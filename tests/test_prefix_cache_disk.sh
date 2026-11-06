#!/bin/bash
# SSD prefix-cache persistence (`--prefix-cache-disk`, src/kv_disk_cache.zig).
#
# End-to-end restart round-trip:
#   1. Boot with an isolated HOME, fire a long-prompt request (cold prefill),
#      capture output + TTFT. Assert `[disk-cache] persisted` fires after the
#      response.
#   2. Kill the server. Assert chunked entries exist on disk
#      (~/.mlx-serve/kv-cache/<fp>/e*/{meta.json,tokens.bin,c*.safetensors}).
#   3. Boot again, re-issue the identical request. Assert:
#        * `[disk-cache] restored N/M tokens from SSD` fires,
#        * the restart TTFT beats the cold TTFT by >= 2x,
#        * the output is byte-identical to run 1 (temp 0).
#   4. Multi-turn extension after restart appends chunks, doesn't rewrite
#      the whole entry (`persisted` with a small chunk count).
#   5. `--prefix-cache-disk off` boots clean, serves, and never touches the
#      kv-cache dir.
#   6. `--no-prefix-cache-ram` keeps RAM residency at zero while SSD still
#      persists and restores prefixes across a restart.
#
# Usage: ./tests/test_prefix_cache_disk.sh [/path/to/model] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-8096}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Legacy flat layout fallback (pre-two-level model dirs).
if [ ! -d "$MODEL" ] && [ -d "$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit" ]; then
    MODEL="$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit"
fi
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_cache_disk: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/lesto}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi

pkill -f "^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)" 2>/dev/null || true
sleep 1

# Isolated HOME so the test never touches the user's real kv-cache.
# On the real disk: /tmp can be a small tmpfs (Linux), where the SSD tier rightly refuses to store.
mkdir -p "$HOME/claude-tmp"
SCRATCH_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk.XXXXXX")
LOGFILE=$(mktemp)
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$SCRATCH_HOME" "$LOGFILE"
}
trap cleanup EXIT

start_server() { # extra args...
    : > "$LOGFILE"
    # The SSD tier is OPT-IN since the off-by-default flip (ad3fd24) — the
    # suite must enable it explicitly. Callers' "$@" comes later, so section
    # 5's `--prefix-cache-disk off` still wins (last flag parses last).
    # SERVER_ULIMIT_N lowers the server's soft open-file limit.
    ( [ -n "$SERVER_ULIMIT_N" ] && ulimit -n "$SERVER_ULIMIT_N"
      HOME="$SCRATCH_HOME" exec "$BINARY" --model "$MODEL" --serve --port "$PORT" \
        --ctx-size 8192 --no-pld --log-level info --prefix-cache-disk 4GB "$@" ) > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 90); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then return 0; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:"; tail -20 "$LOGFILE"; return 1; }
        sleep 1
    done
    return 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 1
}

launchd_stop() { # TERM, 5 s grace, then KILL: what `exit timeout = 5` does
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 1
}

term_stop() { # $1 seconds to wait before KILL. TERM twice, 0.5 s apart; sets STOP_SECS and KILLED
    local t0 now # wall clock: python's monotonic clock is per process on macOS
    t0=$(python3 -c 'import time; print(time.time())')
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 0.5
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    KILLED=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        now=$(python3 -c "import time; print(int(time.time() - $t0 >= $1))")
        if [ "$now" = 1 ]; then kill -KILL "$SERVER_PID" 2>/dev/null || true; KILLED=1; break; fi
        sleep 0.2
    done
    wait "$SERVER_PID" 2>/dev/null || true
    STOP_SECS=$(python3 -c "import time; print(f'{time.time() - $t0:.1f}')")
    SERVER_PID=""
    sleep 1
}

session_turn() { # $1 session label (A|B|C), $2 log lines (default 800, ~25.8k tokens) -> "content"
    python3 - "$BASE" "$1" "${2:-800}" <<'PY'
import json, sys, urllib.request
base, who, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
k = {'A': 1, 'B': 7, 'C': 13}[who]
lines = [f'Session {who}: read this project log fully, then answer.']
lines += [f'Log entry {i}: subsystem {(i*k)%17} reported state change {i*31*k%997} with latency {i*7%131} ms and checksum {i*i*k%9973}.' for i in range(1, n + 1)]
lines.append('Question: which subsystem appears in log entry 42? One short sentence.')
body = json.dumps({'model': 'mlx-serve', 'messages': [{'role': 'user', 'content': '\n'.join(lines)}],
                   'max_tokens': 24, 'temperature': 0.0, 'enable_thinking': False, 'stream': False}).encode()
req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
resp = json.load(urllib.request.urlopen(req, timeout=1200))
print(resp['choices'][0]['message']['content'].replace('\n', ' '))
PY
}

sessions_restore() { # resend sessions A and B; each must restore within one chunk of its prompt
    local who seen line restored total
    for who in A B; do
        seen=$(grep -cE '\[disk-cache\] restored [0-9]+/[0-9]+ tokens' "$LOGFILE" || true)
        session_turn "$who" > /dev/null || { echo -e "${RED}FAIL${NC} session $who after reboot failed"; FAIL=1; }
        line=$(grep -oE '\[disk-cache\] restored [0-9]+/[0-9]+ tokens' "$LOGFILE" | sed -n "$((seen + 1))p")
        restored=$(echo "$line" | grep -oE '[0-9]+/' | tr -d /)
        total=$(echo "$line" | grep -oE '/[0-9]+' | tr -d /)
        if [ -n "$restored" ] && [ "$restored" -ge $((total - 1024)) ]; then
            echo -e "${GREEN}PASS${NC} session $who restored $restored/$total after the reboot"
        else
            echo -e "${RED}FAIL${NC} session $who restored '${restored:-none}' of '${total:-?}' (want within 1024 of the prompt)"; FAIL=1
        fi
    done
}

# Deterministic ~2.5k-token prompt (well above the 512-token persist floor).
LONG_PROMPT=$(python3 -c "
lines = ['You are reviewing a project log. Read it fully, then answer.']
for i in range(1, 81):
    lines.append(f'Log entry {i}: subsystem {i%17} reported state change {i*31%997} with latency {i*7%131} ms and checksum {i*i%9973}.')
lines.append('Question: which subsystem appears in log entry 42? Answer with one short sentence.')
print('\n'.join(lines))")

fire_long() { # -> "elapsed_ms|content"
    local body
    body=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':24,'temperature':0.0,'stream':False}))
" "$LONG_PROMPT")
    python3 - "$BASE" "$body" <<'PY'
import json, sys, time, urllib.request
base, body = sys.argv[1], sys.argv[2]
t0 = time.monotonic()
req = urllib.request.Request(base + "/v1/chat/completions", data=body.encode(), headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=600))
ms = int((time.monotonic() - t0) * 1000)
content = resp["choices"][0]["message"]["content"].replace("|", " ").replace("\n", " ")
print(f"{ms}|{content}")
PY
}

FAIL=0
KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"

echo "== 1. cold boot + long request (persist) =="
start_server || { echo -e "${RED}FAIL${NC} server 1 failed to start"; exit 1; }
OUT1=$(fire_long)
COLD_MS="${OUT1%%|*}"
CONTENT1="${OUT1#*|}"
echo "  cold TTLT(total)=${COLD_MS}ms content='${CONTENT1:0:60}'"
sleep 1  # flush happens post-response on the inference thread
if grep -q '\[disk-cache\] persisted' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} disk persist fired"
else
    echo -e "${RED}FAIL${NC} no [disk-cache] persisted line"
    tail -20 "$LOGFILE"; FAIL=1
fi
stop_server

echo
echo "== 2. on-disk layout =="
META_COUNT=$(find "$KV_DIR" -name meta.json 2>/dev/null | wc -l | tr -d ' ')
CHUNK_COUNT=$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
if [ "$META_COUNT" -ge 1 ] && [ "$CHUNK_COUNT" -ge 2 ]; then
    echo -e "${GREEN}PASS${NC} persisted entry on disk ($META_COUNT meta, $CHUNK_COUNT chunks)"
else
    echo -e "${RED}FAIL${NC} expected >=1 meta + >=2 chunks, got $META_COUNT/$CHUNK_COUNT"
    find "$KV_DIR" -type f 2>/dev/null | head
    FAIL=1
fi

echo
echo "== 3. restart + identical request (restore) =="
start_server || { echo -e "${RED}FAIL${NC} server 2 failed to start"; exit 1; }
OUT2=$(fire_long)
RESTART_MS="${OUT2%%|*}"
CONTENT2="${OUT2#*|}"
echo "  restart total=${RESTART_MS}ms (cold was ${COLD_MS}ms) content='${CONTENT2:0:60}'"
if grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD restore engaged: $(grep -o '\[disk-cache\] restored [^\\n]*' "$LOGFILE" | head -1)"
else
    echo -e "${RED}FAIL${NC} no [disk-cache] restored line after restart"
    tail -30 "$LOGFILE"; FAIL=1
fi
if [ "$CONTENT1" = "$CONTENT2" ]; then
    echo -e "${GREEN}PASS${NC} output byte-identical across restart restore"
else
    echo -e "${RED}FAIL${NC} output diverged: '$CONTENT1' vs '$CONTENT2'"
    FAIL=1
fi
SPEEDUP_OK=$(python3 -c "print(1 if $RESTART_MS * 2 <= $COLD_MS else 0)")
if [ "$SPEEDUP_OK" = "1" ]; then
    echo -e "${GREEN}PASS${NC} restart request >=2x faster than cold (${RESTART_MS}ms vs ${COLD_MS}ms)"
else
    echo -e "${RED}FAIL${NC} restart request not >=2x faster (${RESTART_MS}ms vs ${COLD_MS}ms)"
    FAIL=1
fi

echo
echo "== 4. multi-turn extension appends chunks =="
EXT_BODY=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[
  {'role':'user','content':sys.argv[1]},
  {'role':'assistant','content':sys.argv[2]},
  {'role':'user','content':'Now answer the same question for log entry 43.'}],
  'max_tokens':24,'temperature':0.0,'stream':False}))
" "$LONG_PROMPT" "$CONTENT2")
# Persist lines already in the log (section 3's restore may have written some): only a NEW one proves this request persisted.
PERSIST_BEFORE=$(grep -c '\[disk-cache\] persisted' "$LOGFILE" || true)
EXT_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H "Content-Type: application/json" -d "$EXT_BODY" "$BASE/v1/chat/completions")
sleep 1
PERSIST_LINES=$(grep -c '\[disk-cache\] persisted' "$LOGFILE" || true)
if [ "$EXT_CODE" != "200" ]; then
    echo -e "${RED}FAIL${NC} extension request returned HTTP $EXT_CODE (want 200)"
    tail -20 "$LOGFILE"; FAIL=1
elif [ "$PERSIST_LINES" -gt "$PERSIST_BEFORE" ]; then
    echo -e "${GREEN}PASS${NC} extension turn persisted (append): $((PERSIST_LINES - PERSIST_BEFORE)) new persist line(s)"
else
    echo -e "${RED}FAIL${NC} extension turn did not persist (persist lines $PERSIST_BEFORE before, $PERSIST_LINES after)"
    tail -20 "$LOGFILE"; FAIL=1
fi
stop_server

echo
echo "== 5. --prefix-cache-disk off leaves disk untouched =="
rm -rf "$KV_DIR"
start_server --prefix-cache-disk off || { echo -e "${RED}FAIL${NC} server 3 failed to start"; exit 1; }
OUT3=$(fire_long)
sleep 1
if [ -d "$KV_DIR" ] && [ -n "$(find "$KV_DIR" -name meta.json 2>/dev/null)" ]; then
    echo -e "${RED}FAIL${NC} kv-cache written despite --prefix-cache-disk off"
    FAIL=1
elif grep -q '\[disk-cache\]' "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} disk-cache log lines despite off"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} off switch respected"
fi
stop_server

echo
echo "== 6. SSD-only mode persists without RAM retention =="
rm -rf "$KV_DIR"
start_server --no-prefix-cache-ram || { echo -e "${RED}FAIL${NC} SSD-only server failed to start"; exit 1; }
fire_long > /dev/null
sleep 1
if grep -q 'Prefix cache: SSD ONLY' "$LOGFILE" &&
   grep -q '\[disk-cache\] e[0-9][0-9]* complete on disk' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD-only mode persisted a complete prefix"
else
    echo -e "${RED}FAIL${NC} SSD-only mode did not persist a complete prefix"
    tail -30 "$LOGFILE"; FAIL=1
fi
stop_server
start_server --no-prefix-cache-ram || { echo -e "${RED}FAIL${NC} SSD-only restart failed"; exit 1; }
curl -fsS "$BASE/props" | jq -e '.settings.prefix_cache | .ram_enabled == false and .mem_bytes == 0' > /dev/null
fire_long > /dev/null
if grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE" &&
   ! grep -q '\[hot-cache\] resident=' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD-only restart restored with no hot-cache residency"
else
    echo -e "${RED}FAIL${NC} SSD-only restart did not restore cleanly"
    tail -30 "$LOGFILE"; FAIL=1
fi
stop_server

echo "  -- zero entries disables all reuse even with SSD configured --"
start_server --prefix-cache-entries 0 || { echo -e "${RED}FAIL${NC} cache-off server failed"; exit 1; }
fire_long > /dev/null
if grep -q '\[disk-cache\]\|\[hot-cache\] reused' "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} zero entries still reused or wrote prefixes"; FAIL=1
else
    echo -e "${GREEN}PASS${NC} zero entries disables both tiers"
fi
stop_server

echo
echo "== 8. a restore wider than the fd limit, and a failed restore's fallback =="
# A restore reads one chunk file per 1024 tokens. It must fit under a soft
# RLIMIT_NOFILE smaller than its chunk count, and an unreadable chunk must
# fall back to a cold prefill that answers 200.
MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
FD_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk-fd.XXXXXX")
SCRATCH_HOME="$FD_HOME"
KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"
LONG_PROMPT=$(python3 -c "
lines = ['You are reviewing a project log. Read it fully, then answer.']
for i in range(1, 1201):
    lines.append(f'Log entry {i}: subsystem {i%17} reported state change {i*31%997} with latency {i*7%131} ms and checksum {i*i%9973}.')
lines.append('Question: which subsystem appears in log entry 42? Answer with one short sentence.')
print('\n'.join(lines))")
start_server --ctx-size 65536 || { echo -e "${RED}FAIL${NC} fd server failed to start"; exit 1; }
# One commit flushes at most 512 MB; each repeat extends the entry.
for _ in 1 2 3 4 5 6; do
    fire_long > /dev/null
    sleep 1
    [ "$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l)" -ge 30 ] && break
done
TOP_FD=$(lsof -a -p "$SERVER_PID" -d '0-65535' -F f 2>/dev/null | sed -n 's/^f//p' | sort -n | tail -1)
stop_server
FD_CHUNKS=$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
FD_LIMIT=$((TOP_FD + 16))
echo "  persisted $FD_CHUNKS chunks; serving top fd $TOP_FD; restoring under ulimit -n $FD_LIMIT"
if [ "$FD_CHUNKS" -lt 30 ]; then
    echo -e "${RED}FAIL${NC} need 30+ chunks to exceed the fd headroom, got $FD_CHUNKS"
    FAIL=1
else
    SERVER_ULIMIT_N="$FD_LIMIT" start_server --ctx-size 65536 || { echo -e "${RED}FAIL${NC} fd server 2 failed to start"; exit 1; }
    if fire_long > /dev/null && grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE"; then
        echo -e "${GREEN}PASS${NC} restore under the fd limit: $(grep -oE '\[disk-cache\] restored .* in [0-9]+ms' "$LOGFILE" | head -1)"
    else
        echo -e "${RED}FAIL${NC} restore under ulimit -n $FD_LIMIT did not engage"
        grep -E 'Failed to open|restore failed|prefill failed' "$LOGFILE" | head -5; FAIL=1
    fi
    stop_server

    # After boot: the startup scan salvages an entry whose chunk is already unreadable.
    start_server --ctx-size 65536 || { echo -e "${RED}FAIL${NC} fd server 3 failed to start"; exit 1; }
    chmod 000 "$(find "$KV_DIR" -name 'c000002.safetensors' | head -1)"
    if fire_long > /dev/null && grep -q 'restore failed: .* falling back' "$LOGFILE" && ! grep -q 'prefill failed' "$LOGFILE"; then
        echo -e "${GREEN}PASS${NC} unreadable chunk: restore failed, cold prefill answered"
    else
        echo -e "${RED}FAIL${NC} unreadable chunk did not fall back to a working cold prefill"
        grep -E 'disk-cache|hot-cache|prefill failed|GenerationFailed' "$LOGFILE" | head -8; FAIL=1
    fi
    stop_server
fi
chmod -R u+rw "$FD_HOME" 2>/dev/null; rm -rf "$FD_HOME"


exit $FAIL
