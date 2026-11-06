#!/usr/bin/env bash
# Per-model settings (`~/.mlx-serve/model-settings.json`, issue #269): a model's
# `ctx_size` / `kv_quant` follow the MODEL, apply on its load
# (boot AND cold load), and a second model in the same process keeps the globals.
#
# Runs under a private HOME so the real settings file is never touched.
# NEEDS REAL MODELS: skips when the two small defaults are absent.
#
# Usage: ./tests/test_model_settings.sh [port]
set -uo pipefail
PORT="${1:-11384}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/lesto"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODELS_ROOT="${MODELS_ROOT:-$HOME/.mlx-serve/models}"
MODEL_A="${MODEL_A:-$MODELS_ROOT/mlx-community/gemma-4-e4b-it-4bit}"
MODEL_B="${MODEL_B:-$MODELS_ROOT/mlx-community/gemma-4-e4b-it-8bit}"
if [ ! -f "$MODEL_A/config.json" ] || [ ! -f "$MODEL_B/config.json" ]; then
    echo "SKIP: needs two local chat models (MODEL_A=$MODEL_A, MODEL_B=$MODEL_B)"
    exit 0
fi

PASS=0; FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $1"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.mlx-serve"
SETTINGS="$FAKE_HOME/.mlx-serve/model-settings.json"
LOG="$FAKE_HOME/server.log"
SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    pkill -f "^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)" 2>/dev/null
    rm -rf "$FAKE_HOME"
}
trap cleanup EXIT
pkill -f "^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)" 2>/dev/null
sleep 0.5

write_settings() { # write_settings <ctx> <kv>  — override for MODEL_A only
    cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ctx_size": $1, "kv_quant": "$2" }, "not-a-model": 1 }
JSON
}
write_settings 4096 8

boot_server() { # boot_server <model dir>: a fresh process on the private HOME, waits for /health
    [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
    HOME="$FAKE_HOME" "$BIN" --serve --model "$1" --model-dir "$MODELS_ROOT" --ctx-size 16384 --port "$PORT" --log-file off >"$LOG" 2>&1 &
    SRV=$!
    local up=0
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$SRV" 2>/dev/null || break
        sleep 0.5
    done
    [ "$up" = "1" ] || { echo "FAIL: server never became healthy"; tail -5 "$LOG"; return 1; }
}
boot_server "$MODEL_A" || exit 1

row() { # row <model path> <field> — top-level context_length or meta.kv_quant of the READY row
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
for m in json.load(sys.stdin)['data']:
    if want.endswith('/' + m['id']):
        print(m['context_length'] if sys.argv[2] == 'ctx' else m['meta'].get('kv_quant'))
        break
" "$1" "$2"
}
cold_load() { # cold_load <model path>: a one-token chat request naming the model cold-loads it; prints the HTTP code
    curl -s -o /dev/null -w '%{http_code}' --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}"
}

# [1] boot load honours the file
check "[1] boot: context_length 4096 from the file (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "4096" ] && echo 1 || echo 0)"
check "[1] boot: meta.kv_quant 8 from the file (got $(row "$MODEL_A" kv))" "$([ "$(row "$MODEL_A" kv)" = "8" ] && echo 1 || echo 0)"
check "[1] log names the override" "$(grep -q "\[model-settings\] .*ctx=4096 kv=8" "$LOG" && echo 1 || echo 0)"

# [2] a second model keeps the globals
CODE="$(cold_load "$MODEL_B")"
check "[2] cold load of model B -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[2] model B keeps --ctx-size 16384 (got $(row "$MODEL_B" ctx))" "$([ "$(row "$MODEL_B" ctx)" = "16384" ] && echo 1 || echo 0)"
check "[2] model B keeps kv off (got $(row "$MODEL_B" kv))" "$([ "$(row "$MODEL_B" kv)" = "off" ] && echo 1 || echo 0)"
check "[2] model A still 4096 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "4096" ] && echo 1 || echo 0)"

# [3] edit + restart applies the new values
write_settings 8192 4
boot_server "$MODEL_A" || exit 1
check "[3] model A now 8192 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "8192" ] && echo 1 || echo 0)"
check "[3] model A now kv 4 (got $(row "$MODEL_A" kv))" "$([ "$(row "$MODEL_A" kv)" = "4" ] && echo 1 || echo 0)"

# [4] a malformed file never stops a load
echo '{nope' >"$SETTINGS"
boot_server "$MODEL_A"; BOOT_RC=$?
check "[4] malformed file: boot ok (rc $BOOT_RC), globals apply (ctx $(row "$MODEL_A" ctx))" \
    "$([ "$BOOT_RC" = "0" ] && [ "$(row "$MODEL_A" ctx)" = "16384" ] && echo 1 || echo 0)"
check "[4] malformed file logged" "$(grep -q "\[model-settings\] .*malformed" "$LOG" && echo 1 || echo 0)"

# [5] the per-model drafter: "off" silences a pack's own drafter/, "auto" brings it back
source "$(dirname "$0")/_lib_models.sh"
DRAFT_MODEL="${DRAFT_MODEL:-$(find_model mlx-community/gemma-4-e4b-it-4bit)}"
if [ -f "$DRAFT_MODEL/drafter/config.json" ]; then
    drafter_loaded() {
        curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
print(next((m['meta'].get('drafter_loaded') for m in json.load(sys.stdin)['data'] if want.endswith('/' + m['id'])), None))
" "$DRAFT_MODEL"
    }
    for want in off:False auto:True; do
        printf '{ "%s": { "drafter": "%s" } }' "$DRAFT_MODEL" "${want%%:*}" >"$SETTINGS"
        boot_server "$DRAFT_MODEL"; BOOT_RC=$?
        GOT="$(drafter_loaded)"
        check "[5] drafter ${want%%:*}: boot rc $BOOT_RC, drafter_loaded $GOT" \
            "$([ "$BOOT_RC" = "0" ] && [ "$GOT" = "${want##*:}" ] && echo 1 || echo 0)"
    done
else
    echo "  SKIP [5]: no pack with a drafter/ (DRAFT_MODEL=$DRAFT_MODEL)"
fi

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
