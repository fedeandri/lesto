#!/bin/bash
# test_validator_matrix.sh — API-compliance + agentic matrix across every
# locally-available Zig MLX architecture plus one llama-engine GGUF.
#
# Per model, two layers:
#   1. llmprobe (~/projects/agents/llmprobe) in auto
#      mode - the compliance suite for /v1/chat/completions against a server this script boots and tears down.
#   2. The pi 2-turn agentic html case (multi-turn tool calls: create
#      mlx.html, then add JS) via tests/pi_integration_run.sh, which manages
#      its own server lifecycle and appends audit_format leak markers.
#
# One model loaded at a time; missing weights skip cleanly. A large model
# cannot coexist with another large resident model - this script
# pkills ALL serving lesto instances up front (including the app's).
#
# Usage:
#   ./tests/test_validator_matrix.sh                      # full matrix
#   VALIDATOR_MODELS=gemma4,qwen36 ./tests/...               # csv filter
#   SKIP_PI=1 ./tests/test_validator_matrix.sh            # llmprobe only
#   SKIP_PROBE=1 ./tests/test_validator_matrix.sh         # pi only
#   BINARY=...  PORT=...  LLMPROBE_DIR=...                # overrides
#
# Output: per-model logs in tests/validator-results/, summary table on
# stdout + tests/validator-results/summary.tsv. Exit 1 if anything failed.

set -uo pipefail
cd "$(dirname "$0")/.."

REPO="$(pwd)"
BINARY="${BINARY:-$REPO/zig-out/bin/lesto}"
PORT="${PORT:-11298}"
LLMPROBE_DIR="${LLMPROBE_DIR:-$HOME/projects/agents/llmprobe}"
LLMPROBE_MJS="$LLMPROBE_DIR/bin/dist/llmprobe.mjs"
RESULTS="$REPO/tests/validator-results"
SUMMARY="$RESULTS/summary.tsv"
mkdir -p "$RESULTS"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# logical|display|path|pi_case|probe_timeout_s
# pi_case must exist in pi_integration_run.sh's html matrix.
MODELS=(
    "gemma4|Gemma 4 E4B 8-bit (gemma4)|$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit|html-e4b|90"
    "e4b-gguf|Gemma 4 E4B GGUF (llama engine)|$HOME/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf|html-e4b-gguf|150"
)

if [[ -n "${VALIDATOR_MODELS:-}" ]]; then
    IFS=',' read -r -a WANTED <<< "$VALIDATOR_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do
            [[ "$name" == "$w" ]] && FILTERED+=("$entry")
        done
    done
    MODELS=("${FILTERED[@]:-}")
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fatal] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if [[ -z "${SKIP_PROBE:-}" && ! -f "$LLMPROBE_MJS" ]]; then
    echo "[fatal] llmprobe bundle missing: $LLMPROBE_MJS (cd $LLMPROBE_DIR && npm run build:cli)"
    exit 1
fi

kill_servers() {
    pkill -f "^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)" 2>/dev/null || true
    for _ in $(seq 1 10); do
        pgrep -f "^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)" >/dev/null || return 0
        sleep 0.5
    done
}

start_server() { # path logfile -> 0 on healthy
    "$BINARY" --model "$1" --serve --port "$PORT" --log-level info \
        --ctx-size 32768 > "$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    return 1
}

strip_ansi() { sed -E $'s/\x1b\\[[0-9;]*m//g'; }

trap 'kill_servers' EXIT
kill_servers

[[ -f "$SUMMARY" ]] || printf "timestamp\tmodel\tprobe\tpi\tnotes\n" > "$SUMMARY"

overall_fail=0
declare -a ROWS

for entry in "${MODELS[@]}"; do
    IFS='|' read -r logical display path pi_case probe_timeout <<< "$entry"
    echo -e "\n${YELLOW}===== $logical — $display =====${NC}"
    if [[ ! -e "$path" ]]; then
        echo -e "${YELLOW}SKIP${NC}: model path missing: $path"
        ROWS+=("$logical|SKIP|SKIP|model missing")
        continue
    fi

    probe_cell="skipped"; pi_cell="skipped"; notes=""

    # ---- Layer 1: llmprobe across all three API surfaces -------------------
    if [[ -z "${SKIP_PROBE:-}" ]]; then
        server_log="$RESULTS/$logical.server.log"
        probe_log="$RESULTS/$logical.probe.log"
        if start_server "$path" "$server_log"; then
            node "$LLMPROBE_MJS" "127.0.0.1:$PORT" --timeout "$probe_timeout" \
                2>&1 | strip_ansi > "$probe_log"
            probe_rc=${PIPESTATUS[0]}
            # llmprobe's scorecard line: "ENGINE CONFORMANCE   98.9%"
            probe_cell=$(awk '/^ENGINE CONFORMANCE/ {print $NF; found=1} END { if (!found) print "no-results" }' "$probe_log")
            failed_ids=$(grep -E '^  ✗ [a-z/_ ()]+: ' "$probe_log" | sed 's/^  ✗ //' | paste -sd';' - | head -c 300)
            if [[ "$probe_rc" -ne 0 || "$probe_cell" == "no-results" ]]; then
                overall_fail=1
                notes="$notes probe-fail:[${failed_ids:-see $probe_log}]"
                echo -e "${RED}probe: $probe_cell${NC} ($probe_log)"
            else
                echo -e "${GREEN}probe: $probe_cell${NC}"
            fi
        else
            overall_fail=1
            probe_cell="server-start-fail"
            notes="$notes server-start-fail"
            echo -e "${RED}server failed to boot — tail of $server_log:${NC}"
            tail -n 20 "$server_log"
        fi
        kill_servers
    fi

    # ---- Layer 2: pi agentic 2-turn html case (multi-turn tool calls) ------
    if [[ -z "${SKIP_PI:-}" ]]; then
        pi_log="$RESULTS/$logical.pi.log"
        MLX_BIN="$BINARY" PI_CASES="$pi_case" "$REPO/tests/pi_integration_run.sh" html \
            2>&1 | strip_ansi > "$pi_log"
        score_line=$(grep -E '^SCORE:' "$pi_log" | tail -n1)
        if [[ -n "$score_line" ]]; then
            pi_cell=$(echo "$score_line" | awk '{print $2}')
            pi_notes=$(echo "$score_line" | cut -d' ' -f3-)
            notes="$notes $pi_notes"
            if echo "$score_line" | grep -qE ' -[a-z]' || [[ "${pi_cell%%/*}" != "${pi_cell##*/}" && "${pi_cell%%/*}" -lt "${pi_cell##*/}" ]]; then
                overall_fail=1
                echo -e "${RED}pi: $score_line${NC} ($pi_log)"
            else
                echo -e "${GREEN}pi: $score_line${NC}"
            fi
        else
            overall_fail=1
            pi_cell="no-score"
            notes="$notes pi-no-score"
            echo -e "${RED}pi produced no SCORE line — tail of $pi_log:${NC}"
            tail -n 20 "$pi_log"
        fi
    fi

    ROWS+=("$logical|$probe_cell|$pi_cell|${notes# }")
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$logical" \
        "$probe_cell" "$pi_cell" "${notes# }" >> "$SUMMARY"
done

echo ""
echo "===== validator matrix summary ====="
printf "%-14s %-18s %-10s %s\n" "model" "llmprobe" "pi" "notes"
for row in "${ROWS[@]}"; do
    IFS='|' read -r m p a n <<< "$row"
    printf "%-14s %-18s %-10s %s\n" "$m" "$p" "$a" "$n"
done
echo ""
[[ "$overall_fail" -eq 0 ]] && echo -e "${GREEN}ALL GREEN${NC}" || echo -e "${RED}FAILURES — see $RESULTS${NC}"
exit "$overall_fail"
