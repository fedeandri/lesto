#!/bin/bash
# fetch_test_models.sh — download the checkpoints the release tests use that
# this Mac can load and does not already have on any model root.
#
#   ./tests/fetch_test_models.sh                        # into ~/.mlx-serve/models
#   ./tests/fetch_test_models.sh --dest /Volumes/X/Models
#   ./tests/fetch_test_models.sh --dry-run              # list, download nothing
#
# An entry is "a|b|c": any of them on disk counts, else the first one that fits
# max_model_gb (tests/_lib_models.sh) downloads. A pack this box cannot load is
# disk spent for nothing, so it is skipped. Needs the `hf` CLI.
set -u
cd "$(dirname "$0")/.."
source tests/_lib_models.sh

DEST="$HOME/.mlx-serve/models"
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# The smoke matrix, the tool matrix and the regression scripts' defaults.
ENTRIES=(
    "mlx-community/gemma-4-e4b-it-8bit"
    "mlx-community/gemma-4-e4b-it-4bit"
    "mlx-community/gemma-4-E4B-it-assistant-bf16"
    "mlx-community/gemma-4-e2b-it-4bit"
)

TOKEN_FILE="$HOME/.cache/huggingface/token"
repo_gb() { # HF repo size in whole GB, rounded up; empty when the API refuses
    local auth=()
    [[ -s "$TOKEN_FILE" ]] && auth=(-H "Authorization: Bearer $(cat "$TOKEN_FILE")")
    curl -s ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$1?blobs=true" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(-(-sum(f.get("size") or 0 for f in d["siblings"]) // 10**9))
except Exception:
    pass'
}

MAX="$(max_model_gb)"
echo "=== fetch: dest=$DEST, max pack ${MAX} GB (GPU budget $(gpu_budget_gb) GB) ==="
FAILED=0
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r -a alts <<< "$entry"
    if have=$(find_model "${alts[@]}"); then echo "have  ${have}"; continue; fi
    pick=""
    for repo in "${alts[@]}"; do
        gb=$(repo_gb "$repo")
        [[ -n "$gb" ]] || { echo "skip  $repo (not on the hub, or gated)"; continue; }
        [[ "$gb" -le "$MAX" ]] || { echo "skip  $repo (${gb} GB > ${MAX} GB)"; continue; }
        pick="$repo"; break
    done
    [[ -n "$pick" ]] || continue
    echo "get   $pick (${gb} GB)"
    [[ "$DRY" -eq 1 ]] && continue
    hf download "$pick" --local-dir "$DEST/$pick" >/dev/null || { echo "FAIL  $pick"; FAILED=$((FAILED+1)); }
done
[[ "$FAILED" -eq 0 ]]
