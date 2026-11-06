#!/usr/bin/env bash
# Lesto's stdout must continue after what a redirect already holds, not overwrite it from the start of the file.
#   tests/test_stdout_redirect.sh <lesto binary> [<model dir>]
# With a model directory it also checks the reply path of `--prompt`. Last line: OK (exit 0) or FAILED (exit 1).
set -u
BIN=$1; MODEL=${2:-}
DIR=$(mktemp -d)
FILE=$DIR/out
RC=0
check() {  # check <name> <command...>: "first" and "last" around the command's output must both survive
  local name=$1; shift
  { echo first; "$@"; echo last; } > "$FILE" 2> /dev/null
  if [ "$(head -1 "$FILE")" = first ] && [ "$(tail -1 "$FILE")" = last ] && [ "$(wc -l < "$FILE")" -gt 2 ]; then
    echo "ok: $name"
  else
    echo "FAILED: $name"; head -c 300 "$FILE"; echo; RC=1
  fi
}
check --version "$BIN" --version
check --help "$BIN" --help
if [ -n "$MODEL" ]; then check --prompt "$BIN" --model "$MODEL" --prompt "Say hello in two words." --max-tokens 8; fi
if [ $RC -eq 0 ]; then echo OK; else echo FAILED; fi
exit $RC
