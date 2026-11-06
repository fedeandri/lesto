#!/bin/bash
# Integration test: deep agent loop regression test.
# Simulates 10+ tool call round-trips via curl, verifying at each step:
#   - finish_reason is "tool_calls" (not "stop" or "length")
#   - Tool call has non-empty name and arguments
#   - Arguments parse as valid JSON with required keys
#   - Tracks prompt_tokens and completion_tokens per iteration to detect budget squeeze
#
# Uses a simple repeating task ("read file") with small mock tool results
# to isolate server infrastructure from model comprehension.
#
# Usage: ./tests/test_deep_agent_loop.sh [port]
# Requires a running server with a model loaded.

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
ITERATIONS=12
PASS=0
FAIL=0
TOTAL=0
MIN_COMPLETION_TOKENS=999999

echo "=== Deep Agent Loop Regression Test ==="
echo "Server: $BASE"
echo "Iterations: $ITERATIONS"
echo ""

# Check server is running
if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

MODEL=$(curl -sf "$BASE/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
echo "Model: $MODEL"
echo ""

run_test() {
    local name="$1"
    local result="$2"
    local detail="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name — $detail"
    fi
}

# Use temp files for reliable JSON handling (avoids quoting nightmares)
MSGS_FILE=$(mktemp)
REQ_FILE=$(mktemp)
RESP_FILE=$(mktemp)
trap "rm -f $MSGS_FILE $REQ_FILE $RESP_FILE" EXIT

# Tool definition
TOOL_DEF='[{"type":"function","function":{"name":"readFile","description":"Read a file and return its content","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to read"}},"required":["path"]}}}]'

# Initial messages
python3 -c "
import json
msgs = [
    {'role':'system','content':'You are a helpful assistant. You MUST use the readFile tool for every request. Always call the readFile tool. Never respond with text alone. Read the files one at a time.'},
    {'role':'user','content':'Read these files one by one: /tmp/a.txt, /tmp/b.txt, /tmp/c.txt, /tmp/d.txt, /tmp/e.txt, /tmp/f.txt, /tmp/g.txt, /tmp/h.txt, /tmp/i.txt, /tmp/j.txt, /tmp/k.txt, /tmp/l.txt. Read them one at a time using readFile.'}
]
json.dump(msgs, open('$MSGS_FILE','w'))
"

echo "--- Deep loop: $ITERATIONS iterations ---"
echo ""

for i in $(seq 1 $ITERATIONS); do
    echo "  Iteration $i/$ITERATIONS:"

    # Build request from messages file
    python3 -c "
import json
msgs = json.load(open('$MSGS_FILE'))
tools = json.loads('$TOOL_DEF')
req = {'model':'mlx-serve','messages':msgs,'tools':tools,'max_tokens':512,'temperature':0.3,'stream':False}
json.dump(req, open('$REQ_FILE','w'))
"

    # Send request
    curl -sf "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d @"$REQ_FILE" > "$RESP_FILE" 2>/dev/null

    if [ ! -s "$RESP_FILE" ]; then
        run_test "Iteration $i: server responds" "FAIL" "empty response (server error?)"
        echo ""
        continue
    fi

    # Parse response
    PARSED=$(python3 -c "
import json
r = json.load(open('$RESP_FILE'))
choice = r['choices'][0]
finish = choice['finish_reason']
usage = r.get('usage', {})
pt = usage.get('prompt_tokens', 0)
ct = usage.get('completion_tokens', 0)
tc_name = ''
tc_args = ''
tc_id = ''
tc_args_valid = False
tc_has_path = False
if finish == 'tool_calls' and 'tool_calls' in choice.get('message', {}):
    tc = choice['message']['tool_calls'][0]
    tc_name = tc['function']['name']
    tc_args = tc['function']['arguments']
    tc_id = tc.get('id', 'call_0')
    try:
        parsed = json.loads(tc_args)
        tc_args_valid = True
        tc_has_path = 'path' in parsed
    except: pass
content = (choice.get('message',{}).get('content','') or '')[:100]
print(f'{finish}|{pt}|{ct}|{tc_name}|{tc_args_valid}|{tc_has_path}|{tc_id}|{tc_args}|{content}')
" 2>/dev/null)

    FINISH=$(echo "$PARSED" | cut -d'|' -f1)
    PROMPT_TOKENS=$(echo "$PARSED" | cut -d'|' -f2)
    COMP_TOKENS=$(echo "$PARSED" | cut -d'|' -f3)
    TC_NAME=$(echo "$PARSED" | cut -d'|' -f4)
    TC_ARGS_VALID=$(echo "$PARSED" | cut -d'|' -f5)
    TC_HAS_PATH=$(echo "$PARSED" | cut -d'|' -f6)
    TC_ID=$(echo "$PARSED" | cut -d'|' -f7)
    TC_ARGS=$(echo "$PARSED" | cut -d'|' -f8)
    CONTENT=$(echo "$PARSED" | cut -d'|' -f9)

    echo "    prompt_tokens=$PROMPT_TOKENS, completion_tokens=$COMP_TOKENS, finish=$FINISH"

    # Track minimum completion tokens
    if [ "$COMP_TOKENS" -lt "$MIN_COMPLETION_TOKENS" ] 2>/dev/null; then
        MIN_COMPLETION_TOKENS=$COMP_TOKENS
    fi

    # Check finish reason
    if [ "$FINISH" != "tool_calls" ]; then
        run_test "Iteration $i: finish_reason=tool_calls" "FAIL" "got '$FINISH' (content: ${CONTENT:0:80})"
        echo ""
        break
    fi
    run_test "Iteration $i: finish_reason=tool_calls" "PASS" ""

    if [ -z "$TC_NAME" ]; then
        run_test "Iteration $i: tool call has name" "FAIL" "empty name"
    else
        run_test "Iteration $i: tool call has name" "PASS" ""
    fi

    if [ "$TC_ARGS_VALID" != "True" ]; then
        run_test "Iteration $i: arguments are valid JSON" "FAIL" "raw: $TC_ARGS"
    else
        run_test "Iteration $i: arguments are valid JSON" "PASS" ""
    fi

    if [ "$TC_HAS_PATH" != "True" ]; then
        run_test "Iteration $i: arguments contain 'path' key" "FAIL" "args: $TC_ARGS"
    else
        run_test "Iteration $i: arguments contain 'path' key" "PASS" ""
    fi

    if [ "$COMP_TOKENS" -lt 5 ] 2>/dev/null; then
        run_test "Iteration $i: completion_tokens >= 5" "FAIL" "only $COMP_TOKENS tokens (budget squeeze?)"
    else
        run_test "Iteration $i: completion_tokens >= 5" "PASS" ""
    fi

    # Append assistant+tool+continue to messages file
    MOCK_RESULT="File content from iteration $i: line1=hello, line2=world, line3=done."
    python3 -c "
import json
msgs = json.load(open('$MSGS_FILE'))
msgs.append({'role':'assistant','content':'','tool_calls':[{'id':'$TC_ID','type':'function','function':{'name':'$TC_NAME','arguments':json.load(open('$RESP_FILE'))['choices'][0]['message']['tool_calls'][0]['function']['arguments']}}]})
msgs.append({'role':'tool','tool_call_id':'$TC_ID','content':'$MOCK_RESULT'})
msgs.append({'role':'user','content':'Continue reading the remaining files.'})
json.dump(msgs, open('$MSGS_FILE','w'))
"

    echo ""
done

echo ""
echo "--- Token budget analysis ---"
echo "  Minimum completion_tokens across iterations: $MIN_COMPLETION_TOKENS"
if [ "$MIN_COMPLETION_TOKENS" -lt 10 ] 2>/dev/null; then
    echo "  WARNING: Very low completion token count detected — possible budget squeeze"
fi
echo ""

# ── Summary ──
echo "=== Summary ==="
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: $FAIL"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
