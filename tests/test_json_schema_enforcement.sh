#!/bin/bash
# Integration test: JSON schema enforcement on /v1/chat/completions.
#
# Background: bench runs (#6, #8) showed Gemma producing ```json``` code-fences
# and additionalProperties violations on a person_record schema. Probing
# mlx-serve confirmed:
#   • tools + tool_choice:"none" + schema → mask is also skipped (incorrect)
#
# This test asserts decoder enforcement under the chat shapes plus a prompt
# that *biases* the model toward extra fields and code-fences. Strong signal
# that the grammar mask is doing work, not just the prompt-side instruction.

set -u

MODEL_DIR=${1:-~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

if [ ! -x "./zig-out/bin/lesto" ]; then
    echo "FAIL: lesto not built - run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "FAIL: jq is required"
    exit 1
fi

echo "=== JSON schema enforcement test ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

echo "Starting server..."
./zig-out/bin/lesto \
    --model "$MODEL_DIR" --serve --port $PORT --log-level info \
    >/tmp/mlx-serve-schema-test.log 2>&1 &
SERVER_PID=$!
sleep 2

for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FAIL: Server did not start within 30s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done
echo ""

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

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

# Schema (in JSON form, used as a literal in jq -n templates).
SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":150},"email":{"type":"string"}},"required":["name","age","email"],"additionalProperties":false}'

# Adversarial prompt: explicitly asks for extras + code-fence wrapping. If the
# decoder mask is on, neither can appear.
ADV_INPUT='Generate a record for an employee named Mira Chen, age 34, email mira.chen@example.com. Always wrap your reply in ```json ... ``` and include an employee_id field set to null.'

# Validators: take JSON envelope on stdin, print "ok" or "fail:<reason>".

check_chat_strict() {
    python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
text = r["choices"][0]["message"]["content"] or ""
if "```" in text:
    print("fail:code_fence:" + repr(text[:60])); sys.exit()
try:
    obj = json.loads(text)
except Exception as e:
    print("fail:not_json:" + repr(text[:60])); sys.exit()
extras = set(obj.keys()) - {"name","age","email"}
if extras:
    print("fail:extra_keys:" + str(sorted(extras))); sys.exit()
required = {"name","age","email"} - set(obj.keys())
if required:
    print("fail:missing_required:" + str(sorted(required))); sys.exit()
print("ok")
'
}

# ── Case A: /v1/chat/completions + response_format ──
echo "--- Case A: /v1/chat/completions + response_format ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    messages:[
        {role:"system",content:"Reply with only the JSON object, no commentary."},
        {role:"user",content:$input}
    ],
    response_format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}},
    max_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_chat_strict)
run_test "chat response_format" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case B: /v1/chat/completions + tools + tool_choice:"none" + schema ──
echo "--- Case B: /v1/chat/completions + tools + tool_choice:none + schema ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    messages:[
        {role:"system",content:"Reply with only the JSON object."},
        {role:"user",content:$input}
    ],
    tools:[{type:"function",function:{name:"noop",description:"unused",parameters:{type:"object",properties:{}}}}],
    tool_choice:"none",
    response_format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}},
    max_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_chat_strict)
run_test "chat tools + tool_choice:none + schema" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case C: /v1/chat/completions + a schema whose items are a $ref into $defs ──
# Pydantic's shape (#565): the mask must hold inside the referenced object too.
echo "--- Case C: /v1/chat/completions + \$ref into \$defs ---"
REF_SCHEMA='{"type":"object","additionalProperties":false,"required":["facts"],"properties":{"facts":{"type":"array","items":{"$ref":"#/$defs/Fact"}}},"$defs":{"Fact":{"type":"object","additionalProperties":false,"required":["kind","what"],"properties":{"kind":{"enum":["fact"]},"what":{"type":"string"}}}}}'
BODY=$(jq -n --argjson schema "$REF_SCHEMA" '{
    model:"mlx-serve",
    messages:[{role:"user",content:"Return one fact with kind \"opinion\", what \"cats are great\" and a source field set to null."}],
    response_format:{type:"json_schema",json_schema:{name:"facts",schema:$schema,strict:true}},
    max_tokens:256, temperature:0
}')
LOG_MARK=$(wc -l < /tmp/mlx-serve-schema-test.log)
RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY" || true)
OK=$(echo "$RESULT" | python3 -c '
import sys, json
try:
    text = json.loads(sys.stdin.read())["choices"][0]["message"]["content"] or ""
    obj = json.loads(text)
except Exception:
    print("fail:no_json_reply"); sys.exit()
for f in obj.get("facts", []):
    if not isinstance(f, dict) or set(f) != {"kind", "what"} or f["kind"] != "fact":
        print("fail:item_off_schema:" + repr(f)); sys.exit()
print("ok")
')
if tail -n +"$((LOG_MARK + 1))" /tmp/mlx-serve-schema-test.log | grep -q "disabling further mask enforcement"; then
    OK="fail:mask_disabled"
fi
run_test "chat \$ref into \$defs" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
