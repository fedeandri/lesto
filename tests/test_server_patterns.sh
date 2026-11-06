#!/bin/bash
# The pkill/pgrep pattern the test scripts use to find their own server: it must match the server's command line
# (lesto or the shipped mlx-serve binary) and must not match a test script that merely lives under a path holding
# the repository name. Runs without a server or a model.
PORT=11299
PAT_TEXT='^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)'   # the pattern as the scripts spell it, port variable unexpanded
PAT="^([^ ]*/)?(lesto|mlx-serve) .*--port $PORT( |$)"
fail=0
check() {  # $1 = want (match|nomatch), $2 = command line
    if printf '%s\n' "$2" | grep -E -q -- "$PAT"; then got=match; else got=nomatch; fi
    if [ "$got" = "$1" ]; then echo "PASS $1: $2"; else echo "FAIL want $1 got $got: $2"; fail=1; fi
}
check match   "./zig-out/bin/lesto --serve --port $PORT --model m"
check match   "./zig-out/bin/lesto --serve --model m --port $PORT"
check match   "/Users/x/lesto/zig-out/bin/lesto serve --port $PORT"
check match   "/Applications/MLX-Serve.app/Contents/MacOS/mlx-serve --serve --port $PORT"
check nomatch "bash /Users/x/lesto/tests/test_tokenizer_hf_parity.sh --port $PORT"
check nomatch "python3 /Users/x/lesto/tests/foo.py --port $PORT"
check nomatch "./zig-out/bin/lesto --serve --port 11300"
check nomatch "./zig-out/bin/lesto --serve --port ${PORT}0"
check nomatch "python3 tests/api/contract.py --binary /tmp/lesto --model m --port $PORT"
check match   "lesto --serve --port $PORT"

# Every pkill/pgrep server pattern in tests/*.sh, *.py and *.mjs is this same pattern (the port variable written as $P):
# a script with its own looser or stricter spelling kills a server another job owns, or leaves its own running.
variants=$(python3 - "$(dirname "$0")" "$PAT_TEXT" <<'PY'
import glob, os, re, sys
d, own = sys.argv[1], sys.argv[2]
rx = re.compile(r"\^\(\[\^ \]\*/\)\?\(lesto\|mlx-serve\) [^\"']*")
norm = lambda x: re.sub(r"\$\{?[A-Za-z_]+\}?", "$P", x)
seen = {}
pk = re.compile(r"p(?:kill|grep)\s+(?:-\S+\s+)*[\"']([^\"']*(?:lesto|mlx-serve)[^\"']*)[\"']")
for f in sorted(glob.glob(os.path.join(d, "*.sh")) + glob.glob(os.path.join(d, "*.py")) + glob.glob(os.path.join(d, "*.mjs"))):
    if os.path.basename(f) == "test_server_patterns.sh":
        continue
    for n, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
        for m in rx.finditer(line):
            seen.setdefault(norm(m.group(0)), []).append("%s:%d" % (os.path.basename(f), n))
        # any other pkill/pgrep expression that names the server is a pattern too, even when it is not spelled canonically
        for m in pk.finditer(line):
            if not rx.search(m.group(1)):
                seen.setdefault(norm(m.group(1)), []).append("%s:%d" % (os.path.basename(f), n))
seen.setdefault(norm(own), []).append("test_server_patterns.sh")
for k, where in sorted(seen.items()):
    print("%s\t%s" % (k, ",".join(where[:3])))
PY
)
scan_rc=$?
nvar=$(printf '%s\n' "$variants" | wc -l | tr -d ' ')
if [ "$scan_rc" -ne 0 ] || [ -z "$variants" ]; then
    echo "FAIL the pattern scan did not run (python3 rc=$scan_rc)"; fail=1
elif [ "$nvar" = "1" ]; then
    echo "PASS one server pattern across the test scripts"
else
    echo "FAIL $nvar different server patterns across the test scripts:"; printf '%s\n' "$variants" | sed 's/^/    /'; fail=1
fi
exit $fail
