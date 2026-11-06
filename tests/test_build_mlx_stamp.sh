#!/usr/bin/env bash
# Tests scripts/build-mlx.sh's stamp and patch guard on a throwaway clone of the working tree (committed or not).
# Nothing is built. Prints "ok <name>" or "FAIL <name>: <why>" per case, then "build-mlx stamp: <passed> of 9 passed".
# Exit 0 only when every case passes.
set -u
set -o pipefail
unset CDPATH
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d "${TMPDIR:-/tmp}/lesto-stamp.XXXXXX")" || { echo "could not test: mktemp"; exit 2; }
W="$(mktemp -d "${TMPDIR:-/tmp}/lesto-stamp-work.XXXXXX")" || { echo "could not test: mktemp"; exit 2; }
trap 'rm -rf "$T" "$W"' EXIT
GIT_T() { git -C "$T" -c user.name=t -c user.email=t@t "$@"; }

git clone -q "$REPO" "$T" || { echo "could not test: clone"; exit 2; }
# overlay the uncommitted state of the checkout: tracked edits, then untracked files
git -C "$REPO" diff HEAD --binary > "$W/overlay" || { echo "could not test: overlay diff"; exit 2; }
git -C "$T" apply --allow-empty "$W/overlay" || { echo "could not test: overlay apply"; exit 2; }
git -C "$REPO" ls-files --others --exclude-standard -z > "$W/untracked" || { echo "could not test: untracked list"; exit 2; }
while IFS= read -r -d '' f; do
  mkdir -p "$T/$(dirname "$f")" && cp -p "$REPO/$f" "$T/$f" || { echo "could not test: copy $f"; exit 2; }
done < "$W/untracked"
GIT_T add -A && GIT_T commit -qam overlay --allow-empty || { echo "could not test: overlay commit"; exit 2; }

PASSED=0
pass() { echo "ok $1"; PASSED=$((PASSED+1)); }
fail() { echo "FAIL $1: $2"; }
# stamp prints the raw output (stdout and stderr) untouched; on a non-zero exit it appends one "STAMP-EXIT=<rc>" line
# after it and returns that rc. Every case captures rc right after the substitution and checks it with isstamp/nostamp.
stamp() { local out rc; out="$(cd "$T" && "$@" scripts/build-mlx.sh --print-stamp 2>&1)"; rc=$?; printf '%s\n' "$out"; [ $rc -eq 0 ] || echo "STAMP-EXIT=$rc"; return $rc; }
STAMPRE='^mlx=[0-9a-f]{12}(-dirty[0-9a-f]{12})? mlxc=[0-9a-f]{12}(-dirty[0-9a-f]{12})? patch=[0-9a-f]{12} target=[0-9.]+$'
# isstamp OUT RC: exit 0 and the output is exactly one line, a well-formed stamp
isstamp() { [ "$2" -eq 0 ] && [ "$(wc -l <<< "$1" | tr -d ' ')" = 1 ] && grep -Eq "$STAMPRE" <<< "$1"; }
# nostamp OUT RC: non-zero exit and no line of the output starts like a stamp
# (here-strings, not pipes: under pipefail a quitting grep can hand the writer a SIGPIPE and flip the verdict)
nostamp() { [ "$2" -ne 0 ] && ! grep -Eq '^mlx=[0-9a-f]{12}' <<< "$1"; }
field() { echo "$1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }
# nosetup WHAT: a state-changing setup step failed, so the case cannot be judged; the run stops as "could not test"
nosetup() { echo "could not test: setup step failed: $1"; exit 2; }

# 1 format; an inherited CDPATH (the current directory, or a parent of the clone) must not change the stamp
S="$(stamp env)"; RC1=$?
S1c="$(stamp env CDPATH=.)"; RC1c=$?
S1p="$(stamp env CDPATH="$(dirname "$T")")"; RC1p=$?
if isstamp "$S" $RC1 && echo "$S" | grep -Eq '^mlx=[0-9a-f]{12} mlxc=[0-9a-f]{12} patch=[0-9a-f]{12} target=26\.2$' \
   && [ "$(field "$S" mlx)" = "$(git -C "$T" rev-parse --short=12 HEAD:lib/mlx-src)" ] \
   && isstamp "$S1c" $RC1c && [ "$S1c" = "$S" ] && isstamp "$S1p" $RC1p && [ "$S1p" = "$S" ]; then pass format
else fail format "got '$S'; CDPATH=. got '$S1c' (rc=$RC1c); CDPATH=parent got '$S1p' (rc=$RC1p)"; fi

# 2 unrelated-commit; HEAD must move, or the equal stamps prove nothing
H2a="$(git -C "$T" rev-parse HEAD)" || nosetup "rev-parse before unrelated commit"
echo "stamp test" >> "$T/README.md" || nosetup "edit README.md"
GIT_T commit -qam unrelated || nosetup "unrelated commit"
H2b="$(git -C "$T" rev-parse HEAD)" || nosetup "rev-parse after unrelated commit"
S2="$(stamp env)"; RC2=$?
if [ "$H2a" = "$H2b" ]; then fail unrelated-commit "HEAD did not change across the unrelated commit ($H2a)"
elif isstamp "$S" $RC1 && isstamp "$S2" $RC2 && [ "$S2" = "$S" ]; then pass unrelated-commit
else fail unrelated-commit "'$S' became '$S2'"; fi

# 3 dirty-mlx
echo "# edit" >> "$T/lib/mlx-src/CMakeLists.txt" || nosetup "edit lib/mlx-src/CMakeLists.txt"
S3="$(stamp env)"; RC3=$?
isstamp "$S3" $RC3 && echo "$(field "$S3" mlx)" | grep -Eq -- '-dirty[0-9a-f]{12}$' && pass dirty-mlx || fail dirty-mlx "mlx=$(field "$S3" mlx)"

# 4 committed-mlx
GIT_T commit -qam "mlx edit" || nosetup "mlx edit commit"
S4="$(stamp env)"; RC4=$?
M4="$(field "$S4" mlx)"
if isstamp "$S4" $RC4 && isstamp "$S" $RC1 && [ -n "$M4" ] && [ "$M4" != "$(field "$S" mlx)" ] && ! echo "$M4" | grep -q -- '-dirty'; then pass committed-mlx
else fail committed-mlx "mlx=$M4 (was $(field "$S" mlx))"; fi

# 5 target
S5="$(stamp env MLX_DEPLOYMENT_TARGET=14.0)"; RC5=$?
isstamp "$S5" $RC5 && echo "$S5" | grep -q ' target=14\.0$' && pass target || fail target "got '$S5'"

# 6 unpatched-dies
mkdir -p "$T/lib/mlx/lib" || nosetup "mkdir lib/mlx/lib"
for f in libmlx.dylib libmlxc.dylib mlx.metallib; do : > "$T/lib/mlx/lib/$f" || nosetup "create lib/mlx/lib/$f"; done
S6="$(stamp env)"; RC6=$?
printf '%s\n' "$S6" > "$T/lib/mlx/.version" || nosetup "write lib/mlx/.version"
OUT="$(cd "$T" && scripts/build-mlx.sh 2>&1)"; RC=$?
if ! isstamp "$S6" $RC6; then fail unpatched-dies "stamp for .version: rc=$RC6 out='$S6'"
elif [ $RC -ne 0 ] || ! echo "$OUT" | grep -q 'nothing to do'; then fail unpatched-dies "patched tree: rc=$RC out='$OUT'"
else
  git -C "$T" apply -R -p1 --directory=lib/mlxc-src patches/mlxc-gather-qmm-global-scale.patch || nosetup "revert mlx-c patch"
  git -C "$T" diff --quiet HEAD -- lib/mlxc-src && nosetup "revert left lib/mlxc-src unchanged"
  # the cached stamp is recomputed for the unpatched tree, so only the patch guard (not a cache miss) can stop the build
  S6u="$(stamp env)"; RC6u=$?
  isstamp "$S6u" $RC6u || nosetup "stamp of the unpatched tree: rc=$RC6u out='$S6u'"
  printf '%s\n' "$S6u" > "$T/lib/mlx/.version" || nosetup "write the unpatched lib/mlx/.version"
  OUT="$(cd "$T" && scripts/build-mlx.sh 2>&1)"; RC=$?
  if [ $RC -ne 0 ] && echo "$OUT" | grep -q 'mlxc-gather-qmm-global-scale.patch'; then pass unpatched-dies
  else fail unpatched-dies "unpatched tree: rc=$RC out='$OUT'"; fi
  git -C "$T" apply -p1 --directory=lib/mlxc-src patches/mlxc-gather-qmm-global-scale.patch || nosetup "re-apply mlx-c patch"
  # re-applying restores HEAD's content exactly, so there is nothing to commit: check the tree is back to HEAD instead
  git -C "$T" diff --quiet HEAD -- lib/mlxc-src || nosetup "re-apply did not restore lib/mlxc-src to HEAD"
fi

# 7 untracked-mlx
GIT_T checkout -q -- . || nosetup "checkout to HEAD"
echo "a" > "$T/lib/mlx-src/lesto_probe.h" || nosetup "write lesto_probe.h (a)"
S7a="$(stamp env)"; RC7a=$?
echo "b" > "$T/lib/mlx-src/lesto_probe.h" || nosetup "write lesto_probe.h (b)"
S7b="$(stamp env)"; RC7b=$?
# a file name with a quote and a space must hash like any other (NUL-delimited listing)
echo "q" > "$T/lib/mlx-src/we\"ird name.h" || nosetup "write quoted-name file"
S7q="$(stamp env)"; RC7q=$?
rm -f "$T/lib/mlx-src/we\"ird name.h" || nosetup "remove quoted-name file"
# an untracked file that cannot be read must stop the stamp (non-zero exit, no stamp line), never hash as empty
chmod 000 "$T/lib/mlx-src/lesto_probe.h" || nosetup "chmod 000 lesto_probe.h"
S7c="$(stamp env)"; RC7=$?
chmod 644 "$T/lib/mlx-src/lesto_probe.h" || nosetup "chmod 644 lesto_probe.h"
if isstamp "$S7a" $RC7a && isstamp "$S7b" $RC7b && isstamp "$S7q" $RC7q \
   && echo "$(field "$S7a" mlx)" | grep -q -- '-dirty' && [ "$S7a" != "$S7b" ] && [ "$S7q" != "$S7b" ] \
   && { [ "$(id -u)" = 0 ] || nostamp "$S7c" $RC7; }; then pass untracked-mlx
else fail untracked-mlx "a='$S7a' (rc=$RC7a) b='$S7b' (rc=$RC7b) q='$S7q' (rc=$RC7q) unreadable-rc=$RC7 out='$S7c'"; fi

# 8 target-relaxed: below 26.2 compared numerically (as strings, 9.0 would sort above 26.2)
BAD8=""
for t in 14.0 26.1 9.0; do
  O8="$(cd "$T" && env MLX_DEPLOYMENT_TARGET=$t scripts/build-mlx.sh --check-target 2>&1)"; R8=$?
  [ $R8 -eq 0 ] && [ "$O8" = relaxed ] || BAD8="$BAD8 $t:'$O8'(rc=$R8)"
done
[ -z "$BAD8" ] && pass target-relaxed || fail target-relaxed "not relaxed:$BAD8"

# 9 target-strict
BAD9=""
for t in 26.2 26.10 27.0; do
  O9="$(cd "$T" && env MLX_DEPLOYMENT_TARGET=$t scripts/build-mlx.sh --check-target 2>&1)"; R9=$?
  [ $R9 -eq 0 ] && [ "$O9" = strict ] || BAD9="$BAD9 $t:'$O9'(rc=$R9)"
done
# a malformed target is refused (exit non-zero, neither word printed), never classified
for t in 26. 26..2 .2 26.2.bad 26.2.1.1 abc 26,2; do
  O9="$(cd "$T" && env MLX_DEPLOYMENT_TARGET="$t" scripts/build-mlx.sh --check-target 2>&1)"; R9=$?
  if [ $R9 -eq 0 ] || grep -Eq '^(strict|relaxed)$' <<< "$O9"; then BAD9="$BAD9 malformed '$t':'$O9'(rc=$R9)"; fi
done
[ -z "$BAD9" ] && pass target-strict || fail target-strict "not strict or malformed accepted:$BAD9"

echo "build-mlx stamp: $PASSED of 9 passed"
[ "$PASSED" -eq 9 ]
