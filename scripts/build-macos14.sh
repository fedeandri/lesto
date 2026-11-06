#!/usr/bin/env bash
# Build Lesto for macOS 14 (minimum OS 14.0) on a macOS 26 host with Xcode 26.5, and copy the result to <out-dir>:
#
#   scripts/build-macos14.sh <out-dir>
#
# Run it from the root of a Lesto worktree used only for macOS 14 builds. It builds in that worktree and never touches
# the 26.2 build of another worktree. It needs a macOS 26 host with Xcode 26.5 or later.
#
# What it does, in order:
#   1. MLX and mlx-c at deployment target 14.0 (scripts/build-mlx.sh, which reports the missing NAX kernels and the
#      lower minimum OS at that target instead of failing).
#   2. zig build, then the binary's minimum OS is set to 14.0 (an explicit zig target drops the SDK's include paths, so
#      the native build is lowered afterwards). libwebp is gone: Lesto links no image library.
#   3. Checks: the minimum OS of the binary and of every non-system library it links is 14.0, no Homebrew library is
#      linked, MTLResidencySetDescriptor is not a strong import, and the binary runs (--help exits 0).
#   4. zig-out/bin and lib/mlx/lib are copied into <out-dir> with the same relative layout (the binary finds its
#      libraries through @loader_path/../../lib/mlx/lib).
#
# Last line of output: "build-macos14: OK <out-dir>" (exit 0) or "build-macos14: FAILED: <why>" (exit 1).
set -uo pipefail
unset CDPATH

# a signal (a job's perl alarm wrapper sends SIGALRM) turns into an exit
trap 'exit 143' INT TERM HUP ALRM
# fail: the last line printed is always the verdict
fail() { echo "build-macos14: FAILED: $*"; exit 1; }
ok_exit() { echo "build-macos14: OK $OUT"; exit 0; }

[ $# -eq 1 ] || fail "usage: scripts/build-macos14.sh <out-dir>"
OUT="$1"
# a reused output directory would keep stale libraries the binary could load through its rpath
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then fail "$OUT already exists and is not empty (use a fresh directory)"; fi
[ -f build.zig ] || fail "not a Lesto worktree (run from its root)"
ROOT="$(pwd -P)"
export PATH="$HOME/homebrew/bin:$ROOT/.zig-toolchain:$PATH"

# 1. MLX and mlx-c at 14.0
MLX_DEPLOYMENT_TARGET=14.0 scripts/build-mlx.sh || fail "scripts/build-mlx.sh at target 14.0"

# 2. build, then lower the binary's minimum OS
rm -rf zig-out
zig build -Doptimize=ReleaseFast -Dgit-sha=046b647 || fail "zig build"
B=zig-out/bin/lesto
[ -x "$B" ] || fail "no binary at $B"
xcrun vtool -set-build-version macos 14.0 26.5 -replace -output "$B.m14" "$B" || fail "vtool"
mv "$B.m14" "$B" || fail "mv of the vtool output"
codesign -f -s - "$B" || fail "codesign"

# 3. checks
# awk reads all of otool's output (no early exit), so otool never gets SIGPIPE under pipefail
# A library built for an older macOS loads on macOS 14: its minimum must not be above 14.0.
minos_ok() {
  local v="$1" maj min pat
  [[ "$v" =~ ^([0-9]+)(\.([0-9]+))?(\.[0-9]+)?$ ]] || return 1
  maj="${BASH_REMATCH[1]}"; min="${BASH_REMATCH[3]:-0}"
  pat="${BASH_REMATCH[4]#.}"; pat="${pat:-0}"
  [ "$((10#$maj))" -lt 14 ] || { [ "$((10#$maj))" -eq 14 ] && [ "$((10#$min))" -eq 0 ] && [ "$((10#$pat))" -eq 0 ]; }
}
minos() { otool -l "$1" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/ && !d {print $2; d=1}'; }
BM="$(minos "$B")" || fail "otool on the binary"
echo "binary lesto minos $BM"
[ "$BM" = "14.0" ] || fail "binary minos $BM, expected 14.0"
# Walk the binary's dependencies and theirs: every non-system library must be built for 14.0 and none may be a Homebrew
# library. The part of an @rpath name after "@rpath/" is a relative path (subdirectories kept) under lib/mlx/lib; a
# part that is empty, absolute or has a ".." component is refused.
resolve() {
  local lib="$1" rel hit real droot
  case "$lib" in
    @rpath/*)
      rel="${lib#@rpath/}"
      case "$rel" in ""|/*|..|../*|*/..|*/../*) return 1 ;; esac
      droot=lib/mlx/lib
      hit="$droot/$rel"
      [ -f "$hit" ] || return 1
      # containment of the resolved file; the links themselves are vetted tree-wide by check_links before the walk
      real="$(readlink -f "$hit")" || return 4
      case "$real" in "$(readlink -f "$droot")"/*) ;; *) return 4 ;; esac
      echo "$hit"; return 0 ;;
    # @loader_path, @executable_path and absolute paths outside the system: not supported (an absolute path would not
    # exist on the machine the build is copied to)
    *) return 1 ;;
  esac
}
deps() { # non-system dependencies named by $1, one install name per line (empty output is valid for a leaf)
  local out names
  out="$(otool -L "$1")" || return 2
  # the install name is everything before " (compatibility version", so names with spaces survive
  names="$(printf '%s\n' "$out" | sed -n '2,$ s/^[[:space:]]*\(.*\) (compatibility version .*$/\1/p')" || return 2
  [ -n "$names" ] || return 0
  # system means exactly these two directory prefixes (the trailing slash keeps /usr/libexec or /SystemExtra out)
  printf '%s\n' "$names" | grep -v -e '^/usr/lib/' -e '^/System/Library/'
  [ $? -le 1 ] # grep status 1 means every name was a system library
}
# Every symlink in the two copied folders must be relative and resolve inside its own folder: `cp -a` keeps links as
# they are, so an absolute or escaping one (at any step of a chain, or on a directory) would point at the build host.
check_links() {
  local root link target real rroot list
  for root in zig-out/bin lib/mlx/lib; do
    [ -d "$root" ] || continue
    rroot="$(readlink -f "$root")" || return 1
    list="$(mktemp "${TMPDIR:-/tmp}/lesto-links.XXXXXX")" || { echo "mktemp failed"; return 1; }
    find "$root" -type l -print0 > "$list" || { rm -f "$list"; echo "find failed under $root"; return 1; }
    while IFS= read -r -d '' link; do
      target="$(readlink "$link")" || { rm -f "$list"; echo "cannot read the link $link"; return 1; }
      case "$target" in /*) rm -f "$list"; echo "absolute symlink $link -> $target"; return 1 ;; esac
      # a library link names a sibling: any ".." step could pass through a directory that is not copied
      case "/$target/" in */../*) rm -f "$list"; echo "symlink $link -> $target steps through .."; return 1 ;; esac
      real="$(readlink -f "$link")" || { rm -f "$list"; echo "dangling symlink $link"; return 1; }
      case "$real" in "$rroot"/*) ;; *) rm -f "$list"; echo "symlink $link resolves outside $root"; return 1 ;; esac
    done < "$list"
    rm -f "$list"
  done
  return 0
}
LINKERR="$(check_links)" || fail "${LINKERR:-symlink check failed}"
PENDING="$B"
SEEN=""
while [ -n "$PENDING" ]; do
  cur="${PENDING%%$'\n'*}"
  if [ "$cur" = "$PENDING" ]; then PENDING=""; else PENDING="${PENDING#*$'\n'}"; fi
  DEPS="$(deps "$cur")" || fail "otool -L on $cur"
  if [ "$cur" = "$B" ] && [ -z "$DEPS" ]; then fail "the binary links no non-system library (expected at least libmlx and libmlxc)"; fi
  while IFS= read -r lib; do
    [ -n "$lib" ] || continue
    case "$lib" in */homebrew/*) fail "$cur links a Homebrew library: $lib" ;; esac
    case "$SEEN" in *"|$lib|"*) continue ;; esac
    SEEN="$SEEN|$lib|"
    p="$(resolve "$lib")"; rrc=$?
    case $rrc in
      0) ;;
      4) fail "$cur needs $lib, a symlink that leaves its directory (the copied build would not hold its target)" ;;
      *) fail "$cur needs $lib, which is not an @rpath name found in lib/mlx/lib (it would not exist where the build is copied)" ;;
    esac
    LM="$(minos "$p")" || fail "otool on $p"
    echo "lib $(basename "$lib") minos $LM"
    minos_ok "$LM" || fail "lib $lib minos $LM, above 14.0 (it would not load on macOS 14)"
    PENDING="${PENDING:+$PENDING$'\n'}$p"
  done <<< "$DEPS"
done
NM_UM="$(nm -um "$B")" || fail "nm -um on the binary"
# a here-string, not a pipe: grep reads the whole nm output and its own status is the one checked (0 matches, 1 none, >1 error)
MRS="$(grep -e MTLResidencySetDescriptor <<< "$NM_UM")"; MRS_RC=$?
[ "$MRS_RC" -le 1 ] || fail "grep over the nm -um output exited $MRS_RC (cannot decide the MTLResidencySetDescriptor import)"
if [ "$MRS_RC" -eq 0 ]; then
  while IFS= read -r line; do
    case "$line" in *weak*) ;; *) fail "the binary still links MTLResidencySetDescriptor strongly (macOS 15): $line" ;; esac
  done <<< "$MRS"
fi
"$B" --help > /dev/null 2>&1 || fail "$B --help exits non-zero (the binary does not load on this macOS)"

# 4. copy the two folders
# files that only the removed libwebp step (and earlier builds of this tree) wrote into the staged folder
rm -f lib/mlx/lib/libwebp*.dylib lib/mlx/lib/libsharpyuv*.dylib
mkdir -p "$OUT/zig-out" "$OUT/lib/mlx" || fail "cannot create $OUT"
cp -a zig-out/bin "$OUT/zig-out/" || fail "copy of zig-out/bin"
cp -a lib/mlx/lib "$OUT/lib/mlx/" || fail "copy of lib/mlx/lib"
ok_exit
