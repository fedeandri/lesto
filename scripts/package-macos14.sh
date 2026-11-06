#!/usr/bin/env bash
# Packs a finished macOS 14 build into the release tarball, then checks the tarball from a fresh folder:
#
#   scripts/package-macos14.sh <build-out-dir> <new-dest-dir>
#
# <build-out-dir> is what scripts/build-macos14.sh wrote. <new-dest-dir> must not exist yet. Run it from the root of a Lesto tree,
# which supplies LICENSE, LICENSE-APACHE-2.0 and NOTICE.
# Writes <new-dest-dir>/lesto-macos14-arm64.tar.gz and its .sha256. Last line: "package-macos14: OK <sha256>" or
# "package-macos14: FAILED: <why>".
set -uo pipefail
unset CDPATH
# nothing from the caller may supply a library the tarball lacks
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES DYLD_FRAMEWORK_PATH DYLD_FALLBACK_FRAMEWORK_PATH
fail() { echo "package-macos14: FAILED: $*"; exit 1; }
[ $# -eq 2 ] || fail "usage: scripts/package-macos14.sh <build-out-dir> <new-dest-dir>"
OUT=$1; DEST=$2
NAME=lesto-macos14-arm64
[ -x "$OUT/zig-out/bin/lesto" ] || fail "no binary at $OUT/zig-out/bin/lesto"
[ -f "$OUT/lib/mlx/lib/mlx.metallib" ] || fail "no mlx.metallib under $OUT/lib/mlx/lib"
for f in LICENSE LICENSE-APACHE-2.0 NOTICE; do [ -f "$f" ] || fail "no $f here (run from the root of a Lesto tree)"; done
[ ! -e "$DEST" ] || fail "$DEST already exists"
mkdir -p "$DEST/stage/$NAME" || fail "mkdir $DEST"
DEST=$(cd "$DEST" && pwd -P)
cp -R "$OUT/zig-out" "$OUT/lib" "$DEST/stage/$NAME/" || fail "copy of the build"
cp LICENSE LICENSE-APACHE-2.0 NOTICE "$DEST/stage/$NAME/" || fail "copy of the licences"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$DEST/$NAME.tar.gz" -C "$DEST/stage" "$NAME" || fail "tar"
( cd "$DEST" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256" ) || fail "shasum"

# the tarball must work unpacked somewhere else, and carry its licences
mkdir "$DEST/check" || fail "mkdir check"
tar xzf "$DEST/$NAME.tar.gz" -C "$DEST/check" || fail "unpack"
for f in LICENSE LICENSE-APACHE-2.0 NOTICE zig-out/bin/lesto lib/mlx/lib/mlx.metallib; do
  [ -e "$DEST/check/$NAME/$f" ] || fail "$f is not in the tarball"
done
# every library the binary and its dylibs name must be a system one or relative to the unpacked folder
for f in "$DEST/check/$NAME/zig-out/bin/lesto" "$DEST/check/$NAME"/lib/mlx/lib/*.dylib; do
  OUTSIDE=$(otool -L "$f" | tail -n +2 | awk '{print $1}' | grep -Ev '^(/usr/lib/|/System/|@loader_path/|@rpath/)') && fail "$f links outside the tarball: $OUTSIDE"
  RPATHS=$(otool -l "$f" | awk '/cmd LC_RPATH/{getline; getline; print $2}' | grep -v '^@') && fail "$f has an absolute rpath: $RPATHS"
done
VERSION_LINE=$( cd "$DEST/check/$NAME" && ./zig-out/bin/lesto --version 2>/dev/null | grep -a '^lesto ' ) || fail "the unpacked binary does not run"
echo "unpacked: $VERSION_LINE"
echo "size: $(stat -f %z "$DEST/$NAME.tar.gz") bytes"
SHA=$(cut -d' ' -f1 "$DEST/$NAME.tar.gz.sha256")
echo "package-macos14: OK $SHA"
