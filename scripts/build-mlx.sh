#!/usr/bin/env bash
# Build the pinned mlx + mlx-c submodules (lib/mlx-src, lib/mlxc-src) into
# lib/mlx/{lib,include} with MLX's NAX (M5 neural-accelerator) kernels ENABLED.
#
# Why not Homebrew: the brew bottle is compiled at MACOSX_DEPLOYMENT_TARGET
# 26.0 (the Tahoe builders' point release), which fails MLX's CMake gate
# (SDK >= 26.2 AND deployment target >= 26.2): the NAX kernels are silently
# skipped and MLX_METAL_NO_NAX hard-wires is_nax_available() to false, even on
# M5 hardware. Building ourselves at 26.2 flips the gate; the runtime check in
# MLX still gates dispatch on GPU gen >= 17, so M1–M4 behavior is unchanged.
#
# The gate fails SILENTLY upstream (a configure-time warning only landed after
# v0.32.0), so this script ASSERTS the result: no *_nax kernels in the built
# metallib = hard failure, never a quietly-degraded stage.
#
# Consequence of the 26.2 deployment target: the staged dylibs (and anything
# bundling them) require macOS >= 26.2 at runtime.
#
# This is the single source of truth for the pinned mlx/mlx-c versions: the
# tree hashes of the vendored directories lib/mlx-src and lib/mlxc-src in this
# repository (plus a -dirty suffix for uncommitted edits). Bump by replacing
# the directory contents and committing; local builds rebuild automatically
# (stamp mismatch). Guard test: tests/test_mlx_staged_nax.sh.
set -euo pipefail
# An inherited CDPATH makes `cd` print its destination, which would corrupt SCRIPT_DIR and REPO_ROOT below.
unset CDPATH

DEPLOYMENT_TARGET="${MLX_DEPLOYMENT_TARGET:-26.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLX_SRC="$REPO_ROOT/lib/mlx-src"
MLXC_SRC="$REPO_ROOT/lib/mlxc-src"
STAGE="$REPO_ROOT/lib/mlx"
BUILD_ROOT="$REPO_ROOT/lib/.mlx-build"
STAMP="$STAGE/.version"

die() { echo "[build-mlx] ERROR: $*" >&2; exit 1; }

[ -f "$MLX_SRC/CMakeLists.txt" ] && [ -f "$MLXC_SRC/CMakeLists.txt" ] \
  || die "vendored sources missing: lib/mlx-src or lib/mlxc-src has no CMakeLists.txt"

# Stamp of a vendored directory: its tree hash in HEAD, plus -dirty<12 hex> when it has uncommitted or untracked
# changes. The 12 hex hash `git diff HEAD` of the directory and each untracked file's path and content.
tree_stamp() {
  local dir="$1" base status h f fh tracked list untracked
  base="$(git -C "$REPO_ROOT" rev-parse --short=12 "HEAD:$dir")" || die "cannot read tree hash of $dir"
  status="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all -- "$dir")" || die "git status failed for $dir"
  if [ -z "$status" ]; then echo "$base"; return; fi
  tracked="$(git -C "$REPO_ROOT" diff HEAD -- "$dir")" || die "git diff failed for $dir"
  list="$(mktemp "${TMPDIR:-/tmp}/lesto-untracked.XXXXXX")" || die "mktemp failed"
  git -C "$REPO_ROOT" ls-files --others --exclude-standard -z -- "$dir" > "$list" || { rm -f "$list"; die "git ls-files failed for $dir"; }
  untracked=""
  while IFS= read -r -d '' f; do
    fh="$(shasum < "$REPO_ROOT/$f")" || { rm -f "$list"; die "cannot hash $f"; }
    untracked="$untracked$f ${fh%% *}"$'\n'
  done < "$list"
  rm -f "$list"
  h="$(printf '%s\n%s' "$tracked" "$untracked" | shasum | cut -c1-12)" || die "hashing failed for $dir"
  echo "$base-dirty$h"
}
MLX_SHA="$(tree_stamp lib/mlx-src)"
MLXC_SHA="$(tree_stamp lib/mlxc-src)"
# The mlx-c patch below is part of the build: an edited patch must rebuild.
MLXC_PATCH="$REPO_ROOT/patches/mlxc-gather-qmm-global-scale.patch"
PATCH_SHA="$(shasum "$MLXC_PATCH" | cut -c1-12)"
WANT="mlx=$MLX_SHA mlxc=$MLXC_SHA patch=$PATCH_SHA target=$DEPLOYMENT_TARGET"

# A target below 26.2 builds without the NAX kernels (MLX's CMake gate needs 26.2), which is the macOS 14 build's case
# (scripts/build-macos14.sh). Compared numerically by major and minor: as strings, 9.0 would sort above 26.2.
target_is_strict() {
  local major minor
  # <major> or <major>.<minor> or <major>.<minor>.<patch>, digits only
  [[ "$1" =~ ^([0-9]+)(\.([0-9]+))?(\.[0-9]+)?$ ]] || die "bad deployment target '$1'"
  major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[3]:-0}"
  [ "$((10#$major))" -gt 26 ] || { [ "$((10#$major))" -eq 26 ] && [ "$((10#$minor))" -ge 2 ]; }
}

if [ "${1:-}" = "--print-stamp" ]; then echo "$WANT"; exit 0; fi
if [ "${1:-}" = "--check-target" ]; then
  if target_is_strict "$DEPLOYMENT_TARGET"; then echo strict; else echo relaxed; fi
  exit 0
fi

# Lesto keeps the mlx-c patch applied in lib/mlxc-src (applying it at build time would rewrite tracked files).
git -C "$REPO_ROOT" apply -R --check -p1 --directory=lib/mlxc-src "$MLXC_PATCH" 2>/dev/null \
  || die "lib/mlxc-src lacks patches/mlxc-gather-qmm-global-scale.patch (Lesto keeps it applied in the source)"

# Idempotent: skip when the staged build already matches the pinned SHAs.
if [ -f "$STAMP" ] && [ -f "$STAGE/lib/libmlx.dylib" ] \
   && [ -f "$STAGE/lib/libmlxc.dylib" ] && [ -f "$STAGE/lib/mlx.metallib" ]; then
  if [ "$(cat "$STAMP")" = "$WANT" ]; then
    echo "[build-mlx] lib/mlx already at ($WANT): nothing to do"
    exit 0
  fi
  echo "[build-mlx] staged '$(cat "$STAMP")' != '$WANT': rebuilding"
fi

# ── Toolchain preflight: every one of these failing would otherwise surface
# as a silently NAX-less metallib, not an error. ──────────────────────────────
xcrun -sdk macosx metal --version >/dev/null 2>&1 \
  || die "Metal compiler unavailable: Xcode 26 ships it as a separate download: xcodebuild -downloadComponent MetalToolchain"

SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
SDK_MAJOR="${SDK_VERSION%%.*}"
SDK_MINOR="$(echo "$SDK_VERSION" | cut -d. -f2)"
if [ "$SDK_MAJOR" -lt 26 ] || { [ "$SDK_MAJOR" -eq 26 ] && [ "${SDK_MINOR:-0}" -lt 2 ]; }; then
  die "macOS SDK $SDK_VERSION < 26.2: MLX's NAX kernel gate needs SDK >= 26.2 (update Xcode)"
fi
echo "[build-mlx] SDK $SDK_VERSION, deployment target $DEPLOYMENT_TARGET, $WANT"

NCPU="$(sysctl -n hw.ncpu)"

# A rebuild overwrites the staged libraries: drop the old stamp first, so a build that fails midway can never leave a
# stamp that matches a half-replaced stage (for example after reverting to the previous source state).
rm -f "$STAMP"

# ── mlx (C++ core + metallib) ────────────────────────────────────────────────
cmake -S "$MLX_SRC" -B "$BUILD_ROOT/mlx" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlx" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlx" >/dev/null

# ── mlx-c against the staged mlx (same pairing brew uses: USE_SYSTEM_MLX) ────
# MLX 0.32.3's gather_qmm takes a global_scale before sorted_indices; the
# pinned mlx-c predates it. Lesto keeps the patch applied in lib/mlxc-src and
# checks that above.
cmake -S "$MLXC_SRC" -B "$BUILD_ROOT/mlxc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$STAGE" \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlxc" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlxc" >/dev/null

# ── Assert the point of the exercise ─────────────────────────────────────────
METALLIB="$STAGE/lib/mlx.metallib"
[ -f "$METALLIB" ] || die "mlx.metallib not at $METALLIB: mlx changed its install layout"

STRINGS_OUT="$(strings "$METALLIB")" || die "strings failed on $METALLIB"
NAX_COUNT="$(printf '%s\n' "$STRINGS_OUT" | grep -c "_nax")" || [ "$NAX_COUNT" = 0 ] || die "counting NAX kernels failed"
if [ "$NAX_COUNT" -le 0 ]; then
  if target_is_strict "$DEPLOYMENT_TARGET"; then
    die "built metallib contains ZERO *_nax kernels: the 26.2 gate failed silently (check SDK/deployment target/Metal Toolchain)"
  fi
  echo "[build-mlx] target $DEPLOYMENT_TARGET below 26.2: zero *_nax kernels expected"
fi

MINOS="$(otool -l "$STAGE/lib/libmlx.dylib" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/ && !d {print $2; d=1}')"
if target_is_strict "$DEPLOYMENT_TARGET"; then
  target_is_strict "$MINOS" || die "libmlx.dylib minos '$MINOS' < 26.2: deployment target did not take"
else
  case "$MINOS" in
    "$DEPLOYMENT_TARGET"|"$DEPLOYMENT_TARGET.0") echo "[build-mlx] target $DEPLOYMENT_TARGET below 26.2: libmlx.dylib minos $MINOS" ;;
    *) die "libmlx.dylib minos '$MINOS' != requested target $DEPLOYMENT_TARGET: deployment target did not take" ;;
  esac
fi

[ -f "$STAGE/lib/libmlxc.dylib" ] || die "libmlxc.dylib missing from stage"

echo "$WANT" > "$STAMP"
echo "[build-mlx] staged lib/mlx OK: $NAX_COUNT NAX symbol hits, minos $MINOS"
