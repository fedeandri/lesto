# Zig is NOT a brew dep: `scripts/fetch-zig.sh` fetches the pinned 0.17.0
# release into .zig-toolchain/ (checked at the top of build.zig).
#
# mlx + mlx-c are NOT brew deps: their sources are vendored in lib/mlx-src and
# lib/mlxc-src and built by scripts/build-mlx.sh so the NAX (M5) kernels ship
# enabled; the brew bottle is compiled at deployment target 26.0 and silently
# disables them.
#
# cmake is what BUILDS those sources (four calls in scripts/build-mlx.sh), so it
# is a build-time tool rather than something the binary links against. macOS
# ships no cmake, and a clean Mac stops at the first `cmake -S` without it.
brew "cmake"
