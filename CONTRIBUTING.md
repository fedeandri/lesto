# Contributing

Lesto is a personal project, published as it is. Issues are read; pull requests are not expected and may go unreviewed.

## Build

You need an Apple Silicon Mac, Xcode 26.2 or later and CMake (`brew install cmake`; the `Brewfile` lists it).

```bash
./scripts/fetch-zig.sh          # the pinned Zig into .zig-toolchain/
export PATH="$PWD/.zig-toolchain:$PATH"
scripts/build-mlx.sh            # the MLX and mlx-c sources in lib/
zig build -Doptimize=ReleaseFast
```

The binary is `zig-out/bin/lesto`.

## Run the tests

```bash
zig build test                  # the unit tests, one block at the bottom of each .zig file
```

The scripts in `tests/` drive a running server over HTTP and need a Gemma 4 model on disk (`tests/fetch_test_models.sh`). `tests/replies/` and `tests/api/` hold the checks that a change leaves greedy replies and the API contract as they were.

## Bug reports

An issue helps most when it says what you expected and what happened, the model and its quantization, the macOS version and chip, and the server log (`--log-level debug`; the log file is under `~/.mlx-serve/logs/`).
