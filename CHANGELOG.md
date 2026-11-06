# Changelog

## 0.17.1

First public version.

## 0.17.0

- A prebuilt macOS 14 binary, packed and checked by `scripts/package-macos14.sh`. It is unsigned; the README has the install steps.
- The README introduces the project and states its speed against other Mac engines and its checkpoint.

## 0.16.2

- `--version`, `--help` and `--prompt` output continues after what a redirected file already holds, instead of overwriting it from the start of the file.

## 0.16.1

- `scripts/bench.py` no longer drops an engine that sends a few tokens per stream event after a lone first token: its decode speed is still timed correctly. The first event's limit tightens from 20% to 2% of the characters.

## 0.16.0

- `scripts/bench.py` runs the same benchmark the same way every time: cold decode speed, time to first text and 200-token reply time at about 1k and 8k prompt tokens, then later-turn time to first text in conversations. `--mode release` compares a build with the previous one, and `--mode field` compares it with other Mac engines. `bench.py compare` shows what changed between two runs and refuses runs from different machines or methods. See [docs/benchmark.md](docs/benchmark.md).

## 0.15.1

Lesto is an inference engine for Gemma 4 on Apple silicon Macs, derived from mlx-serve 26.10.3.
[docs/changes-from-mlx-serve.md](docs/changes-from-mlx-serve.md) says what was removed, what was added and what was kept
before this version.

Changes in this version:

- The technique switches are named for what they control. `=1` on an `_OFF` name turns the technique off:
  `MLX_SERVE_KV_REUSE_OFF` (K's projection reused as V, the folded expert scale and the joined Q|K group),
  `MLX_SERVE_SEAM_ROUTER_NORM_OFF` (the router norm from the layer-tail seam and the 512-wide fused QK-norm+RoPE),
  `MLX_SERVE_ROUTER_KERNEL_OFF` (restores the router matmul-then-softmax chain), `MLX_SERVE_LATIN_HEAD_OFF` (keeps the full
  head), `MLX_SERVE_QK_JOINED_OFF` (keeps separate Q and K), `MLX_SERVE_HEAD_TAIL_OFF` (restores the full-row argmax),
  `MLX_SERVE_KV_WRITE_IN_PLACE_OFF` (restores the slice_update path), `MLX_SERVE_MOE_DOWN_ROWS4_OFF` (restores ROWS=1 for
  the MoE down matmul) and `MLX_SERVE_LAYER_PREFETCH_OFF` (restores the plain seam kernels);
  `MLX_SERVE_ROUTER_KERNEL_CHECK=1` compares the router kernel with the chain on every call.
- Responses say `lesto` in `owned_by` and `system_fingerprint`. An empty model id, `lesto` and `mlx-serve` all reach the
  default model.
- `lesto` is now a reserved model alias, like `mlx-serve`: a per-model alias of that name in `~/.mlx-serve/model-settings.json` is
  ignored and the name reaches the default model.
- Messages that named mlx-serve (the port-in-use line, the unsupported-architecture message, the log truncation marker,
  the power assertion, the embedded bundle identity) say lesto.
- The jinja template library is built from its sources; no prebuilt archive is committed.
- `SECURITY.md` states the threat model, and `CONTRIBUTING.md` and the README are rewritten for Lesto.
