# Changelog

## 0.21.1

- On macOS 14, the first request no longer fails with HTTP 500. The fused MoE prefill kernel now compiles there, and a failed warm-up or compile at load switches its path off and clears its own error, instead of leaving it for the first request to report.
- A streamed reply sends a character the model writes as several byte tokens (an emoji outside the vocabulary, for example) whole, instead of one U+FFFD per byte.

## 0.21.0

- `MLX_SERVE_TOPK_SCHEDULE=<k1>:<n>:<k2>` routes each request's prompt and first n reply tokens to k1 experts and the later ones to k2. On Gemma 4's 26B-A4B (8 experts by default), `8:4:6` decodes about 7.4% faster and keeps the reply format; replies change, so it is off unless set. The count is per server, which fits one user at a time.

## 0.20.0

- A thread keeps one CPU core busy while chat turns are served and for 500 ms after, so the performance cores stay clocked up while decoding waits on the GPU: faster decoding and a faster first token on a follow-up turn, with replies unchanged. It costs about 0.4 W of CPU power while a turn is served. `MLX_SERVE_KEEP_WARM_OFF=1` turns it off; `MLX_SERVE_KEEP_WARM_MS` sets the window.

## 0.19.0

- One-token decode steps commit on MLX's ops limit alone, for faster decoding with replies unchanged. A turn sent at once after the previous one sees its first token about 14 ms later. `MLX_SERVE_DECODE_OPS_ONLY_COMMIT_OFF=1` keeps the byte limit.
- `NOTICE` states the changes to the vendored MLX and mlx-c, and names the kernel adapted from MLX.

## 0.18.0

- Prompt reading is faster on Gemma 4's 26B-A4B, with replies unchanged: a short MoE prefill runs one eval for the layer loop, the fused expert reduce admits hidden 2816, and gate, up and GeGLU run in one kernel. Each has a switch, listed in `docs/changes-from-mlx-serve.md`.

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
