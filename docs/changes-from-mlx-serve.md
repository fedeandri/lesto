# Changes from mlx-serve

Lesto starts from [mlx-serve](https://github.com/ddalcu/mlx-serve) 26.10.3 by David Dalcu and keeps its engine, its MLX
build and its HTTP server. This file says what was removed, what was added and what was kept, and why. Every
change that is meant to be lossless was checked against greedy replies: the replies of a Gemma 4 model stay
byte-identical from one change to the next (the hashes are in `tests/replies/`).

## Removed

Lesto serves Gemma 4 text on Apple silicon and nothing else, so everything that served another model, another
medium or another platform is gone.

- **Other model families.** The DeepSeek V4 and V4.1 engines, the llama.cpp and GGUF engines, the mlx-stream plugin
  and its packs, the EXL3 expert format, Qwen (2, 3, 3.5, 4 and their vision and audio parts), GLM, MiMo, Flint,
  Nemotron, Mamba, LFM2, MiniCPM, Muse, gpt-oss, Hy3, Laguna, Kolibri, Inkling, Spark, K2, Llama, Mistral, Bailing,
  DiffusionGemma, BERT, EmbeddingGemma and Gemma 3. A model of any of those types is refused at load with
  `UnsupportedModelType`; discovery serves only the four `gemma4*` types.
- **Everything that is not text.** Image, video, audio and 3D generation, speech and transcription, the image and
  audio encoders and image libraries. A request with an image, video or audio part gets HTTP 400 with the message
  `This engine serves text only: image, video and audio content is not supported`, on every turn of the request.
- **Other platforms and backends.** Linux, Windows, iOS, CUDA, the Apple Neural Engine offload and the Agent Sandbox
  guest.
- **Speculative decoding by multi-token prediction** (`--mtp*`), the SSM and hybrid cache plumbing, and the pooling
  modes other than mean (`cls` and `last_token` are refused at load).
- **Interfaces around the engine.** Only `serve`, `--help` and `--version` remain; `pull`, `list`, `run`, `launch`,
  `worker` and `dist-probe` are gone with LAN sharing, distributed inference, the provider layer, the Ollama routes,
  `/v1/messages`, `/v1/responses`, `/v1/completions`, `/v1/decisions`, `/v1/systemone`, `/v1/load-model`,
  `/v1/unload-model` and the built-in web page. An unknown path answers 404.
- **The macOS app, the website, the Homebrew formula and cask, the Firebase files, upstream's workflows and
  release scripts, its roadmap, its benchmark tables and its documentation.**
- **Dead code** with no caller, and the switches only removed code read (the `MLX_SERVE_GDN_*` family and
  `MLX_SERVE_LAGUNA_UBENCH`). A seam router and gate-up epilogue that the fastest build ran off was removed with its four
  switches, which never existed upstream.

What stays: `POST /v1/chat/completions`, `POST /v1/embeddings`, `GET /v1/models`, `POST /v1/models/rescan`,
`GET /v1/load-progress`, `GET /health`, `HEAD /`, `GET /props`, `GET /metrics`, `GET /metrics.json`,
`POST /tokenize`, `POST /detokenize` and the API-key login page.

## Added

The speed techniques, each described by what it does. Each has a switch that restores the plain path (see Kept).

- **GELU-tanh fused MoE decode kernels.** The fused affine-4 MoE decode kernels handle the GELU-tanh activation, so
  Gemma 4's mixture-of-experts layers use them.
- **Folded expert scale and shared K/V projection.** The per-expert scale is folded into the fused down kernel, and on
  the full-attention layers (which have no V projection) V reuses K's projection.
- **Router norm in the layer-tail seam.** The router's RMSNorm is computed in the seam that ends the previous layer, and
  the QK-norm and RoPE kernel also serves head dimension 512.
- **Router matmul in the top-k kernel.** The router matmul runs inside the softmax top-k router kernel: one dispatch per
  mixture-of-experts layer. `MLX_SERVE_ROUTER_KERNEL_CHECK=1` compares it with the chain as a diagnostic.
- **Float accumulation in the quantized matvec.** The input sums of the affine quantized matvec accumulate in float
  (ported from MLX pull request 4516).
- **Latin-script output head.** A second quantized head holds only the rows of the Latin-script tokens and is scored
  instead of the full vocabulary when the model's weights carry a keep list.
- **Head tail.** At one decode row the argmax runs over the kept rows only and is mapped back to the vocabulary id.
- **KV write in place.** On the single-request decode path the K and V rows are written into the cache by the same
  kernel that applies QK-norm and RoPE.
- **Joined Q and K.** On the full-attention layers Q and K share one row group in one matmul.
- **MoE down kernel at four rows, and layer prefetch.** The down kernel computes four output rows per simdgroup, and the
  idle threadgroups of the seam kernels prefetch the next matmul's weights into the system cache.
- **One eval for a short MoE prefill.** A MoE prefill forward under 1024 tokens (under 384 on a 2-bit checkpoint) runs
  one eval for the layer loop instead of a blocking eval every 4 layers, when the KV cache is not quantized and the
  transient budget kept the default cadence. The admission bill charges such a model the whole old cache on a growing append.
- **Fused prefill expert reduce at Gemma 4's width.** The kernel that replaces the prefill's unsort, multiply and sum also
  admits top-k 8 at hidden 2816, the 26B-A4B's geometry, where a test shows it bit-identical to MLX's own reduction.
- **MoE prefill gate, up and GeGLU in one kernel.** On a Mac without MLX's NAX kernels, an affine 4-bit GELU-tanh MoE whose
  experts average 32 rows or more runs gate, up and GeGLU in one kernel through the row map and down at 32-row tiles,
  instead of MLX's two gathers at 16-row tiles and a separate activation. The bits are the same.
  `MLX_SERVE_MOE_PREFILL_FUSED_CHECK=1` compares them with the stock path as a diagnostic.
- **Decode steps commit on the ops limit alone.** MLX commits a command buffer past an ops limit or a byte limit. At one
  token the bound weights trip the byte limit about twice per layer, so a pipelined decode step turns the byte limit off
  around its own eval; prefill keeps both limits. MLX gains a per-thread switch and a commit count for the test. The bits
  are the same.
- **A busy core during turns.** With fewer command buffers per decode step, the decode thread mostly waits on the GPU and
  macOS clocks the P cores down, so the host builds each step more slowly and a turn that follows at once starts later. A
  thread keeps one core busy while a turn is served and for 500 ms after (`MLX_SERVE_KEEP_WARM_MS`). The bits are the same.
- **Top-k schedule, on request.** `MLX_SERVE_TOPK_SCHEDULE=<k1>:<n>:<k2>` routes each request's prompt and first n reply
  tokens to k1 experts per token and every later reply token to k2 (for example `8:16:6`). Under it the MoE routing runs
  uncompiled, since the compiled closure keeps the k it was traced with. The count is per server, not per request slot, so it
  fits single-user serving; batched decode and speculative verify share it. Cached prompt states keep the k they were
  computed with: a later turn that reuses an earlier reply from the prefix cache does not recompute it at k1. Unset, the
  config's experts per token throughout.
- **macOS 14 compiler fix.** The runtime-compiled Metal kernels unpack with `as_type` instead of a pointer pun that
  the macOS 14 compiler backend crashes on.
- **Fixes.** The fp4 MoE decode gate refuses a GELU model (the kernels apply SiLU only); the third `messages` 400 of
  `/v1/embeddings` reads "'messages' is not supported by this model; send the text in 'input'"; a fallback chat
  format writes the tool block after the system turn when both exist; unloading a model frees its compact head; a bad
  keep list is refused at load and leaves the full head.
- **Build.** The jinja template library is compiled from its sources by `build.zig`, no prebuilt archive is
  committed, and a macOS 14 build (`scripts/build-macos14.sh`) runs the same engine.
- **Names.** The binary is `lesto`. The responses say `lesto` in `owned_by` and `system_fingerprint`; an empty model id,
  `"lesto"` and `"mlx-serve"` all reach the default model.

## Kept

- **Weight formats**: affine 2 to 8 bits, nvfp4, mxfp4, mxfp8 and the generic ModelOpt pack load by the checkpoint's own
  `quantization` block.
- **`MLX_SERVE_*` environment names.** Scripts and benchmark harnesses set them, so renaming them would break those
  without a gain. The sixteen technique switches are the exception: they are named for the technique they control, and each
  `..._OFF=1` restores the plain path. `MLX_SERVE_KV_REUSE_OFF` (shared K/V projection, folded expert scale, joined Q|K),
  `MLX_SERVE_SEAM_ROUTER_NORM_OFF` (router norm in the seam, head dimension 512), `MLX_SERVE_ROUTER_KERNEL_OFF`
  (router matmul and top-k kernel; back to the matmul-then-softmax chain), `MLX_SERVE_ROUTER_KERNEL_CHECK` (`=1` compares
  the router kernel with the chain on every call), `MLX_SERVE_LATIN_HEAD_OFF` (Latin-script head; keeps the full head),
  `MLX_SERVE_QK_JOINED_OFF` (joined Q and K; keeps them separate), `MLX_SERVE_HEAD_TAIL_OFF` (head tail; full-row argmax),
  `MLX_SERVE_KV_WRITE_IN_PLACE_OFF` (KV write in place; the slice_update path), `MLX_SERVE_MOE_DOWN_ROWS4_OFF`
  (four-row down kernel; one row), `MLX_SERVE_LAYER_PREFETCH_OFF` (layer prefetch; plain seam kernels),
  `MLX_SERVE_MOE_SHORT_PREFILL_ONE_EVAL_OFF` (one eval for a short MoE prefill; the eval cadence and its admission bill),
  `MLX_SERVE_MOE_PREFILL_REDUCE_2816_OFF` (fused prefill expert reduce at hidden 2816; unsort, multiply and sum),
  `MLX_SERVE_MOE_PREFILL_FUSED_OFF` (MoE prefill gate, up and GeGLU in one kernel; the stock gathers),
  `MLX_SERVE_DECODE_OPS_ONLY_COMMIT_OFF` (decode steps on the ops limit alone; the byte limit stays on),
  `MLX_SERVE_KEEP_WARM_OFF` (a busy core during turns; no spinning thread) and
  `MLX_SERVE_MOE_PREFILL_FUSED_CHECK` (`=1` compares that kernel with the stock path, first four calls).
- **Paths under `~/.mlx-serve/`** (models, logs, `kv-cache`, `round-cost`, the per-model settings file) and the log file
  name `mlx-serve-<port>.log`. The API contract test checks `~/.mlx-serve/kv-cache`, and moving the paths would orphan
  caches users already have.
- **The KV-cache layout salt** (`mlx-serve-kv-layout`). Changing it would invalidate every saved cache for no gain.
- **Prometheus metric names** `mlx_serve:*`. Like the environment names, renaming them breaks existing dashboards and
  the metrics tests for no user gain.
- **The model alias `"mlx-serve"`**, accepted beside `"lesto"` for clients that already send it.
- **The base in `--version`.** `lesto <VERSION> (base mlx-serve 26.10.3)` credits the project Lesto starts from, and
  comments that describe upstream behaviour stay.
