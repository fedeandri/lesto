# Changelog

## 0.27.1

- Every switch, log tag and identifier has a descriptive name; earlier entries use the new names. Replies are unchanged. A local `lesto tune` table written by 0.27.0 is refused (the kernel's source hash changed) until `lesto tune` runs again.

## 0.27.0

- The drafter verify's MoE down kernel has a second body: the same arithmetic with its per-tile values and accumulators in 1-element arrays, bit for bit identical to the first. A down plan now names its layout and its body (`body=0` scalar, `body=1` array, in `MLX_SERVE_MOE_ROWS_LAYOUT_DOWN` and every log line that prints a plan). Which body is faster depends on the GPU.
- `lesto tune` searches the body too: 58 candidates instead of 34 (plan space 2). Its table is schema 2: a 0.26.0 table is refused after the upgrade (the log says `unknown_schema`); run `lesto tune` again.
- The M3 generation of GPU (G15, not phones) gets a built-in entry: down at SPS 6, TPT 4, AX 0 with the array body, about 15% faster per call than the previous layout in the tune's measures on an M3 Pro (two clean runs). Gate/up stays on the previous layout there, because its gain in the worse routing regime is under the tune's 2% floor. The M4 entry is unchanged and keeps the scalar body.
- `lesto tune --background` installs its LaunchAgent as a `Standard` job instead of a `Background` one. launchd throttles a `Background` job, and on an M3 that made the timings slow and unsteady enough that the tune discarded most batches and picked nothing, while the same binary in the foreground kept every batch. A table that 0.26.0's `--background` wrote may have been timed under that throttling; 0.27.0 refuses it anyway (schema 2), so the retune also fixes that.
- Every reply is unchanged. On an M4 Pro, every reply is byte-identical to 0.25.0's (150 of 150 turns of a multi-turn conversation set, and 210 of 210 general turns) with the default layouts, with `MLX_SERVE_MOE_ROWS_LAYOUT=0`, with `MLX_SERVE_KERNEL_TABLE=off`, and with the array body forced at the M4's layout. On an M3 Pro (macOS 14), 20 of 20 first turns gave the same reply with the new entry and without it, and all 432 kernel checks of both bodies were identical.
- On that M3 Pro with the drafter, decode is about 2% faster with the new entry, in two sessions: paired difference +1.4 tok/s (33 of 40 turns faster; 69.3 against 70.9 tok/s) and +1.7 tok/s (32 of 40; 69.9 against 71.7 tok/s). An A/A run of the previous layout against itself gave +0.0 tok/s (16 of 40). Not preregistered.
- The entry's down layout (SPS 6) and SPS 3 of the same layout tied within about 1 us a call in both clean M3 runs; each run's final stage picked one of the two.

## 0.26.0

- The MoE rows layouts of 0.25.0 are on by default for the M4 generation of GPU (G16, not phones): the drafter verify's gate/up and down kernels launch in the layouts searched on an M4 Pro. Every other device, the M3 included, keeps the previous layouts, as does any shape a layout does not fit. The arithmetic is unchanged. `MLX_SERVE_MOE_ROWS_LAYOUT=0` turns the layouts off; `MLX_SERVE_MOE_ROWS_LAYOUT=1` turns them on on any device.
- `lesto tune` searches those two kernels' launch layouts on your own Mac, only while the Mac is idle (no input for 5 minutes by default, on AC, not in Low Power Mode, no thermal pressure), and stops within seconds of input. It keeps a layout only if the kernel's outputs are identical, bit for bit, to the served layout's, checked at 10 cases for each candidate and on every timed call, and only after two confirmations in two windows. The result is `~/.mlx-serve/tune/kernel-plans.json`, read at the next `lesto serve` start and replacing the built-in entry as a whole. `lesto tune --background` installs a LaunchAgent (macOS shows a "Background Items Added" notice) that removes itself when done; `--stop`, `--show`, `--verify` and `--reset` stop it, show it, re-check it after a macOS update and delete it. [docs/kernel-tune.md](docs/kernel-tune.md) has the details.
- `MLX_SERVE_KERNEL_TABLE` picks the table: unset reads the default file, `off` ignores the local table, any other value is the path of the table to read. A table that fails a check (permissions, checksum, shape, a key field such as the GPU, MLX version or macOS build) is logged once and ignored.
- On an M4 Pro, every reply is byte-identical to 0.25.0's (150 of 150 turns of a multi-turn conversation set, and 210 of 210 general turns) with the default layouts, with `MLX_SERVE_MOE_ROWS_LAYOUT=0`, with `MLX_SERVE_KERNEL_TABLE=off`, and with the table a `lesto tune` run wrote there.
- The LaunchAgent needs a logged-in desktop session: over SSH with no one at the screen, `--background` fails cleanly (exit 2, nothing installed). Whether a LaunchAgent can read the idle time (`HIDIdleTime`) is not yet measured; if `agent.log` shows `hid_idle=unreadable`, use the foreground `lesto tune`.

## 0.25.0

- `MLX_SERVE_MOE_ROWS_LAYOUT=1` launches the drafter verify's two MoE expert kernels (gate/up, and down with its weighted sum) at layouts searched per GPU generation. The table holds one measured entry, the M4 generation (G16, measured on an M4 Pro); every other device, the M3 included, keeps the previous layouts, as does any shape a layout does not fit. The arithmetic is unchanged.
- `MLX_SERVE_DEFERRED_TURN_CLEAR=1` runs the end-of-turn cache clear between passes instead of inside the reply's wait, and `MLX_SERVE_DEFERRED_CACHE_COMMIT=1` moves the finished slot's prefix-cache commit, release and flush to the cleanup drain.
- With all three on and the drafter at k = 3, decode is about 1.9% faster than 0.24.0 on an M4 Pro on 50 multi-turn conversations, twice (+1.9% and +1.9%), and about 2.4% faster on general English and Italian text at 1k, 8k and 16k tokens. Every reply is byte-identical to 0.24.0's (150 of 150, and 210 of 210), and memory is unchanged.
- All three are off by default: without them nothing changes. This release also carries switches that stay off and are not recommended yet: `MLX_SERVE_DRAFTER_OPS_LIMIT_COMMIT=1` and `MLX_SERVE_BUDGET_TRIM_NO_COPY=1` (each slower or not byte-identical in our measurements).

## 0.24.0

- `MLX_SERVE_DRAFTER_HEAD_KEEP_LIST=<file.npy>` cuts the assistant drafter's 4-bit tied head (`MLX_SERVE_DRAFTER_HEAD_4BIT=1`) to a shorter keep list: a 1-D uint32 `.npy` of token ids, ascending and unique, a strict subset of the target's Latin keep list. Any other file stops the server at bind time. A draft then scores only the listed rows and can propose only listed tokens; every kept token is still the target's own choice. Replies can still change, because fewer accepted drafts change the verify's shape and its rounding.
- With an 8,192-token list ranked by how often each token appears in the deployment's own replies, decode is about 3% faster than 0.23.1 on 50 multi-turn conversations, twice (+3.3% and +3.2%), and reply time drops about 1.4%. 109 of 150 replies are byte-identical to 0.23.1's; the changed ones passed a blind meaning read.
- No list ships with Lesto: build it from your own traffic. Without the variable nothing changes.

## 0.23.1

- A drafter round submits its drafts before it builds the verify, so the GPU runs the drafts while the host builds the verify's graph. With the drafter at k = 3 this decodes about 4% faster than 0.23.0 on 50 multi-turn conversations, twice (+4.0% and +4.1%), and every reply is byte-identical to 0.23.0's. `MLX_SERVE_DRAFTS_EARLY_SUBMIT_OFF=1` keeps one submit per round.
- Without a drafter nothing changes.

## 0.23.0

- The Gemma 4 assistant drafter is worth turning on. Served with `--drafter <gemma-4-26B-A4B-it-assistant> --draft-block-size 3` and requests that send `"enable_drafter": true`, it decodes about 9% faster than 0.22.4 on 50 multi-turn conversations, twice (+9.0% and +9.4%), and the reply time per turn drops about 3%. Its replies come from a 3-token verify, so they are not byte-identical to the one-token path's; they passed a blind read against an 8-bit reference.
- A drafter round's kept tokens now advance the top-k schedule, so its verify routes at the scheduled expert count, like a one-token step. Before, the verify stayed at 8 experts for the whole reply.
- The assistant drafter reads the sliding layers' newest window of the target's cache and rotates its query at the last cached position (`MLX_SERVE_DRAFTER_CACHE_VIEW_OFF=1` keeps the old path). A partial accept rolls the cache back to the kept tokens and takes the next hidden from the verify, with no restore and no re-forward (`MLX_SERVE_DRAFTER_ROLLBACK_OFF=1` keeps them). The draft step stays in bf16 (`MLX_SERVE_DRAFTER_BF16_SCALARS_OFF=1` keeps float32). A 2-to-8-token block takes the decode step's attention paths.
- Options the measured setup also turns on: `MLX_SERVE_DRAFTER_LAYERS_4BIT=1` (the drafter's projections in 4-bit), `MLX_SERVE_DRAFTER_HEAD_4BIT=1` (its tied head over the target's Latin keep list, 4-bit), `MLX_SERVE_DRAFTER_NO_SNAPSHOT=1` (no cache snapshot when the rollback can serve every outcome) and `MLX_SERVE_DRAFTER_NO_GATE=1` (the runtime gate never switches drafting off), with the target's `MLX_SERVE_HEAD_SHORTLIST=1` and `MLX_SERVE_TOPK_SCHEDULE=8:4:6`.
- Without a drafter nothing changes: 0.22.5's one-token path is the same code.

## 0.22.5

- The server path builds the first reply token's one-token forward before it waits for the first token's sample, in place of the step 0 shim, and every other reader of a fresh generator reads the same seeded state. `MLX_SERVE_FIRST_TOKEN_PREBUILD_OFF=1` keeps the shim.
- Error exits no longer leak MLX arrays: generator init frees the prefill logits, the first token and the next step's logits on every failure, the forward frees its locals when a later op fails, and a cached Metal kernel config is cleared after its free. Replies are unchanged; only failing requests are affected.
- With the 0.22.4 threshold select, decode speed rises by about 0.9% on 50 multi-turn conversations, twice, against 0.22.3, with every reply byte-identical there.

## 0.22.4

- The 2-bit head shortlist's top-64 pick on bf16 scores runs as a threshold select: each 4,096-score chunk finds its exact 64th score with two 256-bin histograms of a 16-bit order key, keeps its 64 in index order, and one threadgroup repeats this on the candidates and ranks the final 64. No score is sorted that is dropped. The picked indices, their order and the token are unchanged, ties, signed zeros, NaN and infinities included. `MLX_SERVE_HEAD_SHORTLIST_THRESHOLD_OFF=1` keeps the sorted chunks.
- It raises decode speed by about 0.8% on 50 multi-turn conversations, twice, with every reply byte-identical there.

## 0.22.3

- The 2-bit head shortlist's top-64 pick runs as one fused select (256-score chunks sorted in threadgroup memory, then a merge with one thread a chunk) in place of the argpartition, and the head tail takes its argmax over the 64 rescored scores alone; the picked indices, their order and the token are unchanged. `MLX_SERVE_HEAD_SHORTLIST_SELECT_OFF=1` keeps the argpartition, `MLX_SERVE_HEAD_SHORTLIST_ARGMAX_OFF=1` the kept-row argmax.
- The head tail reduces under the request's own mask from the first forward, so it never prepares the other mask first.
- Together they raise decode speed by about 0.8% on 50 multi-turn conversations, twice, with every reply byte-identical there.

## 0.22.2

- A prefill chunk whose output nobody reads no longer evaluates the last layer, and the chunk boundary still evaluates the quantized cache's scales and biases; `MLX_SERVE_PREFILL_LAST_EVAL_SKIP_OFF=1` keeps the old evaluation.
- The suppression mask's -inf is written in the logits' own dtype where an argmax follows, so a bf16 logits row is not widened to float32 before a greedy draw; sampling, probabilities and residual callers keep the float32 row. `MLX_SERVE_SUPPRESS_MASK_NATIVE_DTYPE_OFF=1` keeps the float32 scalar everywhere.
- Together they cut the summed reply time of three-turn conversations by about 2.7% on 50 multi-turn conversations, twice, at unchanged decode speed, with every reply byte-identical there.

## 0.22.1

- With thinking off, a Gemma 4 reply can no longer open the thought channel. The chat template already closes an empty thought before the answer, but the model could still emit `<|channel>` as its first token, and the whole reply then came back as `reasoning_content` with empty `content`. Such requests now sample with that token masked; tool and JSON `response_format` requests are unchanged, and `MLX_SERVE_ANSWER_MASK_OFF=1` turns it off. On 50 multi-turn conversations every reply of the default configuration is unchanged.
- The greedy prompt-lookup verify honours the sampling mask like the other verify paths.

## 0.22.0

- `MLX_SERVE_HEAD_SHORTLIST=1` picks each decode step's 64 likeliest tokens from a 2-bit copy of the Latin-script head and rescores them with the 4-bit head. On Gemma 4's 26B-A4B it decodes about 1.3% faster on 50 multi-turn conversations, twice, with every reply identical there; replies can change where the right token falls outside the 64, so it is off unless set. It holds about 0.15 GB more memory.

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
