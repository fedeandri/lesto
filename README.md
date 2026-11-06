Lesto is an inference engine that runs Gemma 4 models on Apple silicon Macs.

It is a personal experiment, shared as it is. I built it to learn how inference engines work, and because I build AI text and voice agents and wanted a local alternative to OpenAI models, fast enough to replace them in those agents. It is Gemma 4 only for now because most of the agents I build for my clients speak Italian, and of the models I tested, Gemma 4 speaks the best Italian natively. I may add other models later, if I find one worth testing.

Lesto starts from [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu, a much broader project. I removed every other model family, media type and platform, then tuned what was left for Gemma 4 text. If you need images, other models or a maintained product, use mlx-serve or [mlx-lm](https://github.com/ml-explore/mlx-lm).

On an M4 Pro, one user at a time, Lesto decodes Gemma 4 26B-A4B 17 to 26% faster than the latest releases of five other Mac engines at a 1k-token prompt, and more at 8k.

- Runs offline, with no Python and no account, behind an HTTP API for chat, tool calls, JSON schemas and reasoning.
- Apple silicon Macs, M1 or later; tested on M3 Pro and M4 Pro. Download a binary for macOS 14 or later, or build from source.
- [A 4-bit checkpoint](https://huggingface.co/xpressionist/lesto-gemma-4-26b-a4b-text-4bit) made for it: closer to the source model than the standard 4-bit, and about 5% faster on Latin-script languages.
- Text only, Gemma 4 only for now. A request with image, video or audio content gets an HTTP 400. [docs/changes-from-mlx-serve.md](docs/changes-from-mlx-serve.md) says what changed from mlx-serve 26.10.3.

## Speed

Gemma 4 26B-A4B, `mlx-community`'s standard 4-bit checkpoint unless the row says otherwise. Each engine ran with its default flags, one at a time, serving one user at a time: 15 cold requests per prompt size for a 200-token reply. M4 Pro, 20 GPU cores, 64 GB, macOS 26.7.2, on 2026-11-06.

| Engine | Decode tok/s, 1k / 8k | First text s, 1k / 8k | 200-token reply s, 1k / 8k | First text on later turns s, 1k / 8k |
|---|---:|---:|---:|---:|
| Lesto 0.17.1, the checkpoint below | 89.3 / 83.3 | 1.38 / 10.25 | 3.61 / 12.64 | 0.19 / 0.26 |
| Lesto 0.17.1 | 84.5 / 79.1 | 1.38 / 10.34 | 3.74 / 12.86 | 0.19 / 0.26 |
| mlx-serve 26.11.2 | 71.9 / 51.7 | 1.37 / 9.37 | 4.13 / 13.22 | 0.19 / 0.26 |
| oMLX 0.7.4.dev1 | 69.1 / 50.4 | 1.57 / 10.51 | 4.45 / 14.46 | 0.36 / 0.53 |
| vllm-mlx 0.6.0 | 68.7 / 51.1 | 1.40 / 10.40 | 4.29 / 14.29 | 1.52 / 10.89 |
| mlx-vlm 0.7.12 | 68.4 / 50.7 | 1.40 / 10.43 | 4.31 / 14.35 | 1.53 / 10.90 |
| mlx-lm 0.32.2 | 67.3 / 62.6 | 1.44 / 10.90 | 4.39 / 14.10 | 0.23 / 0.28 |

Lesto's lead is in decoding. It does not read a long prompt faster: at 8k, mlx-serve shows first text about 1 second sooner, so on a whole 200-token reply Lesto is only 3% ahead of it there. Several users at once were not measured.

A second launch of the whole benchmark agreed with the first, and the gap between Lesto and each engine held in each of the 3 rounds, on both this standard checkpoint and the one below. To run it yourself: [docs/benchmark.md](docs/benchmark.md).

## The checkpoint

Engine and checkpoint are one experiment. [The checkpoint](https://huggingface.co/xpressionist/lesto-gemma-4-26b-a4b-text-4bit) is a 13 GB 4-bit quantization of Gemma 4 26B-A4B, in standard MLX format, with two changes:

- **Attention rounded with more care.** On one next-token test against the source model, lower is closer: the standard 4-bit scores 0.308, this one 0.171, the 8-bit 0.023. Reading replies blind against the 8-bit, which is twice the size, most pairs were ties. The [card](https://huggingface.co/xpressionist/lesto-gemma-4-26b-a4b-text-4bit) has the counts and their limits.
- **A Latin-script output head.** Lesto scores only the output tokens whose text is Latin script, which makes decoding about 5% faster. It is on by default with this checkpoint. **Replies in other scripts need it off:** set `MLX_SERVE_LATIN_HEAD_OFF=1` for Russian, Chinese, Arabic, Greek, Vietnamese or emoji, or those replies come out wrong. The card lists what I checked.

## Install

The binary runs on macOS 14 or later and leaves out MLX's kernels for M5 chips; on an M5, build from source to get them. It decodes about as fast as the source build on an M4 Pro, with the same replies on the 20 prompts I compared. It is not signed with an Apple Developer ID, so a copy downloaded with a browser needs the `xattr` line:

```bash
curl -LO https://github.com/fedeandri/lesto/releases/latest/download/lesto-macos14-arm64.tar.gz
tar xzf lesto-macos14-arm64.tar.gz
cd lesto-macos14-arm64
xattr -dr com.apple.quarantine .
./zig-out/bin/lesto --version
```

Each release lists the tarball's SHA-256.

## Build

You need an Apple silicon Mac on macOS 26, Xcode 26.2 or later and [Homebrew](https://brew.sh) with CMake (`brew install cmake`). Xcode 26 ships the Metal compiler as a separate download, which the first build needs:

```bash
xcodebuild -downloadComponent MetalToolchain   # once
./scripts/fetch-zig.sh          # the pinned Zig into .zig-toolchain/
export PATH="$PWD/.zig-toolchain:$PATH"
scripts/build-mlx.sh            # the vendored MLX and mlx-c sources in lib/
zig build -Doptimize=ReleaseFast
```

The binary is `zig-out/bin/lesto` and needs macOS 26.2. The commands it knows are `serve`, `--help` and `--version`.

A build for macOS 14 is made on a macOS 26 Mac, in a separate copy of the tree where the first three commands above have run (the script needs `zig` on `PATH`): `scripts/build-macos14.sh <out-dir>` (Xcode 26.5 or later). It builds MLX itself and checks that the binary and its libraries target macOS 14.0.

## Run

Lesto loads MLX-format models. MLX conversions of Gemma 4 are published on [Hugging Face](https://huggingface.co/mlx-community); put each model's folder under one directory, for example `~/models`:

```bash
./zig-out/bin/lesto serve --model-dir ~/models --port 11234
```

`--model-dir <dir>` is that folder; the server discovers the models at start and loads one on the first request that names it. `--port` defaults to 11234. `lesto --help` lists every flag. `lesto --version` prints the Lesto version and the mlx-serve release it is based on.

**The server binds `0.0.0.0` by default**, so it is reachable from the local network. Use `--host 127.0.0.1` to keep it on the machine, and `--api-key <token>` to require a key; see [SECURITY.md](SECURITY.md).

Lesto keeps mlx-serve's paths: models are discovered under `~/.mlx-serve/models` when `--model-dir` is not given, logs go to `~/.mlx-serve/logs/`, and saved prefixes persist under `~/.mlx-serve/kv-cache`. The `MLX_SERVE_*` environment switches keep their names too.

## HTTP API

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | Chat, with streaming, tool calls, JSON schemas and reasoning |
| `POST /v1/embeddings` | Embeddings |
| `GET /v1/models` | The models found under `--model-dir` |
| `POST /v1/models/rescan` | Look for models again |
| `GET /v1/load-progress` | Progress of a model that is loading |
| `GET /health`, `HEAD /` | Liveness |
| `GET /props` | Server and model properties |
| `GET /metrics`, `GET /metrics.json` | Counters |
| `POST /tokenize`, `POST /detokenize` | The loaded model's tokenizer |

An unknown path answers 404. A request whose `model` is empty, `lesto` or `mlx-serve` reaches the default model. With `--api-key` set, requests from other machines need the key, and a login page asks for it.

## Acknowledgements

Lesto builds on mlx-serve by David Dalcu, and a small fix found here went back to it ([PR #1042](https://github.com/ddalcu/mlx-serve/pull/1042), in 26.11.2). It also builds on [MLX](https://github.com/ml-explore/mlx), [mlx-c](https://github.com/ml-explore/mlx-c), [mlx-lm](https://github.com/ml-explore/mlx-lm), [jinja.cpp](https://github.com/wangzhaode/jinja.cpp), [nlohmann/json](https://github.com/nlohmann/json), [Sushi](https://github.com/beamivalice/sushi), [oMLX](https://github.com/jundot/omlx), [MTPLX](https://github.com/youssofal/mtplx), [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) and [TensorFold](https://github.com/ashhart/TensorFold). [NOTICE](NOTICE) lists every third-party work with its license and what Lesto took from it.

## License

MIT, see [LICENSE](LICENSE). Bundled third-party code keeps its own license: [NOTICE](NOTICE) lists it with the required attributions, and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) has the Apache license text.

Lesto is an independent project and is not affiliated with, sponsored by, or endorsed by Google LLC or Apple Inc. Gemma is a trademark of Google LLC. Apple, Mac and Apple silicon are trademarks of Apple Inc., registered in the U.S. and other countries and regions. Lesto does not include Gemma model weights; Google releases Gemma 4 under the [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).
