# Benchmark

`scripts/bench.py` runs the same tests the same way every time, so a result can be compared with an earlier one. The prompts, sizes, rounds and reply length are constants in the script, not options. Changing them means raising `METHOD_VERSION`, and runs of different versions are not compared.

## What it measures

One server runs at a time, and each one is stopped before the next starts.

**Cold requests.** Three rounds, with the engine order rotating each round, and five greedy requests at each prompt size (about 1.2k and 8k tokens). Each reply is 200 tokens. Each prompt's first line is unique to the run and the request, so no engine can reuse a prefix, even one saved to disk earlier.

- **Decode tok/s:** completion tokens minus one, over the time from the first streamed text to the last. This excludes reading the prompt.
- **First text:** from sending the request to the first streamed text.
- **200-token reply:** first text plus decode time. The length is fixed so engines compare fairly; it is not the length of a natural reply.

**Conversations.** Three conversations at each size: a document, then two follow-up questions that resend the history. Turn 0 is cold. The later turns show what the engine reuses, which is what an interactive agent feels.

An engine is dropped and listed with its error, and none of its measurements are scored, when it fails to start, fails a request, sends an error in the stream, ends a stream with no finish reason, returns a cold reply of anything but 200 tokens, reports no token count, buffers tokens into too few stream events to time (under one event per 20 tokens, or more than 2% of the characters in the first event, which would inflate decode speed), or spends tokens on reasoning with thinking off. The run then exits 2. A run that did not finish makes no speed claim.

## Running it

```bash
scripts/bench.py run --mode release --lesto zig-out/bin/lesto --previous ../previous/zig-out/bin/lesto
scripts/bench.py run --mode field --lesto zig-out/bin/lesto
scripts/bench.py compare bench-results/<old> bench-results/<new>
```

Add `--smoke` first: one short pass that proves every engine answers. It is never a result.

- **`release`** compares a build with the previous one, for tracking progress.
- **`field`** compares a build with other Mac engines at whatever versions are installed.

`scripts/bench-config.json` names the engines and checkpoints. Edit its paths, or pass your own file with `--config`. Each run writes a new folder, by default under `bench-results/`, and never overwrites one: `cold.jsonl`, `warm.jsonl`, `report.md`, the server logs, and `meta.json` with the machine, macOS version, load, engine versions and a hash of each checkpoint's files.

## Reading the results

The report says an engine is slower than Lesto only when all of these hold at the 1k prompt, on a checkpoint both ran: Lesto's median decode speed is at least 5% higher, the gap is larger than Lesto's gap against itself (`lesto` and `lesto-again` are the same binary under two names), and Lesto wins every round.

`compare` refuses runs from a different chip, GPU core count, memory size or method, and smoke or unfinished runs. It skips the cells of a checkpoint that changed, and notes a changed macOS version. Numbers from different machines say little about each other.

The tool measures speed only. It says nothing about reply quality.
