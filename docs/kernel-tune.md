# `lesto tune`: kernel layouts searched on your own Mac

This page describes what `lesto tune` does for Gemma 4 26B-A4B and what `lesto serve` does with its result. The README has the short version.

## What it tunes

Gemma 4 26B-A4B's verify step (the drafter's 2-to-8-token check, and the batched decode tick) runs two MoE expert kernels through the engine's "rows" dispatch: gate/up, and down with its weighted sum. Each launches in a layout (how many simdgroups a threadgroup holds, how the grid axes are ordered, how many top-k slots a simdgroup loops over). A layout changes who computes an output, never the arithmetic: the same lane reads the same packs in the same order, with the same roundings. So a layout can be checked against the served one bit for bit, and `lesto tune` keeps only layouts that pass.

The down kernel also has two bodies, the source text a layout runs. The **scalar body** has always served: per-tile values (weight and scale bases, scale, bias) and accumulators are scalars. The **array body** holds the same values in 1-element arrays. Both do the same arithmetic per value, bit for bit, so a body is checked against the served kernel exactly like a layout; which one is faster depends on the GPU. A down plan therefore names a layout and a body. The array body runs the layout dispatch at every layout, the served triple (SPS 1, TPT 1, AX 0) included; only the scalar body at the served triple runs the served kernel itself. A layout that does not fit a call (a top-8 or top-4 call under SPS 6, or an NVFP4 bank) runs the served kernel whatever its body.

The plan space (plan space version 2, `TUNE_PLAN_SPACE_VERSION`):

| kernel | parameters | plans | candidates (the served plan is the reference, not a candidate) |
|---|---|---|---|
| gate/up | SG in {1, 2, 4, 8, 16, 32}, AX in {0, 1} | 12 | 11 |
| down | SPS in {1, 2, 3, 6}, TPT in {1, 2, 4}, AX in {0, 1}, body in {scalar, array} | 48 | 47 |

That is 58 candidates in all. Down's candidates run SPS outermost, then TPT, then AX, then the body (scalar before array), so the first is SPS 1, TPT 1, AX 0 with the array body.

Served plans: gate/up SG 8, AX 0; down SPS 1, TPT 1, AX 0, scalar body. Lesto's built-in table has two entries (not phones). The M4 generation: gate/up SG 32, AX 1; down SPS 6, TPT 1, AX 1, scalar body (a built-in entry names its body like any plan), found on one M4 Pro. The M3 generation: down SPS 6, TPT 4, AX 0, array body, about 15% faster than the served kernel; gate/up stays served, because its gain in the worse routing regime is under the 2% floor. It was found by `lesto tune` on one M3 Pro. Another part of either generation uses its entry unmeasured, and no Mac gets it identity-checked on its own macOS build until `lesto tune --verify` runs.

Only Gemma 4 26B-A4B's expert shape is tuned (hidden 2,816, expert width 704, 4-bit affine, group 64, bf16 activations). Other shapes and quantizations get the served layouts, or the built-in ones where those apply today.

## Files

All under `~/.mlx-serve/tune/` (the folder is `0700`, files `0600`):

| file | content |
|---|---|
| `kernel-plans.json` | the table |
| `progress.jsonl` | one line per finished batch of an unfinished run, so a run resumes |
| `tune.log` | the human log: windows, batches, decisions, the four times of each stop |
| `lock` | the tune's `flock` file; holds the running tune's PID and its start time |
| `install.lock` | serializes `--background` installs and the LaunchAgent's removal |
| `agent.log` | stdout and stderr of the LaunchAgent |

Every `lesto tune` command and the engine use one table path: the file named by `MLX_SERVE_KERNEL_TABLE` when that variable holds a path, else `~/.mlx-serve/tune/kernel-plans.json`. With `MLX_SERVE_KERNEL_TABLE=off` (or empty) the commands use the default path and say the engine will not read it. A table path that names the tune folder, one of the tuner's own files (compared ignoring ASCII case) or the LaunchAgent's plist is refused with exit 2: no command writes or deletes those as a table.

The tuner checks ownership before it writes: a file or folder in `~/.mlx-serve/tune/` that another user owns, a regular file with a second hard link, or a file with an access control list is refused. A mode wider than private on a file of yours is tightened, except a lock file, which is replaced by a new one.

## The table

One JSON object in compact form, written by `std.json`; fields in this order. Every number is an integer (times in nanoseconds), so the checksum's re-serialization is exact. The example below is pretty-printed, and **every number, version and date in it is illustrative**, not a measurement.

```json
{
  "schema": 2,
  "key": {
    "gpu_arch": "applegpu_g16s",
    "gpu_cores": 20,
    "hw_model": "Mac16,7",
    "os_build": "25B78",
    "kernel_hash": "<64 hex characters>",
    "mlx_version": "<version>"
  },
  "shape": { "hidden": 2816, "inter": 704, "bits": 4, "group": 64, "dtype": "bfloat16" },
  "plan": {
    "gateup": { "sg": 32, "ax": 1 },
    "down": { "sps": 1, "tpt": 1, "ax": 0, "body": 1 }
  },
  "evidence": {
    "gateup": {
      "ref_ns": 90000,
      "pick_ns": 82000,
      "confirm": [
        { "median_ns": -8000, "lo_ns": -9500, "hi_ns": -6500, "window": 2 },
        { "median_ns": -7500, "lo_ns": -9000, "hi_ns": -6000, "window": 2 },
        { "median_ns": -8100, "lo_ns": -9600, "hi_ns": -6600, "window": 3 },
        { "median_ns": -7400, "lo_ns": -8900, "hi_ns": -5900, "window": 3 }
      ]
    },
    "down": {
      "ref_ns": 70000,
      "pick_ns": 64000,
      "confirm": [
        { "median_ns": -6000, "lo_ns": -7500, "hi_ns": -4500, "window": 2 },
        { "median_ns": -5800, "lo_ns": -7300, "hi_ns": -4300, "window": 2 },
        { "median_ns": -6100, "lo_ns": -7600, "hi_ns": -4600, "window": 3 },
        { "median_ns": -5900, "lo_ns": -7400, "hi_ns": -4400, "window": 3 }
      ]
    }
  },
  "identity": {
    "plans_checked": 58,
    "cases": 10,
    "calls_compared": 12345,
    "os_build": "25B78",
    "all_identical": true,
    "excluded": [ { "kernel": "down", "cand": "sps=3 tpt=4 ax=0 body=0", "reason": "bits differ in case 4" } ]
  },
  "info": {
    "lesto_version": "<version>",
    "tuned_at": "2026-01-01T03:00:00Z",
    "signals": "real",
    "windows_used": 3,
    "batches_kept": 90,
    "batches_discarded": [
      { "reason": "input", "n": 1 }, { "reason": "power", "n": 0 }, { "reason": "low_power", "n": 0 },
      { "reason": "thermal", "n": 0 }, { "reason": "sleep", "n": 0 }, { "reason": "stopped", "n": 0 },
      { "reason": "gate", "n": 2 }
    ],
    "not_measurable": [],
    "served_reasons": { "gateup": null, "down": null }
  },
  "checksum": "<64 lowercase hex characters>"
}
```

- **`plan`**: per kernel, a layout or `"served"`. A down layout is `{"sps", "tpt", "ax", "body"}`; `body` is `0` for the scalar body and `1` for the array body, an integer like `ax`, and it is required. `"served"` is a result, not a gap: it says no candidate beat the served plan with confirmed evidence, and for down it means exactly the scalar body at SPS 1, TPT 1, AX 0, the served kernel. The served triple with the array body is not `"served"`: it is a layout object with `"body": 1`. A valid table whose plan is `"served"` for both kernels turns the switch off on a Mac whose built-in entry would have turned it on.
- **`evidence`**: for a kernel with a pick, `ref_ns` and `pick_ns` (the reference and the pick's median time a call in the worse regime of the first confirmation) and four confirmations in the order confirmation 1 `indep`, 1 `shared`, 2 `indep`, 2 `shared`. Each holds the median of candidate minus reference (negative is a gain), its 95% bootstrap interval, and the window the batch ran in. `null` for a `"served"` kernel.
- **`identity`**: how many plans the identity pass checked (58 in a full run), the 10 cases, how many kernel calls were compared, the macOS build they ran on, `all_identical` (always `true` in a table that is written: every layout the table names was identical), and `excluded`, every candidate dropped for differing bits or a refused launch with its reason.
- **`info`**: the Lesto version, the UTC time, `"real"` or `"scripted"` signals, the windows used, the batches kept and discarded by reason (seven entries, in this order, zeros included), the candidates that could not be timed quietly, and for a `"served"` kernel its reason. `lesto tune --show` reads all three lists from the table.

The `served_reasons` values: `no_identical_candidate` (every candidate was excluded for differing bits or a refused launch), `no_measurable_candidate` (every other candidate could not be timed quietly), `no_gain_in_screen` (none scored below zero), `no_measurable_finalist`, `final_stage_noisy`, `confirm1_not_confirmed`, `confirm2_not_confirmed`, `below_floor`, `pick_bits_differ`, `confirm2_not_measurable`.

### The key, and why each field is in it

A table is only good on the Mac and binary that produced it. The reader compares these fields in this order and names the first that differs:

| field | source | why |
|---|---|---|
| `gpu_arch` | the Metal device's architecture name (for example `applegpu_g16s`) | a table is measured on one GPU generation |
| `gpu_cores` | the GPU core count | a different core count is a different GPU part |
| `hw_model` | `sysctl hw.model` | a different model is a different machine |
| `kernel_hash` | SHA-256 over the kernel sources and layout rules (below) | a table is tied to the kernels it was tuned for |
| `mlx_version` | MLX's own version string | MLX compiles and launches the kernels |
| `os_build` | `sysctl kern.osversion` | the identity check ran on one macOS build; another build is checked again |

`os_build` is checked last. A table that fails only on `os_build` passed every other check, which is what lets `lesto tune --verify` repair it with the identity pass alone instead of a full tune. A change in any other field needs `lesto tune`.

### `kernel_hash`

SHA-256 over eight inputs in this order, each prefixed by its byte length as a little-endian u64:

1. the affine gate/up rows kernel source (the served one);
2. the affine down-reduce rows kernel source (the served one);
3. the gate/up source of the alternative layouts;
4. the down-reduce source of the alternative layouts (the scalar body);
5. the down-reduce source of the array body;
6. the shared affine kernel header;
7. the text of the three rule functions that decide which layout fits (`downredLanesFor`, `moeRowsLayoutGateUp`, `moeRowsLayoutDown`), cut from the engine's own source file at compile time;
8. `TUNE_PLAN_SPACE_VERSION` as a little-endian u32.

Editing a rule function, a kernel source or the plan space changes the hash, so an old table is refused with `key_mismatch` on `kernel_hash` and `lesto tune` is the fix. The hash is computed once per process.

### The checksum

`checksum` is the SHA-256, lowercase hex, of the compact serialization of the table with `checksum` set to the empty string. The reader parses the file into the typed table (an unknown field is an error), serializes it again, hashes it and compares. It catches a truncated write, disk corruption and a hand edit. It is not a security boundary: whoever can write the file can recompute it, which is why the permission checks below come first.

### Atomic write

The table is written to `<target>.tmp-<pid>` in the target's own folder, `fsync`ed, renamed over the target, and the folder is `fsync`ed. The new file has mode `0600`. A leftover `<target>.tmp-<pid>` from a killed writer is ignored by the reader and removed by the next writer. A table over 1 MiB is neither written nor read.

## What the engine does with a table

At the first MoE model's load, `lesto serve` reads the table once, decides, and logs. A table written later applies at the next start; a running server never reloads it.

### Refusals

The checks run in this order, and the first failure is the reason. A refused table is treated as absent: the engine never deletes or rewrites it. A missing file is not a refusal and prints nothing.

| order | reason | condition |
|---|---|---|
| 1 | `unreadable` | the file or its folder cannot be opened or read, is a symbolic link, a directory or not a regular file, ends the path in `/`, or is over 1 MiB |
| 2 | `unsafe_permissions` | the file or its folder is group- or world-writable, or the file is not owned by the current user; checked on the opened file before any byte is read |
| 3 | `unparseable` | not JSON, or no integer `schema` |
| 4 | `unknown_schema` | `schema` is not 2. Every table written by 0.26.0 (schema 1) is refused here, before the rest of the file is parsed, so the log names the real fix: `lesto tune` writes a new one |
| 5 | `checksum_mismatch` | the checksum does not match |
| 6 | `shape_unknown` | `shape` is not the tuned shape |
| 7 | `plan_value_out_of_range` | a layout value outside SG 1 to 32, SPS and TPT 1 to 32, AX 0 or 1, down `body` 0 or 1 |
| 8 | `plan_outside_space` | a value in range but not in the plan space above |
| 9 | `identity_not_true` | `identity.all_identical` is not `true` |
| 10 | `key_mismatch` | a key field differs from this Mac and binary |

After the schema check the whole file is parsed: a missing or unknown field, or a wrong type, is `unparseable`; so is a schema 2 down layout object without `body`.

The log line for a refusal:

```text
[moe-rows-layout] local table ~/.mlx-serve/tune/kernel-plans.json refused (checksum_mismatch); using the built-in table or the served layouts
```

For `key_mismatch` the parenthesis names the field, both values and the fix, for example `key_mismatch: os_build is <table value> in the table and <this Mac> on this Mac; run `lesto tune --verify``. The fix is `lesto tune --verify` for `os_build` and `lesto tune` for every other field. A path under `$HOME` prints with `~`, and a path longer than 96 bytes is cut at its front.

### Precedence

From strongest to weakest, decided once per process:

1. `MLX_SERVE_MOE_ROWS_LAYOUT` set to anything but `1` (`0` for example): the served layout for every MoE rows call. Neither table is read.
2. `MLX_SERVE_MOE_ROWS_LAYOUT_GATEUP` and `MLX_SERVE_MOE_ROWS_LAYOUT_DOWN` (search-harness parameter lines such as `sg=32 ax=1` and `sps=6 tpt=1 ax=1 body=1`; down's `body` is optional, 0 or 1, and absent means the scalar body): each replaces that kernel's plan as a whole, so a down override without `body` serves the scalar body even where the table or the built-in entry names the array body. They apply only while the switch is on, and a refused value gives that kernel the served layout.
3. The local table, when valid: its plan for both kernels, `"served"` included. A valid local table replaces the built-in entry as a whole; the two kernels are never taken from different tables.
4. The built-in entry for this GPU generation.
5. The served layout.

`MLX_SERVE_KERNEL_TABLE`: unset reads `~/.mlx-serve/tune/kernel-plans.json`; `off` skips step 3; any other non-empty value is the path of the table to read in step 3, with every refusal applied; an empty value is treated as `off`.

The switch, with `MLX_SERVE_MOE_ROWS_LAYOUT` unset, is on when steps 3 and 4 together give a layout other than `"served"` for either kernel. With `MLX_SERVE_MOE_ROWS_LAYOUT=1` it is on on any device.

A plan that came from the local table applies only to calls at the tuned shape (gate/up N 704, K 2,816; down N 2,816, K 704; 4-bit, group 64, bf16). Every other call gets the served layout. A built-in plan and an environment override keep their older behaviour: any shape the layout fits.

### The log lines

At the first MoE load, in order: the table's refusal or skip line (when there is one), the switch line, the override-refused lines (when there are any), and the plan line when the switch is on.

```text
[moe-rows-layout] switch on (local table ~/.mlx-serve/tune/kernel-plans.json, tuned 2026-01-01)
[moe-rows-layout] switch off (local table ~/.mlx-serve/tune/kernel-plans.json, tuned 2026-01-01: no confirmed gain on this Mac): the served MoE rows layouts
[moe-rows-layout] local table skipped (MLX_SERVE_KERNEL_TABLE=off)
[moe-rows-layout] switch on (default: gpu gen=16 phone=false has a table entry; MLX_SERVE_MOE_ROWS_LAYOUT=0 turns it off)
[moe-rows-layout] switch off (default: no table entry for gpu gen=17 phone=false): the served MoE rows layouts
[moe-rows-layout] switch on (MLX_SERVE_MOE_ROWS_LAYOUT=1, gpu gen=16 phone=false)
[moe-rows-layout] switch off (MLX_SERVE_MOE_ROWS_LAYOUT set and not 1): the served MoE rows layouts
[moe-rows-layout] MLX_SERVE_MOE_ROWS_LAYOUT_GATEUP refused (MoeRowsLayoutBadValue): the served gate/up layout serves
[moe-rows-layout] MoE rows layouts on: gpu gen=16 phone=false gateup(local) sg=32 ax=1 down(local) sps=6 tpt=1 ax=1 body=0
```

The plan line names each kernel's source in parentheses: `local`, `table` (the built-in entry), `env` (an override), `refused` (an override that failed to parse) or `none`. Every line that prints a down plan (this one, the `down rows ... does not fit` line, the `engaged` line, and `--show`'s and the tune's plan lines) prints `body=0` or `body=1` after `ax`, in the override's own format, so a printed plan parses back as an override. With no local table and `MLX_SERVE_KERNEL_TABLE` unset, the log is what it was before the local table existed.

## The search

### Idle windows

A window opens when every condition below holds. The scheduler checks them every 5 seconds while waiting (`TUNE_WAIT_POLL_S`) and says what is missing:

```text
[tune] waiting: no input for 3m10s of 5m, on AC
```

| condition | signal | rule | checked |
|---|---|---|---|
| no input | the HID idle time (`HIDIdleTime` of `IOHIDSystem`) | at least `--idle-minutes` (default 5, at most one week); an unreadable value never counts as idle | at open, by the monitor, after each batch |
| on AC power | the providing power source | `AC Power` | at open, by the monitor |
| not in Low Power Mode | `NSProcessInfo` | off | at open, by the monitor |
| no thermal pressure | the `com.apple.system.thermalpressurelevel` notification state | 0 (nominal) | at open, by the monitor |
| GPU quiet | the GPU's `Device Utilization %` | the largest of 8 readings 250 ms apart is below `TUNE_GPU_PCT_MAX` (10) | at open only |
| CPU quiet | host CPU ticks | the busy share over `TUNE_SIGNAL_SPAN_S` (2 seconds) is below `TUNE_CPU_PCT_MAX` (25) | at open only |
| memory | `kern.memorystatus_vm_pressure_level` | normal (1) | at open, and before each batch (a higher level closes the window) |

A GPU, CPU or memory reading that does not exist passes; the statistical gate covers it. GPU and CPU are read only at open because, once the tune runs, both readings include the tune's own work. Other load inside a window shows up in the gate instead.

During a window a monitor thread reads input, AC, Low Power Mode and thermal state every 250 ms (`TUNE_MONITOR_MS`) and raises a stop flag when one fails. SIGINT and SIGTERM raise the same flag. The timing loop checks the flag between pairs, abandons the batch, frees the window's GPU buffers and clears MLX's cache; the requirement is under 2 seconds from input to freed buffers (`TUNE_STOP_LATENCY_MAX_MS`). `tune.log` records four monotonic times for each stop (the input, the flag, the batch abandoned, the buffers freed) and the latency. A window builds its timing banks (about 1.7 GB of GPU memory at the tuned shape) when it opens and frees them when it closes, so a waiting tune holds none.

### The discard rule

A batch is kept whole or discarded whole. After each batch the idle time is read again, and the batch is discarded with reason `input` when it is shorter than the batch's duration plus 250 ms, whatever the monitor saw. The other reasons are `power`, `low_power`, `thermal` (the monitor saw the condition fail), `sleep` (the gap between the continuous and absolute clocks over the batch exceeds `TUNE_SLEEP_GAP_MS`, 1,000 ms), `stopped` (SIGINT or SIGTERM) and `gate` (below). Only a batch that passed every check is written to `progress.jsonl` as kept. A closed window means the run waits for the next one.

### Progress and resume

`progress.jsonl` starts with a header naming the Mac's key and the plan space, then one line per batch (phase, kernel, candidate, regime, window, kept or reason, and the timing numbers). A header for another key or plan space replaces the file and the run starts over, saying so: a progress file from plan space 1 (0.26.0) is replaced, never replayed into the 58-candidate run. A last line that is cut or unparseable is ignored and removed before the next append. A resumed run skips kept batches. One exception: final-stage lines of an attempt that never wrote its `stage_done` marker are void, and the whole final stage runs again as a new attempt. The identity phase writes one line per candidate after all 10 cases ran; a candidate cut by a stop runs all 10 again.

### Stages

Every timing batch compares the served layout (the reference) with one candidate through the engine's own dispatch, in one routing regime: `indep` (each row draws its own experts) or `shared` (rows share experts). The method: per pair, both arms at 8 and at 32 calls, in alternating order, the slope between the two sizes as the time a call, 4 warm-up pairs and 21 counted pairs, the median of the paired differences, its quartiles and a bootstrap 95% interval (2,000 resamples). Before each batch one untimed call of its arm builds the kernel configuration, so no timed batch includes a rebuild. Each call's output is also compared bit for bit with the reference's output on the same inputs.

1. **Identity pass** (no timing). For each of the 58 candidates, 10 cases against the served plan: rows, top-k, how many of the first row's experts later rows keep, and the activation:

   | case | rows | top-k | keep | activation |
   |---|---|---|---|---|
   | 1 | 3 | 6 | 0 | GELU table |
   | 2 | 3 | 6 | 3 | GELU table |
   | 3 | 3 | 6 | 6 | GELU table |
   | 4 | 3 | 6 | 2 | SiLU |
   | 5 | 1 | 6 | 0 | GELU table |
   | 6 | 2 | 6 | 3 | GELU table |
   | 7 | 4 | 6 | 3 | GELU table |
   | 8 | 3 | 8 | 3 | GELU table |
   | 9 | 3 | 4 | 2 | GELU table |
   | 10 | 8 | 6 | 0 | GELU table |

   A candidate with any differing bit, or whose launch MLX refuses, is excluded with its reason (`bits differ in case <n>` or `launch refused: <message>`) and never timed.
2. **Calibration**, at the start of each window after the identity pass: an A/A batch (the reference against itself) for each kernel in each regime, 4 batches. A round passes when every batch has a median difference within `TUNE_CAL_MEDIAN_PCT` (2%) of its reference median and a paired-difference interquartile range below `TUNE_CAL_IQR_PCT` (8%). A failed round waits `TUNE_CAL_RETRY_S` (60 s), rechecks the idle conditions and runs again, up to `TUNE_CAL_ROUNDS` (3) rounds; after the third the window closes as noisy. The passing round sets the window's calibration reference per kernel and regime.
3. **Per-batch gate.** A timing batch passes when its reference median is within `TUNE_BATCH_REF_DRIFT_PCT` (8%) of the window's calibration reference for its kernel and regime and its interquartile range is below `TUNE_BATCH_IQR_PCT` (12%) of its own reference median. A failing batch is discarded with reason `gate` and charged to its candidate. A candidate charged `TUNE_MAX_GATE_FAILS` (5) gate failures over the whole run is not measurable quietly: it leaves every later phase and is listed in `info.not_measurable`.
4. **Screen.** Every candidate that passed identity is timed in both regimes, one kept batch each (at most 116 batches, 2 for each of the 58 candidates). Its screen score is its worse regime, the larger of the two differences as a share of the reference.
5. **Final stage**, in one window. A kernel's finalists are its `TUNE_FINALISTS` (3) best candidates by screen score among those scoring below zero. An attempt times every finalist of both kernels in both regimes (at most 12 batches), then the pick of each kernel, the finalist with the lowest worse-regime result in this attempt, as the first confirmation in both regimes (at most 4 more). An attempt that loses any batch (the window closes, an input, a stop, a gate failure, a bits difference) is abandoned and its lines are void; the next attempt picks its finalists again, so a finalist that became not measurable is replaced. After `TUNE_FINAL_GATE_ABANDONS` (5) attempts abandoned by the gate, every kernel that had finalists is `"served"` with `final_stage_noisy`. Abandonments for other reasons do not count: they come from use of the Mac, and the run waits.
6. **Second confirmation**, in a later window than the final stage's, after that window's calibration: the pick again, both regimes, on fresh pairs and a new bootstrap seed. A gate failure re-queues the batch; a pick that becomes not measurable quietly is `"served"` with `confirm2_not_measurable`; a pick whose bits differ is `"served"` with `pick_bits_differ`.

A pick enters the table only when, in both confirmations and both regimes, the upper end of the 95% interval is below zero, and in both confirmations the gain is at least `TUNE_EFFECT_FLOOR_PCT` (2%) of the reference in both regimes (so in the worse one). Otherwise the kernel records `"served"` with `confirm1_not_confirmed`, `confirm2_not_confirmed` or `below_floor`. A run that records a pick needs at least two windows.

Every `TUNE_*` value is a named constant in `src/tune_stats.zig`, `src/tune_signals.zig`, `src/tune_sched.zig` or `src/kernel_table.zig`. The thresholds are initial values; a later recalibration changes only them.

### Expected GPU time

With no discards, the run needs at most 136 timing batches (the screen's 116, the final stage's 16 and the second confirmation's 4) plus 4 to 12 calibration batches a window. Discards add up to 4 re-timed batches a candidate before it is excluded, and up to 80 batches for abandoned final-stage attempts; an interruption adds the batch or attempt it cut. The printed estimate is about 2 minutes of GPU on an M4 Pro. Measured on an M4 Pro with 0.26.0 (11 + 23 plans): a run with one interruption took 62 s of GPU over 4 windows (121 batches kept, 3 discarded), and the interrupted batch was abandoned 369 ms after the input; 58 candidates are about 1.7 times as many screen batches.

## Commands

Exactly one action flag, or none. Two action flags, an unknown flag or a missing value is a usage error (exit 2, before any lock or file is touched).

| invocation | action |
|---|---|
| `lesto tune` | Prints this Mac's key, the table's status, the candidate count and the expected GPU time. If a valid table exists it says `[tune] the table is current; lesto tune --reset and then lesto tune tunes again` and exits 0. Otherwise it waits for idle windows in the foreground, works through them, writes the table, prints each kernel's result and `applies at the next lesto serve start`. A progress line is rewritten in place on a terminal and printed once per change otherwise. |
| `lesto tune --background` | Says `macOS will show a "Background Items Added" notice for lesto`, then installs and starts the LaunchAgent. Refused (exit 2) while `MLX_SERVE_KERNEL_TABLE` is `off`, because the agent would tune a table the engine does not read. |
| `lesto tune --stop` | Removes the LaunchAgent if installed, and stops a running tune. Progress is kept. |
| `lesto tune --show` | Read-only: no GPU work and no lock (below). |
| `lesto tune --verify` | The identity pass, in an idle window, for the layouts of the table (below). |
| `lesto tune --reset` | Deletes the table and `progress.jsonl`, printing each path removed. Keeps `tune.log`. Refuses (exit 2) while a tune holds the lock. |
| `--idle-minutes <n>` | A whole number from 1 to 10,080 (one week), default 5. Valid with no action flag, `--background` and `--verify`. |

`--check-model` is rejected: `lesto tune: --check-model is not in this release` (exit 2). `--agent` is what the LaunchAgent runs and is not listed in `--help`. `--print-plist-for-test <binary path>` prints the LaunchAgent text for that path and nothing else; it goes alone.

### `--show`

Prints, in order: the table path and its status; each key field with `matches (<value>)` or `table <x>, this Mac <y>`; each kernel's plan (a down plan with its `body=0|1`) with its gain (reference and pick times, and each confirmation's median, interval and window), or `served` with its reason; the source the engine would use now and the lines it would log (the same decision `lesto serve` makes); the candidates excluded by identity, the candidates not measurable quietly and the reason for each served kernel; the progress of an unfinished run (batches kept, discarded by reason, windows used, void final-stage lines, and the same two lists so far); and the LaunchAgent (installed or not, and whether its program path exists). When a built-in entry is in use it adds that the entry is not identity-checked on this Mac's macOS build and that `lesto tune --verify` checks it. It exits 2 when a section failed (the source, progress that exists but cannot be read, the LaunchAgent status), after printing the whole report.

### `--verify`

`--verify` reads the table that `MLX_SERVE_KERNEL_TABLE` names. It ignores `MLX_SERVE_MOE_ROWS_LAYOUT` and the two layout overrides and says so when they are set. It decides the source in order:

- the local table is valid: its plan is checked;
- its only refusal is `key_mismatch` on `os_build` (every other check passed): its own plan is checked;
- otherwise (absent, skipped by `off`, or refused for another reason, which is printed): the built-in entry for this GPU, or nothing when this generation has no entry.

When the plan has a layout, `--verify` waits for an idle window and runs the identity pass for it. On success it prints `[tune] verify: identical (<n> layouts, <m> calls compared)`. For a local table it rewrites `identity` (and `os_build` when that was the mismatch), recomputes the checksum and writes the table atomically to the file it read, so the table is valid at the next server start. For a built-in plan it writes nothing. A differing bit or a refused launch writes nothing and exits 1; for a local plan the output names `MLX_SERVE_KERNEL_TABLE=off` as the immediate undo and `lesto tune --reset`, then `lesto tune` as the fix; for a built-in plan it names `MLX_SERVE_MOE_ROWS_LAYOUT=0` and `lesto tune`.

When there is no layout to check (no source, or `"served"` for both kernels), `--verify` dispatches no GPU work and does not wait for idle. For a table whose `os_build` differs it rewrites `os_build`, says `[tune] the table says "served" for both kernels; os_build refreshed, no GPU check needed` and exits 0: a served kernel runs the served layout, which is the reference every identity check compares against. Otherwise it says `[tune] nothing to verify: the served layouts serve` and writes nothing.

### The lock

`~/.mlx-serve/tune/lock` is opened `O_CREAT` and taken with `flock(LOCK_EX | LOCK_NB)`. The holder writes its PID and its process start time. The default action, `--agent`, `--verify` and `--reset` take it. A second foreground taker prints `[tune] a tune is running (pid <n>)` and exits 2; the agent waits instead, checking every 30 seconds, because it runs once per login and must not end as if it had tuned.

`--stop` tries the lock first. If it gets it, no tune runs, and it signals nothing. If the lock is held, it reads the PID and start time and sends SIGTERM only when that PID's live start time equals the recorded one, and only when the lock file is a private regular file of this user with one link. A PID that was reused by another process has a different start time, so `--stop` never signals another process.

### Exit codes

| code | meaning |
|---|---|
| 0 | done: table written or already current, verify identical or nothing to verify, show, reset, background installed, stop with no tune running or the tune signalled |
| 1 | a check failed: `--verify` found a differing bit or a refused launch |
| 2 | could not run: a usage error, `--check-model`, a lock held, no Metal device, `HOME` unset, the tune folder unusable, a table path that names a tuner file, `launchctl` failed, a GPU error, a `--stop` that could not identify or signal the holder, a `--show` section that failed, an output write that failed |
| 130 | stopped by SIGINT or SIGTERM (or, for the agent, while waiting for the lock); progress kept |

## The LaunchAgent

`lesto tune --background` installs a per-user LaunchAgent:

- label `com.lestoengine.lesto.tune`, plist `~/Library/LaunchAgents/com.lestoengine.lesto.tune.plist`;
- `ProgramArguments`: the absolute real path of the running `lesto` binary, then `tune`, `--agent`, `--idle-minutes`, `<n>`. The first entry is the `lesto` binary itself, never an interpreter or a shell, so macOS's Login Items list shows "lesto";
- `RunAtLoad` true, `KeepAlive` false, `ProcessType` `Standard`, `StandardOutPath` and `StandardErrorPath` both `~/.mlx-serve/tune/agent.log`. 0.26.0 used `Background`: launchd throttles such a job, and on an M3 that made the timings slower and unsteady enough that the noise gate discarded most batches and nothing was picked, while the same binary in the foreground kept every batch. The agent still opens a window only when you are away and the GPU is quiet, and stops at your next input between timing pairs. Unlike a `Background` job it is not deprioritized, so GPU work that starts during a window without any input (a scheduled render, a sync) competes with it on equal terms until the window ends;
- when `MLX_SERVE_KERNEL_TABLE` holds a path, an `EnvironmentVariables` entry carries it (made absolute), because launchd does not pass your shell's environment.

macOS shows a **"Background Items Added"** notice when the agent is installed. The install is one step that completes or leaves the previous state: the plist is staged beside the target, an agent already installed is booted out only once the new file exists, and a failed `launchctl bootstrap` puts the previous plist back. A failed `launchctl` call prints its output and exits 2.

The agent runs `lesto tune --agent`: the default action without the in-place progress line, and with a first line `[tune] agent signals: hid_idle=<seconds or unreadable> ac=<..> low_power=<..> thermal=<..>` in `agent.log`. When it has written the table, or finds a current one at start, it removes its own plist and boots itself out as its last act; being ended by that bootout is its expected end.

**The LaunchAgent needs a logged-in desktop session.** Over SSH with no one logged in at the screen, `launchctl bootstrap` fails (`125: Domain does not support specified action`); `--background` then removes what it staged, says so and exits 2, and the foreground `lesto tune` is the way to tune. **The idle reading under a LaunchAgent works:** on an M3 Pro under macOS 14.4 with a desktop session, `--background` installed and started the agent, its first line read `[tune] agent signals: hid_idle=0 ac=true low_power=false thermal=0`, and its waiting lines counted the time since the last input and fell back to zero on input. If the first line of `agent.log` shows `hid_idle=unreadable` on another Mac, no window ever opens (an unreadable input never counts as idle), and the foreground `lesto tune` is the way to tune.

To remove the agent: `lesto tune --stop`. It deletes the plist, boots the agent out and stops a running tune. By hand: `launchctl bootout gui/$(id -u)/com.lestoengine.lesto.tune` and delete the plist. Installs and removals hold `install.lock`, so a removal cannot race an install.

## `MLX_SERVE_TUNE_FAKE_SIGNALS`

Setting `MLX_SERVE_TUNE_FAKE_SIGNALS=<path>` makes a tune read its idle conditions from a script instead of the Mac. It exists so an unattended machine with no input device can run a tune and script an interruption, and it is in the release binary. A tune on scripted signals prints `[tune] signals: scripted from <path>` and writes `"signals": "scripted"` into the table's `info`, so such a table is never mistaken for a real one.

The file is JSON lines (at most 1 MiB). Each line is an object with `at_ms` (milliseconds from the tune's start, required, never decreasing) and any of:

| field | value | meaning |
|---|---|---|
| `hid_idle_s` | seconds, or `null` | idle time at that moment; it grows with the clock between lines |
| `ac` | `true`, `false` or `null` | on AC power |
| `low_power` | `true`, `false` or `null` | Low Power Mode |
| `thermal` | an integer, or `null` | thermal pressure level (0 is nominal) |
| `gpu_pct` | an integer, or `null` | GPU utilization percent |
| `cpu_pct` | 0 to 100, or `null` | CPU busy share |
| `mem_level` | an integer | memory pressure level (1 is normal) |
| `sleep_ms` | an integer | from this line on, the continuous clock runs this far ahead of the absolute clock (a simulated sleep) |

A field keeps its last value until a later line changes it; `null` reads as unknown from then on. Before the first line the Mac is idle for 0 seconds, on AC, not in Low Power Mode, nominal, with the GPU at 0 and memory normal. An unknown field name, a non-integer where one is required or a time going backwards is an error naming the line.

```text
{"at_ms": 0, "hid_idle_s": 600}
{"at_ms": 8000, "hid_idle_s": 0}
{"at_ms": 20000, "hid_idle_s": 600}
```
