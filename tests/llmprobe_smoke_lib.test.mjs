// Unit tests for the pure decision layer of tests/llmprobe_smoke_test.sh.
//
// The shell script owns only process lifecycle (boot a server, run npx
// llmprobe, kill it). Everything that DECIDES something — which models are
// probeable, which family they belong to, which spec/KV cells to run, whether
// a cell actually engaged the spec path it claimed — lives here, where it can
// be tested without weights.
//
//   node --test tests/llmprobe_smoke_lib.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  familyKey,
  isProbeable,
  selectRepresentatives,
  planCells,
  parseSpecStats,
  cellVerdict,
  buildSummary,
  findDrafters,
  pickDrafterFor,
} from "./llmprobe_smoke_lib.mjs";

const model = (id, arch, extra = {}) => ({
  id,
  capabilities: ["chat", "tool_use", "streaming", "json_schema"],
  bytes_on_disk: 8e9,
  meta: { architecture: arch, is_moe: false, ...(extra.meta ?? {}) },
  ...extra,
});

// ── family grouping ────────────────────────────────────────────────────────

test("familyKey folds _text siblings onto the base arch and splits MoE out", () => {
  assert.equal(familyKey(model("a", "gemma4")), "gemma4");
  assert.equal(familyKey(model("a", "gemma4_text")), "gemma4");
  assert.equal(familyKey(model("a", "gemma4_moe_text")), "gemma4-moe");
  // is_moe is authoritative — a dense-named arch with experts is still MoE
  // (gemma4 26B-A4B is exactly this: model_type "gemma4", 128 experts).
  assert.equal(
    familyKey(model("a", "gemma4", { meta: { architecture: "gemma4", is_moe: true } })),
    "gemma4-moe",
  );
});

// ── what we probe ──────────────────────────────────────────────────────────

test("isProbeable takes MLX chat models only — no gguf, ds4, encoders or media", () => {
  assert.ok(isProbeable(model("ok", "gemma4")));

  assert.ok(!isProbeable(model("ling", "bailing_hybrid")));

  // Engine-served GGUF (llama.cpp / ds4) — out of scope by request.
  assert.ok(!isProbeable(model("g", "gguf")));
  assert.ok(!isProbeable(model("d", "deepseek_v4")));
  // Encoder-only: no chat surface at all.
  assert.ok(
    !isProbeable({ id: "emb", capabilities: ["embeddings"], meta: { architecture: "gemma4" } }),
  );
  // Media models: capabilities carry the modality, never "chat".
  assert.ok(!isProbeable({ id: "flux", capabilities: ["image"], meta: { architecture: "flux2" } }));
  assert.ok(!isProbeable({ id: "tts", capabilities: ["audio"], meta: { architecture: "qwen3_tts" } }));
  // A stub whose config.json couldn't be read has no capabilities — skip it
  // rather than boot a server to find out.
  assert.ok(!isProbeable({ id: "mystery", meta: {} }));
});

test("selectRepresentatives picks one model per family, skipping the giants", () => {
  const models = [
    model("mlx-community/gemma-4-e4b-it-4bit", "gemma4", { bytes_on_disk: 4e9 }),
    model("mlx-community/gemma-4-31b-it-4bit", "gemma4", { bytes_on_disk: 18e9 }),
    model("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma4", { bytes_on_disk: 15e9, meta: { architecture: "gemma4", is_moe: true } }),
    model("x/gemma-4-giant-moe", "gemma4", { bytes_on_disk: 180e9, meta: { architecture: "gemma4", is_moe: true } }),
    model("x/gguf-thing", "gguf"),
  ];
  const reps = selectRepresentatives(models, { maxBytes: 60e9 });
  const ids = reps.map((r) => r.id);

  assert.deepEqual(ids, [
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
  ]);
  // Deterministic tie-break: smallest on disk wins, so a re-run probes the
  // same model and the numbers stay comparable.
  assert.equal(reps.find((r) => r.family === "gemma4").id, "mlx-community/gemma-4-e4b-it-4bit");
  // The giant is excluded by size, not by arch — it must be reported as
  // skipped, never silently dropped (no-silent-caps rule).
  const skipped = selectRepresentatives(models, { maxBytes: 60e9, collectSkipped: true }).skipped;
  assert.ok(skipped.some((s) => s.id === "x/gemma-4-giant-moe" && /too large/.test(s.reason)));
});

test("selectRepresentatives honours an explicit id list verbatim", () => {
  const models = [model("a/one", "gemma4"), model("b/two", "gemma4")];
  const reps = selectRepresentatives(models, { maxBytes: 60e9, only: ["b/two"] });
  assert.deepEqual(reps.map((r) => r.id), ["b/two"]);
});

// ── the cell matrix ────────────────────────────────────────────────────────

test("planCells: baseline is spec-free — PLD is default-ON server-side", () => {
  const cells = planCells(
    { id: "m", family: "gemma4", drafterPath: null },
    { kvCheck: false },
  );
  const none = cells.find((c) => c.id === "none");
  // Without --no-pld the "no speculation" cell would still be running PLD, and without --no-drafter a pack's own
  // drafter/ would still load.
  assert.deepEqual(none.flags, ["--no-pld", "--no-drafter"]);
  assert.equal(none.expectSpec, null);
  // ...and it must PROVE it: a cell that disables speculation and still shows
  // engagements means its flags never reached the server.
  assert.equal(none.expectNoSpec, true);

  // The KV cells deliberately run at the model's default spec config, so
  // engagements there are expected — they must NOT carry the assertion.
  const kv = planCells({ id: "m", family: "gemma4" }, { kvCheck: true }).find((c) => c.id === "kv4");
  assert.ok(!kv.expectNoSpec);
});

test("cellVerdict fails a spec-free cell that engaged anyway — the flags-didn't-land guard", () => {
  const ok = { conformance: { passed: 89, total: 89 } };

  // Live bug this pins: an IFS tab-collapse ate the flags, the 'none' cell
  // booted with the server's defaults (PLD on), and every number in the row was
  // secretly a PLD number. Conformance and throughput both looked perfect.
  const leaked = cellVerdict(
    { id: "none", expectSpec: null, expectNoSpec: true },
    ok,
    { pld: 7, drafter: 0 },
  );
  assert.equal(leaked.ok, false);
  assert.match(leaked.reasons.join(" "), /pld.*7.*--no-pld|flags/i);

  const clean = cellVerdict(
    { id: "none", expectSpec: null, expectNoSpec: true },
    ok,
    { pld: 0, drafter: 0 },
  );
  assert.equal(clean.ok, true);
});

test("planCells: drafter cell only for a Gemma 4 target with a matching assistant", () => {
  const noDrafter = planCells({ id: "m", family: "gemma4", drafterPath: null }, {});
  assert.ok(!noDrafter.some((c) => c.id === "drafter"));

  const cells = planCells({ id: "m", family: "gemma4", drafterPath: "/d/assistant" }, {});
  const d = cells.find((c) => c.id === "drafter");
  assert.deepEqual(d.flags, ["--no-pld", "--drafter", "/d/assistant"]);
});

test("planCells: the KV-quant cells are a crash check on one model, at its real config", () => {
  const plain = planCells({ id: "m", family: "gemma4" }, { kvCheck: false });
  assert.ok(!plain.some((c) => c.id.startsWith("kv")));

  const cells = planCells({ id: "m", family: "gemma4" }, { kvCheck: true });
  const kv4 = cells.find((c) => c.id === "kv4");
  // No --no-pld: the point is "does the server survive the config a
  // user actually runs", so speculation stays at its defaults.
  assert.deepEqual(kv4.flags, ["--kv-quant", "4"]);
});

// ── engagement: perf alone cannot prove a spec path ran ─────────────────────

test("parseSpecStats counts engagements per mode from the server log", () => {
  const log = [
    "[spec-stats] mode=drafter attempts=12 accepted=31 drafted=42 per_draft=73.8%",
    "some other line",
    "[spec-stats] mode=drafter attempts=8 accepted=19",
    "[spec-stats] mode=pld attempts=3",
    "spec-gate: ngram-score=0.004 (novel) -> pld disabled",
  ].join("\n");
  assert.deepEqual(parseSpecStats(log), { pld: 1, drafter: 2 });
});

test("cellVerdict reports a spec cell that never engaged without failing it", () => {
  const ok = { conformance: { passed: 40, total: 40 } };

  // PLD and the drafter ARE gated on the prompt's n-gram score, and llmprobe's
  // prompts are mostly novel — zero engagements there is expected behavior,
  // not a bug. Report it, never fail on it.
  const gated = cellVerdict({ id: "pld", expectSpec: "pld" }, ok, { pld: 0, drafter: 0 });
  assert.equal(gated.ok, true);
  assert.match(gated.reasons.join(" "), /gate/);
});

test("cellVerdict fails a crashed or non-conforming cell", () => {
  const crashed = cellVerdict({ id: "kv4" }, { error: "server exited (rc=139)" }, {});
  assert.equal(crashed.ok, false);
  assert.match(crashed.reasons.join(" "), /rc=139/);

  const broken = cellVerdict({ id: "none" }, { conformance: { passed: 38, total: 40 } }, {});
  assert.equal(broken.ok, false);
  assert.match(broken.reasons.join(" "), /38\/40/);
});

// ── roll-up ────────────────────────────────────────────────────────────────

test("buildSummary flags a cell whose conformance regressed against its own baseline", () => {
  const rows = [
    {
      family: "gemma4",
      cell: "none",
      result: { conformance: { passed: 40, total: 40 }, bench: { decodeTokPerSec: { median: 50 } } },
    },
    {
      family: "gemma4",
      cell: "kv4",
      result: { conformance: { passed: 37, total: 40 }, bench: { decodeTokPerSec: { median: 55 } } },
    },
  ];
  const s = buildSummary(rows);
  assert.equal(s.regressions.length, 1);
  assert.match(s.regressions[0], /gemma4.*kv4.*40 → 37/);
});

test("buildSummary names the fastest cell per family — the birds-eye answer", () => {
  const rows = [
    { family: "gemma4-moe", cell: "none", result: { bench: { decodeTokPerSec: { median: 60 } } } },
    { family: "gemma4-moe", cell: "drafter", result: { bench: { decodeTokPerSec: { median: 79.5 } } } },
    { family: "gemma4-moe", cell: "pld", result: { bench: { decodeTokPerSec: { median: 58 } } } },
  ];
  const s = buildSummary(rows);
  const f = s.families.find((x) => x.family === "gemma4-moe");
  assert.equal(f.bestCell, "drafter");
  assert.equal(f.bestDecode, 79.5);
  assert.equal(f.speedup, 1.33); // vs the spec-free baseline
});

// ── on-disk companions (drafters) ────────────────────────────

test("findDrafters / pickDrafterFor pair a Gemma 4 target with its assistant", () => {
  const root = mkdtempSync(join(tmpdir(), "smoke-"));
  const put = (rel, cfg) => {
    mkdirSync(join(root, rel), { recursive: true });
    writeFileSync(join(root, rel, "config.json"), JSON.stringify(cfg));
  };
  put("mlx-community/gemma-4-E4B-it-assistant-bf16", { model_type: "gemma4_assistant" });
  put("mlx-community/gemma-4-31B-it-assistant-bf16", { model_type: "gemma4_assistant" });
  put("mlx-community/gemma-4-e4b-it-4bit", { model_type: "gemma4" });

  const drafters = findDrafters(root);
  assert.equal(drafters.length, 2);

  // Matched on the size token, case-insensitively (E4B vs e4b) — and the
  // server still validates the pairing, so a wrong guess errors loudly.
  const d = pickDrafterFor("mlx-community/gemma-4-e4b-it-4bit", drafters);
  assert.match(d, /E4B-it-assistant/);
  assert.equal(pickDrafterFor("x/unknown-model-4bit", drafters), null);
  assert.equal(pickDrafterFor("mlx-community/gemma-4-12B-it-4bit", drafters), null);
});
