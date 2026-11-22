//! The pure statistics and stage rules of `lesto tune` (spec 5): paired slopes, the bootstrap interval, the calibration and
//! batch gates, the screen score, finalists, the pick and the confirmation rules. No GPU, no clock, no file: every function
//! takes numbers and returns numbers, so the scheduler's tests run on synthetic input.
//!
//! The timing method is a paired timing with a median, ported (not imported) from a standalone micro-benchmark's
//! `timePair` and `median`.
const std = @import("std");

// ── Thresholds. The initial values come from the design note's "Noise detection" section; the caps are this spec's (5.6). ──

/// Calibration: an A/A batch passes with |median difference| at most this share of its reference median (percent).
pub const TUNE_CAL_MEDIAN_PCT: i64 = 2;
/// Calibration: an A/A batch passes with a paired-difference IQR below this share of its reference median (percent).
pub const TUNE_CAL_IQR_PCT: i64 = 8;
/// Seconds to wait between two calibration rounds of one window.
pub const TUNE_CAL_RETRY_S: u32 = 60;
/// Calibration rounds a window may spend before it closes as noisy.
pub const TUNE_CAL_ROUNDS: u32 = 3;
/// Batch gate: the batch's reference median may differ from the window's calibration reference by this share (percent).
pub const TUNE_BATCH_REF_DRIFT_PCT: i64 = 8;
/// Batch gate: the paired-difference IQR must be below this share of the batch's own reference median (percent).
pub const TUNE_BATCH_IQR_PCT: i64 = 12;
/// Gate failures charged to one candidate before it counts as not measurable quietly.
pub const TUNE_MAX_GATE_FAILS: u8 = 5;
/// Finalists a kernel takes into the final stage.
pub const TUNE_FINALISTS: usize = 3;
/// Final-stage attempts the gate may abandon before the kernels record "served".
pub const TUNE_FINAL_GATE_ABANDONS: u32 = 5;
/// A pick must gain at least this share of the reference (percent) in both confirmations, in its worse regime.
pub const TUNE_EFFECT_FLOOR_PCT: i64 = 2;

// ── Method constants (the probe's) ──

pub const WARM: usize = 4;
pub const PAIRS: usize = 21;
pub const CALLS_SMALL: u64 = 8;
pub const MAX_CALLS: u64 = 32;
pub const NCOPY: usize = 4;
pub const NIND: usize = 32;
pub const BOOT: usize = 2000;

// ── Names (the tag names are the strings written to files) ──

pub const Kernel = enum { gateup, down };
pub const Regime = enum { indep, shared };
pub const Phase = enum { identity, cal, screen, final, confirm1, confirm2 };

pub const ServedReason = enum {
    no_identical_candidate,
    no_measurable_candidate,
    no_gain_in_screen,
    no_measurable_finalist,
    final_stage_noisy,
    confirm1_not_confirmed,
    confirm2_not_confirmed,
    below_floor,
    pick_bits_differ,
    confirm2_not_measurable,
};

// ── Slopes and batch statistics ──

/// The four timings of one pair: the reference and the arm, each at `CALLS_SMALL` and at `MAX_CALLS` calls.
pub const PairTimes = struct { ref_small_ns: u64, ref_large_ns: u64, arm_small_ns: u64, arm_large_ns: u64 };

/// The marginal time of one call: the slope between the two batch sizes.
pub fn slopeNs(small_ns: u64, large_ns: u64) f64 {
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    return (@as(f64, @floatFromInt(large_ns)) - @as(f64, @floatFromInt(small_ns))) / span;
}

/// A batch's result, whole nanoseconds: the reference and candidate slope medians, the paired difference's median,
/// quartiles and bootstrap 95% interval.
pub const BatchStats = struct { ref_ns: i64, cand_ns: i64, diff_ns: i64, q1_ns: i64, q3_ns: i64, lo_ns: i64, hi_ns: i64 };

fn medianOf(buf: []f64) f64 {
    std.mem.sort(f64, buf, {}, std.sort.asc(f64));
    return buf[buf.len / 2];
}

fn whole(x: f64) i64 {
    return @intFromFloat(@round(x));
}

/// The statistics of one batch from its counted pairs (the warm-up pairs are not passed in). At most `PAIRS` pairs are
/// read. The bootstrap resamples the paired differences `BOOT` times from `boot_seed`, so a seed gives one interval.
pub fn batchStats(counted: []const PairTimes, boot_seed: u64) BatchStats {
    const n: usize = @min(counted.len, PAIRS); // typed: @min with a comptime-known bound narrows the type, and 3 * n would wrap
    if (n == 0) return .{ .ref_ns = 0, .cand_ns = 0, .diff_ns = 0, .q1_ns = 0, .q3_ns = 0, .lo_ns = 0, .hi_ns = 0 };
    var ss: [PAIRS]f64 = undefined;
    var vs: [PAIRS]f64 = undefined;
    var ds: [PAIRS]f64 = undefined;
    for (counted[0..n], 0..) |p, k| {
        ss[k] = slopeNs(p.ref_small_ns, p.ref_large_ns);
        vs[k] = slopeNs(p.arm_small_ns, p.arm_large_ns);
        ds[k] = vs[k] - ss[k];
    }
    var sorted_d: [PAIRS]f64 = ds;
    const ref = medianOf(ss[0..n]);
    const cand_med = medianOf(vs[0..n]);
    const diff = medianOf(sorted_d[0..n]); // sorts `sorted_d`, so the quartiles read it
    const q1 = sorted_d[n / 4];
    const q3 = sorted_d[(3 * n) / 4];

    var prng = std.Random.DefaultPrng.init(boot_seed);
    const rnd = prng.random();
    var meds: [BOOT]f64 = undefined;
    for (&meds) |*m| {
        var res: [PAIRS]f64 = undefined;
        for (res[0..n]) |*r| r.* = ds[rnd.uintLessThan(usize, n)];
        m.* = medianOf(res[0..n]);
    }
    std.mem.sort(f64, &meds, {}, std.sort.asc(f64));
    return .{
        .ref_ns = whole(ref),
        .cand_ns = whole(cand_med),
        .diff_ns = whole(diff),
        .q1_ns = whole(q1),
        .q3_ns = whole(q3),
        .lo_ns = whole(meds[(BOOT * 25) / 1000]),
        .hi_ns = whole(meds[(BOOT * 975) / 1000]),
    };
}

/// `x / ref`, the way a difference is compared across batches from windows at different clocks. A reference that is not
/// positive gives 0 (no gain claimed).
pub fn share(x_ns: i64, ref_ns: i64) f64 {
    if (ref_ns <= 0) return 0;
    return @as(f64, @floatFromInt(x_ns)) / @as(f64, @floatFromInt(ref_ns));
}

fn absI(x: i64) i64 {
    return if (x < 0) -x else x;
}

// The gates compare in whole nanoseconds times 100, so a threshold sits exactly on its percentage.

/// An A/A calibration batch: |diff| at most `TUNE_CAL_MEDIAN_PCT` of the reference and IQR below `TUNE_CAL_IQR_PCT` of it.
pub fn calPasses(s: BatchStats) bool {
    if (s.ref_ns <= 0) return false;
    return absI(s.diff_ns) * 100 <= TUNE_CAL_MEDIAN_PCT * s.ref_ns and (s.q3_ns - s.q1_ns) * 100 < TUNE_CAL_IQR_PCT * s.ref_ns;
}

/// A calibration round passes when all four A/A batches pass, indexed `[kernel][regime]`.
pub fn calRoundPasses(r: [2][2]BatchStats) bool {
    for (r) |per_kernel| for (per_kernel) |s| {
        if (!calPasses(s)) return false;
    };
    return true;
}

/// A timing batch: its reference median within `TUNE_BATCH_REF_DRIFT_PCT` of the window's calibration reference, and its
/// IQR below `TUNE_BATCH_IQR_PCT` of its own reference median.
pub fn gatePasses(s: BatchStats, cal_ref_ns: i64) bool {
    if (s.ref_ns <= 0 or cal_ref_ns <= 0) return false;
    return absI(s.ref_ns - cal_ref_ns) * 100 <= TUNE_BATCH_REF_DRIFT_PCT * cal_ref_ns and (s.q3_ns - s.q1_ns) * 100 < TUNE_BATCH_IQR_PCT * s.ref_ns;
}

/// A candidate's screen score: its worse regime, the larger of the two difference shares (negative is a gain).
pub fn screenScore(by_regime: [2]BatchStats) f64 {
    return @max(share(by_regime[0].diff_ns, by_regime[0].ref_ns), share(by_regime[1].diff_ns, by_regime[1].ref_ns));
}

// ── Candidates, finalists, the pick ──

pub const CandState = struct {
    /// The identity pass's verdict: null before it ran, false when the bits differ or the launch was refused.
    identical: ?bool,
    /// Excluded for differing bits or a refused launch, in the identity pass or during timing (spec 6.2).
    bits_excluded: bool,
    /// Gate failures charged over the whole run and every phase.
    gate_fails: u8,
    /// The screen's kept batch per regime.
    screen: [2]?BatchStats,

    /// Charged `TUNE_MAX_GATE_FAILS` failures: excluded from every later phase.
    pub fn notMeasurable(self: CandState) bool {
        return self.gate_fails >= TUNE_MAX_GATE_FAILS;
    }

    fn excludedByBits(self: CandState) bool {
        return self.bits_excluded or (self.identical orelse true) == false;
    }

    /// Screened in both regimes, a screen score below zero, not excluded by bits, measurable.
    pub fn eligible(self: CandState) bool {
        if (self.excludedByBits() or self.notMeasurable()) return false;
        const a = self.screen[0] orelse return false;
        const b = self.screen[1] orelse return false;
        return screenScore(.{ a, b }) < 0;
    }

    fn score(self: CandState) f64 {
        return screenScore(.{ self.screen[0].?, self.screen[1].? });
    }
};

/// Up to `TUNE_FINALISTS` (and `out.len`) eligible candidate indices by ascending screen score, ties by plan-space index.
/// Returns how many were written.
pub fn finalists(cands: []const CandState, out: []usize) usize {
    const want = @min(TUNE_FINALISTS, out.len);
    var n: usize = 0;
    while (n < want) {
        var best: ?usize = null;
        for (cands, 0..) |c, i| {
            if (!c.eligible()) continue;
            var taken = false;
            for (out[0..n]) |t| {
                if (t == i) taken = true;
            }
            if (taken) continue;
            if (best == null or c.score() < cands[best.?].score()) best = i;
        }
        out[n] = best orelse break;
        n += 1;
    }
    return n;
}

/// Why a kernel with no eligible candidate records "served": the first that applies (spec 5.4). The reasons read only the
/// candidates' states; `screened_all` says the screen has finished, the only time the scheduler asks.
pub fn noFinalistReason(cands: []const CandState, screened_all: bool) ServedReason {
    _ = screened_all;
    var any_not_bits = false;
    var all_not_bits_unmeasurable = true;
    for (cands) |c| {
        if (c.excludedByBits()) continue;
        any_not_bits = true;
        if (!c.notMeasurable()) all_not_bits_unmeasurable = false;
    }
    if (!any_not_bits) return .no_identical_candidate;
    if (all_not_bits_unmeasurable) return .no_measurable_candidate;
    return .no_gain_in_screen;
}

/// The finalist an attempt picks: the lowest worse-regime candidate median, taken as the larger of the two regimes'
/// candidate-over-reference shares (the same measure as the screen score), ties by position. `final_by_regime` is indexed
/// by finalist, then regime. Returns the position in that slice.
pub fn pickOfAttempt(final_by_regime: []const [2]BatchStats) usize {
    var best: usize = 0;
    var best_worst: f64 = std.math.inf(f64);
    for (final_by_regime, 0..) |f, i| {
        const w = screenScore(f);
        if (w < best_worst) {
            best_worst = w;
            best = i;
        }
    }
    return best;
}

// ── Confirmation ──

pub const Confirm = enum { ok, not_confirmed, below_floor };

/// One confirmation of a pick, `[regime]`: both intervals' upper ends below zero, else `not_confirmed`; then both regimes'
/// median gain at least `TUNE_EFFECT_FLOOR_PCT` of the reference (so the worse regime's is), else `below_floor`.
pub fn judgeConfirm(c: [2]BatchStats) Confirm {
    for (c) |s| {
        if (s.hi_ns >= 0) return .not_confirmed;
    }
    for (c) |s| {
        if (s.ref_ns <= 0 or -s.diff_ns * 100 < TUNE_EFFECT_FLOOR_PCT * s.ref_ns) return .below_floor;
    }
    return .ok;
}

// ── Tests ──

const testing = std.testing;

/// Pairs whose slopes are exactly `ref_slope` and `9000 + jitter[k]` against the reference 10,000 ns.
fn pairsWith(jitter: ?[PAIRS]i64) [PAIRS]PairTimes {
    var out: [PAIRS]PairTimes = undefined;
    for (&out, 0..) |*p, k| {
        const j: i64 = if (jitter) |js| js[k] else 0;
        p.* = .{
            .ref_small_ns = 80_000,
            .ref_large_ns = 320_000,
            .arm_small_ns = 72_000,
            .arm_large_ns = @intCast(288_000 + 24 * j),
        };
    }
    return out;
}

const JITTER = [PAIRS]i64{ 5, -3, 8, 0, 12, -7, 3, 1, -2, 9, 4, -5, 6, 2, -1, 7, -4, 10, -6, 11, 13 };

test "tune stats: slope and medians match a hand computation" {
    const flat = batchStats(&pairsWith(null), 1);
    try testing.expectEqual(@as(i64, 10_000), flat.ref_ns);
    try testing.expectEqual(@as(i64, 9_000), flat.cand_ns);
    try testing.expectEqual(@as(i64, -1_000), flat.diff_ns);
    try testing.expectEqual(@as(i64, -1_000), flat.q1_ns);
    try testing.expectEqual(@as(i64, -1_000), flat.q3_ns);
    // the jittered set holds each of -7..13 once: the median of 21 is the 11th sorted value, 3
    const j = batchStats(&pairsWith(JITTER), 1);
    try testing.expectEqual(@as(i64, 10_000), j.ref_ns);
    try testing.expectEqual(@as(i64, 9_003), j.cand_ns);
    try testing.expectEqual(@as(i64, -997), j.diff_ns);
    try testing.expectEqual(@as(i64, -1_002), j.q1_ns); // sorted[21 / 4], the 6th value, -2
    try testing.expectEqual(@as(i64, -992), j.q3_ns); // sorted[63 / 4], the 16th value, 8
    try testing.expectEqual(@as(f64, 240_000.0 / 24.0), slopeNs(80_000, 320_000));
}

test "tune stats: the bootstrap interval is deterministic for a seed" {
    const pairs = pairsWith(JITTER);
    const a = batchStats(&pairs, 42);
    const b = batchStats(&pairs, 42);
    try testing.expectEqual(a, b);
    try testing.expect(a.lo_ns <= a.diff_ns and a.diff_ns <= a.hi_ns);
    const c = batchStats(&pairs, 43);
    try testing.expect(c.lo_ns <= c.diff_ns and c.diff_ns <= c.hi_ns);
    // a jitter-free set has no spread: the interval is the point
    const flat = batchStats(&pairsWith(null), 7);
    try testing.expectEqual(@as(i64, -1_000), flat.lo_ns);
    try testing.expectEqual(@as(i64, -1_000), flat.hi_ns);
}

fn stats(ref: i64, diff: i64, q1: i64, q3: i64) BatchStats {
    return .{ .ref_ns = ref, .cand_ns = ref + diff, .diff_ns = diff, .q1_ns = q1, .q3_ns = q3, .lo_ns = diff - 10, .hi_ns = diff + 10 };
}

test "tune stats: calibration gate at, inside and outside each threshold" {
    // reference 100,000 ns: 2.0% is 2,000, 8.0% is 8,000
    for ([_]i64{ 1, -1 }) |sign| {
        try testing.expect(calPasses(stats(100_000, sign * 2_000, 0, 0))); // 2.0% passes
        try testing.expect(!calPasses(stats(100_000, sign * 2_010, 0, 0))); // 2.01% fails
        try testing.expect(calPasses(stats(100_000, sign * 1_990, 0, 0)));
    }
    try testing.expect(calPasses(stats(100_000, 0, 0, 7_990))); // IQR 7.99% passes
    try testing.expect(!calPasses(stats(100_000, 0, 0, 8_000))); // 8.0% fails
    try testing.expect(!calPasses(stats(100_000, 0, 0, 8_010)));
    // a round needs all four batches
    const good = stats(100_000, 0, 0, 100);
    const bad = stats(100_000, 5_000, 0, 100);
    try testing.expect(calRoundPasses(.{ .{ good, good }, .{ good, good } }));
    try testing.expect(!calRoundPasses(.{ .{ good, good }, .{ good, bad } }));
    try testing.expect(!calRoundPasses(.{ .{ bad, good }, .{ good, good } }));
}

test "tune stats: batch gate at, inside and outside each threshold" {
    const cal: i64 = 100_000;
    for ([_]i64{ 1, -1 }) |sign| {
        try testing.expect(gatePasses(stats(cal + sign * 8_000, 0, 0, 100), cal)); // drift 8.0% passes
        try testing.expect(!gatePasses(stats(cal + sign * 8_010, 0, 0, 100), cal)); // 8.01% fails
    }
    try testing.expect(gatePasses(stats(cal, 0, 0, 11_990), cal)); // IQR 11.99% passes
    try testing.expect(!gatePasses(stats(cal, 0, 0, 12_000), cal)); // 12.0% fails
}

test "tune stats: the gate's drift is against the reference it is given" {
    const s = stats(9_000, 0, 0, 100);
    try testing.expect(gatePasses(s, 9_000));
    try testing.expect(!gatePasses(s, 10_000)); // 10% drift
}

test "tune stats: the screen score is the worse regime" {
    const a = stats(10_000, -300, 0, 0); // -3%
    const b = stats(10_000, -100, 0, 0); // -1%
    try testing.expectApproxEqAbs(@as(f64, -0.01), screenScore(.{ a, b }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.01), screenScore(.{ b, a }), 1e-12);
    const c = stats(10_000, 50, 0, 0); // +0.5% in one regime makes the score positive
    try testing.expectApproxEqAbs(@as(f64, 0.005), screenScore(.{ a, c }), 1e-12);
}

fn cand(score_pct: i64) CandState {
    const s = stats(10_000, score_pct * 100, 0, 0);
    return .{ .identical = true, .bits_excluded = false, .gate_fails = 0, .screen = .{ s, s } };
}

test "tune stats: finalists are the three best eligible" {
    var cs = [_]CandState{ cand(-4), cand(-2), cand(-6), cand(-3), cand(-5), cand(-7), cand(0) };
    cs[4].gate_fails = 5; // not measurable
    cs[5].bits_excluded = true;
    cs[6].screen = .{ stats(10_000, 50, 0, 0), stats(10_000, 50, 0, 0) }; // +0.5%
    var out: [8]usize = undefined;
    var n = finalists(&cs, &out);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqualSlices(usize, &.{ 2, 0, 3 }, out[0..n]);
    // the best one becomes not measurable: the fourth replaces it
    cs[2].gate_fails = 5;
    n = finalists(&cs, &out);
    try testing.expectEqualSlices(usize, &.{ 0, 3, 1 }, out[0..n]);
    // fewer eligible than the limit: all of them; an unscreened candidate is not eligible; ties go by index
    cs[0].screen = .{ null, null };
    cs[1].screen = .{ cs[3].screen[0], cs[3].screen[1] };
    n = finalists(&cs, &out);
    try testing.expectEqualSlices(usize, &.{ 1, 3 }, out[0..n]);
    // the output slice limits it too
    var two: [2]usize = undefined;
    try testing.expectEqual(@as(usize, 2), finalists(&[_]CandState{ cand(-1), cand(-2), cand(-3) }, &two));
}

test "tune stats: the served reasons for no finalist" {
    var all_bits = [_]CandState{ cand(-1), cand(-2) };
    all_bits[0].bits_excluded = true;
    all_bits[1].identical = false;
    try testing.expectEqual(ServedReason.no_identical_candidate, noFinalistReason(&all_bits, true));
    var quiet = [_]CandState{ cand(-1), cand(-2), cand(-3) };
    quiet[0].bits_excluded = true;
    quiet[1].gate_fails = 5;
    quiet[2].gate_fails = 6;
    try testing.expectEqual(ServedReason.no_measurable_candidate, noFinalistReason(&quiet, true));
    var flat = [_]CandState{ cand(0), cand(1) };
    flat[0].screen = .{ stats(10_000, 0, 0, 0), stats(10_000, 0, 0, 0) };
    flat[1].screen = .{ stats(10_000, 100, 0, 0), stats(10_000, 100, 0, 0) };
    try testing.expectEqual(ServedReason.no_gain_in_screen, noFinalistReason(&flat, true));
    // one measurable candidate with no gain outweighs the not measurable ones
    var mixed = [_]CandState{ cand(0), cand(-2) };
    mixed[0].screen = flat[0].screen;
    mixed[1].gate_fails = 5;
    try testing.expectEqual(ServedReason.no_gain_in_screen, noFinalistReason(&mixed, true));
    // the pick: the lowest worse-regime share, ties by position
    const f = [_][2]BatchStats{
        .{ stats(10_000, -500, 0, 0), stats(10_000, -100, 0, 0) }, // worse regime -1%
        .{ stats(10_000, -300, 0, 0), stats(10_000, -300, 0, 0) }, // -3%
        .{ stats(10_000, -300, 0, 0), stats(10_000, -300, 0, 0) }, // -3%, a tie
    };
    try testing.expectEqual(@as(usize, 1), pickOfAttempt(&f));
}

fn conf(ref: i64, diff: i64, hi: i64) BatchStats {
    return .{ .ref_ns = ref, .cand_ns = ref + diff, .diff_ns = diff, .q1_ns = diff, .q3_ns = diff, .lo_ns = diff - 20, .hi_ns = hi };
}

test "tune stats: confirmation rules" {
    // -1% with a tight interval: below the 2% floor
    try testing.expectEqual(Confirm.below_floor, judgeConfirm(.{ conf(10_000, -100, -50), conf(10_000, -100, -50) }));
    // -5% in both regimes with hi < 0
    try testing.expectEqual(Confirm.ok, judgeConfirm(.{ conf(10_000, -500, -400), conf(10_000, -500, -400) }));
    // the floor sits at 2.0%: 2.0% passes, 1.99% does not
    try testing.expectEqual(Confirm.ok, judgeConfirm(.{ conf(10_000, -200, -100), conf(10_000, -200, -100) }));
    try testing.expectEqual(Confirm.below_floor, judgeConfirm(.{ conf(10_000, -199, -100), conf(10_000, -200, -100) }));
    // an interval crossing zero in one regime: not confirmed, even when the other is clear
    try testing.expectEqual(Confirm.not_confirmed, judgeConfirm(.{ conf(10_000, -500, -400), conf(10_000, -500, 10) }));
    // the two-window case: confirm1 at -5% is ok, confirm2 at -0.5% (tight) is below the floor, so the kernel is "served"
    try testing.expectEqual(Confirm.ok, judgeConfirm(.{ conf(10_000, -500, -300), conf(10_000, -500, -300) }));
    try testing.expectEqual(Confirm.below_floor, judgeConfirm(.{ conf(10_000, -50, -20), conf(10_000, -50, -20) }));
}

test "tune stats: the five-failure rule" {
    var c = cand(-3);
    c.gate_fails = 4;
    try testing.expect(!c.notMeasurable());
    try testing.expect(c.eligible());
    c.gate_fails = 5;
    try testing.expect(c.notMeasurable());
    try testing.expect(!c.eligible());
}
