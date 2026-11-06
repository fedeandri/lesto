//! Command-line helpers for main.zig: classification of an argument the flag
//! loop could not consume, and the default models root of `mlx-serve serve`.
//! Pure functions, hermetically tested here.

const std = @import("std");

// ── Unparsed-argument reporting ─────────────────────────────────────────

/// Why main.zig's flag loop could not consume an argument.
///
/// The loop matches every flag by EXACT name and reads its value from the
/// NEXT argv slot, and it used to end with no else branch at all, so a
/// misspelled flag, or the `--flag=value` shape it never accepted, was
/// dropped in silence. That is the worst possible outcome for a launcher: the
/// flag parses as far as the user can tell, `--help` documents it, and the
/// server boots clean while ignoring what was asked for. Rejecting loudly is
/// the whole point; the variants exist only to make the message actionable.
pub const ArgReject = enum {
    /// `--model=/path`: value welded to the flag name.
    equals_form,
    /// A flag in the LAST argv slot, so its value never arrived.
    missing_value,
    /// Not a flag we know.
    unknown,

    /// Trailing advice for the error message. Never empty.
    pub fn hint(self: ArgReject) []const u8 {
        return switch (self) {
            .equals_form => "flags take their value as a separate argument (--model <path>, not --model=<path>)",
            .missing_value => "this flag expects a value after it, or it is misspelled",
            .unknown => "see --help for the flag list",
        };
    }
};

/// Classify an argument the flag loop fell through on. `is_last` is true when
/// it occupied the final argv slot, the only way a known value-taking flag
/// can reach the else branch (every such arm is guarded on `i + 1 < args.len`).
pub fn classifyUnparsedArg(arg: []const u8, is_last: bool) ArgReject {
    const is_flag = std.mem.startsWith(u8, arg, "-");
    if (is_flag and std.mem.indexOfScalar(u8, arg, '=') != null) return .equals_form;
    if (is_flag and is_last) return .missing_value;
    return .unknown;
}

pub fn modelsRootPath(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models", .{home});
}

/// Inputs to "which models should this boot discover?", see
/// `shouldDefaultModelsRoot`.
pub const RootDefaulting = struct {
    /// Invoked as `mlx-serve serve` rather than flags.
    subcommand: bool,
    /// The process will serve HTTP (subcommand, or the `--serve` flag).
    serve_mode: bool,
    /// `--model <path>` named a specific checkpoint.
    has_explicit_model: bool,
};

/// Should an unspecified `--model-dir` fall back to `~/.mlx-serve/models`?
///
/// That path is already the single source of truth everywhere else (the app's
/// DownloadManager and both resolvers agree on it), so a server told to serve,
/// but given neither a model nor a directory, has exactly one sensible place to
/// look. Without this a bare `mlx-serve --serve` discovered nothing and
/// answered 503 to everything.
///
/// The one case that must NOT default: `--model <path> --serve`, which asked
/// for one specific model. Registering everything else on disk beside it is a
/// different server than the one requested.
pub fn shouldDefaultModelsRoot(in: RootDefaulting) bool {
    if (!in.serve_mode) return false;
    if (in.subcommand) return true; // `serve` always populates the picker
    return !in.has_explicit_model;
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "cli: a serving boot with no model named falls back to the shared models root" {
    // `mlx-serve serve` already defaults the discovery root; the `--serve`
    // FLAG form did not, so a bare `mlx-serve --serve` booted a server that
    // had discovered nothing and could only answer 503, never what anyone
    // meant by "serve".
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = true, .serve_mode = true, .has_explicit_model = false }));
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = true, .has_explicit_model = false }));

    // `--model <path> --serve` asked for ONE model. Quietly registering the
    // other 28 on disk is a different server than the one requested.
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = true, .has_explicit_model = true }));
    // The `serve` subcommand with a model named still wants the picker
    // populated, which this must not change.
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = true, .serve_mode = true, .has_explicit_model = true }));

    // Not serving at all (one-shot `--prompt`) never scans a root.
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = false, .has_explicit_model = true }));
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = false, .has_explicit_model = false }));
}

test "cli: an unparsed argument is classified, never silently ignored" {
    // `--flag=value`: main.zig's flag loop matches names EXACTLY and takes the
    // value as the NEXT argument, so the '='-joined form is not a near-miss,
    // it is a shape we have never accepted. It used to fall out of the loop in
    // silence: `--model=<path>` booted a healthy-looking headless server that
    // then auto-picked a different model and crashed on it.
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--model=/tmp/m", false));
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--port=1234", false));
    // Even in last position the '=' hint is the useful one.
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--model=/tmp/m", true));

    // A flag in the LAST position fell out because its value is missing;
    // the loop's `i + 1 < args.len` guard is the only way to reach here.
    try testing.expectEqual(ArgReject.missing_value, classifyUnparsedArg("--model", true));
    try testing.expectEqual(ArgReject.missing_value, classifyUnparsedArg("-h", true));

    // Anything else is simply not a flag we know.
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("--frobnicate", false));
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("stray", false));
    // An '=' outside a flag is not the equals form.
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("a=b", false));
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("a=b", true));

    // Every reason carries actionable advice, and the equals-form one names
    // the shape that actually works.
    for ([_]ArgReject{ .equals_form, .missing_value, .unknown }) |r| {
        try testing.expect(r.hint().len > 0);
    }
    try testing.expect(std.mem.indexOf(u8, ArgReject.equals_form.hint(), "separate argument") != null);
}
