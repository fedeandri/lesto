//! `lesto --version` report - the versions of the app and the MLX engine,
//! WITHOUT booting the HTTP server. The macOS app spawns
//! `lesto --version` as a one-shot subprocess and parses this so Settings
//! can show engine versions without a running server (Swift side:
//! `EngineVersions.parse`). Keep this a pure formatter: main.zig gathers the
//! runtime values (`mlx_version()`) and the build-time pins (`build_options`)
//! and calls `writeReport`.

const std = @import("std");

/// Every version string surfaced by `--version`. The runtime-queryable one
/// (`mlx`) comes from the linked library; the rest are build-time pins
/// (Homebrew mlx-c) or derived values.
pub const Info = struct {
    /// Lesto version (`build_options.version`).
    app: []const u8,
    /// MLX core, from `mlx_version()` at runtime.
    mlx: []const u8,
    /// mlx-c C bindings, the pinned submodule revision (no runtime API).
    mlx_c: []const u8,
    /// M5 NAX (neural accelerator) status: "on (...)" / "off (<reason>)",
    /// from `transformer.naxStatus()` (GPU gen + macOS floor; the bundled
    /// MLX always ships the NAX kernels - asserted at build time).
    nax: []const u8,
};

/// Render one `name value` line per component in a stable order. Machine-
/// parseable: the first whitespace-delimited token is the component name, the
/// remainder is its version (which may itself contain spaces, e.g.
/// `nax on (M5 neural accelerators)`). A pin with no value collapses to
/// `unknown` so every line always has a value token.
pub fn writeReport(w: *std.Io.Writer, info: Info) !void {
    try w.print("lesto {s} (base mlx-serve 26.10.3)\n", .{val(info.app)});
    try w.print("mlx {s}\n", .{val(info.mlx)});
    try w.print("mlx-c {s}\n", .{val(info.mlx_c)});
    try w.print("nax {s}\n", .{val(info.nax)});
}

/// Allocate the report as a string (test/caller convenience).
pub fn report(allocator: std.mem.Allocator, info: Info) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try writeReport(&out.writer, info);
    return allocator.dupe(u8, out.written());
}

/// A blank pin reads as `unknown` - never an empty value token, so the Swift
/// parser always gets `name` + `version`.
fn val(s: []const u8) []const u8 {
    return if (s.len == 0) "unknown" else s;
}

test "version: report renders one name-value line per component" {
    const s = try report(std.testing.allocator, .{
        .app = "26.7.9",
        .mlx = "0.32.0",
        .mlx_c = "0.6.0",
        .nax = "on (M5 neural accelerators)",
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\lesto 26.7.9 (base mlx-serve 26.10.3)
        \\mlx 0.32.0
        \\mlx-c 0.6.0
        \\nax on (M5 neural accelerators)
        \\
    , s);
}

test "version: blank pins read as unknown" {
    const s = try report(std.testing.allocator, .{
        .app = "26.7.9",
        .mlx = "0.32.0",
        .mlx_c = "", // build.sh couldn't resolve it (dev build)
        .nax = "",
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\lesto 26.7.9 (base mlx-serve 26.10.3)
        \\mlx 0.32.0
        \\mlx-c unknown
        \\nax unknown
        \\
    , s);
}
