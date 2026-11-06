const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.16.0's bundled libc++ fails to compile against the macOS 27 SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>);
    // fixed in 0.17.0, which is why the floor moved.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "lesto requires Zig 0.17 (have {d}.{d}.{d}). Run ./scripts/fetch-zig.sh or grab it from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Pin LC_BUILD_VERSION minos to macOS 26.2 — the honest floor: the linked
    // libmlx is built at deployment target 26.2 (NAX kernels, scripts/
    // build-mlx.sh), so on older macOS the binary can't run anyway; failing at
    // the binary with a clear dyld version error beats "loading" and dying on
    // the dylib. Matches app LSMinimumSystemVersion + Package.swift. Guard:
    // tests/test_mlx_staged_nax.sh (binary minos check).
    // Host-gated: the minimum applies only when building on macOS.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .macos) .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        } else .{},
    });
    const optimize = b.standardOptimizeOption(.{});

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (target.result.os.tag == .macos) {
        verifyMlxStage(b);
    }

    if (builtin.os.tag != .macos) return;

    // Version. The default is the `VERSION` file at the repository root, so a
    // plain `zig build` reports the version of the checkout; `-Dversion`
    // overrides it. Same pattern as the engine pins below.
    const version = b.option([]const u8, "version", "Version string") orelse readVersionFile(b) orelse "0.0.0-dev";

    const mas = b.option(bool, "mas", "MAS build (no curl/model-pull subprocess)") orelse false;

    // Engine-version pin surfaced by `lesto --version` (the macOS app spawns
    // it and parses the output, see src/version.zig). MLX reports itself at runtime;
    // the mlx-c version has no runtime query API:
    //   --mlx-c-version  pinned mlx-c submodule version; defaults from the
    //                    lib/mlx/.version stamp (written by scripts/build-mlx.sh)
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxcPin(b) orelse "unknown";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption(bool, "mas", mas);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
    build_options.addOption([]const u8, "git_sha", git_sha);
    // The corpus replay and benchmark tests run tens of seconds in Debug (#639), so
    // `zig build test` skips them and `zig build test -Dslow-tests` runs them.
    build_options.addOption(bool, "slow_tests", b.option(bool, "slow-tests", "Also run the slow corpus-replay and benchmark tests") orelse false);

    const mlx_steel_sources = b.createModule(.{
        .root_source_file = b.path("lib/mlx_steel_sources.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "mlx_steel_sources", .module = mlx_steel_sources },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
        },
    });

    // Jinja2 template engine (from the common/jinja code of llama.cpp + nlohmann/json), built from lib/jinja_cpp.
    addJinja(b, mod);
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // The volume free-space probe: ARC Objective-C over Foundation.
    mod.addCSourceFile(.{ .file = b.path("src/volume_free.m"), .flags = &.{ "-O3", "-fobjc-arc", "-Wno-deprecated-declarations" } });

    // mlx + mlx-c: self-built from the pinned submodules (lib/mlx-src,
    // lib/mlxc-src) into lib/mlx by scripts/build-mlx.sh, with NAX kernels
    // enabled (the Homebrew bottle ships without them). No Homebrew library
    // path is added, so a leftover brew mlx-c can never win the link.
    addMlxLib(b, mod);

    if (macos_sdk_frameworks) |fw_path| {
        mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.linkFramework("IOSurface", .{});

    const exe = b.addExecutable(.{
        .name = "lesto",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes (app bundling)
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run lesto");
    run_step.dependOn(&run_cmd.step);

    // Unit tests — reuses the same module config (mlx-c, jinja_cpp, etc.)
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "mlx_steel_sources", .module = mlx_steel_sources },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
        },
    });

    addJinja(b, test_mod);
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addCSourceFile(.{ .file = b.path("src/volume_free.m"), .flags = &.{ "-O3", "-fobjc-arc", "-Wno-deprecated-declarations" } });
    test_mod.linkSystemLibrary("c++", .{});
    addMlxLib(b, test_mod);

    if (macos_sdk_frameworks) |fw_path| {
        test_mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    test_mod.linkFramework("IOKit", .{});
    test_mod.linkFramework("CoreFoundation", .{});
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("Metal", .{});
    test_mod.linkFramework("IOSurface", .{});

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const test_build = b.step("test-build", "Compile unit tests without running them");
    test_build.dependOn(&b.addInstallArtifact(unit_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    // A filtered run is a targeted one, often gated on an env var (QWEN4_TEST_MODEL)
    // that the run's cache key does not see: it reruns every time.
    if (test_filter != null) run_unit_tests.has_side_effects = true;
    if (qwen_preprocess_fixture) |fixture| {
        run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
    }
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // `zig build check`: type-check the engine and its tests without generating
    // code. The build system passes `-fno-emit-bin` when no step asks for the
    // binary, so this is AstGen + Sema alone — the whole tree in seconds, where
    // emitting it costs minutes (LLVM, single-threaded, ~4x more again at
    // ReleaseFast). For the edit loop; a real build still gates every change.
    {
        const check = b.step("check", "Type-check everything, emitting no binary");
        check.dependOn(&b.addExecutable(.{ .name = "lesto", .root_module = mod }).step);
        check.dependOn(&b.addTest(.{ .root_module = test_mod }).step);
    }
}

/// The Jinja2 template engine: the seven C++17 sources of lib/jinja_cpp, compiled with the flags the old prebuilt
/// archive used (-std=c++17 -O2 -DNDEBUG). The module must link libc++ itself (the main and the test module do).
fn addJinja(b: *std.Build, m: *std.Build.Module) void {
    m.addCSourceFiles(.{
        .root = b.path("lib/jinja_cpp"),
        .files = &.{ "jinja_wrapper.cpp", "caps.cpp", "lexer.cpp", "parser.cpp", "runtime.cpp", "jinja_string.cpp", "value.cpp" },
        .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" },
    });
}

/// Translates a single C header into an importable module (`@import("name")`
/// at the call site) via `addTranslateC`, replacing an inline `@cImport` —
/// removed as a language builtin in 0.17.0-dev.
fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    return translate.createModule();
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// Link the self-built mlx + mlx-c staged in lib/mlx by scripts/build-mlx.sh
/// (vendored sources lib/mlx-src + lib/mlxc-src, deployment target 26.2 so
/// MLX's NAX kernels are compiled in - the Homebrew bottle ships without them
/// and hard-wires is_nax_available() false even on M5). Install names are
/// @rpath/...; the build-tree rpath resolves them in the dev tree.
/// Guard test: tests/test_mlx_staged_nax.sh.
fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // use_pkg_config = .no: a leftover Homebrew mlx-c must never hijack this
    // link — we want exactly the staged NAX-enabled pair.
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // @loader_path is relative to the binary itself,
    // so this stays correct regardless of the launching process's cwd and
    // stays relocatable across machines. Two entries for the same reason —
    // the installed exe and the `zig build test` binary sit at different
    // depths under the build root.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Fail loudly with the fix, never let the linker produce a
/// confusing -lmlxc error (or silently pick up a leftover brew copy from
/// /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[lesto] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// The version from the `VERSION` file at the build root, trimmed. Read at
/// configure time so a plain `zig build` reports the version of the checkout.
/// Returns null when the file is missing or empty.
fn readVersionFile(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "VERSION",
        b.allocator,
        .limited(64 * 1024),
    ) catch return null;
    const v = std.mem.trim(u8, bytes, " \t\r\n");
    return if (v.len > 0) b.dupe(v) else null;
}

/// The pinned mlx-c revision from lib/mlx/.version (written by
/// scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>"), surfaced in
/// `lesto --version`. Returns null (→ "unknown") when not staged yet.
fn readMlxcPin(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "mlxc=")) return b.dupe(tok["mlxc=".len..]);
    }
    return null;
}
