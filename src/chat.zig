const std = @import("std");
const jinja_c = @import("jinja_c");
const tokenizer_mod = @import("tokenizer.zig");
const log = @import("log.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8, // JSON string
};

/// One image on a message. A marker: the chat template renders one placeholder per item,
/// and the pixel data is encoded elsewhere.
pub const ImageData = struct {};

/// One audio clip on a message. A marker, like `ImageData`.
pub const AudioData = struct {};

/// One video on a message. A marker, like `ImageData`.
pub const VideoData = struct {};

/// OpenAI spells the system turn `developer` for reasoning models (pi does
/// when `supportsDeveloperRole` is unset). No template we serve tells the two
/// apart and some templates raise on the unknown role, which is a SILENT fallback.
pub fn canonicalRole(role: []const u8) []const u8 {
    return if (std.mem.eql(u8, role, "developer")) "system" else role;
}

/// Join system messages into one leading turn. With `leading_only`, later
/// turns keep their position. Caller owns the joined buffer, if any.
pub fn foldSystemMessages(allocator: std.mem.Allocator, messages: *std.ArrayList(Message), leading_only: bool) !?[]const u8 {
    var end = messages.items.len;
    if (leading_only) {
        end = 0;
        while (end < messages.items.len and std.mem.eql(u8, messages.items[end].role, "system")) : (end += 1) {}
    }
    var extra: usize = 0;
    for (messages.items[@min(end, 1)..end]) |m| {
        if (std.mem.eql(u8, m.role, "system")) extra += 1;
    }
    if (extra == 0) return null;
    const lead_is_system = messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system");
    var joined = std.ArrayList(u8).empty;
    errdefer joined.deinit(allocator);
    if (lead_is_system) try joined.appendSlice(allocator, messages.items[0].content);
    var i: usize = if (lead_is_system) 1 else 0;
    while (i < end) {
        if (!std.mem.eql(u8, messages.items[i].role, "system")) {
            i += 1;
            continue;
        }
        if (joined.items.len > 0) try joined.appendSlice(allocator, "\n\n");
        try joined.appendSlice(allocator, messages.items[i].content);
        _ = messages.orderedRemove(i);
        end -= 1;
    }
    const text = try joined.toOwnedSlice(allocator);
    if (lead_is_system) {
        messages.items[0].content = text;
    } else {
        try messages.insert(allocator, 0, .{ .role = "system", .content = text });
    }
    return text;
}

// ─── media content parts (this engine serves text only) ──────────────────
//
// ONE definition of "this content part carries an image, video or audio",
// shared by every request surface (chat completions, Anthropic messages,
// Responses, embeddings, Ollama). A parse site records the part as a marker
// and the handler refuses the request with the text-only 400; no site keeps
// its own list of type names.

pub const MediaKind = enum { image, video, audio };

const MediaTypeName = struct { name: []const u8, kind: MediaKind };

/// Content-part `type` names that always carry media, whatever the surface calls them.
const media_type_names = [_]MediaTypeName{
    .{ .name = "image", .kind = .image },
    .{ .name = "image_url", .kind = .image },
    .{ .name = "input_image", .kind = .image },
    .{ .name = "video", .kind = .video },
    .{ .name = "video_url", .kind = .video },
    .{ .name = "input_video", .kind = .video },
    .{ .name = "audio", .kind = .audio },
    .{ .name = "audio_url", .kind = .audio },
    .{ .name = "input_audio", .kind = .audio },
};

const media_extensions = [_]MediaTypeName{
    .{ .name = "png", .kind = .image },
    .{ .name = "jpg", .kind = .image },
    .{ .name = "jpeg", .kind = .image },
    .{ .name = "gif", .kind = .image },
    .{ .name = "webp", .kind = .image },
    .{ .name = "bmp", .kind = .image },
    .{ .name = "tif", .kind = .image },
    .{ .name = "tiff", .kind = .image },
    .{ .name = "heic", .kind = .image },
    .{ .name = "heif", .kind = .image },
    .{ .name = "avif", .kind = .image },
    .{ .name = "ico", .kind = .image },
    .{ .name = "mp3", .kind = .audio },
    .{ .name = "wav", .kind = .audio },
    .{ .name = "flac", .kind = .audio },
    .{ .name = "ogg", .kind = .audio },
    .{ .name = "oga", .kind = .audio },
    .{ .name = "opus", .kind = .audio },
    .{ .name = "m4a", .kind = .audio },
    .{ .name = "aac", .kind = .audio },
    .{ .name = "wma", .kind = .audio },
    .{ .name = "aif", .kind = .audio },
    .{ .name = "aiff", .kind = .audio },
    .{ .name = "mp4", .kind = .video },
    .{ .name = "m4v", .kind = .video },
    .{ .name = "mov", .kind = .video },
    .{ .name = "mkv", .kind = .video },
    .{ .name = "webm", .kind = .video },
    .{ .name = "avi", .kind = .video },
    .{ .name = "mpg", .kind = .video },
    .{ .name = "mpeg", .kind = .video },
    .{ .name = "3gp", .kind = .video },
};

/// Kind of a MIME type string (`image/png`, `audio/wav; codecs=x`); null for anything else.
fn mediaKindOfMime(s: []const u8) ?MediaKind {
    if (std.ascii.startsWithIgnoreCase(s, "image/")) return .image;
    if (std.ascii.startsWithIgnoreCase(s, "video/")) return .video;
    if (std.ascii.startsWithIgnoreCase(s, "audio/")) return .audio;
    return null;
}

/// Kind of a `data:<mime>[;base64],<payload>` URL; null when the string is not a data URL or not media.
fn mediaKindOfDataUrl(s: []const u8) ?MediaKind {
    if (!std.ascii.startsWithIgnoreCase(s, "data:")) return null;
    return mediaKindOfMime(s["data:".len..]);
}

/// Kind implied by a file name or URL path extension (query and fragment ignored); null for any other name.
fn mediaKindOfName(s: []const u8) ?MediaKind {
    var end = s.len;
    if (std.mem.indexOfAny(u8, s, "?#")) |q| end = q;
    const path = s[0..end];
    const dot = std.mem.lastIndexOfScalar(u8, path, '.') orelse return null;
    const ext = path[dot + 1 ..];
    for (media_extensions) |e| {
        if (std.ascii.eqlIgnoreCase(ext, e.name)) return e.kind;
    }
    return null;
}

/// Media kind carried by one JSON object that describes a file: a data URL, a
/// media MIME field, or a media file name / URL extension.
fn mediaKindOfFileFields(obj: std.json.ObjectMap) ?MediaKind {
    const data_keys = [_][]const u8{ "file_data", "data", "file_url", "url" };
    for (data_keys) |k| {
        const v = obj.get(k) orelse continue;
        if (v == .string) if (mediaKindOfDataUrl(v.string)) |kind| return kind;
    }
    const mime_keys = [_][]const u8{ "mime", "mime_type", "mimeType", "media_type", "content_type" };
    for (mime_keys) |k| {
        const v = obj.get(k) orelse continue;
        if (v == .string) if (mediaKindOfMime(v.string)) |kind| return kind;
    }
    const name_keys = [_][]const u8{ "filename", "file_name", "name", "file_url", "url" };
    for (name_keys) |k| {
        const v = obj.get(k) orelse continue;
        if (v == .string) if (mediaKindOfName(v.string)) |kind| return kind;
    }
    return null;
}

/// Media kind of a file-style part (`input_file`, `file`, `document`): its own
/// fields, then the objects it nests the file in (`file`, `input_file`, `source`).
fn mediaKindOfFilePart(part: std.json.Value) ?MediaKind {
    if (part != .object) return null;
    if (mediaKindOfFileFields(part.object)) |kind| return kind;
    const nested_keys = [_][]const u8{ "file", "input_file", "source" };
    for (nested_keys) |k| {
        const v = part.object.get(k) orelse continue;
        if (v == .object) if (mediaKindOfFileFields(v.object)) |kind| return kind;
    }
    return null;
}

/// Whether a part of the given `type` carries image, video or audio, and which.
/// File-style parts count only when their data is media (a data URL, a MIME
/// type or a file name); a text or PDF file part is not media.
pub fn mediaKindOfPart(part_type: []const u8, part: std.json.Value) ?MediaKind {
    for (media_type_names) |m| {
        if (std.mem.eql(u8, part_type, m.name)) return m.kind;
    }
    if (std.mem.eql(u8, part_type, "input_file") or std.mem.eql(u8, part_type, "file") or std.mem.eql(u8, part_type, "document")) {
        return mediaKindOfFilePart(part);
    }
    return null;
}

/// `mediaKindOfPart` for a part value: reads the part's own `type`; null for a
/// non-object part or one without a string `type`.
pub fn mediaKindOfPartValue(part: std.json.Value) ?MediaKind {
    if (part != .object) return null;
    const t = part.object.get("type") orelse return null;
    if (t != .string) return null;
    return mediaKindOfPart(t.string, part);
}

/// How many `content` levels the media check descends (a `tool_result` block's `content`,
/// a part nesting its own `content`). Request parsers read at most one nested level, so
/// media below this depth never reaches the model.
pub const media_nesting_limit: usize = 8;

/// Whether one part carries media: the part itself, or any part nested in its `content`
/// (string, array or single object), down to `media_nesting_limit` levels.
fn partHasMediaAt(part: std.json.Value, depth: usize) bool {
    if (mediaKindOfPartValue(part) != null) return true;
    if (part != .object) return false;
    const inner = part.object.get("content") orelse return false;
    return contentHasMediaAt(inner, depth + 1);
}

fn contentHasMediaAt(content: std.json.Value, depth: usize) bool {
    if (depth > media_nesting_limit) return false;
    switch (content) {
        .array => |arr| {
            for (arr.items) |part| {
                if (partHasMediaAt(part, depth)) return true;
            }
            return false;
        },
        .object => return partHasMediaAt(content, depth),
        else => return false,
    }
}

/// Whether any part of a content array carries an image, video or audio, a part nested
/// in a part's `content` (an Anthropic `tool_result` block, for one) included.
pub fn contentHasMediaPart(parts: []const std.json.Value) bool {
    for (parts) |part| {
        if (partHasMediaAt(part, 0)) return true;
    }
    return false;
}

/// Shape-independent media check for a `content` value: a string carries none, an array
/// carries media when any part does, and a single object is itself one part; each part's
/// own `content` is searched the same way (a `tool_result` block's string, array or single
/// object). Every parse site that inspects `content` for media calls this, never its own
/// array or object case.
pub fn contentValueHasMedia(content: std.json.Value) bool {
    return contentHasMediaAt(content, 0);
}

test "contentValueHasMedia: string, array and single-object content" {
    const t = std.testing;
    const cases = [_]struct { json: []const u8, media: bool }{
        .{ .json = "\"just text\"", .media = false },
        .{ .json = "null", .media = false },
        .{ .json = "7", .media = false },
        .{ .json = "[{\"type\":\"text\",\"text\":\"a\"}]", .media = false },
        .{ .json = "[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"input_video\"}]", .media = true },
        .{ .json = "{\"type\":\"text\",\"text\":\"a\"}", .media = false },
        .{ .json = "{\"type\":\"input_video\"}", .media = true },
        .{ .json = "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AAAA\"}", .media = true },
        .{ .json = "{\"type\":\"file\",\"file\":{\"filename\":\"a.pdf\"}}", .media = false },
        .{ .json = "{\"type\":\"input_file\",\"filename\":\"a.mp4\"}", .media = true },
    };
    for (cases) |c| {
        var p = try std.json.parseFromSlice(std.json.Value, t.allocator, c.json, .{});
        defer p.deinit();
        try t.expectEqual(c.media, contentValueHasMedia(p.value));
    }
}

test "contentValueHasMedia: media nested in a tool_result's content is media in every shape" {
    const t = std.testing;
    const media = [_][]const u8{
        "{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":{\"type\":\"input_video\"}}",
        "{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"AAAA\"}}]}",
        "[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":{\"type\":\"input_audio\"}}]",
        "[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":[{\"type\":\"tool_result\",\"content\":[{\"type\":\"video_url\"}]}]}]",
        "{\"type\":\"tool_result\",\"content\":{\"type\":\"document\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"AAAA\"}}}",
    };
    for (media) |c| {
        var p = try std.json.parseFromSlice(std.json.Value, t.allocator, c, .{});
        defer p.deinit();
        try t.expect(contentValueHasMedia(p.value));
        if (p.value == .array) try t.expect(contentHasMediaPart(p.value.array.items));
    }
}

test "contentValueHasMedia: a tool_result whose content is text, PDF or empty is not media" {
    const t = std.testing;
    const plain = [_][]const u8{
        "{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"sunny\"}",
        "{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":{\"type\":\"text\",\"text\":\"sunny\"}}",
        "{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"document\",\"source\":{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"AAAA\"}}]}",
        "{\"type\":\"tool_result\",\"tool_use_id\":\"t\"}",
        "[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":[]},{\"type\":\"text\",\"text\":\"b\"}]",
    };
    for (plain) |c| {
        var p = try std.json.parseFromSlice(std.json.Value, t.allocator, c, .{});
        defer p.deinit();
        try t.expect(!contentValueHasMedia(p.value));
    }
}

test "contentValueHasMedia: nesting deeper than the limit is not searched" {
    const t = std.testing;
    const alloc = t.allocator;
    // `levels` tool_result wrappers around one video part.
    const build = struct {
        fn f(a: std.mem.Allocator, levels: usize) ![]u8 {
            var out: std.Io.Writer.Allocating = .init(a);
            defer out.deinit();
            for (0..levels) |_| try out.writer.writeAll("{\"type\":\"tool_result\",\"content\":");
            try out.writer.writeAll("{\"type\":\"input_video\"}");
            for (0..levels) |_| try out.writer.writeAll("}");
            return a.dupe(u8, out.written());
        }
    }.f;
    const within = try build(alloc, media_nesting_limit);
    defer alloc.free(within);
    var pw = try std.json.parseFromSlice(std.json.Value, alloc, within, .{});
    defer pw.deinit();
    try t.expect(contentValueHasMedia(pw.value));
    const beyond = try build(alloc, media_nesting_limit + 1);
    defer alloc.free(beyond);
    var pb = try std.json.parseFromSlice(std.json.Value, alloc, beyond, .{});
    defer pb.deinit();
    try t.expect(!contentValueHasMedia(pb.value));
}

test "media predicate: every media type name is recognised with its kind" {
    const t = std.testing;
    const cases = [_]struct { json: []const u8, kind: MediaKind }{
        .{ .json = "{\"type\":\"image\"}", .kind = .image },
        .{ .json = "{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}", .kind = .image },
        .{ .json = "{\"type\":\"input_image\"}", .kind = .image },
        .{ .json = "{\"type\":\"video\"}", .kind = .video },
        .{ .json = "{\"type\":\"video_url\"}", .kind = .video },
        .{ .json = "{\"type\":\"input_video\"}", .kind = .video },
        .{ .json = "{\"type\":\"audio\"}", .kind = .audio },
        .{ .json = "{\"type\":\"audio_url\"}", .kind = .audio },
        .{ .json = "{\"type\":\"input_audio\"}", .kind = .audio },
    };
    for (cases) |c| {
        var p = try std.json.parseFromSlice(std.json.Value, t.allocator, c.json, .{});
        defer p.deinit();
        try t.expectEqual(c.kind, mediaKindOfPartValue(p.value).?);
    }
}

test "media predicate: text parts and non-media files are not media" {
    const t = std.testing;
    const cases = [_][]const u8{
        "{\"type\":\"text\",\"text\":\"hi\"}",
        "{\"type\":\"input_text\",\"text\":\"hi\"}",
        "{\"type\":\"output_text\",\"text\":\"hi\"}",
        "{\"type\":\"tool_use\",\"id\":\"a\"}",
        "{\"type\":\"input_file\",\"filename\":\"notes.txt\",\"file_data\":\"data:text/plain;base64,aGk=\"}",
        "{\"type\":\"input_file\",\"filename\":\"paper.pdf\",\"file_data\":\"data:application/pdf;base64,AAAA\"}",
        "{\"type\":\"input_file\",\"file_id\":\"file-123\"}",
        "{\"type\":\"file\",\"file\":{\"filename\":\"main.ts\",\"file_data\":\"data:text/plain;base64,aGk=\"}}",
        "{\"type\":\"document\",\"source\":{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"AAAA\"}}",
        "{\"type\":\"file\"}",
        "{\"no_type\":1}",
        "{\"type\":7}",
    };
    for (cases) |c| {
        var p = try std.json.parseFromSlice(std.json.Value, t.allocator, c, .{});
        defer p.deinit();
        try t.expect(mediaKindOfPartValue(p.value) == null);
    }
    try t.expect(mediaKindOfPartValue(.null) == null);
    try t.expect(!contentHasMediaPart(&.{}));
}

test "media predicate: file-style parts are media when their data, MIME type or name says so" {
    const t = std.testing;
    const cases = [_]struct { json: []const u8, kind: MediaKind }{
        .{ .json = "{\"type\":\"input_file\",\"file_data\":\"data:image/png;base64,AAAA\"}", .kind = .image },
        .{ .json = "{\"type\":\"input_file\",\"file_data\":\"DATA:Audio/WAV;base64,AAAA\"}", .kind = .audio },
        .{ .json = "{\"type\":\"input_file\",\"file_data\":\"data:video/mp4;base64,AAAA\"}", .kind = .video },
        .{ .json = "{\"type\":\"input_file\",\"mime_type\":\"image/jpeg\"}", .kind = .image },
        .{ .json = "{\"type\":\"input_file\",\"filename\":\"clip.MOV\"}", .kind = .video },
        .{ .json = "{\"type\":\"input_file\",\"file_url\":\"https://example.com/a/voice.mp3?sig=1\"}", .kind = .audio },
        .{ .json = "{\"type\":\"file\",\"file\":{\"file_data\":\"data:image/webp;base64,AAAA\"}}", .kind = .image },
        .{ .json = "{\"type\":\"file\",\"file\":{\"filename\":\"scan.png\"}}", .kind = .image },
        .{ .json = "{\"type\":\"document\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"AAAA\"}}", .kind = .image },
    };
    for (cases) |c| {
        var p = try std.json.parseFromSlice(std.json.Value, t.allocator, c.json, .{});
        defer p.deinit();
        try t.expectEqual(c.kind, mediaKindOfPartValue(p.value).?);
    }
    // contentHasMediaPart finds the media part among text parts, wherever it sits.
    var arr = try std.json.parseFromSlice(std.json.Value, t.allocator, "[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"audio_url\"}]", .{});
    defer arr.deinit();
    try t.expect(contentHasMediaPart(arr.value.array.items));
    var txt = try std.json.parseFromSlice(std.json.Value, t.allocator, "[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\"b\"}]", .{});
    defer txt.deinit();
    try t.expect(!contentHasMediaPart(txt.value.array.items));
}

pub const Message = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
    images: ?[]const ImageData = null, // Preprocessed image data for vision
    videos: ?[]const VideoData = null, // Preprocessed video data for vision
    audio: ?[]const AudioData = null, // Raw PCM for the unified audio embedder
    /// Reasoning the client round-trips on assistant HISTORY messages
    /// (`reasoning_content`/`reasoning` on chat completions, `thinking`
    /// blocks on /v1/messages). Templates that persist reasoning across
    /// turns read it; templates that strip history reasoning
    /// (Gemma) never reference the field and render unchanged.
    reasoning_content: ?[]const u8 = null,
};

/// Chat template configuration loaded from tokenizer_config.json.
pub const ChatConfig = struct {
    chat_template: []const u8,
    bos_token: ?[]const u8,
    eos_token: ?[]const u8,
    add_bos_token: bool,
    allocator: std.mem.Allocator,
    /// Template variables as a JSON object: the model's `chat_template_kwargs`
    /// (`model-settings.json`), with a request's own merged over them per request.
    chat_template_kwargs: ?[]const u8 = null,
    /// `templateRendersToolTurn` for this template, probed ONCE at load (a probe render
    /// per request is a second parse of a multi-KB template); null = unknown, probe.
    renders_tool_turn: ?bool = null,
    /// A model-specific video atom emitted only for an attached media part.
    video_placeholder: ?[]const u8 = null,

    pub fn deinit(self: *ChatConfig) void {
        self.allocator.free(self.chat_template);
        if (self.chat_template_kwargs) |k| self.allocator.free(k);
        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
    }
};

/// Is this `chat_template` entry just an `{% include %}` POINTER at the
/// sidecar file? transformers >= 5 saves the real template to
/// `chat_template.jinja` and writes that one-liner into tokenizer_config.json
/// (mlx-community's oQ4e-fast / -oQ5e packs, issue #169). jinja.cpp has
/// no `include` statement, so rendering the pointer fails and drops SILENTLY to
/// the generic format — the model then answers in another family's markers.
/// The file it names is always the sidecar the caller already falls back to.
fn isIncludeStub(template: []const u8) bool {
    const t = std.mem.trim(u8, template, " \t\r\n");
    if (!std.mem.startsWith(u8, t, "{%") or !std.mem.endsWith(u8, t, "%}")) return false;
    const body = std.mem.trim(u8, t[2 .. t.len - 2], " \t\r\n-");
    if (!std.mem.startsWith(u8, body, "include")) return false;
    // Exactly one statement: a second `{%` would make it a real template.
    return std.mem.indexOf(u8, t[2..], "{%") == null;
}

/// The `chat_template` entry of tokenizer_config.json, as a string.
///
/// HF allows TWO shapes and we only handled one: a bare string, or a LIST of
/// `{name, template}` objects (an instruct pack ships `default` +
/// `tool_use`). Reading the list as `v.string` is not a graceful failure — it
/// is a Zig union panic that kills the process at LOAD, so the checkpoint
/// could not be served at all (live 2026-08-07: `access of union field
/// 'string' while field 'array' is active`).
///
/// Selection follows transformers: the entry named "default", which is what
/// `apply_chat_template` uses unless the caller names another. Falls back to
/// the first entry when nothing is named "default", and returns null for any
/// other shape so the caller drops to its chat_template.jinja / family
/// fallback instead of panicking.
fn chatTemplateFromValue(v: ?std.json.Value) ?[]const u8 {
    const val = v orelse return null;
    switch (val) {
        .string => |s| return if (isIncludeStub(s)) null else s,
        .array => |arr| {
            var first: ?[]const u8 = null;
            for (arr.items) |entry| {
                if (entry != .object) continue;
                const tpl = entry.object.get("template") orelse continue;
                if (tpl != .string) continue;
                if (first == null) first = tpl.string;
                const name = entry.object.get("name") orelse continue;
                if (name == .string and std.mem.eql(u8, name.string, "default")) return tpl.string;
            }
            return first;
        },
        else => return null,
    }
}

test "chat_template accepts HF's list-of-named-templates shape" {
    // An instruct pack ships a LIST (`default` + `tool_use`). Reading
    // it as `.string` panicked the process at load, so the checkpoint could
    // not be served at all.
    const a = std.testing.allocator;

    {   // the shape that panicked: pick the entry named "default"
        const json =
            \\{"chat_template":[{"name":"tool_use","template":"TOOLS"},
            \\{"name":"default","template":"PLAIN"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("PLAIN", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // no "default" named: fall back to the first usable entry
        const json =
            \\{"chat_template":[{"name":"tool_use","template":"TOOLS"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("TOOLS", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // the ordinary string shape is untouched
        const json =
            \\{"chat_template":"BARE"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("BARE", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // junk shapes return null so the caller uses its jinja/family fallback
        const json =
            \\{"chat_template":[{"name":"x"},{"nope":1}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    try std.testing.expect(chatTemplateFromValue(null) == null);

    {   // transformers >= 5 saves the template to chat_template.jinja and leaves
        // an `{% include %}` POINTER in tokenizer_config.json (mlx-community
        // oQ4e-fast, issue #169). jinja.cpp has no `include`, so
        // taking the pointer literally fails the render and SILENTLY drops to
        // the generic fallback — wrong-family markers the model then echoes.
        // Read it as "no inline template" so the sidecar file is used.
        const json =
            \\{"chat_template":"{% include 'chat_template.jinja' %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // whitespace/dash variants are the same pointer
        const json =
            \\{"chat_template":"\n  {%- include \"chat_template.jinja\" -%}\n"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // a real template that merely CONTAINS the word include is untouched
        const json =
            \\{"chat_template":"{% if x %}include{% endif %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) != null);
    }
}

/// Load chat template configuration from tokenizer_config.json.
pub fn loadChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const content = blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    };
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    const chat_template: []const u8 = if (chatTemplateFromValue(root.get("chat_template"))) |t|
        try allocator.dupe(u8, t)
    else blk: {
        // Fall back to chat_template.jinja file (e.g. packs that ship a separate template file)
        const jinja_path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{model_dir});
        defer allocator.free(jinja_path);
        if (std.Io.Dir.openFileAbsolute(io, jinja_path, .{})) |f| {
            defer f.close(io);
            var jinja_buf: [4096]u8 = undefined;
            var jinja_reader = f.reader(io, &jinja_buf);
            break :blk try jinja_reader.interface.allocRemaining(allocator, .limited(1 * 1024 * 1024));
        } else |_| {
            break :blk try allocator.dupe(u8, "");
        }
    };

    const bos_token: ?[]const u8 = if (root.get("bos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const eos_token: ?[]const u8 = if (root.get("eos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const add_bos_token = if (root.get("add_bos_token")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;

    const renders_tool_turn: ?bool = templateRendersToolTurn(allocator, chat_template) catch null;
    return .{
        .chat_template = chat_template,
        .bos_token = bos_token,
        .eos_token = eos_token,
        .add_bos_token = add_bos_token,
        .allocator = allocator,
        .renders_tool_turn = renders_tool_turn,
    };
}

/// Format chat messages into token IDs using the model's Jinja chat template.
pub fn formatChat(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    /// Extend the trailing assistant message instead of answering after it.
    continue_final: bool,
) ![]u32 {
    const rendered = try renderChatTemplate(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, continue_final);
    defer allocator.free(rendered);

    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    try encodeWithSpecialTokens(allocator, tok, rendered, &ids);
    log.debug("  prompt: {d} chars -> {d} tokens\n", .{ rendered.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

/// Render the Jinja chat template with the given messages.
pub fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    /// Extend the trailing assistant message instead of answering after it —
    /// the prompt ends mid-turn and generation resumes from its last word.
    continue_final: bool,
) ![]const u8 {
    // Split the partial reply off FIRST, so every transform below (tool
    // fallback synthesis) sees exactly the message list
    // an ordinary turn would have. The template must open the assistant turn
    // once, which it only does when the trailing assistant is not in `messages`.
    const prefill = continuationPrefill(messages, continue_final);
    const msgs = if (prefill != null) messages[0 .. messages.len - 1] else messages;

    if (chat_config.chat_template.len == 0) {
        const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
        if (prefill) |partial| {
            defer allocator.free(base);
            return std.mem.concat(allocator, u8, &.{ base, partial });
        }
        return base;
    }

    // Some chat templates don't reference `tools` or `role == 'tool'`
    // at all — Jinja silently drops the tool definitions, and tool-result messages render
    // as empty. Detect that gap and synthesize an equivalent system-prompt + user-message
    // form so the model still sees the tool context.
    const tpl = chat_config.chat_template;
    const tpl_has_tools = std.mem.indexOf(u8, tpl, "tools") != null;
    const has_tool_content = messagesHaveToolContent(msgs);
    const tpl_has_tool_role = templateReferencesToolRole(tpl) or
        (has_tool_content and (chat_config.renders_tool_turn orelse try templateRendersToolTurn(allocator, tpl)));
    const needs_inject_tools = tools_json != null and !tpl_has_tools;
    const needs_rewrite_tool_role = !tpl_has_tool_role and has_tool_content;

    var fallback_arena: ?std.heap.ArenaAllocator = null;
    defer if (fallback_arena) |*a| a.deinit();
    var effective_messages = msgs;
    var effective_tools_json = tools_json;
    // Optional keys a template may dump blindly (see fillOptionalToolDefKeys).
    // Arena-owned, so the rewritten bytes outlive the render with no free path
    // of their own.
    if (tools_json) |tj| {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        if (try fillOptionalToolDefKeys(fallback_arena.?.allocator(), tj)) |filled| {
            effective_tools_json = filled;
        }
    }
    if (needs_inject_tools or needs_rewrite_tool_role or tool_choice_instruction != null) {
        // NEVER re-init: the tool-def fill above may already own this arena,
        // and clobbering it orphans everything allocated there (a leak here, a
        // use-after-free the moment `effective_tools_json` points into it).
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            msgs,
            tools_json,
            tool_choice_instruction,
            needs_inject_tools,
            !tpl_has_tool_role,
            !tpl_has_tools,
        );
        if (needs_inject_tools) {
            effective_tools_json = null; // already inlined as system content
        }
    }

    // Serialize messages to JSON — Gemma 4 templates handle role:"tool" natively
    // (producing <|turn>tool in the rendered output). No transformation needed.
    // A template that tests only that the content KEY exists and then indexes
    // into it breaks on a null there. Some templates iterate any non-string content, so
    // a null there raises on every tool-call turn.
    const empty_content: EmptyContent = if (std.mem.indexOf(u8, tpl, "if content is not string") != null)
        .empty_string
    else
        .null_literal;
    const extra_json = try serializeExtraContext(allocator, chat_config, enable_thinking);
    defer allocator.free(extra_json);
    if (hasExtraSystemMessages(effective_messages)) {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const a = fallback_arena.?.allocator();
        var prepared = std.ArrayList(Message).empty;
        try prepared.appendSlice(a, effective_messages);
        const preserve = try templatePreservesMidSystem(a, effective_messages, chat_config, effective_tools_json, extra_json, empty_content);
        _ = try foldSystemMessages(a, &prepared, preserve);
        effective_messages = prepared.items;
    }

    if (try renderJinja(allocator, effective_messages, chat_config, effective_tools_json, extra_json, empty_content)) |rendered| {
        defer allocator.free(rendered);
        const collapsed = try collapseDoubledThinkTags(allocator, rendered);
        // A continuation commits the content channel and then hands the model
        // its own unfinished sentence. It runs INSTEAD of the thinking-off
        // tail, never beside it — both append the same channel commit, and
        // twice is a malformed turn header.
        if (prefill) |partial| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{
                collapsed, contentChannelTail(collapsed), partial,
            });
        }
        if (noThinkTailSuffix(collapsed, enable_thinking)) |suffix| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{ collapsed, suffix });
        }
        return collapsed;
    }

    // WARN, not debug: a failed render silently swaps the prompt into
    // fallbackFormatChat's generic format — for families whose stop/turn
    // tokens differ (Gemma 4's <|turn> vs the fallback's <start_of_turn>)
    // that means degenerate generation, so the downgrade must be visible
    // at the default log level.
    if (jinja_c.jinja_last_error()) |e| {
        log.warn("jinja render failed ({s}), falling back to generic chat format\n", .{std.mem.span(e)});
    }
    const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    if (prefill) |partial| {
        defer allocator.free(base);
        return std.mem.concat(allocator, u8, &.{ base, partial });
    }
    return base;
}

fn hasExtraSystemMessages(messages: []const Message) bool {
    for (messages[@min(messages.len, 1)..]) |m| {
        if (std.mem.eql(u8, m.role, "system")) return true;
    }
    return false;
}

fn renderJinja(
    allocator: std.mem.Allocator,
    messages: []const Message,
    config: *const ChatConfig,
    tools_json: ?[]const u8,
    extra_json: []const u8,
    empty_content: EmptyContent,
) !?[]const u8 {
    const json = try serializeMessagesJsonFor(allocator, messages, empty_content, config);
    defer allocator.free(json);
    const tpl = try allocator.dupeSentinel(u8, config.chat_template, 0);
    defer allocator.free(tpl);
    const msgs = try allocator.dupeSentinel(u8, json, 0);
    defer allocator.free(msgs);
    const extra = try allocator.dupeSentinel(u8, extra_json, 0);
    defer allocator.free(extra);
    const tools = if (tools_json) |t| try allocator.dupeSentinel(u8, t, 0) else null;
    defer if (tools) |t| allocator.free(t);
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(tpl.ptr, msgs.ptr, if (tools) |t| t.ptr else null, extra.ptr, 1, &len) orelse return null;
    defer jinja_c.jinja_str_free(ptr);
    return try allocator.dupe(u8, ptr[0..len]);
}

/// Probe each distinct late-system boundary with the request's tools and kwargs.
/// Raising, dropping or moving the note to the front all require consolidation.
fn templatePreservesMidSystem(
    allocator: std.mem.Allocator,
    messages: []const Message,
    config: *const ChatConfig,
    tools_json: ?[]const u8,
    extra_json: []const u8,
    empty_content: EmptyContent,
) !bool {
    const before = "mlxserve_before_note_7c91";
    const note = "mlxserve_system_note_a83d";
    const after = "mlxserve_after_note_b462";
    const calls = [_]ToolCall{.{ .id = "probe_call", .name = "probe", .arguments = "{}" }};
    const Role = enum { user, assistant, tool, tail };
    var checked: u16 = 0;
    var i: usize = 0;
    while (i < messages.len) {
        if (!std.mem.eql(u8, messages[i].role, "system")) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < messages.len and std.mem.eql(u8, messages[i].role, "system")) : (i += 1) {}
        if (start == 0) continue;
        const prev = messages[start - 1].role;
        const next = if (i < messages.len) messages[i].role else "tail";
        const prev_role = std.meta.stringToEnum(Role, prev) orelse return false;
        const next_role = std.meta.stringToEnum(Role, next) orelse return false;
        const bit = @as(u16, 1) << @as(u4, @intCast(@as(u8, @backingInt(prev_role)) * 4 + @backingInt(next_role)));
        if (checked & bit != 0) continue;
        checked |= bit;
        var probe = std.ArrayList(Message).empty;
        defer probe.deinit(allocator);
        try probe.append(allocator, .{ .role = "system", .content = "stable instructions" });
        if (!std.mem.eql(u8, prev, "user")) try probe.append(allocator, .{ .role = "user", .content = "question" });
        if (std.mem.eql(u8, prev, "tool")) try probe.append(allocator, .{ .role = "assistant", .content = "", .tool_calls = &calls });
        try probe.append(allocator, .{ .role = prev, .content = before, .tool_call_id = if (std.mem.eql(u8, prev, "tool")) "probe_call" else null });
        try probe.append(allocator, .{ .role = "system", .content = note });
        if (i < messages.len) try probe.append(allocator, .{ .role = next, .content = after });
        // A tail note must also survive when it becomes history next turn.
        for (0..@as(usize, if (i == messages.len) 2 else 1)) |pass| {
            if (pass == 1) try probe.append(allocator, .{ .role = "assistant", .content = after });
            const rendered = (try renderJinja(allocator, probe.items, config, tools_json, extra_json, empty_content)) orelse return false;
            defer allocator.free(rendered);
            if (!systemNoteInOrder(rendered, before, note, if (i < messages.len or pass == 1) after else null)) return false;
        }
    }
    return true;
}

fn systemNoteInOrder(rendered: []const u8, before: []const u8, note: []const u8, after: ?[]const u8) bool {
    const p = std.mem.indexOf(u8, rendered, before) orelse return false;
    const n = std.mem.indexOf(u8, rendered, note) orelse return false;
    if (n <= p) return false;
    if (after) |marker| {
        const a = std.mem.indexOf(u8, rendered, marker) orelse return false;
        if (a <= n) return false;
    }
    return true;
}

/// the model's chat template emits `</think></think>` between every user→assistant
/// pair in chat mode — the user role appends `</think>` after `<|Assistant|>`
/// and the assistant role prepends `</think>` before its content. The doubling
/// is structurally invalid (no legitimate rendered prompt should contain
/// `</think></think>`), and at 2-bit DQ it pushes the model into prompt-echo
/// / phrase-loop collapse on the second turn (a 24-token "Hello / Hi! / What
/// is 2+2?" chat reliably degenerates). The HF tokenizer renders the same
/// doubling so this is a template-side artifact, not our renderer; mlx-lm
/// must be tolerating it via reduction-order quirks we don't share. Strip
/// it here. Safe across all model families because `</think></think>` never
/// appears legitimately.
fn collapseDoubledThinkTags(allocator: std.mem.Allocator, rendered: []const u8) ![]u8 {
    const tag = "</think>";
    const needle = "</think></think>";
    if (std.mem.indexOf(u8, rendered, needle) == null) {
        return allocator.dupe(u8, rendered);
    }
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, rendered.len);
    var i: usize = 0;
    while (i < rendered.len) {
        // If we see `</think></think>...`, drop the FIRST `</think>` and let
        // the next iteration handle whatever follows — that way 3x, 4x, …
        // also collapse to a single tag without a separate pass.
        if (i + needle.len <= rendered.len and
            std.mem.eql(u8, rendered[i .. i + needle.len], needle))
        {
            i += tag.len; // skip the first `</think>`; next iter writes the survivor
        } else {
            try out.append(allocator, rendered[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(allocator);
}

/// True if the template contains a branch keyed on the literal `tool` role string
/// (`'tool'` or `"tool"` as a standalone token, not `tool_calls` / `tool_call_id`).
fn templateReferencesToolRole(tpl: []const u8) bool {
    const patterns = [_][]const u8{ "'tool'", "\"tool\"" };
    for (patterns) |p| {
        if (std.mem.indexOf(u8, tpl, p) != null) return true;
    }
    return false;
}

/// A template that renders tool CALLS but names no 'tool' role (a generic
/// `<|im_start|>{{ role }}` header) was trained on tool turns: probe-render one
/// and keep it native when its content survives.
fn templateRendersToolTurn(allocator: std.mem.Allocator, tpl: []const u8) !bool {
    if (std.mem.indexOf(u8, tpl, "tool_calls") == null) return false;
    const marker = "__mlx_serve_tool_turn_probe__";
    const tpl_z = try allocator.dupeSentinel(u8, tpl, 0);
    defer allocator.free(tpl_z);
    var len: usize = 0;
    const out = jinja_c.jinja_render_chat(tpl_z.ptr, "[{\"role\":\"tool\",\"content\":\"" ++ marker ++ "\"}]", null, "{}", 0, &len) orelse return false;
    defer jinja_c.jinja_str_free(out);
    return std.mem.indexOf(u8, out[0..len], marker) != null;
}

/// True if any message has `role: "tool"` or an assistant message with tool_calls.
fn messagesHaveToolContent(messages: []const Message) bool {
    for (messages) |m| {
        if (std.mem.eql(u8, m.role, "tool")) return true;
        if (m.tool_calls != null) return true;
    }
    return false;
}

/// Synthesize a messages array suitable for templates that have no tool/role support.
/// - inject_tool_prompt: prepend (or merge into) a system message with the tool prompt
/// - rewrite_tool_role: rewrite role:"tool" messages to role:"user" with <tool_response> wrapping
/// - rewrite_tool_calls: rewrite assistant messages that carry tool_calls into plain
///   content emitting <tool_call>...</tool_call> blocks (matches what we instruct the
///   model to produce, so multi-turn context stays consistent).
fn synthesizeToolFallbackMessages(
    arena: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    inject_tool_prompt: bool,
    rewrite_tool_role: bool,
    rewrite_tool_calls: bool,
) ![]Message {
    var out = std.ArrayList(Message).empty;

    var tool_prompt: ?[]const u8 = null;
    if (inject_tool_prompt and tools_json != null) {
        var buf = std.ArrayList(u8).empty;
        try appendToolSystemPrompt(arena, &buf, tools_json.?, tool_choice_instruction);
        tool_prompt = try buf.toOwnedSlice(arena);
    } else if (tool_choice_instruction) |instr| {
        // The template renders the tools itself; only the choice is ours to add.
        tool_prompt = std.mem.trim(u8, instr, "\n");
    }

    var injected = false;
    if (tool_prompt) |tp| {
        if (messages.len == 0 or !std.mem.eql(u8, messages[0].role, "system")) {
            try out.append(arena, .{ .role = "system", .content = tp });
            injected = true;
        }
    }

    for (messages, 0..) |msg, i| {
        if (tool_prompt) |tp| {
            if (!injected and i == 0 and std.mem.eql(u8, msg.role, "system")) {
                const merged = if (msg.content.len > 0)
                    try std.fmt.allocPrint(arena, "{s}\n\n{s}", .{ msg.content, tp })
                else
                    try arena.dupe(u8, tp);
                try out.append(arena, .{
                    .role = "system",
                    .content = merged,
                    .tool_calls = msg.tool_calls,
                    .tool_call_id = msg.tool_call_id,
                    .images = msg.images,
                });
                injected = true;
                continue;
            }
        }

        if (rewrite_tool_role and std.mem.eql(u8, msg.role, "tool")) {
            const wrapped = try std.fmt.allocPrint(arena, "<tool_response>\n{s}\n</tool_response>", .{msg.content});
            try out.append(arena, .{
                .role = "user",
                .content = wrapped,
                .images = msg.images,
                .videos = msg.videos,
                .audio = msg.audio,
            });
            continue;
        }

        if (rewrite_tool_calls and std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var buf = std.ArrayList(u8).empty;
            if (msg.content.len > 0) {
                try buf.appendSlice(arena, msg.content);
                try buf.append(arena, '\n');
            }
            for (msg.tool_calls.?) |tc| {
                try buf.appendSlice(arena, "<tool_call>\n{\"name\": \"");
                try buf.appendSlice(arena, tc.name);
                try buf.appendSlice(arena, "\", \"arguments\": ");
                try buf.appendSlice(arena, tc.arguments);
                try buf.appendSlice(arena, "}\n</tool_call>\n");
            }
            const owned = try buf.toOwnedSlice(arena);
            try out.append(arena, .{
                .role = "assistant",
                .content = std.mem.trimEnd(u8, owned, "\n"),
                .images = msg.images,
            });
            continue;
        }

        try out.append(arena, msg);
    }

    return out.toOwnedSlice(arena);
}

/// How an assistant message with NO content should serialize.
///
/// Most templates read `message.content is none` / `if message.content`, and
/// `null` is what they expect. A template that tests whether the content KEY
/// exists and then indexes into the VALUE breaks on `"content": null`; a
/// tool-calling assistant turn carries no content, so every tool round-trip
/// would hit it. Serializing `""` satisfies both readings: the key exists, and it
/// is falsy everywhere the template branches on it.
pub const EmptyContent = enum { null_literal, empty_string };

pub fn serializeMessagesJson(allocator: std.mem.Allocator, messages: []const Message) ![]const u8 {
    return serializeMessagesJsonOpts(allocator, messages, .null_literal);
}

pub fn serializeMessagesJsonOpts(allocator: std.mem.Allocator, messages: []const Message, empty_content: EmptyContent) ![]const u8 {
    return serializeMessagesJsonImpl(allocator, messages, empty_content, null);
}

pub fn serializeMessagesJsonFor(allocator: std.mem.Allocator, messages: []const Message, empty_content: EmptyContent, chat_config: *const ChatConfig) ![]const u8 {
    return serializeMessagesJsonImpl(allocator, messages, empty_content, chat_config.video_placeholder);
}

pub fn messageHasMedia(msg: Message) bool {
    return (msg.images != null and msg.images.?.len > 0) or
        (msg.videos != null and msg.videos.?.len > 0) or
        (msg.audio != null and msg.audio.?.len > 0);
}

/// Media as content parts, so the model's own template renders one
/// placeholder per item where the message sits: images, videos, audio (the
/// order the server encodes them), then the text.
fn appendMediaContentParts(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), msg: Message, video_placeholder: ?[]const u8) !void {
    try buf.append(allocator, '[');
    var first = true;
    for (msg.images orelse &[_]ImageData{}) |_| {
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "{\"type\":\"image\"}");
    }
    for (msg.videos orelse &[_]VideoData{}) |_| {
        if (!first) try buf.append(allocator, ',');
        first = false;
        if (video_placeholder) |atom| {
            try buf.appendSlice(allocator, "{\"type\":\"text\",\"text\":");
            try appendJsonString(allocator, buf, atom);
            try buf.append(allocator, '}');
        } else try buf.appendSlice(allocator, "{\"type\":\"video\"}");
    }
    for (msg.audio orelse &[_]AudioData{}) |_| {
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "{\"type\":\"audio\"}");
    }
    if (msg.content.len > 0) {
        try buf.appendSlice(allocator, ",{\"type\":\"text\",\"text\":");
        try appendJsonString(allocator, buf, msg.content);
        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');
}

fn serializeMessagesJsonImpl(allocator: std.mem.Allocator, messages: []const Message, empty_content: EmptyContent, video_placeholder: ?[]const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"role\":");
        try appendJsonString(allocator, &buf, msg.role);

        try buf.appendSlice(allocator, ",\"content\":");
        if (messageHasMedia(msg)) {
            try appendMediaContentParts(allocator, &buf, msg, video_placeholder);
        } else if (msg.content.len > 0) {
            try appendJsonString(allocator, &buf, msg.content);
        } else switch (empty_content) {
            .null_literal => try buf.appendSlice(allocator, "null"),
            .empty_string => try buf.appendSlice(allocator, "\"\""),
        }

        if (msg.tool_calls) |tcs| {
            try buf.appendSlice(allocator, ",\"tool_calls\":[");
            for (tcs, 0..) |tc, ti| {
                if (ti > 0) try buf.append(allocator, ',');
                try buf.appendSlice(allocator, "{\"type\":\"function\",");
                if (tc.id.len > 0) {
                    // A template may name a tool RESULT by matching
                    // `message.tool_call_id` against the history call's `id`.
                    // Inert for templates that never read it.
                    try buf.appendSlice(allocator, "\"id\":");
                    try appendJsonString(allocator, &buf, tc.id);
                    try buf.append(allocator, ',');
                }
                try buf.appendSlice(allocator, "\"function\":{\"name\":");
                try appendJsonString(allocator, &buf, tc.name);
                try buf.appendSlice(allocator, ",\"arguments\":");
                // Embed arguments as a JSON OBJECT when it parses cleanly -
                // some templates do `tool_call.arguments|items` which requires
                // a dict. Templates that need a string (e.g. Gemma 4 with `tojson`
                // or `string`) still get a usable value. Falls back to a string
                // for malformed arguments so downstream still sees something.
                if (std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        try buf.appendSlice(allocator, tc.arguments);
                    } else {
                        try appendJsonString(allocator, &buf, tc.arguments);
                    }
                } else |_| {
                    try appendJsonString(allocator, &buf, tc.arguments);
                }
                try buf.appendSlice(allocator, "}}");
            }
            try buf.append(allocator, ']');
        }

        if (msg.tool_call_id) |tid| {
            try buf.appendSlice(allocator, ",\"tool_call_id\":");
            try appendJsonString(allocator, &buf, tid);
            // No tool_responses field needed — Gemma 4 templates handle role:"tool"
            // natively via the content field. Adding tool_responses causes the template
            // to render duplicate content, wasting tokens.
        }

        if (msg.reasoning_content) |rc| {
            // Key omitted entirely when absent: templates gate on
            // `message.reasoning_content is string`, and an explicit null
            // would still pass `is defined`-style checks.
            try buf.appendSlice(allocator, ",\"reasoning_content\":");
            try appendJsonString(allocator, &buf, rc);
        }

        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');

    return buf.toOwnedSlice(allocator);
}

/// Fill the OPTIONAL tool-definition keys a template may dump blindly.
///
/// `description` is optional in OpenAI's function schema and a no-arg tool
/// needs no `parameters`, but a template that renders `fn.description | tojson`
/// or `tool | tojson` wholesale hits `tojson` on Undefined, which is a
/// jinja RAISE — i.e. a silent `fallbackFormatChat` with no tools preamble at
/// all, for a request that was perfectly legal. Returns null when every
/// function already carries both keys, so the common path re-emits nothing and
/// the caller keeps the client's own bytes (provenance, not content, decides
/// ownership).
///
/// Defaults are the empty ones: `""` says the client supplied no description,
/// and an empty-properties object is what a no-argument tool means. Neither
/// invents a capability the client did not declare.
pub fn fillOptionalToolDefKeys(allocator: std.mem.Allocator, tools_json: []const u8) !?[]u8 {
    // Own the intermediates: growing a parsed ObjectMap allocates OUTSIDE the
    // parse's arena, so mutating with the caller's allocator leaks unless the
    // caller happens to be an arena too. Only the returned bytes are theirs.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, a, tools_json, .{}) catch return null;
    if (parsed.value != .array) return null;

    var changed = false;
    for (parsed.value.array.items) |*tool| {
        if (tool.* != .object) continue;
        const fn_val = tool.object.getPtr("function") orelse continue;
        if (fn_val.* != .object) continue;
        if (fn_val.object.get("description") == null) {
            try fn_val.object.put(a, "description", .{ .string = "" });
            changed = true;
        }
        if (fn_val.object.get("parameters") == null) {
            var params: std.json.ObjectMap = .empty;
            try params.put(a, "type", .{ .string = "object" });
            try params.put(a, "properties", .{ .object = .empty });
            try fn_val.object.put(a, "parameters", .{ .object = params });
            changed = true;
        }
    }
    if (!changed) return null;

    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

pub fn templateConsumesEffort(tpl: []const u8) bool {
    return std.mem.indexOf(u8, tpl, "reasoning_effort") != null;
}

fn appendKwarg(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), key: []const u8, value: std.json.Value) !void {
    if (buf.items.len > 1) try buf.append(allocator, ',');
    try appendJsonString(allocator, buf, key);
    try buf.append(allocator, ':');
    const v = try std.json.Stringify.valueAlloc(allocator, value, .{});
    defer allocator.free(v);
    try buf.appendSlice(allocator, v);
}

/// A request's `chat_template_kwargs` over the model's: one object, request keys win.
pub fn mergeTemplateKwargs(allocator: std.mem.Allocator, model_kwargs: ?[]const u8, request: std.json.ObjectMap) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '{');
    var it = request.iterator();
    while (it.next()) |kv| try appendKwarg(allocator, &buf, kv.key_ptr.*, kv.value_ptr.*);
    if (model_kwargs) |raw| {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
        defer parsed.deinit();
        if (parsed.value == .object) {
            var mit = parsed.value.object.iterator();
            while (mit.next()) |kv| if (!request.contains(kv.key_ptr.*)) try appendKwarg(allocator, &buf, kv.key_ptr.*, kv.value_ptr.*);
        }
    }
    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

fn serializeExtraContext(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '{');
    var need_comma = false;

    if (chat_config.bos_token) |bos| {
        try buf.appendSlice(allocator, "\"bos_token\":");
        try appendJsonString(allocator, &buf, bos);
        need_comma = true;
    }
    if (chat_config.eos_token) |eos| {
        if (need_comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"eos_token\":");
        try appendJsonString(allocator, &buf, eos);
        need_comma = true;
    }
    if (need_comma) try buf.append(allocator, ',');
    if (enable_thinking) {
        try buf.appendSlice(allocator, "\"enable_thinking\":true");
    } else {
        try buf.appendSlice(allocator, "\"enable_thinking\":false");
    }
    // Templates that read `reasoning_effort` get the mapped value; the rest ignore it.
    try buf.appendSlice(allocator, if (enable_thinking)
        ",\"reasoning_effort\":\"high\""
    else
        ",\"reasoning_effort\":\"no_think\"");

    var kwargs: ?std.json.Parsed(std.json.Value) = null;
    defer if (kwargs) |*k| k.deinit();
    if (chat_config.chat_template_kwargs) |raw| {
        kwargs = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch null;
    }
    const kw_obj: ?std.json.ObjectMap = if (kwargs) |k| (if (k.value == .object) k.value.object else null) else null;

    if (kw_obj) |obj| {
        // Set above from resolved values, or the wrapper's own context.
        const reserved = [_][]const u8{ "bos_token", "eos_token", "enable_thinking", "reasoning_effort", "thinking_mode", "reasoning_strength", "messages", "tools", "add_generation_prompt" };
        var it = obj.iterator();
        while (it.next()) |kv| {
            var skip = false;
            for (reserved) |r| skip = skip or std.mem.eql(u8, r, kv.key_ptr.*);
            if (!skip) try appendKwarg(allocator, &buf, kv.key_ptr.*, kv.value_ptr.*);
        }
    }

    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

/// Encode text that may contain special tokens (like <|im_start|>, <bos>, etc.).
fn encodeWithSpecialTokens(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    text: []const u8,
    ids: *std.ArrayList(u32),
) !void {
    // `Tokenizer.encode` already splits around special tokens (earliest
    // occurrence, longest at a position) with an O(text) first-byte-bucketed
    // scan. This wrapper used to duplicate that split with its own
    // O(specials x text) re-search per segment — ~11 s per 66 KB rendered
    // prompt on a 6415-special vocabulary (the tokenizer-side twin
    // of the same class was fixed in tokenizer.zig; keep both on the shared
    // fast path so they can't drift apart again).
    const segment_ids = try tok.encode(allocator, text);
    defer allocator.free(segment_ids);
    try ids.appendSlice(allocator, segment_ids);
}

const SpecialEntry = struct {
    text: []const u8,
    id: u32,
};

/// Fallback chat formatting for when Jinja rendering fails.
fn fallbackFormatChat(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    if (hasExtraSystemMessages(messages)) {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var folded = std.ArrayList(Message).empty;
        try folded.appendSlice(arena.allocator(), messages);
        _ = try foldSystemMessages(arena.allocator(), &folded, false);
        return fallbackFormatChat(allocator, folded.items, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    }

    const has_system = messages.len > 0 and std.mem.eql(u8, messages[0].role, "system");

    // The `<start_of_turn>` format (Decision 6): the one fallback format.
    if (chat_config.bos_token) |bos| {
        try result.appendSlice(allocator, bos);
    }
    if (tools_json != null and !has_system) {
        try result.appendSlice(allocator, "<start_of_turn>user\n");
        try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
        try result.appendSlice(allocator, "<end_of_turn>\n");
    }
    for (messages, 0..) |msg, i| {
        if (std.mem.eql(u8, msg.role, "tool")) {
            try result.appendSlice(allocator, "<start_of_turn>user\n");
            try result.appendSlice(allocator, "Tool result: ");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "<end_of_turn>\n");
        } else {
            try result.appendSlice(allocator, "<start_of_turn>");
            try result.appendSlice(allocator, msg.role);
            try result.appendSlice(allocator, "\n");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "<end_of_turn>\n");
        }
        // The tool block follows the system turn, as it leads the prompt when there is no system message.
        if (i == 0 and has_system and tools_json != null) {
            try result.appendSlice(allocator, "<start_of_turn>user\n");
            try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            try result.appendSlice(allocator, "<end_of_turn>\n");
        }
    }
    try result.appendSlice(allocator, "<start_of_turn>model\n");

    return result.toOwnedSlice(allocator);
}

/// The FIRST unclosed think/thought opener: the earliest `<|channel>thought`
/// or `<think>` in `text` with no matching close tag anywhere after it.
/// Gemma 4 opens a NEW thought channel at the very end of a prose turn (its
/// channel-thought tail behavior) — and sometimes SEVERAL in a row, none of
/// them closed (seen live from the 26B model via pi). Everything from the
/// first such opener onward is dangling thought; cutting at the LAST opener
/// instead leaks the earlier raw tags into visible content. A pos-0 opener is
/// reported too: callers (the split arms / stripTrailingThinkOpen) strip
/// their leading CLOSED block first, so an unclosed opener at the start of the
/// remainder is a genuine dangling re-open (`…<channel|>\n<|channel>thought\n`,
/// 2026-06-19 live) — excluding it leaked the bare opener into content.
const TrailingThinkOpen = struct { pos: usize, after: usize };
fn lastUnclosedThinkOpen(text: []const u8) ?TrailingThinkOpen {
    var from: usize = 0;
    while (nextThinkOpen(text, from)) |o| {
        const close = if (o.is_think_style)
            indexOfThinkCloseTag(text, o.after)
        else if (std.mem.indexOfPos(u8, text, o.after, "<channel|>")) |p|
            TagAt{ .pos = p, .len = "<channel|>".len }
        else
            null;
        if (close) |c| {
            // This block IS closed — keep scanning past its close.
            from = c.pos + c.len;
            continue;
        }
        return .{ .pos = o.pos, .after = o.after };
    }
    return null;
}

// ── Think tag matching ──
//
// The think tags are the literal `<think>` and `</think>`.

pub const TagAt = struct { pos: usize, len: usize };

/// Length of `tag` when `text` starts with it, else null.
fn tagLenAt(text: []const u8, comptime tag: []const u8) ?usize {
    return if (std.mem.startsWith(u8, text, tag)) tag.len else null;
}

/// True when `t` is a strict, non-empty prefix of `tag`: it could still GROW into the tag with more streamed bytes.
fn isPartialTag(t: []const u8, comptime tag: []const u8) bool {
    return t.len > 0 and t.len < tag.len and std.mem.startsWith(u8, tag, t);
}

pub fn thinkOpenTagLenAt(text: []const u8) ?usize {
    return tagLenAt(text, "<think>");
}

pub fn thinkCloseTagLenAt(text: []const u8) ?usize {
    return tagLenAt(text, "</think>");
}

fn indexOfThinkOpenTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "<think")) |p| {
        if (thinkOpenTagLenAt(text[p..])) |l| return .{ .pos = p, .len = l };
        i = p + "<think".len;
    }
    return null;
}

pub fn indexOfThinkCloseTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "</think")) |p| {
        if (thinkCloseTagLenAt(text[p..])) |l| {
            if (!thinkCloseIsToolCallPayload(text, p)) return .{ .pos = p, .len = l };
        }
        i = p + "</think".len;
    }
    return null;
}

/// Whether a `</think>` at `pos` is an ARGUMENT VALUE rather than a block close.
///
/// The think split runs before tool parse, and this is the shared close-tag
/// scan every surface uses — so a model writing about its own prompt format
/// (`<arg_value>…closes a thought with </think>…</arg_value>`) had the split cut
/// straight through the call, losing it entirely and leaking the fragments.
/// Agent traffic writes files about prompts.
///
/// Payload ⟺ the nearest tool opener before `pos` is still OPEN there (no
/// `</tool_call` between it and `pos`) AND the block does close afterwards.
/// Both halves are load-bearing:
///   • without the first, a call the model emitted and CLOSED inside its
///     thought would make the real `</think>` after it look like payload;
///   • without the second, an unclosed opener inside a thought (the leaked-
///     markup case, which `trimLeakedToolMarkup` handles downstream) would
///     swallow the answer that follows the close.
fn thinkCloseIsToolCallPayload(text: []const u8, pos: usize) bool {
    // `streamThinkGate2` calls the close scan once per TOKEN on a growing
    // buffer, so the common case — text with no tool markup at all — must cost
    // one substring scan, not a walk per opener spelling. Both spellings share
    // this needle.
    if (std.mem.indexOf(u8, text[0..pos], "tool_call") == null) return false;
    // Nearest `<tool_call`-family opener before `pos`. Only this family is
    // considered: it is the one whose bodies carry free-form argument text.
    var open: ?usize = null;
    for ([_][]const u8{ "<tool_call", "<|tool_call" }) |m| {
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, text, i, m)) |p| {
            if (p >= pos) break;
            if (open == null or p > open.?) open = p;
            i = p + 1;
        }
    }
    const o = open orelse return false;
    if (std.mem.indexOfPos(u8, text, o, "</tool_call")) |c| {
        if (c < pos) return false; // that call already closed — a real block close
        return true;
    }
    return false; // never closes — treat as leaked markup, not payload
}

/// Length of the trailing bytes of `buf` that could still GROW into a think
/// close tag with more streamed bytes (either family): `</think>` or
/// Gemma's `<channel|>`. Streaming
/// reasoning flushes must hold this suffix back: flushing half a close tag
/// splits it across the flush boundary, so the retained remainder never
/// matches and the stream stays "inside thinking" forever (tag bytes leak
/// into reasoning_content). 0 = safe to flush everything.
pub fn partialThinkCloseSuffixLen(buf: []const u8) usize {
    // Any real close tag is well under 32 bytes; a longer '<'-run is prose.
    const window_start = if (buf.len > 32) buf.len - 32 else 0;
    var i = buf.len;
    while (i > window_start) {
        i -= 1;
        if (buf[i] != '<') continue;
        const tail = buf[i..];
        if (isPartialTag(tail, "</think>")) return tail.len;
        if (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail)) return tail.len;
    }
    return 0;
}

test "partialThinkCloseSuffixLen holds back growing close tags, ignores prose" {
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("plain reasoning text"));
    try testing.expectEqual(@as(usize, 7), partialThinkCloseSuffixLen("thinking…</think"));
    try testing.expectEqual(@as(usize, 5), partialThinkCloseSuffixLen("thinking…<chan"));
    // A COMPLETE tag is not a partial — the scan-side indexOf owns it.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("done</think>"));
    // A '<' too far back is prose, not a growing tag.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("a < b and then lots of ordinary words follow here"));
}

/// Tag length when `text` ENDS with a complete think-close tag.
fn endsWithThinkCloseTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkCloseTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tag length when `text` ENDS with a complete think-open tag.
fn endsWithThinkOpenTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkOpenTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tool-call WRAPPER openers that must never survive into text a client
/// renders. Every entry is the opening marker of a call block, so whatever
/// follows it is call payload, not prose — parseToolCalls has already run by
/// the time these matter, so a marker still present means the block did not
/// parse (a mangled opener, a bare marker with no call behind it, or a call
/// the model emitted inside its think block, which the parser deliberately
/// skips). The whole tail is cut rather than excised: a wrapper we could not
/// parse has no reliable end, and shipping half of it is the same leak.
const tool_markup_openers = [_][]const u8{
    "<｜DSML｜", // native DSML (covers tool_calls/invoke/parameter)
    "<|tool_call", // Gemma 4
    "<tool_call", // Hermes XML + the plural/suffixed variants
};

/// Earliest index of Anthropic-style `<function_calls>` / `<invoke …>` markup, or null. Like other dialects,
/// the opener alone is prose ("wrap it in <invoke name=…>"): a `<parameter ` or `</invoke>` must be there too.
fn bareInvokeMarkupCut(text: []const u8) ?usize {
    if (std.mem.indexOf(u8, text, "<parameter ") == null and
        std.mem.indexOf(u8, text, BARE_INVOKE.invoke_close) == null) return null;
    var cut = std.mem.indexOf(u8, text, FUNCTION_CALLS_TAG);
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, text, scan, BARE_INVOKE.invoke)) |p| {
        const after = p + BARE_INVOKE.invoke.len;
        if (after >= text.len or std.ascii.isWhitespace(text[after])) {
            if (cut == null or p < cut.?) cut = p;
            break;
        }
        scan = after;
    }
    return cut;
}

/// True while `buf` holds, or ends in a growing prefix of, a `<function_calls>` / `<invoke …` opener.
fn invokeOpenerHoldsForTools(buf: []const u8) bool {
    if (std.mem.indexOf(u8, buf, FUNCTION_CALLS_TAG) != null) return true;
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, buf, scan, BARE_INVOKE.invoke)) |p| {
        const after = p + BARE_INVOKE.invoke.len;
        if (after >= buf.len or std.ascii.isWhitespace(buf[after])) return true;
        scan = after;
    }
    inline for (.{ FUNCTION_CALLS_TAG, BARE_INVOKE.invoke }) |marker| {
        var k: usize = 2;
        while (k < marker.len) : (k += 1) if (std.mem.endsWith(u8, buf, marker[0..k])) return true;
    }
    return false;
}

/// Cut visible text at the first unparsed tool-call marker. Returns a prefix
/// slice (no alloc); text with no marker is returned untouched.
pub fn trimLeakedToolMarkup(text: []const u8) []const u8 {
    var cut: usize = text.len;
    for (tool_markup_openers) |m| {
        if (std.mem.indexOf(u8, text, m)) |p| {
            if (p < cut) cut = p;
        }
    }
    if (bareInvokeMarkupCut(text)) |p| {
        if (p < cut) cut = p;
    }
    if (cut == text.len) return text;
    return std.mem.trimEnd(u8, text[0..cut], "\n\r\t ");
}

// NOTE: there is deliberately NO strip-think helper anymore. Reasoning the
// model generated is ALWAYS delivered (reasoning_content / thinking blocks) —
// thinking-off is enforced on the PROMPT side (`noThinkTailSuffix`), so every
// delivery site takes `splitThinkBlock(text, true, …)` and ships both halves.

/// Trim trailing think/channel/tool CLOSE markers (and surrounding whitespace)
/// from visible content. A close marker is never valid at the tail of content —
/// but a degenerate model can spam them (live: a Gemma variant emitted 16 bare
/// `<channel|>` after its answer; the leading strip cuts the FIRST close, the
/// trailing-OPEN strip ignores closes, so they leaked). The Gemma tool CLOSE
/// `<tool_call|>` is here too: a degenerate 1-token bare close with no
/// `<|tool_call>` opener leaked as the entire content (live 2026-07-16 soak,
/// gemma-4-26B). parseToolCalls runs BEFORE this strip and extracts any real
/// call, so any residual `<tool_call|>` reaching here is orphan by construction.
/// Loops so a run of closers is fully removed. Returns a prefix slice (no alloc).
pub fn trimTrailingThinkClosers(content: []const u8) []const u8 {
    var s = content;
    while (true) {
        const t = std.mem.trimEnd(u8, s, "\n \t\r");
        if (std.mem.endsWith(u8, t, "<channel|>")) {
            s = t[0 .. t.len - "<channel|>".len];
        } else if (std.mem.endsWith(u8, t, "<tool_call|>")) {
            s = t[0 .. t.len - "<tool_call|>".len];
        } else if (endsWithThinkCloseTag(t)) |l| {
            s = t[0 .. t.len - l];
        } else {
            return t;
        }
    }
}

/// Truncate a trailing unclosed thought opener (and its dangling thought)
/// out of visible content. Used by the split arms after the leading block has
/// been handled.
fn stripTrailingThinkOpen(content: []const u8) []const u8 {
    if (lastUnclosedThinkOpen(content)) |o| {
        return std.mem.trimEnd(u8, content[0..o.pos], "\n ");
    }
    return content;
}

pub const ThinkSplit = struct {
    reasoning_content: ?[]const u8,
    content: []const u8,
};

/// A think block the MODEL opens when the template did not: a bare `<think>` token, or
/// Gemma 4's TWO tokens, the `<|channel>` marker (dropped from visible deltas) then the
/// word `thought`. `channel_armed` = the previous delta was that marker and nothing visible
/// has been emitted. Returns the opener to seed `think_buf` with.
pub fn modelThinkOpener(channel_armed: bool, token_text: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, std.mem.trimEnd(u8, token_text, "\n"), "<think>")) return "<think>";
    if (channel_armed and std.mem.eql(u8, token_text, "thought")) return "<|channel>thought";
    return null;
}

/// Control-channel marker TOKENS that must never appear in a visible stream
/// delta. Every entry is a single special token in its family's vocab, so
/// exact-match filtering at the delta level is complete — a marker can't be
/// split across deltas. Shared by every streaming surface's flush path.
pub fn isChannelMarkerToken(tt: []const u8) bool {
    return std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
        std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>");
}

pub const BARE_THINK_OPENER = "<think>";
pub const BARE_THINK_CLOSER = "</think>";

/// template opened (for example `…assistant\n<think>\n`). Callers
/// decode the last few prompt tokens and pass the tail here. A thinking-off
/// render ends with a CLOSED `</think>` block and must not match.
pub fn promptTailOpensThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return endsWithThinkOpenTag(trimmed) != null;
}

/// Narrow form used by generation-side phase switches: only a bare `<think>`
/// tail arms the bare `</think>` boundary.
pub fn promptTailOpensBareThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return std.mem.endsWith(u8, trimmed, "<think>");
}

/// Which reasoning opener a rendered generation prompt ENDS with (trailing
/// whitespace ignored): a bare `<think>`, or nothing. Generation-side protocol
/// resolution consumes this.
pub const ThinkTailClass = enum {
    bare,
    none,
};

pub fn promptThinkTailClass(tail: []const u8) ThinkTailClass {
    const trimmed_tail = std.mem.trimEnd(u8, tail, "\n\r\t ");
    if (endsWithThinkOpenTag(trimmed_tail)) |l| {
        const tag = trimmed_tail[trimmed_tail.len - l ..];
        if (std.mem.eql(u8, tag, BARE_THINK_OPENER)) return .bare;
    }
    return .none;
}

/// Suffix that COMMITS the no-think channel in the rendered prompt, or null.
/// Thinking-off on an always-thinking template otherwise means generating the
/// whole reasoning pass and discarding it — the prompt-side commit is what
/// makes thinking-off a real skip.
///   • Unconditional think opener: close the block the
///     template opened. Thinking-off removed-model style renders end CLOSED and never
///     match; no tools gate — tool calls ride outside the think block.
fn noThinkTailSuffix(rendered: []const u8, enable_thinking: bool) ?[]const u8 {
    if (enable_thinking) return null;
    const tail = contentChannelTail(rendered);
    return if (tail.len > 0) tail else null;
}

/// What must be appended to a rendered prompt so generation resumes in the
/// assistant's VISIBLE CONTENT channel rather than inside a reasoning block.
///
/// Two callers, one rule, deliberately: thinking-off enforces the content
/// channel so no reasoning pass runs, and a CONTINUATION needs the same thing
/// for a different reason — the partial text is prose the user has already
/// read, so a prompt that lands inside a think block would return the rest of
/// their answer as reasoning. Written twice, the two callers would
/// drift apart the first time a third family joined.
fn contentChannelTail(rendered: []const u8) []const u8 {
    const tail = std.mem.trimEnd(u8, rendered[rendered.len -| 64 ..], "\n\r\t ");
    if (promptTailOpensThink(tail)) return "</think>";
    return "";
}

/// Whether this conversation SHAPE is a continuation: it ends with an
/// assistant message carrying prose. The one predicate every surface asks, so
/// "what counts as continuable" cannot differ between the endpoint that
/// implies it and the endpoint that takes a flag — and so the answer matches
/// what `continuationPrefill` will actually do with it.
pub fn continuationRequested(messages: []const Message) bool {
    return continuationPrefill(messages, true) != null;
}

/// The partial assistant reply a continuation request is asking to extend, or
/// null when this is an ordinary turn.
///
/// The flag describes the SHAPE of the request, so a client that sets it on a
/// conversation not ending in an assistant message gets an ordinary turn —
/// never the user's own words committed into the assistant's mouth.
///
/// The prefill is trimmed of trailing whitespace: a tokenizer merges a trailing
/// space into the word that follows, so a prompt ending in one tokenizes
/// differently than the same text does mid-reply. (Anthropic's API rejects such
/// a prefill outright; trimming keeps the request serviceable, and only the
/// PROMPT is trimmed — the stored reply keeps its whitespace.)
fn continuationPrefill(messages: []const Message, continue_final: bool) ?[]const u8 {
    if (!continue_final or messages.len == 0) return null;
    const last = messages[messages.len - 1];
    if (!std.mem.eql(u8, last.role, "assistant")) return null;
    // A reply that ended in a tool call has no prose to resume — splicing a
    // continuation onto it would re-run the parse chain over half a call.
    if (last.tool_calls != null) return null;
    const trimmed = std.mem.trimEnd(u8, last.content, " \t\r\n");
    return if (trimmed.len > 0) trimmed else null;
}

/// Split model output into reasoning_content and content.
/// Handles both `<think>...</think>` and Gemma 4's `<|channel>thought\n...<channel|>`.
/// `opened_by_template`: the generation prompt ended with a template-injected
/// think opener (see `promptTailOpensThink`), so output with no literal opener
/// and no close tag is still reasoning — generation started inside the block.
pub fn splitThinkBlock(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    const split = splitThinkBlockKeepingMarkup(text, thinking, opened_by_template);
    const reasoning: ?[]const u8 = if (split.reasoning_content) |r| blk: {
        const t = trimLeakedToolMarkup(r);
        break :blk if (t.len > 0) t else null;
    } else null;
    return .{ .reasoning_content = reasoning, .content = trimLeakedToolMarkup(split.content) };
}

/// First match of a client stop string at or after `from` that the split delivers as
/// answer content, never reasoning. Judged on the text up to the match only, so a
/// stream and its finished text cut at the same byte.
pub fn answerStopIndex(text: []const u8, from: usize, stop: []const u8, opened_by_template: bool) ?usize {
    var pos = from;
    while (std.mem.indexOfPos(u8, text, pos, stop)) |idx| : (pos = idx + 1) {
        const head = text[0 .. idx + stop.len];
        const c = splitThinkBlockKeepingMarkup(head, true, opened_by_template).content;
        const start = @intFromPtr(c.ptr) -% @intFromPtr(head.ptr);
        if (c.len == 0 or start > idx) continue;
        // The split trims trailing whitespace off the content; a match right after it is still answer.
        const end = start + c.len;
        if (idx <= end or std.mem.trim(u8, head[end..idx], " \t\r\n").len == 0) return idx;
    }
    return null;
}

/// The split WITHOUT the leaked-markup cut — for the one caller that feeds the
/// content back to `parseToolCalls` (the /v1/messages non-streaming path).
/// Every arm below returns raw slices; the cut is applied ONCE in the wrapper
/// above so a new arm cannot forget it.
pub fn splitThinkBlockKeepingMarkup(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    // Gemma 4 style: <|channel>thought\n...<channel|>\n<|channel>\ncontent
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        const think_tag = "<|channel>thought\n";
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, think_tag)) think_tag.len else if (std.mem.startsWith(u8, text, "<|channel>thought")) "<|channel>thought".len else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        var content = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
        // Strip the content channel tag: <|channel>\n or <|channel>. A
        // re-opened THOUGHT channel (`<|channel>thought…`) is NOT a content
        // opener — stripping its `<|channel>` prefix here left a bare "thought"
        // in visible content (2026-06-19 live); leave it for stripTrailingThinkOpen.
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>") and !std.mem.startsWith(u8, content, "<|channel>thought")) {
            content = content[10..];
        }
        content = std.mem.trimStart(u8, content, "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    // Standard style: <think>...</think>
    if (indexOfThinkCloseTag(text, 0)) |close| {
        const reasoning_start: usize = thinkOpenTagLenAt(text) orelse 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..close.pos], "\n ");
        const content = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    if (thinking) {
        // Unclosed think block: split policy depends on whether the model's
        // output begins with a literal opener.
        //   • Literal opener present → model definitely entered thinking but
        //     ran out of tokens / didn't close. Treat as reasoning.
        //   • No literal opener + template-injected opener → generation began
        //     INSIDE the block (some templates render `…assistant\n<think>\n`);
        //     an unclosed tail is truncated reasoning, never content.
        //   • No literal opener + no template opener → the model answered
        //     directly (Gemma style); keep the answer visible as content.
        if (thinkOpenTagLenAt(text) != null or std.mem.startsWith(u8, text, "<|channel>thought")) {
            const start: usize = if (thinkOpenTagLenAt(text)) |l| l else if (std.mem.startsWith(u8, text, "<|channel>thought\n")) "<|channel>thought\n".len else "<|channel>thought".len;
            const reasoning = std.mem.trimStart(u8, text[start..], "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        if (opened_by_template) {
            const reasoning = std.mem.trimStart(u8, text, "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        // Prose answer that ENDS by opening a new, unclosed thought block
        // (Gemma 12B tail behavior): the text before the opener is the
        // answer; the dangling thought is reasoning. The raw opener tag must
        // never leak into visible content.
        if (lastUnclosedThinkOpen(text)) |o| {
            const content = std.mem.trimEnd(u8, std.mem.trimStart(u8, text[0..o.pos], "\n "), "\n ");
            const reasoning = std.mem.trim(u8, text[o.after..], "\n ");
            return .{
                .reasoning_content = if (reasoning.len > 0) reasoning else null,
                .content = content,
            };
        }
        // No thought block, but the model may have emitted (or been truncated
        // right after) a dangling Gemma 4 *content* channel opener `<|channel>`
        // / `<|channel>\n`. Strip it so a cut-off reply never leaks the raw
        // control tag into visible content; whatever follows is the answer.
        var content = std.mem.trimStart(u8, text, "\n ");
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>")) {
            content = content[10..];
        }
        return .{ .reasoning_content = null, .content = std.mem.trimStart(u8, content, "\n ") };
    }
    return .{ .reasoning_content = null, .content = text };
}

const ThinkOpen = struct { pos: usize, after: usize, is_think_style: bool };

/// Earliest think/thought opener at or after `from` (either tag family).
fn nextThinkOpen(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<|channel>thought");
    const think = indexOfThinkOpenTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<|channel>thought".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Earliest close tag at or after `from` (either tag family).
fn nextThinkClose(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<channel|>");
    const think = indexOfThinkCloseTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<channel|>".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Skip an optional Gemma 4 CONTENT channel opener (`<|channel>` not followed
/// by `thought`) plus surrounding newlines, right after a thought close.
fn skipContentChannelTag(text: []const u8, start: usize) usize {
    var pos = start;
    while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
    const tag = "<|channel>";
    if (pos + tag.len <= text.len and std.mem.eql(u8, text[pos .. pos + tag.len], tag)) {
        const rest = text[pos + tag.len ..];
        if (!std.mem.startsWith(u8, rest, "thought")) {
            pos += tag.len;
            while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
        }
    }
    return pos;
}

/// Merge ALL closed think/thought blocks in `text` into one leading block.
///
/// The split/strip layer understands exactly one leading block plus an
/// optional UNCLOSED trailing opener. Gemma 4 12B, however, re-opens a thought
/// channel mid-turn and closes it again (`…content<|channel>thought\n…
/// <channel|>more content`) — observed live via Claude Code on /v1/messages,
/// where the raw pair leaked verbatim into the visible text block. This pass
/// rewrites such output to `<|channel>thought\n{all thought text}<channel|>
/// {all content}` so every downstream consumer (splitThinkBlock,
/// parseToolCalls) handles it unchanged.
///
/// Returns null when no rewrite is needed (zero or one LEADING closed block —
/// the overwhelmingly common case, zero-cost). An unclosed trailing opener is
/// left in place for the existing trailing-strip logic.
pub fn normalizeEmbeddedThinkBlocks(allocator: std.mem.Allocator, text: []const u8) !?[]u8 {
    var reasoning_parts = std.ArrayList([]const u8).empty;
    defer reasoning_parts.deinit(allocator);
    var content_parts = std.ArrayList([]const u8).empty;
    defer content_parts.deinit(allocator);

    var style_think = false;
    var style_set = false;
    var closed_blocks: usize = 0;
    var first_block_leading = false;
    var pos: usize = 0;

    // Template-opened leading block: a close tag BEFORE any opener (a template that
    // renders `…assistant\n<think>\n` into the prompt, so output starts
    // mid-thought and contains only the close).
    if (nextThinkClose(text, 0)) |lc| {
        const open_pos = if (nextThinkOpen(text, 0)) |o| o.pos else text.len;
        if (lc.pos < open_pos) {
            try reasoning_parts.append(allocator, std.mem.trim(u8, text[0..lc.pos], "\n "));
            style_think = lc.is_think_style;
            style_set = true;
            closed_blocks += 1;
            first_block_leading = true;
            pos = skipContentChannelTag(text, lc.after);
        }
    }

    while (true) {
        const o = nextThinkOpen(text, pos) orelse {
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        const close_tag: []const u8 = if (o.is_think_style) "</think>" else "<channel|>";
        const close_pos = std.mem.indexOfPos(u8, text, o.after, close_tag) orelse {
            // Unclosed trailing opener — leave verbatim for the trailing-strip
            // logic in splitThinkBlock.
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        try content_parts.append(allocator, text[pos..o.pos]);
        try reasoning_parts.append(allocator, std.mem.trim(u8, text[o.after..close_pos], "\n "));
        if (!style_set) {
            style_think = o.is_think_style;
            style_set = true;
        }
        closed_blocks += 1;
        if (o.pos == 0) first_block_leading = true;
        pos = skipContentChannelTag(text, close_pos + close_tag.len);
    }

    // Rewrite only when a closed block exists beyond the single leading one.
    if (closed_blocks == 0) return null;
    if (closed_blocks == 1 and first_block_leading) return null;

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, if (style_think) "<think>\n" else "<|channel>thought\n");
    var first = true;
    for (reasoning_parts.items) |r| {
        if (r.len == 0) continue;
        if (!first) try out.appendSlice(allocator, "\n\n");
        try out.appendSlice(allocator, r);
        first = false;
    }
    try out.appendSlice(allocator, if (style_think) "</think>\n" else "<channel|>\n");
    first = true;
    for (content_parts.items) |c| {
        const trimmed = std.mem.trim(u8, c, "\n ");
        if (trimmed.len == 0) continue;
        if (!first) try out.append(allocator, '\n');
        try out.appendSlice(allocator, trimmed);
        first = false;
    }
    return try out.toOwnedSlice(allocator);
}

/// Streaming-only: the head of a think block can no longer grow into any
/// opener the stream strips (`<…` markers, `assistant`/`to=` headers), so
/// a short first token is reasoning now instead of waiting for 7 bytes.
pub fn cannotOpenThink(buf: []const u8) bool {
    const rest = std.mem.trimStart(u8, buf, " \n");
    if (rest.len == 0 or rest[0] == '<') return false;
    for ([_][]const u8{ "assistant", "to=" }) |w| {
        if (std.mem.startsWith(u8, w, rest[0..@min(rest.len, w.len)])) return false;
    }
    return true;
}

/// Streaming-only: true when the buffer TAIL is a partial prefix of a think
/// opener (`<think>` / `<|channel>thought`). The buffered-stream flush must
/// hold these bytes back until the tag completes or diverges — flushing them
/// leaks tag fragments as visible content (a pi session showed prose ending
/// in a glued "thought" because `<|channel>` flushed before "thought"
/// arrived and completed the opener).
pub fn endsWithPartialThinkOpen(buf: []const u8) bool {
    const tags = [_][]const u8{ "<|channel>thought", "<think>" };
    for (tags) |tag| {
        // Strictly-partial prefixes only — a COMPLETE opener in the buffer is
        // the caller's contains-check's job; ours is the growing tail.
        var l = @min(buf.len, tag.len - 1);
        while (l > 0) : (l -= 1) {
            if (std.mem.endsWith(u8, buf, tag[0..l])) return true;
        }
    }
    return false;
}

/// Streaming-only: should the chat-completion SSE path defer flushing
/// because `buf` is on track to become a tool call?
///
/// True when:
///   * `buf` contains `<tool` followed by a valid tag terminator
///     (`>`, ` `, `\t`, `\n`, `_`, `|`) somewhere — that's any of the
///     accepted families: `<tool>`, `<tool …>`, `<tool_call…>`, `<tool_calls…>`,
///     `<tool_request…>`, `<tool_requests…>` (mirrors `parseToolCalls`).
///   * `buf` contains the Gemma 4 `<|tool_call` substring.
///     NOTE the bare equals-sign Hermes `<function=NAME>` form is deliberately
///     NOT held here: `parseToolCalls` accepts it wrapper-less, so the gate is
///     narrower than the parser for THAT dialect. Pre-existing on main, tracked
///     separately — two attempts at closing it inside this change were both
///     wrong, and real captures show non-adjacent `<parameter=` openers.
///   * `buf[0] == '{'` and `buf` contains `"name"` (raw JSON tool-call shape).
///   * `buf` ends with a partial prefix that could grow into one of the above
///     in the next token (`<`, `<t`, `<to`, `<too`, `<tool`, or any prefix
///     of `<|tool_call`).
///
/// Conservatively false on `<toolkit>`, `<toolbar>`, etc. — anything where
/// the char after `<tool` isn't a valid tag terminator. That keeps prose
/// that happens to mention HTML-ish tags flowing through normally.
pub fn streamShouldBufferForTools(buf: []const u8) bool {
    if (buf.len == 0) return false;

    // Raw JSON tool-call shape: starts with `{` and has `"name"` somewhere.
    if (buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) return true;

    // Gemma 4 fully-formed canonical open.
    if (std.mem.indexOf(u8, buf, "<|tool_call") != null) return true;

    // DSML: everything from `<｜DSML｜` on is a tool-call block.
    if (std.mem.indexOf(u8, buf, "<｜DSML｜") != null) return true;

    // `<tool…` family: walk every `<tool` occurrence, accept the first one
    // whose terminator is valid (mirrors parseToolCalls' acceptance rule).
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<tool")) |rel| {
        const after = scan + rel + "<tool".len;
        if (after >= buf.len) return true; // truncated mid-prefix → in progress
        const c = buf[after];
        if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '_' or c == '|') return true;
        scan = after;
    }

    // Anthropic-style `<function_calls>` / `<invoke name="…">`.
    if (invokeOpenerHoldsForTools(buf)) return true;

    // Trailing partial prefixes — the next streamed token could complete any
    // of these into a real tool open. Order doesn't matter; first endsWith
    // hit wins. Listed shortest-first for legibility.
    const tail_prefixes = [_][]const u8{
        "<",     "<t",     "<to",     "<too",     "<tool",
        "<|",    "<|t",    "<|to",    "<|too",    "<|tool",
        "<|tool_", "<|tool_c", "<|tool_ca", "<|tool_cal",
        // `<function` openers (Hermes `<function=`) - COMPLETE ladder. `<funct` is a
        // real decomposition in this vocabulary (`<f` id 54303 + `unct` id
        // 14185 decode to exactly `<funct`); omitting that one rung flushed the
        // fragment and leaked the rest of the tag. The rungs are DERIVED in the
        // test, so a future gap fails there instead of shipping.
        "<f",
        "<fu",      "<fun",      "<func",     "<funct",     "<functi",
        "<functio", "<function",
        // DSML fullwidth-bar prefixes (`｜` = 3 bytes; cover mid-codepoint
        // splits too in case the tokenizer spells the marker in pieces)
           "<\xef",     "<\xef\xbd",
        "<｜",
        "<｜D",
        "<｜DS",
        "<｜DSM",
        "<｜DSML",
        "<｜DSML\xef",
        "<｜DSML\xef\xbd",
    };
    for (tail_prefixes) |p| {
        if (std.mem.endsWith(u8, buf, p)) return true;
    }
    return false;
}

/// Streaming-only: what should a tools-enabled SSE path do with the buffered
/// text so far, with respect to thinking? Shared by the chat-completions and
/// Anthropic /v1/messages stream handlers — the two paths drifted apart once
/// already: /v1/messages only recognized think OPENERS present in the output,
/// so removed-model family template-opened thinking (opener injected into the PROMPT)
/// streamed as visible text and a raw `</think>` leaked into Claude Code
/// transcripts (2026-06-10 live).
///
///   .hold_thinking — inside an unclosed think block; buffer, stream only the
///                    reasoning not yet sent (`unstreamedReasoning`)
///   .split_think   — close tag arrived; splitThinkBlock once, emit
///                    reasoning + visible remainder, clear the buffer, and
///                    set think_closed for the rest of the turn
///   .flush_text    — plain visible prose; stream it
///
/// `think_closed` releases the enable_thinking hold after the one split —
/// without it the visible answer sits in the buffer until end-of-stream and
/// gets misfiled as reasoning (the pi hidden-answer bug).
pub const StreamThinkGate = enum { flush_text, hold_thinking, split_think };

/// The gate WITHOUT a prompt-opened think block — every family whose template
/// gates its `<think>` on `enable_thinking` renders the closed `<think></think>`
/// signature (or no tag at all) when thinking is off, so the flag is false for
/// them and this stays their exact behavior.
pub fn streamThinkGate(buf: []const u8, enable_thinking: bool, think_closed: bool) StreamThinkGate {
    return streamThinkGate2(buf, enable_thinking, think_closed, false);
}

/// `prompt_opened_think`: the rendered generation prompt ended with an OPEN
/// think tag (`server.promptOpensThink`). That is a fact about the prompt
/// BYTES, not about our request flag - a template can pre-open `<think>`
/// unconditionally, so with thinking off the model's reasoning arrives as
/// tag-free prose and the 3-arg gate flushed the whole chain-of-thought as the
/// visible answer (live 2026-08-04). It can only ever ADD a hold: when
/// thinking is on `enable_thinking` already held, and once `think_closed` the
/// term is dropped so the answer after the block still flushes.
/// Memoized marker scan for the streaming think gate, owned BESIDE the stream
/// buffer and reset with it.
///
/// Without it `streamThinkGate2` re-scans the whole accumulated buffer on every
/// token for as long as thinking markup is present and no close has arrived —
/// O(buffer) per token, O(n²) over a reasoning block. Measured on a hermetic
/// 4000-token stream growing to 113 KB: 47.99 µs/token with no close in the
/// buffer (every scan runs to the end), 16.98 µs/token with a close near the
/// front (the close scan stops early). ~0.3% of a 15 ms/token model, ~1% at
/// 5 ms/token, and it GROWS with the thought.
///
/// Exactness rests on one property of the marker grammar: every recognized
/// marker contains exactly ONE `<`, at its start. A marker straddling the scanned/unscanned boundary must therefore
/// begin at the LAST `<` in the buffer — and if that `<` cannot still grow into
/// a marker, nothing straddles the boundary at all. So the resume point is
/// exact and bounded, with no overlap constant to get wrong.
pub const ThinkScan = struct {
    /// Bytes already examined. Never trusted when the buffer is SHORTER than
    /// this (the loop reset the buffer without resetting the scan).
    checked_upto: usize = 0,
    /// Index of the last `<` seen, or null. The only position a marker can
    /// straddle the boundary from.
    last_lt: ?usize = null,
    saw_channel_thought: bool = false,
    saw_open: bool = false,
    saw_channel_close: bool = false,
    /// Tool markup anywhere in the buffer. This is what makes a found close
    /// UNSTABLE and therefore un-latchable: `thinkCloseIsToolCallPayload`
    /// reclassifies a close as an argument value once a `</tool_call` arrives
    /// AFTER it, so a close that was real at token N is payload at token N+k.
    /// With no tool markup the close is stable and latching is exact.
    saw_tool_marker: bool = false,
    /// First real close tag, once found and stable.
    close: ?TagAt = null,
    /// Bytes examined by the last call, INCLUDING the exact full re-scan the
    /// tool-markup case falls back to. Test-visible: the invariant is that this
    /// stays FLAT as the buffer grows for traffic with no tool markup — which
    /// is the arm the 47.99 µs/token bench measured.
    last_scan_span: usize = 0,

    /// Longest plain substring needle minus one — the overlap a windowed
    /// substring search needs so a needle split across two arrivals is caught.
    /// `<|channel>thought` is the longest at 17.
    const substr_overlap: usize = 16;

    pub fn reset(self: *ThinkScan) void {
        self.* = .{};
    }

    /// Where the next scan must start so no marker spanning the boundary is
    /// missed: the last `<` if it can still grow into a tag (see the type's
    /// doc comment for why no earlier `<` can), and far enough back for the
    /// plain substring needles either way.
    fn resumeFrom(self: *const ThinkScan, buf: []const u8) usize {
        const upto = @min(self.checked_upto, buf.len);
        const substr_from = upto -| substr_overlap;
        const lt = self.last_lt orelse return substr_from;
        if (lt >= upto) return substr_from;
        const tail = buf[lt..upto];
        const growing = isPartialTag(tail, "<think>") or
            isPartialTag(tail, "</think>") or
            (tail.len < "<|channel>thought".len and std.mem.startsWith(u8, "<|channel>thought", tail)) or
            (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail));
        return if (growing) @min(lt, substr_from) else substr_from;
    }

    fn advance(self: *ThinkScan, buf: []const u8) void {
        if (buf.len < self.checked_upto) self.reset(); // buffer was cleared under us
        if (buf.len == self.checked_upto and self.checked_upto != 0) {
            self.last_scan_span = 0;
        } else {
            const from = self.resumeFrom(buf);
            self.last_scan_span = buf.len - from;
            const win = buf[from..];
            if (std.mem.lastIndexOfScalar(u8, win, '<')) |rel| self.last_lt = from + rel;
            if (!self.saw_channel_thought and std.mem.indexOf(u8, win, "<|channel>thought") != null)
                self.saw_channel_thought = true;
            if (!self.saw_channel_close and std.mem.indexOf(u8, win, "<channel|>") != null)
                self.saw_channel_close = true;
            if (!self.saw_tool_marker and std.mem.indexOf(u8, win, "tool_call") != null)
                self.saw_tool_marker = true;
            // The open scan takes the FULL buffer with a start offset: it
            // consults nothing before the match, but a windowed slice would
            // shift the reported position. Monotone — a complete opener stays
            // one — so latching is exact.
            if (!self.saw_open and indexOfThinkOpenTag(buf, from) != null) self.saw_open = true;
            if (self.close == null and !self.saw_tool_marker)
                self.close = indexOfThinkCloseTag(buf, from);
            self.checked_upto = buf.len;
        }
        if (self.saw_tool_marker) {
            // Exact, and only here: the payload classification looks FORWARD
            // past the close for a `</tool_call`, so it must be recomputed on
            // the whole buffer. The caller latches `think_closed` right after a
            // split, so this runs for a handful of tokens, not a whole block.
            self.close = indexOfThinkCloseTag(buf, 0);
            self.last_scan_span = buf.len;
        }
    }
};

/// `streamThinkGate2` with the per-token rescan memoized. Byte-identical
/// verdicts — pinned prefix-by-prefix against the fresh gate over a corpus that
/// includes every marker family, a close buried in a tool argument, and a
/// suffix longer than the partial-scan's tail window.
pub fn streamThinkGateScan(
    buf: []const u8,
    enable_thinking: bool,
    think_closed: bool,
    prompt_opened_think: bool,
    scan: *ThinkScan,
) StreamThinkGate {
    scan.advance(buf);
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        scan.saw_channel_thought or
        scan.saw_open or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    return if (scan.saw_channel_close or scan.close != null) .split_think else .hold_thinking;
}

pub fn streamThinkGate2(buf: []const u8, enable_thinking: bool, think_closed: bool, prompt_opened_think: bool) StreamThinkGate {
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        std.mem.indexOf(u8, buf, "<|channel>thought") != null or
        indexOfThinkOpenTag(buf, 0) != null or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        // A partial opener at the buffer TAIL (mid-text re-opened channel
        // arriving token by token) must hold the flush — flushing leaks tag
        // fragments like a glued "thought".
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    const has_close = std.mem.indexOf(u8, buf, "<channel|>") != null or
        indexOfThinkCloseTag(buf, 0) != null;
    return if (has_close) .split_think else .hold_thinking;
}

/// End-of-stream classification for a think buffer that never saw a close tag:
/// is it reasoning, or was it content all along?
///
/// The streaming loop starts `in_think_block` from `enable_thinking`, i.e. it
/// ASSUMES the model begins inside a think block whenever thinking was
/// requested. That holds only for templates that pre-inject the opener. Gemma 4
/// renders a bare `<|turn>model\n` and lets the MODEL decide, so a turn it
/// answers directly ("391") carries no think markup at all — and the whole
/// answer was flushed as `reasoning_content` with EMPTY content, while the
/// non-streaming path (`splitThinkBlock`, no opener + no close + not
/// template-opened ⇒ content) returned it correctly (live 2026-08-04).
///
/// So require POSITIVE evidence a block was open: the PROMPT opened one, or the
/// model emitted a literal opener. This is what makes the two paths agree.
pub fn streamTailIsReasoning(in_think_block: bool, prompt_opened_think: bool, saw_think_open: bool) bool {
    return in_think_block and (prompt_opened_think or saw_think_open);
}

/// The reasoning bytes not yet streamed, or null when there is nothing new.
///
/// The tools streaming path emits reasoning INCREMENTALLY (see the
/// `.hold_thinking` arm), so every later site that emits reasoning for the same
/// Streaming counterpart to the leading `trimStart(content, "\n ")` every
/// non-streaming split already applies.
///
/// The two surfaces disagreed on the BYTES of the same answer: one template
/// opens `<think>` unconditionally, so its first generated token is the newline
/// pair that follows the closer, and `splitThinkBlock` drops it while a stream
/// flushing tokens verbatim shipped it — `'One, two…'` non-streaming,
/// `'\n\nOne, two…'` streaming (live 2026-08-13). Any model whose first visible
/// token is whitespace is in this class, not just that one.
///
/// Leading whitespace is the ONE thing a stream can still withhold: nothing
/// visible has been sent yet, so this is a suppression, never a retraction.
///
/// It suppresses a chunk that is ENTIRELY whitespace, never a prefix of one. A
/// partial trim would ship a logprobs entry whose `token` no longer appears in
/// `content` — the collector describes whole tokens and cannot split one — so
/// the two contracts are kept consistent by never cutting inside a token. A
/// single token mixing leading whitespace with visible text is the residual
/// case, and it keeps its whitespace on both surfaces' terms: the stream ships
/// it, and the caller flips `content_started` so nothing later is touched.
///
/// `content_started` must flip on the first NON-EMPTY emission, so a suppressed
/// whitespace token leaves the next one still eligible.
pub fn streamContentLead(chunk: []const u8, content_started: bool) []const u8 {
    if (content_started) return chunk;
    if (std.mem.trimStart(u8, chunk, "\n ").len == 0) return "";
    return chunk;
}

/// turn must send only the remainder or the client sees the thought twice.
///
/// `already >= reasoning.len` returns null rather than underflowing: the split
/// can SHRINK when a tool marker appears mid-thought and `trimLeakedToolMarkup`
/// cuts the tail. An SSE delta cannot be retracted, so the honest behavior is to
/// send nothing further — never to resend from the top.
pub fn unstreamedReasoning(reasoning: []const u8, already: usize) ?[]const u8 {
    if (already >= reasoning.len) return null;
    return reasoning[already..];
}

/// The part of an unclosed thought safe to stream: a close tag still arriving
/// in pieces (`</thi`) is held back until the next token decides it.
pub fn streamableReasoning(so_far: []const u8) []const u8 {
    return so_far[0 .. so_far.len - partialThinkCloseSuffixLen(so_far)];
}

/// Parse tool calls from model output text.
pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !?[]ParsedToolCall {
    // Strip thinking blocks if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (indexOfThinkCloseTag(text, 0)) |close| {
        effective_text = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
    }

    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    // Look for `<tool…>…</tool…>` patterns. Some models hallucinate the
    // spec aggressively; we accept the entire training-bias family:
    //   • `<tool_call>JSON</tool_call>`                      (canonical Hermes)
    //   • `<tool_call name="X">JSON</tool_call>`             (attribute style)
    //   • `<tool_call …>JSON</tool_request>`                 (mismatched close)
    //   • `<tool_calls>JSON</tool_calls>`                    (plural form)
    //   • `<tool_calls name="X">{args}</tool_calls>`         (plural + attr)
    //   • `<tool name="X">{args}</tool_calls>`               (bare-tool open, plural close — nested wrappers)
    //   • `<tool>{json}</tool>`                              (bare tag, body has name+args)
    //   • `<tool name="X" arguments="{...}"/>`               (SELF-CLOSING, which some models emit routinely;
    //                                                          parseSelfClosingToolTag handles broken-quote
    //                                                          variants like `…"}'/>` too)
    // Approach: search for the prefix `<tool` (5 chars). Skip the Gemma 4
    // close marker `<tool_call|>`. Try the self-closing form first (cheap
    // attribute scan + balanced-JSON extraction; tolerates the model's
    // mismatched quotes). If that doesn't match, fall through to the
    // open/close-marker path: scan to the next `>`, optionally capturing
    // `name="X"` / `name='X'`, then locate the earliest of `</tool>`,
    // `</tool_call>`, `</tool_calls>`, `</tool_request>`, `</tool_requests>`.
    // When the body fails to parse, advance only past the OPENING `>` so any
    // inner `<tool…>` blocks get a chance — this is what makes the outer
    // `<tool_calls>` wrapper case work.
    //
    // native DSML (`<｜DSML｜invoke …>`): distinctive fullwidth-bar
    // marker no other family emits, and the generic `<tool` scan never sees
    // it (the byte before "tool_calls" is `｜`, not `<`).
    try parseDsmlToolCalls(allocator, effective_text, &calls);
    var search_pos: usize = if (calls.items.len > 0) effective_text.len else 0;
    while (search_pos < effective_text.len) {
        const rel = std.mem.indexOf(u8, effective_text[search_pos..], "<tool") orelse break;
        const after_tool = search_pos + rel + "<tool".len;
        if (after_tool >= effective_text.len) break;

        // Reject anything that isn't actually a tool open tag — the char
        // right after `<tool` must be `>`, whitespace, `_` (suffix coming),
        // or `|` (Gemma 4 close marker). Anything else is text like
        // `<toolkit>` or `<toolbar>`; advance past `<tool` and keep scanning.
        const next = effective_text[after_tool];
        if (next != '>' and next != ' ' and next != '\t' and next != '\n' and next != '_' and next != '|') {
            search_pos = after_tool;
            continue;
        }
        // Gemma 4 close marker: `<tool_call|>`. Detected when after `<tool`
        // we see `_call|`. Let the Gemma 4 branch below pick this up.
        if (next == '_' and after_tool + 6 <= effective_text.len and std.mem.eql(u8, effective_text[after_tool .. after_tool + 6], "_call|")) {
            search_pos = after_tool + 6;
            continue;
        }
        // If suffix is `_`, the only accepted continuations are
        // `_call`, `_calls`, `_request`, `_requests`. Any OTHER suffix gets
        // one more chance as the XML-element-TAG form before being rejected:
        // some models embed the tool name in the tag itself,
        //   `<tool_read>\n<path>mlx.html</path>\n</tool_read>`
        // (2026-06-10 pi html capture; both calls leaked as text and pi
        // scored 0/4). Conditions kept tight: tag must close with `>` right
        // after the name (no attributes), an EXACT `</tool_NAME>` close must
        // exist, the name must not be a result-ish marker (`<tool_output>`
        // is the model hallucinating a result, not calling a tool named
        // "output"), and the body must be entirely `<key>value</key>` args.
        if (next == '_') {
            const accepted = [_][]const u8{ "_call", "_calls", "_request", "_requests" };
            var matched = false;
            for (accepted) |suf| {
                if (after_tool + suf.len > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[after_tool .. after_tool + suf.len], suf)) continue;
                // Char right after the suffix must be tag-terminating; otherwise
                // `_call` could swallow `<tool_called_X>`.
                const post = after_tool + suf.len;
                if (post >= effective_text.len) break;
                const c = effective_text[post];
                if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '/' or c == '|') {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                const origin = after_tool - "<tool".len;
                if (try parseXmlElementTagToolCall(allocator, effective_text[origin..])) |etc| {
                    try calls.append(allocator, .{ .name = etc.call.name, .arguments = etc.call.arguments });
                    search_pos = origin + etc.consumed;
                    continue;
                }
                search_pos = after_tool;
                continue;
            }
        }

        // Self-closing `<tool …/>` form first, which some models emit routinely
        // and the open/close scan below won't find a `</tool*>` close, so
        // without this branch the call gets dropped silently.
        const tag_origin = after_tool - "<tool".len;
        if (parseSelfClosingToolTag(effective_text[tag_origin..])) |sc| {
            const name_owned = try allocator.dupe(u8, sc.name);
            errdefer allocator.free(name_owned);
            const args_owned = try allocator.dupe(u8, sc.arguments);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
            search_pos = tag_origin + sc.consumed;
            continue;
        }

        // Scan the opening tag to its `>`, capturing optional name attr.
        var attr_name: ?[]const u8 = null;
        var content_start: usize = 0;
        {
            var i: usize = after_tool;
            // Bound the scan so a malformed input can't run wild.
            const limit = @min(effective_text.len, after_tool + 256);
            while (i < limit and effective_text[i] != '>') : (i += 1) {
                if (effective_text[i] != 'n') continue;
                if (i + 5 > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[i .. i + 5], "name=")) continue;
                const q_pos = i + 5;
                if (q_pos >= effective_text.len) break;
                const quote = effective_text[q_pos];
                if (quote != '"' and quote != '\'') continue;
                const val_start = q_pos + 1;
                const val_end_rel = std.mem.indexOfScalar(u8, effective_text[val_start..], quote) orelse break;
                attr_name = effective_text[val_start .. val_start + val_end_rel];
                i = val_start + val_end_rel;
            }
            if (i >= limit or effective_text[i] != '>') {
                // Unclosed opening tag — skip past `<tool` and keep going.
                search_pos = after_tool;
                continue;
            }
            content_start = i + 1;
        }
        // Find the close marker. Prefer the close matching the OPENING tag's
        // name (`<tool_calls>` closes at `</tool_calls>`, never at a
        // `</tool_name>` child element inside the body — the XML-element
        // form puts `</tool…>`-shaped children first). When no exact match
        // exists, fall back to the earliest `</tool…>` with ANY suffix of
        // word characters: some models hallucinate closes freely (`</tool_action>`
        // was captured live closing a `<tool_call>` open), and pinning an
        // exact-name list drops the whole call and leaks it as visible text.
        var close_rel: ?usize = null;
        var close_len: usize = 0;
        {
            const hay = effective_text[content_start..];
            // Opening tag name: "tool" + suffix up to the attr/`>` delimiter.
            var name_end = after_tool;
            while (name_end < effective_text.len and
                (std.ascii.isAlphanumeric(effective_text[name_end]) or effective_text[name_end] == '_')) : (name_end += 1)
            {}
            const open_name = effective_text[tag_origin + 1 .. name_end];
            var exact_buf: [40]u8 = undefined;
            if (open_name.len + 3 <= exact_buf.len) {
                const exact = std.fmt.bufPrint(&exact_buf, "</{s}>", .{open_name}) catch unreachable;
                var from: usize = 0;
                while (std.mem.indexOfPos(u8, hay, from, exact)) |found| {
                    // A `<parameter=…>` value may spell the wrapper's own close.
                    if (hermesParamSpanEnclosing(hay, found)) |past| {
                        from = past;
                        continue;
                    }
                    close_rel = found;
                    close_len = exact.len;
                    break;
                }
            }
            if (close_rel == null) {
                var cpos: usize = 0;
                while (std.mem.indexOf(u8, hay[cpos..], "</tool")) |found| {
                    const open_at = cpos + found;
                    var j = open_at + "</tool".len;
                    const jlimit = @min(hay.len, j + 24);
                    var word_only = true;
                    while (j < jlimit and hay[j] != '>') : (j += 1) {
                        const c = hay[j];
                        if (!std.ascii.isAlphanumeric(c) and c != '_') {
                            word_only = false;
                            break;
                        }
                    }
                    if (word_only and j < jlimit and hay[j] == '>') {
                        if (hermesParamSpanEnclosing(hay, open_at)) |past| {
                            cpos = past;
                            continue;
                        }
                        close_rel = open_at;
                        close_len = j + 1 - open_at;
                        break;
                    }
                    cpos = open_at + "</tool".len;
                }
            }
        }
        if (close_rel == null) {
            // No `</tool…>` close in sight. Two ways this happens in the
            // wild: (1) the model emitted EOS mid-tool-call (`<tool_call>\n{
            // …well-formed args…}` then nothing); (2) max_tokens truncated.
            // The args object itself is usually intact — snap a balanced
            // JSON object from right after the open tag and try to parse it.
            // Function-tag form first: a parameter VALUE is arbitrary bytes and never names the tool.
            const truncated_fn_tag = std.mem.indexOf(u8, effective_text[content_start..], "<function=") != null;
            const truncated_json: ?[]const u8 = if (truncated_fn_tag) null else balancedJsonObject(effective_text[content_start..]);
            if (truncated_json) |json_body| {
                const json_off = @intFromPtr(json_body.ptr) - @intFromPtr(effective_text[content_start..].ptr);
                const advance_to = content_start + json_off + json_body.len;
                // Hermes shape (top-level "name" + "arguments"): use as-is.
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                    search_pos = advance_to;
                    continue;
                }
                // Attribute shape — body is JUST the args, take name from attr.
                if (attr_name) |an| {
                    const name_owned = try allocator.dupe(u8, an);
                    errdefer allocator.free(name_owned);
                    const args_owned = try allocator.dupe(u8, json_body);
                    try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                    search_pos = advance_to;
                    continue;
                }
            }
            // Truncated Hermes/XML tool call: the model emitted
            // `<tool_call><function=NAME><parameter=KEY>…` (Hermes function-tag)
            // or the XML-element `<tool_name>NAME</tool_name>…` shape and ran
            // out of tokens before ANY closing tag — so there's no balanced
            // JSON to snap. The OPENING tags still carry the tool NAME; recover
            // that (parseHermesToolCall breaks out of its parameter loop on a
            // missing `</parameter>`, yielding name + `{}` args) so the call is
            // recognized as a truncated writeFile instead of being DROPPED and
            // leaked into visible content (live JFK-novel capture, 2026-06-20:
            // a 19k-char writeFile cut off mid-content was silently lost and
            // the app fired the wrong "malformed tag" ghost nudge). Recovering
            // the name is enough for the client to fire the right chunk/append
            // nudge; we do NOT salvage the partial content (a half-written file
            // is worse than a re-issued chunked write). The text is truncated
            // to its end here, so advance past it to terminate the scan.
            if (parseHermesToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            if (parseXmlElementToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            // JSON dialect cut off INSIDE the object (EOS mid-string on a
            // large edit): nothing balances, but a top-level `"name"` key is
            // still the tool. NAME + `{}`, never a fragment.
            if (truncatedJsonCallName(effective_text[content_start..])) |name| {
                const name_owned = try allocator.dupe(u8, name);
                errdefer allocator.free(name_owned);
                const args_owned = try allocator.dupe(u8, "{}");
                try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                search_pos = effective_text.len;
                continue;
            }
            break;
        }
        const content = std.mem.trim(u8, effective_text[content_start .. content_start + close_rel.?], " \t\n\r");

        // Pre-clean the body before the parse attempts. Two model quirks:
        //   • The args object may be wrapped in `<parameters>…</parameters>`.
        //   • The model sometimes emits an extra trailing `}` after the
        //     proper JSON close. Find the first balanced JSON object via
        //     depth tracking and use just that.
        const unwrapped = stripParametersWrapper(content);
        // Some models (often when echoing a Jinja `{{ }}` example) wrap the args
        // object in an extra brace layer: `<tool_call>{{"name":…}}</tool_call>`.
        // Strip one layer so the body parses as JSON. Only fires on the `{{…}}`
        // shape (which otherwise fails to parse), so valid single-brace bodies are
        // untouched. Mirrors the unwrap in `parseGemma4ToolCall`.
        const balanced: []const u8 = unwrapDoubleBraces(balancedJsonObject(unwrapped) orelse unwrapped);

        // Try to extract a tool call from the body. Three shapes, tried in
        // priority order:
        //   1. Attribute form (`name="X"` captured, body is JUST args).
        //   2. Canonical Hermes JSON ({"name":"X","arguments":{...}}).
        //   3. Hermes function-tag format (parseHermesToolCall).
        var parsed_ok = false;
        // 0. Function-tag form FIRST when the body actually carries a
        // `<function=` opener. The dialect a hermes-style model's own template
        // mandates rides inside the SAME `<tool_call>` wrapper the JSON form
        // uses, and a `<parameter=…>` VALUE is arbitrary bytes — a written
        // package.json made `balancedJsonObject` snap the FILE CONTENT, whose
        // "name" key then became the tool name (live 2026-08-12: pi answered
        // "Tool voxel-pagoda-garden not found" and the model looped on it).
        // The JSON shapes below can never contain that opener, so this only
        // reorders the case that was being read wrong.
        const repaired_opener = try hermesSpecialOpenerRepair(allocator, content);
        defer if (repaired_opener) |r| allocator.free(r);
        const body = repaired_opener orelse content;
        if (std.mem.indexOf(u8, body, "<function=") != null) {
            if (parseHermesToolCall(allocator, body)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (attr_name) |an| {
                if (std.json.parseFromSlice(std.json.Value, allocator, balanced, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        const name_owned = try allocator.dupe(u8, an);
                        errdefer allocator.free(name_owned);
                        const args_owned = try allocator.dupe(u8, balanced);
                        try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                        parsed_ok = true;
                    }
                } else |_| {}
            }
        }
        if (!parsed_ok) {
            if (tryParseJsonToolCall(allocator, balanced)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseXmlElementToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }

        if (parsed_ok) {
            // Body consumed cleanly — advance past the close marker.
            search_pos = content_start + close_rel.? + close_len;
        } else {
            // Body wasn't a tool call. Advance only past the opening `>`
            // so inner `<tool…>` blocks (if any) still get parsed. This is
            // what makes the outer-wrapper case work: the outer
            // `<tool_calls>` has no JSON body, but its content contains
            // real `<tool name="X">` inner calls.
            search_pos = content_start;
        }
    }

    // Gemma 4 format: <|tool_call>call:name{args}<tool_call|>
    if (calls.items.len == 0) {
        search_pos = 0;
        while (search_pos < effective_text.len) {
            const tag_start = std.mem.indexOf(u8, effective_text[search_pos..], "<|tool_call>") orelse break;
            const content_start = search_pos + tag_start + "<|tool_call>".len;
            const tag_end_opt = std.mem.indexOf(u8, effective_text[content_start..], "<tool_call|>");
            const content = if (tag_end_opt) |tag_end|
                std.mem.trim(u8, effective_text[content_start .. content_start + tag_end], " \t\n\r")
            else
                // Incomplete tool call (model hit EOS before closing tag) — use rest of text
                std.mem.trim(u8, effective_text[content_start..], " \t\n\r");

            search_pos = if (tag_end_opt) |tag_end|
                content_start + tag_end + "<tool_call|>".len
            else
                effective_text.len;

            // No <tool_call|> close = the generation was CUT mid-call (EOS,
            // max_tokens, or the degenerate-tail-loop guard) — fragment values
            // are dropped inside the parse, only completed args ship.
            if (parseGemma4ToolCall(allocator, content, tag_end_opt == null)) |tc| {
                try calls.append(allocator, tc);
            } else {
                log.info("  [tool-parse] Gemma4 parse FAILED for: {s}\n", .{content[0..@min(content.len, 200)]});
            }
        }
    }

    // Bare Hermes function tag with NO <tool_call> wrapper. The outer scan
    // triggers on the substring `<tool`, which a lone `<function=…></function>`
    // (even with a trailing `</tool_call>`, since that is `</too…`, not `<tool`)
    // never provides — so a model that drops the OPENING <tool_call> would leak
    // the whole call as visible text. Only fire when a `<function=` opener AND a
    // `<parameter=` are both present, so prose that merely mentions the tag can't
    // false-fire (parseHermesToolCall needs a real parameter to recover args).
    if (calls.items.len == 0) {
        if (std.mem.indexOf(u8, effective_text, "<function=") != null and
            std.mem.indexOf(u8, effective_text, "<parameter=") != null)
        {
            if (parseHermesToolCall(allocator, effective_text)) |tc| {
                try calls.append(allocator, tc);
            }
        }
    }

    // Anthropic's `<invoke>` is another family's dialect, so it runs after every native one:
    // inside one of their parameter values it is payload, never the call.
    if (calls.items.len == 0) try parseInvokeToolCalls(allocator, effective_text, &calls, BARE_INVOKE);

    // If no <tool_call> tags, try to find raw JSON tool call(s).
    if (calls.items.len == 0) {
        var trimmed = std.mem.trim(u8, effective_text, " \t\n\r");
        if (std.mem.startsWith(u8, trimmed, "</tool_call>")) {
            trimmed = std.mem.trim(u8, trimmed["</tool_call>".len..], " \t\n\r");
        }
        // JSON ARRAY of {name, arguments} objects — parallel tool calls from
        // models without a trained tool format. Such a model emits a ```json fence
        // around the array; skip one leading fence line before the check so
        // the array (not the first object inside it) is what we parse —
        // otherwise only the first call survives and the rest silently drop.
        var array_probe = trimmed;
        if (std.mem.startsWith(u8, array_probe, "```")) {
            if (std.mem.indexOfScalar(u8, array_probe, '\n')) |nl| {
                array_probe = std.mem.trim(u8, array_probe[nl + 1 ..], " \t\n\r");
            }
        }
        if (array_probe.len > 0 and array_probe[0] == '[') {
            if (balancedJsonArray(array_probe)) |arr_body| {
                try appendJsonToolCallArray(allocator, arr_body, &calls);
            }
        }
        if (calls.items.len == 0) {
            if (std.mem.indexOf(u8, trimmed, "{")) |brace_pos| {
                const json_start = trimmed[brace_pos..];
                // Snap the first balanced object so trailing garbage can't poison
                // the parse: models without a trained tool format emit
                // the call inside a ```json fence, leaving "\n```" after the `}`.
                const json_body = balancedJsonObject(json_start) orelse json_start;
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                }
            }
        }
        // Everything appended in this block was inferred from BARE JSON — no
        // tag syntax anywhere. Mark provenance so the request chokepoint
        // (server.parseToolCallsForRequest → filterInferredBySchema) can
        // validate the name against the declared tools: the first balanced
        // {"name": …} object in truncated prose/code is routinely a DATA
        // record, not a call (the George Washington class).
        for (calls.items) |*tc| tc.inferred = true;
    }

    if (calls.items.len == 0) return null;

    // Final safety net: EVERY emitted call carries valid-JSON arguments, whatever
    // converter built them. The direct-construction converters (Gemma custom
    // format, Hermes) can still emit invalid JSON on a pathological value — e.g. a
    // JSON-style string with a bad escape (`\q`) copied verbatim. std.json is the
    // arbiter: if the args don't parse, try the tolerant re-serialize (which
    // re-escapes lone backslashes / control bytes / inner quotes), and if THAT
    // still fails, fall back to `{}` — a client can retry a named call with empty
    // args, but invalid JSON it cannot parse at all. This makes the "args are
    // always valid JSON" invariant structural, not per-converter.
    for (calls.items) |*tc| {
        if (jsonIsValidObject(allocator, tc.arguments)) continue;
        if (looseRepairToolCallJson(allocator, tc.arguments)) |repaired| {
            if (jsonIsValidObject(allocator, repaired)) {
                allocator.free(tc.arguments);
                tc.arguments = repaired;
                continue;
            }
            allocator.free(repaired);
        }
        log.info("  [tool-parse] unrepairable args for '{s}' → empty object (never ship invalid JSON)\n", .{tc.name});
        const empty = try allocator.dupe(u8, "{}");
        allocator.free(tc.arguments);
        tc.arguments = empty;
    }

    return try calls.toOwnedSlice(allocator);
}

/// True when `s` strict-parses as a JSON object. Cheap arbiter for the
/// parseToolCalls safety net.
fn jsonIsValidObject(allocator: std.mem.Allocator, s: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, s, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
    /// True when the call was HEURISTICALLY inferred from bare JSON in the
    /// output (no tag syntax — the model never said "this is a tool call").
    /// The request chokepoint validates inferred names against the declared
    /// tools schema: a truncated DATA object like {"name": "George
    /// Washington", "num": 1, …} must never become a tool call (live pi
    /// capture 2026-07-13). Explicit tag-format calls keep flowing even with
    /// unknown names — "tool not found" is feedback the model corrects from.
    inferred: bool = false,
};

/// Locate a tool's `properties` map in an OpenAI-shaped tools array. Accepts
/// both the wrapped (`{"type":"function","function":{…}}`) and flat forms.
/// The tool's JSON-Schema `parameters` object (Anthropic `input_schema` too).
/// Callers that need only the property map use `toolPropertiesFor`; the hoist
/// also needs `required`, which lives beside `properties` here.
fn toolParametersFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    if (tools != .array) return null;
    for (tools.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv != .string or !std.mem.eql(u8, nv.string, name)) continue;
        const params = func.get("parameters") orelse func.get("input_schema") orelse return null;
        if (params != .object) return null;
        return params.object;
    }
    return null;
}

fn toolPropertiesFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    const params = toolParametersFor(tools, name) orelse return null;
    const props = params.get("properties") orelse return null;
    if (props != .object) return null;
    return props.object;
}

/// The types a misplaced value may be hoisted on. A required OBJECT/ARRAY param
/// found inside another container is too speculative to move — only scalars.
fn isScalarJsonType(want: []const u8) bool {
    return std.mem.eql(u8, want, "string") or std.mem.eql(u8, want, "integer") or
        std.mem.eql(u8, want, "number") or std.mem.eql(u8, want, "boolean");
}

/// The JSON type a property declares. `"type"` may be a union array
/// (`["string","null"]`) — the first non-null entry wins. Absent/odd → null,
/// which means "leave the value alone". Without a `"type"`, a `oneOf`/`anyOf`
/// union declares one only when every non-null branch names the same.
fn declaredJsonType(prop: std.json.Value) ?[]const u8 {
    if (prop != .object) return null;
    const t = prop.object.get("type") orelse return unionJsonType(prop.object);
    switch (t) {
        .string => |s| return s,
        .array => |arr| {
            for (arr.items) |item| {
                if (item != .string) continue;
                if (std.mem.eql(u8, item.string, "null")) continue;
                return item.string;
            }
            return null;
        },
        else => return null,
    }
}

fn unionJsonType(prop: std.json.ObjectMap) ?[]const u8 {
    const branches = (prop.get("oneOf") orelse prop.get("anyOf") orelse return null);
    if (branches != .array) return null;
    var agreed: ?[]const u8 = null;
    for (branches.array.items) |b| {
        const bt = declaredJsonType(b) orelse return null;
        if (std.mem.eql(u8, bt, "null")) continue;
        if (agreed) |a| {
            if (!std.mem.eql(u8, a, bt)) return null;
        } else agreed = bt;
    }
    return agreed;
}

/// Tolerant boolean spelling — the union of what weak models actually emit:
/// Python's `False`, shell's `0`/`1`, prose `yes`/`no`, plus the trailing-comma
/// weld the app's `appendFlagIsTrue` already tolerates. Anything else → null
/// (never guess; an honest client validation error beats a wrong value).
fn looseBoolSpelling(raw: []const u8) ?bool {
    const s = std.mem.trim(u8, raw, " \t\r\n,");
    if (s.len == 0 or s.len > 5) return null;
    var buf: [5]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..s.len], s);
    if (std.mem.eql(u8, lower, "true") or std.mem.eql(u8, lower, "yes") or
        std.mem.eql(u8, lower, "on") or std.mem.eql(u8, lower, "1")) return true;
    if (std.mem.eql(u8, lower, "false") or std.mem.eql(u8, lower, "no") or
        std.mem.eql(u8, lower, "off") or std.mem.eql(u8, lower, "0")) return false;
    return null;
}

/// Strict parse of a container string that admits a key repeated with the SAME
/// value (issue #402: a model spelling `header` twice per item) and refuses a
/// key repeated with different values: the two are told apart by parsing
/// first-wins and last-wins and comparing the documents.
fn parseContainerAllowingRepeats(allocator: std.mem.Allocator, text: []const u8) ?std.json.Parsed(std.json.Value) {
    var first = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_first }) catch return null;
    var last = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_last }) catch {
        first.deinit();
        return null;
    };
    defer last.deinit();
    const a = std.json.Stringify.valueAlloc(allocator, first.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(a);
    const b = std.json.Stringify.valueAlloc(allocator, last.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(b);
    if (!std.mem.eql(u8, a, b)) {
        first.deinit();
        return null;
    }
    return first;
}

/// Coerce ONE value to the declared type. Returns null when the value already
/// matches or the spelling is undecidable. Allocated replacements are appended
/// to `owned`; parsed sub-documents to `docs` — both are freed by the caller
/// AFTER the object is re-serialized, because the returned Value borrows them.
fn coerceValueToType(
    allocator: std.mem.Allocator,
    v: std.json.Value,
    want: []const u8,
    owned: *std.ArrayList([]const u8),
    docs: *std.ArrayList(std.json.Parsed(std.json.Value)),
) !?std.json.Value {
    // A CONTAINER param carried through a tag format arrives as a STRING — no tag
    // format types its values. Live (15 hits in one session): pi's `edit.edits`,
    // an array of {oldText,newText}, shipped as `"[{\"oldText\":…}]"` on every
    // call. Strict parse only: a value that isn't well-formed JSON of the
    // DECLARED kind is left alone rather than guessed at.
    if (std.mem.eql(u8, want, "array") or std.mem.eql(u8, want, "object")) {
        if (v != .string) return null;
        const t = std.mem.trim(u8, v.string, " \t\r\n");
        const want_array = want[0] == 'a';
        if (t.len == 0 or t[0] != (if (want_array) @as(u8, '[') else @as(u8, '{'))) return null;

        // Strict first; a well-formed container is the common case.
        if (parseContainerAllowingRepeats(allocator, t)) |doc| {
            const kind_ok = if (want_array) doc.value == .array else doc.value == .object;
            if (kind_ok) {
                try docs.append(allocator, doc);
                return doc.value;
            }
            var d = doc;
            d.deinit();
            return null;
        }

        // The container string is itself MANGLED (live: pi's `edits` array with a
        // missing comma between two object values). Same big-file escaping class
        // as looseRepairToolCallJson — re-serialize tolerantly, then STRICT-parse
        // the result so a mis-repair that yields invalid JSON is discarded.
        const repaired = looseRepairContainer(allocator, t) orelse return null;
        defer allocator.free(repaired);
        var rdoc = parseContainerAllowingRepeats(allocator, repaired) orelse {
            log.debug("[tool-autocorrect] {s} param left as a string: not well-formed JSON of that kind\n", .{want});
            return null;
        };
        const rkind_ok = if (want_array) rdoc.value == .array else rdoc.value == .object;
        if (!rkind_ok) {
            rdoc.deinit();
            return null;
        }
        try docs.append(allocator, rdoc); // ownership transfers; never deinit here
        return rdoc.value;
    }
    if (std.mem.eql(u8, want, "boolean")) {
        return switch (v) {
            .bool => null,
            .string => |s| if (looseBoolSpelling(s)) |b| std.json.Value{ .bool = b } else null,
            .integer => |i| if (i == 0 or i == 1) std.json.Value{ .bool = i == 1 } else null,
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "integer")) {
        return switch (v) {
            .string => |s| blk: {
                const n = std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\r\n"), 10) catch break :blk null;
                break :blk std.json.Value{ .integer = n };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "number")) {
        return switch (v) {
            .string => |s| blk: {
                const t = std.mem.trim(u8, s, " \t\r\n");
                if (std.fmt.parseInt(i64, t, 10)) |n| break :blk std.json.Value{ .integer = n } else |_| {}
                const f = std.fmt.parseFloat(f64, t) catch break :blk null;
                if (std.math.isNan(f) or std.math.isInf(f)) break :blk null; // not representable in JSON
                break :blk std.json.Value{ .float = f };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "string")) {
        // The inverse repair: the tag/custom formats infer type from the value's
        // SPELLING (isJsonLiteral), so a string param whose content happens to
        // read as `false`/`42` was shipped as a bool/number. The schema is the
        // only thing that can tell those apart.
        return switch (v) {
            .bool => |b| std.json.Value{ .string = if (b) "true" else "false" },
            .integer => |i| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{i});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .float => |f| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .number_string => |s| std.json.Value{ .string = s },
            else => null,
        };
    }
    return null; // a type we don't model — never guessed
}

fn jsonTypeMatches(v: std.json.Value, want: []const u8) bool {
    if (v == .null) return true; // an explicit null satisfies any optional field
    if (std.mem.eql(u8, want, "boolean")) return v == .bool;
    if (std.mem.eql(u8, want, "integer")) return v == .integer or v == .number_string;
    if (std.mem.eql(u8, want, "number")) return v == .integer or v == .float or v == .number_string;
    if (std.mem.eql(u8, want, "string")) return v == .string;
    if (std.mem.eql(u8, want, "array")) return v == .array;
    if (std.mem.eql(u8, want, "object")) return v == .object;
    return true; // a type we don't model constrains nothing
}

/// True when every argument whose type the tool DECLARES actually carries that
/// JSON type. This is the invariant strict clients (Claude Code, pi, opencode)
/// enforce and reject on; the format corpus asserts it universally so a new
/// entry in any family is covered without a bespoke assertion. Undeclared keys
/// and unmodelled types constrain nothing.
pub fn toolCallConformsToSchema(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer tools.deinit();
    const props = toolPropertiesFor(tools.value, call.name) orelse return true;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return true;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const prop = props.get(entry.key_ptr.*) orelse continue;
        const want = declaredJsonType(prop) orelse continue;
        if (!jsonTypeMatches(entry.value_ptr.*, want)) return false;
    }
    return true;
}

/// True when a REQUIRED scalar param is MISSING at the top level while sitting
/// buried inside one of the tool's declared container args — exactly the shape
/// `hoistMisplacedRequiredParams` exists to repair, and exactly what a strict
/// client rejects with "must have required properties X". The format corpus
/// asserts this is FALSE for every entry that declares tools, so a future family
/// that produces the buried shape is covered without a bespoke assertion.
pub fn requiredParamIsBuried(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return false;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return false;
    const props_v = params.get("properties") orelse return false;
    const required_v = params.get("required") orelse return false;
    if (props_v != .object or required_v != .array) return false;
    const props = props_v.object;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;

    for (required_v.array.items) |req_v| {
        if (req_v != .string) continue;
        const name = req_v.string;
        if (parsed.value.object.get(name) != null) continue;
        const want = declaredJsonType(props.get(name) orelse continue) orelse continue;
        if (!isScalarJsonType(want)) continue;

        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const prop = props.get(e.key_ptr.*) orelse continue;
            if (containerItemDeclares(prop, name)) continue;
            if (buriedParamIn(e.value_ptr, name, want, false) != null) return true;
        }
    }
    return false;
}

/// Coerce every parsed call's `arguments` to the types its tool DECLARES.
///
/// Class: **type inference from value spelling.** Neither the tag formats
/// (`<parameter=x>False</parameter>`, Gemma's `key:false`) nor a model writing
/// JSON by hand carries type information a parser can trust — `isJsonLiteral`
/// has to guess from the bytes, and it guesses wrong in both directions:
///   • `False` / `"false"` on a boolean param → shipped as the STRING "false";
///   • `false` / `42` as a string param's CONTENT → shipped as a bool/number.
/// Strict clients (Claude Code, pi, opencode) reject both and the model, which
/// cannot see its own serialized request, loops forever "fixing" a value that
/// was already correct. The tool schema is the only disambiguator, and it is
/// already threaded to every parse site as `tools_json`.
///
/// Contract: only where the schema declares a type, and only when the spelling
/// is unambiguous. An undecidable value is left alone so the client's
/// validation error stays honest. Correct calls from strong models are
/// byte-unchanged (nothing re-serializes unless a coercion actually fired).
///
/// Scalars AND containers: a tag format types nothing, so a declared
/// `array`/`object` arrives as a string of JSON and `coerceValueToType` parses
/// it back (strict, then one tolerant repair re-validated strictly). This
/// comment used to say "only SCALARS are touched" — it was two weeks stale by
/// the time it was read as a spec and filed as issue #94. A contract comment is
/// load-bearing documentation: pin it with a test (`issue #94: a container
/// param carried as literal TEXT…`), don't let it drift.
pub fn coerceToolArgsToSchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const props = toolPropertiesFor(tools.value, call.name) orelse continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var owned = std.ArrayList([]const u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }
        // Parsed container sub-documents: the rewritten object BORROWS from these,
        // so they must outlive the Stringify below.
        var docs = std.ArrayList(std.json.Parsed(std.json.Value)).empty;
        defer {
            for (docs.items) |*d| d.deinit();
            docs.deinit(allocator);
        }

        var changed = false;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const prop = props.get(entry.key_ptr.*) orelse continue;
            const want = declaredJsonType(prop) orelse continue;
            const new_val = try coerceValueToType(allocator, entry.value_ptr.*, want, &owned, &docs) orelse continue;
            entry.value_ptr.* = new_val;
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        log.info("  [tool-parse] coerced {s} arguments to the declared schema types\n", .{call.name});
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does this container's ITEM schema declare `name` as one of its own
/// properties? If it does, a `name` sitting inside it BELONGS there and must
/// never be lifted out — a multi-file edit tool would lose its per-item paths.
/// This is the whole safety argument for the hoist below, so it is deliberately
/// conservative: no declared item `properties` map ⇒ nothing is proven ⇒ false.
fn containerItemDeclares(prop: std.json.Value, name: []const u8) bool {
    if (prop != .object) return false;
    // Array property → the item schema. Object property → the schema itself.
    const inner = if (prop.object.get("items")) |items| items else prop;
    if (inner != .object) return false;
    const props = inner.object.get("properties") orelse return false;
    if (props != .object) return false;
    return props.object.get(name) != null;
}

/// Scalar equality — the only shapes we ever hoist on. A container value is
/// never a candidate, so a structural compare is not needed.
fn jsonScalarEql(a: std.json.Value, b: std.json.Value) bool {
    return switch (a) {
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .integer => |i| b == .integer and i == b.integer,
        .bool => |x| b == .bool and x == b.bool,
        else => false,
    };
}

/// Find `name` buried one level down inside `container`, requiring UNANIMITY:
/// every object in the container carries `name`, all with the same scalar value,
/// and that value matches the type the schema declares for the top-level param.
/// Anything less is ambiguous → null. With `remove`, also deletes it from each
/// object (the returned value keeps pointing into the parse arena, which
/// outlives the removal).
fn buriedParamIn(
    container: *std.json.Value,
    name: []const u8,
    want: []const u8,
    remove: bool,
) ?std.json.Value {
    switch (container.*) {
        .array => |*arr| {
            if (arr.items.len == 0) return null;
            var found: ?std.json.Value = null;
            for (arr.items) |*item| {
                if (item.* != .object) return null;
                const v = item.object.get(name) orelse return null; // must be in EVERY item
                if (v == .null or !jsonTypeMatches(v, want)) return null;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) return null; // items disagree → ambiguous
                } else found = v;
            }
            if (remove) for (arr.items) |*item| {
                _ = item.object.orderedRemove(name);
            };
            return found;
        },
        .object => |*obj| {
            const v = obj.get(name) orelse return null;
            if (v == .null or !jsonTypeMatches(v, want)) return null;
            if (remove) _ = obj.orderedRemove(name);
            return v;
        },
        else => return null,
    }
}

/// Lift a REQUIRED top-level parameter the model BURIED inside a container arg.
///
/// Class: **misplaced required param.** A weak model that has internalized "the
/// edit object holds everything about the edit" writes the file path INSIDE each
/// `edits[]` item and never emits it at the top level:
///
///     {"edits":[{"newText":"…","oldText":"…","path":"us_presidents/generate_site.sh"}]}
///
/// The call is valid JSON and correctly escaped, so no repair path fires and no
/// type is wrong — it is purely in the wrong PLACE, which only the schema can
/// know. Strict clients reject it ("path: must have required properties path"),
/// and the model, which cannot see its own serialized request, re-emits the same
/// call until it gives up and falls back to rewriting the whole file with
/// `write` — three dead multi-thousand-token rounds in the live 2026-07-13 pi
/// session (gemma-4-26B-A4B). Same shape as the Claude Code replace_all loop.
///
/// Contract — every condition is read off the SCHEMA, never guessed:
///   • the param is declared REQUIRED at the top level and is ABSENT there;
///   • it is declared a SCALAR (hoisting a container is too speculative);
///   • it sits inside a DECLARED container arg whose item schema does NOT
///     declare it (`containerItemDeclares` — otherwise it belongs where it is);
///   • every object in that container carries it with the SAME value and the
///     declared type.
/// Ambiguity of any kind (items disagreeing, two containers offering different
/// values, a wrong-typed value) leaves the call untouched so the client's
/// validation error stays honest. A compliant call never re-serializes.
pub fn hoistMisplacedRequiredParams(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props_v = params.get("properties") orelse continue;
        const required_v = params.get("required") orelse continue;
        if (props_v != .object or required_v != .array) continue;
        const props = props_v.object;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var changed = false;
        for (required_v.array.items) |req_v| {
            if (req_v != .string) continue;
            const name = req_v.string; // borrows from `tools`, which outlives the Stringify
            if (parsed.value.object.get(name) != null) continue; // present — nothing misplaced

            const want = declaredJsonType(props.get(name) orelse continue) orelse continue;
            if (!isScalarJsonType(want)) continue;

            // Probe every declared container WITHOUT mutating: a value found in
            // two containers that disagree is ambiguous, and we must not have
            // already gutted the first one by then.
            var found: ?std.json.Value = null;
            var ambiguous = false;
            var it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                const v = buriedParamIn(e.value_ptr, name, want, false) orelse continue;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) {
                        ambiguous = true;
                        break;
                    }
                } else found = v;
            }
            if (ambiguous or found == null) continue;

            // Commit: strip it from the containers, install it at the top level.
            it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                _ = buriedParamIn(e.value_ptr, name, want, true);
            }
            try parsed.value.object.put(parsed.arena.allocator(), name, found.?);
            log.info("  [tool-parse] hoisted misplaced required arg '{s}' to the top level for {s}\n", .{ name, call.name });
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does the OpenAI-shaped tools array declare a function with this name?
/// Accepts both the wrapped ({"type":"function","function":{…}}) and flat
/// forms, mirroring `toolPropertiesFor`. An unparseable/odd `tools_json`
/// returns TRUE — never drop a call because OUR schema parse failed.
pub fn toolNameIsDeclared(allocator: std.mem.Allocator, tools_json: []const u8, name: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .array) return true;
    for (parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv == .string and std.mem.eql(u8, nv.string, name)) return true;
    }
    return false;
}

/// Drop HEURISTICALLY-inferred raw-JSON calls (`ParsedToolCall.inferred`)
/// whose name the request never declared. The George Washington class: a
/// generation truncated mid-data-script leaves `{"name": "George Washington",
/// "num": 1, …}` as the first balanced object, and the flat-shape synthesis
/// promoted it to a tool call the client can only answer with "tool not
/// found" — burning the turn. Explicit tag-format calls are NEVER dropped
/// here (undeclared name → honest client feedback the model corrects from).
///
/// Returns the retained slice (shrunk in place), or null when every call was
/// dropped — the caller treats that exactly like "no tool calls", so the text
/// stays visible content and finish_reason (e.g. "length") is preserved.
pub fn filterInferredBySchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !?[]ParsedToolCall {
    var kept: usize = 0;
    for (calls) |tc| {
        if (!tc.inferred or toolNameIsDeclared(allocator, tools_json, tc.name)) {
            calls[kept] = tc;
            kept += 1;
        } else {
            log.warn("  [tool-parse] dropped inferred raw-JSON call '{s}' — name not among the request's tools (data object, not a call)\n", .{tc.name});
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    if (kept == calls.len) return calls;
    if (kept == 0) {
        allocator.free(calls);
        return null;
    }
    return try allocator.realloc(calls, kept);
}

/// Last-resort tool-call inference for models that emit JUST the arguments
/// object — no tool name, no wrapper syntax. Observed live from Gemma 4 12B
/// via Claude Code: ```` ```json\n{"file_path": …, "content": …}\n``` ````
/// with the Write tool defined; parseToolCalls finds nothing (no "name" key)
/// and the un-executed JSON leaked into the visible text.
///
/// Conservative contract — fires ONLY when parseToolCalls returned null:
///   • the visible content (after a leading think block) must START with the
///     JSON object, optionally wrapped in a markdown fence;
///   • the object's keys must satisfy exactly ONE tool in `tools_json`
///     (OpenAI shape): required ⊆ keys AND keys ⊆ properties;
///   • zero or multiple matching tools → null (ambiguity never guesses).
pub fn inferBareJsonToolCalls(allocator: std.mem.Allocator, text: []const u8, tools_json: []const u8) !?[]ParsedToolCall {
    // Strip a leading think block, mirroring parseToolCalls.
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }
    var trimmed = std.mem.trimStart(u8, effective_text, "\n \t");
    // Skip one opening markdown fence line (``` or ```json etc.).
    if (std.mem.startsWith(u8, trimmed, "```")) {
        const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return null;
        trimmed = std.mem.trimStart(u8, trimmed[nl + 1 ..], "\n \t");
    }
    // The object must LEAD the visible content — a JSON example mid-prose is
    // never promoted to a call.
    if (!std.mem.startsWith(u8, trimmed, "{")) return null;
    const obj_slice = balancedJsonObject(trimmed) orelse return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, obj_slice, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    if (obj.count() == 0) return null;

    const tools_parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return null;
    defer tools_parsed.deinit();
    if (tools_parsed.value != .array) return null;

    var match_name: ?[]const u8 = null;
    for (tools_parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func_val = tool_val.object.get("function") orelse continue;
        if (func_val != .object) continue;
        const name_val = func_val.object.get("name") orelse continue;
        if (name_val != .string) continue;
        const params_val = func_val.object.get("parameters") orelse continue;
        if (params_val != .object) continue;
        const props_val = params_val.object.get("properties") orelse continue;
        if (props_val != .object) continue;
        const props = props_val.object;
        if (props.count() == 0) continue;

        // keys ⊆ properties
        var keys_ok = true;
        var it = obj.iterator();
        while (it.next()) |kv| {
            if (props.get(kv.key_ptr.*) == null) {
                keys_ok = false;
                break;
            }
        }
        if (!keys_ok) continue;

        // required ⊆ keys
        if (params_val.object.get("required")) |req_val| {
            if (req_val == .array) {
                var req_ok = true;
                for (req_val.array.items) |r| {
                    if (r != .string) continue;
                    if (obj.get(r.string) == null) {
                        req_ok = false;
                        break;
                    }
                }
                if (!req_ok) continue;
            }
        }

        if (match_name != null) return null; // ambiguous — never guess
        match_name = name_val.string;
    }

    const name = match_name orelse return null;
    const calls = try allocator.alloc(ParsedToolCall, 1);
    errdefer allocator.free(calls);
    const name_owned = try allocator.dupe(u8, name);
    errdefer allocator.free(name_owned);
    calls[0] = .{ .name = name_owned, .arguments = try allocator.dupe(u8, obj_slice) };
    log.info("  [tool-parse] inferred bare-args call to '{s}' via unique schema match\n", .{name});
    return calls;
}

/// Strip a leading `<parameters>` / trailing `</parameters>` pair if both
/// are present. the model occasionally wraps the args object in this XML
/// container — leaving it in place would defeat the JSON parser. Trims
/// surrounding whitespace.
fn stripParametersWrapper(content: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const open = "<parameters>";
    const close = "</parameters>";
    if (trimmed.len < open.len + close.len) return trimmed;
    if (!std.mem.startsWith(u8, trimmed, open)) return trimmed;
    if (!std.mem.endsWith(u8, trimmed, close)) return trimmed;
    const inner = trimmed[open.len .. trimmed.len - close.len];
    return std.mem.trim(u8, inner, " \t\n\r");
}

/// Return the substring of `content` covering the first brace-balanced JSON
/// object, or null if no balanced object is present. Handles quoted strings
/// (so `}` inside a string doesn't close the object) and escape sequences.
/// Used to tolerate the model `{"…"}}` extra-trailing-`}` artifact.
/// Try to parse a tool tag whose body lives entirely in XML-ish
/// attributes. Two terminator shapes are accepted:
///
///   Self-closing /> :
///     <tool name="shell" arguments="{...}"/>
///     <tool name="X" arguments='{...}' />
///     <tool name="X" arguments="{...}'/>             (broken quote)
///
///   Empty-body explicit close ></tool*> :
///     <tool_call name="cwd" arguments="{...}"></tool_call>
///     <tool name="X" arguments="{...}"></tool_calls>  (mismatched close OK)
///
/// `slice` MUST start with `<tool` (the caller has already validated the
/// terminator char after that prefix). Returns name + arguments slices into
/// `slice` plus the number of bytes consumed (up to and including the final
/// `>` or `</tool*>` close), or null if this isn't a shape we can parse.
///
/// We never trust the XML quoting around `arguments=` — the model frequently
/// uses `"` to open and `'` to close, or doesn't escape inner `"` at all.
/// Instead we expect a `{` immediately (optionally after a single opening
/// quote) and snap a balanced JSON object via `balancedJsonObject`. That
/// strategy is what makes the XML fragment parse cleanly even though the
/// surrounding XML is malformed.
fn parseSelfClosingToolTag(slice: []const u8) ?struct { name: []const u8, arguments: []const u8, consumed: usize } {
    if (slice.len < "<tool".len) return null;
    if (!std.mem.startsWith(u8, slice, "<tool")) return null;
    // Bound the scan window so a runaway tag can't pin us forever.
    const limit = @min(slice.len, 8192);

    // 1. Locate `name="X"` (or `'X'`) attribute. We don't enforce ordering
    //    against `arguments=`: some models interleave them.
    var name: ?[]const u8 = null;
    var name_end: usize = "<tool".len;
    {
        var i: usize = "<tool".len;
        while (i + "name=".len < limit) : (i += 1) {
            if (slice[i] != 'n') continue;
            if (!std.mem.eql(u8, slice[i .. i + "name=".len], "name=")) continue;
            const q_pos = i + "name=".len;
            if (q_pos >= limit) break;
            const quote = slice[q_pos];
            if (quote != '"' and quote != '\'') continue;
            const val_start = q_pos + 1;
            const val_end_rel = std.mem.indexOfScalar(u8, slice[val_start..limit], quote) orelse break;
            name = slice[val_start .. val_start + val_end_rel];
            name_end = val_start + val_end_rel + 1;
            break;
        }
    }

    // 2. Locate `arguments=` and snap the next balanced JSON object as
    //    the value. We deliberately don't parse the XML quote — see
    //    function docs.
    var args: ?[]const u8 = null;
    var args_end: usize = 0;
    {
        var i: usize = "<tool".len;
        while (i + "arguments=".len < limit) : (i += 1) {
            if (slice[i] != 'a') continue;
            if (!std.mem.eql(u8, slice[i .. i + "arguments=".len], "arguments=")) continue;
            var j: usize = i + "arguments=".len;
            // Skip an opening quote if present.
            if (j < limit and (slice[j] == '"' or slice[j] == '\'')) j += 1;
            // Skip whitespace between the quote and the JSON object.
            while (j < limit and (slice[j] == ' ' or slice[j] == '\t')) j += 1;
            if (j >= limit or slice[j] != '{') break;
            const sub = slice[j..limit];
            const json = balancedJsonObject(sub) orelse break;
            args = json;
            const json_off = @intFromPtr(json.ptr) - @intFromPtr(sub.ptr);
            args_end = j + json_off + json.len;
            break;
        }
    }

    if (name == null or args == null) return null;

    // 3. Scan forward from the end of whichever attribute landed last to
    //    the terminator. Tolerate a stray closing quote (model often closes
    //    args with `'` even though the opener was `"`) and any intervening
    //    whitespace. We accept either:
    //      a) `/>` — self-closing.
    //      b) `>` followed (possibly across whitespace) by `</tool*>` —
    //         empty body, explicit close marker.
    var k: usize = @max(name_end, args_end);
    while (k < limit and (slice[k] == ' ' or slice[k] == '\t' or slice[k] == '\n' or slice[k] == '\r' or slice[k] == '"' or slice[k] == '\'')) k += 1;
    if (k >= limit) return null;

    if (slice[k] == '/') {
        k += 1;
        if (k >= limit or slice[k] != '>') return null;
        return .{ .name = name.?, .arguments = args.?, .consumed = k + 1 };
    }
    if (slice[k] == '>') {
        var w: usize = k + 1;
        while (w < slice.len and (slice[w] == ' ' or slice[w] == '\t' or slice[w] == '\n' or slice[w] == '\r')) w += 1;
        const close_markers = [_][]const u8{
            "</tool_call>",   "</tool_calls>",
            "</tool_request>", "</tool_requests>",
            "</tool>",
        };
        for (close_markers) |m| {
            if (w + m.len <= slice.len and std.mem.eql(u8, slice[w .. w + m.len], m)) {
                return .{ .name = name.?, .arguments = args.?, .consumed = w + m.len };
            }
        }
    }
    return null;
}

/// The value of a depth-1 `"name"` key in an UNBALANCED JSON object (a tool
/// call truncated mid-body). String-aware, so a `"name"` inside a nested
/// value or a string never counts. Null when the text is not an object or
/// carries no complete top-level name.
fn truncatedJsonCallName(content: []const u8) ?[]const u8 {
    const t = std.mem.trimStart(u8, content, " \t\n\r");
    if (!std.mem.startsWith(u8, t, "{")) return null;
    var depth: usize = 0;
    var i: usize = 0;
    var last_str: ?[]const u8 = null; // most recent complete string at depth 1
    var after_name_colon = false;
    while (i < t.len) : (i += 1) {
        const c = t[i];
        if (c == '"') {
            const start = i + 1;
            var j = start;
            while (j < t.len and t[j] != '"') : (j += 1) {
                if (t[j] == '\\') j += 1;
            }
            if (j >= t.len) return null; // unterminated string = the cut
            const str = t[start..j];
            if (depth == 1) {
                if (after_name_colon) return if (str.len > 0) str else null;
                last_str = str;
            }
            i = j;
            continue;
        }
        switch (c) {
            '{', '[' => {
                depth += 1;
                after_name_colon = false;
            },
            '}', ']' => {
                if (depth == 0) return null;
                depth -= 1;
            },
            ':' => if (depth == 1) {
                if (last_str) |k| after_name_colon = std.mem.eql(u8, k, "name");
            },
            ',' => {
                after_name_colon = false;
                if (depth == 1) last_str = null;
            },
            ' ', '\t', '\n', '\r' => {},
            else => after_name_colon = false,
        }
    }
    return null;
}

fn balancedJsonObject(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return trimmed[start .. i + 1];
            if (depth < 0) return null; // mismatched — give up
        }
    }
    return null;
}

/// XML-element tool form: the body of a `<tool_calls>`-style
/// wrapper carries the tool name and each argument as plain child elements —
/// no JSON anywhere (captured live, 2026-06-10):
///     <tool_name>shell</tool_name>
///     <command>df -h / | grep -v "Filesystem"</command>
/// The name comes from a <tool_name>, <name>, or <tool> element; every other
/// simple `<key>text</key>` pair becomes a string argument. Strict on shape:
/// non-whitespace between elements, an unclosed element, a duplicate name
/// element, or a missing name returns null so prose-ish markup can't
/// half-execute.
/// Tag suffixes after `tool_` that mark tool RESULTS or metadata, never tool
/// names. `<tool_output>…</tool_output>` is the model hallucinating the result of
/// a call it never made — mapping it onto a tool named "output" would
/// fabricate a call out of thin air. `name` guards the bare XML-element-form
/// child (`<tool_name>shell</tool_name>`) appearing outside its wrapper.
const xml_tag_reserved_names = [_][]const u8{
    "output", "outputs", "result", "results", "response", "responses", "error", "errors", "name",
};

/// XML-element-TAG tool form: `<tool_NAME><key>value</key>…</tool_NAME>`,
/// the tool name embedded in the tag itself (training-bias family;
/// captured live 2026-06-10: `<tool_read>` / `<tool_edit>`). `slice` starts
/// at `<tool`. Returns the parsed call plus bytes consumed through the close
/// tag, or null when the shape doesn't hold (no attributes allowed, exact
/// close required, reserved names rejected, body must be all elements).
fn parseXmlElementTagToolCall(
    allocator: std.mem.Allocator,
    slice: []const u8,
) !?struct { call: ParsedToolCall, consumed: usize } {
    std.debug.assert(std.mem.startsWith(u8, slice, "<tool"));
    const name_start = "<tool_".len;
    if (slice.len <= name_start) return null;
    var name_end = name_start;
    while (name_end < slice.len and
        (std.ascii.isAlphanumeric(slice[name_end]) or slice[name_end] == '_')) : (name_end += 1)
    {}
    if (name_end == name_start) return null; // bare `<tool_>`
    if (name_end >= slice.len or slice[name_end] != '>') return null; // attributes → not this form
    const tool_name = slice[name_start..name_end];
    for (xml_tag_reserved_names) |reserved| {
        if (std.ascii.eqlIgnoreCase(tool_name, reserved)) return null;
    }
    var close_buf: [72]u8 = undefined;
    if (tool_name.len + "</tool_>".len > close_buf.len) return null;
    const close_tag = std.fmt.bufPrint(&close_buf, "</tool_{s}>", .{tool_name}) catch return null;
    const body_start = name_end + 1;
    const close_rel = std.mem.indexOf(u8, slice[body_start..], close_tag) orelse return null;
    const body = slice[body_start .. body_start + close_rel];
    const args = parseXmlElementArgsJson(allocator, body) orelse blk: {
        // JSON-args body variant of the same form, also captured live:
        // `<tool_write>\n{"path": …, "content": …}\n</tool_write>`. The whole
        // object IS the args. The trimmed body must START with `{` so prose
        // that merely contains braces never produces arguments.
        const trimmed = std.mem.trim(u8, body, " \t\n\r");
        if (trimmed.len == 0 or trimmed[0] != '{') return null;
        const json_body = balancedJsonObject(trimmed) orelse return null;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_body, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        break :blk allocator.dupe(u8, json_body) catch return null;
    };
    const name_owned = allocator.dupe(u8, tool_name) catch {
        allocator.free(args);
        return null;
    };
    return .{
        .call = .{ .name = name_owned, .arguments = args },
        .consumed = body_start + close_rel + close_tag.len,
    };
}

/// Parse a body consisting ENTIRELY of `<key>value</key>` child elements into
/// a JSON args string. Values are kept verbatim, so nested markup (the model's
/// `<edits><oldText>…</oldText>…</edits>`) survives as the arg's string
/// value. Returns null unless every non-whitespace byte belongs to an element
/// and at least one element is present — a plain-text body must never
/// produce arguments.
fn parseXmlElementArgsJson(allocator: std.mem.Allocator, body: []const u8) ?[]u8 {
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);
    var count: usize = 0;
    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        args_map.put(allocator, key, .{ .string = body[val_start .. val_start + rel] }) catch return null;
        count += 1;
        i = val_start + rel + close_tag.len;
    }
    if (count == 0) return null;
    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch null;
}

fn isBareToolName(body: []const u8) bool {
    const name = std.mem.trim(u8, body, " \t\n\r");
    if (name.len == 0) return false;
    for (name) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.')) return false;
    }
    return true;
}

/// Earliest position at/after `from` where any of `needles` occurs, or null.
fn earliestIndexOfAny(text: []const u8, from: usize, needles: []const []const u8) ?usize {
    var best: ?usize = null;
    for (needles) |n| {
        if (std.mem.indexOfPos(u8, text, from, n)) |q| {
            if (best == null or q < best.?) best = q;
        }
    }
    return best;
}

// native DSML markers. `｜` is the FULLWIDTH vertical bar
// (U+FF5C, 3 bytes) — the dsml_token "｜DSML｜" appears in no other family.
// A DSML variant puts a space before every tag name (`<｜DSML｜ invoke`).
const DsmlTag = struct { open: []const u8, name: []const u8 };
const DSML_INVOKE_TAG: DsmlTag = .{ .open = "<｜DSML｜", .name = "invoke" };
const DSML_INVOKE_CLOSE: DsmlTag = .{ .open = "</｜DSML｜", .name = "invoke>" };
const DSML_PARAM_TAG: DsmlTag = .{ .open = "<｜DSML｜", .name = "parameter" };
const DSML_PARAM_CLOSE: DsmlTag = .{ .open = "</｜DSML｜", .name = "parameter>" };

const DsmlHit = struct { at: usize, end: usize };

fn dsmlFind(text: []const u8, from: usize, comptime tag: DsmlTag) ?DsmlHit {
    var p = from;
    while (std.mem.indexOfPos(u8, text, p, tag.open)) |at| {
        var q = at + tag.open.len;
        if (q < text.len and text[q] == ' ') q += 1;
        if (std.mem.startsWith(u8, text[q..], tag.name)) return .{ .at = at, .end = q + tag.name.len };
        p = at + 1;
    }
    return null;
}

/// Tolerant quoted-attribute extraction from a tag-attribute segment:
/// `name="X"` / `name='X'` / mangled unquoted `name=X` (to whitespace/`>`).
fn dsmlAttr(seg: []const u8, comptime key: []const u8) ?[]const u8 {
    const pat = key ++ "=";
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, seg, start, pat)) |at| {
        if (at > 0 and (std.ascii.isAlphanumeric(seg[at - 1]) or seg[at - 1] == '_')) {
            start = at + pat.len;
            continue;
        }
        var v = at + pat.len;
        if (v >= seg.len) return null;
        const q = seg[v];
        if (q == '"' or q == '\'') {
            v += 1;
            const end = std.mem.indexOfScalarPos(u8, seg, v, q) orelse seg.len;
            return seg[v..end];
        }
        const end = std.mem.indexOfAnyPos(u8, seg, v, " \t\n>") orelse seg.len;
        return seg[v..end];
    }
    return null;
}

/// DSML tool calls:
///   `<｜DSML｜tool_calls>` wrapping
///   `<｜DSML｜invoke name="X">` +
///   `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`.
/// string="true" values are raw text (JSON-escaped here); string="false"
/// values are already JSON (validated; invalid degrades to an honest string).
/// House conventions: the parser keys on the INVOKE marker (a dropped
/// wrapper or invoke-close still parses — delimiter-drop class), values run
/// to the CONFIRMED closing tag (rich values with `<`/newlines survive),
/// truncation salvages NAME + completed pairs (fragments dropped), keys are
/// escaped + deduped (last wins) so emitted arguments are ALWAYS valid JSON.
fn parseDsmlToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    const Param = struct { key: []const u8, value: []const u8, is_string: bool };
    var pos: usize = 0;
    while (dsmlFind(text, pos, DSML_INVOKE_TAG)) |inv| {
        var p = inv.end;
        const tag_end = std.mem.indexOfScalarPos(u8, text, p, '>') orelse break;
        const name = dsmlAttr(text[p..tag_end], "name") orelse {
            pos = tag_end + 1;
            continue;
        };
        p = tag_end + 1;

        var params = std.ArrayList(Param).empty;
        defer params.deinit(allocator);
        while (true) {
            const next_param = dsmlFind(text, p, DSML_PARAM_TAG);
            const next_close = dsmlFind(text, p, DSML_INVOKE_CLOSE);
            const next_invoke = dsmlFind(text, p, DSML_INVOKE_TAG);
            const np = if (next_param) |h| h.at else std.math.maxInt(usize);
            const nc = if (next_close) |h| h.at else std.math.maxInt(usize);
            const ni = if (next_invoke) |h| h.at else std.math.maxInt(usize);
            if (np == std.math.maxInt(usize) and nc == std.math.maxInt(usize) and ni == std.math.maxInt(usize)) {
                // truncated invoke: ship NAME + completed pairs
                p = text.len;
                break;
            }
            if (nc <= np and nc <= ni) {
                p = next_close.?.end;
                break;
            }
            if (ni < np) {
                // dropped invoke-close — the next call starts here; the outer
                // loop picks it up from `p`.
                p = ni;
                break;
            }
            // parameter
            var q = next_param.?.end;
            const ptag_end = std.mem.indexOfScalarPos(u8, text, q, '>') orelse {
                p = text.len;
                break;
            };
            const attrs = text[q..ptag_end];
            const pkey = dsmlAttr(attrs, "name");
            const pstring = dsmlAttr(attrs, "string");
            const vstart = ptag_end + 1;
            const vclose = dsmlFind(text, vstart, DSML_PARAM_CLOSE) orelse {
                // unterminated value = server-cut fragment; drop it, end scan
                p = text.len;
                break;
            };
            q = vclose.end;
            if (pkey) |k| {
                const value = text[vstart..vclose.at];
                // string attr absent (mangled opener): treat JSON-parseable
                // values as JSON, everything else as a string.
                const is_string = if (pstring) |sf|
                    !std.mem.eql(u8, sf, "false")
                else
                    !dsmlValueIsJson(allocator, value);
                // dedup: last wins
                var replaced = false;
                for (params.items) |*prm| {
                    if (std.mem.eql(u8, prm.key, k)) {
                        prm.value = value;
                        prm.is_string = is_string;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) try params.append(allocator, .{ .key = k, .value = value, .is_string = is_string });
            }
            p = q;
        }

        // Build the arguments object: escaped keys, escaped string values,
        // raw (validated) JSON values.
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        try out.append(allocator, '{');
        for (params.items, 0..) |prm, i| {
            if (i > 0) try out.append(allocator, ',');
            try appendJsonString(allocator, &out, prm.key);
            try out.append(allocator, ':');
            if (!prm.is_string and dsmlValueIsJson(allocator, prm.value)) {
                try out.appendSlice(allocator, std.mem.trim(u8, prm.value, " \t\n\r"));
            } else {
                try appendJsonString(allocator, &out, prm.value);
            }
        }
        try out.append(allocator, '}');
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = try out.toOwnedSlice(allocator) });
        pos = p;
    }
}

fn dsmlValueIsJson(allocator: std.mem.Allocator, value: []const u8) bool {
    const trimmed = std.mem.trim(u8, value, " \t\n\r");
    if (trimmed.len == 0) return false;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return false;
    parsed.deinit();
    return true;
}

// ── Invoke-style tool calls (Anthropic spelling) ───────────────────────
//
// String parameters are RAW bytes; lists/objects/bools/null/numbers are
// spelled as JSON and keep their spelled type (the schema coercion chokepoint
// still gets the final word). The parser keys on the INVOKE tag so a dropped
// wrapper still parses (delimiter-drop class), and truncation salvages NAME +
// completed parameters, never fragment values.
const InvokeDialect = struct {
    invoke: []const u8,
    param: []const u8,
    param_close: []const u8,
    invoke_close: []const u8,
    /// Not the model's own dialect: kept only when it names a declared tool (`filterInferredBySchema`).
    inferred: bool,
};
/// Anthropic's spelling of the same grammar (`<function_calls><invoke name="X"><parameter name="k">`),
/// which models trained on Claude transcripts emit in place of their own (#928).
const BARE_INVOKE: InvokeDialect = .{ .invoke = "<invoke", .param = "<parameter", .param_close = "</parameter>", .invoke_close = "</invoke>", .inferred = true };
const FUNCTION_CALLS_TAG = "<function_calls>";

fn parseInvokeToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall), comptime d: InvokeDialect) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, d.invoke)) |inv| {
        const after_tag = inv + d.invoke.len;
        const seg = text[after_tag..];
        // `<invoked>` is a word, not the tag.
        if (seg.len > 0 and !std.ascii.isWhitespace(seg[0])) {
            pos = after_tag;
            continue;
        }
        // Attribute scan bounded so a runaway unclosed quote can't eat the
        // rest of the output.
        const raw_name = dsmlAttr(seg[0..@min(seg.len, 512)], "name") orelse {
            pos = after_tag;
            continue;
        };
        // No real tool name carries a channel marker: such a name is an echoed marker, not a call.
        const name_slice = raw_name;
        if (std.mem.indexOf(u8, name_slice, "<|") != null) {
            pos = after_tag;
            continue;
        }
        if (name_slice.len == 0) {
            pos = after_tag;
            continue;
        }

        var args_json: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, seg, '>')) |te| {
            const body_all = seg[te + 1 ..];
            const body_end = std.mem.indexOf(u8, body_all, d.invoke_close) orelse body_all.len;
            // An inferred opener with no parameter and no close is prose, as `bareInvokeMarkupCut` reads it.
            if (d.inferred and body_end == body_all.len and std.mem.indexOf(u8, body_all, d.param) == null) {
                pos = after_tag;
                continue;
            }
            args_json = try invokeParamsToJson(allocator, body_all[0..body_end], d);
            pos = if (body_end < body_all.len)
                after_tag + te + 1 + body_end + d.invoke_close.len
            else
                text.len; // truncated body — completed params already salvaged
        } else {
            if (d.inferred) break; // an unclosed opener alone is prose
            pos = text.len; // truncated inside the opening tag: NAME + {}
        }

        const name = try allocator.dupe(u8, name_slice);
        errdefer allocator.free(name);
        const arguments = args_json orelse try allocator.dupe(u8, "{}");
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = d.inferred });
    }
}

/// Assemble `<parameter name="k">v</parameter>` runs into a JSON
/// object string. A value that parses as COMPLETE JSON keeps its spelled type;
/// everything else is a raw string, exact bytes. Keys are escaped + deduped
/// (last wins) via ObjectMap+Stringify, so emitted arguments are ALWAYS valid
/// JSON. Returns null when no complete parameter was found.
fn invokeParamsToJson(allocator: std.mem.Allocator, body: []const u8, comptime d: InvokeDialect) !?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var map: std.json.ObjectMap = .empty;
    var pos: usize = 0;
    var found = false;
    while (std.mem.indexOfPos(u8, body, pos, d.param)) |p| {
        const after = p + d.param.len;
        const seg = body[after..];
        const te = std.mem.indexOfScalar(u8, seg, '>') orelse break; // truncated open tag
        const pname = dsmlAttr(seg[0 .. te + 1], "name") orelse {
            pos = after;
            continue;
        };
        const vstart = after + te + 1;
        // The value runs to the CONFIRMED closing tag — rich values with
        // newlines/quotes/`<` survive; no close by end of text = fragment,
        // dropped.
        const vend_rel = std.mem.indexOf(u8, body[vstart..], d.param_close) orelse break;
        const raw = body[vstart .. vstart + vend_rel];
        const v: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, aa, raw, .{}) catch
            std.json.Value{ .string = raw };
        try map.put(aa, pname, v);
        found = true;
        pos = vstart + vend_rel + d.param_close.len;
    }
    if (!found) return null;
    return try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = map }, .{});
}

fn parseXmlElementToolCall(allocator: std.mem.Allocator, body: []const u8) ?ParsedToolCall {
    var name: ?[]const u8 = null;
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);

    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        const value = body[val_start .. val_start + rel];
        i = val_start + rel + close_tag.len;

        if (std.mem.eql(u8, key, "tool_name") or std.mem.eql(u8, key, "name") or std.mem.eql(u8, key, "tool")) {
            if (name != null) return null;
            const trimmed_name = std.mem.trim(u8, value, " \t\n\r");
            if (trimmed_name.len == 0) return null;
            name = trimmed_name;
        } else {
            args_map.put(allocator, key, .{ .string = value }) catch return null;
        }
    }

    const n = name orelse return null;
    const args_str = std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch return null;
    const name_owned = allocator.dupe(u8, n) catch {
        allocator.free(args_str);
        return null;
    };
    return .{ .name = name_owned, .arguments = args_str };
}

/// Repair a JSON object whose trailing closer(s) were dropped — e.g. the
/// model wrote `{"name":"edit","arguments":{…,"edits":[{…}]}` and went
/// straight to its close tag, one `}` short. String/escape-aware bracket
/// stack; appends exactly the missing closers in nesting order. Returns null
/// when the text doesn't start with `{`, is already balanced (trailing
/// garbage is someone else's problem), ends mid-string, or nests deeper
/// than the stack (not worth guessing).
fn completeUnbalancedJsonObject(allocator: std.mem.Allocator, content: []const u8) ?[]u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    if (trimmed.len == 0 or trimmed[0] != '{') return null;
    var stack: [16]u8 = undefined;
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    for (trimmed) |c| {
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '{' => {
                if (depth == stack.len) return null;
                stack[depth] = '}';
                depth += 1;
            },
            '[' => {
                if (depth == stack.len) return null;
                stack[depth] = ']';
                depth += 1;
            },
            '}', ']' => {
                if (depth == 0 or stack[depth - 1] != c) return null;
                depth -= 1;
                // Balanced before the end → trailing garbage, not a
                // truncated tail. balancedJsonObject owns that case.
                if (depth == 0) return null;
            },
            else => {},
        }
    }
    if (in_string or depth == 0) return null;
    var out = allocator.alloc(u8, trimmed.len + depth) catch return null;
    @memcpy(out[0..trimmed.len], trimmed);
    var i: usize = 0;
    while (i < depth) : (i += 1) {
        out[trimmed.len + i] = stack[depth - 1 - i];
    }
    return out;
}

/// Snap a balanced JSON array starting at the first `[`. Mirrors
/// balancedJsonObject: string/escape aware, tracks BOTH bracket kinds so
/// objects nested in the array can't fool the depth count.
fn balancedJsonArray(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '[') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '[', '{' => depth += 1,
            ']', '}' => {
                depth -= 1;
                if (depth == 0) return trimmed[start .. i + 1];
                if (depth < 0) return null; // mismatched — give up
            },
            else => {},
        }
    }
    return null;
}

/// Parse a JSON array of tool-call objects (`[{"name":…,"arguments":…}, …]`)
/// and append EVERY call. All-or-nothing: if any element fails to parse as a
/// tool call, nothing is appended — a prose-ish array must pass through as
/// text rather than half-execute.
fn appendJsonToolCallArray(
    allocator: std.mem.Allocator,
    arr_text: []const u8,
    calls: *std.ArrayList(ParsedToolCall),
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, arr_text, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .array) return;
    const items = parsed.value.array.items;
    if (items.len == 0) return;

    var pending = std.ArrayList(ParsedToolCall).empty;
    var ok = true;
    defer {
        for (pending.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        pending.deinit(allocator);
    }
    for (items) |item| {
        if (item != .object) {
            ok = false;
            break;
        }
        // Reuse tryParseJsonToolCall (leaf-name walk, flat-shape synthesis)
        // by round-tripping the element through its text form.
        const item_text = std.json.Stringify.valueAlloc(allocator, item, .{}) catch {
            ok = false;
            break;
        };
        defer allocator.free(item_text);
        const tc = tryParseJsonToolCall(allocator, item_text) orelse {
            ok = false;
            break;
        };
        try pending.append(allocator, tc);
    }
    if (!ok) return;
    try calls.appendSlice(allocator, pending.items);
    pending.clearRetainingCapacity();
}

/// Lenient recovery for tool-call argument JSON that small models mangle when
/// emitting large string values (file contents). Two failure modes, both of
/// which make strict std.json reject the WHOLE blob so the call would be dropped
/// and leak as visible text:
///   • raw control bytes (literal newlines/tabs) inside a string instead of
///     `\n`/`\t` — the dominant big-file failure;
///   • unescaped inner double-quotes (`<meta charset="UTF-8">`) and invalid
///     backslash escapes (Windows paths, regex) inside a string.
/// Re-serializes a CLEAN object by walking the input with a position-aware
/// tolerant parser (key vs value context): control bytes are re-escaped, lone/
/// invalid backslashes are escaped, and a `"` closes a string only at a
/// structural delimiter (`:` after a key; `,`/`}`/`]`/end after a value) — any
/// other `"` is an inner content quote and gets escaped. The CALLER strict-parses
/// the result, so a mis-recovery that yields invalid JSON is discarded; the only
/// residual risk is a value string closed early on pathological content (a
/// literal `"}` / `",` byte-sequence inside the file), which still beats dropping
/// the call. Does NOT tolerate truncation (no auto-close of open containers) —
/// that stays with completeUnbalancedJsonObject. Returns an allocator-owned
/// normalized JSON string, or null when no well-formed leading object recovers.
fn looseRepairToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const obj_start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitObject(allocator, &out, trimmed, obj_start, 0) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

/// Tolerant re-serialize of a top-level JSON CONTAINER (`[…]` or `{…}`) — the
/// array-aware sibling of looseRepairToolCallJson. Used by the schema coercion to
/// salvage a container-typed argument the model mangled (e.g. a missing comma
/// inside an `edits` array). Same discipline: the caller strict-parses the
/// result, so a mis-repair yielding invalid JSON is discarded. Returns null when
/// the text doesn't start with `[`/`{` or can't be re-serialized.
fn looseRepairContainer(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0 or (trimmed[0] != '[' and trimmed[0] != '{')) return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitValue(allocator, &out, trimmed, 0, 0, .object_value) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

const LooseStringCtx = enum { key, object_value, array_value };
const LooseError = error{ Malformed, OutOfMemory };
const loose_max_depth = 32;

fn looseSkipWs(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return i;
}

fn looseSkipWsCommas(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return i;
}

fn looseIsHex4(s: []const u8) bool {
    if (s.len != 4) return false;
    for (s) |c| {
        const ok = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
        if (!ok) return false;
    }
    return true;
}

fn looseEmitObject(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '{') return error.Malformed;
    i += 1;
    try out.append(allocator, '{');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed; // unterminated (not truncation-tolerant)
        if (body[i] == '}') {
            try out.append(allocator, '}');
            return i + 1;
        }
        if (body[i] != '"') return error.Malformed; // key must be a string
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitString(allocator, out, body, i, .key);
        i = looseSkipWs(body, i);
        if (i >= body.len or body[i] != ':') return error.Malformed;
        i += 1;
        try out.append(allocator, ':');
        i = try looseEmitValue(allocator, out, body, i, depth, .object_value);
    }
}

fn looseEmitArray(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '[') return error.Malformed;
    i += 1;
    try out.append(allocator, '[');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed;
        if (body[i] == ']') {
            try out.append(allocator, ']');
            return i + 1;
        }
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitValue(allocator, out, body, i, depth, .array_value);
    }
}

fn looseEmitValue(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    str_ctx: LooseStringCtx,
) LooseError!usize {
    const i = looseSkipWs(body, start);
    if (i >= body.len) return error.Malformed;
    return switch (body[i]) {
        '"' => try looseEmitString(allocator, out, body, i, str_ctx),
        '{' => try looseEmitObject(allocator, out, body, i, depth + 1),
        '[' => try looseEmitArray(allocator, out, body, i, depth + 1),
        else => try looseEmitScalar(allocator, out, body, i),
    };
}

fn looseEmitScalar(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
) LooseError!usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ',', '}', ']', ' ', '\t', '\n', '\r' => break,
            else => try out.append(allocator, body[i]),
        }
    }
    if (i == start) return error.Malformed;
    return i; // strict re-parse validates the token was a real number/bool/null
}

fn looseEmitString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    ctx: LooseStringCtx,
) LooseError!usize {
    // body[start] == '"'
    try out.append(allocator, '"');
    var i = start + 1;
    while (i < body.len) {
        const c = body[i];
        if (c == '\\') {
            if (i + 1 >= body.len) {
                try out.appendSlice(allocator, "\\\\"); // trailing backslash → literal
                i += 1;
                continue;
            }
            const n = body[i + 1];
            switch (n) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {
                    try out.append(allocator, '\\');
                    try out.append(allocator, n);
                    i += 2;
                },
                'u' => {
                    if (i + 6 <= body.len and looseIsHex4(body[i + 2 .. i + 6])) {
                        try out.appendSlice(allocator, body[i .. i + 6]);
                        i += 6;
                    } else {
                        try out.appendSlice(allocator, "\\\\"); // invalid \u → literal backslash
                        i += 1;
                    }
                },
                else => {
                    try out.appendSlice(allocator, "\\\\"); // invalid escape → literal backslash
                    i += 1;
                },
            }
            continue;
        }
        if (c == '"') {
            const j = looseSkipWs(body, i + 1);
            const is_close = switch (ctx) {
                .key => j < body.len and body[j] == ':',
                .object_value => j >= body.len or body[j] == ',' or body[j] == '}',
                .array_value => j >= body.len or body[j] == ',' or body[j] == ']',
            };
            if (is_close) {
                try out.append(allocator, '"');
                return i + 1;
            }
            try out.appendSlice(allocator, "\\\""); // inner content quote → escape
            i += 1;
            continue;
        }
        if (c < 0x20) {
            switch (c) {
                '\n' => try out.appendSlice(allocator, "\\n"),
                '\t' => try out.appendSlice(allocator, "\\t"),
                '\r' => try out.appendSlice(allocator, "\\r"),
                0x08 => try out.appendSlice(allocator, "\\b"),
                0x0c => try out.appendSlice(allocator, "\\f"),
                else => {
                    var buf: [6]u8 = undefined;
                    const esc = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try out.appendSlice(allocator, esc);
                },
            }
            i += 1;
            continue;
        }
        try out.append(allocator, c);
        i += 1;
    }
    // Ran off the end with no structural close — unterminated. The enclosing
    // container won't find its close either → looseEmitObject returns Malformed
    // and the whole recovery is discarded (truncation handled elsewhere). Emit
    // a closing quote so the discarded buffer stays well-formed.
    try out.append(allocator, '"');
    return i;
}

fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch blk: {
        // Strict parse failed. Try a chain of repairs for known shapes:
        //   1. {"name":"shell", {"command":"ls"}}           - missing `"arguments":` key entirely
        //   2. {"name":"shell", arguments":{"command":..}}   - missing OPENING quote on `arguments`
        //   3. {"name":"edit", "arguments": {…}]}            — truncated tail, final closer(s) dropped
        //      before the close tag (captured live)
        // Repairs are cheap and run only on the parse-failure path.
        if (repairBrokenToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                break :blk reparsed;
            } else |_| {}
        }
        //   4. Mangled big-file content: raw control bytes (literal newlines/
        //      tabs) and unescaped inner quotes inside a string value — what
        //      small models emit when writing a large file in one shot. The
        //      re-escaped copy is re-validated by the strict parse below, so a
        //      mis-recovery that yields invalid JSON is silently discarded.
        if (looseRepairToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                log.info("  [tool-parse] loose-repair recovered mangled tool-call JSON (raw control bytes / unescaped quotes)\n", .{});
                break :blk reparsed;
            } else |_| {}
        }
        const completed = completeUnbalancedJsonObject(allocator, text) orelse return null;
        defer allocator.free(completed);
        const reparsed = std.json.parseFromSlice(std.json.Value, allocator, completed, .{}) catch return null;
        break :blk reparsed;
    };
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    var obj = parsed.value.object;

    // A MoE model sometimes emits nested-name garbage like
    //   {"name":{"name":{"name":"write","arguments":{...}}}}
    // Walk down through up to a few levels of nested "name" objects to find the
    // leaf object that has a string name + arguments/parameters. Observed in the
    // wild with pi in non-thinking mode.
    {
        var depth: u8 = 0;
        while (depth < 4) : (depth += 1) {
            const nv = obj.get("name") orelse return null;
            switch (nv) {
                .string => break,
                .object => |inner| obj = inner,
                else => return null,
            }
        }
    }

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_str: []const u8 = blk: {
        if (obj.get("arguments") orelse obj.get("parameters")) |args_val| {
            break :blk switch (args_val) {
                .object => std.json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null,
                .string => |s| allocator.dupe(u8, s) catch return null,
                else => return null,
            };
        }
        // Flat shape (e.g. a MoE model): {"name":"shell","command":"ls"} -
        // parameters live at top level. Synthesize an arguments object from
        // every non-metadata key.
        var flat_map: std.json.ObjectMap = .empty;
        defer flat_map.deinit(allocator);
        var it = obj.iterator();
        while (it.next()) |entry| {
            const k = entry.key_ptr.*;
            if (std.mem.eql(u8, k, "name") or
                std.mem.eql(u8, k, "id") or
                std.mem.eql(u8, k, "type"))
            {
                continue;
            }
            flat_map.put(allocator, k, entry.value_ptr.*) catch return null;
        }
        if (flat_map.count() == 0) return null;
        const flat_value = std.json.Value{ .object = flat_map };
        break :blk std.json.Stringify.valueAlloc(allocator, flat_value, .{}) catch return null;
    };

    return .{
        .name = allocator.dupe(u8, name_val.string) catch {
            allocator.free(args_str);
            return null;
        },
        .arguments = args_str,
    };
}

/// Run known removed-model MoE-broken-JSON repairs in sequence; return the first
/// successfully-repaired string (allocator-owned), or null if none match.
fn repairBrokenToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    if (repairFlatBraceToolCallJson(allocator, text)) |s| return s;
    if (repairUnquotedArgsKey(allocator, text)) |s| return s;
    return null;
}

/// Repair `{"name":"x", arguments":{...}}` (missing OPENING quote on
/// `arguments`/`parameters`) by injecting the quote.
fn repairUnquotedArgsKey(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    // Look for the unquoted-key pattern. Both `arguments` and `parameters`
    // have been observed; both are short fixed strings, so direct search is fine.
    const candidates = [_][]const u8{ ", arguments\":", ",arguments\":", ", parameters\":", ",parameters\":" };
    for (candidates) |needle| {
        if (std.mem.indexOf(u8, trimmed, needle)) |at| {
            const insert_at = at + 1; // right after the comma
            // Skip leading whitespace after the comma so the injected `"` lands on the identifier.
            var p = insert_at;
            while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
            return std.fmt.allocPrint(allocator, "{s}\"{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
        }
    }
    return null;
}

/// Repair the removed-model MoE-broken shape `{"name":"x", {"k":"v"}}` by injecting
/// the missing `"arguments":` key between the comma and the inner object.
/// Returns an allocator-owned repaired JSON string, or null if the pattern
/// doesn't match.
fn repairFlatBraceToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    if (trimmed.len < 4 or trimmed[0] != '{') return null;

    // Locate `"name"` key.
    const name_at = std.mem.indexOf(u8, trimmed, "\"name\"") orelse return null;
    var p = name_at + 6;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ':') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '"') return null;

    // Skip the string value.
    p += 1;
    while (p < trimmed.len and trimmed[p] != '"') {
        if (trimmed[p] == '\\' and p + 1 < trimmed.len) p += 2 else p += 1;
    }
    if (p >= trimmed.len) return null;
    p += 1;

    // Expect `, {` (any whitespace).
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ',') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '{') return null;

    return std.fmt.allocPrint(allocator, "{s}\"arguments\":{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
}

/// Strip one layer of outer braces from a `{{…}}`-wrapped JSON object. Models
/// sometimes emit args with a doubled brace layer (a literal-brace artifact from
/// Jinja `{{ }}` templates). Only touches strings that start with `{{` and end
/// with `}}`, so well-formed single-brace JSON passes through unchanged.
fn unwrapDoubleBraces(s: []const u8) []const u8 {
    if (s.len >= 4 and std.mem.startsWith(u8, s, "{{") and std.mem.endsWith(u8, s, "}}")) {
        return s[1 .. s.len - 1];
    }
    return s;
}

/// Strip artifacts models sometimes append to a tool name — surrounding
/// whitespace and a trailing ':' (Gemma 4 12B leaks one via its `call:NAME:`
/// format). A valid OpenAI tool name matches `^[A-Za-z0-9_-]{1,64}$`, so a
/// trailing colon is never legitimate and is always safe to drop.
fn sanitizeToolName(raw: []const u8) []const u8 {
    var name = std.mem.trim(u8, raw, " \t\n\r");
    while (name.len > 0 and name[name.len - 1] == ':') {
        name = std.mem.trim(u8, name[0 .. name.len - 1], " \t\n\r");
    }
    return name;
}

/// Parse Gemma 4 tool call format: "call:function_name{json_args}"
/// `input_truncated`: the call had NO `<tool_call|>` close tag — the
/// generation was cut (EOS mid-call, max_tokens, or the scheduler's
/// degenerate-tail-loop guard). Under truncation, a value whose scan runs to
/// end-of-body without its terminator is a FRAGMENT and is dropped (the
/// Hermes-truncation rule: a half-written file is worse than a re-issued
/// write); with the close tag present, behavior is byte-identical to before.
fn parseGemma4ToolCall(allocator: std.mem.Allocator, content: []const u8, input_truncated: bool) ?ParsedToolCall {
    const prefix = "call:";
    if (!std.mem.startsWith(u8, content, prefix)) return null;
    const after_prefix = content[prefix.len..];

    // Find the opening brace
    const brace_pos = std.mem.indexOf(u8, after_prefix, "{") orelse return null;
    const name = sanitizeToolName(after_prefix[0..brace_pos]);
    if (name.len == 0) return null;

    var args_str = after_prefix[brace_pos..];

    // Gemma 4 uses {{ }} (double braces) for literal braces in Jinja templates.
    // The model often generates {{"key":"value"}} — unwrap the outer braces.
    args_str = unwrapDoubleBraces(args_str);

    // Try JSON first (model sometimes outputs valid JSON arguments)
    if (std.json.parseFromSlice(std.json.Value, allocator, args_str, .{})) |parsed| {
        defer parsed.deinit();
        if (parsed.value == .object) {
            return .{
                .name = allocator.dupe(u8, name) catch return null,
                .arguments = allocator.dupe(u8, args_str) catch return null,
            };
        }
    } else |_| {}

    // Strict JSON failed — but the model may have emitted standard JSON with
    // mangled escaping (raw newlines / unescaped quotes in big content) rather
    // than the custom <|"|> format. Recover that BEFORE the custom-format
    // converter, which would garble standard JSON. looseRepair returns null for
    // the bare-key `{key:<|"|>v<|"|>}` form, so the custom path still runs.
    if (looseRepairToolCallJson(allocator, args_str)) |repaired| {
        var keep = false;
        defer if (!keep) allocator.free(repaired);
        if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
            defer reparsed.deinit();
            if (reparsed.value == .object) {
                const name_owned = allocator.dupe(u8, name) catch return null;
                keep = true;
                log.info("  [tool-parse] loose-repair recovered mangled Gemma 4 tool-call JSON\n", .{});
                return .{ .name = name_owned, .arguments = repaired };
            }
        } else |_| {}
    }

    // Convert Gemma 4 custom format to JSON:
    // {key:<|"|>value<|"|>,key2:<|"|>value2<|"|>} → {"key":"value","key2":"value2"}
    const json = convertGemma4ArgsToJson(allocator, args_str, input_truncated) orelse {
        log.info("  [tool-parse] convertGemma4ArgsToJson FAILED for: {s}\n", .{args_str[0..@min(args_str.len, 200)]});
        return null;
    };
    return .{
        .name = allocator.dupe(u8, name) catch return null,
        .arguments = json,
    };
}

/// Convert Gemma 4's custom key-value format to JSON.
/// Input:  {key:<|"|>value<|"|>,nested:{k:<|"|>v<|"|>},arr:[<|"|>a<|"|>]}
/// Output: {"key":"value","nested":{"k":"v"},"arr":["a"]}
///
/// Gemma 4 emits its own object/array/string syntax (bare keys, `<|"|>…<|"|>`
/// string delimiters) that can nest arbitrarily, so every value is converted
/// recursively — see `convertGemma4Value`. Nested structures that already use
/// valid JSON (regular `"…"` strings, numbers, bools, null) pass through
/// unchanged. The args are always an object, with or without a literal `{…}`.
fn convertGemma4ArgsToJson(allocator: std.mem.Allocator, input: []const u8, input_truncated: bool) ?[]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    // `cut` = the truncation landed INSIDE a value somewhere below; the pair
    // holding it has already been rolled back (fragment values never ship).
    var cut = false;
    _ = convertGemma4Object(allocator, &result, input, 0, 0, input_truncated, &cut) orelse return null;
    return result.toOwnedSlice(allocator) catch return null;
}

const gemma4_str_delim = "<|\"|>";
/// Recursion-depth guard against adversarial deeply-nested model output.
const gemma4_max_depth = 64;

fn gemma4SkipWs(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return pos;
}

fn gemma4SkipWsCommas(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return pos;
}

/// Parse a Gemma 4 object beginning at `start` (an optional leading `{` is
/// consumed; absent braces are tolerated so a brace-less body still parses).
/// Appends the JSON object to `result`; returns the index just past it.
fn convertGemma4Object(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = gemma4SkipWs(body, start);
    if (pos < body.len and body[pos] == '{') pos += 1;
    result.append(allocator, '{') catch return null;

    // Dedup keys within this object (a repeated key would make std.json reject
    // the args with error.DuplicateField — same class as parseHermesToolCall's
    // dup-param bug). Keys are slices into the stable input `body`.
    var seen = std.ArrayList([]const u8).empty;
    defer seen.deinit(allocator);

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == '}') {
            pos += 1;
            closed = true;
            break;
        }

        // Key: everything up to the first ':'.
        const colon = std.mem.indexOfScalar(u8, body[pos..], ':') orelse break;
        const key_raw = std.mem.trim(u8, body[pos .. pos + colon], " \t\n\r");
        // Strip surrounding quotes if present (model sometimes quotes keys).
        const key = if (key_raw.len >= 2 and key_raw[0] == '"' and key_raw[key_raw.len - 1] == '"')
            key_raw[1 .. key_raw.len - 1]
        else
            key_raw;
        pos = pos + colon + 1;

        // Tentatively emit comma+key+value; roll back if this key is a dup. The
        // value is still consumed (pos advances) so the dup's payload is skipped.
        const mark = result.items.len;
        if (!first) result.append(allocator, ',') catch return null;
        // Escape the KEY — a raw `"`/control byte in the key would emit invalid
        // JSON (the Gemma converter's output is not strict-re-validated).
        appendJsonString(allocator, result, key) catch return null;
        result.append(allocator, ':') catch return null;

        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;

        // The truncation landed INSIDE this value — it is a fragment (partial
        // file content, partial url) and must never ship as a real argument
        // (the Hermes-truncation rule, extended to the Gemma arm 2026-07-14:
        // php.html loop-stop capture). Roll the whole pair back; the scan
        // consumed to end-of-body, so this object is done.
        if (cut.*) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            break;
        }

        var is_dup = false;
        for (seen.items) |s| {
            if (std.mem.eql(u8, s, key)) {
                is_dup = true;
                break;
            }
        }
        if (is_dup) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            continue;
        }
        seen.append(allocator, key) catch return null;
        first = false;
    }

    // A NESTED object the cut landed inside (it never saw its `}`) is itself a
    // fragment — even when every pair inside completed, more were coming; the
    // parent rolls the whole container back. The ROOT object (depth 0) is
    // exempt: pairs completed before the cut are intact values and survive —
    // only the value holding the cut drops.
    if (input_truncated and depth > 0 and !closed) cut.* = true;

    result.append(allocator, '}') catch return null;
    return pos;
}

/// Parse a Gemma 4 array beginning at `start` (`body[start] == '['`). Appends
/// the JSON array to `result`; returns the index just past it.
fn convertGemma4Array(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = start + 1; // consume '['
    result.append(allocator, '[') catch return null;

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == ']') {
            pos += 1;
            closed = true;
            break;
        }
        if (!first) result.append(allocator, ',') catch return null;
        first = false;
        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;
        // Fragment below: stop consuming; the parent object rolls the whole
        // container pair back, so no local cleanup is needed.
        if (cut.*) break;
    }

    // Same fragment rule as nested objects: an unclosed array under truncation
    // is a partial LIST — more elements were coming (a partial edits[] applies
    // fragmentary work, the same hazard as partial file content).
    if (input_truncated and !closed) cut.* = true;

    result.append(allocator, ']') catch return null;
    return pos;
}

/// Convert one Gemma 4 value (custom string / JSON string / object / array /
/// bare literal) beginning at `start`. String branches are checked first so
/// braces inside a string stay string content. Returns the index past it.
fn convertGemma4Value(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    var pos = gemma4SkipWs(body, start);
    if (pos >= body.len) {
        // `key:` then end-of-body. Under truncation the value is a fragment
        // that never arrived — drop the pair (pre-fix this shipped
        // {"path":""} and a client would act on the bogus empty value).
        if (input_truncated) {
            cut.* = true;
            return pos;
        }
        result.appendSlice(allocator, "\"\"") catch return null;
        return pos;
    }

    // Gemma custom string: <|"|>…<|"|> (closing delimiter may be missing if
    // the model output was truncated).
    if (pos + gemma4_str_delim.len <= body.len and
        std.mem.eql(u8, body[pos .. pos + gemma4_str_delim.len], gemma4_str_delim))
    {
        pos += gemma4_str_delim.len;
        const end_idx = std.mem.indexOf(u8, body[pos..], gemma4_str_delim);
        const value = if (end_idx) |e| body[pos .. pos + e] else blk: {
            // Closing <|"|> missing.
            if (input_truncated) {
                // The call itself was cut (no <tool_call|>): the truncation
                // landed INSIDE this string — a fragment (partial file
                // content, partial url) that must never ship as a real
                // argument (live 2026-07-14 php.html: the loop-stop guard cut
                // a write call mid-`content` and the 1.1 KB fragment shipped).
                cut.* = true;
                return body.len;
            }
            // Complete call (close tag present), delimiter just dropped: a
            // plain to-end-of-body scan swallows the args object's own `}`
            // and any stray fence garbage the model tacked on — a real write
            // call reached disk as "mlx_pi1.html`}". Trim one trailing `}`
            // (the enclosing object's closer) plus backtick/whitespace junk.
            var v = std.mem.trimEnd(u8, body[pos..], " \t\n\r");
            if (std.mem.endsWith(u8, v, "}")) v = v[0 .. v.len - 1];
            v = std.mem.trimEnd(u8, v, "` \t\n\r");
            break :blk v;
        };
        pos = if (end_idx) |e|
            pos + e + gemma4_str_delim.len
        else
            body.len;
        appendJsonString(allocator, result, value) catch return null;
        return pos;
    }

    // Regular JSON string: already valid, copy verbatim (respecting \" escapes).
    if (body[pos] == '"') {
        var end = pos + 1;
        var closed_quote = false;
        while (end < body.len) : (end += 1) {
            if (body[end] == '\\' and end + 1 < body.len) {
                end += 1;
                continue;
            }
            if (body[end] == '"') {
                closed_quote = true;
                end += 1;
                break;
            }
        }
        // Unclosed JSON string at end-of-body under truncation: same fragment
        // rule — and copying it verbatim would emit an UNCLOSED string, i.e.
        // invalid JSON that only the {}-fallback safety net could catch.
        if (!closed_quote and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, body[pos..end]) catch return null;
        return end;
    }

    if (body[pos] == '{') return convertGemma4Object(allocator, result, body, pos, depth + 1, input_truncated, cut);
    if (body[pos] == '[') return convertGemma4Array(allocator, result, body, pos, depth + 1, input_truncated, cut);

    // Bare value — a JSON literal (number/bool/null) terminates at the
    // enclosing separator and is emitted verbatim.
    const sep_rel = std.mem.indexOfAny(u8, body[pos..], ",}]");
    const first_sep = sep_rel orelse (body.len - pos);
    const head = std.mem.trim(u8, body[pos .. pos + first_sep], " \t\n\r");
    if (isJsonLiteral(head)) {
        // A literal that ran to end-of-body under truncation has no
        // terminator either — uniform fragment rule (a cut `5` may have been
        // `50`; the client re-asks and gets the real value).
        if (sep_rel == null and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, head) catch return null;
        return pos + first_sep;
    }

    // Unquoted STRING. The bare scan above stops at the first `,`/`}`/`]`, which
    // truncates rich content (HTML/markdown) that legitimately contains those
    // bytes. Observed live on gemma-4-e4b-it writing a full page: it dropped the
    // OPENING <|"|> on `content` but kept the CLOSING one
    // (`content:<!DOCTYPE…>…</html><|"|>,path:…`), so content got cut at the
    // viewport meta's comma and the rest became bogus keys → invalid args. Only
    // rich content (multi-line or markup) gets the wider scan; a plain short
    // bare token keeps the first-separator behavior so it can't swallow a
    // sibling field (e.g. `command:ls -la`).
    const rich = std.mem.indexOfScalar(u8, head, '\n') != null or
        std.mem.indexOfScalar(u8, head, '<') != null;
    if (rich) {
        // Prefer a CLOSING <|"|> as the boundary (dropped-opener case). Confirm
        // it's a closer — the byte after it (past whitespace) must be a
        // separator/closer — so a LATER field's OPENING delimiter isn't grabbed.
        if (std.mem.indexOf(u8, body[pos..], gemma4_str_delim)) |close_rel| {
            const after = gemma4SkipWs(body, pos + close_rel + gemma4_str_delim.len);
            const is_closer = after >= body.len or body[after] == ',' or body[after] == '}' or body[after] == ']';
            if (is_closer) {
                const value = std.mem.trim(u8, body[pos .. pos + close_rel], " \t\n\r");
                appendJsonString(allocator, result, value) catch return null;
                return pos + close_rel + gemma4_str_delim.len;
            }
        }
        // No usable closing delimiter at all. Under truncation the rest of the
        // body IS the fragment: consuming to the first comma would ship a
        // partial value AND shred the remainder into bogus keys, and the
        // last-`}` heuristic below would cut a truncated page at some interior
        // CSS brace — drop the whole value instead.
        if (input_truncated) {
            cut.* = true;
            return body.len;
        }
        // Complete call, both delimiters dropped. At the TOP level the value
        // runs to the object's closing `}` (the last one); nested values keep
        // the narrow scan since the outer `}` isn't theirs.
        if (depth == 0) {
            if (std.mem.lastIndexOfScalar(u8, body, '}')) |last_brace| {
                if (last_brace > pos) {
                    const value = std.mem.trim(u8, body[pos..last_brace], " \t\n\r");
                    appendJsonString(allocator, result, value) catch return null;
                    return last_brace;
                }
            }
        }
    }

    // A short bare string that ran to end-of-body under truncation is a
    // fragment too (`command:ls -la` cut mid-flags) — uniform rule.
    if (sep_rel == null and input_truncated) {
        cut.* = true;
        return body.len;
    }

    // Plain short bare string — terminate at the first separator (unchanged).
    appendJsonString(allocator, result, head) catch return null;
    return pos + first_sep;
}

/// End of a `<parameter=…>` value: the LAST `</parameter>` before the next opener (a value may
/// spell its own close tag); an unclosed parameter falls back to the first close.
fn hermesValueEnd(body: []const u8, val_start: usize) ?usize {
    const bound = std.mem.indexOfPos(u8, body, val_start, "<parameter=") orelse body.len;
    var best: ?usize = null;
    var i = val_start;
    while (std.mem.indexOfPos(u8, body, i, "</parameter>")) |at| {
        if (at >= bound) break;
        best = at;
        i = at + 1;
    }
    return best orelse std.mem.indexOfPos(u8, body, val_start, "</parameter>");
}

/// If `at` falls inside a `<parameter=…>` value, the position just past that value's close; else null.
fn hermesParamSpanEnclosing(body: []const u8, at: usize) ?usize {
    const open = std.mem.lastIndexOf(u8, body[0..at], "<parameter=") orelse return null;
    const gt = std.mem.indexOfScalarPos(u8, body, open, '>') orelse return null;
    if (gt >= at) return null;
    const end = hermesValueEnd(body, gt + 1) orelse return null;
    if (end <= at) return null;
    return end + "</parameter>".len;
}

/// A `<tool_call>` body whose `<function=` opener lost its `<` to a sampled special token
/// (`<|im_start|>function=read_file>`, live under speculative): one mangled delimiter,
/// name and arguments intact. Returns the body with that special spelled as `<`, null when the
/// body already has a `<function=` or no `<|…|>function=` run; the caller owns the copy.
fn hermesSpecialOpenerRepair(allocator: std.mem.Allocator, body: []const u8) !?[]u8 {
    if (std.mem.indexOf(u8, body, "<function=") != null) return null;
    const fn_at = std.mem.indexOf(u8, body, "function=") orelse return null;
    if (fn_at < 4 or !std.mem.endsWith(u8, body[0..fn_at], "|>")) return null;
    const open = std.mem.lastIndexOf(u8, body[0..fn_at], "<|") orelse return null;
    if (std.mem.indexOfAny(u8, body[open + 2 .. fn_at - 2], " \t\r\n<>") != null) return null;
    const out = try allocator.alloc(u8, body.len - (fn_at - open) + 1);
    @memcpy(out[0..open], body[0..open]);
    out[open] = '<';
    @memcpy(out[open + 1 ..], body[fn_at..]);
    return out;
}

fn parseHermesToolCall(allocator: std.mem.Allocator, block: []const u8) ?ParsedToolCall {
    const fn_start_tag = "<function=";
    const fn_start = std.mem.indexOf(u8, block, fn_start_tag) orelse return null;
    var name_start = fn_start + fn_start_tag.len;
    while (name_start < block.len and std.ascii.isWhitespace(block[name_start])) name_start += 1;
    // A name never spans a line or a tag: a newline or `<` before the `>` means
    // the model dropped the `>`, so the name ends there and the body starts there.
    const name_end = std.mem.indexOfAnyPos(u8, block, name_start, ">\n\r<") orelse return null;
    const fn_name = std.mem.trim(u8, block[name_start..name_end], " \t");
    if (fn_name.len == 0) return null;

    var args_map = std.ArrayList(u8).empty;
    defer args_map.deinit(allocator);
    args_map.append(allocator, '{') catch return null;

    const fn_body_start = if (block[name_end] == '>') name_end + 1 else name_end;
    const fn_end = blk: {
        var from = fn_body_start;
        while (std.mem.indexOfPos(u8, block, from, "</function>")) |found| {
            if (hermesParamSpanEnclosing(block, found)) |past| {
                from = past;
                continue;
            }
            break :blk found - fn_body_start;
        }
        break :blk block.len - fn_body_start;
    };
    const fn_body = block[fn_body_start .. fn_body_start + fn_end];

    // Track emitted parameter names so a repeated `<parameter=NAME>` can't
    // produce a duplicate JSON key (std.json rejects those with DuplicateField —
    // invalid JSON to the client; live soak record 443, a 0.8B model emitting two
    // `<parameter=edits>` blocks). First occurrence wins.
    var seen_names = std.ArrayList([]const u8).empty;
    defer seen_names.deinit(allocator);

    var param_search: usize = 0;
    var first_param = true;
    while (std.mem.indexOf(u8, fn_body[param_search..], "<parameter=")) |ps| {
        const p_name_start = param_search + ps + "<parameter=".len;
        const p_name_end = std.mem.indexOf(u8, fn_body[p_name_start..], ">") orelse break;
        const p_name = std.mem.trim(u8, fn_body[p_name_start .. p_name_start + p_name_end], " \n");

        // A well-formed `<parameter=NAME>` name is a clean token. When the model
        // malforms the tag — e.g. `<parameter=limit=1` with no closing `>` — the
        // `>`-scan spills across a newline into `</parameter`, and interpolating
        // that raw would emit INVALID JSON (live soak record 317). Detect the
        // malformed name and skip just this opener, so the well-formed sibling
        // parameter after it is still recovered. Advancing to p_name_start (past
        // the `<parameter=` we matched) guarantees forward progress.
        if (!isPlausibleParamName(p_name)) {
            param_search = p_name_start;
            continue;
        }

        const p_val_start = p_name_start + p_name_end + 1;
        const p_val_end = (hermesValueEnd(fn_body, p_val_start) orelse break) - p_val_start;
        const p_val = stripHermesValueFraming(fn_body[p_val_start .. p_val_start + p_val_end]);

        // Skip a duplicate name (first wins); still advance past its block.
        var dup = false;
        for (seen_names.items) |seen| {
            if (std.mem.eql(u8, seen, p_name)) {
                dup = true;
                break;
            }
        }
        if (dup) {
            param_search = p_val_start + p_val_end + "</parameter>".len;
            continue;
        }
        seen_names.append(allocator, p_name) catch return null;

        if (!first_param) args_map.append(allocator, ',') catch return null;
        first_param = false;

        // Escape the NAME too — belt-and-braces so no parameter name can ever
        // produce invalid JSON, even if a future malformed shape slips the guard.
        appendJsonString(allocator, &args_map, p_name) catch return null;
        args_map.append(allocator, ':') catch return null;

        // Whitespace around a NUMBER/BOOLEAN/null carries no meaning, so a
        // padded scalar is still typed from its spelling. A STRING's own
        // whitespace is PAYLOAD and ships verbatim: an `old_string` must match
        // the file "exactly, including indentation", and trimming it either
        // failed the edit or — worse — matched a different, un-indented
        // occurrence mid-line. Only the template framing is removed.
        const literal_val = std.mem.trim(u8, p_val, " \t\r\n");
        if (isJsonLiteral(literal_val)) {
            args_map.appendSlice(allocator, literal_val) catch return null;
        } else {
            appendJsonString(allocator, &args_map, p_val) catch return null;
        }

        param_search = p_val_start + p_val_end + "</parameter>".len;
    }

    args_map.append(allocator, '}') catch return null;

    return .{
        .name = allocator.dupe(u8, fn_name) catch return null,
        .arguments = allocator.dupe(u8, args_map.items) catch return null,
    };
}

/// The Hermes XML tool-call template frames a parameter value with EXACTLY
/// one newline on each side (`<parameter=NAME>\nVALUE\n</parameter>` — pinned by the
/// the render test in this file), so exactly that framing is what the parser
/// may remove. Everything else between the tags is the value's own bytes: eating
/// a run of newlines, or the spaces an `old_string` needs to match a file
/// "exactly, including indentation", silently mis-edits.
fn stripHermesValueFraming(raw: []const u8) []const u8 {
    var v = raw;
    if (std.mem.startsWith(u8, v, "\r\n")) {
        v = v[2..];
    } else if (std.mem.startsWith(u8, v, "\n")) {
        v = v[1..];
    }
    if (std.mem.endsWith(u8, v, "\r\n")) {
        v = v[0 .. v.len - 2];
    } else if (std.mem.endsWith(u8, v, "\n")) {
        v = v[0 .. v.len - 1];
    }
    return v;
}

/// A parameter name from a well-formed `<parameter=NAME>` tag is a short token
/// with no whitespace or markup. Reject names carrying a newline, angle bracket,
/// quote, or whitespace - those signal the `>`-scan spilled past a malformed tag.
fn isPlausibleParamName(name: []const u8) bool {
    if (name.len == 0 or name.len > 128) return false;
    for (name) |c| {
        switch (c) {
            '\n', '\r', '\t', ' ', '<', '>', '"' => return false,
            else => {},
        }
    }
    return true;
}

fn isJsonLiteral(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
    // Objects/arrays are handled by brace-matching in the caller, not here
    return isJsonNumber(s);
}

/// JSON's number grammar; `parseFloat` is wider (`Infinity`, `0755`, `.5`) and the literal is spliced UNQUOTED.
pub fn isJsonNumber(s: []const u8) bool {
    var i: usize = 0;
    if (i < s.len and s[i] == '-') i += 1;
    if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
    if (s[i] == '0') {
        i += 1;
    } else {
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    if (i < s.len and s[i] == '.') {
        i += 1;
        if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    return i == s.len;
}

/// One UTF-8 sequence starting at `i`: `valid` = well-formed, else `end` is the maximal
/// subpart to replace with U+FFFD (a character split across two tokens costs ONE replacement).
pub const Utf8Seq = struct { end: usize, valid: bool };

pub fn utf8Next(s: []const u8, i: usize) Utf8Seq {
    const len = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .end = i + 1, .valid = false };
    if (i + len <= s.len and std.unicode.utf8ValidateSlice(s[i .. i + len])) return .{ .end = i + len, .valid = true };
    var j = i + 1;
    while (j < s.len and j < i + len and s[j] & 0xC0 == 0x80) j += 1;
    return .{ .end = j, .valid = false };
}

/// The lossy view of bytes that are not valid UTF-8: one U+FFFD per invalid sequence.
/// A token is a BPE fragment, so ANY string built from model bytes can end mid-character;
/// raw, that makes a response body unparseable and a prompt render fall back silently.
pub fn utf8Sanitize(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var clean = std.ArrayList(u8).empty;
    errdefer clean.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        const seq = utf8Next(input, i);
        try clean.appendSlice(allocator, if (seq.valid) input[i..seq.end] else "\u{FFFD}");
        i = seq.end;
    }
    return clean.toOwnedSlice(allocator);
}

/// THE JSON string escaper (quoted, control bytes \u-escaped). Public so other
/// request builders reuse it instead of hand-rolling a second one — a duplicate
/// escaper is exactly how the control-byte class shipped twice before.
pub fn appendJsonString(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    try buf.append(allocator, '"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            // nlohmann rejects ill-formed UTF-8 like a raw control byte.
            const seq = utf8Next(s, i);
            try buf.appendSlice(allocator, if (seq.valid) s[i..seq.end] else "\u{FFFD}");
            i = seq.end;
            continue;
        }
        i += 1;
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            // Every other control char (e.g. ESC from ANSI codes in tool
            // results) must be \u-escaped — nlohmann inside jinja_render_chat
            // rejects raw control bytes, and the render failure silently
            // downgrades the prompt to fallbackFormatChat.
            0...8, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const n = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try buf.appendSlice(allocator, n);
            },
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
}

/// `tool_choice` that obliges a call: any declared tool, or the named one.
pub const ForcedTool = union(enum) { any, name: []const u8 };

/// The prompt line for a forced choice; the caller frees it.
pub fn toolChoiceInstruction(allocator: std.mem.Allocator, forced: ForcedTool) ![]u8 {
    return switch (forced) {
        .any => allocator.dupe(u8, "\nYou MUST call one of the available functions. Do not respond with text."),
        .name => |n| std.fmt.allocPrint(allocator, "\nYou MUST call the function \"{s}\". Do not respond with text.", .{n}),
    };
}

/// Whether a request's `tools` array declares `name`, as `function.name`
/// (chat completions) or a top-level `name` (Anthropic, Responses).
pub fn toolsDeclare(tools: ?std.json.Value, name: []const u8) bool {
    const arr = tools orelse return false;
    if (arr != .array) return false;
    for (arr.array.items) |t| {
        if (t != .object) continue;
        const holder = if (t.object.get("function")) |f| f else t;
        if (holder != .object) continue;
        const n = holder.object.get("name") orelse continue;
        if (n == .string and std.mem.eql(u8, n.string, name)) return true;
    }
    return false;
}

/// The assistant text that commits a tool call in the template's own dialect,
/// or null for a dialect we cannot spell (the instruction line alone asks then).
pub fn forcedToolOpener(allocator: std.mem.Allocator, template: []const u8, forced: ForcedTool) !?[]u8 {
    // The pair, escaped or literal: Another model family spells a different `<function=` dialect.
    if (std.mem.indexOf(u8, template, "<tool_call>\\n<function=") == null and
        std.mem.indexOf(u8, template, "<tool_call>\n<function=") == null) return null;
    return switch (forced) {
        // Not past `=`: the pre-tokenizer joins it to the name (`=get`), so a lone `=` steers the name.
        .any => try allocator.dupe(u8, "<tool_call>\n<function"),
        .name => |n| try std.fmt.allocPrint(allocator, "<tool_call>\n<function={s}>\n", .{n}),
    };
}

/// Append tool definitions as a system prompt section.
fn appendToolSystemPrompt(allocator: std.mem.Allocator, result_buf: *std.ArrayList(u8), tools_json: []const u8, tool_choice_instruction: ?[]const u8) !void {
    try result_buf.appendSlice(allocator,
        \\You are a helpful assistant with access to the following functions. To call a function, respond with a JSON object in the following format:
        \\<tool_call>
        \\{"name": "function_name", "arguments": {"arg1": "value1"}}
        \\</tool_call>
        \\
        \\Available functions:
        \\
    );
    try result_buf.appendSlice(allocator, tools_json);
    if (tool_choice_instruction) |instr| {
        try result_buf.appendSlice(allocator, instr);
    }
}

// ── Tests ──

const testing = std.testing;

test "collapseDoubledThinkTags collapses 2x → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<|Assistant|></think></think>Hi!");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<|Assistant|></think>Hi!", out);
}

test "collapseDoubledThinkTags collapses triple → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "</think></think></think>X");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("</think>X", out);
}

test "collapseDoubledThinkTags leaves single </think> unchanged" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<think>r</think>Hello");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<think>r</think>Hello", out);
}

test "collapseDoubledThinkTags handles multiple separated doublings" {
    const out = try collapseDoubledThinkTags(testing.allocator, "A</think></think>B</think></think>C");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("A</think>B</think>C", out);
}

test "split content: think tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\nHello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\n\nHello", true, false).content);
}

test "split content: empty for open think tag (truncated thought is never content)" {
    try testing.expectEqualStrings("", splitThinkBlock("<think>still thinking...", true, false).content);
}

test "indexOfThinkCloseTag: a close inside an OPEN tool call is argument payload, not a block close" {
    // Agent traffic writes files about prompts, and the think split runs
    // BEFORE the tool parse — so this cut used to destroy the whole call.
    const payload = "<tool_call>write\n<arg_key>content</arg_key>\n" ++
        "<arg_value>a thought ends with </think> here</arg_value>\n</tool_call>";
    try testing.expect(indexOfThinkCloseTag(payload, 0) == null);

    // A call the model emitted and CLOSED inside its thought: the `</think>`
    // after it is a REAL close and the answer stays content.
    const in_thought = "<think>let me try <tool_call>read</tool_call> hmm</think>The answer is 4.";
    const c = indexOfThinkCloseTag(in_thought, 0) orelse return error.MissingClose;
    try testing.expectEqualStrings("The answer is 4.", in_thought[c.pos + c.len ..]);
    try testing.expectEqualStrings("The answer is 4.", splitThinkBlock(in_thought, true, false).content);

    // An opener that NEVER closes is leaked markup, not a payload container:
    // the close still ends the block, and trimLeakedToolMarkup takes the tail.
    const unclosed = "<think>starting <tool_call>partial</think>Answer.";
    try testing.expect(indexOfThinkCloseTag(unclosed, 0) != null);

    // Ordinary shapes are untouched.
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a", 0).?.pos == 8);
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a<tool_call>x</tool_call>", 0).?.pos == 8);
}

test "split content: text without think tags passes through" {
    try testing.expectEqualStrings("Hello world", splitThinkBlock("Hello world", true, false).content);
}

test "splitThinkBlock with complete think block" {
    const result = splitThinkBlock("<think>reasoning here</think>answer here", false, false);
    try testing.expectEqualStrings("reasoning here", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock with empty reasoning" {
    const result = splitThinkBlock("<think>\n\n</think>\n\nactual content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("actual content", result.content);
}

test "splitThinkBlock thinking=true no close tag, literal opener present" {
    // Model entered thinking but ran out of tokens before closing.
    const result = splitThinkBlock("<think>partial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, template-opened block" {
    // Truncated-thinking leak (regression): the chat template injects
    // `<think>\n` into the generation prompt, so the model's output starts
    // INSIDE the think block with no literal opener. If generation stops
    // (length) before `</think>`, every token so far is reasoning — it must
    // land in reasoning_content, never in content. Matches the streaming path.
    const result = splitThinkBlock("The user wants 17*23. Let me compute", true, true);
    try testing.expectEqualStrings("The user wants 17*23. Let me compute", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, no opener, template did NOT open" {
    // Gemma-style direct answer: thinking enabled but the template injects no
    // opener and the model answered without a thought channel. The answer must
    // stay visible as content.
    const result = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("It is currently 8:15 AM PDT.", result.content);
}

test "splitThinkBlock template-opened block with close tag still splits" {
    // Normal round: no literal opener (template-injected), close tag
    // present — reasoning before it, content after.
    const result = splitThinkBlock("compute 340+51=391</think>\n\n391.", true, true);
    try testing.expectEqualStrings("compute 340+51=391", result.reasoning_content.?);
    try testing.expectEqualStrings("391.", result.content);
}

test "splitThinkBlock thinking=false no tags" {
    const result = splitThinkBlock("just content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("just content", result.content);
}

test "splitThinkBlock strips think prefix in thinking mode" {
    const result = splitThinkBlock("<think>my reasoning", true, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "promptTailOpensThink detects template-injected opener" {
    // A generation prompt tail with thinking on
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>\n"));
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
    // Thinking off renders a CLOSED empty block — must not match
    try testing.expect(!promptTailOpensThink("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // Gemma 4 prompt tail (no injected opener)
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    try testing.expect(!promptTailOpensThink(""));
}

test "promptThinkTailClass resolves bare and plain tails" {
    try testing.expectEqual(ThinkTailClass.bare, promptThinkTailClass("<|im_start|>assistant\n<think>\n"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<|im_start|>assistant\n"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<|start|>assistant"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass(""));
    // A PARTIAL opener is not an opener.
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<think"));
}

test "parseToolCalls JSON format" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // arguments should be valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls MoE nested-name garbage (real capture)" {
    // Captured verbatim from a model generating a tool call in
    // streaming mode with enable_thinking=false and tools present.
    // parseToolCalls previously returned null here because the top-level "name"
    // is an object, not a string - tokens then leaked as plain-text content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": {"name": {"name":  "write", "arguments": {"path": "/tmp/x/app.test.js", "content": "hello"}}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/x/app.test.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello", parsed.value.object.get("content").?.string);
}

test "parseToolCalls MoE double-nested name" {
    // Two layers of nesting — defensive test for a slightly different shape.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name":{"name":"shell","arguments":{"command":"ls"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls flat MoE shape (no arguments wrapper)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\", \"command\": \"mkdir -p 2ddungeon\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p 2ddungeon", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("name") == null);
}

test "parseToolCalls flat shape with multiple top-level args" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"writeFile\", \"path\": \"/tmp/a\", \"content\": \"hi\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls flat shape ignores id/type metadata" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"id\": \"call_1\", \"type\": \"function\", \"name\": \"shell\", \"command\": \"ls\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("id") == null);
    try testing.expect(parsed.value.object.get("type") == null);
}

test "parseToolCalls repairs MoE missing-arguments-key shape" {
    const allocator = testing.allocator;
    // Real broken output observed from a model:
    // {"name": "shell",  {"command":"ls"}}  — `, {` instead of `, "arguments": {`
    const text = "<tool_call>\n{\"name\":  \"shell\",     {\"command\":\"ls -la\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls repairs MoE missing-opening-quote on arguments key" {
    const allocator = testing.allocator;
    // Real broken output observed from a model:
    // {"name": "shell", arguments": {"command":"mkdir -p src/app"}}
    // — missing the OPENING `"` on the `arguments` key.
    const text = "<tool_call>\n{\"name\": \"shell\", arguments\": {\"command\": \"mkdir -p src/app\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p src/app", parsed.value.object.get("command").?.string);
}

test "parseToolCalls recovers writeFile content with RAW newlines (small-model big-file escaping)" {
    const allocator = testing.allocator;
    // Small models writing a big file often emit literal newlines inside the
    // JSON `content` string instead of `\n` — strict JSON rejects raw control
    // bytes, so pre-fix the whole call was dropped and the file leaked as text.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"a.js\",\"content\":\"const x = 1;\nconst y = 2;\n\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("const x = 1;\nconst y = 2;\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with RAW tab" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"m.py\",\"content\":\"def f():\n\treturn 1\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("def f():\n\treturn 1", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with UNESCAPED inner quotes (HTML)" {
    const allocator = testing.allocator;
    // HTML/code with attribute quotes — `<meta charset="UTF-8">` — is the other
    // half of the escaping class: the model forgets to backslash the inner `"`.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"i.html\",\"content\":\"<meta charset=\"UTF-8\"><a href=\"x\">go</a>\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("i.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("<meta charset=\"UTF-8\"><a href=\"x\">go</a>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with BOTH raw newlines and unescaped quotes" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"p.html\",\"content\":\"<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers content with lone backslash (Windows path / regex)" {
    const allocator = testing.allocator;
    // `\U` and `\d` are invalid JSON escapes — strict parse rejects them.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"C:\\Users\\app.js\",\"content\":\"re.match(\\d+)\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("C:\\Users\\app.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("re.match(\\d+)", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers Gemma 4 call:NAME{json} with raw newline content" {
    const allocator = testing.allocator;
    // Gemma 4's JSON-first branch hits the same strict-parse wall on raw bytes.
    const text = "<|tool_call>call:writeFile{\"path\":\"g.txt\",\"content\":\"line1\nline2\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("line1\nline2", parsed.value.object.get("content").?.string);
}

test "parseToolCalls leaves valid escaped content untouched (no regression)" {
    const allocator = testing.allocator;
    // Already-correct JSON must pass straight through — the recovery path only
    // runs after strict parse fails, so this never touches looseRepair.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"ok.js\",\"content\":\"a\\nb\\t\\\"q\\\"\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a\nb\t\"q\"", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers truncated <function=writeFile> (max_tokens mid-content)" {
    const allocator = testing.allocator;
    // Live JFK-novel capture (2026-06-20): a Hermes-format writeFile dumped a
    // 19k-char novel into one <parameter=content> and hit the token cap before
    // any closing tag. Pre-fix the close_rel==null branch only tried JSON, so
    // the whole call was DROPPED and leaked as visible text. We must recover at
    // least the tool NAME (args may be empty — content is truncated, not
    // salvaged) so the client fires the chunk/append nudge instead of "use JSON".
    const text = "<tool_call>\n<function=writeFile>\n<parameter=content>\n# THE LION OF MASSACHUSETTS\n\nChapter 1. The young senator rose before dawn";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    // Args are valid JSON (empty object) — the parameter never closed.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
}

test "parseToolCalls recovers EOS-before-close-tag <function=> with full args" {
    const allocator = testing.allocator;
    // Bonus from the same fix: a Hermes call that closed </parameter></function>
    // but hit EOS before </tool_call> now recovers WITH its args, not just the
    // name (parseHermesToolCall reads the closed parameter).
    const text = "<tool_call>\n<function=shell>\n<parameter=command>ls -la</parameter>\n</function>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls recovers a bare <function=> call with NO <tool_call> opener" {
    const allocator = testing.allocator;
    // Live capture (soak, 2026-07-09, via Claude Code): the model
    // emitted the CLOSING </tool_call> but DROPPED the opening <tool_call>, so the
    // whole call arrived as `<function=Write>…</function></tool_call>`. The outer
    // scan triggers on the substring `<tool` — which `</tool_call>` does NOT
    // contain (it is `</too…`) — so parseHermesToolCall was never reached and the
    // entire Write (with its raw <function=/<parameter= markup) leaked into the
    // user-visible text. Same big-file-write dropped-delimiter CLASS as the
    // truncated-opener bugs; recover the call so the markup never leaks.
    const text = "\n\n<function=Write>\n<parameter=file_path>\n/tmp/index.html\n</parameter>\n" ++
        "<parameter=content>\n<!DOCTYPE html>\n<html></html>\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("Write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/index.html", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("<!DOCTYPE html>\n<html></html>", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall never emits invalid JSON on a malformed <parameter=> tag" {
    // Live soak capture (record 317): the model wrote `<parameter=limit=1` — using
    // `=1` instead of closing the tag with `>` and a value. The `>`-scan then
    // spilled the 'name' across a newline into `</parameter`, and the raw
    // (unescaped) name interpolation produced INVALID JSON args that flowed to the
    // client. Two guarantees: (1) args are ALWAYS valid JSON; (2) the well-formed
    // sibling `<parameter=path>` is still recovered.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=cat>\n<parameter=limit=1\n</parameter>\n<parameter=path>\n./parse.py\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cat", calls[0].name);
    // Hard invariant: valid JSON object, no matter how malformed the tag was.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    // The clean sibling parameter is recovered.
    try testing.expectEqualStrings("./parse.py", parsed.value.object.get("path").?.string);
}

test "parseHermesToolCall ends a function name whose > is missing at the line break (#748)" {
    const allocator = testing.allocator;
    const cases = [_]struct { text: []const u8, path: ?[]const u8 }{
        .{ .text = "<tool_call>\n<function=read_file\n</parameter>\n</function>\n</tool_call>", .path = null },
        .{ .text = "<tool_call>\n<function=read_file\n<parameter=path>\n./a.txt\n</parameter>\n</function>\n</tool_call>", .path = "./a.txt" },
        .{ .text = "<tool_call>\n<function=read_file<parameter=path>\n./a.txt\n</parameter>\n</function>\n</tool_call>", .path = "./a.txt" },
    };
    for (cases) |case| {
        const calls = (try parseToolCalls(allocator, case.text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("read_file", calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        if (case.path) |p| {
            try testing.expectEqualStrings(p, parsed.value.object.get("path").?.string);
        } else {
            try testing.expectEqual(@as(usize, 0), parsed.value.object.count());
        }
    }
}

test "parseHermesToolCall dedups a repeated <parameter=> name (no duplicate JSON key)" {
    // Live soak capture (record 443): the model emitted TWO
    // `<parameter=edits>` blocks in one call. Pre-fix that produced
    // `{"edits":…,"edits":…,"path":…}`, which std.json rejects with
    // error.DuplicateField — invalid JSON to the client. Dedup so the args are
    // always a valid object.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=edit>\n<parameter=edits>\nfirst\n</parameter>\n" ++
        "<parameter=edits>\nsecond\n</parameter>\n<parameter=path>\n./notes.md\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expectEqualStrings("./notes.md", parsed.value.object.get("path").?.string);
    // Exactly one `edits` key survives.
    try testing.expect(parsed.value.object.get("edits") != null);
}

test "parseHermesToolCall keeps leading/trailing whitespace inside a parameter value" {
    // The Hermes XML template frames a value with EXACTLY one newline on
    // each side (`<parameter=NAME>\nVALUE\n</parameter>` — pinned by the
    // the render test above), so that framing is the only thing the parser
    // may remove. It also did `std.mem.trim(u8, p_val, " ")`, which ate an
    // `old_string`/`new_string`'s own indentation: an Edit whose needle must
    // match the file "exactly, including indentation" then either failed, or —
    // worse — matched a DIFFERENT, un-indented occurrence mid-line and wrote
    // the replacement at the wrong nesting. Trailing whitespace is payload too
    // (a `new_string` that must end in two spaces, a markdown hard line break).
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/tmp/app.py\n</parameter>\n" ++
        "<parameter=old_string>\n        return None\n</parameter>\n" ++
        "<parameter=new_string>\n        return self.value  \n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/app.py", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("        return None", parsed.value.object.get("old_string").?.string);
    try testing.expectEqualStrings("        return self.value  ", parsed.value.object.get("new_string").?.string);
}

test "parseHermesToolCall strips only the ONE framing newline on each side" {
    // A value that legitimately begins or ends with a blank line keeps it: the
    // template adds one newline, not "any run of newlines". A multi-line
    // `content` written into a file is byte-exact or the file is wrong.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=write>\n<parameter=path>\n./a.md\n</parameter>\n" ++
        "<parameter=content>\n\n# Title\n\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("\n# Title\n", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall still types a space-padded scalar from its spelling" {
    // The other half: whitespace around a NUMBER or BOOLEAN carries no meaning,
    // so a one-line `<parameter=limit> 5 </parameter>` must still ship JSON 5,
    // not the string " 5 " — fixing the string class must not break the literal
    // class (`isJsonLiteral` never matches a padded token).
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=ls>\n<parameter=path>\n./src\n</parameter>\n" ++
        "<parameter=limit> 5 </parameter>\n<parameter=recursive>\n true \n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    try testing.expectEqual(true, parsed.value.object.get("recursive").?.bool);
}

test "parseHermesToolCall strips a CRLF framing pair as one unit" {
    // A model that emits Windows line endings frames the value with
    // `>\r\nVALUE\r\n</parameter>`. Dropping the "\n" half alone would leave a
    // stray CR at both ends of every value — the framing is one unit.
    const allocator = testing.allocator;
    const text = "<tool_call>\r\n<function=write>\r\n<parameter=content>\r\n" ++
        "  hi\r\n</parameter>\r\n</function>\r\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("  hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls does NOT treat prose mentioning <function=> as a call" {
    const allocator = testing.allocator;
    // The bare-<function=> fallback must not fire on prose. Without a matching
    // </function> AND a real <parameter=, parseHermesToolCall recovers nothing
    // useful — but guard against a false positive on a lone mention.
    const text = "To define one you write `<function=name>` in the Hermes format.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        defer {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        }
        // If it parsed anything, it must at least be well-formed JSON args —
        // never leak the raw prose as a tool name.
        try testing.expect(cs.len == 0);
    }
}

test "looseRepair does not fabricate a tool call from non-JSON prose" {
    const allocator = testing.allocator;
    // A stray `{` in prose must not become a tool call via the recovery path.
    const text = "Here is the plan {step one, step two} and we proceed.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    }
    try testing.expect(calls == null);
}

test "cannotOpenThink decides a short think head only once no opener can follow" {
    try testing.expect(cannotOpenThink("The"));
    try testing.expect(cannotOpenThink("\nOk"));
    try testing.expect(!cannotOpenThink("<"));
    try testing.expect(!cannotOpenThink("<thi"));
    try testing.expect(!cannotOpenThink(" \n"));
    try testing.expect(!cannotOpenThink("to"));
    try testing.expect(!cannotOpenThink("assis"));
    try testing.expect(!cannotOpenThink(""));
}

test "endsWithPartialThinkOpen holds back partial opener tails (pi leak repro)" {
    // The exact failure shape: `<|channel>` flushed as content before
    // "thought" arrived; the buffer tail was a partial opener at every step.
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|chan"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>thoug"));
    try testing.expect(endsWithPartialThinkOpen("prose <"));
    try testing.expect(endsWithPartialThinkOpen("prose <think"));
    try testing.expect(endsWithPartialThinkOpen("<th"));
    // Prose and HTML-ish tags keep flowing.
    try testing.expect(!endsWithPartialThinkOpen("just prose."));
    try testing.expect(!endsWithPartialThinkOpen("prose <table"));
    try testing.expect(!endsWithPartialThinkOpen("a > b"));
    try testing.expect(!endsWithPartialThinkOpen(""));
}

const test_tools_write_bash =
    \\[{"type":"function","function":{"name":"Write","description":"Write a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\ {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
;

test "inferBareJsonToolCalls maps bare-args fenced JSON to the unique matching tool (Claude Code capture)" {
    const allocator = testing.allocator;
    // Shape captured live from gemma-4-12b via Claude Code /v1/messages: thought
    // block, then a ```json fence holding ONLY the Write tool's arguments.
    const text = "<|channel>thought\nI will create this file using Write.<channel|>```json\n{\n  \"file_path\": \"/Users/example/mlx_info.html\",\n  \"content\": \"<h1>MLX</h1>\"\n}\n```\nI've created the file.";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("Write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/Users/example/mlx_info.html", parsed.value.object.get("file_path").?.string);
}

test "inferBareJsonToolCalls unfenced bare object at content start" {
    const allocator = testing.allocator;
    const text = "{\"command\": \"ls -la\"}";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("Bash", calls[0].name);
}

test "inferBareJsonToolCalls refuses ambiguous and non-matching objects" {
    const allocator = testing.allocator;
    // Two tools share the same key → ambiguous → null.
    const dup_tools =
        \\[{"type":"function","function":{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}},
        \\ {"type":"function","function":{"name":"b","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}}]
    ;
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"x\": \"1\"}", dup_tools)) == null);
    // Keys not in any tool's properties → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"zzz\": 1}", test_tools_write_bash)) == null);
    // Missing a required key → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"file_path\": \"a.txt\"}", test_tools_write_bash)) == null);
}

test "inferBareJsonToolCalls ignores JSON that does not lead the content" {
    const allocator = testing.allocator;
    // Example object mid-prose must never be promoted to a call.
    const text = "Here is an example of the payload you could send: {\"command\": \"ls\"} — adjust as needed.";
    try testing.expect((try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) == null);
}

test "parseToolCalls returns null for name-only object (no real args)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\"}\n</tool_call>";
    const result = try parseToolCalls(allocator, text);
    try testing.expect(result == null);
}

test "parseToolCalls marks raw-JSON fallback calls as inferred, tag calls as explicit" {
    const allocator = testing.allocator;
    // Tagged call → explicit (never schema-name-filtered).
    {
        const calls = (try parseToolCalls(allocator, "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(!calls[0].inferred);
    }
    // Bare raw-JSON object → inferred (flat shape included).
    {
        const calls = (try parseToolCalls(allocator, "prefix text {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None\"} suffix")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(calls[0].inferred);
        try testing.expectEqualStrings("George Washington", calls[0].name);
    }
}

test "filterInferredBySchema drops undeclared inferred calls, keeps declared and explicit ones" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    // Inferred + undeclared → dropped; slice becomes null.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "George Washington"),
            .arguments = try allocator.dupe(u8, "{\"num\":1}"),
            .inferred = true,
        };
        try testing.expect((try filterInferredBySchema(allocator, calls, tools)) == null);
    }
    // Inferred + declared → kept.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "write"),
            .arguments = try allocator.dupe(u8, "{\"path\":\"a\"}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
    }
    // Explicit + undeclared → kept (client feedback, not our guess); the
    // inferred sibling with an undeclared name is dropped and the slice shrinks.
    {
        var calls = try allocator.alloc(ParsedToolCall, 2);
        calls[0] = .{
            .name = try allocator.dupe(u8, "searchWeb"),
            .arguments = try allocator.dupe(u8, "{\"q\":\"zig\"}"),
            .inferred = false,
        };
        calls[1] = .{
            .name = try allocator.dupe(u8, "John Adams"),
            .arguments = try allocator.dupe(u8, "{\"num\":2}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
        try testing.expectEqualStrings("searchWeb", kept[0].name);
    }
}

test "toolNameIsDeclared: wrapped + flat forms; unparseable schema never drops" {
    const allocator = testing.allocator;
    const wrapped =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{}}}}]
    ;
    const flat =
        \\[{"name":"write","parameters":{"type":"object","properties":{}}}]
    ;
    try testing.expect(toolNameIsDeclared(allocator, wrapped, "write"));
    try testing.expect(toolNameIsDeclared(allocator, flat, "write"));
    try testing.expect(!toolNameIsDeclared(allocator, wrapped, "George Washington"));
    // Broken schema JSON → safe default TRUE (never drop on OUR parse failure).
    try testing.expect(toolNameIsDeclared(allocator, "not json", "anything"));
}

test "parseToolCalls multiple calls" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "func_a", "arguments": {"x": 1}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "func_b", "arguments": {"y": 2}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("func_a", calls[0].name);
    try testing.expectEqualStrings("func_b", calls[1].name);
}

test "parseToolCalls returns null for no tool calls" {
    const allocator = testing.allocator;
    const result = try parseToolCalls(allocator, "Hello, how can I help you?");
    try testing.expect(result == null);
}

// ── training-bias tool-call shapes ──
// the model emits tool calls in two non-standard variants the
// reference parser previously dropped on the floor:
//   1. Attribute form: `<tool_call name="X">{args}</tool_call>` — name as an
//      XML attribute, body is JUST the args object.
//   2. Mismatched closing tag: `<tool_call>{full json}</tool_request>` — open
//      tag is `tool_call`, close is `tool_request`. Hermes-style trained on a
//      mixed corpus.
// These tests are the regression guard. Without the fix they fail because
// the parser hard-matched on `<tool_call>…</tool_call>` exactly.

test "parseToolCalls XML attribute form: <tool_call name=\"X\">{args}</tool_call>" {
    const allocator = testing.allocator;
    const text = "<tool_call name=\"webSearch\">{\"query\":\"top open source inference applications\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("top open source inference applications", parsed.value.object.get("query").?.string);
}

test "parseToolCalls XML mismatched close: <tool_call>{json}</tool_request>" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "mkdir -p prisma src/lib"}}
        \\</tool_request>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p prisma src/lib", parsed.value.object.get("command").?.string);
}

test "parseToolCalls XML attribute + mismatched close combined" {
    // Worst-case combo observed live: attribute-style open, body is
    // just the args object (no top-level "name" key), and the closing tag
    // is `</tool_request>` instead of `</tool_call>`.
    const allocator = testing.allocator;
    const text = "<tool_call name=\"shell\">{\"command\":\"ls /tmp\"}</tool_request>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls /tmp", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Hermes double-brace body: <tool_call>{{json}}</tool_call>" {
    // Some models (seen on small instruct models echoing a Jinja `{{ }}`
    // example) wrap the args object in an extra brace layer. The body must still
    // parse into a tool call rather than leaking through as raw content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{{"name": "get_weather", "arguments": {"city": "Paris"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Paris", parsed.value.object.get("city").?.string);
}

test "parseToolCalls XML attribute single-quoted: <tool_call name='X'>" {
    // Some captures used single quotes around the attribute value. Same
    // semantics — accept both quote styles.
    const allocator = testing.allocator;
    const text = "<tool_call name='browse'>{\"action\":\"navigate\",\"url\":\"https://example.com\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
}

test "parseToolCalls XML plural-tag form: <tool_calls>{json}</tool_calls>" {
    // Real capture from a live chat session:
    //   <tool_calls>
    //   {"name": "shell", "arguments": {"command": "df -h /"}}
    //   </tool_calls>
    // Plural open + plural close, single JSON object inside. Without the
    // fix the parser hard-matches `<tool_call` (the suffix `s>` makes the
    // `>`-scan walk past the tool name), the body never extracts, and the
    // raw XML leaks into the chat bubble.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\{"name": "shell", "arguments": {"command": "df -h /"}}
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("df -h /", parsed.value.object.get("command").?.string);
}

test "parseToolCalls XML plural-tag with attribute: <tool_calls name=\"X\">{args}</tool_calls>" {
    const allocator = testing.allocator;
    const text = "<tool_calls name=\"webSearch\">{\"query\":\"hi\"}</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
}

test "hermes xml: the live <function=…>/<parameter=…> capture (package.json class)" {
    // Verbatim capture via pi (2026-08-12).
    // The model's OWN template dictates this dialect ("ONLY reply in the
    // following format"), so it is the checkpoint's contract, not a mangle.
    // The JSON-only reader mined the first balanced object out of the body —
    // the file CONTENT — and promoted its "name" key to the tool name, so pi
    // answered "Tool voxel-pagoda-garden not found" and the model looped.
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n" ++
        "/Users/example/pi-lfm/package.json\n</parameter>\n<parameter=content>\n" ++
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}\n" ++
        "</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("/Users/example/pi-lfm/package.json", o.get("path").?.string);
    // The value is RAW BYTES: a JSON file stays a STRING (typing by spelling
    // is what turned this call into `{"name": "voxel-pagoda-garden", …}`).
    // Types come from the schema at the request chokepoint, never from here.
    try testing.expectEqualStrings(
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}",
        o.get("content").?.string,
    );
}

test "hermes xml: truncated mid-parameter ships NAME + the completed params only" {
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.ts\n</parameter>\n" ++
        "<parameter=content>\nexport const half = ";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a.ts", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "issue #94: a container param carried as literal TEXT coerces to a real array/object" {
    // #94 reports this as an open gap on the strength of the doc comment
    // ("only SCALARS are touched"), which was stale — `coerceValueToType` has
    // had its array/object arm since 60ba5ec (2026-07-09), two weeks before
    // the issue was filed. Pinned as a TEST rather than left as a code read,
    // so the claim is settled by evidence and cannot silently regress.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"stats","parameters":{"type":"object","properties":{"nums":{"type":"array","items":{"type":"integer"}},"opts":{"type":"object"},"label":{"type":"string"}}}}}]
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "stats"),
        .arguments = try allocator.dupe(u8, "{\"nums\":\"[1, 2, 3]\",\"opts\":\"{\\\"grid\\\": true}\",\"label\":\"[not an array]\"}"),
    }};
    defer {
        allocator.free(calls[0].name);
        allocator.free(calls[0].arguments);
    }

    try coerceToolArgsToSchema(allocator, &calls, tools);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    const nums = o.get("nums").?.array;
    try testing.expectEqual(@as(usize, 3), nums.items.len);
    try testing.expectEqual(@as(i64, 2), nums.items[1].integer);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    // A STRING param whose content merely looks bracket-y is untouched — the
    // schema decides, never the spelling.
    try testing.expectEqualStrings("[not an array]", o.get("label").?.string);
}

test "parseToolCalls XML nested <tool name=\"X\"> inside <tool_calls> wrapper" {
    // Real capture from the model:
    //   <tool_calls>
    //   <tool name="webSearch">{"query": "top 10 ..."}</tool_calls>
    //   <tool name="webSearch">{"query": "best ..."}</tool_calls>
    //   </tool_calls>
    // Two inner calls, each opening with `<tool name="X">` and closing with
    // `</tool_calls>` (the model conflates open and close tag names). The
    // outer `<tool_calls>` wrapper has no content of its own — it's a
    // decorative parent.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="webSearch">{"query": "top 10 open source AI inference engines 2024 2025"}</tool_calls>
        \\<tool name="webSearch">{"query": "best open source AI inference engines vllm tensorrt mlx comparison"}</tool_calls>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Both inner calls must be recovered.
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    try testing.expectEqualStrings("webSearch", calls[1].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expect(std.mem.indexOf(u8, a0.value.object.get("query").?.string, "top 10") != null);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expect(std.mem.indexOf(u8, a1.value.object.get("query").?.string, "best") != null);
}

test "parseToolCalls bare <tool>{json}</tool>" {
    // Hypothetical minimal form — `<tool>` open with no attribute, JSON
    // body carries the name. Belt-and-suspenders for future model output.
    const allocator = testing.allocator;
    const text = "<tool>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}</tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls XML <parameters> wrapper inside <tool name=\"X\">" {
    // Real capture from the model:
    //   <tool_calls>
    //   <tool name="browse"><parameters>{"action":"navigate","url":"…"}</parameters></tool>
    //   …
    //   </tool_calls>
    // The args object is wrapped in <parameters>…</parameters> before being
    // placed inside the <tool> body. Strip that wrapper before JSON-parsing.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.com"}</parameters></tool>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.org"}</parameters></tool>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
    try testing.expectEqualStrings("https://example.com", a0.value.object.get("url").?.string);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expectEqualStrings("https://example.org", a1.value.object.get("url").?.string);
}

test "parseToolCalls XML trailing extra closing brace in args body" {
    // some models emit an extra `}` after the args object, likely a
    // training-data artifact. Be tolerant: find the first balanced JSON
    // object and ignore trailing garbage.
    const allocator = testing.allocator;
    const text =
        "<tool name=\"browse\"><parameters>{\"action\":\"navigate\",\"url\":\"https://dev.to/agdex_ai/post\"}}</parameters></tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
}

test "splitThinkBlock with think block before tool call" {
    const text = "<think>I need to call the calculator</think>\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 5}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("I need to call the calculator", result.reasoning_content.?);
    // `content` is VISIBLE text, so the call block is cut: parseToolCalls runs
    // on the raw output, and whatever it leaves behind is markup the client
    // must never render.
    try testing.expectEqualStrings("", result.content);
    // The keeping-markup variant is what the one parse-from-content caller
    // (/v1/messages non-streaming) uses, and it still hands over the block.
    const raw = splitThinkBlockKeepingMarkup(text, false, false);
    try testing.expect(std.mem.startsWith(u8, raw.content, "<tool_call>"));
}

test "splitThinkBlock with think block before regular content" {
    const text = "<think>Let me think about this</think>\n\nThe answer is 42.";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("Let me think about this", result.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 42.", result.content);
}

test "splitThinkBlock with empty think block" {
    const text = "<think>\n\n</think>\n\nJust content here.";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("Just content here.", result.content);
}

test "splitThinkBlock no think tags with tool call" {
    const text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("", result.content);
    try testing.expectEqualStrings(text, splitThinkBlockKeepingMarkup(text, false, false).content);
}

test "parseToolCalls with think block" {
    const allocator = testing.allocator;
    const text = "<think>reasoning</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("search", calls[0].name);
}

test "parseToolCalls raw JSON without tags" {
    const allocator = testing.allocator;
    const text = "{\"name\": \"get_time\", \"arguments\": {}}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
}

test "parseToolCalls Hermes format" {
    const allocator = testing.allocator;
    const text = "<tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // Should have {"location":"Tokyo"}
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "split content: Gemma 4 channel tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning here<channel|>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning<channel|>\nHello", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nstill thinking...", true, false).content);
}

test "splitThinkBlock Gemma 4 channel tags" {
    const result = splitThinkBlock("<|channel>thought\nmy reasoning<channel|>answer here", false, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock Gemma 4 thinking in progress" {
    const result = splitThinkBlock("<|channel>thought\npartial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock truncated mid-thinking does not leak channel tag" {
    // Truncation regression: the model emitted the Gemma 4 *content* channel
    // opener (`<|channel>\n…`) directly — no thought block and no `<channel|>`
    // close — then hit the output cap. The raw `<|channel>` control tag must
    // never reach visible content (it used to leak straight through).
    {
        const r = splitThinkBlock("<|channel>\nThe answer is 42.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("The answer is 42.", r.content);
    }
    // Bare dangling opener (cut off right after the tag) → nothing visible.
    {
        const r = splitThinkBlock("<|channel>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        const r = splitThinkBlock("<|channel>\n", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    // The template-injected-but-no-tags case must still pass through untouched.
    {
        const r = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("It is currently 8:15 AM PDT.", r.content);
    }
}

test "splitThinkBlock trailing unclosed thought opener does not leak (Gemma 12B pi regression)" {
    // Gemma 4 12B answers in prose, then opens a NEW thought channel right
    // before the turn ends (its known channel-thought tail behavior). The raw
    // `<|channel>thought` opener — and any unclosed thought text after it —
    // must never reach visible content; pi rendered the literal tag to users.
    {
        // Bare trailing opener, nothing after.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expect(r.reasoning_content == null);
    }
    {
        // Trailing opener with unclosed thought text → thought is reasoning.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write the file", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expectEqualStrings("I should now write the file", r.reasoning_content.?);
    }
    {
        // Same shape for the <think> family.
        const r = splitThinkBlock("Done.\n<think>wait, maybe I", true, false);
        try testing.expectEqualStrings("Done.", r.content);
        try testing.expectEqualStrings("wait, maybe I", r.reasoning_content.?);
    }
}

test "split content: trailing unclosed thought opener does not leak" {
    // The visible text must be truncated at a trailing unclosed opener (the
    // tag and dangling thought are reasoning, never content).
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought", true, false).content);
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write", true, false).content);
    try testing.expectEqualStrings("Done.", splitThinkBlock("Done.\n<think>hmm", true, false).content);
}

test "trimLeakedToolMarkup cuts at the first unparsed tool-call opener" {
    // Untouched when there is no marker (the overwhelmingly common case —
    // the returned slice must be the input itself, not a trimmed copy).
    try testing.expectEqualStrings("plain prose", trimLeakedToolMarkup("plain prose"));
    try testing.expectEqualStrings("ends in a tag <b>", trimLeakedToolMarkup("ends in a tag <b>"));

    // Every wrapper family cuts, and the trailing whitespace before the
    // marker goes with it.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"x\">"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<|tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_calls:opensource>"));

    // A marker with nothing before it leaves nothing behind — a bare marker
    // is not an answer (live: the model answered a question and then emitted a
    // lone `<｜DSML｜` with no call behind it).
    try testing.expectEqualStrings("", trimLeakedToolMarkup("\n\n<｜DSML｜\n"));

    // Earliest marker wins when several families appear.
    try testing.expectEqualStrings("a", trimLeakedToolMarkup("a<tool_call>b<｜DSML｜c"));
}

test "in-thought tool markup never rides out as reasoning" {
    // Live 2026-08-01 (a live agent session): the model
    // emitted a complete DSML call INSIDE its think block, closed the block,
    // then issued a different call in content. parseToolCalls works on the
    // post-think text, so the in-thought call is deliberately skipped — but
    // it rode out verbatim in reasoning_content, and the client round-tripped
    // it into every later prompt.
    const raw = "Let me check the directory.<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"listFiles\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">quake</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls></think>I'll build it.";
    const split = splitThinkBlock(raw, true, true);
    try testing.expectEqualStrings("Let me check the directory.", split.reasoning_content.?);
    try testing.expectEqualStrings("I'll build it.", split.content);

    // Reasoning that is NOTHING but markup reports no reasoning at all.
    const only_markup = splitThinkBlock("<｜DSML｜tool_calls>\n</｜DSML｜tool_calls></think>answer", true, true);
    try testing.expectEqual(@as(?[]const u8, null), only_markup.reasoning_content);
    try testing.expectEqualStrings("answer", only_markup.content);
}

test "a tool marker split across tokens is only visible once concatenated" {
    // The streaming tools flush holds a tail that looked like a tool call and
    // ships it when nothing parses. the model spells the DSML marker as `<` then
    // `｜DSML｜` (two tokens), so a per-token filter sees neither piece as a
    // marker and both went out as visible content (live 2026-08-01). The
    // flush concatenates first for exactly this reason.
    const pieces = [_][]const u8{ "My name is", "<", "｜DSML｜", "\n" };
    var joined: [64]u8 = undefined;
    var n: usize = 0;
    for (pieces) |p| {
        @memcpy(joined[n .. n + p.len], p);
        n += p.len;
    }
    for (pieces) |p| try testing.expectEqualStrings(p, trimLeakedToolMarkup(p));
    try testing.expectEqualStrings("My name is", trimLeakedToolMarkup(joined[0..n]));
}

test "splitThinkBlock re-opened thought opener right after close does not leak" {
    // 2026-06-19 live regression (gemma-4): the model closed its thought
    // channel and IMMEDIATELY re-opened a fresh one with nothing between, then
    // the turn ended. The leading-strip consumed the first closed block,
    // leaving the bare re-opened opener at pos 0 of the remainder — it must
    // still vanish, and `<|channel>thought` must never be mis-stripped as a
    // CONTENT channel (`<|channel>`), which leaked a glued "thought".
    {
        const r = splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false);
        try testing.expectEqualStrings("", r.content);
        try testing.expectEqualStrings("Let me plan.", r.reasoning_content.?);
    }
    {
        // <think> family equivalent.
        const r = splitThinkBlock("<think>plan</think>\n<think>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        // Content BETWEEN the close and the re-open must survive the cut.
        const r = splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false);
        try testing.expectEqualStrings("Ready.", r.content);
    }
}

test "split content: re-opened thought opener right after close does not leak" {
    // Same regression, content side — this is the exact form that reached
    // chat-history.json (`<|channel>thought\n` as the whole content).
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("Ready.", splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<think>plan</think>\n<think>", true, false).content);
}

test "parseToolCalls Gemma 4 format" {
    const allocator = testing.allocator;
    const text = "<|tool_call>call:get_weather{\"location\": \"Tokyo\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Gemma 4 with channel thinking" {
    const allocator = testing.allocator;
    const text = "<|channel>thought\nLet me check the weather<channel|>\n<|tool_call>call:get_weather{\"city\": \"Paris\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
}

test "parseToolCalls Gemma 4 truncated (no closing tag)" {
    const allocator = testing.allocator;
    // Model hit EOS before generating <tool_call|>
    const text = "<|tool_call>call:browse{action:<|\"|>browse<|\"|>,url:<|\"|>https://finance.yahoo.com<|\"|>}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    // Should have parsed at least the action argument
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("browse", parsed.value.object.get("action").?.string);
}

test "parseToolCalls Gemma 4 truncated mid-value drops the partial value" {
    const allocator = testing.allocator;
    // Model stopped mid-URL: no closing <|"|>, no `}`, no <tool_call|>. POLICY
    // (2026-07-14, aligned with the Hermes truncation rule): a value whose scan
    // runs to end-of-body without its terminator is a truncation FRAGMENT —
    // partial bytes must never ship as a real argument (a client executes them:
    // a fragmentary url browses garbage, a fragmentary content writes a corrupt
    // file "successfully"). The completed pair before it survives, and the tool
    // NAME is always recovered so the client's steering fires.
    const text = "<|tool_call>call:browse{action:<|\"|>navigate<|\"|>,url:<|\"|>https://finance.";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("navigate", parsed.value.object.get("action").?.string);
    // The truncated URL must be DROPPED, never shipped as partial bytes.
    try testing.expect(parsed.value.object.get("url") == null);
}

test "parseToolCalls Gemma 4 loop-stop truncated content dropped, name recovered (php.html capture)" {
    const allocator = testing.allocator;
    // Live 2026-07-14 (pi → gemma-4-26B-A4B-it-qat-4bit, plang/php.html): the
    // model repetition-looped INSIDE the write call's content string and the
    // scheduler's degenerate-tail-loop guard cut generation mid-word — the
    // buffered call arrived with NO closing delimiter, NO `}`, NO <tool_call|>,
    // and `path` never emitted at all. Pre-fix the salvage shipped
    // {"content":"<1.1 KB of loop garbage>"}; pi rejected the call on the
    // missing path and echoed the garbage back into the model's context.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <title>PHP</title>\n</head>\n<body>\n    <p>PHP is a widely-used general-purpose scripting language. It is a server-side scripting language, server-side scripting language, server-side scripting language, server-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("content") == null);
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 truncated content after a complete path keeps path, drops fragment" {
    const allocator = testing.allocator;
    // The ordering that would CORRUPT a file: `path` completed BEFORE the cut.
    // Pre-fix salvage = {path, <partial garbage content>} — a schema-valid call
    // the client executes, writing a fragment to a real path and reporting
    // success. The completed path survives (the client's missing-content error
    // steers a clean re-issue); the fragment never ships.
    const text = "<|tool_call>call:write{path:<|\"|>plang/php.html<|\"|>,content:<|\"|><!DOCTYPE html>\n<p>PHP is a server-side scripting language, server-side scripting language, server-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 bare rich value truncated at end-of-body is dropped" {
    const allocator = testing.allocator;
    // Both <|"|> delimiters dropped AND the generation truncated (no `}`, no
    // close tag anywhere in the rest): the run-to-end scan must not ship the
    // fragment as content.
    const text = "<|tool_call>call:write{content:<!DOCTYPE html>\n<h1>Hi</h1>\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 truncated edits container dropped, complete path kept" {
    const allocator = testing.allocator;
    // Cut INSIDE a container arg (pi-style edits array): the list is a fragment
    // — more edits were coming — and applying a partial edit list is the same
    // fragmentary-work hazard as partial content. The whole container drops;
    // the completed scalar before it survives.
    const text = "<|tool_call>call:edit{path:<|\"|>a.sh<|\"|>,edits:[{oldText:<|\"|>x<|\"|>,newText:<|\"|>y<|\"|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.sh", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("edits") == null);
}

test "parseToolCalls Gemma 4 EOS after a complete pair keeps the pair (no over-drop)" {
    const allocator = testing.allocator;
    // Truncation landing BETWEEN pairs (after a completed value + comma) must
    // not over-drop: only the value the cut landed INSIDE is a fragment.
    const text = "<|tool_call>call:write{path:<|\"|>a.html<|\"|>,";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.html", parsed.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 EOS right after the key colon drops the key" {
    const allocator = testing.allocator;
    // `path:` then nothing — pre-fix this shipped {"path":""} and a client
    // would create a file with an EMPTY name (or reject on a bogus value).
    // An absent value is a fragment; drop the key, recover the name.
    const text = "<|tool_call>call:write{path:";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 JSON-style truncated string value is dropped" {
    const allocator = testing.allocator;
    // Same truncation class through the JSON-quoted spelling: the closing `"`
    // never arrived. Pre-fix the verbatim copy emitted an UNCLOSED JSON string
    // (invalid args, caught only by the {}-fallback safety net); the complete
    // first pair must survive and the fragment must drop.
    const text = "<|tool_call>call:write{\"path\":\"plang/php.html\",\"content\":\"<!DOCTYPE html>\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 unterminated string must not swallow the closing brace (pi write regression)" {
    const allocator = testing.allocator;
    // Gemma 4 12B emitted a write call whose LAST string value was missing
    // its closing <|"|> delimiter and carried a stray markdown backtick
    // before the args object's `}`. The unterminated-string scan ran to end
    // of body, so the parsed path was literally "mlx_pi1.html`}" — and pi
    // created a file with that name on disk.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html><html></html><|\"|>,path:<|\"|>mlx_pi1.html`}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx_pi1.html", parsed.value.object.get("path").?.string);

    // Plain missing-delimiter variant (no backtick): same brace exclusion.
    const text2 = "<|tool_call>call:write{content:<|\"|>X<|\"|>,path:<|\"|>out.html}<tool_call|>";
    const calls2 = (try parseToolCalls(allocator, text2)).?;
    defer {
        for (calls2) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls2);
    }
    const parsed2 = try std.json.parseFromSlice(std.json.Value, allocator, calls2[0].arguments, .{});
    defer parsed2.deinit();
    try testing.expectEqualStrings("out.html", parsed2.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with custom delimiters" {
    const allocator = testing.allocator;
    // Model mixes JSON-style quoted keys with <|"|> delimiters — JSON parse fails,
    // convertGemma4ArgsToJson must strip quotes from keys to produce valid JSON.
    const text =
        \\<|tool_call>call:shell{"command":<|"|>ls -la<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with bare values" {
    const allocator = testing.allocator;
    // Model uses quoted keys but bare (non-JSON, non-delimited) values
    const text = "<|tool_call>call:shell{\"command\":ls -la}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 dropped OPENING delimiter on big content (E4B live)" {
    const allocator = testing.allocator;
    // Verbatim shape captured from gemma-4-e4b-it-4bit writing a full HTML page
    // (test_tool_matrix_small.sh): the model dropped the OPENING <|"|> on
    // `content` but kept the CLOSING one, so the bare-value scan cut content at
    // the FIRST comma (inside `<meta ... content="width=device-width, ...">`)
    // and shredded the rest of the markup into bogus keys → invalid args. The
    // closing <|"|> (followed by `,path`) is the real boundary.
    const text = "<|tool_call>call:write_file{content:<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html><|\"|>,path:<|\"|>mars.html<|\"|>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mars.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings(
        "<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html>",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls Gemma 4 dropped BOTH delimiters on single content field" {
    const allocator = testing.allocator;
    // Both delimiters dropped, content is the only/last field → run to the
    // object's closing brace.
    const text = "<|tool_call>call:write_file{content:<h1>Hi</h1>\n<p>a, b, c</p>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("<h1>Hi</h1>\n<p>a, b, c</p>", parsed.value.object.get("content").?.string);
}

test "convertGemma4ArgsToJson nested braces in value" {
    const allocator = testing.allocator;
    // Content value contains JSON-like structures (e.g., JavaScript code with objects)
    const input = "{path:<|\"|>server.js<|\"|>,content:<|\"|>const x = {a: 1, b: {c: 2}};<|\"|>}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    // Verify it produces valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("server.js", parsed.value.object.get("path").?.string);
    const content = parsed.value.object.get("content").?.string;
    try testing.expect(std.mem.indexOf(u8, content, "{a: 1") != null);
}

test "convertGemma4ArgsToJson bare array value" {
    const allocator = testing.allocator;
    // Bare array value should be preserved via brace-matching
    const input = "{stops:[\"Rome\",\"Venice\",\"Athens\"],price:1200}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.get("stops").?.array.items.len);
    try testing.expectEqual(@as(i64, 1200), parsed.value.object.get("price").?.integer);
}

test "convertGemma4ArgsToJson bare nested object value" {
    const allocator = testing.allocator;
    // Bare nested object should be preserved via brace-matching
    const input = "{name:test,config:{\"port\":3000,\"host\":\"localhost\"}}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try testing.expectEqual(@as(i64, 3000), config.get("port").?.integer);
}

// Issue #16: nested objects/arrays that themselves use Gemma 4's custom
// `<|"|>`-delimited string format (or bare keys) must be converted recursively,
// not passed through verbatim.
test "convertGemma4ArgsToJson issue#16 nested array of custom strings" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array:[<|"|>foo<|"|>,<|"|>bar<|"|>]}
    ;
    const expected =
        \\{"nested_array":["foo","bar"]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object custom format" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object:{foo:<|"|>bar<|"|>}}
    ;
    const expected =
        \\{"nested_object":{"foo":"bar"}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object inside array" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object_in_array:[{foo:<|"|>bar<|"|>}]}
    ;
    const expected =
        \\{"nested_object_in_array":[{"foo":"bar"}]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested array inside object" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array_in_object:{foo:[<|"|>bar<|"|>]}}
    ;
    const expected =
        \\{"nested_array_in_object":{"foo":["bar"]}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

// Issue #16, real capture: this is the verbatim output of gemma-4-e4b-it-8bit
// for a tool with nested-object + nested-array params. The model nests its
// custom `<|"|>`/bare-key format, so the client must still receive valid,
// fully-converted JSON arguments (no surviving <|"|> delimiters / bare keys).
test "parseToolCalls Gemma 4 nested object + array args (issue#16 real capture)" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:send_notification{message:<|"|>Deploy complete<|"|>,metadata:{priority:<|"|>high<|"|>,tags:[<|"|>ci<|"|>,<|"|>release<|"|>]},recipients:[<|"|>alice<|"|>,<|"|>bob<|"|>]}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("send_notification", calls[0].name);
    // The arguments must be valid JSON (would fail to parse with raw <|"|> in them).
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Deploy complete", parsed.value.object.get("message").?.string);
    const metadata = parsed.value.object.get("metadata").?.object;
    try testing.expectEqualStrings("high", metadata.get("priority").?.string);
    const tags = metadata.get("tags").?.array;
    try testing.expectEqual(@as(usize, 2), tags.items.len);
    try testing.expectEqualStrings("ci", tags.items[0].string);
    try testing.expectEqualStrings("release", tags.items[1].string);
    const recipients = parsed.value.object.get("recipients").?.array;
    try testing.expectEqual(@as(usize, 2), recipients.items.len);
    try testing.expectEqualStrings("alice", recipients.items[0].string);
    try testing.expectEqualStrings("bob", recipients.items[1].string);
}

// Gemma 4 12B leaks a trailing ':' into the function name (`call:shell:{…}`),
// producing an unresolvable tool name `shell:`. The parser must strip it so the
// client resolves a real tool instead of looping on "Unknown tool 'shell:'".
test "parseToolCalls Gemma 4 strips trailing colon from tool name" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:shell:{"command":<|"|>ls<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
}

test "sanitizeToolName strips trailing colons and whitespace" {
    try testing.expectEqualStrings("shell", sanitizeToolName("shell:"));
    try testing.expectEqualStrings("shell", sanitizeToolName("  shell  "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell : "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell::"));
    try testing.expectEqualStrings("cwd", sanitizeToolName("cwd"));
    try testing.expectEqualStrings("", sanitizeToolName(":"));
}

test "isJsonLiteral" {
    try testing.expect(isJsonLiteral("true"));
    try testing.expect(isJsonLiteral("false"));
    try testing.expect(isJsonLiteral("null"));
    // Objects/arrays are now handled by brace-matching in convertGemma4ArgsToJson, not isJsonLiteral
    try testing.expect(!isJsonLiteral("{\"key\":1}"));
    try testing.expect(!isJsonLiteral("[1,2,3]"));
    try testing.expect(isJsonLiteral("42"));
    try testing.expect(isJsonLiteral("3.14"));
    try testing.expect(!isJsonLiteral("hello"));
    try testing.expect(!isJsonLiteral(""));
}

test "appendJsonString escapes special chars" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "hello \"world\"\nnew\\line");
    try testing.expectEqualStrings("\"hello \\\"world\\\"\\nnew\\\\line\"", buf.items);
}

test "appendJsonString empty string" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "");
    try testing.expectEqualStrings("\"\"", buf.items);
}

test "appendJsonString escapes ALL control characters (2026-06-11 ESC-byte regression)" {
    // Live failure: a tool result carrying raw ANSI terminal codes (ESC = 0x1B
    // from `\x1b[?25l`) passed through unescaped, nlohmann::json rejected the
    // messages JSON inside jinja_render_chat, and the prompt silently fell
    // back to the wrong-format fallbackFormatChat — gemma-4-31b then
    // hallucinated entire conversations. JSON strings must escape EVERY
    // control char < 0x20, not just \n \r \t.
    const allocator = testing.allocator;

    var input: [0x20]u8 = undefined;
    for (&input, 0..) |*c, i| c.* = @intCast(i);

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, &input);

    // No raw control byte may survive in the serialized form.
    for (buf.items) |c| {
        try testing.expect(c >= 0x20);
    }

    // And the result must round-trip through a strict JSON parser byte-exact.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(&input, parsed.value.string);
}

test "renderChatTemplate: a tool-aware template with a generic role header renders tool turns natively" {
    // The template names no 'tool' literal: every non-assistant turn is
    // `<|im_start|>{{ role }}`. It renders tool CALLS, so tool results are its
    // own turns; a template that renders no tool calls keeps the rewrite.
    const a = testing.allocator;
    const generic = "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}" ++
        "{%- if m.tool_calls is defined and m.tool_calls -%}{%- for tc in m.tool_calls -%}<tool_call>{{ tc.function.name }}</tool_call>{%- endfor -%}{%- endif -%}<|im_end|>{%- endfor -%}";
    const calls = [_]ToolCall{.{ .id = "c1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "Weather?" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls },
        .{ .role = "tool", .content = "18C", .tool_call_id = "c1" },
    };
    var config = ChatConfig{ .chat_template = generic, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
    const native = try renderChatTemplate(a, &messages, &config, null, null, true, false);
    defer a.free(native);
    try testing.expect(std.mem.indexOf(u8, native, "<|im_start|>tool\n18C<|im_end|>") != null);
    try testing.expect(std.mem.indexOf(u8, native, "<tool_response>") == null);

    config.chat_template = "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{%- endfor -%}";
    const plain = try renderChatTemplate(a, &messages, &config, null, null, true, false);
    defer a.free(plain);
    try testing.expect(std.mem.indexOf(u8, plain, "<|im_start|>user\n<tool_response>\n18C\n</tool_response>") != null);
}

test "renderChatTemplate: tool result with raw ANSI escapes still renders via Jinja" {
    // Regression for the 2026-06-11 pi/gemma-4-31b session: the third request
    // was the first whose history held a tool result with a raw ESC byte (the
    // interactive `npx sv create` output). The Jinja render failed on the
    // unescaped messages JSON and silently downgraded to fallbackFormatChat's
    // Gemma-2/3 `<start_of_turn>` text format — not special tokens for Gemma
    // 4, so generation never stopped at turn end. The render must SUCCEED via
    // the template (template marker present, fallback marker absent).
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'tool' -%}<|tool_response>{{ message['content'] }}<turn|>
        \\{%- else -%}<|turn>{{ message['role'] }}
        \\{{ message['content'] }}{% if message.tool_calls %}{% for tc in message.tool_calls %}<|tool_call>{{ tc.function.name }}<tool_call|>{% endfor %}{% endif %}<turn|>
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{# tools referenced so no synthesis kicks in: {{ tools }} #}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "bash", .arguments = "{\"command\": \"npx sv create .\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "make me a sveltekit app" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        // Verbatim shape from the live failure: hide-cursor ANSI code + prompt UI.
        .{ .role = "tool", .content = "\x1b[?25l\u{2502}\n\u{25c6}  Which template would you like?", .tool_call_id = "tc_0" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "<|tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "Which template") != null);
    // Fallback format must NOT have been used.
    try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
}

test "renderChatTemplate: |min picks the smaller element, |max the larger" {
    // Guards against a min/max implementation that just returns first/last.
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\{{ [3, 1, 2]|min }}/{{ [3, 1, 2]|max }}
        \\{%- endfor -%}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "1/3") != null);
}

test "renderChatTemplate: a namespace attribute named like a dict method is the attribute" {
    // `ns.items` resolved to dict.items(), `+` raised, and the render fell back to the generic format.
    const allocator = testing.allocator;
    const tpl =
        \\{%- set ns = namespace(items=[]) -%}
        \\{%- for message in messages -%}{%- set ns.items = ns.items + [message.content] -%}{%- endfor -%}
        \\ITEMS={{ ns.items | join(',') }}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{ .{ .role = "user", .content = "hi" }, .{ .role = "assistant", .content = "there" } };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ITEMS=hi,there") != null);
}

test "templateConsumesEffort: the real templates that READ the effort word" {
    // A template that acts on `reasoning_effort`: the word is the lever and an
    // effort-derived TOKEN budget on top is pure truncation (see the fn doc).
    try testing.expect(templateConsumesEffort("{%- set e = reasoning_effort|default('high') %}"));

    // A template that never reads it (Gemma 4's included): thinking is a bool there.
    try testing.expect(!templateConsumesEffort("{%- if enable_thinking %}<think>\n{%- endif %}"));
    try testing.expect(!templateConsumesEffort(""));
}

test "fillOptionalToolDefKeys: a tool with no description/parameters gets defaults, a complete one is left alone" {
    // OpenAI's schema makes `description` optional and a no-arg tool needs no
    // `parameters`, but a template that dumps all three through `tojson`
    // RAISES on Undefined and silently falls back to ChatML with no tools
    // preamble. Every family that dumps `tool | tojson` wholesale is one
    // missing key from the same class, so the defaults are filled for ALL templates.
    const allocator = testing.allocator;
    // A COMPLETE declaration re-emits nothing: null means the caller keeps the
    // client's own bytes, so no request that works today is re-serialized
    // (key order, number formatting and unicode escaping all stay theirs).
    const complete =
        \\[{"type":"function","function":{"name":"note","description":"d","parameters":{"type":"object","properties":{}}}}]
    ;
    try testing.expect(try fillOptionalToolDefKeys(allocator, complete) == null);
    // Only the missing keys are added, and only to the function that lacks them.
    const partial =
        \\[{"type":"function","function":{"name":"a","description":"d"}},{"type":"function","function":{"name":"b","description":"e","parameters":{"type":"object"}}}]
    ;
    const filled = (try fillOptionalToolDefKeys(allocator, partial)).?;
    defer allocator.free(filled);
    try testing.expect(std.mem.indexOf(u8, filled, "\"parameters\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"d\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"e\"") != null);
}

test "toolChoiceInstruction: one line per forced choice" {
    const allocator = testing.allocator;
    const any = try toolChoiceInstruction(allocator, .any);
    defer allocator.free(any);
    try testing.expectEqualStrings("\nYou MUST call one of the available functions. Do not respond with text.", any);
    const named = try toolChoiceInstruction(allocator, .{ .name = "calculator" });
    defer allocator.free(named);
    try testing.expectEqualStrings("\nYou MUST call the function \"calculator\". Do not respond with text.", named);
}

test "toolsDeclare: a name in the OpenAI, Anthropic or Responses tool shape" {
    const body =
        \\{"chat":[{"type":"function","function":{"name":"calculator"}}],"flat":[{"name":"probe_ping"}]}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const chat = parsed.value.object.get("chat");
    const flat = parsed.value.object.get("flat");
    try testing.expect(toolsDeclare(chat, "calculator"));
    try testing.expect(!toolsDeclare(chat, "bogus"));
    try testing.expect(toolsDeclare(flat, "probe_ping"));
    try testing.expect(!toolsDeclare(null, "calculator"));
}

test "forcedToolOpener: the XML dialect commits a call in the template's own spelling" {
    const allocator = testing.allocator;
    const hermes = "{{- '<tool_call>\\n<function=' + name + '>' }}";
    const any = (try forcedToolOpener(allocator, hermes, .any)).?;
    defer allocator.free(any);
    try testing.expectEqualStrings("<tool_call>\n<function", any);
    const named = (try forcedToolOpener(allocator, hermes, .{ .name = "calculator" })).?;
    defer allocator.free(named);
    try testing.expectEqualStrings("<tool_call>\n<function=calculator>\n", named);
    try testing.expect(try forcedToolOpener(allocator, "<function=example>{}</function>", .any) == null);
}

test "serializeExtraContext with thinking enabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, true);
    defer allocator.free(result);
    // Should contain enable_thinking:true
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":true") != null);
}

test "serializeExtraContext with thinking disabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, false);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":false") != null);
}

test "serializeExtraContext: a template without a family effort vocabulary keeps high/no_think" {
    const allocator = testing.allocator;
    var other = ChatConfig{
        .chat_template = "{{ reasoning_effort }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const on = try serializeExtraContext(allocator, &other, true);
    defer allocator.free(on);
    try testing.expect(std.mem.indexOf(u8, on, "\"reasoning_effort\":\"high\"") != null);
    const off = try serializeExtraContext(allocator, &other, false);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"reasoning_effort\":\"no_think\"") != null);
}

test "serializeMessagesJson basic" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Parse it back to verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.array.items.len);
    try testing.expectEqualStrings("user", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("Hello", parsed.value.array.items[0].object.get("content").?.string);
}

test "serializeMessagesJson with tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"location\":\"Tokyo\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should contain tool_calls array
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "get_weather") != null);
}

test "serializeMessagesJson carries assistant reasoning_content" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const asst = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("two plus two is four", asst.get("reasoning_content").?.string);
    // A message without the field must not carry the key at all — templates
    // gate on `is string`, and an explicit null would still pass `is defined`.
    try testing.expect(parsed.value.array.items[0].object.get("reasoning_content") == null);
}

test "assistant history reasoning_content reaches a template that reads it (history round-trip)" {
    // The chat_template.jinja persists reasoning across turns: history
    // assistant messages render <think>{message.reasoning|reasoning_content}</think>.
    // When the parse layer drops the field, every history turn renders the
    // empty <think></think> nothink signature and the model stops thinking
    // from turn 2 of a session (live 2026-07-29, pi agent: one
    // reasoning delta on the 2-msg turn, zero across the next 13k chunks).
    // Template fragment mirrors the assistant-history branch.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "{%- for message in messages -%}" ++
            "{%- if message.role == 'assistant' -%}" ++
            "{%- set rc = '' -%}" ++
            "{%- if message.reasoning is string -%}{%- set rc = message.reasoning -%}" ++
            "{%- elif message.reasoning_content is string -%}{%- set rc = message.reasoning_content -%}" ++
            "{%- endif -%}" ++
            "<think>{{ rc }}</think>{{ message.content }}" ++
            "{%- endif -%}{%- endfor -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, true, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "<think>two plus two is four</think>4") != null);
}

test "serializeMessagesJson embeds valid-JSON arguments as object (not string)" {
    // Required by templates that do `tool_call.arguments|items`.
    // Without this, the Jinja `items` filter fails on a string and the server
    // falls back to ChatML — losing the `<think>\n` injection that primes the
    // model's reasoning + close-tag behavior on the next turn.
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Object form: ..."arguments":{"command":"date"}...
    // String form (rejected): ..."arguments":"{\"command\":\"date\"}"...
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{\"command\":\"date\"}") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"{") == null);
}

test "serializeMessagesJson keeps malformed arguments as string" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "not valid json" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"not valid json\"") != null);
}

test "serializeMessagesJson tool response has tool_call_id and content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Must contain tool_call_id
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"call_1\"") != null);
    // Must contain the response content
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Must NOT contain tool_responses (templates handle role:"tool" natively)
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "serializeMessagesJson full tool calling conversation" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_99_0", .name = "shell", .arguments = "{\"command\":\"echo hello\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Run echo hello" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "hello", .tool_call_id = "call_99_0" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 4), parsed.value.array.items.len);

    // Verify roles
    try testing.expectEqualStrings("system", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("user", parsed.value.array.items[1].object.get("role").?.string);
    try testing.expectEqualStrings("assistant", parsed.value.array.items[2].object.get("role").?.string);
    try testing.expectEqualStrings("tool", parsed.value.array.items[3].object.get("role").?.string);

    // Verify assistant has tool_calls
    const assistant = parsed.value.array.items[2].object;
    try testing.expect(assistant.get("tool_calls") != null);

    // Verify tool message has tool_call_id and content, no tool_responses
    const tool_msg = parsed.value.array.items[3].object;
    try testing.expectEqualStrings("call_99_0", tool_msg.get("tool_call_id").?.string);
    try testing.expectEqualStrings("hello", tool_msg.get("content").?.string);
    try testing.expect(tool_msg.get("tool_responses") == null);
}

test "serializeMessagesJson multiple parallel tool responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "readFile", .arguments = "{\"path\":\"a.txt\"}" },
        .{ .id = "call_2", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file content", .tool_call_id = "call_1" },
        .{ .role = "tool", .content = "Fri Apr 4", .tool_call_id = "call_2" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();

    // Both tool messages should have tool_call_id and content
    const tool1 = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("call_1", tool1.get("tool_call_id").?.string);
    try testing.expectEqualStrings("file content", tool1.get("content").?.string);

    const tool2 = parsed.value.array.items[2].object;
    try testing.expectEqualStrings("call_2", tool2.get("tool_call_id").?.string);
}

test "serializeMessagesJson tool response with empty content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"mkdir test\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should have tool_call_id but no tool_responses
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

// ── Fallback formatter tests ──

test "fallbackFormatChat Gemma tool response uses user role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Run ls" },
        .{ .role = "tool", .content = "file1.txt" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<eos>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Gemma tool results should be in user turn
    try testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>user\nTool result: file1.txt") != null);
    // Should NOT have <start_of_turn>tool (invalid role)
    try testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>tool") == null);
}

test "fallbackFormatChat: tools with a system message write the tool block after the system turn" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "Be brief." },
        .{ .role = "user", .content = "Run ls" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<eos>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const result = try fallbackFormatChat(allocator, &messages, &config, tools_json, null, false);
    defer allocator.free(result);

    const expected = "<start_of_turn>system\nBe brief.<end_of_turn>\n" ++
        "<start_of_turn>user\n" ++
        "You are a helpful assistant with access to the following functions. To call a function, respond with a JSON object in the following format:\n" ++
        "<tool_call>\n" ++
        "{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n" ++
        "</tool_call>\n" ++
        "\n" ++
        "Available functions:\n" ++
        tools_json ++ "<end_of_turn>\n" ++
        "<start_of_turn>user\nRun ls<end_of_turn>\n" ++
        "<start_of_turn>model\n";
    try testing.expectEqualStrings(expected, result);

    // Without tools the system turn stands alone, and a conversation with no system message still leads with the tool block.
    const plain = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(plain);
    try testing.expectEqualStrings("<start_of_turn>system\nBe brief.<end_of_turn>\n<start_of_turn>user\nRun ls<end_of_turn>\n<start_of_turn>model\n", plain);
    const no_sys = try fallbackFormatChat(allocator, messages[1..], &config, tools_json, null, false);
    defer allocator.free(no_sys);
    try testing.expect(std.mem.startsWith(u8, no_sys, "<start_of_turn>user\nYou are a helpful assistant with access"));
    try testing.expect(std.mem.endsWith(u8, no_sys, "<start_of_turn>user\nRun ls<end_of_turn>\n<start_of_turn>model\n"));
}

// ── Tool fallback (XML-tool + any template lacking tools support) ──

test "templateReferencesToolRole detects role branches" {
    try testing.expect(templateReferencesToolRole("{% if message['role'] == 'tool' %}"));
    try testing.expect(templateReferencesToolRole("{% if message.role == \"tool\" %}"));
    try testing.expect(!templateReferencesToolRole("{% if message['role'] == 'user' %}{{ message['tool_calls'] }}{% endif %}"));
    try testing.expect(!templateReferencesToolRole(""));
}

test "messagesHaveToolContent" {
    const tc = [_]ToolCall{.{ .id = "x", .name = "n", .arguments = "{}" }};
    const m_tool = [_]Message{.{ .role = "tool", .content = "x" }};
    const m_tc = [_]Message{.{ .role = "assistant", .content = "", .tool_calls = &tc }};
    const m_plain = [_]Message{.{ .role = "user", .content = "hi" }};
    try testing.expect(messagesHaveToolContent(&m_tool));
    try testing.expect(messagesHaveToolContent(&m_tc));
    try testing.expect(!messagesHaveToolContent(&m_plain));
}

test "synthesizeToolFallbackMessages: inject tool prompt with no existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // System content carries the tool prompt + the tools_json blob.
    try testing.expect(std.mem.indexOf(u8, out[0].content, "tool_call") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("List files", out[1].content);
}

test "synthesizeToolFallbackMessages: merge tool prompt into existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // Existing system content preserved, tool prompt appended.
    try testing.expect(std.mem.startsWith(u8, out[0].content, "You are helpful."));
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
}

test "synthesizeToolFallbackMessages: rewrite role:tool to role:user" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "do it" },
        .{ .role = "tool", .content = "result: 42" },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, true, false);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "result: 42") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_response>") != null);
}

test "synthesizeToolFallbackMessages: rewrite assistant tool_calls into <tool_call> blocks" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "ls" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, false, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("assistant", out[1].role);
    try testing.expect(out[1].tool_calls == null); // moved into content
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_call>") != null);
}

test "renderChatTemplate: XML-tool-style template gets tool fallback applied" {
    const allocator = testing.allocator;
    // Minimal template that mirrors an XML-tool template: has system/user/assistant, no tools, no role:tool.
    const tpl =
        \\<bos>
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'system' -%}{{ message['content'] }}{%- elif message['role'] == 'user' -%}<U>{{ message['content'] }}</U><A>{%- elif message['role'] == 'assistant' -%}{{ message['content'] }}<EOS>{%- endif -%}
        \\{%- endfor -%}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"run a shell command\"}}]";
    const messages = [_]Message{
        .{ .role = "user", .content = "list my files" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, false);
    defer allocator.free(rendered);
    // Tool prompt should have been injected as a system message.
    try testing.expect(std.mem.indexOf(u8, rendered, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "shell") != null);
    // User message still rendered after the synthesized system content.
    try testing.expect(std.mem.indexOf(u8, rendered, "list my files") != null);
}

// ── XML-tool agentic-flow compositions (round-trip + multi-turn + edge cases) ──

test "XML-tool agentic round-trip: model-generated <tool_call> parses + replays" {
    // Simulates the full agent loop step:
    //   1. Model emits a `<tool_call>` block in its generated content.
    //   2. parseToolCalls extracts the structured ToolCall(s).
    //   3. The agent builds the next prompt with role:assistant(tool_calls=...) +
    //      role:tool(content=result).
    //   4. synthesizeToolFallbackMessages rewrites both for the XML-tool template
    //      (assistant tool_calls → <tool_call> block, tool → user
    //      <tool_response>).
    //   5. The same call should round-trip: the rewritten assistant content
    //      should contain a parseable <tool_call> matching the original.
    const allocator = testing.allocator;

    // 1. Pretend the model wrote this:
    const generated =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "date"}}
        \\</tool_call>
    ;
    const parsed_calls = (try parseToolCalls(allocator, generated)).?;
    defer {
        for (parsed_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(parsed_calls);
    }
    try testing.expectEqual(@as(usize, 1), parsed_calls.len);
    try testing.expectEqualStrings("shell", parsed_calls[0].name);

    // 2. Agent builds the next-turn history with the parsed call + the result.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "tc_0", .name = parsed_calls[0].name, .arguments = parsed_calls[0].arguments },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "what is the date?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "tc_0" },
    };

    // 3. Run through the tool fallback. inject_tool_prompt=true mirrors how
    // renderChatTemplate calls it when the template lacks tool support.
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Should produce: system(tool-prompt), user, assistant(<tool_call>), user(<tool_response>).
    try testing.expectEqual(@as(usize, 4), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expectEqualStrings("user", out[3].role);

    // 4. The assistant's rewritten content must contain a <tool_call> block
    //    that re-parses back to the same name + arguments. This is the
    //    closed-loop test.
    try testing.expect(out[2].tool_calls == null);
    const reparsed = (try parseToolCalls(allocator, out[2].content)).?;
    defer {
        for (reparsed) |t| {
            allocator.free(t.name);
            allocator.free(t.arguments);
        }
        allocator.free(reparsed);
    }
    try testing.expectEqual(@as(usize, 1), reparsed.len);
    try testing.expectEqualStrings("shell", reparsed[0].name);
    try testing.expect(std.mem.indexOf(u8, reparsed[0].arguments, "date") != null);

    // 5. The tool-result rewrite must wrap the result in <tool_response>...
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "</tool_response>") != null);
}

test "XML-tool multi-turn agent history: every role correctly rewritten" {
    // Realistic 6-message agent transcript:
    //   1. user (initial question)
    //   2. assistant tool_call #1 (date)
    //   3. tool result #1
    //   4. assistant intermediate reasoning + tool_call #2 (ls)
    //   5. tool result #2
    //   6. user follow-up
    // The synthesizer must:
    //   - inject the tool-prompt system message at the top
    //   - rewrite each role:tool into role:user with <tool_response>
    //   - rewrite each role:assistant with tool_calls into <tool_call> blocks
    //   - preserve user messages untouched
    //   - preserve assistant intermediate text (msg #4 has BOTH content and tool_calls)
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc1 = [_]ToolCall{.{ .id = "t1", .name = "shell", .arguments = "{\"command\":\"date\"}" }};
    const tc2 = [_]ToolCall{.{ .id = "t2", .name = "shell", .arguments = "{\"command\":\"ls\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "what files exist as of today?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "t1" },
        .{ .role = "assistant", .content = "Got the date. Now listing files.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "README.md\nsrc/\ntests/", .tool_call_id = "t2" },
        .{ .role = "user", .content = "summarize" },
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Expected layout: system + 6 rewritten messages.
    try testing.expectEqual(@as(usize, 7), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);

    // Original index 0 (user) → out[1] user, unchanged.
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("what files exist as of today?", out[1].content);

    // Original index 1 (assistant tool_call) → out[2] assistant with <tool_call> in content, no tool_calls.
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expect(out[2].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "\"command\":\"date\"") != null);

    // Original index 2 (tool) → out[3] user with <tool_response>.
    try testing.expectEqualStrings("user", out[3].role);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);

    // Original index 3 (assistant content + tool_call) → out[4] preserves intermediate text.
    try testing.expectEqualStrings("assistant", out[4].role);
    try testing.expect(out[4].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "Got the date. Now listing files.") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "\"command\":\"ls\"") != null);

    // Original index 4 (tool) → out[5] user with <tool_response>.
    try testing.expectEqualStrings("user", out[5].role);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "README.md") != null);

    // Original index 5 (user follow-up) → out[6] user, unchanged.
    try testing.expectEqualStrings("user", out[6].role);
    try testing.expectEqualStrings("summarize", out[6].content);
}

test "tool fallback: null tools_json with inject_tool_prompt does not synthesize" {
    // When a request has NO tools (chat without tools param), the synthesizer
    // must NOT inject a system message even with inject_tool_prompt=true.
    // Only role:tool / assistant tool_calls rewrites still fire (in case the
    // template still lacks support and history was injected by a previous turn).
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "hello" },
    };
    // tools_json = null → no synthesis.
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, true, true, true);
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqualStrings("user", out[0].role);
    try testing.expectEqualStrings("hello", out[0].content);
}

test "streamShouldBufferForTools: empty + plain prose" {
    try testing.expect(!streamShouldBufferForTools(""));
    try testing.expect(!streamShouldBufferForTools("hello world"));
    try testing.expect(!streamShouldBufferForTools("Let me look at this."));
}

test "streamShouldBufferForTools: trailing partial prefixes of <tool" {
    // The exact regression from the screenshot: model emits `<tool` as a
    // single BPE token, server must buffer instead of streaming it as
    // content (so the next token can either complete the open or be flushed
    // together as harmless prose).
    try testing.expect(streamShouldBufferForTools("<"));
    try testing.expect(streamShouldBufferForTools("<t"));
    try testing.expect(streamShouldBufferForTools("<to"));
    try testing.expect(streamShouldBufferForTools("<too"));
    try testing.expect(streamShouldBufferForTools("<tool"));
    try testing.expect(streamShouldBufferForTools("Let me look at the CLI help:\n<tool"));
}

test "streamShouldBufferForTools: trailing partial prefixes of <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|"));
    try testing.expect(streamShouldBufferForTools("<|t"));
    try testing.expect(streamShouldBufferForTools("<|too"));
    try testing.expect(streamShouldBufferForTools("<|tool"));
    try testing.expect(streamShouldBufferForTools("<|tool_"));
    try testing.expect(streamShouldBufferForTools("<|tool_c"));
    try testing.expect(streamShouldBufferForTools("<|tool_cal"));
}

test "streamShouldBufferForTools: bare <tool> open (XML form)" {
    try testing.expect(streamShouldBufferForTools("<tool>"));
    try testing.expect(streamShouldBufferForTools("<tool>{\"name\":\"shell\""));
    try testing.expect(streamShouldBufferForTools("<tool name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("hello <tool>{}"));
}

test "streamShouldBufferForTools: canonical <tool_call*> opens" {
    try testing.expect(streamShouldBufferForTools("<tool_call>"));
    try testing.expect(streamShouldBufferForTools("<tool_call>{\"name\":"));
    try testing.expect(streamShouldBufferForTools("<tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<tool_request>"));
    try testing.expect(streamShouldBufferForTools("prose first then <tool_call>"));
}

test "streamShouldBufferForTools: Gemma 4 <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|tool_call>"));
    try testing.expect(streamShouldBufferForTools("hello <|tool_call name=\"x\">"));
}

test "streamShouldBufferForTools: false positives — <toolkit> etc." {
    // `<toolkit>` and `<toolbar>` have `<tool` as a prefix but the char
    // after is `k` / `b` — not a tag terminator. Must NOT buffer.
    try testing.expect(!streamShouldBufferForTools("here is <toolkit>"));
    try testing.expect(!streamShouldBufferForTools("here is <toolbar>"));
    try testing.expect(!streamShouldBufferForTools("HTML: <toolset>"));
    // But a later real `<tool>` should still trip detection:
    try testing.expect(streamShouldBufferForTools("HTML: <toolset> then <tool>{"));
}

test "streamShouldBufferForTools: raw JSON shape" {
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\",\"arguments\":{}}"));
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\""));
    // Just `{` or JSON without `"name"` shouldn't false-trigger.
    try testing.expect(!streamShouldBufferForTools("{"));
    try testing.expect(!streamShouldBufferForTools("{\"foo\":1}"));
}

test "streamThinkGate: template-opened thinking holds tag-free prose" {
    // The Claude Code leak class: with thinking on and the opener injected by
    // the template, the model's prose has NO tags — it must still be held,
    // never flushed as visible text.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The user is asking about their system specs.", true, false));
    // …and split exactly when the close tag arrives.
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("The user is asking.\n</think>\n\nI don't have direct access.", true, false));
}

test "streamThinkGate: after the split, prose flushes (hidden-answer guard)" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("The visible answer streams normally.", true, true));
}

test "streamThinkGate: explicit openers hold even with thinking off or closed" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Sure.<|channel>thought\nlet me reconsider", true, true));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("<think>hmm", false, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("<|channel>thought\nplan<channel|>done", false, false));
}

test "streamThinkGate: partial opener at the buffer tail holds" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The answer is 391.\n<|channel>", false, false));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Done. <thi", false, false));
}

test "streamThinkGate: plain prose with thinking off flushes" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("17 × 23 = 391.", false, false));
}

test "streamThinkGate: a prompt-opened block is held even with thinking OFF" {
    // This template renders `…assistant\n<think>` UNCONDITIONALLY - there is no
    // enable_thinking branch in its template, so the model always reasons and
    // the opener is never in the OUTPUT. The gate saw tag-free prose with
    // thinking off and flushed the entire chain-of-thought as the answer
    // (live 2026-08-04, streaming only — non-streaming strips it because the
    // `</think>` is present by the time the text is split).
    //
    // Whether the prompt ends inside a think block is a fact about the
    // RENDERED BYTES, not about our request flag, which is why it needs its
    // own argument rather than riding `enable_thinking`.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, true));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2("The user wants 5 words.</think>Hi there, how are you?", false, false, true));

    // Containment, both directions. Same buffers with the prompt flag FALSE
    // behave exactly as before — every other family's template renders the
    // closed `<think></think>` signature (or no tag) when thinking is off, so
    // `promptTailOpensThink` is false for them and this term can never fire.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, false));
    // And once the block has closed, a prompt-opened flag must not re-hold the
    // visible answer — that would hide every reply after the first.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("Hi there, how are you?", false, true, true));
}

test "unstreamedReasoning: only the remainder, and never a resend" {
    try testing.expectEqualStrings("cde", unstreamedReasoning("abcde", 2).?);
    try testing.expectEqualStrings("abcde", unstreamedReasoning("abcde", 0).?);
    // Nothing new since the last delta.
    try testing.expect(unstreamedReasoning("abcde", 5) == null);
    // The split SHRANK (a tool marker appeared mid-thought and the leaked-markup
    // cut took the tail). Deltas cannot be retracted, so send nothing more —
    // resending from the top would duplicate the whole thought.
    try testing.expect(unstreamedReasoning("ab", 5) == null);
    try testing.expect(unstreamedReasoning("", 0) == null);
}

test "streamTailIsReasoning: an unopened block was content all along" {
    // Gemma 4 with enable_thinking:true answering "391" directly — no opener
    // in the prompt (its template renders a bare `<|turn>model\n`) and none in
    // the output. Streamed, the whole answer arrived as reasoning_content with
    // EMPTY content while non-streaming returned it as content (live
    // 2026-08-04). Content is the correct reading: nothing ever opened.
    try testing.expect(!streamTailIsReasoning(true, false, false));

    // A template-opened block that never closed IS reasoning — the truncated
    // thought must not leak into content (some templates pre-inject the opener).
    try testing.expect(streamTailIsReasoning(true, true, false));
    // A model-emitted opener is the same evidence (Gemma's `<|channel>thought`),
    // even with no prompt opener.
    try testing.expect(streamTailIsReasoning(true, false, true));
    // Already closed and split ⇒ the tail is the visible answer.
    try testing.expect(!streamTailIsReasoning(false, true, true));
}

test "promptTailOpensThink does not match Gemma's channel opener" {
    // The containment boundary for the prompt-opened-think stream arm: it
    // detects the close with `</think>` ONLY, which is sound exactly because
    // the flag can only be set by a `<think`-family opener. Gemma 4 opens with
    // `<|channel>thought` and closes with `<channel|>`, so it never enters
    // that arm and its streaming is byte-unchanged.
    //
    // If `thinkOpenTagLenAt` ever grows the channel opener, this test fails —
    // and `server.zig`'s arm then needs `<channel|>` in its close check, or a
    // Gemma turn would be held forever waiting for a tag that never arrives.
    try testing.expect(!promptTailOpensThink("<|turn>model\n<|channel>thought\n"));
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    // The `<think>` family it DOES match.
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
}

test "streamThinkGate: the 3-arg form is the 4-arg form with no prompt opener" {
    // The old signature is kept as the no-prompt-opener case so every existing
    // call site and test pins the SAME behavior it always did.
    const cases = [_][]const u8{
        "The user is asking about their system specs.",
        "The user is asking.\n</think>\n\nI don't have direct access.",
        "<think>hmm",
        "17 × 23 = 391.",
        "The answer is 391.\n<|channel>",
    };
    for (cases) |c| {
        for ([_]bool{ true, false }) |think| {
            for ([_]bool{ true, false }) |closed| {
                try testing.expectEqual(streamThinkGate(c, think, closed), streamThinkGate2(c, think, closed, false));
            }
        }
    }
}

test "parseToolCalls: self-closing <tool name=... arguments=... />" {
    const allocator = testing.allocator;
    // Clean form. We use the literal byte sequence the model actually emits —
    // unescaped `"` inside the attribute value. That's XML-invalid but
    // perfectly JSON-parseable once we snap to the inner `{…}`.
    const text =
        \\<tool name="shell" arguments="{"command": "ls -la"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: XML broken-quote self-closing fragment (regression repro)" {
    // Verbatim capture from the failing turn in test_agent_stop_repro.py.
    // The model opened arguments with `"`, closed with `'`, embedded
    // unescaped `"` inside the JSON, and finished with `'/>`. The parser
    // must still recover the JSON object and produce a single tool call.
    const allocator = testing.allocator;
    const text =
        "\n\n<tool_calls>\n" ++
        "<tool name=\"shell\" arguments=\"{\"command\": \"ls -la && cat package.json 2>/dev/null || echo 'no package.json yet'\"}'/>\n" ++
        "</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    // Args should be the balanced JSON object — starts with `{` and ends `}`.
    try testing.expect(calls[0].arguments.len > 0);
    try testing.expectEqual(@as(u8, '{'), calls[0].arguments[0]);
    try testing.expectEqual(@as(u8, '}'), calls[0].arguments[calls[0].arguments.len - 1]);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: multiple self-closing tool tags in one response" {
    const allocator = testing.allocator;
    const text =
        \\<tool name="cwd" arguments="{"path": "src"}"/>
        \\<tool name="listFiles" arguments="{"recursive": "true"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expectEqualStrings("listFiles", calls[1].name);
}

test "parseToolCalls: canonical <tool_call>JSON</tool_call> still parses" {
    // Regression guard — adding the self-closing branch must not break the
    // canonical Hermes form.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls: empty body + attribute-args + explicit </tool_call> close" {
    // Verbatim capture from the user's failed chat-history.json. The model
    // emitted args via XML attribute and used `></tool_call>` (not `/>`)
    // as the terminator, with empty body. Without explicit handling this
    // shape parses as "empty body" → no JSON to extract → no tool call.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool_call name="cwd" arguments="{"path": "/Users/example/.mlx-serve/workspace"}"></tool_call>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, ".mlx-serve/workspace") != null);
}

test "parseToolCalls: <tool_call>{JSON} truncated before </tool_call>" {
    // The model emitted `<tool_call>` + well-formed args JSON, then hit
    // EOS before completing the close marker. Output looks like:
    //   "\n\n<tool_call>\n{\"name\": \"writeFile\", \"arguments\":{...}}\n</tool_cal"
    // Previously this fell through to the orphan branch and got silently
    // dropped — that's the second failure mode the agent-stops repro
    // surfaced. We snap to a balanced JSON body and parse it.
    const allocator = testing.allocator;
    const text = "\n\n<tool_call>\n" ++
        \\{"name": "writeFile", "arguments": {"path": "vite.config.js", "content": "ok"}}
        ++ "\n</tool_cal";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"content\"") != null);
}

test "parseToolCalls: mismatched </tool_action> close still parses" {
    // Verbatim capture (2026-06-10 pi html turn 2): opened
    // with <tool_call>, closed with the hallucinated </tool_action>. The
    // edit call must parse; pre-fix it leaked into visible text and the
    // agent executed nothing.
    const allocator = testing.allocator;
    const text = "<tool_call>\n" ++
        \\{"name": "edit", "arguments": {"path":"mlx.html", "edits":[{"oldText": "  </ul>\n</body>", "newText": "  </ul>\n  <button onclick=\"alert('Hello from MLX')\">Click me</button>\n</body>"}]}
        ++ "\n</tool_action>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
}

test "parseToolCalls: fenced JSON array of parallel calls parses all" {
    const allocator = testing.allocator;
    const text = "```json\n[\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris, France\"}},\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo, Japan\"}}\n]\n```";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "Paris, France") != null);
    try testing.expect(std.mem.indexOf(u8, calls[1].arguments, "Tokyo, Japan") != null);
}

test "parseToolCalls: XML-element tool form (tool_name + arg children)" {
    // Verbatim capture (2026-06-10, MLX Core agent chat): the
    // tool name and each argument arrive as XML child elements — no JSON.
    const allocator = testing.allocator;
    const text = "Let me check the available disk space on this device.\n\n<tool_calls>\n<tool_name>shell</tool_name>\n<command>df -h / | grep -v \"Filesystem\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "df -h / | grep -v \\\"Filesystem\\\"") != null);
}

test "parseToolCalls: XML-element-TAG form (<tool_NAME> with arg children)" {
    // Verbatim capture (2026-06-10 pi html turn 2): the tool
    // name rides in the tag itself — <tool_read>, <tool_edit> — with each
    // argument as an XML child element. Nested elements (the <edits> body)
    // stay verbatim as the arg's string value, matching what working models
    // send pi for the same tool.
    const allocator = testing.allocator;
    const text = "Let me read the current file first.\n\n<tool_read>\n<path>mlx.html</path>\n</tool_read>Now I'll add a button:\n\n<tool_edit>\n<path>mlx.html</path>\n<edits>\n  <oldText>    <h1>MLX</h1></oldText>\n  <newText>    <h1>MLX</h1>\n    <button onclick=\"alert('hi')\">Say Hello</button></newText>\n</edits>\n</tool_edit>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("read", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
    try testing.expectEqualStrings("edit", calls[1].name);
    // Args must be valid JSON with the nested XML preserved as a string value.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx.html", parsed.value.object.get("path").?.string);
    const edits = parsed.value.object.get("edits").?.string;
    try testing.expect(std.mem.indexOf(u8, edits, "<oldText>") != null);
    try testing.expect(std.mem.indexOf(u8, edits, "Say Hello") != null);
}

test "parseToolCalls: XML-element-TAG form with JSON args body" {
    // Verbatim-shape capture (2026-06-10 pi html, second
    // sampling): same name-in-tag form but the body is a bare JSON args
    // object instead of XML elements — `<tool_write>\n{…}\n</tool_write>`.
    const allocator = testing.allocator;
    const text = "Here's the HTML page:\n\n<tool_write>\n{\"path\": \"/tmp/ws/mlx.html\", \"content\": \"<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<body>\\n</body>\\n</html>\"}\n</tool_write>\n\npage ready";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/ws/mlx.html", parsed.value.object.get("path").?.string);
    try testing.expect(std.mem.indexOf(u8, parsed.value.object.get("content").?.string, "<!DOCTYPE html>") != null);
}

test "parseToolCalls: <tool_NAME> with prose body containing braces is NOT a tool call" {
    // Prose with incidental {braces} inside an unknown tool-ish tag must not
    // produce arguments — the JSON-body variant requires the body to BE the
    // args object, not merely contain one.
    const allocator = testing.allocator;
    const text = "<tool_summary>The config {a: 1} was applied.</tool_summary>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: hallucinated <tool_output> result tag is NOT a tool call" {
    // Verbatim capture (same session, turn 1): the model invented
    // a tool RESULT without calling anything. A plain-text body must never
    // produce a call to a tool named "output".
    const allocator = testing.allocator;
    const text = "Here's the page I created for you:\n\n<tool_output>Page ready: mlx.html</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <tool_output> with element children is still NOT a tool call" {
    // A hallucinated result tag can carry element-shaped content (e.g. an
    // echoed HTML fragment). Result-ish names are denylisted outright.
    const allocator = testing.allocator;
    const text = "<tool_output>\n<status>ok</status>\n</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: XML-element form without a tool_name is NOT a tool call" {
    // Prose-ish markup inside a <tool_calls> wrapper must not half-execute.
    const allocator = testing.allocator;
    const text = "<tool_calls>\n<note>this has no tool name element</note>\n</tool_calls>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: prose list array is NOT a tool call" {
    // All-or-nothing: an array whose elements aren't {name, arguments}
    // objects must pass through as text.
    const allocator = testing.allocator;
    const text = "[\"apples\", \"oranges\", \"pears\"]";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <toolkit> is not a tool tag" {
    // The self-closing scan must not false-fire on HTML-ish prose. `<toolkit>`
    // has `<tool` as a prefix but the terminator (`k`) isn't valid, so the
    // outer scan rejects it before parseSelfClosingToolTag even runs.
    const allocator = testing.allocator;
    const text = "Here's a doc about <toolkit> and <toolbar>. No actual tool call.";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

// ── Schema-aware argument coercion (value-spelling type-inference class) ──

/// The Edit tool as Claude Code declares it (OpenAI shape, post
/// `buildOpenAIToolsJson`): `replace_all` is a BOOLEAN, the rest are strings.
const edit_tools_json_test =
    \\[{"type":"function","function":{"name":"Edit","description":"Edit a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}}]
;

fn parseArgsObj(allocator: std.mem.Allocator, args: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, args, .{});
}

test "coerceToolArgsToSchema: Python-style False on a boolean param becomes JSON false" {
    // VERBATIM capture, 2026-07-09, via Claude Code on
    // ~/.mlx-serve/logs/mlx-serve-11234.log:109471. The model emits the Hermes
    // XML parameter form with Python's `False`; parseHermesToolCall's
    // isJsonLiteral only knows lowercase, so it shipped the STRING "False" and
    // Claude Code rejected every Edit with
    //   "The parameter `replace_all` type is expected as `boolean` but provided as `string`"
    // — six times in a row, until the model gave up and rewrote the whole file.
    const allocator = testing.allocator;
    const raw =
        "\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/Users/example/doom/index.html\n</parameter>\n" ++
        "<parameter=old_string>\n  <script src=\"game.js\"></script>\n</parameter>\n" ++
        "<parameter=new_string>\n  <script src=\"game.js\" type=\"module\"></script>\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(false, v.bool);
    // Same capture, the whitespace half of the class: `old_string` must match
    // the file "exactly, including indentation", so the two leading spaces the
    // model sent are payload. Trimming them either failed the edit or matched a
    // DIFFERENT, un-indented occurrence mid-line and rewrote it at the wrong
    // nesting; coercion leaves a declared string alone, so the parser is the
    // only layer that can preserve them.
    try testing.expectEqualStrings(
        "  <script src=\"game.js\"></script>",
        parsed.value.object.get("old_string").?.string,
    );
    try testing.expectEqualStrings(
        "  <script src=\"game.js\" type=\"module\"></script>",
        parsed.value.object.get("new_string").?.string,
    );
}

test "coerceToolArgsToSchema: a STRING param spelled `false` stays a string" {
    // The inverse half of the same class: isJsonLiteral guesses the type from
    // the value's SPELLING, so a code edit whose old_string is literally the
    // token `false` was shipped as JSON `false` (boolean) — "expected string,
    // provided boolean". The schema is the only thing that can disambiguate.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\nfalse\n</parameter>\n<parameter=new_string>\n42\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const old = parsed.value.object.get("old_string").?;
    try testing.expect(old == .string);
    try testing.expectEqualStrings("false", old.string);
    const new = parsed.value.object.get("new_string").?;
    try testing.expect(new == .string);
    try testing.expectEqualStrings("42", new.string);
}

test "coerceToolArgsToSchema: quoted boolean in a JSON body is coerced too" {
    // Same class, different producer: the model emits well-formed JSON but
    // quotes the boolean. Strict parse succeeds, so no repair path ever runs.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\"," ++
        "\"old_string\":\"a\",\"new_string\":\"b\",\"replace_all\":\"true\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(true, v.bool);
}

test "coerceToolArgsToSchema: unknown tool / unknown key / undecidable value pass through" {
    const allocator = testing.allocator;
    // Unknown tool name → args untouched.
    {
        const raw = "<tool_call>{\"name\":\"Nope\",\"arguments\":{\"replace_all\":\"False\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        try testing.expect(parsed.value.object.get("replace_all").? == .string);
    }
    // Known tool, boolean param, value that is not a recognizable boolean →
    // left alone so the client's validation error stays honest.
    {
        const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"replace_all\":\"maybe\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const v = parsed.value.object.get("replace_all").?;
        try testing.expect(v == .string);
        try testing.expectEqualStrings("maybe", v.string);
    }
}

/// pi's `edit` tool, verbatim from its own schema
/// (@earendil-works/pi-coding-agent dist/core/tools/edit.js — typebox
/// `editSchema`). Two facts the tests below lean on: `path` is REQUIRED at the
/// TOP level, and the array's item schema declares ONLY oldText/newText — it
/// never declares `path`. That asymmetry is what makes a buried `path`
/// provably misplaced rather than a judgment call.
const pi_edit_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]},"description":"One or more targeted replacements."}},"required":["path","edits"]}}}]
;

const edit_array_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object"}},"dry_run":{"type":"boolean"}},"required":["path","edits"]}}}]
;

test "coerceToolArgsToSchema: an ARRAY param passed through Hermes XML is not left a string" {
    // Live capture class (15 occurrences in one session): `<parameter=edits>` holds
    // a JSON array, but parseHermesToolCall's isJsonLiteral only recognizes
    // scalars, so the whole array shipped as a STRING. Strict clients reject it
    // ("edits: want array, got str"); lenient ones silently mis-execute.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/a.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"const a = 1;\", \"newText\": \"const a = 2;\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("const a = 2;", edits.array.items[0].object.get("newText").?.string);
    // sibling scalars still coerce/pass through
    try testing.expectEqualStrings("/tmp/a.js", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: identical duplicate keys inside a container param still coerce; conflicting ones do not" {
    // Issue #402: Code's ask_user_question array arrived as a string whose
    // items repeated `header` with the same value; strict parse rejects the
    // duplicate, the tolerant re-emit keeps it, and the string reached the
    // client unchanged. Identical duplicates carry no ambiguity; conflicting
    // ones do, and guessing a value is worse than leaving the string alone.
    const allocator = testing.allocator;
    const cases = [_]struct { inner: []const u8, want_array: bool }{
        .{ .inner = "[{\"header\":\"Scope\",\"question\":\"Which scope?\",\"header\":\"Scope\",\"options\":[{\"label\":\"Small\"}]}]", .want_array = true },
        .{ .inner = "[{\"header\":\"Scope\",\"header\":\"Size\"}]", .want_array = false },
    };
    for (cases) |c| {
        const raw = try std.fmt.allocPrint(allocator, "<tool_call>\n<function=edit>\n<parameter=edits>\n{s}\n</parameter>\n</function>\n</tool_call>", .{c.inner});
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const edits = parsed.value.object.get("edits").?;
        if (c.want_array) {
            try testing.expect(edits == .array);
            try testing.expectEqualStrings("Scope", edits.array.items[0].object.get("header").?.string);
            try testing.expectEqual(@as(usize, 3), edits.array.items[0].object.count());
        } else {
            try testing.expect(edits == .string);
        }
    }
}

test "coerceToolArgsToSchema: a STRING param whose content is JSON stays a string" {
    // The inverse guard for arrays/objects: `old_string` may legitimately BE the
    // text `[1,2]`. Only the schema decides.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\n[1,2]\n</parameter>\n<parameter=new_string>\n{\"k\":1}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("old_string").? == .string);
    try testing.expectEqualStrings("[1,2]", parsed.value.object.get("old_string").?.string);
    try testing.expect(parsed.value.object.get("new_string").? == .string);
}

test "coerceToolArgsToSchema: already-conforming args are byte-identical (no-op guard)" {
    // The auto-correct layer must never touch a good call. Property: when every
    // declared type already matches, `arguments` comes out byte-for-byte the same.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\",\"old_string\":\"a\"," ++
        "\"new_string\":\"b\",\"replace_all\":true}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

// ── Auto-correct safety properties (deterministic fuzz) ─────────────────────
// The repair layer is only ever allowed to touch input that is actually broken.
// These generate well-formed tool calls whose values deliberately SPELL other
// JSON types ("false", "42", "[1,2]", "null") — exactly the strings the
// value-spelling inference used to mis-type — and assert round-trip identity.

fn jsonValueEql(a: std.json.Value, b: std.json.Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => a.bool == b.bool,
        .integer => a.integer == b.integer,
        .float => a.float == b.float,
        .number_string => std.mem.eql(u8, a.number_string, b.number_string),
        .string => std.mem.eql(u8, a.string, b.string),
        .array => blk: {
            if (a.array.items.len != b.array.items.len) break :blk false;
            for (a.array.items, b.array.items) |x, y| if (!jsonValueEql(x, y)) break :blk false;
            break :blk true;
        },
        .object => blk: {
            if (a.object.count() != b.object.count()) break :blk false;
            var it = a.object.iterator();
            while (it.next()) |e| {
                const other = b.object.get(e.key_ptr.*) orelse break :blk false;
                if (!jsonValueEql(e.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}

/// Value strings chosen to collide with every JSON literal spelling the
/// tag-format parsers used to guess from.
const adversarial_strings = [_][]const u8{
    "false",      "true",         "False",   "True",    "null",   "None",
    "42",         "-7",           "3.14",    "0",       "1",      "",
    "[1,2]",      "{\"k\":1}",    "nan",     "inf",     "  ",     "yes",
    "a\"b",  "line\nline", "tab\there", "back\\slash",
    "café ☕",
    "0x1f",  "{not json",  "[unclosed", "1e400",       "00",   "+5",
    "-",
};

test "fuzz: a conforming tool call round-trips byte-identical through parse+coerce" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE_1234);
    const rand = prng.random();

    const types = [_][]const u8{ "string", "integer", "number", "boolean", "array", "object" };

    var iter: usize = 0;
    while (iter < 400) : (iter += 1) {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Build a random schema + a CONFORMING args object for it.
        var props: std.json.ObjectMap = .empty;
        var args: std.json.ObjectMap = .empty;
        const nprops = 1 + rand.uintLessThan(usize, 5);
        var p: usize = 0;
        while (p < nprops) : (p += 1) {
            const key = try std.fmt.allocPrint(arena, "p{d}", .{p});
            const want = types[rand.uintLessThan(usize, types.len)];

            var prop: std.json.ObjectMap = .empty;
            try prop.put(arena, "type", .{ .string = want });
            try props.put(arena, key, .{ .object = prop });

            const v: std.json.Value = if (std.mem.eql(u8, want, "string"))
                .{ .string = adversarial_strings[rand.uintLessThan(usize, adversarial_strings.len)] }
            else if (std.mem.eql(u8, want, "integer"))
                .{ .integer = @as(i64, @intCast(rand.uintLessThan(u32, 1000))) - 500 }
            else if (std.mem.eql(u8, want, "number"))
                .{ .float = 1.5 }
            else if (std.mem.eql(u8, want, "boolean"))
                .{ .bool = rand.boolean() }
            else if (std.mem.eql(u8, want, "array")) blk: {
                var arr = std.json.Array.init(arena);
                try arr.append(.{ .string = "x" });
                try arr.append(.{ .integer = 1 });
                break :blk .{ .array = arr };
            } else blk: {
                var o: std.json.ObjectMap = .empty;
                try o.put(arena, "k", .{ .string = "v" });
                break :blk .{ .object = o };
            };
            try args.put(arena, key, v);
        }

        var func: std.json.ObjectMap = .empty;
        try func.put(arena, "name", .{ .string = "t" });
        var params: std.json.ObjectMap = .empty;
        try params.put(arena, "type", .{ .string = "object" });
        try params.put(arena, "properties", .{ .object = props });
        try func.put(arena, "parameters", .{ .object = params });
        var tool: std.json.ObjectMap = .empty;
        try tool.put(arena, "type", .{ .string = "function" });
        try tool.put(arena, "function", .{ .object = func });
        var tools_arr = std.json.Array.init(arena);
        try tools_arr.append(.{ .object = tool });
        const tools_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .array = tools_arr }, .{});

        // Serialize as a canonical Hermes JSON call — always VALID input.
        const args_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = args }, .{});
        const raw = try std.fmt.allocPrint(arena, "<tool_call>{{\"name\":\"t\",\"arguments\":{s}}}</tool_call>", .{args_json});

        const calls = (try parseToolCalls(allocator, raw)) orelse {
            std.debug.print("\n[fuzz iter {d}] valid tool call did not parse\n  {s}\n", .{ iter, raw });
            return error.FuzzFailed;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }

        const before = try arena.dupe(u8, calls[0].arguments);
        try coerceToolArgsToSchema(allocator, calls, tools_json);

        // Property 1: a conforming call is left BYTE-identical.
        if (!std.mem.eql(u8, before, calls[0].arguments)) {
            std.debug.print("\n[fuzz iter {d}] auto-correct mutated a conforming call\n  before: {s}\n  after : {s}\n", .{ iter, before, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 2: the values survive verbatim (no spelling-based retyping).
        var got = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer got.deinit();
        if (!jsonValueEql(std.json.Value{ .object = args }, got.value)) {
            std.debug.print("\n[fuzz iter {d}] value mismatch\n  want: {s}\n  got : {s}\n", .{ iter, args_json, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 3: conformance holds, and coercion is idempotent.
        try testing.expect(toolCallConformsToSchema(allocator, calls[0], tools_json));
        try coerceToolArgsToSchema(allocator, calls, tools_json);
        try testing.expectEqualStrings(before, calls[0].arguments);
    }
}

test "coerceToolArgsToSchema: a MANGLED array-typed param is tolerantly repaired" {
    // Live soak capture (record 268): pi's `edits` array with a MISSING COMMA
    // between two object values — `"newText": "..." "path": "..."`. Strict parse
    // rejects it, so pre-fix `edits` stayed a STRING (schema contradiction). The
    // tolerant container repair recovers a real array.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n./game.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"  speed: 8,\", \"newText\": \"  vy: 0,\" \"extra\": \"x\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("  speed: 8,", edits.array.items[0].object.get("oldText").?.string);
}

// ── Misplaced required param (buried-`path` class) ──────────────────────────
//
// VERBATIM captured arguments, 2026-07-13 pi session (gemma-4-26B-A4B-it-qat-4bit,
// ~/.mlx-serve/logs/mlx-serve-11234.log around the us_presidents run). The model
// put `path` INSIDE each edits[] item and emitted no top-level `path`, so pi
// answered
//     Validation failed for tool "edit":
//       - path: must have required properties path
// three times in a row. Each rejection was a full multi-thousand-token
// generation; the model then abandoned `edit` and rewrote the whole file with
// `write` — the same "give up on the cheap tool, re-emit the expensive one"
// degradation as the Claude Code replace_all loop.
//
// The parse layer is NOT at fault (verified: convertGemma4Object/Array are a
// structural walk that preserves the model's own nesting and key order, and
// coerceToolArgsToSchema only rewrites values in place) — the model genuinely
// emitted it this way. The schema is what makes it provably fixable: `path` is
// required at the TOP level and the item schema declares only oldText/newText.
//
// The script bodies are the real ones, excerpted for readability — their LENGTH
// is not load-bearing (the JSON was already valid and correctly escaped; that is
// exactly why no repair path fired). The nesting and the escaping are verbatim.
const pi_buried_path_args_captured =
    \\{"edits":[{"newText":"#!/bin/bash\n\n# Get the directory where the script is located\nSCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" && pwd )\"\ncd \"$SCRIPT_DIR\"\n","oldText":"#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n","path":"us_presidents/generate_site.sh"}]}
;

test "hoistMisplacedRequiredParams: pi's buried `path` is lifted out of the edits items" {
    const allocator = testing.allocator;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, pi_buried_path_args_captured),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }

    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();

    // The required param now satisfies the schema at the top level…
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expect(path == .string);
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);

    // …and is gone from the item, whose schema never declared it.
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    const item = edits.array.items[0].object;
    try testing.expect(item.get("path") == null);

    // The payload the model worked for survives byte-exact, escaping intact.
    try testing.expectEqualStrings(
        "#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n",
        item.get("oldText").?.string,
    );
    try testing.expect(std.mem.startsWith(u8, item.get("newText").?.string, "#!/bin/bash\n"));
    try testing.expect(std.mem.indexOf(u8, item.get("newText").?.string, "${BASH_SOURCE[0]}") != null);

    // The whole call now conforms — which is the point: pi accepts it.
    try testing.expect(toolCallConformsToSchema(allocator, calls[0], pi_edit_tools_json_test));
}

test "hoistMisplacedRequiredParams: a buried `path` survives the Gemma 4 parse path end to end" {
    // The same class through the REAL parse chain (Gemma's call:name{...} form —
    // this session's model). The raw pre-parse bytes were not dumped (the server
    // was not at --log-level debug), so the tag wrapper here is reconstructed;
    // the ARGUMENT SHAPE it produces is the verbatim captured one asserted above,
    // which is what the hoist actually operates on.
    const allocator = testing.allocator;
    const raw = "<|tool_call>call:edit{edits:[{oldText:<|\"|>old line<|\"|>,newText:<|\"|>new line<|\"|>," ++
        "path:<|\"|>us_presidents/generate_site.sh<|\"|>}]}<tool_call|>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try hoistMisplacedRequiredParams(allocator, calls, pi_edit_tools_json_test);
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits.array.items[0].object.get("path") == null);
    try testing.expectEqualStrings("old line", edits.array.items[0].object.get("oldText").?.string);
}

test "hoistMisplacedRequiredParams: a compliant call is byte-identical" {
    // The strong-model path: nothing may re-serialize when nothing was misplaced.
    const allocator = testing.allocator;
    const args =
        \\{"path":"a.js","edits":[{"oldText":"a","newText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: never touches a key the item schema DECLARES" {
    // Safety gate. A tool whose items legitimately carry their own `path` (a
    // multi-file edit) must never have it stripped out from under them — hoisting
    // there would DESTROY data, not repair it. The item schema declaring the key
    // is the proof that it belongs where it is.
    const allocator = testing.allocator;
    const multifile_schema =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"}},"required":["path","oldText"]}}},"required":["path","edits"]}}}]
    ;
    const args =
        \\{"edits":[{"path":"a.js","oldText":"a"},{"path":"b.js","oldText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, multifile_schema);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: items that DISAGREE on the value stay honest" {
    // Two items, two different paths, and the tool edits ONE file. There is no
    // correct hoist here — picking either would silently write the wrong file.
    // Leave it broken so the client's validation error reaches the model.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":"a.js"},{"oldText":"c","newText":"d","path":"b.js"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: a wrong-typed buried value is never hoisted" {
    // `path` is declared a string; an object spelled `path` inside the item is
    // not the missing param. Hoisting it would just move the schema violation.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":{"nested":"x"}}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "coerceToolArgsToSchema: quoted integer/number strings coerce to numeric" {
    // A model emitting `"limit":"5"` for an integer param → strict clients reject
    // "expected integer, got string". Coerce string→numeric per the schema.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"scale":{"type":"number"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"Read\",\"arguments\":{\"path\":\"a.txt\",\"limit\":\"5\",\"scale\":\"1.5\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    const scale = parsed.value.object.get("scale").?;
    try testing.expect(scale == .float or scale == .integer);
    // A string param is untouched.
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: nullable union type [\"string\",\"null\"] coerces to the non-null member" {
    // Real schemas declare optional params as a type UNION. declaredJsonType
    // picks the first non-null member, so a boolean-spelled value under
    // ["boolean","null"] still coerces.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"flag":{"type":["boolean","null"]},"note":{"type":["string","null"]}}}}}]
    ;
    const raw = "<tool_call>\n<function=E>\n<parameter=flag>\nFalse\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("flag").? == .bool);
    try testing.expectEqual(false, parsed.value.object.get("flag").?.bool);
    try testing.expectEqualStrings("hi", parsed.value.object.get("note").?.string);
}

test "coerceToolArgsToSchema: an object param declared as a oneOf union coerces from its JSON text" {
    // fx's shell tool: `request` is a discriminated union with no top-level "type".
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"shell","parameters":{"type":"object","required":["request"],"properties":{"request":{"oneOf":[{"type":"object","required":["action","command"],"properties":{"action":{"const":"run"},"command":{"type":"string"}}},{"type":"object","required":["action"],"properties":{"action":{"const":"cancel"}}}]}}}}}]
    ;
    const raw = "<tool_call>\n<function=shell>\n<parameter=request>\n{\"action\":\"run\",\"command\":\"ls\"}\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const req = parsed.value.object.get("request").?;
    try testing.expect(req == .object);
    try testing.expectEqualStrings("ls", req.object.get("command").?.string);
}

test "declaredJsonType: a oneOf/anyOf union declares a type only when its branches agree" {
    const allocator = testing.allocator;
    const cases = [_]struct { schema: []const u8, want: ?[]const u8 }{
        .{ .schema = "{\"oneOf\":[{\"type\":\"object\"},{\"type\":\"object\"}]}", .want = "object" },
        .{ .schema = "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}", .want = "string" },
        .{ .schema = "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"object\"}]}", .want = null },
        .{ .schema = "{\"oneOf\":[{\"type\":\"object\"},{\"const\":1}]}", .want = null },
    };
    for (cases) |c| {
        var doc = try std.json.parseFromSlice(std.json.Value, allocator, c.schema, .{});
        defer doc.deinit();
        const got = declaredJsonType(doc.value);
        if (c.want) |w| {
            try testing.expectEqualStrings(w, got.?);
        } else {
            try testing.expect(got == null);
        }
    }
}

test "coerceToolArgsToSchema: an explicit JSON null satisfies any typed param (not coerced away)" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"limit":{"type":"integer"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"limit\":null}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    // null is a valid absent-value marker for any type — left byte-identical.
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: coercion is SHALLOW — values nested inside a param are untouched" {
    // Documented boundary: the layer coerces TOP-LEVEL params against the tool's
    // declared property types. It does NOT recurse into array/object element
    // properties (the tool schema's `items`/nested `properties` are the client's
    // to validate). A nested `"port":"8080"` stays a string — byte-identical.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"W","parameters":{"type":"object","properties":{"config":{"type":"object"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"W\",\"arguments\":{\"config\":{\"port\":\"8080\",\"tls\":\"false\"}}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: an unmodelled declared type leaves the value alone" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"x":{"type":"weird"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"x\":\"False\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "convertGemma4 dedups a repeated key + escapes a key with special chars (valid JSON)" {
    // Same class as the parseHermesToolCall dup-key / raw-name bugs, but in the
    // Gemma `call:name{...}` converter: a repeated key produced a DUPLICATE JSON
    // key (std.json → error.DuplicateField), and a key with a `"` was appended
    // raw → invalid JSON. Both must yield a valid JSON object.
    const allocator = testing.allocator;
    // Duplicate key.
    {
        const raw = "<|tool_call>call:shell{command:<|\"|>ls<|\"|>,command:<|\"|>pwd<|\"|>}<tool_call|>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expect(parsed.value == .object);
        try testing.expect(parsed.value.object.get("command") != null);
    }
    // Key containing a double-quote must not break the JSON.
    {
        const raw = "<|tool_call>call:shell{a\"b:<|\"|>v<|\"|>}<tool_call|>";
        const calls = try parseToolCalls(allocator, raw);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
            defer parsed.deinit();
            try testing.expect(parsed.value == .object);
        }
    }
}

test "parseXmlElementToolCall is already dup-key + escape safe (characterization)" {
    // The XML-element path builds args via std.json.ObjectMap.put + Stringify,
    // which dedups keys (last wins) and escapes values — so it is structurally
    // immune to the dup-key/raw-interpolation class that bit parseHermesToolCall
    // and convertGemma4Object. This characterization test pins that guarantee.
    const allocator = testing.allocator;
    const raw = "<tool_calls>\n<tool_name>shell</tool_name>\n<command>echo \"a\"</command>\n<command>echo \"b\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Valid JSON despite the duplicate <command> and the inner quotes.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("command") != null);
}

test "parseToolCalls: NO path emits invalid JSON args (final safety net)" {
    // Bullet-proof invariant: whatever a converter builds, the arguments a client
    // receives are ALWAYS valid JSON. convertGemma4Value copies a JSON-style
    // string verbatim, so a bad escape (`\q`) would otherwise emit invalid JSON.
    const allocator = testing.allocator;
    const cases = [_][]const u8{
        // Gemma regular-JSON-string value with an INVALID escape.
        "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>",
        // Gemma custom-string is fine, but mix a bad JSON string sibling.
        "<|tool_call>call:write{path:<|\"|>ok<|\"|>,x:\"y\\z\"}<tool_call|>",
    };
    for (cases) |text| {
        const calls = try parseToolCalls(allocator, text);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            for (cs) |tc| {
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{}) catch {
                    std.debug.print("\nINVALID JSON args emitted: {s}\n", .{tc.arguments});
                    return error.InvalidJsonEmitted;
                };
                parsed.deinit();
            }
        }
    }
    // The `\q` case is REPAIRABLE (looseRepair doubles the bad backslash), so the
    // path survives as the literal `a\qb` — the safety net repairs, only falling
    // back to `{}` when even tolerant repair fails.
    {
        const calls = (try parseToolCalls(allocator, "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("a\\qb", parsed.value.object.get("path").?.string);
    }
}

test "split content: trailing <channel|> close-marker spam never leaks (degenerate model output)" {
    // Live soak capture (record 2151, a Gemma reasoning variant): the model emitted
    // reasoning, one <channel|> close, a scrap of content, then SPAMMED 16 more bare
    // <channel|> close markers. The leading cut takes the FIRST close, and the
    // trailing-open cut only handles unclosed OPENERS — so the 16 stray CLOSE markers
    // leaked into visible content. A close marker is never valid at the tail of
    // content; trailing closers (and openers) are trimmed so <channel|> can't leak.
    const text = "Reasoning here about the file.\n<channel|>`glob` `./game.js`\n\n" ++
        "<channel|><channel|><channel|><channel|><channel|>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "<channel|>") == null);
    // The scrap of real content before the spam survives.
    try testing.expect(std.mem.indexOf(u8, content, "glob") != null);
}

test "split content: trailing </think> close spam never leaks" {
    const text = "<think>plan</think>The answer.</think></think></think>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "</think>") == null);
    try testing.expect(std.mem.indexOf(u8, content, "The answer.") != null);
}

test "split content: orphan Gemma <tool_call|> close never leaks (no-tag-leak)" {
    // Live 2026-07-16 soak (gemma-4-26B-A4B-it-qat-4bit, tools present, a "no
    // tools needed" probe at temp 0.7): the model degenerated into a bare
    // 1-token <tool_call|> CLOSE with NO <|tool_call> opener, so parseToolCalls
    // found no call and the orphan control token leaked as the ENTIRE visible
    // content (server response content == "<tool_call|>"). A tool CLOSE marker is
    // never valid at the tail of content — same class as the <channel|> spam.
    // parseToolCalls already extracted any real call before this split runs, so
    // any residual <tool_call|> here is orphan by construction.
    try testing.expectEqualStrings("", splitThinkBlock("<tool_call|>", true, false).content);
    try testing.expectEqualStrings("Sure.", splitThinkBlock("Sure.<tool_call|>", true, false).content);
    // Legit prose (no control token) is untouched.
    try testing.expectEqualStrings("2 + 2 = 4", splitThinkBlock("2 + 2 = 4", true, false).content);
}

test "splitThinkBlock content never keeps trailing channel close spam" {
    const text = "<|channel>thought\nplan<channel|>\n<|channel>\nThe answer.<channel|><channel|><channel|>";
    const split = splitThinkBlock(text, true, false);
    try testing.expect(std.mem.indexOf(u8, split.content, "<channel|>") == null);
    try testing.expect(std.mem.indexOf(u8, split.content, "The answer.") != null);
}

// ── native DSML tool calls ─────────────────────────────────

fn freeParsedCalls(calls: []ParsedToolCall) void {
    for (calls) |tc| {
        testing.allocator.free(tc.name);
        testing.allocator.free(tc.arguments);
    }
    testing.allocator.free(calls);
}

test "parseToolCalls dsml: canonical two-call block, string and JSON params" {
    const raw = "I'll check the weather.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"get_weather\">\n" ++
        "<｜DSML｜parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "<｜DSML｜invoke name=\"get_time\">\n" ++
        "<｜DSML｜parameter name=\"tz\" string=\"true\">PST</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{\"tz\":\"PST\"}", calls[1].arguments);
    try testing.expect(!calls[0].inferred);
}

test "parseToolCalls dsml: spaced tag names parse like unspaced ones" {
    const raw = "On it.\n\n<｜DSML｜ calls>\n" ++
        "<｜DSML｜ invoke name=\"get_weather\">\n" ++
        "<｜DSML｜ parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n" ++
        "</｜DSML｜ invoke>\n" ++
        "<｜DSML｜ invoke name=\"get_time\">\n" ++
        "</｜DSML｜ invoke>\n" ++
        "</｜DSML｜ calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{}", calls[1].arguments);
}

test "parseToolCalls dsml: dropped invoke close before the next call still parses both" {
    // Delimiter-drop tolerance: the model omits </｜DSML｜invoke>
    // between back-to-back calls.
    const raw = "<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"first\">\n" ++
        "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n" ++
        "<｜DSML｜invoke name=\"second\">\n" ++
        "<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("first", calls[0].name);
    try testing.expectEqualStrings("{\"a\":\"x\"}", calls[0].arguments);
    try testing.expectEqualStrings("second", calls[1].name);
    try testing.expectEqualStrings("{\"b\":true}", calls[1].arguments);
}

test "parseToolCalls dsml: truncation salvages NAME + completed pairs, never fragments" {
    // Server-side cut mid-value: the unterminated pair is dropped (a half-
    // written file is worse than a re-issued write), completed pairs survive.
    const raw = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a.txt</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"content\" string=\"true\">first half of a long fi";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"/tmp/a.txt\"}", calls[0].arguments);
}

test "parseToolCalls dsml: rich string value runs to the confirmed close" {
    // Newlines, quotes, braces, `<` inside a string value — value ends ONLY
    // at the closing parameter tag (the Gemma dropped-quote class).
    const raw = "<｜DSML｜invoke name=\"bash\">\n" ++
        "<｜DSML｜parameter name=\"cmd\" string=\"true\">grep -r \"a < b\" src/\necho {done}</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("grep -r \"a < b\" src/\necho {done}", parsed.value.object.get("cmd").?.string);
}

test "parseToolCalls dsml: duplicate keys dedup (last wins) and invalid JSON value degrades to string" {
    const raw = "<｜DSML｜invoke name=\"t\">\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">old</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">new</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"n\" string=\"false\">not[json</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("new", parsed.value.object.get("k").?.string);
    try testing.expectEqualStrings("not[json", parsed.value.object.get("n").?.string);
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
}

test "parseToolCalls dsml: zero-arg call and thinking prefix" {
    const raw = "thinking about it</think>Sure.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"list_files\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "streamShouldBufferForTools: DSML marker and partial fullwidth prefixes hold" {
    try testing.expect(streamShouldBufferForTools("Sure.\n\n<｜DSML｜tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<｜DSML｜invoke name=\"x\">"));
    // partial fullwidth-bar prefix at the tail — next token may complete it
    try testing.expect(streamShouldBufferForTools("answer text <｜DSML"));
    try testing.expect(streamShouldBufferForTools("answer text <｜"));
    // plain prose with a fullwidth bar elsewhere must NOT hold
    try testing.expect(!streamShouldBufferForTools("the ｜ character is fun"));
}

test "loadChatConfig: a template-less checkpoint keeps the empty fallback" {
    // No family fallback template exists: a model dir without a chat template
    // gets "" (fallbackFormatChat territory), whatever its model_type.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\": \"e\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\": \"llama\"}" });
    var pbuf: [512]u8 = undefined;
    const plen = try tmp.dir.realPath(io, &pbuf);

    var cc = try loadChatConfig(io, allocator, pbuf[0..plen]);
    defer cc.deinit();
    try testing.expectEqual(@as(usize, 0), cc.chat_template.len);
}

test "renderChatTemplate: thinking-off closes a template-opened think block (unconditional-opener class)" {
    // This template opens `<think>` unconditionally; thinking-off used to
    // mean generating the whole reasoning pass and discarding it. Closing the
    // block in the prompt skips it. Thinking ON keeps the opener untouched.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    const off = try renderChatTemplate(allocator, &messages, &config, null, null, false, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<think></think>"));

    const on = try renderChatTemplate(allocator, &messages, &config, null, null, true, false);
    defer allocator.free(on);
    try testing.expect(std.mem.endsWith(u8, on, "<think>"));
}

test "renderChatTemplate: continuation appends the partial reply into the OPEN assistant turn" {
    // Continuing a reply means the prompt ends mid-assistant-turn. The trailing
    // assistant message is a PREFIX to extend, not history: rendering it as
    // history and then letting add_generation_prompt open a second assistant
    // turn is what the server did before this existed, and the model answers
    // the doubled turn by starting over.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "count to three" },
        .{ .role = "assistant", .content = "one, two," },
    };

    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, true);
    defer allocator.free(cont);
    // Ends INSIDE the turn, on the partial text — no close tag after it.
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistantone, two,"));
    // Exactly one assistant turn was opened.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "<|im_start|>assistant"));
    // The partial appears once — as the prefill, not also as history.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "one, two,"));

    // Same messages WITHOUT the flag stay history + a fresh turn, unchanged.
    const plain = try renderChatTemplate(allocator, &messages, &config, null, null, true, false);
    defer allocator.free(plain);
    try testing.expect(std.mem.endsWith(u8, plain, "<|im_start|>assistant"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, plain, "<|im_start|>assistant"));
}

test "renderChatTemplate: continuation resumes in the CONTENT channel, not a think block" {
    // A template that opens `<think>` unconditionally (this class) would
    // otherwise put the partial reply — visible prose the user already read —
    // inside a reasoning block, and the continuation would come back as
    // reasoning. The same tail that thinking-off uses closes it first.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Hello ther" },
    };

    // Thinking ON is the hard case: nothing else would close the block.
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<think></think>Hello ther"));
}

test "renderChatTemplate: a continuation prefill is trimmed of trailing whitespace" {
    // The tokenizer merges a trailing space with the word that follows it, so
    // a prompt ending in one splits differently than the same text mid-reply —
    // the hazard Anthropic's API rejects prefills outright for. Trim it: the
    // stored reply keeps its whitespace, only the PROMPT is trimmed.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Once upon a time \n" },
    };
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "Once upon a time"));
}

test "renderChatTemplate: continuation is declined when the last message is not an assistant" {
    // The flag describes the SHAPE of the request; a client that sets it on a
    // user-final conversation gets an ordinary turn rather than a prompt with
    // the user's own words committed into the assistant's mouth.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistant"));
}

test "parseToolCalls: Anthropic-style bare <invoke> markup is an inferred call (#928)" {
    const allocator = testing.allocator;
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">cd /x && ls -la</parameter>\n<parameter name=\"timeout\">30</parameter>\n</invoke>\n</function_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    // Not the model's own dialect: it must name a declared tool to survive.
    try testing.expect(calls[0].inferred);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("cd /x && ls -la", parsed.value.object.get("command").?.string);
    try testing.expectEqual(@as(i64, 30), parsed.value.object.get("timeout").?.integer);
    // Prose words are not the tag.
    try testing.expect((try parseToolCalls(allocator, "I <invoked> it with <parameters> set.")) == null);
    // Nothing of an unparsed block reaches content; prose naming the tag stays.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls"));
    const prose = "Claude's format wraps each call in <invoke name=\"x\"> tags.";
    try testing.expectEqualStrings(prose, trimLeakedToolMarkup(prose));
}

test "parseToolCalls: a bare <invoke> in prose or inside another dialect's value is not a call" {
    const allocator = testing.allocator;
    // The opener alone is prose, as the scrubber reads it.
    try testing.expect((try parseToolCalls(allocator, "Claude's format wraps each call in <invoke name=\"bash\"> tags.")) == null);
    // A Hermes parameter VALUE carrying invoke markup never decides the call.
    const text = "<tool_call>\n<function=write>\n<parameter=content>\nrun <invoke name=\"bash\"><parameter name=\"command\">ls</parameter></invoke>\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    try testing.expect(!calls[0].inferred);
}

test "streamShouldBufferForTools: every prefix of a bare <invoke> call holds; prose releases" {
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>\n</function_calls>";
    for ([_]usize{ 0, std.mem.indexOf(u8, text, "<invoke").? }) |start| {
        var end = start + 2;
        while (end <= text.len) : (end += 1) {
            testing.expect(streamShouldBufferForTools(text[start..end])) catch |e| {
                std.debug.print("gate flushed a prefix of an accepted call: '{s}'\n", .{text[start..end]});
                return e;
            };
        }
    }
    try testing.expect(!streamShouldBufferForTools("I <invoked> the <function_callback> helper"));
}

test "serializeExtraContext: preserve_thinking keeps the template's default; chat_template_kwargs fill what the request did not decide" {
    // A template that reuses prior-turn reasoning by default: only an explicit kwarg turns it off.
    const allocator = testing.allocator;
    var reasoning_reuse = ChatConfig{
        .chat_template = "…{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}…",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const r = try serializeExtraContext(allocator, &reasoning_reuse, true);
    defer allocator.free(r);
    try testing.expect(std.mem.indexOf(u8, r, "preserve_thinking") == null);
    // The per-model kwargs can turn it off and carry any other key; a key the
    // request decides (enable_thinking) is not theirs.
    reasoning_reuse.chat_template_kwargs = "{\"preserve_thinking\":false,\"custom\":{\"n\":1},\"enable_thinking\":false}";
    const off = try serializeExtraContext(allocator, &reasoning_reuse, true);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"preserve_thinking\":false") != null);
    try testing.expect(std.mem.indexOf(u8, off, "\"custom\":{\"n\":1}") != null);
    try testing.expect(std.mem.indexOf(u8, off, "\"enable_thinking\":true") != null);
    try testing.expect(std.mem.indexOf(u8, off, "\"enable_thinking\":false") == null);

    // A kwarg can never replace the conversation the wrapper puts in context.
    reasoning_reuse.chat_template_kwargs = "{\"messages\":[],\"tools\":[],\"add_generation_prompt\":false}";
    const core = try serializeExtraContext(allocator, &reasoning_reuse, true);
    defer allocator.free(core);
    for ([_][]const u8{ "messages", "tools", "add_generation_prompt" }) |k| try testing.expect(std.mem.indexOf(u8, core, k) == null);

    var plain = ChatConfig{
        .chat_template = "{{ messages }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const p = try serializeExtraContext(allocator, &plain, true);
    defer allocator.free(p);
    try testing.expect(std.mem.indexOf(u8, p, "preserve_thinking") == null);
}

test "mergeTemplateKwargs: request keys win, the model's fill the rest" {
    const allocator = testing.allocator;
    const req = try std.json.parseFromSlice(std.json.Value, allocator, "{\"preserve_thinking\":true,\"x\":1}", .{});
    defer req.deinit();
    const merged = try mergeTemplateKwargs(allocator, "{\"preserve_thinking\":false,\"y\":\"m\"}", req.value.object);
    defer allocator.free(merged);
    const got = try std.json.parseFromSlice(std.json.Value, allocator, merged, .{});
    defer got.deinit();
    try testing.expectEqual(true, got.value.object.get("preserve_thinking").?.bool);
    try testing.expectEqual(@as(i64, 1), got.value.object.get("x").?.integer);
    try testing.expectEqualStrings("m", got.value.object.get("y").?.string);
    try testing.expectEqual(@as(usize, 3), got.value.object.count());
}

test "ThinkScan: incremental gate matches the fresh gate at every prefix length" {
    // The gate is O(buffer) per token today (measured 47.99 us/token at 113 KB
    // with no close in the buffer), i.e. O(n^2) over a reasoning block. A
    // cursor makes it O(new bytes) — but only if it is EXACT, and the failure
    // a naive cursor introduces is a marker split across two token arrivals.
    // Feeding prefix-by-prefix IS that case, at every possible split point.
    const corpus = [_][]const u8{
        "plain visible prose with no markup at all",
        "<think>reasoning here</think>the answer",
        // A </think> inside a tool ARGUMENT is payload, not a block close.
        "<think>plan<tool_call>f<arg_key>k</arg_key><arg_value>closes with </think> inside</arg_value></tool_call>done</think>visible",
        "<|channel>thought hidden<channel|>visible",
        "prose where a < b and nothing else happens",
        "trailing partial opener at the end <thi",
        "trailing partial close </thin",
    };
    const flags = [_][2]bool{ .{ true, false }, .{ false, false }, .{ false, true } };
    for (corpus) |full| {
        for (flags) |f| {
            const enable_thinking = f[0];
            const prompt_opened = f[1];
            var scan: ThinkScan = .{};
            var n: usize = 0;
            while (n <= full.len) : (n += 1) {
                const buf = full[0..n];
                const want = streamThinkGate2(buf, enable_thinking, false, prompt_opened);
                const got = streamThinkGateScan(buf, enable_thinking, false, prompt_opened, &scan);
                if (want != got) {
                    std.debug.print("prefix {d} of \"{s}\" (think={}, opened={}): {any} != {any}\n", .{ n, full, enable_thinking, prompt_opened, got, want });
                    return error.IncrementalGateDiverged;
                }
            }
        }
    }
}

test "ThinkScan: per-token work is FLAT in buffer size, not linear" {
    // The whole point. A smaller constant is not the fix — the re-examined
    // span must not grow with the buffer, or a 32K-token thought pays 8x what
    // the 113 KB bench measured.
    var scan: ThinkScan = .{};
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(testing.allocator);
    var max_span: usize = 0;
    var i: usize = 0;
    while (i < 4000) : (i += 1) {
        // A think block that never closes — the expensive arm of the bench.
        try buf.appendSlice(testing.allocator, if (i == 0) "<think>" else "token words and more ");
        _ = streamThinkGateScan(buf.items, true, false, false, &scan);
        if (i > 0 and scan.last_scan_span > max_span) max_span = scan.last_scan_span;
    }
    try testing.expect(buf.items.len > 60_000);
    // Bounded by one token's bytes plus the trailing partial-marker candidate.
    try testing.expect(max_span <= 64);

    // A growing partial marker at the tail is re-examined, and only it.
    var scan2: ThinkScan = .{};
    var buf2 = std.ArrayList(u8).empty;
    defer buf2.deinit(testing.allocator);
    try buf2.appendSlice(testing.allocator, "<think>");
    var j: usize = 0;
    while (j < 500) : (j += 1) try buf2.appendSlice(testing.allocator, "filler ");
    _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
    for ("</think") |c| {
        try buf2.append(testing.allocator, c);
        _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
        try testing.expect(scan2.last_scan_span <= 64);
    }
}

test "ThinkScan: reset() re-arms the cursor when the stream buffer is cleared" {
    // The buffer is NOT append-only — the stream loop trims and resets it at
    // every emit. A stale cursor then points past the end or into different
    // bytes, which is the second failure mode a naive cursor introduces.
    var scan: ThinkScan = .{};
    const first = "<think>reasoning</think>";
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGateScan(first, true, false, false, &scan));
    scan.reset();
    // Same scan struct, fresh (shorter) buffer: must not carry the old close.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGateScan("<think>more", true, false, false, &scan));
}

test "the streaming handlers route the think gate through a persistent ThinkScan" {
    // A memoized scan nobody threads through is a silent no-op that is
    // output-identical to the unmemoized gate — the exact shape of the
    // hardcoded-use_drafter=false class. The streaming surface must hold a
    // scan beside its buffer AND reset it where it clears that buffer.
    // Needles are ++-split so this test's own source cannot satisfy the scan.
    const src = @embedFile("server.zig");
    const call = "streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think" ++ "_scan)";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, call));
    // The unmemoized 4-arg gate must be GONE from the streaming paths.
    const old = "chat_mod.streamThinkGate2(" ++ "buf,";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
    // One reset per buffer clear in those handlers.
    const reset = "think_scan" ++ ".reset();";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, reset));
}

test "bench: streaming think gate scans BYTES, memoized vs fresh" {
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    // The 5.10 acceptance bar is that per-token work is FLAT in buffer size,
    // not merely smaller. Wall clock is unavailable in a hermetic test under
    // Zig 0.17 (clocks live under std.Io) and would be noise anyway — bytes
    // examined is the thing that was quadratic, and it is deterministic.
    // Mirrors the 4b.4 harness: 4000 tokens growing to ~104 KB.
    const a = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(a);
    var scan: ThinkScan = .{};
    try buf.appendSlice(a, "<think>");
    var memo_bytes: u64 = 0;
    var fresh_bytes: u64 = 0;
    var i: usize = 0;
    while (i < 4000) : (i += 1) {
        try buf.appendSlice(a, "some reasoning words here ");
        const g = streamThinkGateScan(buf.items, true, false, false, &scan);
        try testing.expectEqual(StreamThinkGate.hold_thinking, g);
        memo_bytes += scan.last_scan_span;
        // What the unmemoized gate re-reads every token: the whole buffer, and
        // more than once (open scan + close scan + two substring scans).
        fresh_bytes += buf.items.len;
    }
    std.debug.print(
        "[think-gate] {d} tokens to {d} KB: fresh >= {d} KB scanned, memoized {d} KB ({d}x)\n",
        .{ 4000, buf.items.len / 1024, fresh_bytes / 1024, memo_bytes / 1024, fresh_bytes / @max(memo_bytes, 1) },
    );
    // Linear-in-total, not quadratic: the memoized scan reads each byte a
    // bounded number of times.
    try testing.expect(memo_bytes < buf.items.len * 4);
}

test "streaming content agrees with the non-streaming split on leading whitespace" {
    // Template-opened class: the template opens <think> unconditionally, so the first
    // generated token is the newline pair after the closer. The non-streaming
    // split drops it; the stream must reach the same bytes.
    const out = "\n\nOne, two, three.";
    const expected = splitThinkBlock(out, true, false).content;
    try testing.expectEqualStrings("One, two, three.", expected);

    const tokens = [_][]const u8{ "\n\n", "One", ", two", ", three." };
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(testing.allocator);
    var content_started = false;
    for (tokens) |tt| {
        const emit = streamContentLead(tt, content_started);
        if (emit.len == 0) continue;
        content_started = true;
        try streamed.appendSlice(testing.allocator, emit);
    }
    try testing.expectEqualStrings(expected, streamed.items);
}
test "streamContentLead never touches whitespace inside the answer" {
    // Only the LEAD is suppressed: once anything visible has been sent, a
    // newline token is ordinary content (markdown paragraphs, code blocks).
    try testing.expectEqualStrings("\n\n", streamContentLead("\n\n", true));
    try testing.expectEqualStrings("  x", streamContentLead("  x", true));
    // And a lead that is not whitespace is untouched.
    try testing.expectEqualStrings("Hello", streamContentLead("Hello", false));
    // A token is never cut in half: one carrying whitespace AND text rides
    // whole, so its logprobs entry still describes bytes that reached content.
    try testing.expectEqualStrings("\nHello", streamContentLead("\nHello", false));
}

test "parseToolCalls: <tool_call>{JSON} truncated mid-string recovers NAME + {} (never a fragment)" {
    // A 4 KB edit call that hit EOS inside a string value. The object never
    // balances, so the balanced-snap arm declines; without a name salvage the
    // whole turn was dropped at the flush (empty `stop`, nothing for the client
    // to nudge on). Same rule as every other dialect: NAME + `{}`.
    const allocator = testing.allocator;
    const text = "Fixing it.\n<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"a.ts\", \"oldText\": \"function foo() {\\n  return 1;";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
    // Reversed key order + whitespace variants still find the name.
    const text2 = "<tool_call>{ \"arguments\" : {\"name\": \"nope\", \"s\": \"a}b\"}, \"name\" : \"real\", \"extra\": \"unterm";
    const c2 = (try parseToolCalls(allocator, text2)) orelse return error.NoCalls;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("real", c2[0].name);
    // No name anywhere: still nothing (no invented call).
    try testing.expect((try parseToolCalls(allocator, "<tool_call>\n{\"arguments\": {\"path\": \"a")) == null);
}

// ── `<function` opener streaming gate ──

test "streamShouldBufferForTools: EVERY growing prefix of <function holds" {
    // DERIVED, not enumerated. The previous version of this test was a second
    // hand-written copy of `tail_prefixes` and inherited its hole: `<funct` was
    // missing from BOTH, so a token boundary landing exactly there flushed the
    // fragment and leaked the rest of the tag (`<f` id 54303 + `unct` id 14185
    // decode to exactly `<funct` in this vocabulary — a real decomposition, not
    // a hypothetical). Generating the assertions from the marker string means a
    // rung missing from the array fails here even if nobody remembers to add an
    // assertion for it.
    const marker = "<function";
    var i: usize = 2;
    while (i <= marker.len) : (i += 1) {
        try testing.expect(streamShouldBufferForTools(marker[0..i]));
    }
}

test "streamShouldBufferForTools: false positive — bare <function> with no attribute" {
    try testing.expect(!streamShouldBufferForTools("here is <function>"));
    try testing.expect(!streamShouldBufferForTools("Wrap it in a <functional> block"));
}

test "renderChatTemplate: a NUL byte in message content does not truncate the prompt" {
    // Bar: content after the NUL, and the assistant header, survive the render.
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\<|im_start|>{{ message['role'] }}
        \\{{ message['content'] }}<|im_end|>
        \\{% endfor %}{%- if add_generation_prompt -%}<|im_start|>assistant
        \\{% endif %}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const messages = [_]Message{
        .{ .role = "user", .content = "before\x00after" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "before") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "after") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>assistant") != null);
}

test "parseToolCalls hermes: a truncated <function= call is never renamed by a JSON parameter value" {
    // Bar: a call cut before any close tag keeps the name its `<function=`
    // opener declared, whatever the parameter value looks like.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/package.json\n</parameter>\n" ++
        "<parameter=content>\n{\"name\": \"voxel-pagoda-garden\", \"version\": \"1.0.0\"}";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("write_file", calls[0].name);
}

test "parseToolCalls hermes: a parameter value carrying </parameter> is not emptied" {
    // Bar: the value ends at the close tag that actually ends it, byte-exact.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=content>\nsee </parameter> in the docs\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("see </parameter> in the docs", parsed.value.object.get("content").?.string);
}

test "parseToolCalls hermes: a parameter value carrying </tool_call> is not dropped" {
    // Bar: a required parameter survives a value that spells the wrapper close.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=content>\nemit </tool_call> here\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("emit </tool_call> here", parsed.value.object.get("content").?.string);
}

test "parseToolCalls hermes: a parameter value types from JSON's number grammar, not parseFloat" {
    // Bar: only a spelling JSON itself calls a number stays unquoted.
    const allocator = testing.allocator;
    const Case = struct { spelling: []const u8, number: bool };
    const cases = [_]Case{
        .{ .spelling = "Infinity", .number = false },
        .{ .spelling = "nan", .number = false },
        .{ .spelling = "+1", .number = false },
        .{ .spelling = "0755", .number = false },
        .{ .spelling = ".5", .number = false },
        .{ .spelling = "5.", .number = false },
        .{ .spelling = "1.5", .number = true },
        .{ .spelling = "-2", .number = true },
        .{ .spelling = "3e5", .number = true },
        .{ .spelling = "0", .number = true },
    };
    for (cases) |c| {
        const raw = try std.mem.concat(allocator, u8, &.{
            "<tool_call>\n<function=edit>\n<parameter=mode>\n", c.spelling, "\n</parameter>\n</function>\n</tool_call>",
        });
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{}) catch {
            std.debug.print("\n{s}: args not valid JSON: {s}\n", .{ c.spelling, calls[0].arguments });
            return error.ArgsNotValidJson;
        };
        defer parsed.deinit();
        const v = parsed.value.object.get("mode") orelse return error.MissingParam;
        if (c.number) {
            if (v == .string) {
                std.debug.print("\n{s}: shipped as a string\n", .{c.spelling});
                return error.NumberQuoted;
            }
        } else {
            if (v != .string) {
                std.debug.print("\n{s}: shipped unquoted as {s}\n", .{ c.spelling, @tagName(v) });
                return error.NonNumberUnquoted;
            }
            try testing.expectEqualStrings(c.spelling, v.string);
        }
    }
}

test "appendJsonString: invalid UTF-8 becomes U+FFFD, valid multibyte is byte-exact" {
    // Bar: the escaper's output is always well-formed UTF-8 (nlohmann rejects
    // anything else), and every valid sequence survives unchanged.
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, "\xff\xfe\x80 binary");
    try testing.expect(std.unicode.utf8ValidateSlice(buf.items));
    try testing.expectEqualStrings("\"\u{FFFD}\u{FFFD}\u{FFFD} binary\"", buf.items);

    buf.clearRetainingCapacity();
    const valid = "héllo \u{4e2d}\u{6587} \u{1F600} ok";
    try appendJsonString(allocator, &buf, valid);
    try testing.expectEqualStrings("\"" ++ valid ++ "\"", buf.items);

    // A truncated multibyte sequence at end-of-input is one replacement.
    buf.clearRetainingCapacity();
    try appendJsonString(allocator, &buf, "a\xe4\xb8");
    try testing.expectEqualStrings("\"a\u{FFFD}\"", buf.items);
}

test "a model-opened think block latches from its own opener token" {
    // Both openers are dropped from visible deltas as channel markers, so without
    // this latch the whole thought streams as the answer (Gemma 4).
    try std.testing.expectEqualStrings("<think>", modelThinkOpener(false, "<think>").?);
    try std.testing.expectEqualStrings("<think>", modelThinkOpener(false, "<think>\n").?);
    try std.testing.expectEqualStrings("<|channel>thought", modelThinkOpener(true, "thought").?);
    try std.testing.expect(modelThinkOpener(false, "thought") == null);
    // Prose that merely BEGINS with the word is not the opener token.
    try std.testing.expect(modelThinkOpener(true, "thoughts") == null);
    try std.testing.expect(modelThinkOpener(true, "\n") == null);
    try std.testing.expect(modelThinkOpener(false, "<think>x") == null);
}

const mid_system_test_template =
    \\{# tools 'tool' #}
    \\{%- for m in messages -%}{{ '<|im_start|>' + m.role + '\n' + (m.content or '') + '<|im_end|>\n' }}{%- endfor -%}
    \\{{ '<|im_start|>assistant\n' }}
;

test "mid-system: supported templates preserve late notes and stable history" {
    const allocator = testing.allocator;
    const config = ChatConfig{ .chat_template = mid_system_test_template, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable instructions" },
        .{ .role = "user", .content = "first question" },
        .{ .role = "system", .content = "<total_tokens>15000000 tokens left</total_tokens>" },
        .{ .role = "assistant", .content = "checking", .tool_calls = &calls },
        .{ .role = "tool", .content = "result", .tool_call_id = "call_1" },
        .{ .role = "system", .content = "<total_tokens>14999999 tokens left</total_tokens>" },
    };
    const before = try renderChatTemplate(allocator, messages[0..3], &config, "[]", null, true, false);
    defer allocator.free(before);
    const after = try renderChatTemplate(allocator, &messages, &config, "[]", null, true, false);
    defer allocator.free(after);
    try testing.expect(std.mem.startsWith(u8, after, before));
    try testing.expect(std.mem.indexOf(u8, after, "result<|im_end|>\n<|im_start|>system\n<total_tokens>14999999") != null);
}

test "mid-system: a tail-only template uses stable consolidation across turns" {
    const allocator = testing.allocator;
    const tpl =
        \\{% for m in messages %}{% if m.role == 'system' and not loop.first and not loop.last %}{{ raise_exception('system must be first or last') }}{% endif %}{{ m.role + ':' + (m.content or '') + ';' }}{% endfor %}
    ;
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, false);
    defer allocator.free(rendered);
    try testing.expectEqualStrings("system:stable\n\nlate note;user:question;", rendered);
}

test "mid-system: a template dropping tail notes cannot preserve them" {
    const allocator = testing.allocator;
    const tpl =
        \\{% for m in messages %}{% if m.role != 'system' or loop.first or not loop.last %}{{ m.role + ':' + (m.content or '') + ';' }}{% endif %}{% endfor %}
    ;
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "late note") != null);
}

test "generic fallback injects tools once when no system message leads" {
    const allocator = testing.allocator;
    const config = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
    const messages = [_]Message{
        .{ .role = "user", .content = "question" },
    };
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"unique_tool_marker\",\"parameters\":{}}}]";
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools, null, false, false);
    defer allocator.free(rendered);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "unique_tool_marker"));
}

test "mid-system: unsupported templates fold without generic fallback or lost notes" {
    const allocator = testing.allocator;
    const templates = [_][]const u8{
        "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}{{ m.role + ':' + (m.content or '') + ';' }}{% endfor %}",
        "{% for m in messages %}{% if m.role != 'system' or loop.first %}{{ m.role + ':' + (m.content or '') + ';' }}{% endif %}{% endfor %}",
        "{% for m in messages %}{% if m.role == 'system' %}{{ 'system:' + m.content + ';' }}{% endif %}{% endfor %}{% for m in messages %}{% if m.role != 'system' %}{{ m.role + ':' + m.content + ';' }}{% endif %}{% endfor %}",
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "system", .content = "initial hook" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    for (templates) |tpl| {
        const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
        const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, false);
        defer allocator.free(rendered);
        try testing.expectEqualStrings("system:stable\n\ninitial hook\n\nlate note;user:question;", rendered);
    }
}

test "foldSystemMessages: a system turn past index 0 joins the leading system message" {
    // Live 2026-09-15: Claude Code carries SessionStart hook output as a
    // `system`-role message INSIDE `messages`, after the top-level system
    // prompt. Some templates raise on a system turn that is not first, and
    // the raise is a silent generic fallback (the model loses its stop token).
    const al = std.testing.allocator;
    var msgs = std.ArrayList(Message).empty;
    defer msgs.deinit(al);
    try msgs.append(al, .{ .role = "system", .content = "You are S." });
    try msgs.append(al, .{ .role = "system", .content = "hook output" });
    try msgs.append(al, .{ .role = "user", .content = "hi" });
    try msgs.append(al, .{ .role = "system", .content = "late note" });
    const owned = try foldSystemMessages(al, &msgs, false);
    defer if (owned) |o| al.free(o);
    try std.testing.expectEqual(@as(usize, 2), msgs.items.len);
    try std.testing.expectEqualStrings("You are S.\n\nhook output\n\nlate note", msgs.items[0].content);
    try std.testing.expectEqualStrings("user", msgs.items[1].role);

    // No leading system: the fold creates one at index 0.
    var lone = std.ArrayList(Message).empty;
    defer lone.deinit(al);
    try lone.append(al, .{ .role = "user", .content = "hi" });
    try lone.append(al, .{ .role = "system", .content = "hook output" });
    const owned2 = try foldSystemMessages(al, &lone, false);
    defer if (owned2) |o| al.free(o);
    try std.testing.expectEqualStrings("system", lone.items[0].role);
    try std.testing.expectEqualStrings("hook output", lone.items[0].content);
    try std.testing.expectEqual(@as(usize, 2), lone.items.len);

    // Nothing to fold: untouched, nothing allocated.
    var plain = std.ArrayList(Message).empty;
    defer plain.deinit(al);
    try plain.append(al, .{ .role = "system", .content = "You are S." });
    try plain.append(al, .{ .role = "user", .content = "hi" });
    try std.testing.expect((try foldSystemMessages(al, &plain, false)) == null);
    try std.testing.expectEqual(@as(usize, 2), plain.items.len);
}

test "answerStopIndex: a stop after trailing whitespace in the answer still cuts" {
    const t = std.testing;
    const text = "</think>\n\nHi \n\nmore";
    try t.expectEqual(@as(?usize, std.mem.indexOf(u8, text, "Hi ").? + 3), answerStopIndex(text, 0, "\n\n", true));
    const hard_break = "</think>\n\nline  \nnext";
    try t.expectEqual(@as(?usize, std.mem.indexOf(u8, hard_break, "  \n").? + 2), answerStopIndex(hard_break, 0, "\n", true));
    // Whitespace leading the answer is never delivered, so it never matches.
    try t.expectEqual(@as(?usize, null), answerStopIndex("</think>\n\nHi", 0, "\n\n", true));
}

test "renderChatTemplate: a `\\xHH` escape is Python's code point U+00HH, so 0x80 and up render as UTF-8" {
    const cfg = ChatConfig{ .chat_template = "{{ messages[0].content }}{{ \"\\xe9|\\x41\" }}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const out = try renderChatTemplate(testing.allocator, &messages, &cfg, null, null, false, false);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hi\u{e9}|A", out);
}

test "a sampled special token where <function='s `<` belongs still names the call" {
    const allocator = std.testing.allocator;
    const raw = "<tool_call>\n<|im_start|>function=read_file>\n<parameter=path>\nsrc/game.js\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.TestUnexpectedResult;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("read_file", calls[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"src/game.js\"}", calls[0].arguments);
    // A body with its real opener is left alone; a special that is not glued to `function=` is not touched.
    try std.testing.expect((try hermesSpecialOpenerRepair(allocator, "<function=x>\n</function>")) == null);
    try std.testing.expect((try hermesSpecialOpenerRepair(allocator, "<|im_start|>\nfunction=x>")) == null);
}
