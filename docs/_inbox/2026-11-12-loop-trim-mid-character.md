# A non-streamed reply's loop cut can split a character

`loopTrimmedIds` (src/server.zig, the non-stream chat path near `const emit_ids = loopTrimmedIds(...)`) cuts the
reply at `slot.loop_trim_start`, a token index. When the model spells a character as several `<0xNN>` byte tokens
(emoji outside the vocabulary, or any emoji under a head that drops emoji tokens), the cut can land inside that
character, and the decode renders the leftover bytes as U+FFFD.

Seen 2026-11-12 on the M3 (0.21.1 candidate build, greedy): a reply that repeated a four-byte emoji ended in
`... 🦝 ��` non-streamed, while the stream sent every repeat whole.

Fix: move the cut back to the start of the byte-token run it lands in (or drop a trailing incomplete sequence after
the decode). Needs a unit test with a byte-fallback tokenizer and the greedy-reply check on the M4.
