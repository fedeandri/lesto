# bf16Scalar failures during the model load are logged, not refused

Found by Codex while reviewing the m14-latch fix (merged 2026-11-12, 8ba6fb1..05e8cfe).

`bf16Scalar` (src/transformer.zig, for example the embedding scale at line 6043 and the softcap scalars at 6179 and
6186) builds a scalar array without checking the MLX error latch. If one of them fails during the load, the
load-wide guard in `doLoadOnInferenceThread` (src/scheduler.zig) now takes the latch and logs
"[load] dropped an MLX error ..." instead of letting it fail the first request. The broken scalar then fails loudly at
the first forward, so this is not a silent wrong answer, but the load should refuse at the failing call.

Fix: check the latch right after each required `bf16Scalar` at load time and fail the load with its message. Optional
steps keep their own drop. Needs a unit test with `armLatchingFaultForTest`, a Codex review, and the greedy-reply check
on the M4 (lossless change).
