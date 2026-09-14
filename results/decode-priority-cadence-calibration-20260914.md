# Decode-priority cadence calibration — 2026-09-14

## Verdict

**IMPLEMENTED DEFAULT-OFF / PROMOTION REJECTED AT CALIBRATION.** The scheduler
control and receipts work, but no tested cadence met the precommitted requirement
to improve p99 inter-token latency by at least 20%. The compatibility default
therefore remains `--decode-priority-cadence 1`; no held-out promotion run was
authorized by the calibration result.

## Configuration

- Runtime commit: `3bd6813c` on `codex/prefill-cadence-20260914`.
- Model: `Qwen3.8-Flash-Next-MLX-4bit-MTP`, served in plain non-MTP mode.
- Fixed server shape: decode concurrency 8, prompt concurrency 4, prompt batch
  window 16, prefill chunk 512, APC capacity 0, greedy sampling, thinking off.
- Frozen seed-427 Poisson trace: 18 requests across 8 tenants over 28.18 seconds;
  each prompt was about 3,900 tokens and requested 80 output tokens.
- Each cadence used a fresh process and one fixed warm-up request. macOS reported
  no thermal or performance warning throughout; system free memory was 91–93%.

## Results

| Cadence | p99 ITL | vs c1 | p90 ITL | Output t/s | TTFT p95 | Jain fairness | Scheduler receipts |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1530.3 ms | control | 1182.1 ms | 20.22 | 38.10 s | 0.9923 | 51 prefill |
| 2 | 1530.9 ms | -0.0% | 1379.4 ms | 19.67 | 38.85 s | 0.9936 | 41 deferred / 38 release |
| 4 | 1408.4 ms | +8.0% | 1295.3 ms | 21.05 | 35.26 s | 0.9923 | 108 deferred / 38 release |
| 8 | 1371.0 ms | +10.4% | 665.9 ms | 20.37 | 37.19 s | 0.9921 | 257 deferred / 38 release |

Cadence 8 materially improved the median and p90 shape, but the fixed 512-token
prefill slices still create roughly 1.37-second decode interruptions. Cadence
reduces their frequency; it does not reduce the duration of an individual
interruption, so the p99 target remains missed. A follow-up should combine
decode priority with a smaller decode-active prefill slice or a time/token
budget, rather than increasing cadence alone and extending prompt starvation.

All four arms completed 18/18 requests with no client errors and no foreign
canary leakage. The automatic quality gate is intentionally recorded as failed:
the calibration used an 80-token cap and only 3/18 baseline responses reached
the requested end-of-answer canary. This is a calibration-design limitation,
not a candidate-specific quality attribution. Since the performance gate already
failed, the ladder did not spend GPU on a corrected held-out quality schedule.

## Incidental blocker repaired

The first warm-up exposed an existing APC failure when a newly inserted entry
is immediately evicted by its configured capacity. Post-store bookkeeping
unconditionally fetched the removed trie path and raised `KeyError`, terminating
the generation worker. Commit `3bd6813c` now performs access/spill bookkeeping
only when the exact entry survived eviction. The new zero-capacity regression
test passes, and the sweep then ran with APC capacity zero to isolate scheduling.

## Verification

- Cadence/APC focused suite: 94 passed.
- Expanded server, runtime, reload, APC, cadence, and scorer suite: 213 passed,
  6 failed, 68 subtests passed. All six failures reproduce unchanged on the
  untouched `unified` checkout and are stale soft-reload fixtures that omit the
  current admission method's required request and tenant identifiers.
- All result files share schedule SHA-256
  `b2b5c4a036c339bce49e1586372a5a36033cdad0f19fbf3913b4bb65eb8cc650`.
