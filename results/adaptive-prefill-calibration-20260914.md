# Adaptive prefill calibration — 2026-09-14

## Outcome

The default-off adaptive prefill scheduler passed the mixed-load acceptance gate
at a 1,500 ms target ITL and a 2,000 ms maximum defer deadline. It chooses among
64/128/256/512-token slices using measured prefill cost, defers prefill after an
observed ITL miss, forces a minimum slice when a queued or active prompt reaches
its service deadline, and ranks APC-trimmed residual work behind an oldest-first
fairness floor.

The implementation remains experimental and is not enabled in the restored
production service.

## Controlled comparison

Both arms used the same 18-request schedule, 3,887–3,920-token prompts, 80 output
tokens, decode concurrency 8, prompt concurrency 4, and Qwen3.8 Flash Next 4-bit
MTP checkpoint. APC was disabled for the performance comparison so cache reuse
could not confound the scheduler result. Runs used the production PLE/QSA
environment and were separated by service restart and a clean thermal-status
check.

| Metric | Baseline, 512-token mixed prefill | Adaptive, target 1,500 ms | Change |
|---|---:|---:|---:|
| p90 ITL | 1,127.1 ms | 773.1 ms | 31.4% better |
| p99 ITL | 1,490.4 ms | 864.2 ms | 42.0% better |
| p95 TTFT | 36,992.1 ms | 37,695.4 ms | 1.9% worse |
| Aggregate output throughput | 20.739 tok/s | 20.385 tok/s | 1.7% worse |
| Jain tenant-rate fairness | 0.9926 | 0.9920 | effectively flat |

All 18 requests completed in both arms. The adaptive arm recorded 70 released
prefill rounds, one ITL-triggered defer, one deadline-forced release, and the
following maximum-width histogram: 256×61, 512×6, 64×1, plus two terminal tail
fragments (47 and 3 tokens). All acceptance gates passed: p99 improvement at
least 20%, throughput within 5%, p95 TTFT within 25%, fairness at least 0.90,
matching request counts, terminal completion, and mechanism engagement.

Raw evidence:

- `adaptive-prefill-baseline-c1.json`
- `adaptive-prefill-candidate-t1500-d2000.json`
- `adaptive-prefill-t1500-d2000-gate.json`

## Calibration note

A corrected 1,000 ms arm reduced p99 ITL to 772.5 ms (48.2%) and kept p95 TTFT
within the guardrail, but reduced aggregate output throughput by 13.8%; it was
rejected. This is useful evidence that 128-token-heavy operation is too costly
at this operating point, while the 256-token-heavy 1,500 ms policy occupies the
better tradeoff region.

Two earlier diagnostics are invalid as performance evidence: one used a stale
MLX runtime, and one omitted the production PLE environment. An initial deadline
implementation also allowed old queued work to force every active slice; live
calibration caught this and the policy was corrected so the deadline now bounds
time to first service and then the gap between successive prefill slices.

## APC functional check

With the same adaptive policy and a four-entry, 2 GiB APC enabled, two identical
24-token requests returned 0 then 23 cached prompt tokens. The second request's
TTFT was 55.0 ms versus 685.0 ms for the first. This confirms cache lookup and
residual trimming occur before prefill. Admission-order behavior for concurrent
cached residuals is covered by the focused scheduler regression test; it was not
part of the cache-disabled performance comparison.

## Decision

Keep the feature default-off. The 1,500/2,000 ms configuration is the first
passing operating point and is suitable for a longer soak or production-shadow
trial. Do not use the 1,000 ms configuration for general service because its
tail gain does not justify its measured throughput loss.
