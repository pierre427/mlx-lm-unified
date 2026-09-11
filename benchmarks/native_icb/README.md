# Native compute ICB P0 gate

This isolated benchmark tests whether one reusable Metal indirect command buffer
can reduce host encoding cost for a long verifier command sequence. It expands
the dependent `write -> transform -> read` pattern to exactly 242 and 640
commands, then compares direct concurrent dispatch encoding with ICB replay.

The result is fail-closed. All three pipeline states must report indirect-command
support, the ICB must have nonzero size, and direct and ICB lanes must match the
host-simulated state, every read checkpoint, and the GPU command counter. The
counter must equal `command_count * repetitions`, which proves that the reusable
ICB actually executed instead of silently taking a direct or empty path. A
validation failure exits nonzero after printing exact diagnostics and writing a
JSON result with `passed: false`.

Build without running GPU work:

```bash
benchmarks/native_icb/build.sh
benchmarks/native_icb/build/compute_icb_bench --help
```

Run only while holding the repository CPG GPU lease and filesystem lock:

```bash
MLX_UAG_GPU_LEASE_ACK=1 \
MLX_UAG_GPU_LEASE_AGENT=<current-cpg-agent-id> \
benchmarks/native_icb/run.sh \
  --commands 242,640 \
  --repetitions 1000 \
  --warmup-repetitions 5 \
  --json-out /absolute/path/native-icb-p0.json
```

The JSON reports one-time ICB pre-encoding time; direct and replay host encoding
time; effective wall and GPU microseconds per command; speedups; and exactness
and mechanism counters. It also accounts for every direct barrier, pre-encoded
ICB barrier, execution range, and command-buffer submission. The harness does
not wire ICB into the MLX runtime.

Metal's compute-ICB barrier belongs to the command that waits: command `i`
calls `setBarrier` when `i > 0`, ensuring commands before `i` complete before it
executes. The first prototype placed barriers on all commands except the last,
which left the final command unsynchronized; this version corrects that ordering.

The real one-layer P1 gate is documented in `P1_SPEC.md`. It uses a
digest-bound production Qwen4 pack and `MirrorExecutor` oracle, generated Metal
source extracted from the current exact Q4 helpers, and a standalone direct vs
reusable-ICB native runner. Direct and ICB must remain bitwise identical;
`MirrorExecutor` is an independent numeric oracle with a fixed `rtol=1e-4`,
`atol=1e-5` gate and per-range max/mean error telemetry. Nothing is wired into
the core runtime.
