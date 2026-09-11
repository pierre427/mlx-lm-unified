# Qwen4 phase-family ICB P1 gate

P1 replaces the synthetic P0 commands with a six-command prefix from one real
Flash-Next linear-attention layer. It does not launch the old whole-token
megakernel. The artifact preparer uses the current pack and schedule ABIs and
the `MirrorExecutor` opcode definition as the oracle.

The extracted schedule uses the supported non-fused spelling so every family
boundary is observable:

1. `GROUP_RMSNORM`: four 2,560-wide hyper streams to `NORMED`.
2. `QMV`: Q4 10,240-to-320 down projection with scaled SiLU to `HC_LR`.
3. `QMV`: Q4 320-to-10,240 up projection with sigmoid to `HC_W`.
4. `HC_MIX`: weighted four-stream reduction to `MIXED`.
5. `QMV`: Q4 10,240-to-4 inject gate to `INJECT`.
6. `QMV`: Q4 2,560-to-10,240 GDN input projection to `GDN_QKV`.

This covers the affine-Q4 family, norm, hyper-connection glue, and the GDN
activation/state boundary. `NORMED`, `HC_LR`, `HC_W`, `MIXED`, `INJECT`, and
`GDN_QKV` are exposed. Direct and ICB outputs are compared as raw float32 bits.
The independent MLX oracle uses `rtol=1e-4`, `atol=1e-5` and records maximum
and mean absolute error for every range.

## Bound artifact

`prepare_qwen4_phase_family_p1.py` writes a fresh artifact directory containing:

- the one-group production pack and exact offset table;
- the six ABI schedule rows;
- deterministic full scratch input and `MirrorExecutor` expected scratch;
- generated Metal source whose Q4 dot/dequant/reduction and activation helpers
  are extracted from the current `MEGA_QMV` and `BODY_HELPERS` strings;
- source, helper, and kernel-header SHA-256 digests;
- file size and SHA-256 for every binary;
- declared scratch input/output ranges and mandatory mechanism counts.

Metadata-only inspection is safe without a GPU lease:

```bash
benchmarks/native_icb/prepare_qwen4_phase_family_p1.py --mode inspect
```

Artifact preparation is GPU work because MLX evaluates the real pack and
oracle. Run it only through the repository GPU-job wrapper while holding the
lease:

```bash
/usr/bin/env MLX_UAG_GPU_LEASE_ACK=1 \
  MLX_UAG_GPU_LEASE_AGENT=<current-cpg-agent-id> \
  .venv/bin/python benchmarks/native_icb/prepare_qwen4_phase_family_p1.py \
  --mode prepare \
  --out-dir /absolute/new/artifact-directory
```

## Native runner acceptance contract

The native runner must refuse before dispatch unless all file and implementation
digests match, the pack has one group, every projection is Q4/64, table stride
is 12, schedule stride is 8, and the schedule is exactly the six declared rows.
It then executes two fresh scratch copies:

- direct: six explicit Metal dispatches with five dependency barriers;
- candidate: the same pipelines and arguments encoded once in an ICB, barriers
  set before commands 1 through 5, then one `executeCommandsInBuffer` call.

Both lanes must report six device-side command receipts. Each declared direct
output must be bitwise equal to the corresponding ICB output. The independently
executed MLX `MirrorExecutor` output must satisfy
`abs(candidate-oracle) <= 1e-5 + 1e-4*abs(oracle)`. The report records max/mean
absolute error and the count/first eight offsets outside tolerance for each
lane and range. It also includes direct commands/barriers/submissions, ICB
encoded commands/barriers, execute calls, pre-encode time, replay host time,
GPU time, and microseconds per command. Any missing receipt, digest drift,
direct/ICB bit mismatch, out-of-tolerance oracle result, or unsupported ICB is
a hard failure.

`verify_qwen4_phase_family_p1.py` is the CPU-only acceptance checker. The
native runner writes a `mlx-uag.qwen4-phase-family-icb-p1-receipt.v1` receipt
that includes the exact `manifest_sha256`, plus the full direct scratch image
and the full ICB scratch image. The verifier revalidates every artifact and
implementation-source digest, refuses a manifest that differs from the fixed
six-command schedule or output-range contract, checks all nine mechanism
counters, compares direct-to-ICB as raw uint32 words, and independently applies
the declared numeric oracle tolerance to both native lanes. It always writes a
result JSON, including on a fail-closed exception:

```bash
../.venv/bin/python benchmarks/native_icb/verify_qwen4_phase_family_p1.py \
  --artifact /absolute/artifact-directory \
  --receipt /absolute/native-receipt.json \
  --direct-scratch /absolute/direct-scratch.f32 \
  --icb-scratch /absolute/icb-scratch.f32 \
  --out /absolute/verification.json
```

## Running the native gate

`qwen4_phase_family_icb.mm` is a standalone native runner, not a relaunch of
the old whole-token megakernel. It refuses before creating a Metal device unless
explicitly acknowledged, then independently checks the manifest digest, schema,
five-entry pack geometry, exact six schedule rows, fixed input/output ranges,
scratch size, and native geometry. The direct correctness lane submits six
commands with five buffer barriers. The ICB correctness lane binds the same
three pipeline states, buffers, parameter offsets, grid, and threadgroup-memory
lengths once; commands 1 through 5 carry `setBarrier`.

The safe wrapper requires the live CPG and filesystem GPU lease, creates a new
output directory, runs the native receipt through the CPU verifier, and refuses
to overwrite prior evidence:

```bash
MLX_UAG_GPU_LEASE_ACK=1 \
MLX_UAG_GPU_LEASE_AGENT=<current-cpg-agent-id> \
benchmarks/native_icb/run_qwen4_phase_family_p1.sh \
  /absolute/artifact-directory \
  /absolute/new-result-directory \
  100
```

The generated QMV kernel uses the current helper text exactly. The phase
wrappers preserve the persistent body's 40 x 512 grid, 1/2/4 rows per
simdgroup, conditional 2,560-float source staging, GroupRMSNorm reduction
order, and activation functions. `HC_MIX` uses the mirror/fused-up arithmetic
order because the current persistent body no longer has a standalone opcode-5
branch. This is called out rather than hidden as a claim that the removed
branch still exists.

## Why the oracle is numeric rather than bitwise

The first real P1 gate produced bitwise-identical direct and ICB scratch over
the full buffer and every exposed range. Both native lanes differed slightly
from `MirrorExecutor` because MLX and the standalone Metal phase kernels use
different valid floating-point reduction/execution orders. Maximum absolute
errors were `1.91e-6` (`NORMED`), `3.46e-6` (`HC_LR`), `2.09e-6` (`HC_W`),
`5.25e-6` (`MIXED`), `1.79e-7` (`INJECT`), and `6.91e-6` (`GDN_QKV`); every
value satisfied the declared tolerance. Requiring cross-runtime bit identity
therefore rejected a correct ICB mechanism comparison. The revised contract
keeps bitwise identity where it is meaningful—direct versus ICB using the same
pipelines—and keeps the independent MLX implementation as a tight numeric
correctness oracle.
