"""Default-off persistent per-token decode megakernel for Qwen3.8-Flash-Next.

One dispatch per token.  The layer chain runs inside it as a sequence of
phases separated by the grid-wide device-scope barrier settled by the
2026-09-03 feasibility spike, replacing ~1,500 dependent dispatches at a ~9 us
GPU-side floor with a handful of long-running launches.

Nothing here runs at import time.  ``MLX_QWEN4_MEGAKERNEL`` defaults OFF and
admission fails closed.

Three structural decisions, and why.

**The phase dispatcher reads a device-visible schedule, it is not unrolled.**
The alternative -- emitting 48 layers x ~12 phases of straight-line MSL -- was
rejected on three counts.  The source would be ~600 phase bodies, so compile
time and the shader cache grow with the layer count; the binary would
re-specialise on every layer-type mix, so the MTP head (one full-attention
layer) could not share it; and the usual argument FOR unrolling does not apply
here.  That argument is register pressure: a single dispatch has ONE register
allocation whichever way it is written, and the spike measured the residency
ceiling as a thread/register budget (~2,048 threads/core), so a 48-layer kernel
must fit its worst phase's occupancy either way.  What unrolling would buy is
constant address arithmetic, and the offset table already reduces that to one
indexed load per phase against a 2.1--5.2 us barrier.  The schedule buffer also
puts the barrier map -- which phase boundary needs a device barrier and which
needs only a threadgroup barrier -- in DATA, so adopting a tuned map is a
host-side change rather than a kernel edit.

**Scratch is ping-ponged, never written in place.**  The spike's U2 result is
that the first cross-threadgroup handoff of an address is correct even at
threadgroup scope and every later REUSE of that address is stale.  A persistent
kernel reuses the same scratch every phase of every layer, so device scope is
mandatory, and any phase that both reads and writes the residual streams writes
to the other slab.

**Admission fails closed.**  ``M != 1`` is refused at first: batch and
speculative lanes come later.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

# ------------------------------------------------------------------ geometry
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HC_COUNT * HIDDEN
HC_LOWRANK = 320
NUM_LAYERS = 48
RMS_EPS = 1e-6

# GDN
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_KEY_DIM = 128
GDN_VALUE_DIM = 128
GDN_RATIO = GDN_VALUE_HEADS // GDN_KEY_HEADS
CONV_KERNEL = 4
KEY_DIM = GDN_KEY_HEADS * GDN_KEY_DIM
VALUE_DIM = GDN_VALUE_HEADS * GDN_VALUE_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM

# full attention
N_Q_HEADS = 24
N_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
Q_DIM = N_Q_HEADS * HEAD_DIM

# indexer
IDX_HEADS = 4
IDX_HEAD_DIM = 128
IDX_COMPRESS = 4
IDX_BUDGET = 2048
BLOCK_TOPK = IDX_BUDGET // IDX_COMPRESS  # 512

# MoE
NUM_EXPERTS = 512
TOPK = 10
FF = 640
VOCAB = 248320

# The block grid the in-kernel selector must handle.  16,384 blocks is 65,536
# tokens of context at compress ratio 4.
MAX_BLOCKS = 16384


# -------------------------------------------------------------------- opcodes
OP_NOP = 0
OP_QMV = 1          # quantized matvec, optionally activated
OP_DENSE_MV = 2     # bfloat16 matvec
OP_RMSNORM = 3      # plain RMSNorm with a weight
OP_GROUP_RMSNORM = 4
OP_HC_MIX = 5       # silu/sigmoid gate, reshape, mean over the H streams
OP_INJECT = 6       # residual + branch * inject, broadcast over H streams
OP_GDN_CORE = 7
OP_MOE_TOPK = 8
OP_MOE_E1 = 9
OP_MOE_E2 = 10
OP_ATTN = 11
OP_INDEX_TOPB = 12  # bounded top-BLOCK_TOPK selection over the block scores
OP_COPY = 13
OP_ADD = 14
OP_SILU_MUL = 15

OP_NAMES = {
    value: name
    for name, value in sorted(globals().items())
    if name.startswith("OP_") and isinstance(value, int)
}

# Barrier kind, per schedule step.  The spike measured 2.1--3.3 us for a
# device barrier in a light kernel and 5.2 us in the GDN kernel, against ~0 for
# a threadgroup barrier, so which boundary needs which is worth tuning -- and
# it lives here, in data.
BAR_NONE = 0
BAR_THREADGROUP = 1
BAR_DEVICE = 2

# Schedule stride, in uint32 words.
STEP_STRIDE = 8
STEP_FIELDS = ("op", "entry", "src", "dst", "arg0", "arg1", "arg2", "barrier")
assert len(STEP_FIELDS) == STEP_STRIDE


@dataclass(frozen=True)
class Step:
    op: int
    entry: int = 0xFFFFFFFF
    src: int = 0
    dst: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    barrier: int = BAR_DEVICE

    def row(self) -> list[int]:
        return [
            self.op, self.entry, self.src, self.dst,
            self.arg0, self.arg1, self.arg2, self.barrier,
        ]


class Schedule:
    """An ordered list of phases plus the buffer the kernel walks."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def add(self, step: Step) -> int:
        self.steps.append(step)
        return len(self.steps) - 1

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def device_barriers(self) -> int:
        return sum(1 for step in self.steps if step.barrier == BAR_DEVICE)

    def to_array(self) -> mx.array:
        return mx.array(
            [value for step in self.steps for value in step.row()], mx.uint32
        )


# ------------------------------------------------------------- scratch layout
_SCRATCH_BLOCKS = (
    ("RESID_A", HC_HIDDEN),
    ("RESID_B", HC_HIDDEN),
    ("NORMED", HC_HIDDEN),
    ("HC_LR", HC_LOWRANK),
    ("HC_W", HC_HIDDEN),
    ("MIXED", HIDDEN),
    ("INJECT", HC_COUNT),
    ("BRANCH", HIDDEN),
    ("GDN_QKV", CONV_DIM),
    ("GDN_Z", VALUE_DIM),
    ("GDN_BA", 2 * GDN_VALUE_HEADS),
    ("GDN_Y", VALUE_DIM),
    ("ATT_QG", 2 * Q_DIM),
    ("ATT_K", N_KV_HEADS * HEAD_DIM),
    ("ATT_V", N_KV_HEADS * HEAD_DIM),
    ("ATT_O", Q_DIM),
    ("IDX_QK", (IDX_HEADS + 1) * IDX_HEAD_DIM),
    ("IDX_SCORE", MAX_BLOCKS),
    ("IDX_SEL", BLOCK_TOPK),
    ("MOE_LOGITS", NUM_EXPERTS),
    ("MOE_TOPI", TOPK),
    ("MOE_TOPW", TOPK),
    ("MOE_ACT", TOPK * FF),
    ("SHARED_ACT", FF),
    ("SHARED_UP", FF),
    ("SHARED_OUT", HIDDEN),
    ("SHARED_GATE", 1),
    ("SCRATCH_TMP", 1024),
)

SCRATCH: dict[str, int] = {}
_offset = 0
for _name, _size in _SCRATCH_BLOCKS:
    SCRATCH[_name] = _offset
    _offset += _size
SCRATCH_FLOATS = _offset
del _offset, _name, _size


# ------------------------------------------------------------------ env/state
def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{name} must be 0/off or 1/on; got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


_MEGAKERNEL_ENABLED = _env_flag("MLX_QWEN4_MEGAKERNEL")
# Geometry.  Provisional: the spike cleared G in {40, 80} x T in {256, 512} for
# residency and measured the two-block chain fastest at G=80/T=256, but its
# own phase profile says the geometry is not settled (T=512 lost on that kernel
# and won by 1.5x on a plain stream).  ``opus-mk-tune`` owns these numbers;
# they are read from the environment so adopting the spec cannot need an edit.
_THREADGROUPS = _env_int("MLX_QWEN4_MEGAKERNEL_GROUPS", 80, minimum=1)
_THREADS = _env_int("MLX_QWEN4_MEGAKERNEL_THREADS", 256, minimum=32)
_SPIN_CAP = _env_int("MLX_QWEN4_MEGAKERNEL_SPIN_CAP", 400_000, minimum=1)

_STATUS_LOCK = threading.Lock()
_STATUS_COUNTS: Counter = Counter()
_STATUS_LAST: Optional[dict[str, Any]] = None
_STATUS_ABORTS = 0
_STATUS_LAUNCHES = 0
_STATUS_PHASES = 0


@dataclass(frozen=True)
class MegakernelAdmission:
    accepted: bool
    reason: str


def admit_megakernel_decode(
    *,
    width: int,
    batch: int,
    pack: Any,
    schedule: Any,
    speculating: bool,
    training: bool,
    sharded: bool,
    mask: Any = None,
) -> MegakernelAdmission:
    """Pure structural admission; safe to exercise without MLX eval.

    Refuses everything the first cut does not serve.  ``M != 1`` is the
    headline: the kernel holds one token's activations in threadgroup memory
    and one recurrent state per value head per threadgroup, so a wider slab is
    not a parameter change.
    """
    if not _megakernel_enabled():
        return MegakernelAdmission(False, "disabled")
    if training:
        return MegakernelAdmission(False, "training")
    if sharded:
        return MegakernelAdmission(False, "distributed sharding")
    if speculating:
        return MegakernelAdmission(False, "speculative rollback")
    if batch != 1:
        return MegakernelAdmission(False, f"batch {batch}")
    if width != 1:
        return MegakernelAdmission(False, f"query width {width}")
    if mask is not None:
        return MegakernelAdmission(False, "masked decode")
    if pack is None:
        return MegakernelAdmission(False, "weights not packed")
    if schedule is None or len(schedule) == 0:
        return MegakernelAdmission(False, "empty schedule")
    if not _device_supported():
        return MegakernelAdmission(False, "device unsupported")
    return MegakernelAdmission(True, "engaged")


def _megakernel_enabled() -> bool:
    return _MEGAKERNEL_ENABLED


def set_qwen4_megakernel(enabled: bool) -> bool:
    """Live-toggle the megakernel without changing resident arrays."""
    global _MEGAKERNEL_ENABLED
    _MEGAKERNEL_ENABLED = bool(enabled)
    return _MEGAKERNEL_ENABLED


def _device_supported() -> bool:
    """Apple GPU with a device-scope barrier, i.e. Metal 3.2 or newer.

    The spike proved the barrier on ``applegpu_g17s`` at ``__METAL_VERSION__``
    400.  MLX exposes no version query, so this checks the architecture family
    and leaves the real proof to the kernel's own abort flag, which reports a
    non-resident grid rather than hanging.
    """
    try:
        info = mx.device_info()
    except Exception:  # pragma: no cover - no Metal device
        return False
    return str(info.get("architecture", "")).startswith("applegpu")


def record_megakernel_receipt(
    *, engaged: bool, reason: str, phases: int = 0, aborted: bool = False,
    **fields: Any,
) -> None:
    global _STATUS_LAST, _STATUS_ABORTS, _STATUS_LAUNCHES, _STATUS_PHASES
    receipt = {
        "engaged": bool(engaged),
        "reason": reason,
        "phases": int(phases),
        "aborted": bool(aborted),
        **fields,
    }
    with _STATUS_LOCK:
        _STATUS_COUNTS[reason] += 1
        if engaged:
            _STATUS_LAUNCHES += 1
            _STATUS_PHASES += int(phases)
        if aborted:
            _STATUS_ABORTS += 1
        _STATUS_LAST = receipt


def qwen4_megakernel_status(*, reset: bool = False) -> dict[str, Any]:
    """Admission, geometry and abort evidence for the megakernel path."""
    global _STATUS_LAST, _STATUS_ABORTS, _STATUS_LAUNCHES, _STATUS_PHASES
    with _STATUS_LOCK:
        report = {
            "enabled": _megakernel_enabled(),
            "device_supported": _device_supported(),
            "threadgroups": _THREADGROUPS,
            "threads": _THREADS,
            "spin_cap": _SPIN_CAP,
            "scratch_floats": SCRATCH_FLOATS,
            "scratch_bytes": SCRATCH_FLOATS * 4,
            "step_stride": STEP_STRIDE,
            "counts": dict(_STATUS_COUNTS),
            "launches": _STATUS_LAUNCHES,
            "phases": _STATUS_PHASES,
            "aborts": _STATUS_ABORTS,
            "mlx_version": str(getattr(mx, "__version__", "unknown")),
            "last_decision": _STATUS_LAST,
        }
        if reset:
            _STATUS_COUNTS.clear()
            _STATUS_LAST = None
            _STATUS_ABORTS = 0
            _STATUS_LAUNCHES = 0
            _STATUS_PHASES = 0
    return report


# --------------------------------------------------------- top-block selection
def float_sort_key(values):
    """IEEE-754 float32 -> uint32 whose unsigned order is the float order.

    The in-kernel selector is a radix select over this key, so the CPU mirror
    and the kernel must agree on it exactly.  Positive floats keep their bit
    pattern with the sign bit set; negative floats are inverted.  ``-inf`` maps
    to ``0x007FFFFF``, the lowest key any non-NaN value can take, which is what
    makes an invalid block sort below every real score.  A NaN score would sort
    outside the finite range at whichever end its sign puts it; the indexer's
    score is a sum of ``relu`` terms over finite pooled keys, so a NaN there is
    a bug upstream and this selector does not paper over it.
    """
    import numpy as np

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    sign = np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    return (bits ^ sign).astype(np.uint32)


def select_top_blocks_mirror(scores, k: int):
    """CPU mirror of ``OP_INDEX_TOPB``: the exact algorithm the kernel runs.

    Four 8-bit radix passes narrow the score distribution to the bucket that
    contains the k-th largest key, then the selection is emitted in two parts:
    every block strictly above the threshold, then blocks equal to it, taken in
    ASCENDING BLOCK INDEX until k slots are full.

    That tie rule is a choice, and it has to be stated.  ``mx.argpartition``
    leaves ties unspecified, and ties are not exotic here: the indexer's score
    is a sum of ``relu`` terms, so a block none of whose heads scores positive
    is exactly ``0.0``, and an invalid block is exactly ``-inf``.  Downstream
    consumes ``selected`` only as a SET, so any tie-break yields the same
    attention as long as the selected multiset of SCORES matches -- which it
    does by construction.  Lowest block index wins, deterministically.

    Returns the selected block ids, ascending.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = scores.shape[0]
    k = min(int(k), n)
    if k == n:
        return np.arange(n, dtype=np.uint32)
    keys = float_sort_key(scores)

    prefix = np.uint32(0)
    remaining = k
    threshold = np.uint32(0)
    for shift in (24, 16, 8, 0):
        mask = np.uint32(0xFFFFFFFF) << np.uint32(shift + 8) if shift < 24 else np.uint32(0)
        candidates = keys if shift == 24 else keys[(keys & mask) == prefix]
        digits = (candidates >> np.uint32(shift)) & np.uint32(0xFF)
        counts = np.bincount(digits, minlength=256)
        # walk buckets from the top; the bucket that crosses `remaining` holds
        # the k-th largest key
        cumulative = 0
        chosen = 0
        for bucket in range(255, -1, -1):
            if cumulative + counts[bucket] >= remaining:
                chosen = bucket
                break
            cumulative += counts[bucket]
        remaining -= cumulative
        prefix = np.uint32(prefix | (np.uint32(chosen) << np.uint32(shift)))
        threshold = prefix
    above = np.flatnonzero(keys > threshold).astype(np.uint32)
    equal = np.flatnonzero(keys == threshold).astype(np.uint32)
    take = k - above.shape[0]
    return np.sort(np.concatenate([above, equal[:take]])).astype(np.uint32)


def select_top_blocks_threaded_mirror(scores, k: int, nt: int = 256):
    """Line-for-line mirror of ``select_top_blocks``, including its threading.

    ``select_top_blocks_mirror`` proves the ALGORITHM; this proves the
    PARALLELISATION -- the chunked counts, the exclusive scan and the slot
    formula ``gt_before + min(eq_before, need)`` that lets an order-free emit
    still land globally ascending.  A bug in that formula is invisible to a
    single-threaded mirror and would show up on the GPU as a duplicated or
    dropped block id.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = scores.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.uint32)
    keys = float_sort_key(scores)

    prefix = np.uint32(0)
    want = int(k)
    for pass_index in range(4):
        shift = 24 - 8 * pass_index
        hi_mask = np.uint32(0) if pass_index == 0 else np.uint32(
            (0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF
        )
        counts = np.zeros(256, dtype=np.int64)
        selected = keys[(keys & hi_mask) == prefix]
        if selected.size:
            counts = np.bincount(
                (selected >> np.uint32(shift)) & np.uint32(0xFF), minlength=256
            )
        cumulative = 0
        chosen = 0
        for bucket in range(255, -1, -1):
            if cumulative + counts[bucket] >= want:
                chosen = bucket
                break
            cumulative += counts[bucket]
        want -= cumulative
        prefix = np.uint32(prefix | (np.uint32(chosen) << np.uint32(shift)))
    threshold, need = prefix, want

    chunk = (n + nt - 1) // nt
    gtc, eqc = [], []
    for tid in range(nt):
        lo, hi = min(tid * chunk, n), min(tid * chunk + chunk, n)
        block = keys[lo:hi]
        gtc.append(int((block > threshold).sum()))
        eqc.append(int((block == threshold).sum()))
    gt_scan, eq_scan, a, b = [], [], 0, 0
    for tid in range(nt):
        gt_scan.append(a)
        eq_scan.append(b)
        a += gtc[tid]
        b += eqc[tid]

    out = np.full(k, 0xFFFFFFFF, dtype=np.uint32)
    for tid in range(nt):
        lo, hi = min(tid * chunk, n), min(tid * chunk + chunk, n)
        gt_before, eq_before = gt_scan[tid], eq_scan[tid]
        for i in range(lo, hi):
            key = keys[i]
            if key > threshold:
                out[gt_before + min(eq_before, need)] = i
                gt_before += 1
            elif key == threshold:
                if eq_before < need:
                    out[gt_before + eq_before] = i
                eq_before += 1
    return out


# -------------------------------------------------------------- kernel source
# The grid barrier is the spike's, verbatim: the arrival counter is never
# reset, so no per-call memset is needed; the last arriver publishes a
# generation word nobody else writes, so spinners do not contend the counter;
# and both fences are DEVICE scope, which the spike measured as the difference
# between 0 and 1,023,897,600 stale words in 1.024e9 reads on REUSED addresses.
MEGA_BARRIER = r"""
#include <metal_atomic>

inline bool gbar(device atomic_uint* ctr, device atomic_uint* ab,
                 uint ntg, uint tid, thread uint& phase, uint cap) {
  device atomic_uint* gen = ctr + 2;
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  if (tid == 0u) {
    if (atomic_load_explicit(ab, memory_order_relaxed) == 0u) {
      uint want = phase + 1u;
      uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
      if (prev + 1u == want * ntg) {
        atomic_store_explicit(gen, want, memory_order_relaxed);
      } else {
        uint spins = 0u;
        while (atomic_load_explicit(gen, memory_order_relaxed) < want) {
          if (atomic_load_explicit(ab, memory_order_relaxed) != 0u) break;
          if (++spins >= cap) {
            atomic_store_explicit(ab, 1u, memory_order_relaxed);
            break;
          }
        }
      }
    }
  }
  phase += 1u;
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  return atomic_load_explicit(ab, memory_order_relaxed) == 0u;
}
"""

# Dequant-on-the-fly matvec helpers, ported from the spike.  R rows per
# simdgroup so the activation slice is loaded once and reused: one row per
# simdgroup spends eight load slots on x for every one on weights, which is
# what made the spike's first version 0.69x.  The one change from the spike is
# the scale/bias addressing -- the pack splits them on the leading axis, so the
# bias base is a second offset rather than ``soff + ng``.
MEGA_QMV = r"""
inline float dot8(uint p, float4 a0, float4 a1) {
  return a0.x * float(p & 0xFu)
       + a0.y * float((p >> 4u) & 0xFu)
       + a0.z * float((p >> 8u) & 0xFu)
       + a0.w * float((p >> 12u) & 0xFu)
       + a1.x * float((p >> 16u) & 0xFu)
       + a1.y * float((p >> 20u) & 0xFu)
       + a1.z * float((p >> 24u) & 0xFu)
       + a1.w * float((p >> 28u) & 0xFu);
}

inline float dot4x8(uint p, float4 a) {
  return a.x * float(p & 0xFFu)
       + a.y * float((p >> 8u) & 0xFFu)
       + a.z * float((p >> 16u) & 0xFFu)
       + a.w * float((p >> 24u) & 0xFFu);
}

inline float hsum4(float4 v) { return v.x + v.y + v.z + v.w; }

template <typename F4>
inline void qdot4_rows(const device uint* w, const device BF* sc,
                       const device BF* bi, uint ng, uint nwords, F4 x4,
                       uint lane, const thread uint* woff,
                       const thread uint* soff, uint R, thread float* acc) {
  const device uint2* w2 = reinterpret_cast<const device uint2*>(w);
  uint n2 = nwords >> 1u;
  for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
  for (uint t = 0; t * 32u < n2; ++t) {
    uint bl = t * 32u + lane;
    if (bl >= n2) break;
    float4 a0 = x4[bl * 4u + 0u], a1 = x4[bl * 4u + 1u];
    float4 a2 = x4[bl * 4u + 2u], a3 = x4[bl * 4u + 3u];
    float xs = hsum4(a0) + hsum4(a1) + hsum4(a2) + hsum4(a3);
    uint g = bl >> 2u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[(woff[r] >> 1u) + bl];
      float part = dot8(p.x, a0, a1) + dot8(p.y, a2, a3);
      acc[r] += float(sc[soff[r] + g]) * part + float(bi[soff[r] + g]) * xs;
    }
  }
  for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}

template <typename F4>
inline void qdot8_rows(const device uint* w, const device BF* sc,
                       const device BF* bi, uint ng, uint nwords, F4 x4,
                       uint lane, const thread uint* woff,
                       const thread uint* soff, uint R, thread float* acc) {
  const device uint2* w2 = reinterpret_cast<const device uint2*>(w);
  uint n2 = nwords >> 1u;
  for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
  for (uint t = 0; t * 32u < n2; ++t) {
    uint bl = t * 32u + lane;
    if (bl >= n2) break;
    float4 a0 = x4[bl * 2u + 0u], a1 = x4[bl * 2u + 1u];
    float xs = hsum4(a0) + hsum4(a1);
    uint g = bl >> 3u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[(woff[r] >> 1u) + bl];
      float part = dot4x8(p.x, a0) + dot4x8(p.y, a1);
      acc[r] += float(sc[soff[r] + g]) * part + float(bi[soff[r] + g]) * xs;
    }
  }
  for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}

inline float silu_f(float x) { return x / (1.0f + metal::precise::exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

inline float softplus_f(float x) {
  float hi = metal::max(x, 0.0f), lo = metal::min(x, 0.0f);
  return hi + metal::precise::log(1.0f + metal::precise::exp(lo - hi));
}
"""

# Radix select over the monotone float key.  Four 8-bit passes plus one emit
# pass, all inside ONE threadgroup, so it needs no grid barrier: 16,384 scores
# is 5 x 16 KiB of reads, against a 5.2 us barrier it would otherwise pay.
MEGA_TOPB = r"""
inline uint fkey(float v) {
  uint b = as_type<uint>(v);
  return b ^ ((b >> 31u) ? 0xFFFFFFFFu : 0x80000000u);
}

// Top-`k` of `n` scores into `out`, ASCENDING, ties broken by lowest index.
//
// Four 8-bit radix passes over the monotone float key narrow the distribution
// to the exact key of the k-th largest score, and leave `need` -- how many
// blocks EQUAL to that key are still wanted.  The emit is then order-free and
// still deterministic, because every element's output slot is a function of
// counts before it rather than of arrival:
//
//     slot(i) = (# j < i with key > T) + min(# j < i with key == T, need)
//
// Both counts come from one per-thread chunk count plus an exclusive scan, so
// the pass is parallel and the result is globally ascending -- which is what
// lets a kernel-vs-mirror test compare arrays rather than sets.
//
// It runs inside ONE threadgroup and needs no grid barrier: 16,384 scores is
// six 64 KiB sweeps against a 5.2 us device barrier it would otherwise pay.
//
// `hist` is 256 atomics, `gtc`/`eqc` are one uint per thread, `shared` is 2.
inline void select_top_blocks(const device float* scores, uint n, uint k,
                              device uint* out, threadgroup atomic_uint* hist,
                              threadgroup uint* gtc, threadgroup uint* eqc,
                              threadgroup uint* shared, uint tid, uint nt) {
  if (k >= n) {
    for (uint i = tid; i < n; i += nt) out[i] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
  }
  if (tid == 0u) { shared[0] = 0u; shared[1] = k; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint pass = 0; pass < 4u; ++pass) {
    uint shift = 24u - 8u * pass;
    uint hi_mask = (pass == 0u) ? 0u : (0xFFFFFFFFu << (shift + 8u));
    for (uint i = tid; i < 256u; i += nt)
      atomic_store_explicit(hist + i, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint prefix = shared[0];
    for (uint i = tid; i < n; i += nt) {
      uint key = fkey(scores[i]);
      if ((key & hi_mask) != prefix) continue;
      atomic_fetch_add_explicit(hist + ((key >> shift) & 0xFFu), 1u,
                                memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      uint want = shared[1];
      uint cum = 0u, chosen = 0u;
      for (int b = 255; b >= 0; --b) {
        uint c = atomic_load_explicit(hist + b, memory_order_relaxed);
        if (cum + c >= want) { chosen = uint(b); break; }
        cum += c;
      }
      shared[1] = want - cum;          // equal-key slots still to fill
      shared[0] = prefix | (chosen << shift);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint thr = shared[0];
  const uint need = shared[1];
  const uint chunk = (n + nt - 1u) / nt;
  const uint lo = metal::min(tid * chunk, n);
  const uint hi = metal::min(lo + chunk, n);

  uint gt = 0u, eq = 0u;
  for (uint i = lo; i < hi; ++i) {
    uint key = fkey(scores[i]);
    if (key > thr) ++gt; else if (key == thr) ++eq;
  }
  gtc[tid] = gt; eqc[tid] = eq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Exclusive scan.  nt <= 1024, and this runs once per attention layer per
  // token, so the serial scan on one thread is cheaper than a tree over
  // threadgroup memory with its own barriers.
  if (tid == 0u) {
    uint a = 0u, b = 0u;
    for (uint t = 0; t < nt; ++t) {
      uint ga = gtc[t], ea = eqc[t];
      gtc[t] = a; eqc[t] = b;
      a += ga; b += ea;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint gt_before = gtc[tid], eq_before = eqc[tid];
  for (uint i = lo; i < hi; ++i) {
    uint key = fkey(scores[i]);
    if (key > thr) {
      out[gt_before + metal::min(eq_before, need)] = i;
      ++gt_before;
    } else if (key == thr) {
      if (eq_before < need) out[gt_before + eq_before] = i;
      ++eq_before;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""


def kernel_header() -> str:
    """The full MSL header: barrier, matvec helpers, top-block selection."""
    return MEGA_BARRIER + MEGA_QMV.replace("BF", "bfloat16_t") + MEGA_TOPB
