# Copyright © 2025 Apple Inc.
# Metal reduction and precision structure adapted from mlx-vlm #2105
# (Copyright © 2025 Prince Canuma, MIT).

"""Default-off fused Qwen4-Exp GDN operator for the speculative-verify block.

The single-token kernel in :mod:`qwen4_fused_gdn` refuses every forward on a
speculating cache, so it never engages on the self-MTP verify forward that
dominates a speculative decode cycle. This module fuses the same
post-projection chain (depthwise causal convolution, SiLU, direct-L2 q/k
normalization, decay and beta gates, the fp32 delta recurrence and the gated
RMSNorm) across ``2 <= S <= MAX_VERIFY_STEPS`` sequential tokens in one
dispatch, and additionally emits the restore points the replay-closure
rollback contract of ``ArraysCache.record_rollback`` needs:

* ``state_snapshots[:, p]``: the recurrent state after ``p + 1`` tokens;
* ``conv_snapshots[:, p]``: the convolution window after ``p + 1`` tokens, the
  same rows the stock path slices out of ``conv_input``.

The layer records ``fn(m) -> [conv_snapshots[:, m-1], state_snapshots[:, m-1]]``
instead of the stock replay closure, so a rejection restores from the
kernel's own per-token states with no recomputation.

Every unary boundary uses the form an exhaustive sweep over all finite bf16
values proved exact against the stock MLX op (see
``tests/test_qwen4_fused_gdn_verify_metal.py``): SiLU as
``x * mlx_sigmoid_fast(x)``, the beta gate as ``mlx_sigmoid_precise<bf16>``,
the float32 output gate as ``mlx_sigmoid_precise<float>``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import Any, Optional

import mlx.core as mx

from .qwen4_fused_gdn import (
    _HEADER,
    _THREADGROUP_Y_CANDIDATES,
    CONV_DIM,
    CONV_KERNEL,
    KEY_HEAD_DIM,
    NUM_KEY_HEADS,
    NUM_VALUE_HEADS,
    VALUE_DIM,
    VALUE_HEAD_DIM,
    FusedGdnAdmission,
    _dtype,
    _shape,
    fused_gdn_runtime_supported,
    probe_qwen4_fused_gdn_decode,
)

logger = logging.getLogger(__name__)

# Production self-MTP verifies ``k + 1`` tokens (k = 2 on Flash-Next); adaptive
# prompt lookup proposes spans up to 16 wide, and the bound is what decides
# whether those reach this kernel at all.
#
# Nothing in the kernel's geometry depends on ``S``: the threadgroup is
# ``(32, TY, 1)`` over one value head, the register tile is ``st[DV/TY][DK/32]``
# and every threadgroup array is sized by ``DK``/``DV``. ``S`` is a template
# constant that sets the trip count of the token loop and the leading extent of
# the two snapshot outputs. So raising the bound compiles more specializations
# (one per width) and holds more snapshot memory until the accept boundary --
# ``state_snapshots`` is ``(S - 1) * HV * DV * DK`` float32, about 3 MiB per
# extra step per linear layer -- but it does not change the arithmetic of any
# width that was already admitted. The kernel source is pinned by hash in
# ``tests/test_qwen4_fused_gdn_verify_contract.py`` to keep that true.
MAX_VERIFY_STEPS = 16


def admit_qwen4_fused_gdn_verify(
    *,
    qkv: Any,
    z: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_weight: Any,
    A_log: Any,
    dt_bias: Any,
    norm_weight: Any,
    mask: Any,
    spans: Any,
    speculating: bool,
    training: bool,
    sharded: bool,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel: int,
    gate_activation: str,
) -> FusedGdnAdmission:
    """Pure structural admission for a B=1 verify block; no MLX evaluation."""
    if training:
        return FusedGdnAdmission(False, "training")
    if sharded:
        return FusedGdnAdmission(False, "distributed sharding")
    if not speculating:
        return FusedGdnAdmission(False, "not a speculative verify")
    # ``spans`` is the cache's host-side rollback geometry for this forward
    # (``ArraysCache.rollback_spans``). The kernel is mask-free, so only a
    # slab whose single row advances every position is exact: ``()`` (no
    # length metadata, no mask) or a one-row span equal to the width, which
    # is how a ragged engine describes a fully valid lane; its mask, derived
    # from that same metadata, is then all ones. A shorter span is right
    # padding and ``None`` is geometry the cache cannot describe.
    if spans is None:
        return FusedGdnAdmission(False, "rollback geometry not describable")
    if spans == ():
        if mask is not None:
            return FusedGdnAdmission(False, "masked verify")
    elif len(spans) != 1 or int(spans[0]) != int(qkv.shape[1]):
        return FusedGdnAdmission(False, "padded rollback geometry")
    if gate_activation != "sigmoid":
        return FusedGdnAdmission(False, f"output gate {gate_activation!r}")

    geometry = (
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel,
    )
    if geometry != (
        NUM_KEY_HEADS,
        NUM_VALUE_HEADS,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        CONV_KERNEL,
    ):
        return FusedGdnAdmission(False, f"unsupported geometry {geometry}")

    qkv_shape = _shape(qkv)
    if len(qkv_shape) != 3 or qkv_shape[0] != 1 or qkv_shape[2] != CONV_DIM:
        return FusedGdnAdmission(
            False, f"qkv shape {qkv_shape}, expected (1, S, {CONV_DIM})"
        )
    steps = qkv_shape[1]
    if steps < 2:
        return FusedGdnAdmission(False, f"verify width {steps} below 2")
    if steps > MAX_VERIFY_STEPS:
        return FusedGdnAdmission(
            False, f"verify width {steps} above {MAX_VERIFY_STEPS}"
        )

    expected = {
        "z": (1, steps, VALUE_DIM),
        "a": (1, steps, NUM_VALUE_HEADS),
        "b": (1, steps, NUM_VALUE_HEADS),
        "conv_state": (1, CONV_KERNEL - 1, CONV_DIM),
        "recurrent_state": (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
        "conv_weight": (CONV_DIM, CONV_KERNEL, 1),
        "A_log": (NUM_VALUE_HEADS,),
        "dt_bias": (NUM_VALUE_HEADS,),
        "norm_weight": (VALUE_HEAD_DIM,),
    }
    values = {
        "z": z,
        "a": a,
        "b": b,
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "conv_weight": conv_weight,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "norm_weight": norm_weight,
    }
    for name, expected_shape in expected.items():
        if _shape(values[name]) != expected_shape:
            return FusedGdnAdmission(
                False,
                f"{name} shape {_shape(values[name])}, expected {expected_shape}",
            )

    value_dtype = _dtype(qkv)
    if value_dtype != mx.bfloat16:
        return FusedGdnAdmission(False, f"unsupported activation dtype {value_dtype}")
    for name in ("z", "a", "b", "conv_state", "conv_weight", "dt_bias", "norm_weight"):
        if _dtype(values[name]) != value_dtype:
            return FusedGdnAdmission(False, f"{name} dtype {_dtype(values[name])}")
    if _dtype(recurrent_state) != mx.float32:
        return FusedGdnAdmission(False, "recurrent_state must be float32")
    if _dtype(A_log) not in (value_dtype, mx.float32):
        return FusedGdnAdmission(False, f"A_log dtype {_dtype(A_log)}")
    return FusedGdnAdmission(True, "eligible")


# One threadgroup per value head, exactly like the single-token kernel; the
# token loop keeps the recurrent state in registers between steps, and every
# per-token phase reproduces the single-token kernel's rounding boundaries and
# reduction order. Window row ``r`` of the causal convolution input is
# ``conv_state[r]`` for ``r < K - 1`` and ``qkv[r - (K - 1)]`` otherwise.
_SOURCE = r"""
  const uint hv = threadgroup_position_in_grid.z;
  const uint hk = hv / RATIO;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty = thread_position_in_threadgroup.y;
  const uint tid = thread_index_in_threadgroup;

  constexpr int NT = 32 * TY;
  constexpr int NDK = DK / 32;
  constexpr int NDV = DV / TY;
  constexpr uint KD = (uint)(HK * DK);
  constexpr uint VD = (uint)(HV * DV);
  constexpr uint CD = 2u * KD + VD;
  constexpr uint KEEP = (uint)K - 1u;
  constexpr uint SNAPS = (uint)S - 1u;

  threadgroup float sq[DK];
  threadgroup float sk[DK];
  threadgroup T sq_squared[DK];
  threadgroup T sk_squared[DK];
  threadgroup float sv[DV];
  threadgroup float sy[DV];
  threadgroup float shr[4];

  device const float* si = recurrent_state + (size_t)hv * DV * DK;
  device float* so = recurrent_state_out + (size_t)hv * DV * DK;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i)
      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];
  }

  const bool owns_shared = (hv % RATIO) == 0u;

  // Convolution window bookkeeping is token independent: publish the final
  // window (the next conv cache) and every intermediate window the layer
  // records as a restore point.
  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
    uint part = idx / (uint)DK;
    uint d = idx - part * (uint)DK;
    uint c = part == 0u ? hk * DK + d
           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
    if (part == 2u || owns_shared) {
      for (uint tap = 0; tap < KEEP; ++tap) {
        uint row = (uint)S + tap;
        conv_state_out[(size_t)tap * CD + c] =
            row < KEEP ? conv_state[(size_t)row * CD + c]
                       : qkv[(size_t)(row - KEEP) * CD + c];
      }
      for (uint p = 1; p <= SNAPS; ++p) {
        for (uint tap = 0; tap < KEEP; ++tap) {
          uint row = p + tap;
          conv_snapshots[((size_t)(p - 1u) * KEEP + tap) * CD + c] =
              row < KEEP ? conv_state[(size_t)row * CD + c]
                         : qkv[(size_t)(row - KEEP) * CD + c];
        }
      }
    }
  }

  for (uint t = 0; t < (uint)S; ++t) {
    for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
      uint part = idx / (uint)DK;
      uint d = idx - part * (uint)DK;
      uint c = part == 0u ? hk * DK + d
             : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
      device const T* wc = conv_weight + (size_t)c * K;
      float acc = 0.0f;
      for (uint tap = 0; tap < (uint)K; ++tap) {
        uint row = t + tap;
        T xv = row < KEEP ? conv_state[(size_t)row * CD + c]
                          : qkv[(size_t)(row - KEEP) * CD + c];
        acc += float(xv) * float(wc[tap]);
      }
      T xb = static_cast<T>(acc);
      // nn.silu is reproduced by the fast sigmoid form on every finite bf16.
      T sl = xb * mlx_sigmoid_fast(xb);
      if (part == 0u) sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else sv[d] = float(sl);
    }

    if (tid == 0u) {
      T av = a[t * HV + hv] + dt_bias[hv];
      T sp = mlx_softplus_fast(av);
      shr[2] = metal::precise::exp(
          -metal::precise::exp(float(A_log[hv])) * float(sp));
      // mx.sigmoid on bf16 is the precise form on every finite bf16 input;
      // the fast form differs on one (x ~ -6.85), which real activations reach.
      shr[3] = float(mlx_sigmoid_precise(b[t * HV + hv]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < (uint)DK; d += NT) {
      T qv = static_cast<T>(sq[d]);
      T kv = static_cast<T>(sk[d]);
      sq_squared[d] = static_cast<T>(qv * qv);
      sk_squared[d] = static_cast<T>(kv * kv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0u) {
      T pq = static_cast<T>(0), pk = static_cast<T>(0);
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) {
        pq = static_cast<T>(sq_squared[base + i] + pq);
        pk = static_cast<T>(sk_squared[base + i] + pk);
      }
      pq = static_cast<T>(simd_sum(float(pq)));
      pk = static_cast<T>(simd_sum(float(pk)));
      if (lane == 0u) {
        T eps = static_cast<T>(1.0e-6f);
        T qdenom = pq + eps;
        T kdenom = pk + eps;
        shr[0] = float(static_cast<T>(metal::precise::rsqrt(qdenom)));
        shr[1] = float(static_cast<T>(metal::precise::rsqrt(kdenom)));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T qscale = static_cast<T>(0.08838834764831845f);
    for (uint d = tid; d < (uint)DK; d += NT) {
      T q_normalized = static_cast<T>(static_cast<T>(sq[d]) * static_cast<T>(shr[0]));
      T k_normalized = static_cast<T>(static_cast<T>(sk[d]) * static_cast<T>(shr[1]));
      sq[d] = float(static_cast<T>(q_normalized * qscale));
      sk[d] = float(k_normalized);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* state_dst =
        t < SNAPS ? state_snapshots + ((size_t)t * HV + hv) * DV * DK : so;
    for (int j = 0; j < NDV; ++j) {
      uint dv = ty + (uint)TY * (uint)j;
      float kv = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] * shr[2];
        kv += st[j][i] * sk[s];
      }
      kv = simd_sum(kv);
      float delta = (sv[dv] - kv) * shr[3];
      float out = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] + sk[s] * delta;
        out += st[j][i] * sq[s];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0u)
        sy[dv] = float(static_cast<T>(out));
      for (int i = 0; i < NDK; ++i)
        state_dst[(size_t)dv * DK + NDK * lane + i] = st[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0u) {
      float po = 0.0f;
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];
      po = simd_sum(po);
      if (lane == 0u)
        shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < (uint)DV; d += NT) {
      T normalized = static_cast<T>(sy[d] * shr[0]);
      normalized = norm_weight[d] * normalized;
      // float32 sigmoid of a bf16-valued gate: the precise form matches
      // mx.sigmoid on every finite bf16 input; the fast form differs on ~1%.
      float x = float(normalized) *
                mlx_sigmoid_precise<float>(float(z[t * VD + hv * DV + d]));
      output[t * VD + hv * DV + d] = static_cast<T>(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_verify",
        input_names=[
            "qkv",
            "z",
            "b",
            "a",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
            "norm_eps",
        ],
        output_names=[
            "output",
            "conv_state_out",
            "recurrent_state_out",
            "state_snapshots",
            "conv_snapshots",
        ],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def qwen4_fused_gdn_verify(
    qkv,
    z,
    b,
    a,
    conv_state,
    conv_weight,
    A_log,
    dt_bias,
    recurrent_state,
    norm_weight,
    norm_eps: float,
    *,
    threadgroup_y: int,
):
    """Build the fused verify graph. Callers must run structural admission first.

    Returns ``(output, conv_state_out, recurrent_state_out, state_snapshots,
    conv_snapshots)``; ``state_snapshots[:, p]`` and ``conv_snapshots[:, p]``
    are the recurrent state and convolution window after ``p + 1`` tokens for
    ``p`` in ``range(S - 1)``.
    """
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; "
            f"expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    steps = int(qkv.shape[1])
    if not 2 <= steps <= MAX_VERIFY_STEPS:
        raise ValueError(
            f"unsupported verify width {steps}; expected 2..{MAX_VERIFY_STEPS}"
        )
    outputs = _kernel()(
        inputs=[
            qkv,
            z,
            b,
            a,
            conv_state,
            conv_weight,
            A_log,
            dt_bias,
            recurrent_state,
            norm_weight,
            float(norm_eps),
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("S", steps),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, CONV_KERNEL - 1, CONV_DIM),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, mx.float32, qkv.dtype],
    )
    return tuple(outputs)


_PROBED_STEPS: dict[int, Optional[int]] = {}
_PROBE_LOCK = Lock()


def probe_qwen4_fused_gdn_verify(dtype, steps: int) -> Optional[int]:
    """Compile-and-run one verify specialization once per width.

    The decode probe's published ``threadgroup_y`` is tried first, then every
    smaller candidate, like the decode ladder. A width whose specializations
    all fail stays on the stock path. Cached widths are read lock-free.
    """
    steps = int(steps)
    if steps in _PROBED_STEPS:
        return _PROBED_STEPS[steps]
    with _PROBE_LOCK:
        if steps in _PROBED_STEPS:
            return _PROBED_STEPS[steps]
        if not 2 <= steps <= MAX_VERIFY_STEPS or not fused_gdn_runtime_supported():
            _PROBED_STEPS[steps] = None
            return None
        start = probe_qwen4_fused_gdn_decode(dtype)
        if start is None:
            _PROBED_STEPS[steps] = None
            return None

        qkv = mx.zeros((1, steps, CONV_DIM), dtype=dtype)
        z = mx.zeros((1, steps, VALUE_DIM), dtype=dtype)
        gates = mx.zeros((1, steps, NUM_VALUE_HEADS), dtype=dtype)
        conv_state = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        conv_weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)
        recurrent_state = mx.zeros(
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
        )
        vector = mx.zeros((NUM_VALUE_HEADS,), dtype=dtype)
        A_log = mx.zeros((NUM_VALUE_HEADS,), dtype=mx.float32)
        norm_weight = mx.ones((VALUE_HEAD_DIM,), dtype=dtype)
        result: Optional[int] = None
        for threadgroup_y in [c for c in _THREADGROUP_Y_CANDIDATES if c <= start]:
            try:
                outputs = qwen4_fused_gdn_verify(
                    qkv,
                    z,
                    gates,
                    gates,
                    conv_state,
                    conv_weight,
                    A_log,
                    vector,
                    recurrent_state,
                    norm_weight,
                    1.0e-6,
                    threadgroup_y=threadgroup_y,
                )
                mx.eval(*outputs)
                result = threadgroup_y
                break
            except ValueError as exc:
                if "threads per threadgroup" in str(exc):
                    continue
                logger.info("Qwen4 fused GDN verify probe failed: %s", exc)
                break
            except RuntimeError as exc:
                logger.info(
                    "Qwen4 fused GDN verify width %d unavailable at "
                    "threadgroup_y=%d: %s",
                    steps,
                    threadgroup_y,
                    exc,
                )
                continue
        _PROBED_STEPS[steps] = result
        return result


__all__ = [
    "MAX_VERIFY_STEPS",
    "admit_qwen4_fused_gdn_verify",
    "probe_qwen4_fused_gdn_verify",
    "qwen4_fused_gdn_verify",
]
