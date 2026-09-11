"""Rejected default-off Qwen4 S=7 routed-expert union prototype.

The stock sorted gather orders 70 assignments by expert, but it still launches
one QMV per assignment.  This component instead walks the deterministic union
of the seven top-10 lists and keeps each affine-Q4 weight word outside the
query loop.  It consumes the resident fused gate/up and down tables directly;
there is no expert-table gather, concatenation, dequantization, or second
weight layout.

This module is intentionally not wired into ``qwen3_next``.  It is a component
gate for the exact production geometry only.  ``MLX_QWEN4_S7_EXPERT_UNION`` is
off by default, and even enabling it has no effect unless an explicit caller
invokes :func:`qwen4_s7_expert_union`.

The real-weight gate on 2026-09-11 rejected this implementation: its runtime-
compiled QMV arithmetic was not raw-bit equal to MLX's prebuilt gather-QMV and
the component was about 4.06x slower at the observed U=49 operating point.
It remains isolated for negative-result archaeology and must not be wired into
serving.

The seven query-slot bytes require two uint32 words.  Queries 0..3 occupy the
low word and queries 4..6 the high word.  E2 never reduces in union order: it
writes a rounded, router-weighted value into the query's original slot and
then sums slots 0..9 in order.  That ordering is a correctness boundary, not a
performance detail.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import mlx.core as mx


TOKENS = 7
TOP_K = 10
NUM_EXPERTS = 512
HIDDEN_SIZE = 2560
EXPERT_HIDDEN_SIZE = 640
GROUP_SIZE = 64
BITS = 4
PACK_FACTOR = 8
MAX_UNION = TOKENS * TOP_K
_SIMD_WIDTH = 32
_COUNTER_MAX = (1 << 63) - 1


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


_ENABLED = _env_flag("MLX_QWEN4_S7_EXPERT_UNION", default=False)
_STATS = {
    "admission_calls": 0,
    "admitted": 0,
    "disabled": 0,
    "rejected": 0,
    "component_calls": 0,
    "e1_dispatches": 0,
    "e2_dispatches": 0,
}


def _bump(name: str) -> None:
    """Increment a bounded host counter without evaluating an MLX array."""

    _STATS[name] = min(_COUNTER_MAX, _STATS[name] + 1)


def set_qwen4_s7_expert_union(enabled: bool) -> bool:
    """Live-toggle the isolated component's explicit admission gate."""

    global _ENABLED
    _ENABLED = bool(enabled)
    return _ENABLED


def qwen4_s7_expert_union_status(*, reset: bool = False) -> dict:
    """Return reachability receipts; no device value is read or synchronized."""

    report = {"enabled": bool(_ENABLED), "counts": dict(_STATS)}
    if reset:
        for name in _STATS:
            _STATS[name] = 0
    return report


@dataclass(frozen=True)
class ExpertUnionPlan:
    """Host mirror of the kernel's deterministic union and two-word slot map."""

    experts: tuple[int, ...]
    slot_words: tuple[tuple[int, int], ...]

    def slot(self, union_index: int, query: int) -> int | None:
        """Return the original zero-based top-k slot for one query/expert."""

        if not 0 <= query < TOKENS:
            raise IndexError(query)
        low, high = self.slot_words[union_index]
        if query < 4:
            encoded = (low >> (8 * query)) & 0xFF
        else:
            encoded = (high >> (8 * (query - 4))) & 0xFF
        return None if encoded == 0 else encoded - 1


def build_expert_union(
    routes: Iterable[Sequence[int]], *, num_experts: int = NUM_EXPERTS
) -> ExpertUnionPlan:
    """Build the exact query-major union used by both Metal kernels.

    Each query must contain ten distinct experts.  Top-k routing supplies that
    invariant; checking it in this host mirror catches malformed fixtures and
    prevents two slots being OR-ed into one byte.
    """

    rows = tuple(tuple(int(expert) for expert in row) for row in routes)
    if len(rows) != TOKENS or any(len(row) != TOP_K for row in rows):
        raise ValueError(f"routes must have shape ({TOKENS}, {TOP_K})")

    experts: list[int] = []
    slots: list[list[int]] = []
    positions: dict[int, int] = {}
    for query, row in enumerate(rows):
        if len(set(row)) != TOP_K:
            raise ValueError(f"query {query} contains a duplicate expert")
        for slot, expert in enumerate(row):
            if not 0 <= expert < num_experts:
                raise ValueError(
                    f"expert {expert} is outside [0, {num_experts})"
                )
            at = positions.get(expert)
            if at is None:
                at = len(experts)
                positions[expert] = at
                experts.append(expert)
                slots.append([0, 0])
            word = 0 if query < 4 else 1
            shift = 8 * (query if query < 4 else query - 4)
            slots[at][word] |= (slot + 1) << shift

    return ExpertUnionPlan(
        tuple(experts), tuple((words[0], words[1]) for words in slots)
    )


@dataclass(frozen=True)
class S7ExpertUnionAdmission:
    accepted: bool
    reason: str
    tokens: int = 0


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


def _dtype_in(value, allowed: Sequence) -> bool:
    return getattr(value, "dtype", None) in allowed


def _table_check(value, shape, dtype, label: str) -> str | None:
    if _shape(value) != shape:
        return f"{label} shape {_shape(value)} does not match {shape}"
    if getattr(value, "dtype", None) != dtype:
        return f"{label} dtype must be {dtype}"
    return None


def admit_qwen4_s7_expert_union(
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    gate_up_weight: mx.array,
    gate_up_scales: mx.array,
    gate_up_biases: mx.array | None,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array | None,
    *,
    num_experts: int = NUM_EXPERTS,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    mode: str = "affine",
    enabled: bool | None = None,
) -> S7ExpertUnionAdmission:
    """Admit only the deployed Qwen4 affine-Q4 S=7 component geometry.

    The check is shape-only and never evaluates ``indices``.  Top-k uniqueness
    remains the caller/router contract, as it is for the stock switch layer.
    """

    _bump("admission_calls")
    active = _ENABLED if enabled is None else bool(enabled)
    if not active:
        _bump("disabled")
        return S7ExpertUnionAdmission(False, "S=7 expert union is disabled")

    x_shape = _shape(x)
    if x_shape is None or len(x_shape) < 2 or x_shape[-1] != HIDDEN_SIZE:
        _bump("rejected")
        return S7ExpertUnionAdmission(False, "x must end in hidden_size=2560")
    tokens = 1
    for extent in x_shape[:-1]:
        tokens *= extent
    if tokens != TOKENS:
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, f"flattened token width S={tokens} is not S=7", tokens
        )
    routed_shape = x_shape[:-1] + (TOP_K,)
    if _shape(indices) != routed_shape or _shape(scores) != routed_shape:
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "indices and scores must match x prefix plus top_k=10", tokens
        )
    if x.dtype != mx.bfloat16 or scores.dtype != mx.bfloat16:
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "x and scores must be bfloat16", tokens
        )
    if not _dtype_in(indices, (mx.int32, mx.uint32)):
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "indices must be int32 or uint32", tokens
        )
    if num_experts != NUM_EXPERTS:
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "only 512 routed experts are supported", tokens
        )
    if (group_size, bits, mode) != (GROUP_SIZE, BITS, "affine"):
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "only affine q4 with group_size=64 is supported", tokens
        )
    if gate_up_biases is None or down_biases is None:
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "affine q4 requires both bias tables", tokens
        )

    table_specs = (
        (
            gate_up_weight,
            (NUM_EXPERTS, 2 * EXPERT_HIDDEN_SIZE, HIDDEN_SIZE // PACK_FACTOR),
            mx.uint32,
            "gate_up_weight",
        ),
        (
            gate_up_scales,
            (NUM_EXPERTS, 2 * EXPERT_HIDDEN_SIZE, HIDDEN_SIZE // GROUP_SIZE),
            mx.bfloat16,
            "gate_up_scales",
        ),
        (
            gate_up_biases,
            (NUM_EXPERTS, 2 * EXPERT_HIDDEN_SIZE, HIDDEN_SIZE // GROUP_SIZE),
            mx.bfloat16,
            "gate_up_biases",
        ),
        (
            down_weight,
            (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // PACK_FACTOR),
            mx.uint32,
            "down_weight",
        ),
        (
            down_scales,
            (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // GROUP_SIZE),
            mx.bfloat16,
            "down_scales",
        ),
        (
            down_biases,
            (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // GROUP_SIZE),
            mx.bfloat16,
            "down_biases",
        ),
    )
    for value, shape, dtype, label in table_specs:
        error = _table_check(value, shape, dtype, label)
        if error is not None:
            _bump("rejected")
            return S7ExpertUnionAdmission(False, error, tokens)

    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        _bump("rejected")
        return S7ExpertUnionAdmission(False, "MLX Metal is unavailable", tokens)
    if mx.default_device() != mx.gpu:
        _bump("rejected")
        return S7ExpertUnionAdmission(
            False, "the default MLX device is not GPU", tokens
        )
    _bump("admitted")
    return S7ExpertUnionAdmission(True, "eligible", tokens)


_UNION_HELPERS = r"""
    constexpr uint S = 7;
    constexpr uint TOPK = 10;
    constexpr uint MAXU = S * TOPK;
"""


# Arithmetic helpers mirror MLX's affine-Q4 ``qmv_impl``.  Exactness requires
# both its fast-path sixteen-input lane partition and its masked-uint16/scaled
# input arithmetic; a shifted-nibble dot is mathematically equivalent but not
# raw-bit equivalent after float rounding.
_QMV_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float qwen4_s7_qdot4(ushort p, float4 a) {
  return a.x * float(p & 0x000Fu)
       + (a.y / 16.0f) * float(p & 0x00F0u)
       + (a.z / 256.0f) * float(p & 0x0F00u)
       + (a.w / 4096.0f) * float(p & 0xF000u);
}

inline float qwen4_s7_qdot8(uint p, float4 a0, float4 a1) {
  float accum = 0.0f;
  accum += qwen4_s7_qdot4(ushort(p & 0xFFFFu), a0);
  accum += qwen4_s7_qdot4(ushort(p >> 16u), a1);
  return accum;
}

inline float qwen4_s7_qdot16(
    uint2 p, float4 a0, float4 a1, float4 a2, float4 a3) {
  float accum = 0.0f;
  accum += qwen4_s7_qdot4(ushort(p.x & 0xFFFFu), a0);
  accum += qwen4_s7_qdot4(ushort(p.x >> 16u), a1);
  accum += qwen4_s7_qdot4(ushort(p.y & 0xFFFFu), a2);
  accum += qwen4_s7_qdot4(ushort(p.y >> 16u), a3);
  return accum;
}

inline float qwen4_s7_xsum16(
    float4 a0, float4 a1, float4 a2, float4 a3) {
  float sum = 0.0f;
  sum += a0.x + a0.y + a0.z + a0.w;
  sum += a1.x + a1.y + a1.z + a1.w;
  sum += a2.x + a2.y + a2.z + a2.w;
  sum += a3.x + a3.y + a3.z + a3.w;
  return sum;
}
"""


_UNION_BUILD = r"""
    if (tid == 0) {
      uint count = 0;
      for (uint query = 0; query < S; ++query) {
        for (uint slot = 0; slot < TOPK; ++slot) {
          uint elem = query * TOPK + slot;
          uint expert = uint(indices[elem_to_loc(
              elem, indices_shape, indices_strides, indices_ndim)]);
          uint at = count;
          for (uint u = 0; u < count; ++u) {
            if (union_ids[u] == expert) { at = u; break; }
          }
          if (at == count) {
            union_ids[count] = expert;
            union_slot_lo[count] = 0;
            union_slot_hi[count] = 0;
            count += 1;
          }
          uint encoded = slot + 1;
          if (query < 4)
            union_slot_lo[at] |= encoded << (8 * query);
          else
            union_slot_hi[at] |= encoded << (8 * (query - 4));
        }
      }
      union_n[0] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


_E1_SOURCE = _UNION_HELPERS + r"""
    constexpr uint H = 2560;
    constexpr uint EH = 640;
    constexpr uint GU_ROWS = 2 * EH;
    constexpr uint IN_BLOCKS = (H / 8) / 2;
    constexpr uint IN_GROUPS = H / 64;

    uint lane = thread_index_in_simdgroup;
    uint tid = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    threadgroup uint union_ids[MAXU];
    threadgroup uint union_slot_lo[MAXU];
    threadgroup uint union_slot_hi[MAXU];
    threadgroup uint union_n[1];
""" + _UNION_BUILD + r"""

    for (uint u = 0; u < union_n[0]; ++u) {
      uint expert = union_ids[u];
      size_t expert_base = size_t(expert) * GU_ROWS;
      size_t gate_row = expert_base + row;
      size_t up_row = expert_base + EH + row;
      float gate[S];
      float up[S];
#pragma unroll
      for (uint query = 0; query < S; ++query) {
        gate[query] = 0.0f;
        up[query] = 0.0f;
      }

      // QMV affine-Q4 walk: both packed weight words are outside the seven
      // query loop. E1 deliberately evaluates every query for each union
      // expert; active-row branching lost in the earlier megakernel probe.
      const device uint2* gu2 =
          reinterpret_cast<const device uint2*>(gate_up_weight);
      for (uint block = lane; block < IN_BLOCKS; block += 32) {
        uint2 pg = gu2[gate_row * IN_BLOCKS + block];
        uint2 pu = gu2[up_row * IN_BLOCKS + block];
        uint group = block >> 2;
        float gs = float(gate_up_scales[gate_row * IN_GROUPS + group]);
        float gb = float(gate_up_biases[gate_row * IN_GROUPS + group]);
        float us = float(gate_up_scales[up_row * IN_GROUPS + group]);
        float ub = float(gate_up_biases[up_row * IN_GROUPS + group]);
#pragma unroll
        for (uint query = 0; query < S; ++query) {
          const device T* xrow = x + elem_to_loc(
              query * H, x_shape, x_strides, x_ndim);
          size_t base = size_t(block) * 16;
          float4 a0 = float4(
              float(xrow[base + 0]), float(xrow[base + 1]),
              float(xrow[base + 2]), float(xrow[base + 3]));
          float4 a1 = float4(
              float(xrow[base + 4]), float(xrow[base + 5]),
              float(xrow[base + 6]), float(xrow[base + 7]));
          float4 a2 = float4(
              float(xrow[base + 8]), float(xrow[base + 9]),
              float(xrow[base + 10]), float(xrow[base + 11]));
          float4 a3 = float4(
              float(xrow[base + 12]), float(xrow[base + 13]),
              float(xrow[base + 14]), float(xrow[base + 15]));
          float xsum = qwen4_s7_xsum16(a0, a1, a2, a3);
          float gpart = qwen4_s7_qdot16(pg, a0, a1, a2, a3);
          float upart = qwen4_s7_qdot16(pu, a0, a1, a2, a3);
          gate[query] += gs * gpart + gb * xsum;
          up[query] += us * upart + ub * xsum;
        }
      }
#pragma unroll
      for (uint query = 0; query < S; ++query) {
        gate[query] = simd_sum(gate[query]);
        up[query] = simd_sum(up[query]);
      }
      if (lane == 0) {
#pragma unroll
        for (uint query = 0; query < S; ++query) {
          // Two words are mandatory at S=7: queries 4..6 restart at bit 0.
          uint slot1 = query < 4
              ? ((union_slot_lo[u] >> (8 * query)) & 0xFFu)
              : ((union_slot_hi[u] >> (8 * (query - 4))) & 0xFFu);
          if (slot1 == 0) continue;
          // gather_qmv produces T before the compiled SwiGLU reads it.  The
          // compiled activation uses the runtime fast ``metal::exp`` and T
          // intermediates; this is intentionally not the standalone precise
          // sigmoid primitive.
          T gate_value = static_cast<T>(gate[query]);
          T up_value = static_cast<T>(up[query]);
          T e = static_cast<T>(metal::exp(metal::abs(gate_value)));
          T sigmoid = gate_value < T(0)
              ? T(1) / (T(1) + e)
              : T(1) - T(1) / (T(1) + e);
          T silu = gate_value * sigmoid;
          T value = silu * up_value;
          hidden[(query * TOPK + (slot1 - 1)) * EH + row] = value;
        }
      }
    }
"""


_E2_SOURCE = _UNION_HELPERS + r"""
    constexpr uint H = 2560;
    constexpr uint EH = 640;
    constexpr uint DOWN_BLOCKS = (EH / 8) / 2;
    constexpr uint DOWN_GROUPS = EH / 64;

    uint lane = thread_index_in_simdgroup;
    uint tid = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    threadgroup uint union_ids[MAXU];
    threadgroup uint union_slot_lo[MAXU];
    threadgroup uint union_slot_hi[MAXU];
    threadgroup uint union_n[1];
    threadgroup float slot_values[S * TOPK];
    for (uint i = tid; i < S * TOPK; i += 32) slot_values[i] = 0.0f;
""" + _UNION_BUILD + r"""

    for (uint u = 0; u < union_n[0]; ++u) {
      uint expert = union_ids[u];
      size_t weight_row = size_t(expert) * H + row;
      float value[S];
#pragma unroll
      for (uint query = 0; query < S; ++query) value[query] = 0.0f;

      const device uint2* down2 =
          reinterpret_cast<const device uint2*>(down_weight);
      for (uint block = lane; block < DOWN_BLOCKS; block += 32) {
        // One affine-Q4 weight read serves every query that selected expert.
        uint2 packed = down2[weight_row * DOWN_BLOCKS + block];
        uint group = block >> 2;
        float scale = float(
            down_scales[weight_row * DOWN_GROUPS + group]);
        float bias = float(
            down_biases[weight_row * DOWN_GROUPS + group]);
#pragma unroll
        for (uint query = 0; query < S; ++query) {
          uint slot1 = query < 4
              ? ((union_slot_lo[u] >> (8 * query)) & 0xFFu)
              : ((union_slot_hi[u] >> (8 * (query - 4))) & 0xFFu);
          if (slot1 == 0) continue;
          uint elem = query * TOPK + (slot1 - 1);
          const device T* hrow = hidden + elem_to_loc(
              elem * EH, hidden_shape, hidden_strides, hidden_ndim);
          size_t base = size_t(block) * 16;
          float4 a0 = float4(
              float(hrow[base + 0]), float(hrow[base + 1]),
              float(hrow[base + 2]), float(hrow[base + 3]));
          float4 a1 = float4(
              float(hrow[base + 4]), float(hrow[base + 5]),
              float(hrow[base + 6]), float(hrow[base + 7]));
          float4 a2 = float4(
              float(hrow[base + 8]), float(hrow[base + 9]),
              float(hrow[base + 10]), float(hrow[base + 11]));
          float4 a3 = float4(
              float(hrow[base + 12]), float(hrow[base + 13]),
              float(hrow[base + 14]), float(hrow[base + 15]));
          float xsum = qwen4_s7_xsum16(a0, a1, a2, a3);
          float qsum = qwen4_s7_qdot16(packed, a0, a1, a2, a3);
          value[query] += scale * qsum + bias * xsum;
        }
      }
#pragma unroll
      for (uint query = 0; query < S; ++query)
        value[query] = simd_sum(value[query]);
      if (lane == 0) {
#pragma unroll
        for (uint query = 0; query < S; ++query) {
          uint slot1 = query < 4
              ? ((union_slot_lo[u] >> (8 * query)) & 0xFFu)
              : ((union_slot_hi[u] >> (8 * (query - 4))) & 0xFFu);
          if (slot1 == 0) continue;
          uint elem = query * TOPK + (slot1 - 1);
          T expert_value = static_cast<T>(value[query]);
          float score = float(scores[elem_to_loc(
              elem, scores_shape, scores_strides, scores_ndim)]);
          T weighted_value = static_cast<T>(float(expert_value) * score);
          // The union decides read order only. It never decides sum order.
          slot_values[elem] = float(weighted_value);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One lane per query, strictly original top-k slot order. The old M=3
    // megakernel union accumulated above and therefore changed this order.
    if (lane < S) {
      uint query = lane;
      // MLX's length-10 BF16 reduction rounds after each add. Preserve both
      // that boundary and the original slot order, not just mathematical sum.
      T routed = T(0.0f);
#pragma unroll
      for (uint slot = 0; slot < TOPK; ++slot)
        routed = T(float(routed) + slot_values[query * TOPK + slot]);
      out[query * H + row] = routed;
    }
"""


_e1_kernel = None
_e2_kernel = None


def _get_e1_kernel():
    global _e1_kernel
    if _e1_kernel is None:
        _e1_kernel = mx.fast.metal_kernel(
            name="qwen4_s7_expert_union_e1",
            input_names=[
                "x",
                "indices",
                "gate_up_weight",
                "gate_up_scales",
                "gate_up_biases",
            ],
            output_names=["hidden"],
            header=_QMV_HEADER,
            source=_E1_SOURCE,
            ensure_row_contiguous=False,
        )
    return _e1_kernel


def _get_e2_kernel():
    global _e2_kernel
    if _e2_kernel is None:
        _e2_kernel = mx.fast.metal_kernel(
            name="qwen4_s7_expert_union_e2",
            input_names=[
                "hidden",
                "indices",
                "scores",
                "down_weight",
                "down_scales",
                "down_biases",
            ],
            output_names=["out"],
            header=_QMV_HEADER,
            source=_E2_SOURCE,
            ensure_row_contiguous=False,
        )
    return _e2_kernel


def qwen4_s7_expert_union(
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    gate_up_weight: mx.array,
    gate_up_scales: mx.array,
    gate_up_biases: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
    *,
    num_experts: int = NUM_EXPERTS,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    mode: str = "affine",
    enabled: bool | None = None,
) -> mx.array:
    """Run the isolated two-dispatch S=7 component or refuse before launch."""

    admission = admit_qwen4_s7_expert_union(
        x,
        indices,
        scores,
        gate_up_weight,
        gate_up_scales,
        gate_up_biases,
        down_weight,
        down_scales,
        down_biases,
        num_experts=num_experts,
        group_size=group_size,
        bits=bits,
        mode=mode,
        enabled=enabled,
    )
    if not admission.accepted:
        raise ValueError(f"Qwen4 S=7 expert union is not eligible: {admission.reason}")

    _bump("component_calls")
    hidden = _get_e1_kernel()(
        inputs=[
            x,
            indices,
            gate_up_weight,
            gate_up_scales,
            gate_up_biases,
        ],
        template=[("T", x.dtype)],
        grid=(_SIMD_WIDTH, EXPERT_HIDDEN_SIZE, 1),
        threadgroup=(_SIMD_WIDTH, 1, 1),
        output_shapes=[(TOKENS, TOP_K, EXPERT_HIDDEN_SIZE)],
        output_dtypes=[x.dtype],
    )[0]
    _bump("e1_dispatches")
    out = _get_e2_kernel()(
        inputs=[
            hidden,
            indices,
            scores,
            down_weight,
            down_scales,
            down_biases,
        ],
        template=[("T", x.dtype)],
        grid=(_SIMD_WIDTH, HIDDEN_SIZE, 1),
        threadgroup=(_SIMD_WIDTH, 1, 1),
        output_shapes=[(TOKENS, HIDDEN_SIZE)],
        output_dtypes=[x.dtype],
    )[0]
    _bump("e2_dispatches")
    return out.reshape(x.shape)


__all__ = [
    "BITS",
    "EXPERT_HIDDEN_SIZE",
    "ExpertUnionPlan",
    "GROUP_SIZE",
    "HIDDEN_SIZE",
    "MAX_UNION",
    "NUM_EXPERTS",
    "S7ExpertUnionAdmission",
    "TOKENS",
    "TOP_K",
    "admit_qwen4_s7_expert_union",
    "build_expert_union",
    "qwen4_s7_expert_union",
    "qwen4_s7_expert_union_status",
    "set_qwen4_s7_expert_union",
]
