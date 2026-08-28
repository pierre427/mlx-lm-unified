# Copyright © 2025 Apple Inc.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache, RotatingKVCache
from .gated_delta import gated_delta_update, normalize_gdn_qk
from .rope_utils import initialize_rope
from . import switch_layers as _switch_layers
from .switch_layers import (
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)


logger = logging.getLogger(__name__)


class MaterializationTooLarge(RuntimeError):
    """A runtime weight materialization does not fit in device memory."""


def _env_flag(name: str, default: bool = False) -> bool:
    """Read one performance flag once, at import time.

    ``default=True`` carries a lever that has been PROMOTED into the shipped
    path: unset means ON, and an operator turns it OFF with ``=0`` instead of
    on with ``=1``.  An unset or empty value takes ``default``; any set value
    is parsed, so ``=0`` reverts a promoted lever.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


# Compiles the expert-selection chain of EVERY user of
# Qwen3NextSparseMoeBlock: qwen3_next itself, qwen3_5, and qwen4_exp.
_MOE_GATE_COMPILE = _env_flag("MLX_QWEN4_MOE_GATE_COMPILE")

# Shaped compile traces are keyed by exact width, and production widths are
# not a small stable set: the final prefill chunk has arbitrary width per
# prompt length, so compiling it would accumulate unbounded traces in a
# long-lived server. Only the stable narrow shapes — decode (1 token) and
# MTP verify (k+1 tokens) — take the compiled path; prefill stays eager.
_MOE_GATE_COMPILE_MAX_TOKENS = 8


# 2026-08-27 decode-decomposition levers (results/qwen38-decode-decomposition
# -20260827.json): the decode GPU window is 86% of the step at only 22% of
# bandwidth — latency/occupancy-bound on tiny 640-wide expert tiles, not
# bandwidth-bound. That reopens tile aggregation: the old fused-projection
# lesson (D13, "same bytes, low single digits") was measured in the
# bandwidth-bound regime and does not govern this operating point.
#
# MLX_QWEN4_MOE_FUSED_GATE_UP: keep gate and up as the one [gate|up]
# tensor the checkpoint ships and run them as ONE gather matmul. Halves
# routed dispatches per layer and doubles N-tile fill. The wide matmul may
# regroup accumulation => tolerance-level lever.
# Default-on 2026-08-28: ~+3.6% prefill, clean greedy digest (unlike
# moe_shared_in_gather, which stays off). A 0.32.2 remeasure is still pending,
# so the magnitude is the 0.32.0 figure; set MLX_QWEN4_MOE_FUSED_GATE_UP=0 to
# revert.
_MOE_FUSED_GATE_UP = _env_flag("MLX_QWEN4_MOE_FUSED_GATE_UP", default=True)

# MLX_QWEN4_MOE_SHARED_IN_GATHER: fold the shared expert into the routed
# table as expert index E, so top_k+1 rows go through one dispatch. The
# output composition is unchanged, but the shared row moves to the gather
# kernel family, which accumulates differently on M5 (<= 1.5e-3 of output
# scale; <= 2e-7 with MLX_ENABLE_TF32=0) => tolerance-level lever.
_MOE_SHARED_IN_GATHER = _env_flag("MLX_QWEN4_MOE_SHARED_IN_GATHER")

# MLX_QWEN4_FUSED_EXPERT_KERNEL: experimental production-shape kernel for the
# remaining routed-MoE boundary. Gate/up stay on MLX's gather path; the q4
# down projection, router weighting, and top-k reduction become one dispatch.
# It is exact-geometry, default-off, and falls back structurally before a
# kernel is emitted. See qwen4_fused_moe.py.
_MOE_FUSED_EXPERT_KERNEL = _env_flag("MLX_QWEN4_FUSED_EXPERT_KERNEL")
_MOE_FUSED_EXPERT_MODES = ("stock", "scalar", "tile4")

# Both MoE levers are LOAD-TIME weight transforms, not runtime tables. A
# runtime re-fusion was OOM-killed on the 104 GB serving artifact: the
# fused [gate|up] set is 944 MB per layer x 48 = 45 GB, a full second copy
# beside the split tensors. See wiki lessons/moe-runtime-fusion-oom.md.


def _proj_signature(module):
    """Eligibility signature of one expert projection module."""
    if "bias" in module:
        return None
    if isinstance(module, (QuantizedSwitchLinear, nn.QuantizedLinear)):
        return (
            "quantized",
            module.group_size,
            module.bits,
            getattr(module, "mode", "affine"),
            getattr(module, "biases", None) is not None,
        )
    if isinstance(module, (SwitchLinear, nn.Linear)):
        return ("float",)
    return None


def _proj_identity(module):
    """Identity key of every array a projection contributes to a table.

    Scales and biases are included so a partial ``update()`` that replaces
    only them (weight untouched) still invalidates the lazy tables.
    """
    return (
        module["weight"],
        getattr(module, "scales", None),
        getattr(module, "biases", None),
    )


def _proj_table(module):
    """(weight, scales, biases, group_size, bits, mode) view of a projection."""
    if _proj_signature(module)[0] == "quantized":
        return (
            module["weight"],
            module["scales"],
            getattr(module, "biases", None),
            module.group_size,
            module.bits,
            getattr(module, "mode", "affine"),
        )
    return (module["weight"], None, None, None, None, None)


def _concat_tables(tables, axis):
    """Concatenate projection tables along one weight axis, exactly.

    Quantization groups run along the input (K) axis of each output row, so
    concatenation along N (axis -2) or along the expert axis (0) preserves
    every group, scale, and bias byte-for-byte; re-fusing the split
    [gate|up] quantized tensors reproduces the checkpoint's fused layout.
    """
    weight = mx.concatenate([t[0] for t in tables], axis=axis)
    scales = biases = None
    if tables[0][1] is not None:
        scales = mx.concatenate([t[1] for t in tables], axis=axis)
        if tables[0][2] is not None:
            biases = mx.concatenate([t[2] for t in tables], axis=axis)
    return (weight, scales, biases, *tables[0][3:])


def table_bytes(table) -> int:
    """Bytes a materialized projection table occupies."""
    return sum(part.nbytes for part in table[:3] if part is not None)


def _fmt_bytes(nbytes: float) -> str:
    """Size text that keeps a small request legible instead of rounding it away."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(nbytes) >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes:.0f} B"


def _working_set_bytes() -> Optional[int]:
    """Recommended working-set size, or None when the device has no such limit.

    Only a POSITIVELY detected absence returns None. A broken API or a
    malformed reply propagates: a memory guard that fails open on its own bugs
    is worse than no guard at all, because it still reads as protection.
    """
    metal = getattr(mx, "metal", None)
    is_available = getattr(metal, "is_available", None)
    if is_available is not None and not is_available():
        return None
    # mx.metal.device_info is deprecated; prefer mx.device_info where it exists
    # so the guard survives the old spelling being removed.
    info = getattr(mx, "device_info", None) or getattr(metal, "device_info", None)
    if info is None:
        return None
    # A device that reports no working-set limit is an absence, not a fault;
    # anything else raising here is a fault and is left to propagate.
    reported = info()
    if "max_recommended_working_set_size" not in reported:
        return None
    budget = reported["max_recommended_working_set_size"]
    # Present but null or non-positive is a MALFORMED reply, not an absence.
    # Do not quietly turn it into "no device": that disables the guard while
    # it still reads as protection.
    if budget is None or budget <= 0:
        raise ValueError(f"device reported an unusable working set: {budget!r}")
    return budget


def materialization_headroom(active: int, budget: int, headroom: float) -> float:
    """Bytes a NEW allocation may claim beside `active` bytes already held.

    The primary rule is the one this guard has always applied: hold `headroom`
    of the recommended working set back from TOTAL occupancy, leaving
    `budget * (1 - headroom) - active` for the request.

    That rule goes unsatisfiable the moment resident weights alone exceed it.
    A 104.3 GB model in a 120.3 GB working set leaves 102.2 - 104.3 < 0, so
    `active + nbytes <= allowed` is false for EVERY nbytes -- zero included --
    and the guard refuses 20 MB tables that plainly fit. Loading the model
    already spent the reserve; refusing a projection table cannot win it back.

    So when, and only when, the reserve is already gone, fall back to a
    DEGRADED allowance: the same fraction of what is still unclaimed, capped at
    `headroom` of the reserve that should have been there. The cap is the point
    -- uncapped, the allowance leaps from ~0 to 15.3 GB the instant a resident
    model crosses the reserve line, so being slightly bigger would buy a much
    larger claim. Capped, that step is 2.7 GB at the shipped constants: enough
    for the incidental per-layer tables this regime exists to stop refusing,
    and nowhere near a deliberate multi-GB materialization.

    The fallback is reachable only where the primary rule refused a request of
    nothing, so no allocation the primary rule judged -- admitted or refused --
    changes verdict.
    """
    if not 0.0 <= headroom < 1.0:
        raise ValueError(f"headroom must be in [0, 1), got {headroom!r}")
    reserved = budget * (1.0 - headroom) - active
    if reserved >= 0:
        # Note >=, not >: at exactly zero the primary rule still admitted a
        # request of nothing, so it must keep governing or that one verdict
        # would change too.
        return reserved
    degraded = max(0, budget - active) * (1.0 - headroom)
    return min(degraded, budget * headroom * headroom)


def check_materialization_budget(nbytes: int, what: str, headroom: float = 0.15):
    """Refuse a runtime materialization that does not fit in memory.

    A runtime weight table is a SECOND copy of resident weights, so size it
    against the remaining working-set allowance before building it. Returns the
    recorded estimate and raises MaterializationTooLarge when it does not fit.
    """
    budget = _working_set_bytes()
    if budget is None:
        return {"bytes": nbytes, "checked": False}
    active = mx.get_active_memory()
    allowed = materialization_headroom(active, budget, headroom)
    estimate = {
        "bytes": nbytes,
        "active_bytes": active,
        "budget_bytes": budget,
        "allowed_bytes": allowed,
        "headroom": headroom,
        "checked": True,
        "fits": nbytes <= allowed,
    }
    logger.debug("materialization estimate for %s: %s", what, estimate)
    if not estimate["fits"]:
        raise MaterializationTooLarge(
            f"{what} needs {_fmt_bytes(nbytes)} beside {_fmt_bytes(active)} "
            f"active; the allowance is {_fmt_bytes(allowed)} of the "
            f"{_fmt_bytes(budget)} recommended working set"
        )
    return estimate


def switch_layers_sort_min() -> int:
    """Read the sorted-gather threshold at call time so an A/B can switch it."""
    return _switch_layers._GATHER_SORT_MIN_ASSIGNMENTS

class FusedGateUpSwitchGLU(nn.Module):
    """SwitchGLU holding gate and up as one [gate|up] projection.

    One gather matmul of width 2*hidden_dims replaces two of width
    hidden_dims. The halves are split from the OUTPUT, so the weights are
    never copied.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(
            hidden_dims, input_dims, num_experts, bias=bias
        )
        self.activation = activation

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        scores: Optional[mx.array] = None,
        variant: str = "scalar",
    ) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= switch_layers_sort_min()
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        half = self.hidden_dims
        hidden = self.activation(gate_up[..., half:], gate_up[..., :half])
        fused = _try_qwen4_fused_down(
            hidden, idx, scores, self.down_proj, do_sort, variant
        )
        if fused is not None:
            return fused
        x = self.down_proj(hidden, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        x = x.squeeze(-2)
        if scores is not None:
            return (x * scores[..., None]).sum(axis=-2)
        return x


class FusedDownSwitchGLU(SwitchGLU):
    """Stock split gate/up projections with the experimental fused down tail."""

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        scores: Optional[mx.array] = None,
        variant: str = "scalar",
    ) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= switch_layers_sort_min()
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        hidden = self.activation(
            self.up_proj(x, idx, sorted_indices=do_sort),
            self.gate_proj(x, idx, sorted_indices=do_sort),
        )
        fused = _try_qwen4_fused_down(
            hidden, idx, scores, self.down_proj, do_sort, variant
        )
        if fused is not None:
            return fused
        x = self.down_proj(hidden, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        x = x.squeeze(-2)
        if scores is not None:
            return (x * scores[..., None]).sum(axis=-2)
        return x


def _try_qwen4_fused_down(
    hidden, indices, scores, down_proj, sorted_indices, variant="scalar"
):
    """Return the fused routed result, or None before dispatch when ineligible."""
    if (
        scores is None
        or sorted_indices
        or down_proj.training
        or not isinstance(down_proj, QuantizedSwitchLinear)
        or "bias" in down_proj
    ):
        return None

    from .qwen4_fused_moe import admit_qwen4_fused_down, qwen4_fused_down

    # SwitchLinear preserves its singleton matrix row as [..., top_k, 1, K].
    # The custom GEMV consumes the equivalent compact [..., top_k, K] view.
    if hidden.ndim < 3 or hidden.shape[-2] != 1:
        return None
    compact_hidden = hidden.squeeze(-2)
    biases = getattr(down_proj, "biases", None)
    admission = admit_qwen4_fused_down(
        compact_hidden,
        indices,
        scores,
        down_proj["weight"],
        down_proj["scales"],
        biases,
        num_experts=down_proj.num_experts,
        group_size=down_proj.group_size,
        bits=down_proj.bits,
        mode=down_proj.mode,
    )
    if not admission.accepted:
        return None
    return qwen4_fused_down(
        compact_hidden,
        indices,
        scores,
        down_proj["weight"],
        down_proj["scales"],
        biases,
        num_experts=down_proj.num_experts,
        group_size=down_proj.group_size,
        bits=down_proj.bits,
        mode=down_proj.mode,
        variant=variant,
    )


def _parts(weights: dict, path: str) -> dict:
    """The weight/scales/biases a module path contributes, if present."""
    found = {}
    for suffix in ("weight", "scales", "biases"):
        key = f"{path}.{suffix}"
        if key in weights:
            found[suffix] = weights[key]
    return found


def _drop(weights: dict, path: str):
    for suffix in ("weight", "scales", "biases"):
        weights.pop(f"{path}.{suffix}", None)


def _store(weights: dict, path: str, parts: dict):
    for suffix, value in parts.items():
        weights[f"{path}.{suffix}"] = value


def _concat_parts(parts_list, axis: int):
    """Concatenate matching part sets, or None when they do not match.

    Affine groups run along each row's K axis, so concatenating along the
    output-row axis or the expert axis keeps every packed value, scale and
    bias byte for byte.
    """
    if not all(parts_list):
        return None
    suffixes = set(parts_list[0])
    if any(set(parts) != suffixes for parts in parts_list):
        return None
    for suffix in suffixes:
        shapes = [parts[suffix].shape for parts in parts_list]
        if len({len(shape) for shape in shapes}) > 1:
            return None
        at = axis % len(shapes[0])
        if len({shape[:at] + shape[at + 1 :] for shape in shapes}) > 1:
            return None
    return {
        suffix: mx.concatenate(
            [parts[suffix] for parts in parts_list], axis=axis
        )
        for suffix in suffixes
    }


def transform_moe_weights(
    weights: dict, prefixes, *, fuse_gate_up: bool, fold_shared: bool
) -> int:
    """Apply the load-time MoE lever transforms in place.

    ``fuse_gate_up`` keeps (or restores) the shipped [gate|up] tensor.
    ``fold_shared`` appends the shared expert as routed expert index E and
    drops the separate shared tensors. Both consume file-backed checkpoint
    arrays and leave ONE resident tensor per projection, so neither adds a
    second full-size copy. Returns the number of layers transformed.
    """
    if not (fuse_gate_up or fold_shared):
        return 0
    changed = 0
    for prefix in prefixes:
        switch = f"{prefix}.switch_mlp"
        shared = f"{prefix}.shared_expert"
        names = ["gate_up_proj"] if fuse_gate_up else ["gate_proj", "up_proj"]
        routed = {}
        if fuse_gate_up:
            fused = _parts(weights, f"{switch}.gate_up_proj") or _concat_parts(
                [
                    _parts(weights, f"{switch}.gate_proj"),
                    _parts(weights, f"{switch}.up_proj"),
                ],
                axis=-2,
            )
            if fused is None:
                continue
            routed["gate_up_proj"] = fused
        else:
            for name in names:
                routed[name] = _parts(weights, f"{switch}.{name}")
        routed["down_proj"] = _parts(weights, f"{switch}.down_proj")
        if not all(routed.values()):
            continue

        if fold_shared:
            shared_parts = {"down_proj": _parts(weights, f"{shared}.down_proj")}
            if fuse_gate_up:
                shared_parts["gate_up_proj"] = _parts(
                    weights, f"{shared}.gate_up_proj"
                ) or _concat_parts(
                    [
                        _parts(weights, f"{shared}.gate_proj"),
                        _parts(weights, f"{shared}.up_proj"),
                    ],
                    axis=-2,
                )
            else:
                for name in names:
                    shared_parts[name] = _parts(weights, f"{shared}.{name}")
            folded = {}
            for name, parts in routed.items():
                one = shared_parts.get(name)
                if not one:
                    folded = None
                    break
                merged = _concat_parts(
                    [parts, {s: v[None] for s, v in one.items()}], axis=0
                )
                if merged is None:
                    folded = None
                    break
                folded[name] = merged
            if folded is None:
                continue
            routed = folded
            for name in ("gate_proj", "up_proj", "gate_up_proj", "down_proj"):
                _drop(weights, f"{shared}.{name}")

        for name in ("gate_proj", "up_proj", "gate_up_proj", "down_proj"):
            _drop(weights, f"{switch}.{name}")
        for name, parts in routed.items():
            _store(weights, f"{switch}.{name}", parts)
        changed += 1
    return changed


# ``shapeless=True`` is rejected here: the top-k slice cannot infer output
# shapes, so this follows the shaped glm4_moe/dots1 compile pattern.
@mx.compile
def _select_experts(gates: mx.array, top_k: int, norm_topk_prob: bool):
    gates = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    head_dim: int
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    full_attention_interval: int = 4
    # Multi-token-prediction (nextn) head. 0 = no head (default; community MLX
    # repacks strip it). Set >0 (typically 1) to build a head for training or a
    # grafted/trained "-mtp" checkpoint, enabling self-speculative decoding.
    mtp_num_hidden_layers: int = 0


@partial(mx.compile, shapeless=True)
def _precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False
        )
        self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(
        self, mixed_qkvz: mx.array, mixed_ba: mx.array
    ) -> mx.array:
        nk, dn, nv, dv = (
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )
        mixed_qkvz = mixed_qkvz.reshape(*mixed_qkvz.shape[:-1], nk, -1)
        mixed_ba = mixed_ba.reshape(*mixed_ba.shape[:-1], nk, -1)
        q, k, v, z = mx.split(mixed_qkvz, [dn, 2 * dn, 2 * dn + nv // nk * dv], axis=-1)
        b, a = mx.split(mixed_ba, [nv // nk], axis=-1)
        return (
            q,
            k,
            v.reshape(*v.shape[:2], -1, dv),
            z.reshape(*z.shape[:2], -1, dv),
            b.reshape(*b.shape[:2], nv),
            a.reshape(*a.shape[:2], nv),
        )

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        q, k = normalize_gdn_qk(q, k)

        # Gate on the geometry being describable per row, not on it being
        # unpadded: a right-padded speculative slab is exactly the case that
        # has to roll back. ``rollback_spans`` returns None where a scalar
        # depth would lie, and the cache credits each row its own span.
        spans = ()
        if cache is not None:
            describe = getattr(cache, "rollback_spans", None)
            if describe is not None:
                spans = describe(S, mask)

        if (
            cache is not None
            and getattr(cache, "speculating", False)
            and spans is not None
        ):
            # Record an exact rollback for speculative decoding: replaying the
            # recurrence from the pre-forward state over the first m of the
            # exact per-token inputs the kernel consumes reproduces the state
            # after m tokens bit-for-bit; the conv state after m tokens is a
            # slice of conv_input. See ArraysCache.record_rollback.
            # The replay is mask-free, which is exact for a row only up to its
            # own span: masked steps are no-ops live, and no row has leading
            # pads here, so positions [0, m) are that row's own tokens.
            n_keep = self.conv_kernel_size - 1
            use_kernel = not self.training

            def _rollback(
                m, q=q, k=k, v=v, a=a, b=b, S0=state, ci=conv_input, nk=n_keep
            ):
                _, s_m = gated_delta_update(
                    q[:, :m],
                    k[:, :m],
                    v[:, :m],
                    a[:, :m],
                    b[:, :m],
                    self.A_log,
                    self.dt_bias,
                    S0,
                    None,
                    use_kernel,
                )
                return [mx.contiguous(ci[:, m : m + nk, :]), s_m]

            cache.record_rollback(S, _rollback, [conv_state, state])

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        # Lever layouts are structural: the module is built to match the
        # transformed checkpoint, so no runtime weight copy is ever made.
        self.fused_gate_up = _MOE_FUSED_GATE_UP
        self.shared_folded = _MOE_SHARED_IN_GATHER and (
            shared_expert_intermediate_size == intermediate_size
        )
        self.fused_expert_kernel_mode = (
            "scalar"
            if _MOE_FUSED_EXPERT_KERNEL and not self.shared_folded
            else "stock"
        )
        if self.fused_gate_up:
            switch_cls = FusedGateUpSwitchGLU
        else:
            # This wrapper is weight-layout-identical to SwitchGLU. Keeping it
            # present in stock mode allows a loaded model to switch modes
            # without rebuilding or reloading any parameter.
            switch_cls = FusedDownSwitchGLU
        self.switch_mlp = switch_cls(
            dim,
            intermediate_size,
            num_experts + (1 if self.shared_folded else 0),
        )
        if not self.shared_folded:
            self.shared_expert = Qwen3NextMLP(
                dim, shared_expert_intermediate_size
            )
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        self.sharding_group = None

    @property
    def fused_expert_kernel_enabled(self):
        return self.fused_expert_kernel_mode != "stock"

    def set_fused_expert_kernel_mode(self, mode: str):
        """Select stock/scalar/tile4 for subsequent forwards without reload."""
        if mode not in _MOE_FUSED_EXPERT_MODES:
            raise ValueError(
                f"unknown fused expert mode {mode!r}; "
                f"expected one of {_MOE_FUSED_EXPERT_MODES}"
            )
        if mode != "stock" and self.shared_folded:
            raise ValueError("fused expert kernels do not support a folded shared row")
        self.fused_expert_kernel_mode = mode

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x)
        if (
            _MOE_GATE_COMPILE
            and gates.size // gates.shape[-1] <= _MOE_GATE_COMPILE_MAX_TOKENS
        ):
            inds, scores = _select_experts(
                gates, self.top_k, bool(self.norm_topk_prob)
            )
        else:
            gates = mx.softmax(gates, axis=-1, precise=True)

            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)

        if self.shared_folded:
            # The shared expert is routed row E; it is added ungated by the
            # router, exactly as the separate shared branch was.
            shared_col = mx.full(
                inds.shape[:-1] + (1,), self.num_experts, dtype=inds.dtype
            )
            rows = self.switch_mlp(
                x, mx.concatenate([inds, shared_col], axis=-1)
            )
            y = (rows[..., : self.top_k, :] * scores[..., None]).sum(axis=-2)
            shared_y = rows[..., self.top_k, :]
        else:
            if self.fused_expert_kernel_enabled and self.sharding_group is None:
                y = self.switch_mlp(
                    x,
                    inds,
                    scores=scores,
                    variant=self.fused_expert_kernel_mode,
                )
            else:
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


def set_qwen4_fused_expert_mode(model: nn.Module, mode: str) -> int:
    """Switch every compatible MoE block in a resident model atomically.

    This changes only Python dispatch state; parameters and MLX arrays are
    untouched. Call it only between requests/benchmark observations because a
    concurrent forward could otherwise see a mixture of old and new modes.
    Returns the number of updated sparse blocks.
    """
    if mode not in _MOE_FUSED_EXPERT_MODES:
        raise ValueError(
            f"unknown fused expert mode {mode!r}; "
            f"expected one of {_MOE_FUSED_EXPERT_MODES}"
        )
    blocks = [
        module
        for _, module in model.named_modules()
        if isinstance(module, Qwen3NextSparseMoeBlock)
    ]
    incompatible = [
        block for block in blocks if mode != "stock" and block.shared_folded
    ]
    if incompatible:
        raise ValueError(
            f"{len(incompatible)} sparse blocks use a folded shared row; "
            "select stock mode"
        )
    for block in blocks:
        block.set_fused_expert_kernel_mode(mode)
    return len(blocks)


def qwen4_fused_expert_mode_counts(model: nn.Module) -> dict[str, int]:
    """Return resident MoE mode counts without evaluating model arrays."""
    counts = {mode: 0 for mode in _MOE_FUSED_EXPERT_MODES}
    for _, module in model.named_modules():
        if isinstance(module, Qwen3NextSparseMoeBlock):
            counts[module.fused_expert_kernel_mode] += 1
    return counts


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3NextGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(args)
        else:
            self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3NextDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class Qwen3NextMTP(nn.Module):
    """Multi-token-prediction (nextn) head for qwen3_next — mirrors the upstream
    Qwen3-Next MTP module and the qwen3_5 port:

        fc([norm(embed(t_{p+1})); norm(hidden_p)]) -> one FULL-ATTENTION decoder
        layer (own KV cache) -> norm -> the trunk's lm_head.

    Predicts token p+2 from the trunk's hidden at p and the committed token
    p+1, enabling self-speculative decoding with no external draft model. Built
    only when args.mtp_num_hidden_layers > 0 (a trained or grafted head)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # layer_idx chosen so is_linear=False -> the MTP block is full attention.
        self.layers = [
            Qwen3NextDecoderLayer(args, layer_idx=args.full_attention_interval - 1)
            for _ in range(args.mtp_num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class Model(nn.Module):
    # The GDN layers record exact rollbacks on their ArraysCache during
    # speculative decoding, making the hybrid cache trimmable (see
    # Qwen3NextGatedDeltaNet.__call__ and ArraysCache.record_rollback).
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.mtp_num_hidden_layers > 0:
            self.mtp = Qwen3NextMTP(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, max_kv_size: Optional[int] = None):
        # Recurrent (GDN) layers keep a fixed-size ArraysCache regardless. Only
        # the full-attention layers grow unbounded, so cap just those with a
        # RotatingKVCache when a budget is given (keep=4 preserves the sink
        # tokens). Quantized KV is not compatible with RotatingKVCache
        # (toQuantized NYI), so callers must not combine --max-kv-size with
        # --kv-bits on this model.
        def attn_cache():
            if max_kv_size is not None:
                return RotatingKVCache(max_size=max_kv_size, keep=4)
            return KVCache()

        return [
            ArraysCache(size=2) if l.is_linear else attn_cache()
            for l in self.layers
        ]

    def logits(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def make_mtp_cache(self):
        return [KVCache() for _ in self.mtp.layers]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """One MTP forward over S positions.

        hidden: [B, S, H] post-final-norm trunk hiddens at positions p..p+S-1.
        tokens: [B, S] the committed/drafted token FOLLOWING each hidden's
        position (p+1..p+S). Returns (logits [B, S, V], post_norm_hidden).
        """
        e = self.mtp.pre_fc_norm_embedding(self.model.embed_tokens(tokens))
        h = self.mtp.pre_fc_norm_hidden(hidden)
        x = self.mtp.fc(mx.concatenate([e, h], axis=-1))
        mask = create_attention_mask(x, mtp_cache[0])
        x = self.mtp.layers[0](x, mask=mask, cache=mtp_cache[0])
        post = self.mtp.norm(x)
        return self.logits(post), post

    def sanitize(self, weights):
        # MTP (nextn) head: keep its tensors only if the checkpoint has them AND
        # this model was built with an MTP module (config mtp_num_hidden_layers
        # > 0); otherwise drop them (and the module) so strict loading stays
        # consistent. No-op for the common MLX repacks that ship no MTP head.
        has_mtp_weights = any("mtp." in k for k in weights)
        if not (has_mtp_weights and getattr(self, "mtp", None) is not None):
            weights = {k: v for k, v in weights.items() if "mtp." not in k}
            if getattr(self, "mtp", None) is not None:
                self.mtp = None

        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
