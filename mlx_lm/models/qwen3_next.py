# Copyright © 2025 Apple Inc.

from __future__ import annotations

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
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)


def _env_flag(name: str) -> bool:
    """Read one opt-in performance flag once, at import time."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "on", "yes"}


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
# MLX_QWEN4_MOE_FUSED_GATE_UP: run the routed experts' gate and up
# projections as ONE gather matmul over the re-fused [gate|up] weight (the
# layout the checkpoint ships before sanitize splits it) — halves routed
# dispatches per layer and doubles N-tile fill.  Accumulation grouping in
# the wide matmul may change => tolerance-level lever.
_MOE_FUSED_GATE_UP = _env_flag("MLX_QWEN4_MOE_FUSED_GATE_UP")

# MLX_QWEN4_MOE_SHARED_IN_GATHER: fold the shared expert into the routed
# gather as constant extra expert index E, giving top_k+1 rows through one
# dispatch instead of separate plain matmuls.  The output composition
# (routed weighted sum + sigmoid-gated shared, added in stock order) is
# preserved exactly, but the shared row moves from the plain (q)mm kernel
# family to the gather family, which accumulates differently on M5 (the
# fp32 gap is the NAX TF32 path; measured <= 1.5e-3 of output scale, and
# <= 2e-7 with MLX_ENABLE_TF32=0) => tolerance-level lever, NOT bitwise.
_MOE_SHARED_IN_GATHER = _env_flag("MLX_QWEN4_MOE_SHARED_IN_GATHER")


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


def _gather_table_apply(table, x, idx, do_sort):
    weight, scales, biases, group_size, bits, mode = table
    if scales is None:
        return mx.gather_mm(
            x, weight.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=do_sort
        )
    return mx.gather_qmm(
        x,
        weight,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )


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
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        if (
            cache is not None
            and getattr(cache, "speculating", False)
            and mask is None
            and cache.lengths is None
            and cache.left_padding is None
        ):
            # Record an exact rollback for speculative decoding: replaying the
            # recurrence from the pre-forward state over the first m of the
            # exact per-token inputs the kernel consumes reproduces the state
            # after m tokens bit-for-bit; the conv state after m tokens is a
            # slice of conv_input. See ArraysCache.record_rollback.
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
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        self.sharding_group = None
        # Lazy lever tables (fused [gate|up], shared-as-expert-E), keyed by
        # the identity of the source weight arrays so a later
        # load_weights/update invalidates them (the stale-snapshot lesson).
        # Kept in __dict__ (not Module items) so it never reaches
        # parameters()/state.
        object.__setattr__(self, "_moe_lever_cache", {})

    def _moe_lever_tables(self, fused, folded):
        """Build (or reuse) the lever tables; returns None when ineligible."""
        routed = (
            self.switch_mlp.gate_proj,
            self.switch_mlp.up_proj,
            self.switch_mlp.down_proj,
        )
        shared = (
            self.shared_expert.gate_proj,
            self.shared_expert.up_proj,
            self.shared_expert.down_proj,
        )
        sources = routed + (shared if folded else ())
        key = tuple(part for m in sources for part in _proj_identity(m))
        cached = self._moe_lever_cache.get((fused, folded))
        if (
            cached is not None
            and len(cached[0]) == len(key)
            and all(new is old for new, old in zip(key, cached[0]))
        ):
            return cached[1]

        signatures = [_proj_signature(m) for m in sources]
        eligible = signatures[0] is not None and all(
            s == signatures[0] for s in signatures
        )
        if eligible and folded:
            # The shared expert must be shape-compatible with one routed
            # expert so its tables concatenate as expert index E.
            eligible = all(
                s["weight"].shape == r["weight"].shape[1:]
                for r, s in zip(routed, shared)
            )
        if not eligible:
            tables = None
        else:
            gate, up, down = (_proj_table(m) for m in routed)
            if folded:

                def as_expert_row(module):
                    table = _proj_table(module)
                    return tuple(
                        None if part is None else part[None]
                        for part in table[:3]
                    ) + table[3:]

                gate = _concat_tables([gate, as_expert_row(shared[0])], axis=0)
                up = _concat_tables([up, as_expert_row(shared[1])], axis=0)
                down = _concat_tables([down, as_expert_row(shared[2])], axis=0)
            tables = {"down": down}
            if fused:
                tables["gate_up"] = _concat_tables([gate, up], axis=-2)
            else:
                tables["gate"], tables["up"] = gate, up
        self._moe_lever_cache[(fused, folded)] = (key, tables)
        return tables

    def _moe_lever_forward(self, x, inds, scores, fused, folded, tables):
        top_k = inds.shape[-1]
        idx_all = inds
        if folded:
            shared_col = mx.full(
                inds.shape[:-1] + (1,), self.num_experts, dtype=inds.dtype
            )
            idx_all = mx.concatenate([inds, shared_col], axis=-1)
        xe = mx.expand_dims(x, (-2, -3))
        do_sort = idx_all.size >= 64
        idx = idx_all
        inv_order = None
        if do_sort:
            xe, idx, inv_order = _gather_sort(xe, idx_all)
        if fused:
            gate_up = _gather_table_apply(tables["gate_up"], xe, idx, do_sort)
            hidden = gate_up.shape[-1] // 2
            x_gate, x_up = gate_up[..., :hidden], gate_up[..., hidden:]
        else:
            x_up = _gather_table_apply(tables["up"], xe, idx, do_sort)
            x_gate = _gather_table_apply(tables["gate"], xe, idx, do_sort)
        h = self.switch_mlp.activation(x_up, x_gate)
        if folded:
            y = _gather_table_apply(tables["down"], h, idx, do_sort)
        else:
            y = self.switch_mlp.down_proj(h, idx, sorted_indices=do_sort)
        if do_sort:
            y = _scatter_unsort(y, inv_order, idx_all.shape)
        y = y.squeeze(-2)
        if folded:
            routed = (y[..., :top_k, :] * scores[..., None]).sum(axis=-2)
            shared_y = y[..., top_k, :]
        else:
            routed = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x)
        # Stock composition order: routed sum + sigmoid-gated shared.
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return routed + shared_y

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

        if (
            (_MOE_FUSED_GATE_UP or _MOE_SHARED_IN_GATHER)
            and self.sharding_group is None
            and not self.training
        ):
            fused, folded = _MOE_FUSED_GATE_UP, _MOE_SHARED_IN_GATHER
            tables = self._moe_lever_tables(fused, folded)
            if tables is None and fused and folded:
                # Shared expert ineligible to fold: keep the fusion alone.
                folded = False
                tables = self._moe_lever_tables(fused, folded)
            if tables is not None:
                return self._moe_lever_forward(
                    x, inds, scores, fused, folded, tables
                )

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


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
