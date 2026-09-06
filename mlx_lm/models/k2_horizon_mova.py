# Copyright © 2026 mlx-uag lab
#
# IFM K2-Horizon-MoVA (`k2_horizon_mova` / K2HorizonForCausalLM).
#
# A Llama-shaped decoder with three departures:
#   * grouped RMSNorm (`layernorm_num_groups` contiguous channel groups, one
#     shared weight vector),
#   * sigmoid-routed FFN experts plus one shared expert, where the router
#     bias only affects expert *selection* and not the mixing weights,
#   * "MoVA" attention on the sparse layers: `v_proj` is replaced by a
#     second sigmoid router over `SwitchLinear` value experts (silu on each
#     expert output), and every attention layer has a softplus output gate
#     applied per head before `o_proj`.
#
# Attention itself is plain GQA + RoPE through `base.scaled_dot_product_attention`
# on a `KVCache`, so quantized KV, prefix trimming and prompt-lookup decoding
# come from the shared paths. Numerics mirror the checkpoint's vendored
# reference (`k2_horizon_mova_mlx.py`) op for op; tests assert bit identity.

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from . import switch_layers
from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU, SwitchLinear, _gather_sort, _scatter_unsort


def _env_flag(name: str, default: bool = False) -> bool:
    """Read one performance flag once, at import time (qwen3_next convention)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


# MLX_K2_ROUTE_COMPILE: mx.compile the array tail of `sigmoid_route` (sigmoid,
# top-k, normalize). It fires twice per sparse layer -- the MoE gate and the
# MoVA v_router -- so it is the model's biggest uncompiled decode cost. Only the
# stable narrow widths (decode + PLD-verify) take it; prefill stays eager to
# avoid unbounded traces. Pure op reorder -> gated by a CPU bit-identity test.
_K2_ROUTE_COMPILE = _env_flag("MLX_K2_ROUTE_COMPILE")
_ROUTE_COMPILE_MAX_TOKENS = 8

# MLX_K2_EAGER_DISPATCH: after each decoder layer, `mx.async_eval` the running
# hidden so the GPU runs layer i while Python builds layer i+1. Pure scheduling,
# bit-identical. Gated to small row counts (decode / verify slabs only).
_K2_EAGER_DISPATCH = _env_flag("MLX_K2_EAGER_DISPATCH")
_K2_EAGER_DISPATCH_MAX_ROWS = max(
    1, int(os.environ.get("MLX_K2_EAGER_DISPATCH_MAX_ROWS", "64"))
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    mova_num_experts: int
    mova_num_experts_per_tok: int
    rms_norm_eps: float
    vocab_size: int
    num_shared_experts: int = 1
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[List[int]] = None
    head_dim: Optional[int] = None
    rope_head_dim: Optional[int] = None
    max_position_embeddings: int = 524288
    norm_topk_prob: bool = True
    router_score_func: str = "sigmoid"
    router_scaling_factor: float = 1.0
    tie_word_embeddings: bool = False
    layernorm_num_groups: int = 1
    rope_theta: float = 10_000_000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False
    moe_gate_bias: bool = True
    attention_gate_func: Optional[str] = None

    def __post_init__(self):
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_head_dim is None:
            self.rope_head_dim = self.head_dim
        if self.rope_head_dim != self.head_dim:
            raise NotImplementedError(
                "k2_horizon_mova: partial rotary (rope_head_dim != head_dim) is not supported"
            )
        # HF stores theta/type under rope_parameters; the vendored reference reads
        # a flat rope_theta. Accept both, flat key wins when explicitly present.
        if self.rope_parameters:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            rope_type = self.rope_parameters.get("rope_type", "default")
            if rope_type != "default" and self.rope_scaling is None:
                self.rope_scaling = dict(self.rope_parameters)
        if self.router_score_func != "sigmoid":
            raise NotImplementedError(
                f"k2_horizon_mova: router_score_func={self.router_score_func!r} is not supported"
            )
        if self.attention_gate_func not in (None, "silu", "softplus"):
            raise NotImplementedError(
                f"k2_horizon_mova: attention_gate_func={self.attention_gate_func!r} is not supported"
            )
        if self.hidden_size % self.layernorm_num_groups:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by "
                f"layernorm_num_groups {self.layernorm_num_groups}"
            )


class GroupRMSNorm(nn.Module):
    """RMSNorm over `groups` contiguous channel groups with one full-width weight."""

    def __init__(self, dims: int, eps: float, groups: int):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


def _sigmoid_route_core(
    logits: mx.array, bias: mx.array, top_k: int, normalize: bool, has_bias: bool
):
    """Array-only tail of `sigmoid_route`, safe to mx.compile. `has_bias` is a
    Python constant so the compiled trace omits the bias ops when there is none;
    `bias` is then an unused placeholder."""
    if has_bias:
        logits = logits - bias
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice = scores + bias.astype(scores.dtype) if has_bias else scores
    indices = mx.argpartition(choice, kth=-top_k, axis=-1)[..., -top_k:]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights, indices


_sigmoid_route_core_compiled = mx.compile(_sigmoid_route_core)


def sigmoid_route(
    x: mx.array, gate: nn.Module, top_k: int, normalize: bool, scale: float
):
    """K2 sigmoid router. The gate bias only affects top-k selection."""
    # Call the module so this also works once `gate` is a QuantizedLinear.
    logits = gate(x)
    has_bias = "bias" in gate
    bias = gate.bias if has_bias else logits  # placeholder when bias-free
    n_tokens = logits.size // logits.shape[-1]
    core = (
        _sigmoid_route_core_compiled
        if (_K2_ROUTE_COMPILE and n_tokens <= _ROUTE_COMPILE_MAX_TOKENS)
        else _sigmoid_route_core
    )
    weights, indices = core(logits, bias, top_k, normalize, has_bias)
    return weights.astype(x.dtype) * scale, indices


def attention_gate(gate: mx.array, func: str) -> mx.array:
    if func == "silu":
        return nn.silu(gate)
    # softplus with beta = ln 2, as in the reference implementation.
    beta = math.log(2.0)
    return mx.log1p(mx.exp(gate * beta)) / beta


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, mova: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.mova = mova

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        if mova:
            self.top_k = args.mova_num_experts_per_tok
            self.router_scale = args.router_scaling_factor
            self.v_router = nn.Linear(
                dim, args.mova_num_experts, bias=args.moe_gate_bias
            )
            self.v_experts = SwitchLinear(
                dim, self.n_kv_heads * self.head_dim, args.mova_num_experts, bias=False
            )
        else:
            self.v_proj = nn.Linear(
                dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
            )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

        self.gate_func = args.attention_gate_func
        if self.gate_func is not None:
            self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def _mova_values(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = sigmoid_route(
            flat, self.v_router, self.top_k, True, self.router_scale
        )
        # Same expert-major sorted gather (and A/B threshold) as SwitchGLU.
        xe = mx.expand_dims(flat, (-2, -3))
        do_sort = indices.size >= switch_layers._GATHER_SORT_MIN_ASSIGNMENTS
        idx = indices
        if do_sort:
            xe, idx, inv_order = _gather_sort(xe, indices)
        routed = self.v_experts(xe, idx, sorted_indices=do_sort)
        if do_sort:
            routed = _scatter_unsort(routed, inv_order, indices.shape)
        routed = routed.squeeze(-2)
        return (nn.silu(routed) * weights[..., None]).sum(axis=-2)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self._mova_values(x) if self.mova else self.v_proj(x)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        if self.gate_func is not None:
            gate = self.gate_proj(x).reshape(B, L, self.n_heads, self.head_dim)
            output = output * attention_gate(gate, self.gate_func).transpose(0, 2, 1, 3)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.router_scale = args.router_scaling_factor
        self.gate = nn.Linear(
            args.hidden_size, args.num_experts, bias=args.moe_gate_bias
        )
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts, bias=False
        )
        self.shared_experts = MLP(
            args.hidden_size, args.moe_intermediate_size * args.num_shared_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        weights, indices = sigmoid_route(
            x, self.gate, self.top_k, self.norm_topk_prob, self.router_scale
        )
        y = self.switch_mlp(x, indices)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        sparse = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.self_attn = Attention(args, mova=sparse and args.mova_num_experts > 0)
        self.mlp = (
            SparseMoE(args) if sparse else MLP(args.hidden_size, args.intermediate_size)
        )
        self.input_layernorm = GroupRMSNorm(
            args.hidden_size, args.rms_norm_eps, args.layernorm_num_groups
        )
        self.post_attention_layernorm = GroupRMSNorm(
            args.hidden_size, args.rms_norm_eps, args.layernorm_num_groups
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class K2HorizonModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GroupRMSNorm(
            args.hidden_size, args.rms_norm_eps, args.layernorm_num_groups
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        eager = _K2_EAGER_DISPATCH and (
            h.shape[0] * h.shape[1] <= _K2_EAGER_DISPATCH_MAX_ROWS
        )
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
            if eager:
                mx.async_eval(h)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        weights.pop("model.rotary_emb.inv_freq", None)
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # HF-style per-expert tensors -> stacked SwitchLinear tensors. MLX
        # exports already ship the stacked form and pass through untouched.
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            if f"{prefix}.mlp.experts.0.up_proj.weight" in weights:
                for n in ("up_proj", "down_proj", "gate_proj"):
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
            if f"{prefix}.self_attn.v_experts.0.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.self_attn.v_experts.{e}.weight")
                    for e in range(self.args.mova_num_experts)
                ]
                weights[f"{prefix}.self_attn.v_experts.weight"] = mx.stack(to_join)
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Routers stay at 8-bit; matches the published checkpoint.
            if path.endswith("mlp.gate") or path.endswith("self_attn.v_router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
