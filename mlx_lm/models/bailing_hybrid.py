# Copyright © 2026 mlx-uag lab
#
# inclusionAI Ling-3.0-flash (`bailing_hybrid` / BailingMoeV3ForCausalLM).
#
# A hybrid decoder: every `layer_group_size`-th layer (and every layer past
# the last full group) is gated multi-latent attention (MLA); the other layers
# are Kimi Delta Attention (KDA) linear layers with a short causal conv on
# q/k/v. The first `first_k_dense_replace` layers use a dense MLP, the rest a
# sigmoid `noaux_tc` router (group-limited top-k with a selection-only expert
# bias) over `SwitchGLU` experts plus one shared expert.
#
# Wiring and numerics mirror the checkpoint's vendored reference
# (`bailing_hybrid.py`) op for op; tests assert bit identity. Departures, each
# config-driven or algebraically identical to the reference:
#   * MLA takes the tree's absorbed/expanded split from `mla.use_absorbed_path`
#     (the reference gates on L == 1 only) and reads a quantized KV cache.
#   * `kda_lower_bound` from config.json is honored: the HF module passes it
#     into FLA's KDA kernels, the reference drops it. MLX_LM_BAILING_HYBRID_KDA_GATE
#     selects `config` (default) or `plain` (the reference's softplus gate).
#   * KDA layers record exact rollbacks on their ArraysCache while speculating,
#     so prompt-lookup / self-MTP verify rounds can trim them.
#   * A_log / dt_bias / expert_bias are loaded as float32, as HF keeps them.

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    _contiguous_quant,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .mla import (
    MultiLinear,
    absorbed_max_query,
    refuse_asymmetric_mla_kv_bits,
    use_absorbed_path,
)
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU

_KDA_GATE_ENV = "MLX_LM_BAILING_HYBRID_KDA_GATE"
_KDA_GATE_MODES = ("config", "plain")


def _kda_gate_mode_from_env() -> str:
    raw = os.environ.get(_KDA_GATE_ENV)
    mode = (raw or "config").strip().lower() or "config"
    if mode not in _KDA_GATE_MODES:
        # Fail closed: a typo in an A/B run must not silently pick a gate.
        raise ValueError(
            f"{_KDA_GATE_ENV}={raw!r} is not one of {_KDA_GATE_MODES}. "
            "'config' honors kda_lower_bound from config.json, 'plain' uses "
            "the unbounded softplus gate of the vendored reference."
        )
    return mode


#: Process-wide KDA gate selection, resolved once from the environment.
KDA_GATE_MODE = _kda_gate_mode_from_env()


def set_kda_gate_mode(mode: Optional[str]):
    """Set (``None`` restores the environment value) the KDA gate mode."""
    global KDA_GATE_MODE
    if mode is None:
        KDA_GATE_MODE = _kda_gate_mode_from_env()
        return
    if mode not in _KDA_GATE_MODES:
        raise ValueError(f"kda gate mode {mode!r} is not one of {_KDA_GATE_MODES}")
    KDA_GATE_MODE = mode


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    routed_scaling_factor: float
    head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    short_conv_kernel_size: int = 4
    moe_shared_expert_intermediate_size: Optional[int] = None
    q_lora_rank: Optional[int] = None
    rope_interleave: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_qkv_bias: bool = False
    use_bias: bool = False
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    no_kda_lora: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: Optional[float] = None
    gated_attention_proj_granularity_type: Optional[str] = None
    # Accepted for config.json completeness; the HF module never reads them.
    # KDA l2-normalizes q/k in-kernel regardless of use_qk_norm, and the MLA
    # rope always spans qk_rope_head_dim (HF forces partial_rotary_factor=1).
    use_qk_norm: bool = True
    partial_rotary_factor: float = 1.0
    use_kda_lora: bool = False

    def __post_init__(self):
        if self.score_function not in ("sigmoid", "softmax"):
            raise ValueError(
                f"bailing_hybrid: unsupported score_function {self.score_function!r}"
            )
        if self.gated_attention_proj_granularity_type not in (
            None,
            "head_wise",
            "element_wise",
        ):
            raise ValueError(
                "bailing_hybrid: unsupported gated_attention_proj_granularity_type "
                f"{self.gated_attention_proj_granularity_type!r}"
            )
        if self.num_experts and self.n_group > 1 and self.num_experts % self.n_group:
            raise ValueError(
                f"num_experts {self.num_experts} is not divisible by n_group "
                f"{self.n_group}"
            )
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")


def is_global_layer(layer_idx: int, layer_group_size: int, num_layers: int) -> bool:
    """MLA on every group's last layer and on every layer past the last full group."""
    return (
        (layer_idx + 1) % layer_group_size == 0
        or layer_idx >= (num_layers // layer_group_size) * layer_group_size
    )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, hidden, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, hidden, bias=args.use_bias)
        self.down_proj = nn.Linear(hidden, args.hidden_size, bias=args.use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MultiLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.gate_type = args.gated_attention_proj_granularity_type
        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, self.q_lora_rank, bias=args.use_qkv_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        # kv_b_proj split into the absorbed-MLA pair (see sanitize).
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        # Query-width crossover of the absorbed branch; the forward resolves
        # the cache-length-aware limit per call (see mla.absorbed_max_query).
        self.absorbed_geometry = (
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        self.absorbed_max_query = absorbed_max_query(*self.absorbed_geometry)

        if self.gate_type is None:
            self.g_proj = None
        elif self.gate_type == "head_wise":
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        else:
            self.g_proj = nn.Linear(
                args.hidden_size, self.num_heads * self.v_head_dim, bias=False
            )

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim, args.hidden_size, bias=args.use_qkv_bias
        )

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = args.rope_scaling.get("factor", 1)
            if mscale_all_dim and scaling_factor > 1:
                mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scale *= mscale * mscale

        # HF's interleaved rope pairs adjacent dims, i.e. MLX "traditional".
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        # A QuantizedKVCache returns (weight, scales, biases) tuples.
        quantized = not isinstance(k_pe, mx.array)
        if quantized:
            refuse_asymmetric_mla_kv_bits(cache, "bailing_hybrid")
            pe_scores = mx.quantized_matmul(
                q_pe * self.scale,
                *k_pe,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        absorbed = use_absorbed_path(
            L, pe_scores.shape[-1], self.absorbed_geometry
        )
        if absorbed:
            q_nope = self.embed_q(q_nope)
            keys = values = kv_latent
            output = scaled_dot_product_attention(
                q_nope, keys, values, cache=cache, scale=self.scale, mask=pe_scores
            )
        else:
            if quantized:
                kv_latent = mx.dequantize(
                    *_contiguous_quant(kv_latent),
                    group_size=cache.group_size,
                    bits=cache.bits,
                )
            keys = self.embed_q(kv_latent, transpose=False)
            values = self.unembed_out(kv_latent)
            # keys/values are materialized arrays, so plain SDPA even when the
            # cache itself is quantized.
            output = scaled_dot_product_attention(
                q_nope, keys, values, cache=None, scale=self.scale, mask=pe_scores
            )
        if absorbed:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gate_type == "head_wise":
                output = output * gate[..., None]
            else:
                output = output * gate.reshape(B, L, self.num_heads, self.v_head_dim)
        return self.dense(output.reshape(B, L, -1))


class ShortConv1d(nn.Module):
    """Depthwise causal conv with silu; state is the last kernel-1 inputs."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        output = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])
        # conv_input is returned so a speculative rollback can slice the state
        # after any prefix of this forward.
        return output, new_state, conv_input


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.conv_kernel = args.short_conv_kernel_size
        self.projection_dim = self.num_heads * self.head_dim
        self.no_kda_lora = args.no_kda_lora
        self.scale = float(self.head_dim) ** -0.5
        # The bounded gate: HF hands both to FLA; fused_recurrent_kda (decode)
        # only takes lower_bound, so the bound alone fixes the formula.
        self.lower_bound = args.kda_lower_bound
        self.safe_gate = args.kda_safe_gate

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)

        if self.no_kda_lora:
            self.f_proj = nn.Linear(hidden, self.projection_dim, bias=False)
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.log(
            mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))
        )
        self.dt_bias = mx.zeros((self.projection_dim,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def effective_lower_bound(self) -> Optional[float]:
        return None if KDA_GATE_MODE == "plain" else self.lower_bound

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if cache is not None:
            q_state, k_state, v_state, recurrent_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = recurrent_state = None
            lengths = None
        if q_state is None:
            state = mx.zeros(
                (B, self.conv_kernel - 1, self.projection_dim), dtype=x.dtype
            )
            q_state = k_state = v_state = state
        snapshot = [q_state, k_state, v_state, recurrent_state]

        q, q_state, q_in = self.q_conv1d(self.q_proj(x), q_state, mask, lengths)
        k, k_state, k_in = self.k_conv1d(self.k_proj(x), k_state, mask, lengths)
        v, v_state, v_in = self.v_conv1d(self.v_proj(x), v_state, mask, lengths)
        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        # Reference l2norm as written in the vendored file: eps on the mean of
        # squares (FLA adds 1e-6 to the sum; gated_delta.normalize_gdn_qk has
        # that form). Kept for bit identity with the reference.
        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        if self.no_kda_lora:
            decay_logits = self.f_proj(x)
        else:
            decay_logits = self.f_b_proj(self.f_a_proj(x))
        decay_logits = decay_logits.reshape(B, L, self.num_heads, self.head_dim)
        beta_logits = self.b_proj(x).reshape(B, L, self.num_heads)
        A_log = self.A_log.reshape(self.num_heads, 1)
        dt_bias = self.dt_bias.reshape(self.num_heads, self.head_dim)
        lower_bound = self.effective_lower_bound()
        use_kernel = not self.training

        # Exact speculative rollback, as the tree's GDN layers record it: the
        # state after m tokens is the recurrence replayed over the first m of
        # the inputs this forward consumes; each conv state is a slice of its
        # conv_input. rollback_spans() is None where a per-row depth would
        # misdescribe the rows, and then nothing is staged.
        spans = ()
        if cache is not None:
            describe = getattr(cache, "rollback_spans", None)
            if describe is not None:
                spans = describe(L, mask)
        if (
            cache is not None
            and getattr(cache, "speculating", False)
            and spans is not None
        ):
            n_keep = self.conv_kernel - 1

            def _rollback(m, S0=recurrent_state):
                _, s_m = gated_delta_update(
                    q[:, :m],
                    k[:, :m],
                    v[:, :m],
                    decay_logits[:, :m],
                    beta_logits[:, :m],
                    A_log,
                    dt_bias,
                    state=S0,
                    mask=None,
                    use_kernel=use_kernel,
                    lower_bound=lower_bound,
                )
                return [
                    mx.contiguous(q_in[:, m : m + n_keep, :]),
                    mx.contiguous(k_in[:, m : m + n_keep, :]),
                    mx.contiguous(v_in[:, m : m + n_keep, :]),
                    s_m,
                ]

            cache.record_rollback(L, _rollback, snapshot)

        output, recurrent_state = gated_delta_update(
            q,
            k,
            v,
            decay_logits,
            beta_logits,
            A_log,
            dt_bias,
            state=recurrent_state,
            mask=mask,
            use_kernel=use_kernel,
            lower_bound=lower_bound,
        )
        if cache is not None:
            cache[3] = recurrent_state
            cache.advance(L)

        if self.no_kda_lora:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, L, self.num_heads, self.head_dim)
        output = self.o_norm(output) * mx.sigmoid(gate)
        return self.o_proj(output.reshape(B, L, -1))


@mx.compile
def group_expert_select(
    gates: mx.array,
    expert_bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    input_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1, precise=True)

    # The bias steers selection only; mixing weights use the raw scores.
    original_scores = scores
    if expert_bias is not None:
        scores = scores + expert_bias

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        n_drop = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=n_drop - 1, axis=-2)[
            ..., :n_drop, :
        ]
        # Dropped groups are zeroed (HF fills -inf); identical unless fewer
        # than top_k kept experts score above zero after the bias.
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    indices = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(original_scores, indices, axis=-1)
    if top_k > 1 and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return indices, weights.astype(input_type)


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function
        # nn.Linear so the checkpoint's 8-bit router loads as QuantizedLinear.
        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,), dtype=mx.float32)
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = Gate(args)
        shared_size = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        ) * args.num_shared_experts
        self.shared_experts = (
            MLP(args, intermediate_size=shared_size)
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = self.gate(x)
        output = self.switch_mlp(x, indices)
        output = (output * weights[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_global = is_global_layer(
            layer_idx, args.layer_group_size, args.num_hidden_layers
        )
        self.attention = (
            MultiLatentAttention(args) if self.is_global else KimiDeltaAttention(args)
        )
        self.mlp = (
            SparseMoeBlock(args)
            if args.num_experts and layer_idx >= args.first_k_dense_replace
            else MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attention(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attention_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_global), None
        )
        self._kda_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_global), None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        attention_mask = None
        if self._attention_idx is not None:
            attention_mask = create_attention_mask(
                h, cache[self._attention_idx], return_array=True
            )
        kda_mask = None
        if self._kda_idx is not None:
            kda_mask = create_ssm_mask(h, cache[self._kda_idx])

        for layer, layer_cache in zip(self.layers, cache):
            mask = attention_mask if layer.is_global else kda_mask
            h = layer(h, mask=mask, cache=layer_cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[Any]:
        return [
            KVCache() if layer.is_global else ArraysCache(size=4)
            for layer in self.layers
        ]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        args = self.args
        n_layers = args.num_hidden_layers

        def layer_of(key):
            parts = key.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                return int(parts[2]) if parts[2].isdigit() else None
            return None

        # Drop MTP heads (HF stores them as layers >= num_hidden_layers).
        weights = {
            k: v
            for k, v in weights.items()
            if not ((idx := layer_of(k)) is not None and idx >= n_layers)
        }
        if args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, SparseMoeBlock):
                for projection in ("gate_proj", "down_proj", "up_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        first = f"{prefix}.mlp.experts.0.{projection}.{suffix}"
                        if first in weights:
                            stacked = [
                                weights.pop(
                                    f"{prefix}.mlp.experts.{e}.{projection}.{suffix}"
                                )
                                for e in range(args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{projection}.{suffix}"] = (
                                mx.stack(stacked)
                            )
                # HF router: gate.weight -> gate.gate_proj.weight.
                for suffix in ("weight", "scales", "biases"):
                    gate_key = f"{prefix}.mlp.gate.{suffix}"
                    if gate_key in weights:
                        weights[f"{prefix}.mlp.gate.gate_proj.{suffix}"] = weights.pop(
                            gate_key
                        )

            attn = f"{prefix}.attention"
            if layer.is_global:
                kv_b_key = f"{attn}.kv_b_proj.weight"
                if kv_b_key in weights:
                    value = weights.pop(kv_b_key)
                    quantized = f"{attn}.kv_b_proj.scales" in weights
                    if quantized:
                        scales = weights.pop(f"{attn}.kv_b_proj.scales")
                        biases = weights.pop(f"{attn}.kv_b_proj.biases")
                        bits = (value.shape[-1] * 32) // args.kv_lora_rank
                        group_size = args.kv_lora_rank // scales.shape[-1]
                        value = mx.dequantize(
                            value, scales, biases, bits=bits, group_size=group_size
                        )
                    combined = args.qk_nope_head_dim + args.v_head_dim
                    value = value.reshape(args.num_attention_heads, combined, -1)
                    wk = mx.contiguous(
                        value[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2)
                    )
                    wv = mx.contiguous(value[:, args.qk_nope_head_dim :, :])
                    if quantized:
                        wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                        wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                        weights[f"{attn}.embed_q.scales"] = wk_s
                        weights[f"{attn}.embed_q.biases"] = wk_b
                        weights[f"{attn}.unembed_out.scales"] = wv_s
                        weights[f"{attn}.unembed_out.biases"] = wv_b
                    weights[f"{attn}.embed_q.weight"] = wk
                    weights[f"{attn}.unembed_out.weight"] = wv
            else:
                # HF conv weight (C, 1, K) -> MLX Conv1d (C, K, 1), under the
                # ShortConv1d submodule name.
                for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    source = f"{attn}.{name}.weight"
                    if source in weights:
                        value = weights.pop(source)
                        if value.ndim == 3 and value.shape[-1] != 1:
                            value = value.moveaxis(2, 1)
                        weights[f"{attn}.{name}.conv.weight"] = value
                # HF keeps the gate parameters in float32; some MLX exports
                # store them in the activation dtype.
                for name in ("A_log", "dt_bias"):
                    key = f"{attn}.{name}"
                    if key in weights and weights[key].dtype != mx.float32:
                        weights[key] = weights[key].astype(mx.float32)
                dt_key = f"{attn}.dt_bias"
                if dt_key in weights and weights[dt_key].ndim > 1:
                    weights[dt_key] = weights[dt_key].reshape(-1)

            bias_key = f"{prefix}.mlp.gate.expert_bias"
            if bias_key in weights and weights[bias_key].dtype != mx.float32:
                weights[bias_key] = weights[bias_key].astype(mx.float32)

        return weights

    @property
    def quant_predicate(self):
        def predicate(path: str, _module: nn.Module):
            # Routers stay at 8-bit; matches the published checkpoint.
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            if "expert_bias" in path:
                return False
            return True

        return predicate
