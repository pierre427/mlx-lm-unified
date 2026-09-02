# Copyright © 2026 Apple Inc.

import math
import os

import mlx.core as mx
import mlx.nn as nn


# --- Absorbed-vs-expanded MLA query-width crossover -------------------------
#
# MLA attention has two algebraically identical forms.  Writing the per-head
# geometry as r = kv_lora_rank, dn = qk_nope_head_dim, dv = v_head_dim, with a
# query width L and a cache length S, the per-head multiply counts are:
#
#   absorbed  (fold W_k / W_v into the query and the output; attend in latent
#              space against the cached r-dim latent directly)
#       embed_q             L * dn * r
#       scores              L * S  * r
#       softmax @ latent    L * S  * r
#       unembed_out         L * r  * dv
#     total = 2*L*S*r + L*r*(dn + dv)
#
#   expanded  (materialize k and v for the whole cache from the latent)
#       k = latent @ W_k    S * r * dn
#       v = latent @ W_v    S * r * dv
#       scores              L * S * dn
#       attn @ v            L * S * dv
#     total = S*r*(dn + dv) + L*S*(dn + dv)
#
# The absorbed cost is linear in L; the expanded cost carries an
# L-independent S*r*(dn + dv) term because it rebuilds k and v over the entire
# cache no matter how few query rows there are.  Dropping the absorbed
# L*r*(dn + dv) term (it does not scale with S, and is small whenever S >> r)
# and dividing through by S:
#
#       2*L*r  =  r*(dn + dv) + L*(dn + dv)
#   =>  L*     =  r*(dn + dv) / (2*r - dn - dv)
#
# For the DeepSeek-V2/V3 and Sarvam geometry (r=512, dn=dv=128) this is
# 512*256 / (1024-256) = 170.67, i.e. the absorbed form is the cheaper one for
# every speculative-decode verify width (L = k+1, typically 2..8) and for
# modest chunked prefill -- not only for L == 1, which is what these models
# gated on until now.
#
# Degenerate case: when 2*r <= dn + dv the expanded form is never the more
# expensive one at large S, so fall back to decode-only (L == 1).

_ABSORBED_MAX_QUERY_ENV = "MLX_LM_MLA_ABSORBED_MAX_QUERY"


def _absorbed_env_override():
    raw = os.environ.get(_ABSORBED_MAX_QUERY_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return None


#: Process-wide override of the absorbed-path query-width limit, resolved once
#: from ``MLX_LM_MLA_ABSORBED_MAX_QUERY`` at import.  ``0`` forces the expanded
#: branch at every width (including decode); a large value forces the absorbed
#: branch.  Tests and A/B benchmarks flip it in place with
#: :func:`set_absorbed_max_query_override` so the model need not be reloaded.
ABSORBED_MAX_QUERY_OVERRIDE = _absorbed_env_override()


def set_absorbed_max_query_override(value):
    """Set (``None`` clears) the process-wide absorbed query-width override."""
    global ABSORBED_MAX_QUERY_OVERRIDE
    ABSORBED_MAX_QUERY_OVERRIDE = None if value is None else max(0, int(value))


def absorbed_max_query(kv_lora_rank, qk_nope_head_dim, v_head_dim) -> int:
    """Largest query width L for which absorbed MLA beats the expanded form.

    Pass the *resolved* attention geometry, not raw config fields: some models
    (e.g. ``kimi_linear``) leave ``qk_nope_head_dim`` / ``v_head_dim`` unset in
    the config and fall back to ``head_dim`` in the attention module.
    """
    denom = 2 * kv_lora_rank - qk_nope_head_dim - v_head_dim
    if denom <= 0:
        return 1
    return max(1, (kv_lora_rank * (qk_nope_head_dim + v_head_dim)) // denom)


def absorbed_query_limit(geometric_limit: int) -> int:
    """Effective absorbed query-width limit, honoring the env/test override."""
    override = ABSORBED_MAX_QUERY_OVERRIDE
    return geometric_limit if override is None else override


class MultiLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )

    def __call__(self, x, transpose=True):
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight

    def to_quantized(
        self,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ):
        num_heads, output_dims, input_dims = self.weight.shape
        ql = QuantizedMultiLinear(
            input_dims, output_dims, num_heads, group_size, bits, mode
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
        return ql


class QuantizedMultiLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_heads: int,
        group_size: int,
        bits: int,
        mode: str,
    ):
        super().__init__()

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        self.freeze()

    def __call__(self, x, transpose=True):
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
