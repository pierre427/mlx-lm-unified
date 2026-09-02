# Copyright © 2026 Apple Inc.

import math
import os

import mlx.core as mx
import mlx.nn as nn


# --- Absorbed-vs-expanded MLA query-width crossover -------------------------
#
# MLA attention has two algebraically identical forms.  Writing the per-head
# geometry as r = kv_lora_rank, D = qk_nope_head_dim + v_head_dim, with a query
# width L and a cache length S (the keys actually attended -- prefix + L), the
# per-head multiply counts are:
#
#   absorbed  (fold W_k / W_v into the query and the output; attend in latent
#              space against the cached r-dim latent directly)
#       embed_q             L * dn * r
#       scores              L * S  * r
#       softmax @ latent    L * S  * r
#       unembed_out         L * r  * dv
#     total = 2*L*S*r + L*r*D
#
#   expanded  (materialize k and v for the whole cache from the latent)
#       k = latent @ W_k    S * r * dn
#       v = latent @ W_v    S * r * dv
#       scores              L * S * dn
#       attn @ v            L * S * dv
#     total = S*r*D + L*S*D
#
# The expanded cost carries an L-independent S*r*D term because it rebuilds k
# and v over the entire cache no matter how few query rows there are.  Absorbed
# is the cheaper form when
#
#       2*L*S*r + L*r*D  <=  S*r*D + L*S*D
#   =>  L * (S*(2*r - D) + r*D)  <=  S*r*D
#   =>  L*(S)  =  S*r*D / (S*(2*r - D) + r*D)              [bracket > 0]
#
# S does not cancel and must not be dropped.  As S -> infinity the limit tends
# to the asymptotic crossover r*D / (2*r - D) -- 170 for the DeepSeek-V2/V3 and
# Sarvam geometry (r=512, dn=dv=128) -- which is the regime this gate exists to
# serve: a few verify rows (L = k+1, typically 2..8) against a long cache.
#
# The asymptotic form is *wrong* for a cold prefill, where S ~ L.  Substituting
# S = L reduces the condition to 2*r <= D, which no real MLA geometry
# satisfies, so a fresh L-token prompt is always cheaper on the expanded
# branch: at r=512, D=256 a 128-token cold prompt costs 33.5M multiplies
# absorbed against 21.0M expanded.  Gating on the asymptotic limit alone would
# have regressed short-prompt prefill, so the runtime gate uses the S-aware
# form with the cache length actually attended.
#
# Degenerate geometry: when the bracket S*(2*r - D) + r*D is <= 0 -- only
# reachable at 2*r < D, a latent rank below half the expanded head geometry --
# the inequality holds for every L, i.e. absorbed is cheaper at every width.
# No shipped MLA model has that geometry.

_ABSORBED_MAX_QUERY_ENV = "MLX_LM_MLA_ABSORBED_MAX_QUERY"

#: Returned instead of a finite crossover for degenerate geometries where the
#: absorbed form is cheaper at every query width.
ABSORBED_UNBOUNDED = 1 << 30


def _absorbed_env_override():
    raw = os.environ.get(_ABSORBED_MAX_QUERY_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        return max(0, int(raw.strip()))
    except ValueError as exc:
        # Fail closed rather than silently ignoring the knob: an unparseable
        # value in an A/B run would otherwise look exactly like a null result.
        raise ValueError(
            f"{_ABSORBED_MAX_QUERY_ENV}={raw!r} is not an integer. Unset it, or "
            "set 0 to force the expanded branch and a large value to force the "
            "absorbed branch."
        ) from exc


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


def absorbed_max_query(
    kv_lora_rank, qk_nope_head_dim, v_head_dim, cache_len=None
) -> int:
    """Largest query width L for which absorbed MLA beats the expanded form.

    ``cache_len`` is the number of keys attended (prefix + query rows).  Pass it
    whenever it is known: the crossover depends on it, and the asymptotic
    (``cache_len=None``) value overstates the limit badly for a cold prefill,
    where the expanded branch is the cheaper one at every width above 1.

    Pass the *resolved* attention geometry, not raw config fields: some models
    (e.g. ``kimi_linear``) leave ``qk_nope_head_dim`` / ``v_head_dim`` unset in
    the config and fall back to ``head_dim`` in the attention module.
    """
    d = qk_nope_head_dim + v_head_dim
    if cache_len is None:
        numerator = kv_lora_rank * d
        denom = 2 * kv_lora_rank - d
    else:
        numerator = cache_len * kv_lora_rank * d
        denom = cache_len * (2 * kv_lora_rank - d) + kv_lora_rank * d
    if denom <= 0:
        return ABSORBED_UNBOUNDED
    return max(1, numerator // denom)


def absorbed_query_limit(geometric_limit: int) -> int:
    """Effective absorbed query-width limit, honoring the env/test override."""
    override = ABSORBED_MAX_QUERY_OVERRIDE
    return geometric_limit if override is None else override


def use_absorbed_path(query_len: int, cache_len: int, geometry) -> bool:
    """Whether an (L, S)-shaped MLA attention should take the absorbed branch.

    ``geometry`` is the ``(kv_lora_rank, qk_nope_head_dim, v_head_dim)`` tuple
    the attention module resolved at construction.  ``L == 1`` always answers
    True (``absorbed_max_query`` floors at 1), so decode dispatch is exactly
    what it was before the gate was widened.
    """
    limit = absorbed_query_limit(
        absorbed_max_query(*geometry, cache_len=cache_len)
    )
    return query_len <= limit


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
