"""Speculative (PFlash-style) prefill for MLX.

This module implements a **lossy** prefill approximation that trades answer
fidelity for time-to-first-token (TTFT) at long context. It is the compute-bound
dual of speculative *decoding*: instead of a drafter proposing decode tokens
that the target verifies losslessly, a small drafter scores the *importance* of
every prompt token and the target prefills only the top-``k%`` most important
ones (block-sparse). Prefill is the lab's compute-bound side (M5 NAX prefill
battles), so shrinking the target's prefill quadratic is the lever.

References
----------
- Liu et al., "Speculative Prefill: Turbocharging TTFT with Lightweight and
  Training-Free Token Importance Estimation", ICML 2025 (arXiv:2502.02789).
- llama.cpp feature request #24126 (open request that names MLX), "Adaptive
  PFlash": drafter attention-mass scoring + block-sparse target prefill,
  ~10x prefill at 128k / ~3x TTFT on M-series.

IMPORTANT — this is LOSSY, unlike our speculative *decoding* which is bit-exact.
Dropping prompt tokens from the target's KV cache changes the target's outputs.
The knob ``k_frac`` makes the quality-vs-TTFT trade measurable: ``k_frac=1.0``
degenerates to the exact dense prefill; smaller keeps fewer tokens, faster but
lossier. Force-keeping the first tokens (attention sinks) and a recent tail is
the standard mitigation and is on by default.

Position preservation
---------------------
The kept tokens are non-contiguous in the original prompt, but downstream decode
must attend to them with their *true* RoPE positions so relative rotations are
correct. ``mx.fast.rope`` only accepts a scalar (or 1-element) offset, so we
cannot hand it arbitrary per-token positions. Instead we exploit two facts:
1. Every stock mlx-lm model applies RoPE with the scalar ``cache.offset`` and
   delegates its attention mask to ``cache.make_mask(...)``.
2. Kept tokens can be grouped into maximal **contiguous runs**; a run fed in one
   forward with ``cache.offset = run_start`` gets exactly its true positions.
``SparsePromptCache`` stores K/V *compactly* (only kept tokens) while ``.offset``
carries the true RoPE position, and its ``make_mask`` builds the causal mask
over the *compact* index space. Storage index and RoPE position are thus
decoupled — the whole point.
"""

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

from .models.base import create_causal_mask
from .models.cache import make_prompt_cache

__all__ = [
    "SparsePromptCache",
    "aggregate_attention_scores",
    "select_topk_mask",
    "contiguous_runs",
    "assign_rope_positions",
    "AttentionImportanceScorer",
    "speculative_prefill",
]


# --------------------------------------------------------------------------- #
# Compact, position-preserving KV cache
# --------------------------------------------------------------------------- #
class SparsePromptCache:
    """KV cache that stores only kept tokens but preserves their true positions.

    Two counters are tracked separately:

    - ``offset``: the *true* RoPE sequence position of the next token to be
      processed. Read by the model as ``cache.offset`` for ``self.rope(...)``.
      The prefill driver sets this to each run's true start; ``update_and_fetch``
      then advances it contiguously within the run (and during decode).
    - ``_compact``: the number of tokens actually stored, i.e. the write pointer
      into the compact K/V buffers. Used to build the attention mask.

    For a dense (``k=100%``) prefill the two counters stay equal and this reduces
    to an ordinary concatenating KV cache.
    """

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0  # true RoPE position of the next token
        self._compact: int = 0  # number of stored tokens (write pointer)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # keys/values are (B, n_kv_heads, L, head_dim), already RoPE'd at the
        # true positions [offset, offset+L) by the model (contiguous run).
        L = keys.shape[2]
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self._compact += L
        self.offset += L
        return self.keys, self.values

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
    ):
        """Causal mask over the *compact* index space.

        The ``N`` new queries attend to the ``_compact`` already-stored keys plus
        each other, causally. Because runs are processed in ascending true-position
        order, compact order preserves causal order, so a plain causal mask offset
        by the current compact length is exactly right.
        """
        if N == 1:
            # Single decode query attends to every stored (earlier) key.
            return None
        return create_causal_mask(N, offset=self._compact, window_size=window_size)

    # --- trimming (decode / speculation rollback) --------------------------- #
    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._compact, n)
        if n <= 0:
            return 0
        self._compact -= n
        self.offset -= n
        self.keys = self.keys[..., : self._compact, :]
        self.values = self.values[..., : self._compact, :]
        return n

    def size(self) -> int:
        return self._compact

    def empty(self) -> bool:
        return self.keys is None

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self._compact = 0 if self.keys is None else self.keys.shape[2]
        self.offset = self._compact

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v:
            raise ValueError("SparsePromptCache has no meta_state.")

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


# --------------------------------------------------------------------------- #
# Importance aggregation (pure, synthetic-testable)
# --------------------------------------------------------------------------- #
def aggregate_attention_scores(
    probs: mx.array,
    head_agg: str = "mean",
) -> mx.array:
    """Reduce one attention block's lookahead probabilities to per-key scores.

    Args:
        probs: attention *probabilities* (post-softmax) of shape
            ``(n_heads, w, S)`` — the ``w`` lookahead query rows attending to all
            ``S`` keys.
        head_agg: ``"mean"`` or ``"max"`` over heads.

    Returns:
        ``(S,)`` per-key importance for this block, mean-pooled over the ``w``
        lookahead rows then aggregated over heads.
    """
    if probs.ndim != 3:
        raise ValueError(f"probs must be (n_heads, w, S), got shape {probs.shape}")
    per_head = probs.mean(axis=1)  # (n_heads, S) — mean over lookahead rows
    if head_agg == "mean":
        return per_head.mean(axis=0)
    elif head_agg == "max":
        return per_head.max(axis=0)
    raise ValueError(f"Unknown head_agg={head_agg!r} (use 'mean' or 'max').")


# --------------------------------------------------------------------------- #
# Top-k% block selection (pure, synthetic-testable)
# --------------------------------------------------------------------------- #
def select_topk_mask(
    scores: Sequence[float] | mx.array,
    k_frac: float,
    block_size: int = 32,
    keep_first: int = 4,
    keep_last: int = 64,
) -> Tuple[mx.array, List[int]]:
    """Select the top ``k_frac`` fraction of prompt tokens, at block granularity.

    Tokens are grouped into contiguous ``block_size`` blocks (PFlash keeps/drops
    whole blocks so kept tokens form long contiguous runs, which is what makes
    the position-preserving prefill efficient). Blocks are ranked by mean score;
    the top ``ceil(k_frac * num_blocks)`` are kept. The first ``keep_first``
    tokens (attention sinks) and last ``keep_last`` tokens (recency) are always
    kept regardless of score.

    Args:
        scores: per-token importance, length ``N``.
        k_frac: fraction of *blocks* to keep in ``(0, 1]``. ``>= 1.0`` keeps all
            tokens (dense / identity).
        block_size: token block granularity.
        keep_first: always-kept prefix length (attention sinks).
        keep_last: always-kept suffix length (recency).

    Returns:
        ``(mask, kept_indices)`` where ``mask`` is a bool ``mx.array`` of length
        ``N`` and ``kept_indices`` is the sorted list of kept positions.
    """
    scores = mx.array(scores, dtype=mx.float32).reshape(-1)
    N = int(scores.shape[0])
    if N == 0:
        return mx.zeros((0,), dtype=mx.bool_), []
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    mask = [False] * N

    if k_frac >= 1.0:
        return mx.ones((N,), dtype=mx.bool_), list(range(N))

    num_blocks = math.ceil(N / block_size)
    # Mean score per block (partial trailing block handled by slicing).
    block_scores = []
    for b in range(num_blocks):
        lo = b * block_size
        hi = min(lo + block_size, N)
        block_scores.append(float(scores[lo:hi].mean()))

    n_keep = max(1, math.ceil(k_frac * num_blocks))
    # Rank blocks by score, keep the top n_keep (ties broken by earlier block).
    order = sorted(range(num_blocks), key=lambda b: (-block_scores[b], b))
    for b in order[:n_keep]:
        lo = b * block_size
        hi = min(lo + block_size, N)
        for i in range(lo, hi):
            mask[i] = True

    # Force-keep sinks and recency.
    for i in range(min(keep_first, N)):
        mask[i] = True
    for i in range(max(0, N - keep_last), N):
        mask[i] = True

    kept = [i for i in range(N) if mask[i]]
    return mx.array(mask, dtype=mx.bool_), kept


def contiguous_runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Group a sorted index list into maximal ``(start, length)`` contiguous runs."""
    runs: List[Tuple[int, int]] = []
    start = None
    prev = None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev - start + 1))
            start = prev = i
    if start is not None:
        runs.append((start, prev - start + 1))
    return runs


def assign_rope_positions(runs: Sequence[Tuple[int, int]]) -> List[int]:
    """True RoPE positions assigned to stored tokens, in storage (compact) order.

    Each run at true start ``s`` of length ``l`` contributes positions
    ``[s, s+1, ..., s+l-1]``. The concatenation equals the kept indices, i.e.
    every stored token keeps its *original* position — this is the property the
    position-preservation unit test checks.
    """
    positions: List[int] = []
    for start, length in runs:
        positions.extend(range(start, start + length))
    return positions


# --------------------------------------------------------------------------- #
# Drafter attention-mass importance scorer (GPU path)
# --------------------------------------------------------------------------- #
class AttentionImportanceScorer:
    """Score prompt-token importance from a small drafter's attention mass.

    Primary method (Liu et al. 2025 / PFlash): run the drafter over the prompt,
    and for each attention layer take the last ``lookahead`` query rows'
    attention to every prompt key, ``softmax(Q_lookahead·Kᵀ/√d)``. Aggregate:
    mean over lookahead rows, ``head_agg`` over heads, summed over layers. The
    resulting per-token vector is the importance score consumed by
    ``select_topk_mask``.

    Implementation note: stock mlx-lm models use fused ``mx.fast.scaled_dot_
    product_attention`` and do not expose attention weights. Rather than fork
    every model, we temporarily wrap ``mx.fast.scaled_dot_product_attention``
    for the drafter forward (context manager below): it computes the real output
    (so deeper layers stay correct) and *additionally* the cheap ``(w, S)``
    last-window score slice, which it accumulates. Cost added is one small
    ``Q_lookahead·Kᵀ`` matmul per layer — negligible vs. the drafter forward.

    Alternatives (not primary): a training-free saliency proxy such as key-norm
    ‖K‖ per token, or hidden-state velocity. Attention-mass is chosen because it
    is the method the reference paper/impl validate and it directly measures
    "how much the model looks at this token".

    NOTE: the scoring forward itself is a *dense* drafter prefill (O(N²) on a
    tiny model). The win is that the expensive *target* prefill becomes sparse.
    """

    def __init__(
        self,
        drafter: Any,
        lookahead: int = 8,
        head_agg: str = "max",
    ):
        self.drafter = drafter
        self.lookahead = lookahead
        self.head_agg = head_agg

    def score(self, prompt_tokens: mx.array) -> mx.array:
        """Return a per-prompt-token importance vector of length ``N``.

        GPU path — runs the drafter. ``prompt_tokens`` is a 1-D int array.
        """
        prompt_tokens = mx.array(prompt_tokens).reshape(-1)
        N = int(prompt_tokens.shape[0])
        acc = mx.zeros((N,), dtype=mx.float32)

        with _capture_attention(self.lookahead, self.head_agg) as sink:
            cache = make_prompt_cache(self.drafter)
            self.drafter(prompt_tokens[None], cache=cache)
            mx.eval([c.state for c in cache if hasattr(c, "state")])

        for layer_scores in sink:
            # layer_scores is (S,) with S == N (single dense forward).
            if layer_scores.shape[0] == N:
                acc = acc + layer_scores
        return acc


class _capture_attention:
    """Context manager: wrap SDPA to accumulate last-window attention mass.

    Patches ``mx.fast.scaled_dot_product_attention`` for the duration. The wrapper
    returns the true attention output and stashes, per call, a ``(S,)`` per-key
    importance slice computed from the last ``lookahead`` query rows.
    """

    def __init__(self, lookahead: int, head_agg: str):
        self.lookahead = lookahead
        self.head_agg = head_agg
        self.sink: List[mx.array] = []
        self._orig = None

    def __enter__(self):
        self._orig = mx.fast.scaled_dot_product_attention
        orig = self._orig
        lookahead = self.lookahead
        head_agg = self.head_agg
        sink = self.sink

        def wrapped(queries, keys, values, *, scale, mask=None, **kwargs):
            out = orig(queries, keys, values, scale=scale, mask=mask, **kwargs)
            try:
                # queries: (B, n_q_heads, L, D); keys: (B, n_kv_heads, S, D)
                B, n_q_heads, L, D = queries.shape
                S = keys.shape[2]
                w = min(lookahead, L)
                q = queries[0, :, L - w : L, :]  # (n_q_heads, w, D)
                k = keys[0]  # (n_kv_heads, S, D)
                n_rep = n_q_heads // k.shape[0]
                if n_rep > 1:
                    k = mx.repeat(k, n_rep, axis=0)  # (n_q_heads, S, D)
                logits = (q @ k.transpose(0, 2, 1)) * scale  # (n_q_heads, w, S)
                # Causal masking for the lookahead rows (row r -> abs pos S-w+r).
                rows = mx.arange(S - w, S).reshape(w, 1)
                cols = mx.arange(S).reshape(1, S)
                causal = rows >= cols
                logits = mx.where(causal, logits, -mx.inf)
                probs = mx.softmax(logits, axis=-1, precise=True)  # (H, w, S)
                sink.append(aggregate_attention_scores(probs, head_agg=head_agg))
            except Exception:
                # Never let scoring instrumentation break the forward.
                pass
            return out

        mx.fast.scaled_dot_product_attention = wrapped
        return self.sink

    def __exit__(self, *exc):
        mx.fast.scaled_dot_product_attention = self._orig
        return False


# --------------------------------------------------------------------------- #
# The sparse prefill driver
# --------------------------------------------------------------------------- #
def speculative_prefill(
    target_model: Any,
    prompt_tokens: mx.array,
    scorer: Callable[[mx.array], mx.array] | mx.array,
    *,
    k_frac: float = 0.4,
    block_size: int = 32,
    keep_first: int = 4,
    keep_last: int = 64,
    prefill_step_size: int = 2048,
    head_agg: str = "max",
) -> Tuple[List[SparsePromptCache], mx.array]:
    """Prime a target prompt cache on only the top-``k_frac`` important tokens.

    LOSSY. See module docstring. ``k_frac=1.0`` reproduces the exact dense
    prefill.

    Args:
        target_model: the large model to prefill.
        prompt_tokens: 1-D int array of prompt token ids (length ``N``).
        scorer: either a precomputed per-token score array of length ``N``, or a
            callable ``prompt_tokens -> scores`` (e.g. ``AttentionImportanceScorer
            (drafter).score``).
        k_frac: fraction of blocks the target actually prefills.
        block_size, keep_first, keep_last: forwarded to ``select_topk_mask``.
        prefill_step_size: sub-chunk size within a run (memory bound only; runs
            stay contiguous so sub-chunks keep correct positions automatically).
        head_agg: unused here; kept for signature symmetry with the scorer.

    Returns:
        ``(prompt_cache, anchor)`` where ``prompt_cache`` is one
        ``SparsePromptCache`` per target layer primed on the kept prompt tokens
        *except the final one*, with ``offset`` set to ``N-1``; and ``anchor`` is
        the last prompt token (shape ``(1,)``). Feed the anchor to ``generate``/
        ``stream_generate`` with ``prompt_cache=prompt_cache`` to produce the
        first logits — mirroring the stock "prefill all but last, then step the
        last token" pattern, so it drops in as an opt-in replacement for prefill.
    """
    prompt_tokens = mx.array(prompt_tokens).reshape(-1)
    N = int(prompt_tokens.shape[0])
    if N == 0:
        raise ValueError("prompt_tokens must be non-empty.")

    scores = scorer(prompt_tokens) if callable(scorer) else mx.array(scorer)
    scores = mx.array(scores, dtype=mx.float32).reshape(-1)
    if int(scores.shape[0]) != N:
        raise ValueError(
            f"scorer returned {scores.shape[0]} scores for {N} prompt tokens."
        )

    _mask, kept = select_topk_mask(
        scores, k_frac, block_size=block_size, keep_first=keep_first, keep_last=keep_last
    )
    # The final prompt token must be present to produce the first logits.
    if not kept or kept[-1] != N - 1:
        kept = sorted(set(kept) | {N - 1})

    # Everything but the last kept token is prefilled; the last token is the
    # anchor fed to generate() to emit the first logits.
    prefill_indices = kept[:-1]
    anchor = prompt_tokens[N - 1 : N]

    caches = [SparsePromptCache() for _ in make_prompt_cache(target_model)]

    runs = contiguous_runs(prefill_indices)
    for start, length in runs:
        for c in caches:
            c.offset = start  # true RoPE position of this run's first token
        processed = 0
        while processed < length:
            step = min(prefill_step_size, length - processed)
            lo = start + processed
            chunk = prompt_tokens[lo : lo + step][None]
            target_model(chunk, cache=caches)
            mx.eval([c.state for c in caches])
            # offset advanced contiguously by update_and_fetch; sub-chunks of the
            # same run therefore keep correct positions without re-setting offset.
            processed += step

    # Set the true position of the anchor token (last prompt token at index N-1).
    for c in caches:
        c.offset = N - 1

    return caches, anchor
