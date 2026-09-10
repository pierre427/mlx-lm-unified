"""Hybrid speculative decoding: retrieval-first + confidence-gated draft chain.

Greedy / temperature-0, same-tokenizer only. Per cycle the proposer picks the
cheapest credible source of speculation:

  1. RETRIEVAL FIRST — a suffix automaton over (prompt + generated so far)
     finds the longest suffix of the committed sequence that occurred earlier;
     if the match is long enough (``min_match``) the continuation after the
     earlier occurrence is proposed verbatim, up to ``max_span`` tokens.
     Zero draft-model cost; the design center is copy-heavy output (quoting,
     diffs, structured repetition) where spans verify at ~5 tokens per
     weight-stream.
  2. ELSE a CONFIDENCE-GATED DRAFT CHAIN — the draft model runs
     autoregressively up to ``num_draft_tokens``, but drafting stops at the
     first token whose draft probability falls below ``tau``. Only the
     confident prefix is submitted for verification.
  3. If neither source proposes anything, a plain single-token target step is
     taken: no draft cost, no verify overhead — exactly baseline cost for
     that token. This is what lets the hybrid not lose on open-ended prose.

Verification is a single target forward over
``[pending committed tokens, proposal...]`` with standard greedy
longest-prefix acceptance plus the target's bonus/correction token —
identical semantics (and cache trim bookkeeping) to
``speculative_generate_step``. The draft model is optional: with
``draft_model=None`` this is pure prompt-lookup decoding (PLD) backed by a
suffix automaton.
"""

import copy
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .generate import (
    GenerationResponse,
    draft_tokens_for_budget,
    generate_step,
    generation_stream,
    wired_limit,
)
from .models.cache import (
    KVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
    trim_ragged_prompt_cache,
)
from .sample_utils import LaneRNG, draw_key, make_sampler, make_transformed_logprobs
from .spec_policy import draft_depth_for
from .tokenizer_utils import TokenizerWrapper
from .prompt_lookup import HybridStats as _PromptLookupStatsBase
from .prompt_lookup import plan_proposal_around_verify_cliff
from .speculation_router import RoutedSpeculationPolicy
from .verify_sync import (
    record_verify_sync,
    trace_verify_syncs,
    verify_sync_round,
    verify_sync_status,
)
from . import host_timing as _ht  # uncommitted lab host-stall attribution (default off)
from . import round_levers as _lv  # lab round levers (default off)

_GREEDY = make_sampler(temp=0.0)

# MTP rate gate (see _mtp_draft_verify_loop): one-shot measured keep-or-drop
# decision for the self-spec loop. Warmup cycles gather the spec-rate sample;
# the probe decodes that many tokens plainly (still delivered output); spec
# must beat plain by the margin to stay on.
_RATE_GATE_WARMUP_CYCLES = 8
_RATE_GATE_PROBE_TOKENS = 12
_RATE_GATE_MARGIN = 0.03


def _start_speculation_or_cleanup(caches, required_caches, error_message):
    """Enable rollback atomically and fail without leaking cache state."""
    try:
        for c in caches:
            c.start_speculation()
        if not can_trim_prompt_cache(required_caches):
            raise ValueError(error_message)
    except Exception:
        # Validation happens after start because recurrent caches only become
        # trimmable while recording rollback.  A failed capability gate or a
        # later cache's start failure must still unwind every earlier cache.
        for c in caches:
            try:
                c.stop_speculation()
            except Exception:
                pass
        raise


def _stop_all_speculation(caches):
    """Run ``stop_speculation`` on every cache even when one raises.

    A failing cleanup hook must not leave later caches speculating (holding
    rollback stashes and reporting trimmable state). Every hook runs; the
    first error is re-raised only after all caches had their turn.
    """
    first_error = None
    for c in caches:
        try:
            c.stop_speculation()
        except BaseException as e:  # noqa: BLE001 — cleanup must reach every cache
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error


class SuffixAutomaton:
    """Online suffix automaton over token ids.

    Built incrementally (``extend`` per committed token), it answers the
    retrieval query in O(suffix-link chain) per call: *what is the longest
    suffix of the current sequence that also occurs ending at some earlier
    position, and where does that earlier occurrence end?*

    Each state stores ``first_end`` — the end position (0-based, inclusive)
    of the FIRST occurrence of the substrings it represents. ``first_end``
    is fixed at state creation (clones inherit the original's, since a
    clone's endpos is a superset of the original's and new positions are
    always larger), so no propagation pass is ever needed.

    Pure Python on purpose: per-token construction cost is a few
    microseconds, invisible next to ~76 ms target-model steps.
    """

    __slots__ = ("seq", "_len", "_link", "_next", "_first_end", "_last")

    def __init__(self, tokens: Sequence[int] = ()):
        # State 0 is the root (empty string).
        self.seq: List[int] = []
        self._len = [0]
        self._link = [-1]
        self._next: List[dict] = [{}]
        self._first_end = [-1]
        self._last = 0
        for t in tokens:
            self.extend(t)

    def __len__(self) -> int:
        return len(self.seq)

    def extend(self, token: int) -> None:
        """Append one token to the indexed sequence."""
        token = int(token)
        pos = len(self.seq)
        self.seq.append(token)

        lens, link, nxt, first_end = self._len, self._link, self._next, self._first_end
        cur = len(lens)
        lens.append(lens[self._last] + 1)
        link.append(-1)
        nxt.append({})
        first_end.append(pos)

        p = self._last
        while p != -1 and token not in nxt[p]:
            nxt[p][token] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][token]
            if lens[p] + 1 == lens[q]:
                link[cur] = q
            else:
                clone = len(lens)
                lens.append(lens[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                first_end.append(first_end[q])  # endpos(clone) ⊇ endpos(q)
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def longest_suffix_match(self, max_len: int = 16) -> Tuple[int, int]:
        """Longest suffix of the current sequence with an earlier occurrence.

        Returns ``(match_len, next_pos)`` where ``match_len`` is the length of
        the longest suffix (capped at ``max_len``) that also occurs ending
        strictly before the current end, and ``next_pos`` is the index right
        after that earlier occurrence — i.e. ``seq[next_pos:]`` is the
        retrieval continuation. Returns ``(0, -1)`` when no suffix repeats.

        The earlier occurrence used is the FIRST one in the sequence. Walks
        the suffix-link chain from the last state: endpos sets only grow going
        up the chain, so the deepest state whose ``first_end`` precedes the
        current end holds the longest repeated suffix.
        """
        n = len(self.seq)
        if n < 2:
            return 0, -1
        v = self._last
        while v != 0 and self._first_end[v] >= n - 1:
            v = self._link[v]
        if v == 0:
            return 0, -1
        match_len = min(self._len[v], max_len)
        return match_len, self._first_end[v] + 1


def _require_hybrid_stats(stats) -> None:
    """F7' guard: hybrid/MTP/adaptive paths write draft_*/external_cache_*
    fields that only hybrid_speculative.HybridStats carries. Reject a base
    prompt_lookup stats object at ENTRY with a clear error instead of an
    AttributeError mid-generation."""
    if not hasattr(stats, "draft_proposed"):
        raise TypeError(
            "this generator needs hybrid_speculative.HybridStats (with "
            "draft_*/external_cache_* fields); got a stats object without "
            "them - likely prompt_lookup.HybridStats (F7' footgun)."
        )


@dataclass
class HybridStats(_PromptLookupStatsBase):
    """Per-source accounting for one hybrid generation run.

    Extends prompt_lookup.HybridStats (the F7' unification: shared retrieval/
    plain/span fields are defined ONCE, there) with the draft-chain and
    external-cache accounting the MTP paths need. isinstance-compatible with
    the base, so a hybrid stats object can be passed anywhere the
    prompt-lookup one is expected; the reverse (base into an MTP path) is
    rejected early with a clear error instead of an attribute crash mid-run.
    """

    draft_cycles: int = 0  # cycles whose proposal came from the draft chain
    draft_proposed: int = 0  # tokens proposed by the draft chain
    draft_accepted: int = 0  # ... of which the target accepted
    external_cache_reconciled: bool = False
    external_cache_trimmed_tokens: int = 0
    # MTP rate gate (opt-in): one inline plain probe vs the measured spec
    # rate, then a one-way keep-or-de-latch decision.
    rate_gate_probed: bool = False
    rate_gate_delatched: bool = False
    rate_gate_spec_ms_per_tok: float = 0.0
    rate_gate_plain_ms_per_tok: float = 0.0
    router_plain_cycles: int = 0
    router_reengagements: int = 0
    router_last_num_draft: int = 0
    router_accept_prob: float = 0.0

    @property
    def total_emitted(self) -> int:
        return (
            self.retrieval_accepted
            + self.draft_accepted
            + self.bonus_tokens
            + self.plain_tokens
        )

    @property
    def mean_retrieval_span_proposed(self) -> float:
        return self.retrieval_proposed / max(self.retrieval_cycles, 1)

    @property
    def mean_retrieval_span_accepted(self) -> float:
        return self.retrieval_accepted / max(self.retrieval_cycles, 1)

    @property
    def mean_draft_span_proposed(self) -> float:
        return self.draft_proposed / max(self.draft_cycles, 1)

    @property
    def mean_draft_span_accepted(self) -> float:
        return self.draft_accepted / max(self.draft_cycles, 1)

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        lines = [
            f"cycles: {self.cycles} "
            f"(retrieval {self.retrieval_cycles}, draft {self.draft_cycles}, "
            f"plain {self.plain_cycles})",
            f"tokens: {self.total_emitted} = "
            f"retrieval {self.retrieval_accepted} ({self.retrieval_accepted / tot:.1%})"
            f" + draft {self.draft_accepted} ({self.draft_accepted / tot:.1%})"
            f" + bonus {self.bonus_tokens} ({self.bonus_tokens / tot:.1%})"
            f" + plain {self.plain_tokens} ({self.plain_tokens / tot:.1%})",
        ]
        if self.retrieval_cycles:
            lines.append(
                f"retrieval span: proposed {self.mean_retrieval_span_proposed:.2f} / "
                f"accepted {self.mean_retrieval_span_accepted:.2f} "
                f"(acceptance {self.retrieval_accepted / max(self.retrieval_proposed, 1):.1%})"
            )
        if self.draft_cycles:
            lines.append(
                f"draft span:     proposed {self.mean_draft_span_proposed:.2f} / "
                f"accepted {self.mean_draft_span_accepted:.2f} "
                f"(acceptance {self.draft_accepted / max(self.draft_proposed, 1):.1%})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MTPToken:
    """One token authorized by a self-MTP target verification row."""

    token: int
    logprobs: mx.array
    from_draft: bool


@dataclass
class SelfMTPCachePair:
    target: List[Any]   # standalone when detached, merged when attached
    draft: List[Any]    # persistent MTP-head cache


@dataclass
class SelfMTPLane:
    uid: int
    cur: int                         # committed, emitted, not in target cache
    seed_h: mx.array                 # [1, 1, H], hidden that predicted cur
    pending_hs: Optional[mx.array]   # [1, p, H]
    pending_ts: List[int]            # length p
    token_prefix: mx.array           # committed processor history before cur
    rng: Optional[LaneRNG]
    ntoks: int
    max_tokens: int
    num_draft: int
    sampling_temp: float
    accept_rule: str
    logprob_transform: Optional[Callable]
    logits_processors: List[Callable]
    stats: HybridStats
    share_qsa_indices: bool = False


@dataclass
class DetachedSelfMTPLane:
    lane: SelfMTPLane
    caches: SelfMTPCachePair


@dataclass
class BatchedSelfMTPState:
    lanes: List[SelfMTPLane]          # row order is authoritative
    caches: SelfMTPCachePair          # both groups are merged
    membership_epoch: int
    proposal_open: bool = False
    poisoned: bool = False
    poison_reason: Optional[str] = None
    _open_proposal: Optional["SelfMTPCycleResult"] = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class SelfMTPCycleResult:
    membership_epoch: int
    lane_uids: Tuple[int, ...]
    draft_depths: Tuple[int, ...]       # k_i
    accepted_lengths: Tuple[int, ...]   # a_i
    target_drops: Tuple[int, ...]       # k_i - a_i
    head_drops: Tuple[int, ...]         # k_i
    outputs: Tuple[Tuple[MTPToken, ...], ...]
    # Private cycle tensors retained until commit.
    _old_curs: Tuple[int, ...] = field(default=(), repr=False, compare=False)
    _old_seed_hs: Tuple[mx.array, ...] = field(
        default=(), repr=False, compare=False
    )
    _drafts: Tuple[Tuple[int, ...], ...] = field(
        default=(), repr=False, compare=False
    )
    _vhidden: Tuple[mx.array, ...] = field(default=(), repr=False, compare=False)
    _logprobs: Tuple[mx.array, ...] = field(default=(), repr=False, compare=False)
    _bonuses: Tuple[int, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class GreedyBatchDiagnostic:
    """Classify cross-shape greedy flips without treating near ties as bugs."""

    compared: int
    flip_positions: Tuple[int, ...]
    near_tie_flip_positions: Tuple[int, ...]
    non_near_tie_flip_positions: Tuple[int, ...]
    max_relative_error: float


def classify_greedy_batch_divergence(
    single_logits: mx.array,
    batched_logits: mx.array,
    *,
    shape_noise_band: float = 3e-3,
) -> GreedyBatchDiagnostic:
    """Return the diagnostic used by the real-hardware digest battery.

    The reference top-1/top-2 margin is compared with the documented Qwen4
    cross-shape noise band. A token flip inside that margin is recorded as a
    near tie; a flip outside it is a logical divergence and must fail the
    gate. This helper deliberately does not claim logits are bit-identical.
    """
    if shape_noise_band < 0:
        raise ValueError("shape_noise_band must be non-negative")
    if single_logits.shape != batched_logits.shape or single_logits.ndim < 1:
        raise ValueError(
            "single_logits and batched_logits must have the same [..., V] shape"
        )
    vocab = int(single_logits.shape[-1])
    if vocab < 2:
        raise ValueError("greedy divergence classification requires V >= 2")

    single = single_logits.reshape((-1, vocab)).astype(mx.float32)
    batched = batched_logits.reshape((-1, vocab)).astype(mx.float32)
    if not bool(mx.all(mx.isfinite(single)).item()) or not bool(
        mx.all(mx.isfinite(batched)).item()
    ):
        raise ValueError("greedy divergence classification requires finite logits")

    flips: List[int] = []
    near: List[int] = []
    non_near: List[int] = []
    max_relative_error = 0.0
    for pos in range(int(single.shape[0])):
        want = single[pos]
        got = batched[pos]
        scale = float(mx.max(mx.abs(want)).item())
        denom = max(scale, float(mx.finfo(mx.float32).smallest_normal))
        absolute_error = float(mx.max(mx.abs(got - want)).item())
        error = absolute_error / denom
        max_relative_error = max(max_relative_error, error)
        want_token = int(mx.argmax(want).item())
        got_token = int(mx.argmax(got).item())
        if want_token == got_token:
            continue
        flips.append(pos)
        ordered = mx.sort(want)
        margin = float((ordered[-1] - ordered[-2]).item())
        noise_limit = shape_noise_band * scale
        if margin <= noise_limit and absolute_error <= noise_limit:
            near.append(pos)
        else:
            non_near.append(pos)
    return GreedyBatchDiagnostic(
        compared=int(single.shape[0]),
        flip_positions=tuple(flips),
        near_tie_flip_positions=tuple(near),
        non_near_tie_flip_positions=tuple(non_near),
        max_relative_error=max_relative_error,
    )


def require_only_near_tie_greedy_flips(
    single_logits: mx.array,
    batched_logits: mx.array,
    *,
    shape_noise_band: float = 3e-3,
) -> GreedyBatchDiagnostic:
    """Classify a digest mismatch and fail on every non-near-tie flip."""
    diagnostic = classify_greedy_batch_divergence(
        single_logits,
        batched_logits,
        shape_noise_band=shape_noise_band,
    )
    if diagnostic.non_near_tie_flip_positions:
        raise AssertionError(
            "batched greedy output diverged outside the documented near-tie "
            "band at flattened positions "
            f"{diagnostic.non_near_tie_flip_positions}"
        )
    return diagnostic


def hybrid_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: Optional[nn.Module] = None,
    *,
    num_draft_tokens: int = 3,
    tau: float = 0.55,
    min_match: int = 3,
    max_span: int = 10,
    max_lookback: int = 16,
    max_tokens: int = 256,
    sampler: Optional[Any] = None,
    logits_processors: Optional[Any] = None,
    draft_tokenizer: Optional[Any] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    stats: Optional[HybridStats] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """A generator producing token ids with hybrid (retrieval-first +
    confidence-gated draft chain) speculative decoding. Greedy only.

    Args:
        prompt (mx.array): The input prompt token ids.
        model (nn.Module): The target model.
        draft_model (nn.Module, optional): The draft model (same tokenizer as
          the target). ``None`` gives retrieval-only mode (pure PLD backed by
          the suffix automaton).
        num_draft_tokens (int): Max draft-chain length ``k`` per cycle.
        tau (float): Draft confidence gate — drafting stops at the first token
          whose draft probability is below ``tau``.
        min_match (int): Minimum suffix-match length before retrieval proposes.
        max_span (int): Maximum retrieval continuation span per cycle.
        max_lookback (int): Cap on the suffix-match length considered.
        max_tokens (int): Maximum number of tokens to generate.
        prefill_step_size (int): Chunk size for prompt prefill.
        stats (HybridStats, optional): Updated in place with per-source
          accounting; pass one in to read it after (or during) generation.

    ``sampler``, ``logits_processors``, ``draft_tokenizer`` and
    ``prompt_cache`` are accepted for signature compatibility but must be
    left at their defaults: the hybrid path is greedy-only, same-tokenizer
    only, and manages its own plain ``KVCache`` lists.

    Yields:
        Tuple[int, mx.array, bool]: One committed token, a 1-D vector of log
        probabilities over the vocabulary, and whether the token came from an
        accepted proposal (retrieval or draft) — the same contract as
        ``speculative_generate_step`` (``from_draft`` is ``False`` for bonus/
        correction tokens and plain steps).
    """
    if sampler is not None:
        raise ValueError("hybrid speculative decoding is greedy-only; do not pass a sampler")
    if logits_processors:
        raise ValueError("hybrid speculative decoding does not support logits_processors")
    if draft_tokenizer is not None:
        raise ValueError(
            "hybrid speculative decoding requires target and draft to share a tokenizer"
        )
    if prompt_cache is not None:
        raise NotImplementedError(
            "hybrid speculative decoding manages its own KVCache; "
            "external prompt_cache is not supported"
        )
    if num_draft_tokens < 1:
        raise ValueError("num_draft_tokens must be >= 1")
    if min_match < 1:
        raise ValueError("min_match must be >= 1")
    if not (1 <= max_span):
        raise ValueError("max_span must be >= 1")

    stats = stats if stats is not None else HybridStats()
    _require_hybrid_stats(stats)

    if max_tokens <= 0:
        # Zero-token budget: yield nothing and do no prefill or sampling work.
        return

    y = prompt.astype(mx.uint32)
    # Use each model's own cache layout so hybrid architectures get the right
    # caches (e.g. qwen3_next GatedDeltaNet layers -> ArraysCache, full-attn ->
    # KVCache), not a uniform plain KVCache. Rollback-capable caches make the
    # per-cycle speculative trim exact even for recurrent (non-trimmable) state.
    model_cache = make_prompt_cache(model)
    use_draft = draft_model is not None
    draft_cache = make_prompt_cache(draft_model) if use_draft else None

    # Committed sequence (prompt + everything yielded) and its automaton.
    history: List[int] = [int(t) for t in prompt.tolist()]
    sam = SuffixAutomaton(history)

    def _prefill(m, c, toks):
        # Leave exactly one token unprocessed (mirrors speculative_generate_step's
        # prefill) so the first verify window is a single token + proposal, not the
        # whole prompt tail — keeps the incremental-cache numerics close to plain
        # decode instead of drifting off a large first forward.
        while toks.size > 1:
            n = min(prefill_step_size, toks.size - 1)
            m(toks[:n][None], cache=c)
            mx.eval([layer.state for layer in c])
            toks = toks[n:]
            mx.clear_cache()
        return toks

    def _draft_chain(pending: List[int], k: int) -> Tuple[List[int], int]:
        """Run the confidence-gated draft chain.

        Feeds ``pending`` (committed tokens the draft cache has not seen yet)
        then drafts greedily, stopping at the first token whose probability is
        below ``tau``. Returns ``(proposal, n_fed)`` where ``n_fed`` is how
        many PROPOSED tokens were fed into the draft cache (the caller must
        trim the fed-but-rejected ones after verification).
        """
        proposal: List[int] = []
        n_fed = 0
        feed = mx.array(pending, mx.uint32)
        with mx.stream(generation_stream):
            for _ in range(k):
                logits = draft_model(feed[None], cache=draft_cache)[0, -1, :]
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                tok = mx.argmax(logprobs)
                lp = logprobs[tok]
                mx.eval(tok, lp)
                if math.exp(lp.item()) < tau:
                    break
                proposal.append(int(tok.item()))
                if len(proposal) == k:
                    break  # never feed the last drafted token
                feed = mx.array(proposal[-1:], mx.uint32)
                n_fed += 1
        return proposal, n_fed

    with mx.stream(generation_stream):
        if use_draft:
            draft_tail = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    # After prefill, let rollback-capable caches (GatedDeltaNet ArraysCache) begin
    # recording so each verify forward can be trimmed exactly on rejection; a
    # no-op for plain KV caches. Done post-prefill so we never stash prompt-sized
    # recurrent state — only the small per-cycle verify windows.
    spec_caches = list(model_cache) + (list(draft_cache) if use_draft else [])
    # Rejected proposals must be trimmable; trim_prompt_cache silently no-ops on a
    # non-trimmable cache, which would leave rejected tokens committed and corrupt
    # the output. Both target and draft caches may be trimmed, so validate both.
    _start_speculation_or_cleanup(
        spec_caches,
        spec_caches,
        (
            "hybrid speculative decoding requires a trimmable prompt cache "
            "(recurrent layers need supports_speculative_rollback)."
        ),
    )

    # Tokens committed to `history` but not yet in each model's KV cache.
    pending_target: List[int] = [int(t) for t in y.tolist()]
    pending_draft: List[int] = [int(t) for t in draft_tail.tolist()] if use_draft else []

    ntoks = 0
    try:
        while ntoks < max_tokens:
            stats.cycles += 1
            remaining = max_tokens - ntoks

            # ---- 1. choose a proposal --------------------------------------
            proposal: List[int] = []
            source = "plain"
            n_fed_draft = 0
            budget = remaining - 1  # the bonus token always fills the last slot
            if budget > 0:
                mlen, nxt = sam.longest_suffix_match(max_lookback)
                if mlen >= min_match and 0 <= nxt < len(history):
                    proposal = history[nxt : nxt + min(max_span, budget)]
                    source = "retrieval"
                elif use_draft:
                    proposal, n_fed_draft = _draft_chain(
                        pending_draft, min(num_draft_tokens, budget)
                    )
                    pending_draft = []
                    if proposal:
                        source = "draft"

            # ---- 2. verify: one target forward over [pending, proposal] ----
            n_prop = len(proposal)
            y_verify = mx.array(pending_target + proposal, mx.uint32)
            with mx.stream(generation_stream):
                logits = model(y_verify[None], cache=model_cache)
                rel = logits[0, -(n_prop + 1) :, :]
                logprobs = rel - mx.logsumexp(rel, axis=-1, keepdims=True)
                choices = mx.argmax(logprobs, axis=-1)
            mx.eval(choices)
            choices = choices.tolist()

            n_accept = 0
            while n_accept < n_prop and choices[n_accept] == proposal[n_accept]:
                n_accept += 1
            bonus = choices[n_accept]

            # ---- 3. bookkeeping BEFORE yielding (safe to close at any yield)
            # Target cache: drop the rejected proposal rows. For GatedDeltaNet
            # ArraysCache layers this applies the recorded exact rollback; for
            # KV layers it is the usual offset trim.
            trim_prompt_cache(model_cache, n_prop - n_accept)
            # Draft cache: drop fed-but-rejected draft rows; queue the committed
            # tokens it has not seen for the next draft run.
            if use_draft:
                if source == "draft":
                    trim_prompt_cache(draft_cache, max(n_fed_draft - n_accept, 0))
                    pending_draft = proposal[min(n_fed_draft, n_accept) : n_accept] + [bonus]
                else:
                    # Draft cache untouched this cycle (or consumed only committed
                    # tokens); everything newly committed rides in pending_draft.
                    pending_draft = pending_draft + proposal[:n_accept] + [bonus]
            emitted = proposal[:n_accept] + [bonus]
            history.extend(emitted)
            for t in emitted:
                sam.extend(t)
            pending_target = [bonus]

            if source == "retrieval":
                stats.retrieval_cycles += 1
                stats.retrieval_proposed += n_prop
            elif source == "draft":
                stats.draft_cycles += 1
                stats.draft_proposed += n_prop
            else:
                stats.plain_cycles += 1

            # ---- 4. yield ----------------------------------------------------
            # Delivered-token telemetry is updated exactly at each yield
            # boundary: the consumer may close the generator at any yield
            # (e.g. on EOS), and eager batch accounting would overstate the
            # accepted/bonus/plain token counts.
            for i in range(n_accept):
                ntoks += 1
                if source == "retrieval":
                    stats.retrieval_accepted += 1
                else:
                    stats.draft_accepted += 1
                yield proposal[i], logprobs[i], True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                if source == "plain":
                    stats.plain_tokens += 1
                else:
                    stats.bonus_tokens += 1
                yield bonus, logprobs[n_accept], False
    finally:
        # Stop recording and free rollback stashes on normal completion or when
        # the consumer closes the generator early (e.g. on eos). Every cache's
        # hook runs even if one raises.
        _stop_all_speculation(spec_caches)


def adaptive_pld_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    min_match: int = 3,
    max_span: int = 16,
    cliff_aware_span: bool = False,
    max_lookback: int = 32,
    warmup: int = 48,
    gate: float = 0.12,
    mtp_tail: bool = False,
    num_draft: int = 1,
    persistent_mtp: bool = False,
    mtp_rate_gate: bool = False,
    speculation_router: Optional[RoutedSpeculationPolicy] = None,
    batch_size: int = 1,
    depth_table: Optional[Any] = None,
    prefill_step_size: int = 512,
    stats: Optional[HybridStats] = None,
    prompt_cache: Optional[Any] = None,
    history_prompt: Optional[mx.array] = None,
    mtp_state: Optional[Tuple[Any, Optional[mx.array]]] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Retrieval-only PLD with a one-way latch so it does not lose on no-copy
    output.

    Runs the suffix-automaton retrieval-verify cycle (the ~2x copy-heavy win).
    After ``warmup`` tokens, if the fraction of output that came from retrieval
    is below ``gate``, the work is not copy-heavy, so it **latches once** to a
    tail for all remaining tokens — with no mid-stream thrashing. Copy-heavy work
    never latches, keeping the full PLD speedup with zero regression vs pure PLD.

    The latched tail is a single plain ``generate_step`` (bit-exact, full
    baseline throughput) unless ``mtp_tail=True`` and the model has an MTP head —
    then it self-speculates with the head (``self_mtp_generate_step`` tail),
    giving ~1.1x on the novel code PLD can't help with. So one greedy path serves
    both regimes: copy-heavy -> PLD, novel -> MTP.

    ``persistent_mtp=True`` (with ``mtp_tail``) keeps the MTP tail's KV cache in
    sync with the full committed sequence — the drafting regime vendor-trained
    heads need (see ``self_mtp_generate_step``). The PLD phase routes its verify
    forwards through ``model.model``/``model.logits`` (bit-identical values) so
    the trunk hiddens of every committed token are available, and lazily
    accumulates the (hidden, next_token) pairs; they are teacher-forced into the
    MTP cache in batches (and finally at the latch handoff), so copy-heavy work
    that never latches pays no extra MTP forwards until a flush. Incompatible
    with an external ``prompt_cache``: the cached prefix's hiddens don't exist,
    and an MTP cache missing those positions drafts at wrong RoPE offsets — the
    exact failure persistence exists to fix — so that combination raises.

    ``mtp_state`` lifts that restriction for snapshot restores: pass
    ``(mtp_cache, prev_tail_hidden)`` covering EXACTLY the tokens already in
    ``prompt_cache`` — the teacher-forced pair protocol of
    ``self_mtp_generate_step``'s prefill, with ``prev_tail_hidden`` the trunk
    hidden of the last cached token — and the persistent-MTP path resumes from
    it: the uncached tail is teacher-forced on top, and the MTP tail drafts
    with full context at real RoPE positions. The caller owns the pairing's
    correctness: the prefix snapshot and its draft sidecar must have been
    captured together (see prefix_snapshot_cache.attach_draft_state). The
    structural half is validated at entry — a non-empty cached prefix with a
    missing ``prev_tail_hidden`` or an MTP-cache offset other than
    ``prefix_len - 1`` raises before either cache is mutated.

    ``mtp_rate_gate=True`` protects the MTP tail with a one-shot measured
    break-even check (see ``_mtp_draft_verify_loop``): after a few cycles it
    probes plain decode inline and de-latches the tail permanently if
    speculating isn't actually faster — the tail's acceptance and verify cost
    are workload- and context-dependent, so a config that wins on code
    generation can lose on prose or at long context.

    ``batch_size`` is a concurrency hint from the caller (a multi-lane server
    running several single-stream generators): the MTP tail depth becomes
    ``spec_policy.draft_depth_for(batch_size, num_draft, depth_table)`` — the
    per-batch-size table caps how DEEP the tail drafts, then the hard M5
    verify-width cap (drafts + bonus <= 8) applies. Defaults (``batch_size=1``,
    no table) leave ``num_draft`` unchanged below the cap. Composes with the
    rate gate: the gate decides IF the tail keeps speculating, the policy
    decides HOW DEEP. ``depth_table`` accepts a ``spec_policy.DepthTable`` or
    a ``"1:6,2-4:3,5+:2"``-style spec string; the
    ``MLX_LM_SPEC_DEPTH_TABLE`` env var overrides the default table.

    Greedy only, draft-free; one shared prompt cache. ``prompt_cache`` may hold
    a prefilled prefix; when it is provided, ``prompt`` is the uncached tail and
    ``history_prompt`` must be the full prompt used to seed PLD retrieval history.
    Like all speculative
    decoders, output matches the target's own (batched) greedy — not bit-identical
    to sequential ``generate_step``, since batched/incremental-cache verify forwards
    differ numerically from single-token decode (this holds for upstream
    ``speculative_generate_step`` too). The plain tail runs ``generate_step``
    directly.

    Yields ``(token, logprobs, from_retrieval)``.
    """
    stats = stats if stats is not None else HybridStats()
    _require_hybrid_stats(stats)

    if max_tokens <= 0:
        # Zero-token budget: yield nothing and do no prefill or sampling work
        # (an external prompt_cache is left untouched).
        return

    # Per-batch-size draft-depth policy + hard M5 verify-width cap. Resolved
    # once at entry (concurrency is a per-request hint, not a per-cycle
    # signal); the per-cycle budget clamp in _mtp_draft_verify_loop still
    # applies on top. Resolving reads MLX_LM_SPEC_DEPTH_TABLE, which raises
    # loudly on malformed values — only pay that when an MTP tail can
    # actually run, so pure retrieval-PLD requests never abort on a spec
    # knob they don't use.
    if mtp_tail:
        num_draft = draft_depth_for(batch_size, num_draft, depth_table)

    external_prompt_cache = prompt_cache is not None
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)

    persistent = (
        persistent_mtp and mtp_tail and getattr(model, "mtp", None) is not None
    )
    restored_seed_h = None
    if mtp_state is not None:
        if not (mtp_tail and getattr(model, "mtp", None) is not None):
            raise ValueError(
                "mtp_state requires mtp_tail=True and a model with an MTP head"
            )
        persistent = True
        mtp_cache, restored_seed_h = _restore_mtp_state(cache, mtp_state)
    elif persistent and external_prompt_cache:
        raise ValueError(
            "persistent_mtp is incompatible with an external prompt_cache: the "
            "cached prefix's trunk hiddens are unavailable, so the MTP cache "
            "would draft at wrong RoPE offsets. Pass the full prompt instead, "
            "or provide the matching mtp_state draft sidecar."
        )
    else:
        mtp_cache = model.make_mtp_cache() if persistent else None
    # Committed (hidden, next_token) pairs not yet teacher-forced into
    # mtp_cache; flushed in batches so PLD cycles stay MTP-forward-free.
    mtp_p_hs: List[mx.array] = []
    mtp_p_ts: List[int] = []
    MTP_FLUSH = 256

    y = prompt.astype(mx.uint32)
    with mx.stream(generation_stream):
        # Trunk hidden of the previous chunk's last position; a restored draft
        # sidecar seeds it so the first tail pair lands at the right position.
        prev_h = restored_seed_h if mtp_state is not None else None
        while y.size > 1:  # leave one token for the first verify window
            n = min(prefill_step_size, y.size - 1)
            if persistent:
                # Teacher-force the MTP over pairs (hidden_i, token_{i+1}) so
                # its KV covers the prompt with real positions (same protocol
                # as self_mtp_generate_step's prefill).
                _, h_chunk = _mtp_backbone(model, y[:n][None], cache)
                if prev_h is None:
                    hs, ts = h_chunk[:, :-1], y[1:n][None]
                else:
                    hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
                    ts = y[:n][None]
                if ts.size > 0:
                    model.mtp_step(hs, ts, mtp_cache)
                prev_h = h_chunk[:, -1:, :]
                mx.eval([c.state for c in mtp_cache])
            else:
                model(y[:n][None], cache=cache)
            mx.eval([c.state for c in cache])
            y = y[n:]
            mx.clear_cache()
        if persistent and prev_h is not None:
            # Pair (h_{L-2}, t_{L-1}); t_{L-1} is the pending verify token.
            model.mtp_step(prev_h, y[None], mtp_cache)
    _start_speculation_or_cleanup(
        cache,
        cache,
        (
            "adaptive PLD requires a trimmable prompt cache "
            "(recurrent layers need supports_speculative_rollback)."
        ),
    )

    history_src = history_prompt if history_prompt is not None else prompt
    history: List[int] = [int(t) for t in history_src.tolist()]
    sam = SuffixAutomaton(history)
    pending: List[int] = [int(t) for t in y.tolist()]  # committed, not yet cached

    ntoks = 0
    retrieved = 0  # tokens emitted from accepted retrieval spans
    latched = False
    cached_unyielded = 0
    try:
        # ---- PLD phase: retrieval-verify cycles until latch or done ----------
        while ntoks < max_tokens and not latched:
            stats.cycles += 1
            budget = (max_tokens - ntoks) - 1  # bonus fills the last slot
            proposal: List[int] = []
            if budget > 0:
                mlen, nxt = sam.longest_suffix_match(max_lookback)
                if mlen >= min_match and 0 <= nxt < len(history):
                    nominal_span = min(max_span, budget)
                    available_span = min(len(history) - nxt, budget)
                    chosen_span = min(nominal_span, available_span)
                    if cliff_aware_span:
                        chosen_span = plan_proposal_around_verify_cliff(
                            nominal_span, available_span, len(pending)
                        )
                        if chosen_span < min(nominal_span, available_span):
                            stats.span_snap_cycles += 1
                            stats.span_snap_tokens += (
                                min(nominal_span, available_span) - chosen_span
                            )
                        elif chosen_span > nominal_span:
                            stats.span_extend_cycles += 1
                            stats.span_extend_tokens += chosen_span - nominal_span
                    proposal = history[nxt : nxt + chosen_span]
            n_prop = len(proposal)

            y_verify = mx.array(pending + proposal, mx.uint32)
            verify_rows = len(pending) + n_prop
            stats.verify_span_hist[verify_rows] = (
                stats.verify_span_hist.get(verify_rows, 0) + 1
            )
            with mx.stream(generation_stream):
                if persistent:
                    # Same values as model(...) — logits = lm_head(model.model)
                    # — but the hiddens stay visible for MTP teacher-forcing.
                    vlogit_hidden, vhidden = _mtp_backbone(
                        model, y_verify[None], cache
                    )
                    logits = model.logits(vlogit_hidden)
                else:
                    logits = model(y_verify[None], cache=cache)
                rel = logits[0, -(n_prop + 1) :, :]
                logprobs = rel - mx.logsumexp(rel, axis=-1, keepdims=True)
                choices = mx.argmax(logprobs, axis=-1)
            mx.eval(choices)
            choices = choices.tolist()

            n_accept = 0
            while n_accept < n_prop and choices[n_accept] == proposal[n_accept]:
                n_accept += 1
            bonus = choices[n_accept]

            trim_prompt_cache(cache, n_prop - n_accept)
            # Accepted proposal tokens are already in the cache before they are
            # yielded. If the caller closes early, trim any accepted tokens that
            # never reached the caller so an external cache is not over-advanced.
            cached_unyielded = n_accept
            emitted = proposal[:n_accept] + [bonus]
            if persistent:
                # Predecessor hiddens of the committed span: rows for
                # pending[-1] (predicts emitted[0]) through the last accepted
                # proposal token (predicts the bonus). Rejected rows are
                # excluded, so only committed pairs ever reach the MTP cache.
                pre = len(pending) - 1
                mtp_p_hs.append(vhidden[:, pre : pre + n_accept + 1, :])
                mtp_p_ts.extend(emitted)
                if len(mtp_p_ts) >= MTP_FLUSH:
                    with mx.stream(generation_stream):
                        model.mtp_step(
                            mx.concatenate(mtp_p_hs, axis=1),
                            mx.array([mtp_p_ts], mx.uint32),
                            mtp_cache,
                        )
                        mx.eval([c.state for c in mtp_cache])
                    mtp_p_hs, mtp_p_ts = [], []
            history.extend(emitted)
            for t in emitted:
                sam.extend(t)
            pending = [bonus]
            retrieved += n_accept

            if n_prop:
                stats.retrieval_cycles += 1
                stats.retrieval_proposed += n_prop
            else:
                stats.plain_cycles += 1

            # Delivered-token telemetry updates exactly at each yield boundary
            # so an early close (e.g. EOS) never overstates accepted/bonus
            # counts (see the same pattern in prompt_lookup_generate_step).
            for i in range(n_accept):
                ntoks += 1
                cached_unyielded -= 1
                stats.retrieval_accepted += 1
                yield proposal[i], logprobs[i], True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                if n_prop:
                    stats.bonus_tokens += 1
                else:
                    stats.plain_tokens += 1
                yield bonus, logprobs[n_accept], False

            # Latch to plain once we have enough evidence the work isn't copy-heavy.
            if ntoks >= warmup and retrieved / ntoks < gate:
                latched = True

        # ---- latched tail: self-MTP if available+requested, else plain -------
        if latched and ntoks < max_tokens:
            if mtp_tail and getattr(model, "mtp", None) is not None:
                # Keep speculation ON (MTP verify trims on reject). Bootstrap the
                # seed hidden by forwarding the pending token through the trunk.
                with mx.stream(generation_stream):
                    if persistent and mtp_p_ts:
                        # Bring the MTP cache current: pairs end at (·, bonus);
                        # the loop's first draft call then appends the
                        # (h(bonus), nxt) pair with correct positions.
                        model.mtp_step(
                            mx.concatenate(mtp_p_hs, axis=1),
                            mx.array([mtp_p_ts], mx.uint32),
                            mtp_cache,
                        )
                        mtp_p_hs, mtp_p_ts = [], []
                    blogit_hidden, bh = _mtp_backbone(
                        model, mx.array(pending, mx.uint32)[None], cache
                    )
                    blp = model.logits(blogit_hidden)[0, -1]
                    blp = blp - mx.logsumexp(blp)
                    nxt = int(mx.argmax(blp).item())
                ntoks += 1
                stats.plain_tokens += 1
                yield nxt, blp, False
                yield from _mtp_draft_verify_loop(
                    model,
                    cache,
                    nxt,
                    bh[:, -1:, :],
                    ntoks,
                    max_tokens,
                    num_draft,
                    stats,
                    mtp_cache=mtp_cache,
                    rate_gate=mtp_rate_gate,
                    speculation_router=speculation_router,
                )
            else:
                _stop_all_speculation(cache)
                for tok, lp in generate_step(
                    mx.array(pending, mx.uint32), model,
                    max_tokens=max_tokens - ntoks, prompt_cache=cache, sampler=_GREEDY,
                ):
                    it = int(tok)
                    sam.extend(it)
                    pending = [it]
                    stats.cycles += 1
                    stats.plain_cycles += 1
                    stats.plain_tokens += 1
                    ntoks += 1
                    yield it, lp, False
                    if ntoks == max_tokens:
                        break
    finally:
        try:
            if cached_unyielded > 0:
                trimmed = trim_prompt_cache(cache, cached_unyielded)
                if external_prompt_cache:
                    stats.external_cache_reconciled = True
                    stats.external_cache_trimmed_tokens += int(trimmed or 0)
        finally:
            _stop_all_speculation(cache)


def _mtp_backbone(model, tokens, cache):
    """Return (LM-head hidden, MTP seed hidden) for one trunk forward.

    Conventional MTP models use the same post-norm hidden for both. Qwen4
    scheme A keeps its pre-final-mixer HC multi-stream tensor for drafting.
    """
    if hasattr(model, "mtp_backbone"):
        return model.mtp_backbone(tokens, cache=cache)
    hidden = model.model(tokens, cache=cache)
    return hidden, hidden


def _restore_mtp_state(cache, mtp_state):
    """Validate and unpack an MTP sidecar paired with ``cache``.

    A persistent MTP cache contains one fewer teacher-forced pair than the
    target cache contains tokens.  The sidecar also needs the trunk hidden of
    the final cached token so the first uncached token can form the boundary
    pair.  Validate this relationship before mutating either cache.
    """
    mtp_cache, restored_seed_h = mtp_state
    prefix_len = max((getattr(c, "offset", 0) for c in cache), default=0)
    mtp_offset = max((getattr(c, "offset", 0) for c in mtp_cache), default=0)
    if prefix_len > 0:
        if restored_seed_h is None:
            raise ValueError(
                "mtp_state with a non-empty prompt_cache prefix requires "
                "prev_tail_hidden (the trunk hidden of the last cached "
                "token); without it the boundary pair is skipped and the "
                "MTP cache drafts one position behind."
            )
        if mtp_offset != prefix_len - 1:
            raise ValueError(
                f"mtp_state offset mismatch: MTP cache covers "
                f"{mtp_offset} pairs but the prompt_cache prefix has "
                f"{prefix_len} tokens (expected {prefix_len - 1} pairs). "
                "The prefix snapshot and its draft sidecar were not "
                "captured together."
            )
    elif restored_seed_h is not None or mtp_offset != 0:
        raise ValueError(
            "mtp_state carries a restored draft context "
            f"({mtp_offset} pairs, prev_tail_hidden "
            f"{'set' if restored_seed_h is not None else 'unset'}) but the "
            "prompt_cache prefix is empty; the sidecar must cover exactly "
            "the cached tokens."
        )
    return mtp_cache, restored_seed_h


def self_mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    num_draft: int = 1,
    max_tokens: int = 256,
    prefill_step_size: int = 512,
    sampling_temp: float = 0.0,
    sampling_top_p: float = 1.0,
    sampling_top_k: int = 0,
    sampling_min_p: float = 0.0,
    accept_rule: str = "residual",
    persistent_mtp: bool = False,
    mtp_window_size: Optional[int] = None,
    mtp_sink_size: int = 4,
    mtp_share_qsa_indices: bool = False,
    rate_gate: bool = False,
    speculation_router: Optional[RoutedSpeculationPolicy] = None,
    batch_size: int = 1,
    depth_table: Optional[Any] = None,
    stats: Optional[HybridStats] = None,
    prompt_cache: Optional[List[Any]] = None,
    lane_rng: Optional[LaneRNG] = None,
    mtp_state: Optional[Tuple[Any, Optional[mx.array]]] = None,
    mtp_state_out: Optional[dict] = None,
    logits_processors: Optional[
        List[Callable[[mx.array, mx.array], mx.array]]
    ] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Self-speculative decoding with the model's own MTP (nextn) head.

    The head drafts the next ``num_draft`` tokens from the trunk's last hidden
    state (no external draft model); the trunk verifies them in one batched
    forward and accepts the greedy-matching prefix. Requires ``model.mtp`` +
    ``mtp_step``/``make_mtp_cache``/``logits`` and (for the hybrid trunk) the GDN
    ``record_rollback`` protocol, so verify steps trim exactly on rejection.
    The head is depth-1, so ``num_draft=1`` is the trained regime; k>1 chains the
    head on its own hidden (out of training distribution — acceptance decays).

    Greedy by default. When ``sampling_temp > 0``, the MTP path uses standard
    speculative rejection sampling with temperature-scaled target and draft
    distributions. This is exact for temperature-only sampling. When
    ``sampling_top_p``/``sampling_top_k``/``sampling_min_p`` request a
    transformed distribution, the SAME ``make_sampler``-order transform is
    applied to both draft and target log-probabilities and the residual rule
    runs on the transformed pair — exact for the fully transformed
    distribution as well (XTC is not supported).

    ``accept_rule`` selects the temp>0 verification rule; greedy (temp=0)
    decisions are rule-independent and bit-identical. ``"residual"`` (the
    default — the incumbent behavior) is Leviathan/SpecDec rejection
    sampling: accept draft x ~ q with probability min(1, p(x)/q(x)), resample
    rejects from normalized relu(p - q). ``"block"`` is block verification
    (arXiv 2403.10444): accepts draft PREFIXES by cumulative joint
    likelihood ratio — provably accepts at least as many tokens as per-token
    rejection sampling with the SAME output distribution. ``"exact"`` is the
    conservative sampled-token-match rule (upstream
    ``speculative_generate_step`` semantics: accept while the target's own
    sample equals the draft's), kept as an A/B baseline.

    ``persistent_mtp=True`` keeps ONE MTP KV cache in sync with the committed
    sequence (teacher-forced from trunk hiddens during prefill and after each
    verify) instead of a fresh empty cache per draft cycle. The head then
    drafts with full context and real RoPE positions — the regime it was
    trained in — which can lift acceptance dramatically (Hy3-REAP50: 36.8% ->
    84.4% on code). Costs one extra (single-layer) MTP forward per cycle plus
    ~1 layer-equivalent of prefill; requires the MTP cache to be trimmable.

    ``rate_gate=True`` adds the one-shot measured break-even check from
    ``_mtp_draft_verify_loop``: keep speculating only if it is actually faster
    than an inline plain-decode probe; otherwise fall back to plain for the
    rest of the generation.

    ``mtp_window_size`` bounds only the persistent draft head to attention
    sinks plus a recent window. The target cache and verification remain
    full-context, so the A/B changes proposal quality and cost, never which
    model authorizes the emitted token.

    ``mtp_share_qsa_indices=True`` lets model-specific sparse-attention draft
    heads compute top-k blocks on the first MTP step and reuse them on later
    chained steps.  Verification is unchanged.  Models without the optional
    ``mtp_start_cycle`` hook ignore it.

    ``batch_size``/``depth_table`` apply the per-batch-size draft-depth policy
    (``spec_policy.draft_depth_for``): the table caps how deep the head drafts
    when the caller runs multiple lanes, then the hard M5 verify-width cap
    (drafts + bonus <= 8) applies. Defaults leave ``num_draft`` unchanged
    below the cap; the gate decides IF, the policy decides HOW DEEP.

    ``logits_processors`` are applied to each target verification position
    with its exact speculative token prefix. Stateful processors therefore use
    the same rewind-on-shorter-history contract as external-draft speculative
    generation and make decisions only from committed-or-tentatively-accepted
    prefixes.

    ``lane_rng`` is this request's ``sample_utils.LaneRNG``: every draw of the
    loop takes a subkey from it, so the request's tokens do not depend on
    traffic decoded beside it. Leave it ``None`` for the global ``mx.random``
    stream (byte-identical to a build without lane keys). Greedy requests draw
    nothing and are unaffected either way.

    ``prompt_cache`` lets serving own the target cache so the verified result
    can be inserted into its automatic prefix cache. ``mtp_state`` restores
    the matching persistent draft sidecar: ``(mtp_cache, prev_tail_hidden)``.
    Its offsets are validated against the target cache before either is
    mutated. ``mtp_state_out`` is a caller-owned mapping populated on exit
    with an exact, fully evaluated sidecar, plus the lane's ``rng_key`` /
    ``rng_draws`` so a resumed request continues its own stream.

    Yields ``(token, logprobs, from_draft)``.
    """
    if getattr(model, "mtp", None) is None:
        raise ValueError("model has no MTP head (build with mtp_num_hidden_layers>0)")
    if accept_rule not in ("exact", "residual", "block"):
        raise ValueError(
            f"accept_rule must be 'exact', 'residual', or 'block'; got {accept_rule!r}"
        )
    if lane_rng is not None and not isinstance(lane_rng, LaneRNG):
        # A bare seed is the footgun this exists to stop: a lane must CARRY and
        # split one key, not re-derive key(seed) per call, which repeats draws.
        raise TypeError(
            "lane_rng must be a sample_utils.LaneRNG (carried across calls); "
            f"got {type(lane_rng).__name__}"
        )
    logprob_transform = _make_sampling_transform(
        sampling_temp, sampling_top_p, sampling_top_k, sampling_min_p
    )
    if logprob_transform is not None and accept_rule != "residual":
        raise ValueError(
            "transformed sampling (top-p/top-k/min-p) supports only "
            f"accept_rule='residual'; got {accept_rule!r}"
        )
    stats = stats if stats is not None else HybridStats()
    _require_hybrid_stats(stats)

    # Per-batch-size draft-depth policy + hard M5 verify-width cap (resolved
    # once at entry; the per-cycle budget clamp still applies on top).
    num_draft = draft_depth_for(batch_size, num_draft, depth_table)

    if max_tokens <= 0:
        # Zero-token budget: yield nothing and do no prefill or sampling work.
        return
    if mtp_window_size is not None and not persistent_mtp:
        raise ValueError("mtp_window_size requires persistent_mtp=True")

    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    if mtp_state is not None:
        if not persistent_mtp:
            raise ValueError("mtp_state requires persistent_mtp=True")
        mtp_cache, restored_seed_h = _restore_mtp_state(cache, mtp_state)
    else:
        restored_seed_h = None
        mtp_cache = (
            model.make_mtp_cache(mtp_window_size, mtp_sink_size)
            if persistent_mtp and mtp_window_size is not None
            else model.make_mtp_cache() if persistent_mtp else None
        )

    y = prompt.astype(mx.uint32)
    processor_prompt = y
    with mx.stream(generation_stream):
        prev_h = restored_seed_h
        while y.size > 1:  # leave one token to produce the seed hidden
            n = min(prefill_step_size, y.size - 1)
            _, h_chunk = _mtp_backbone(model, y[:n][None], cache)
            if persistent_mtp:
                # Teacher-force the MTP over pairs (hidden_i, token_{i+1}) so
                # its KV covers the prompt with real positions.
                if prev_h is None:
                    hs, ts = h_chunk[:, :-1], y[1:n][None]
                else:
                    hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
                    ts = y[:n][None]
                if ts.size > 0:
                    model.mtp_step(hs, ts, mtp_cache)
                prev_h = h_chunk[:, -1:, :]
                mx.eval([c.state for c in mtp_cache])
            mx.eval([c.state for c in cache])
            y = y[n:]
            mx.clear_cache()
        if persistent_mtp and prev_h is not None:
            model.mtp_step(prev_h, y[None], mtp_cache)  # pair (h_{L-2}, t_{L-1})
        logit_hidden, hidden = _mtp_backbone(model, y[None], cache)
        seed_h = hidden[:, -1:, :]
        first_logits = model.logits(logit_hidden[:, -1:, :])[0, -1]
        first_logits = _apply_logits_processors(
            logits_processors, y=processor_prompt, logits=first_logits
        )
        first_lp = (
            logprob_transform(first_logits)
            if logprob_transform is not None
            else _temperature_logprobs(first_logits, sampling_temp)
        )
        cur = _sample_from_logprobs(first_lp, sampling_temp, rng=lane_rng)
    _start_speculation_or_cleanup(
        cache,
        cache,
        (
            "self-MTP decoding requires a trimmable prompt cache "
            "(recurrent layers need supports_speculative_rollback)."
        ),
    )

    try:
        # Count the token at its yield boundary (not after): an immediate
        # close must still account for the delivered first token.
        stats.plain_tokens += 1
        yield cur, first_lp, False
        yield from _mtp_draft_verify_loop(
            model,
            cache,
            cur,
            seed_h,
            1,
            max_tokens,
            num_draft,
            stats,
            sampling_temp,
            accept_rule=accept_rule,
            logprob_transform=logprob_transform,
            mtp_cache=mtp_cache,
            share_qsa_indices=mtp_share_qsa_indices,
            rate_gate=rate_gate,
            speculation_router=speculation_router,
            logits_processors=logits_processors,
            rng=lane_rng,
            # Match generate_step's processor contract: the immutable prompt
            # prefix is present before tentative draft tokens are appended and
            # rewound at commit boundaries.
            token_prefix=processor_prompt,
            mtp_state_out=mtp_state_out,
        )
    finally:
        _stop_all_speculation(cache)


def _temperature_logprobs(logits, sampling_temp: float = 0.0):
    logits = logits.astype(mx.float32)
    if sampling_temp and sampling_temp > 0:
        logits = logits / float(sampling_temp)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _apply_logits_processors(logits_processors, y, logits):
    """Apply processors with the same rank convention as ``generate_step``."""
    if not logits_processors:
        return logits
    batched = logits[None] if logits.ndim == 1 else logits
    for processor in logits_processors:
        batched = processor(y, batched)
    return batched[0] if logits.ndim == 1 else batched


def _sample_from_logprobs(logprobs, sampling_temp: float = 0.0, *, rng=None) -> int:
    # Greedy takes no draw, so it consumes no lane key (see the RNG note in
    # ``_mtp_draft_verify_loop_impl``).
    if sampling_temp and sampling_temp > 0:
        record_verify_sync("hybrid.sample.categorical_item")
        return int(mx.random.categorical(logprobs, key=draw_key(rng)).item())
    record_verify_sync("hybrid.sample.argmax_item")
    return int(mx.argmax(logprobs).item())


def _residual_sample(
    target_logprobs,
    draft_logprobs,
    sampling_temp: float,
    scale: float = 1.0,
    *,
    rng=None,
) -> int:
    # ``scale`` is the block-verification cumulative ratio p_tau; 1.0 (the
    # per-token rule) multiplies bit-exactly, so the default is unchanged.
    residual = mx.maximum(scale * mx.exp(target_logprobs) - mx.exp(draft_logprobs), 0.0)
    total = mx.sum(residual)
    record_verify_sync("hybrid.residual.total_eval")
    mx.eval(total)
    record_verify_sync("hybrid.residual.total_item")
    if float(total.item()) <= 0.0:
        return _sample_from_logprobs(target_logprobs, sampling_temp, rng=rng)
    residual_logprobs = mx.log(residual / total)
    record_verify_sync("hybrid.residual.categorical_item")
    return int(mx.random.categorical(residual_logprobs, key=draw_key(rng)).item())


def _make_sampling_transform(
    sampling_temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> Optional[Callable[[mx.array], mx.array]]:
    """Return a shared draft/target logprob transform, or ``None``.

    ``None`` means no filter is active (or ``temp == 0``, where the filters
    cannot change the argmax) and the caller keeps the incumbent
    temperature-only path bit-exactly. Otherwise the returned callable maps
    raw logits to the log-probabilities ``make_sampler`` samples from, with
    filtered tokens exactly ``-inf`` — applied identically to draft and
    target so the residual acceptance ratio is well-defined.
    """
    filtered = (0.0 < top_p < 1.0) or top_k > 0 or min_p > 0.0
    if not filtered or not sampling_temp or sampling_temp <= 0:
        return None
    return make_transformed_logprobs(
        sampling_temp, top_p=top_p, top_k=top_k, min_p=min_p
    )


def _batched_residual_verify(
    logprobs, draft_logprobs, drafts, sampling_temp: float, *, rng=None
):
    """Residual acceptance over all k positions with one GPU sync.

    Same rule as ``_accept_sampled_draft`` scanned per position, but every
    uniform and log-ratio is computed in one graph and drained with a single
    ``mx.eval`` (vs one per accepted position). A draft the target transform
    filtered has ratio exactly 0 and is always rejected — ``u <= 0`` cannot
    rescue it. Returns ``(n_accept, bonus)``.
    """
    k = len(drafts)
    d = mx.array(drafts)[:, None]
    target_at = mx.take_along_axis(logprobs[:k], d, axis=-1)[:, 0]
    draft_at = mx.take_along_axis(mx.stack(draft_logprobs), d, axis=-1)[:, 0]
    ratios = mx.exp(mx.minimum(target_at - draft_at, 0.0))
    us = _draw_mtp_acceptance_uniforms(k, rng=rng)
    record_verify_sync("hybrid.residual_verify.eval")
    mx.eval(ratios, us)
    record_verify_sync("hybrid.residual_verify.ratios_tolist")
    record_verify_sync("hybrid.residual_verify.uniforms_tolist")
    ratios, us = ratios.tolist(), us.tolist()
    n_accept = 0
    while (
        n_accept < k
        and ratios[n_accept] > 0.0
        and us[n_accept] <= ratios[n_accept]
    ):
        n_accept += 1
    if n_accept < k:
        bonus = _residual_sample(
            logprobs[n_accept], draft_logprobs[n_accept], sampling_temp, rng=rng
        )
    else:
        bonus = _sample_from_logprobs(logprobs[n_accept], sampling_temp, rng=rng)
    return n_accept, bonus


def _draw_mtp_acceptance_uniforms(k: int, *, rng=None) -> mx.array:
    """Draw one lane's native acceptance vector, never a padded batch shape."""
    if k < 0:
        raise ValueError("acceptance width must be non-negative")
    return mx.random.uniform(shape=(k,), key=draw_key(rng))


def _accept_sampled_draft(
    target_logprobs, draft_logprobs, token: int, *, rng=None
) -> bool:
    # min(1, p/q) computed in log space: exp(min(log p - log q, 0)). A
    # linear-space q floor (e.g. max(q, 1e-30)) would bias acceptance for
    # representable sub-floor q — q=1e-35, p=1e-34 must accept with
    # probability 1, not p/floor.
    log_ratio = mx.minimum(target_logprobs[token] - draft_logprobs[token], 0.0)
    ratio = mx.exp(log_ratio)
    u = mx.random.uniform(shape=(), key=draw_key(rng))
    record_verify_sync("hybrid.accept_sampled.eval")
    mx.eval(ratio, u)
    record_verify_sync("hybrid.accept_sampled.uniform_item")
    record_verify_sync("hybrid.accept_sampled.ratio_item")
    return float(u.item()) <= float(ratio.item())


def _block_verify(logprobs, draft_logprobs, drafts, sampling_temp: float, *, rng=None):
    """Block verification (Sun et al., arXiv 2403.10444): accept a draft
    PREFIX by cumulative joint likelihood ratio instead of independent
    per-token coin flips.

    ``p_cum_i = min(p_cum_{i-1} * p_i(x_i)/q_i(x_i), 1)`` tracks the joint
    target/draft ratio of the drafted prefix ``x_1..x_i``. Each prefix length
    ``i`` is checked against a threshold ``h_i``: the full block uses
    ``h_k = p_cum_k``; shorter prefixes use the residual-mass form
    ``h_i = S_i / (S_i + (1 - p_cum_i))`` with
    ``S_i = sum(relu(p_cum_i * p_{i+1} - q_{i+1}))`` over the vocabulary.
    The accepted length ``tau`` is the LARGEST ``i`` whose check
    ``eta_i <= h_i`` passes — a later position can rescue an earlier
    failure, which is why block verification provably accepts at least as
    many tokens in expectation as per-token rejection sampling while
    preserving the target distribution exactly. On ``tau < k`` the
    correction token comes from the scaled residual
    ``relu(p_cum_tau * p - q)``; on ``tau == k`` it is a plain target
    sample (the bonus).

    ``logprobs`` has ``k+1`` target rows (row ``i`` conditions on
    ``x_1..x_i``), ``draft_logprobs`` the ``k`` draft rows that produced
    ``drafts``. Returns ``(n_accept, bonus)``.
    """
    k = len(drafts)
    etas = _draw_mtp_acceptance_uniforms(k, rng=rng)
    record_verify_sync("hybrid.block_verify.uniforms_eval")
    mx.eval(etas)
    p_cums = [1.0]
    p_cum = 1.0
    tau = 0
    for i in range(k):
        d = drafts[i]
        record_verify_sync("hybrid.block_verify.log_ratio_item")
        log_ratio = float((logprobs[i][d] - draft_logprobs[i][d]).item())
        p_cum = min(p_cum * math.exp(log_ratio), 1.0)
        p_cums.append(p_cum)
        if i == k - 1:
            h = p_cum
        else:
            record_verify_sync("hybrid.block_verify.residual_item")
            s = float(
                mx.sum(
                    mx.maximum(
                        p_cum * mx.exp(logprobs[i + 1]) - mx.exp(draft_logprobs[i + 1]),
                        0.0,
                    )
                ).item()
            )
            denom = s + (1.0 - p_cum)
            h = 1.0 if denom <= 0.0 else s / denom
        record_verify_sync("hybrid.block_verify.uniform_item")
        if float(etas[i].item()) <= h:
            tau = i + 1
    if tau == k:
        bonus = _sample_from_logprobs(logprobs[k], sampling_temp, rng=rng)
    else:
        bonus = _residual_sample(
            logprobs[tau],
            draft_logprobs[tau],
            sampling_temp,
            scale=p_cums[tau],
            rng=rng,
        )
    return tau, bonus


def _self_mtp_cache_offset(cache) -> int:
    offset = getattr(cache, "offset", 0)
    if isinstance(offset, mx.array):
        if int(offset.size) != 1:
            raise ValueError("detached self-MTP caches must contain exactly one row")
        return int(offset.item())
    return int(offset)


def _self_mtp_group_offset(caches: Sequence[Any]) -> int:
    return max((_self_mtp_cache_offset(c) for c in caches), default=0)


def _reject_unsupported_self_mtp_caches(caches: Sequence[Any]) -> None:
    # Windowed (sink/rotating) caches cannot do the per-row ragged speculative
    # rollback the transaction needs, so they stay unsupported. Quantized caches
    # CAN (the batched transaction is bit-exact on a BatchQuantizedKVCache); they
    # are gated by policy (allow_quantized_kv) at the server/constructor, not
    # refused here as a capability limit.
    unsupported = [
        type(cache).__name__
        for cache in caches
        if "SinkWindow" in type(cache).__name__
        or "Rotating" in type(cache).__name__
    ]
    if unsupported:
        raise ValueError(
            "batched self-MTP excludes windowed caches; got "
            + ", ".join(unsupported)
        )


def _validate_detached_self_mtp(detached: DetachedSelfMTPLane) -> None:
    lane = detached.lane
    if lane.pending_hs is not None or lane.pending_ts:
        raise ValueError("a detached self-MTP lane must have no pending pairs")
    if lane.seed_h is None or lane.seed_h.ndim != 3 or lane.seed_h.shape[:2] != (1, 1):
        raise ValueError("a detached lane seed_h must have shape [1, 1, H]")
    if not detached.caches.target or not detached.caches.draft:
        raise ValueError("a detached self-MTP lane requires target and draft caches")
    _reject_unsupported_self_mtp_caches(detached.caches.target)
    _reject_unsupported_self_mtp_caches(detached.caches.draft)
    covered = _self_mtp_group_offset(detached.caches.target)
    draft_offset = _self_mtp_group_offset(detached.caches.draft)
    if covered <= 0 or draft_offset != covered - 1:
        raise ValueError(
            "detached self-MTP cache mismatch: target covers "
            f"{covered} tokens but draft covers {draft_offset} pairs"
        )


def _merge_self_mtp_cache_groups(groups: Sequence[Sequence[Any]]) -> List[Any]:
    if not groups:
        return []
    width = len(groups[0])
    if width == 0 or any(len(group) != width for group in groups):
        raise ValueError("self-MTP cache groups must have the same non-zero width")
    merged = []
    for rows in zip(*groups):
        merge = getattr(type(rows[0]), "merge", None)
        if merge is None:
            raise ValueError(f"{type(rows[0]).__name__} cannot merge cache rows")
        merged.append(merge(list(rows)))
    return merged


def _extract_self_mtp_cache_pair(
    caches: SelfMTPCachePair,
    indices: Sequence[int],
    *,
    batched: bool = False,
) -> SelfMTPCachePair:
    """Copy-build a cache pair for ``indices`` without mutating the live pair."""
    indices = [int(index) for index in indices]
    if not indices:
        return SelfMTPCachePair(target=[], draft=[])
    if len(indices) == 1 and not batched:
        index = indices[0]
        return SelfMTPCachePair(
            target=[cache.extract(index) for cache in caches.target],
            draft=[cache.extract(index) for cache in caches.draft],
        )
    rows = [
        SelfMTPCachePair(
            target=[cache.extract(index) for cache in caches.target],
            draft=[cache.extract(index) for cache in caches.draft],
        )
        for index in indices
    ]
    return SelfMTPCachePair(
        target=_merge_self_mtp_cache_groups([row.target for row in rows]),
        draft=_merge_self_mtp_cache_groups([row.draft for row in rows]),
    )


def _copy_build_self_mtp_cache_pair(
    current: Optional[SelfMTPCachePair],
    current_rows: int,
    joining: Sequence[SelfMTPCachePair],
) -> SelfMTPCachePair:
    """Build a complete replacement pair before publishing membership.

    ``extend`` and ``filter`` mutate layer objects one at a time. A late layer
    failure can therefore leave target and draft groups with different row
    membership. Extracting canonical rows and merging replacements keeps the
    old pair untouched until every layer has succeeded.
    """
    rows = []
    if current is not None:
        rows.extend(
            SelfMTPCachePair(
                target=[cache.extract(index) for cache in current.target],
                draft=[cache.extract(index) for cache in current.draft],
            )
            for index in range(current_rows)
        )
    rows.extend(joining)
    if not rows:
        return SelfMTPCachePair(target=[], draft=[])
    return SelfMTPCachePair(
        target=_merge_self_mtp_cache_groups([row.target for row in rows]),
        draft=_merge_self_mtp_cache_groups([row.draft for row in rows]),
    )


def _poison_self_mtp_batch(batch: BatchedSelfMTPState, reason: str) -> None:
    batch.poisoned = True
    batch.poison_reason = str(reason)


def _require_healthy_self_mtp_batch(batch: BatchedSelfMTPState) -> None:
    if batch.poisoned:
        reason = batch.poison_reason or "unproved transaction rollback"
        raise RuntimeError(f"self-MTP batch is poisoned: {reason}")


def _restart_live_self_mtp_or_poison(
    batch: BatchedSelfMTPState,
    cause: BaseException,
) -> None:
    """Restore rollback recording after a failed copy-build operation."""
    try:
        _start_speculation_or_cleanup(
            batch.caches.target,
            batch.caches.target,
            "batched self-MTP requires ragged-trimmable target caches",
        )
    except BaseException as restart_error:
        _poison_self_mtp_batch(
            batch,
            f"membership rebuild failed ({cause}); rollback restart failed: "
            f"{restart_error}",
        )
        raise RuntimeError(batch.poison_reason) from restart_error


def _prepare_self_mtp_cache_group(caches, lengths, right_padding) -> None:
    for cache in caches:
        prepare = getattr(cache, "prepare_self_mtp_step", cache.prepare)
        prepare(lengths=lengths, right_padding=right_padding)


def _finalize_self_mtp_cache_group(caches) -> None:
    first_error = None
    for cache in caches:
        try:
            finalize = getattr(cache, "finalize_self_mtp_step", cache.finalize)
            finalize()
        except BaseException as exc:  # noqa: BLE001 - finish every cache entry
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _eval_self_mtp_lane_state(detached: DetachedSelfMTPLane) -> None:
    values = [c.state for c in detached.caches.target]
    values.extend(c.state for c in detached.caches.draft)
    values.append(detached.lane.seed_h)
    if detached.lane.rng is not None:
        values.append(detached.lane.rng.key)
    mx.eval(*values)


def _lane_mtp_logprobs(lane: SelfMTPLane, logits: mx.array) -> mx.array:
    if lane.logprob_transform is not None:
        return lane.logprob_transform(logits)
    return _temperature_logprobs(logits, lane.sampling_temp)


def prepare_self_mtp_lane(
    prompt: mx.array,
    model: nn.Module,
    *,
    uid: int,
    max_tokens: int,
    prompt_cache: Optional[List[Any]],
    mtp_state: Optional[Tuple[List[Any], mx.array]],
    lane_rng: Optional[LaneRNG],
    num_draft: int,
    sampling_temp: float,
    sampling_top_p: float,
    sampling_top_k: int,
    sampling_min_p: float,
    accept_rule: str,
    logits_processors: List[Callable],
    prefill_step_size: int,
    share_qsa_indices: bool,
) -> Tuple[DetachedSelfMTPLane, MTPToken]:
    """Prefill one canonical persistent self-MTP lane without attaching it."""
    if getattr(model, "mtp", None) is None:
        raise ValueError("model has no MTP head")
    if max_tokens <= 0:
        raise ValueError("prepare_self_mtp_lane requires max_tokens > 0")
    if num_draft < 1:
        raise ValueError("prepare_self_mtp_lane requires num_draft >= 1")
    if prefill_step_size <= 0:
        raise ValueError("prefill_step_size must be positive")
    if accept_rule not in ("exact", "residual", "block"):
        raise ValueError(
            f"accept_rule must be 'exact', 'residual', or 'block'; got {accept_rule!r}"
        )
    if lane_rng is not None and not isinstance(lane_rng, LaneRNG):
        raise TypeError("lane_rng must be a sample_utils.LaneRNG")
    if prompt.ndim != 1 or int(prompt.size) == 0:
        raise ValueError("prompt must be a non-empty rank-1 token array")

    transform = _make_sampling_transform(
        sampling_temp, sampling_top_p, sampling_top_k, sampling_min_p
    )
    if transform is not None and accept_rule != "residual":
        raise ValueError(
            "transformed sampling supports only accept_rule='residual'"
        )

    target_cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    _reject_unsupported_self_mtp_caches(target_cache)
    if mtp_state is None:
        draft_cache = model.make_mtp_cache()
        restored_seed_h = None
    else:
        draft_cache, restored_seed_h = _restore_mtp_state(target_cache, mtp_state)
    _reject_unsupported_self_mtp_caches(draft_cache)

    processor_prompt = prompt.astype(mx.uint32)
    y = processor_prompt
    prev_h = restored_seed_h
    with mx.stream(generation_stream):
        while y.size > 1:
            n = min(prefill_step_size, int(y.size) - 1)
            _, h_chunk = _mtp_backbone(model, y[:n][None], target_cache)
            if prev_h is None:
                hs, ts = h_chunk[:, :-1], y[1:n][None]
            else:
                hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
                ts = y[:n][None]
            if ts.size > 0:
                model.mtp_step(hs, ts, draft_cache)
            prev_h = h_chunk[:, -1:, :]
            mx.eval([c.state for c in target_cache], [c.state for c in draft_cache])
            y = y[n:]
            mx.clear_cache()
        if prev_h is not None:
            model.mtp_step(prev_h, y[None], draft_cache)
        logit_hidden, hidden = _mtp_backbone(model, y[None], target_cache)
        seed_h = hidden[:, -1:, :]
        logits = model.logits(logit_hidden[:, -1:, :])[0, -1]
        logits = _apply_logits_processors(logits_processors, processor_prompt, logits)
        first_lp = transform(logits) if transform is not None else _temperature_logprobs(
            logits, sampling_temp
        )
        cur = _sample_from_logprobs(first_lp, sampling_temp, rng=lane_rng)

    stats = HybridStats()
    stats.plain_tokens = 1
    lane = SelfMTPLane(
        uid=int(uid),
        cur=cur,
        seed_h=seed_h,
        pending_hs=None,
        pending_ts=[],
        token_prefix=processor_prompt,
        rng=lane_rng,
        ntoks=1,
        max_tokens=int(max_tokens),
        num_draft=int(num_draft),
        sampling_temp=float(sampling_temp),
        accept_rule=accept_rule,
        logprob_transform=transform,
        logits_processors=list(logits_processors or []),
        stats=stats,
        share_qsa_indices=bool(share_qsa_indices),
    )
    detached = DetachedSelfMTPLane(
        lane=lane,
        caches=SelfMTPCachePair(target=target_cache, draft=draft_cache),
    )
    _eval_self_mtp_lane_state(detached)
    _validate_detached_self_mtp(detached)
    return detached, MTPToken(cur, first_lp, False)


def attach_self_mtp_lanes(
    model: nn.Module,
    batch: Optional[BatchedSelfMTPState],
    joining: Sequence[DetachedSelfMTPLane],
) -> BatchedSelfMTPState:
    """Attach canonical rows at a transaction boundary and restart rollback."""
    joining = list(joining)
    if batch is not None:
        _require_healthy_self_mtp_batch(batch)
        if batch.proposal_open:
            raise RuntimeError("cannot attach self-MTP lanes while a proposal is open")
    if not joining:
        if batch is None:
            raise ValueError("cannot create an empty self-MTP batch")
        return batch
    for detached in joining:
        _validate_detached_self_mtp(detached)

    old_lanes = [] if batch is None else batch.lanes
    uids = [lane.uid for lane in old_lanes] + [item.lane.uid for item in joining]
    if len(set(uids)) != len(uids):
        raise ValueError("self-MTP lane uid values must be unique")
    configured = {lane.num_draft for lane in old_lanes}
    configured.update(item.lane.num_draft for item in joining)
    if len(configured) != 1:
        raise ValueError("adaptive per-lane self-MTP depth is excluded")
    share_modes = {lane.share_qsa_indices for lane in old_lanes}
    share_modes.update(item.lane.share_qsa_indices for item in joining)
    if len(share_modes) != 1:
        raise ValueError("mixed shared-QSA modes cannot enter one self-MTP batch")

    if batch is None or not batch.lanes:
        incoming = _copy_build_self_mtp_cache_pair(
            None,
            0,
            [item.caches for item in joining],
        )
        epoch = 1 if batch is None else batch.membership_epoch + 1
        result = BatchedSelfMTPState(
            lanes=[item.lane for item in joining],
            caches=incoming,
            membership_epoch=epoch,
        )
    else:
        try:
            _stop_all_speculation(batch.caches.target)
        except BaseException as error:
            _poison_self_mtp_batch(batch, f"rollback stop failed: {error}")
            raise
        try:
            replacement = _copy_build_self_mtp_cache_pair(
                batch.caches,
                len(batch.lanes),
                [item.caches for item in joining],
            )
            _start_speculation_or_cleanup(
                replacement.target,
                replacement.target,
                "batched self-MTP requires ragged-trimmable target caches",
            )
        except BaseException as error:
            _restart_live_self_mtp_or_poison(batch, error)
            raise
        batch.caches = replacement
        batch.lanes = [*batch.lanes, *(item.lane for item in joining)]
        batch.membership_epoch += 1
        result = batch
    if batch is None or not old_lanes:
        _start_speculation_or_cleanup(
            result.caches.target,
            result.caches.target,
            "batched self-MTP requires ragged-trimmable target caches",
        )
    return result


def _propose_batched_self_mtp_impl(
    model: nn.Module,
    batch: BatchedSelfMTPState,
) -> SelfMTPCycleResult:
    """Open one batched draft/verify transaction over the current membership."""
    with verify_sync_round():
        return _propose_batched_self_mtp_round(model, batch)


def _propose_batched_self_mtp_round(
    model: nn.Module,
    batch: BatchedSelfMTPState,
) -> SelfMTPCycleResult:
    if batch.proposal_open:
        raise RuntimeError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot propose on an empty self-MTP batch")
    n_lanes = len(batch.lanes)
    lane_uids = tuple(lane.uid for lane in batch.lanes)
    if len(set(lane_uids)) != n_lanes:
        raise ValueError("self-MTP batch contains duplicate lane uid values")

    k_vector = tuple(
        min(lane.num_draft, max(lane.max_tokens - lane.ntoks - 1, 0))
        for lane in batch.lanes
    )
    active_share_modes = {
        lane.share_qsa_indices
        for lane, k in zip(batch.lanes, k_vector)
        if k > 1
    }
    if len(active_share_modes) > 1:
        raise ValueError("mixed shared-QSA modes cannot share a draft cycle")

    drafts: List[List[int]] = [[] for _ in batch.lanes]
    draft_tokens: List[List[mx.array]] = [[] for _ in batch.lanes]
    draft_logprobs: List[List[mx.array]] = [[] for _ in batch.lanes]
    draft_h = [lane.seed_h for lane in batch.lanes]
    draft_steps = [0] * n_lanes
    max_k = max(k_vector)
    greedy_cycle = all(lane.sampling_temp <= 0 for lane in batch.lanes)
    if max_k > 0:
        start_cycle = getattr(model, "mtp_start_cycle", None)
        if start_cycle is not None:
            # Shared QSA deliberately omits raw-index keys after the first
            # draft step, which is safe only when the whole batch rewinds the
            # same drafted span. A lane at its terminal budget makes the tail
            # rewind ragged even though configured depths are homogeneous;
            # keep that cycle exact by recomputing QSA normally.
            share_qsa_this_cycle = bool(
                active_share_modes
                and next(iter(active_share_modes))
                and len(set(k_vector)) == 1
            )
            start_cycle(
                batch.caches.draft,
                share_qsa_this_cycle,
            )
        try:
            first_lengths = [
                len(lane.pending_ts) + 1 if k > 0 else 0
                for lane, k in zip(batch.lanes, k_vector)
            ]
            width = max(first_lengths)
            hidden_rows = []
            token_rows = []
            for lane, valid in zip(batch.lanes, first_lengths):
                if valid:
                    if lane.pending_hs is None:
                        if lane.pending_ts:
                            raise RuntimeError("pending token list has no hidden tensor")
                        hs = lane.seed_h
                    else:
                        if lane.pending_hs.shape[1] != len(lane.pending_ts):
                            raise RuntimeError("pending hidden/token lengths disagree")
                        hs = mx.concatenate([lane.pending_hs, lane.seed_h], axis=1)
                    ts = mx.array([lane.pending_ts + [lane.cur]], mx.uint32)
                else:
                    hs = mx.zeros_like(lane.seed_h)
                    ts = mx.zeros((1, 1), mx.uint32)
                pad = width - valid if valid else width - 1
                hidden_rows.append(mx.pad(hs, [(0, 0), (0, pad), (0, 0)]))
                token_rows.append(mx.pad(ts, [(0, 0), (0, pad)]))
            right_padding = [width - valid for valid in first_lengths]
            _prepare_self_mtp_cache_group(
                batch.caches.draft, first_lengths, right_padding
            )
            try:
                d_logits, post = model.mtp_step(
                    mx.concatenate(hidden_rows),
                    mx.concatenate(token_rows),
                    batch.caches.draft,
                )
            finally:
                _finalize_self_mtp_cache_group(batch.caches.draft)
            for row, (lane, k, valid) in enumerate(
                zip(batch.lanes, k_vector, first_lengths)
            ):
                if k == 0:
                    continue
                pos = valid - 1
                draft_h[row] = post[row : row + 1, pos : pos + 1, :]
                lp = _lane_mtp_logprobs(lane, d_logits[row, pos])
                if greedy_cycle:
                    token = mx.argmax(lp).astype(mx.uint32)
                else:
                    hosted_token = _sample_from_logprobs(
                        lp, lane.sampling_temp, rng=lane.rng
                    )
                    token = mx.array(hosted_token, mx.uint32)
                    drafts[row].append(hosted_token)
                draft_tokens[row].append(token)
                draft_logprobs[row].append(lp)
                draft_steps[row] += 1
                lane.pending_hs = None
                lane.pending_ts = []

            if greedy_cycle:
                mx.async_eval(
                    *(row[-1] for row in draft_tokens if row),
                    *(draft_h[row] for row, k in enumerate(k_vector) if k),
                )

            for depth in range(1, max_k):
                lengths = [1 if depth < k else 0 for k in k_vector]
                right_padding = [1 - length for length in lengths]
                hidden = mx.concatenate(draft_h)
                tokens = mx.concatenate(
                    [
                        mx.reshape(draft_tokens[row][-1], (1, 1))
                        if lengths[row]
                        else mx.zeros((1, 1), mx.uint32)
                        for row in range(n_lanes)
                    ],
                    axis=0,
                )
                _prepare_self_mtp_cache_group(
                    batch.caches.draft, lengths, right_padding
                )
                try:
                    d_logits, post = model.mtp_step(
                        hidden, tokens, batch.caches.draft
                    )
                finally:
                    _finalize_self_mtp_cache_group(batch.caches.draft)
                for row, (lane, active) in enumerate(zip(batch.lanes, lengths)):
                    if not active:
                        continue
                    draft_h[row] = post[row : row + 1, -1:, :]
                    lp = _lane_mtp_logprobs(lane, d_logits[row, -1])
                    if greedy_cycle:
                        token = mx.argmax(lp).astype(mx.uint32)
                    else:
                        hosted_token = _sample_from_logprobs(
                            lp, lane.sampling_temp, rng=lane.rng
                        )
                        token = mx.array(hosted_token, mx.uint32)
                        drafts[row].append(hosted_token)
                    draft_tokens[row].append(token)
                    draft_logprobs[row].append(lp)
                    draft_steps[row] += 1
                if greedy_cycle:
                    mx.async_eval(
                        *(
                            draft_tokens[row][-1]
                            for row, active in enumerate(lengths)
                            if active
                        ),
                        *(
                            draft_h[row]
                            for row, active in enumerate(lengths)
                            if active
                        ),
                    )
        finally:
            if any(draft_steps):
                trim_ragged_prompt_cache(
                    batch.caches.draft, draft_steps, validate=False
                )
            end_cycle = getattr(model, "mtp_end_cycle", None)
            if end_cycle is not None:
                end_cycle(batch.caches.draft)
        if tuple(draft_steps) != k_vector:
            raise RuntimeError(
                f"draft head advanced {tuple(draft_steps)}, expected {k_vector}"
            )

    valid_lengths = tuple(k + 1 for k in k_vector)
    width = max(valid_lengths)
    right_padding = tuple(width - valid for valid in valid_lengths)
    verify_rows = []
    for lane, row in zip(batch.lanes, draft_tokens):
        verify_rows.append(
            mx.concatenate(
                [mx.array([[lane.cur]], mx.uint32)]
                + [mx.reshape(token, (1, 1)) for token in row]
                + [mx.zeros((1, width - len(row) - 1), mx.uint32)],
                axis=1,
            )
        )
    verify_ids = mx.concatenate(verify_rows, axis=0)
    _prepare_self_mtp_cache_group(
        batch.caches.target, valid_lengths, right_padding
    )
    try:
        vlogit_hidden, batched_hidden = _mtp_backbone(
            model, verify_ids, batch.caches.target
        )
        batched_logits = model.logits(vlogit_hidden)
    finally:
        _finalize_self_mtp_cache_group(batch.caches.target)

    old_curs = tuple(lane.cur for lane in batch.lanes)
    old_seed_hs = tuple(lane.seed_h for lane in batch.lanes)
    lane_logprobs: List[mx.array] = []
    lane_hiddens: List[mx.array] = []
    for row, (lane, k, valid) in enumerate(
        zip(batch.lanes, k_vector, valid_lengths)
    ):
        if lane.logits_processors:
            processed = []
            for pos in range(valid):
                processor_tokens = mx.concatenate(
                    [lane.token_prefix, mx.array([lane.cur], mx.uint32)]
                    + [
                        mx.reshape(token, (1,))
                        for token in draft_tokens[row][:pos]
                    ]
                )
                processed.append(
                    _apply_logits_processors(
                        lane.logits_processors,
                        processor_tokens,
                        batched_logits[row, pos],
                    )
                )
            logits = mx.stack(processed)
        else:
            logits = batched_logits[row, :valid]
        logprobs = _lane_mtp_logprobs(lane, logits)
        hidden = batched_hidden[row : row + 1, :valid, :]
        lane_logprobs.append(logprobs)
        lane_hiddens.append(hidden)

    greedy_targets = None
    if greedy_cycle:
        target_rows = []
        drafted_rows = []
        for row, (k, valid) in enumerate(zip(k_vector, valid_lengths)):
            target = mx.argmax(lane_logprobs[row], axis=-1).astype(mx.uint32)
            target_rows.append(mx.pad(target, [(0, width - valid)]))
            drafted = (
                mx.stack(draft_tokens[row])
                if k
                else mx.zeros((0,), mx.uint32)
            )
            drafted_rows.append(mx.pad(drafted, [(0, width - k)]))
        accept_payload = mx.stack(
            [mx.stack(target_rows), mx.stack(drafted_rows)]
        )
        record_verify_sync("hybrid.greedy.accept_boundary")
        mx.eval(accept_payload)
        greedy_targets, hosted_drafts = accept_payload.tolist()
        drafts = [row[:k] for row, k in zip(hosted_drafts, k_vector)]

    accepted: List[int] = []
    bonuses: List[int] = []
    output_rows: List[Tuple[MTPToken, ...]] = []
    for row, (lane, k) in enumerate(zip(batch.lanes, k_vector)):
        logprobs = lane_logprobs[row]

        if k == 0:
            if greedy_cycle:
                n_accept = 0
                bonus = int(greedy_targets[row][0])
            else:
                n_accept = 0
                bonus = _sample_from_logprobs(
                    logprobs[0], lane.sampling_temp, rng=lane.rng
                )
        elif lane.sampling_temp > 0:
            if lane.logprob_transform is not None:
                n_accept, bonus = _batched_residual_verify(
                    logprobs,
                    draft_logprobs[row],
                    drafts[row],
                    lane.sampling_temp,
                    rng=lane.rng,
                )
            elif lane.accept_rule == "block":
                n_accept, bonus = _block_verify(
                    logprobs,
                    draft_logprobs[row],
                    drafts[row],
                    lane.sampling_temp,
                    rng=lane.rng,
                )
            elif lane.accept_rule == "exact":
                sampled = mx.random.categorical(logprobs, key=draw_key(lane.rng))
                record_verify_sync("hybrid.exact.sampled_eval")
                mx.eval(sampled)
                record_verify_sync("hybrid.exact.sampled_tolist")
                sampled = sampled.tolist()
                n_accept = 0
                while n_accept < k and sampled[n_accept] == drafts[row][n_accept]:
                    n_accept += 1
                bonus = int(sampled[n_accept])
            else:
                n_accept = 0
                while n_accept < k and _accept_sampled_draft(
                    logprobs[n_accept],
                    draft_logprobs[row][n_accept],
                    drafts[row][n_accept],
                    rng=lane.rng,
                ):
                    n_accept += 1
                if n_accept < k:
                    bonus = _residual_sample(
                        logprobs[n_accept],
                        draft_logprobs[row][n_accept],
                        lane.sampling_temp,
                        rng=lane.rng,
                    )
                else:
                    bonus = _sample_from_logprobs(
                        logprobs[n_accept], lane.sampling_temp, rng=lane.rng
                    )
        else:
            if greedy_cycle:
                targets = greedy_targets[row]
            else:
                record_verify_sync("hybrid.greedy.targets_tolist")
                targets = mx.argmax(logprobs, axis=-1).tolist()
            n_accept = 0
            while n_accept < k and targets[n_accept] == drafts[row][n_accept]:
                n_accept += 1
            bonus = int(targets[n_accept])

        accepted.append(n_accept)
        bonuses.append(bonus)
        output_rows.append(
            tuple(
                [
                    MTPToken(drafts[row][pos], logprobs[pos], True)
                    for pos in range(n_accept)
                ]
                + [MTPToken(bonus, logprobs[n_accept], False)]
            )
        )

    target_drops = tuple(k - a for k, a in zip(k_vector, accepted))
    trim_ragged_prompt_cache(
        batch.caches.target, target_drops, validate=False
    )
    proposal = SelfMTPCycleResult(
        membership_epoch=batch.membership_epoch,
        lane_uids=lane_uids,
        draft_depths=k_vector,
        accepted_lengths=tuple(accepted),
        target_drops=target_drops,
        head_drops=k_vector,
        outputs=tuple(output_rows),
        _old_curs=old_curs,
        _old_seed_hs=old_seed_hs,
        _drafts=tuple(tuple(row) for row in drafts),
        _vhidden=tuple(lane_hiddens),
        _logprobs=tuple(lane_logprobs),
        _bonuses=tuple(bonuses),
    )
    batch.proposal_open = True
    batch._open_proposal = proposal
    return proposal


def propose_batched_self_mtp(
    model: nn.Module,
    batch: BatchedSelfMTPState,
) -> SelfMTPCycleResult:
    """Open a proposal, poisoning state when rollback cannot be proved."""
    _require_healthy_self_mtp_batch(batch)
    if batch.proposal_open:
        raise RuntimeError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot propose on an empty self-MTP batch")
    try:
        return _propose_batched_self_mtp_impl(model, batch)
    except BaseException as error:
        batch.proposal_open = False
        batch._open_proposal = None
        _poison_self_mtp_batch(batch, f"proposal rollback unproved: {error}")
        raise


def commit_batched_self_mtp(
    batch: BatchedSelfMTPState,
    proposal: SelfMTPCycleResult,
    *,
    emitted_counts: Sequence[int],
    terminal: Sequence[bool],
) -> None:
    """Commit exactly the delivered prefix of one open proposal."""
    _require_healthy_self_mtp_batch(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise RuntimeError("commit requires the currently open self-MTP proposal")
    if proposal.membership_epoch != batch.membership_epoch:
        raise RuntimeError("self-MTP membership changed during an open proposal")
    if proposal.lane_uids != tuple(lane.uid for lane in batch.lanes):
        raise RuntimeError("self-MTP lane order changed during an open proposal")
    n_lanes = len(batch.lanes)
    if len(emitted_counts) != n_lanes or len(terminal) != n_lanes:
        raise ValueError("commit vectors must have one entry per lane")

    emitted = tuple(int(value) for value in emitted_counts)
    terminal = tuple(bool(value) for value in terminal)
    delivery_drops = []
    for row, (count, is_terminal, outputs, accepted) in enumerate(
        zip(emitted, terminal, proposal.outputs, proposal.accepted_lengths)
    ):
        if count < 0 or count > len(outputs):
            raise ValueError(f"lane {row} emitted_count {count} is out of range")
        if not is_terminal and count != len(outputs):
            raise ValueError("a nonterminal lane must consume its entire proposal")
        if is_terminal and count < len(outputs) and count > accepted:
            raise ValueError("a terminal prefix cannot skip part of the bonus token")
        # A terminal short prefix also drops the final emitted token, which
        # becomes ``cur`` and must stay outside the target cache.
        delivery_drops.append(
            accepted - count + 1 if is_terminal and count <= accepted else 0
        )

    try:
        if any(delivery_drops):
            trim_ragged_prompt_cache(
                batch.caches.target, delivery_drops, validate=False
            )

        for row, lane in enumerate(batch.lanes):
            accepted = proposal.accepted_lengths[row]
            count = emitted[row]
            old_cur = proposal._old_curs[row]
            old_seed_h = proposal._old_seed_hs[row]
            drafts = list(proposal._drafts[row])
            hidden = proposal._vhidden[row]
            consumed_accepted = min(count, accepted)

            if terminal[row] and count <= accepted:
                # The cycle ends at the last emitted token: it becomes ``cur``
                # (outside the target cache), ``seed_h`` is the hidden that
                # predicts it, and pending pairs stop one token before it.
                if count > 0:
                    lane.pending_hs = mx.concatenate(
                        [old_seed_h, hidden[:, : count - 1, :]], axis=1
                    )
                    lane.pending_ts = [old_cur] + drafts[: count - 1]
                    lane.seed_h = hidden[:, count - 1 : count, :]
                    lane.cur = proposal.outputs[row][count - 1].token
                    lane.token_prefix = mx.concatenate(
                        [
                            lane.token_prefix,
                            mx.array(lane.pending_ts, mx.uint32),
                        ]
                    )
                # count == 0 keeps the pre-cycle state: the delivery trim
                # already rolled the target cache back to before ``cur``.
            else:
                new_hs = mx.concatenate(
                    [old_seed_h, hidden[:, :accepted, :]], axis=1
                )
                new_ts = [old_cur] + drafts[:accepted]
                if lane.pending_ts:
                    # A k == 0 lane skips the draft flush, so pairs retained
                    # from the prior cycle are still owed to the draft cache.
                    new_hs = mx.concatenate([lane.pending_hs, new_hs], axis=1)
                    new_ts = lane.pending_ts + new_ts
                lane.pending_hs = new_hs
                lane.pending_ts = new_ts
                lane.seed_h = hidden[:, accepted : accepted + 1, :]
                lane.cur = proposal._bonuses[row]
                lane.token_prefix = mx.concatenate(
                    [
                        lane.token_prefix,
                        mx.array([old_cur] + drafts[:accepted], mx.uint32),
                    ]
                )

            lane.ntoks += count
            lane.stats.cycles += 1
            lane.stats.draft_cycles += 1
            lane.stats.draft_proposed += proposal.draft_depths[row]
            lane.stats.draft_accepted += consumed_accepted
            if count > accepted:
                lane.stats.bonus_tokens += 1
    except BaseException as error:
        batch.proposal_open = False
        batch._open_proposal = None
        _poison_self_mtp_batch(batch, f"commit rollback unproved: {error}")
        raise

    batch.proposal_open = False
    batch._open_proposal = None


def abort_batched_self_mtp(
    batch: BatchedSelfMTPState,
    proposal: SelfMTPCycleResult,
    *,
    cause: Optional[BaseException] = None,
) -> None:
    """Close an interrupted proposal and permanently isolate its state.

    The integrated MLX backend does not yet expose a model-neutral exact-abort
    ABI. Once proposal compute has advanced recurrent/KV state or consumed a
    per-lane RNG key, replay safety cannot be proved. Match Rapid-MLX's abort
    contract by failing closed: close the transaction, poison the cohort, and
    require the scheduler to discard it rather than detach, retry, or reuse it.
    """
    _require_healthy_self_mtp_batch(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise RuntimeError("abort requires the currently open self-MTP proposal")
    batch.proposal_open = False
    batch._open_proposal = None
    detail = "explicit proposal abort"
    if cause is not None:
        detail = f"proposal delivery aborted: {cause}"
    _poison_self_mtp_batch(batch, detail)


def detach_self_mtp_lanes(
    model: nn.Module,
    batch: BatchedSelfMTPState,
    indices: Sequence[int],
) -> Tuple[BatchedSelfMTPState, List[DetachedSelfMTPLane]]:
    """Extract canonical rows before filtering the old batch membership."""
    _require_healthy_self_mtp_batch(batch)
    if batch.proposal_open:
        raise RuntimeError("cannot detach self-MTP lanes while a proposal is open")
    requested = [int(index) for index in indices]
    if len(set(requested)) != len(requested):
        raise ValueError("detach indices must be unique")
    if any(index < 0 or index >= len(batch.lanes) for index in requested):
        raise IndexError("detach index is outside the self-MTP batch")
    if not requested:
        return batch, []

    leaving = set(requested)
    keep = [index for index in range(len(batch.lanes)) if index not in leaving]
    try:
        _stop_all_speculation(batch.caches.target)
    except BaseException as error:
        _poison_self_mtp_batch(batch, f"rollback stop failed: {error}")
        raise
    try:
        detached: List[DetachedSelfMTPLane] = []
        for index in requested:
            lane = copy.copy(batch.lanes[index])
            caches = _extract_self_mtp_cache_pair(batch.caches, [index])
            if lane.pending_hs is not None and lane.pending_ts:
                model.mtp_step(
                    lane.pending_hs,
                    mx.array([lane.pending_ts], mx.uint32),
                    caches.draft,
                )
            lane.pending_hs = None
            lane.pending_ts = []
            item = DetachedSelfMTPLane(lane=lane, caches=caches)
            _eval_self_mtp_lane_state(item)
            _validate_detached_self_mtp(item)
            detached.append(item)

        replacement = _extract_self_mtp_cache_pair(
            batch.caches,
            keep,
            batched=True,
        )
        if keep:
            _start_speculation_or_cleanup(
                replacement.target,
                replacement.target,
                "batched self-MTP requires ragged-trimmable target caches",
            )
    except BaseException as error:
        _restart_live_self_mtp_or_poison(batch, error)
        raise

    batch.lanes = [batch.lanes[index] for index in keep]
    batch.caches = replacement
    batch.membership_epoch += 1
    return batch, detached



def _discard_hedge(model, mtp_cache, hedge) -> None:
    """Rewind a queued hedge chain's k speculative head entries.

    Host offsets only: the chain's GPU work completes and is ignored. Used
    when a hit is not consumed (budget change, plain step, generator exit).
    """
    trim_prompt_cache(mtp_cache, hedge[3])
    end_cycle = getattr(model, "mtp_end_cycle", None)
    if end_cycle is not None:
        end_cycle(mtp_cache)
    _lv.bump("hedge_discarded")


def _mtp_draft_verify_loop_impl(
    model,
    cache,
    cur,
    seed_h,
    ntoks,
    max_tokens,
    num_draft,
    stats,
    sampling_temp: float = 0.0,
    accept_rule: str = "residual",
    logprob_transform=None,
    mtp_cache=None,
    rate_gate: bool = False,
    speculation_router: Optional[RoutedSpeculationPolicy] = None,
    logits_processors=None,
    token_prefix=None,
    share_qsa_indices: bool = False,
    rng: Optional[LaneRNG] = None,
    mtp_state_tracker: Optional[dict] = None,
    lever_state: Optional[dict] = None,
):
    """Shared MTP tail: the head drafts, the trunk verifies, GDN rollback trims
    rejects. Assumes cache speculation is already ON; ``cur`` is the last
    committed-but-uncached token and ``seed_h`` the trunk hidden that predicted
    it. Yields (token, logprobs, from_draft). Does NOT start/stop speculation.

    When ``mtp_cache`` is given it is a PERSISTENT context cache covering the
    committed pairs (hidden_i, token_{i+1}), so the head drafts with full
    context at real RoPE positions (its trained regime). After each verify the
    k draft entries are rewound; the newly committed span (with TRUNK hiddens)
    is carried as ``pending`` pairs and teacher-forced as a prefix of the next
    cycle's first draft call — one MTP forward per cycle, no separate
    catch-up pass. ``None`` keeps the legacy fresh-cache-per-cycle behavior.

    ``rate_gate=True`` adds a one-shot empirical break-even check: after
    ``_RATE_GATE_WARMUP_CYCLES`` measured draft/verify cycles, it decodes
    ``_RATE_GATE_PROBE_TOKENS`` tokens plainly INLINE (the probe tokens are
    delivered output, nothing is wasted), compares wall-clock ms per delivered
    token, and decides ONCE: keep speculating only if the spec rate beats the
    plain rate by ``_RATE_GATE_MARGIN``; otherwise de-latch to plain decode for
    the rest of the generation. Measured, not modeled — the verify-cost
    break-even is target- AND context-dependent (see
    lessons/persistent-mtp-context-cache) — and one-way, so no mid-stream
    thrashing (the adaptive-PLD latch philosophy; per the D-Cut lesson,
    continuous adaptivity loses to simple decisions).

    ``rng`` is this request's ``LaneRNG``. Every stochastic operation of the
    loop — draft, plain-step and bonus sampling, the acceptance uniforms, the
    block and exact rule draws, and residual correction — takes a subkey from
    it, so a lane's draws depend on its own seed and history alone, never on
    co-scheduled traffic (the batch-composition P1 contract). ``None`` keeps
    the global ``mx.random`` stream, byte-identically. A rejected draft rewinds
    tokens and caches but NEVER the key: the key advances once per draw made,
    so a correction draw is independent of the proposal it replaces. Greedy
    (``sampling_temp == 0``) takes no draw and consumes no key."""
    persistent = mtp_cache is not None

    # ---- round levers (2026-09-08, default off; see round_levers.py) ------
    # (a) PLE verify-row prefetch: needs a file-backed PLE table on the model.
    # (c) hedge: greedy, persistent head only (checked per round below).
    lever_state = lever_state if lever_state is not None else {}
    lever_ple = _lv.PLE_VERIFY_PREFETCH and hasattr(model, "ple_prefetch_verify")
    lever_hedge = _lv.HEDGE_DRAFT and persistent
    end_cycle_hook = getattr(model, "mtp_end_cycle", None)
    recent_tokens: deque = deque(maxlen=8)  # host mirror of the committed tail
    prebuilt = None  # (c): (tokens, logprobs, h, k, appended) for the next round
    if persistent:
        lever_state["mtp_cache"] = mtp_cache

    def _logprobs(logits):
        # ``logprob_transform`` is the shared draft/target transformed
        # distribution; ``None`` is the incumbent temperature-only path.
        if logprob_transform is not None:
            return logprob_transform(logits)
        return _temperature_logprobs(logits, sampling_temp)

    pending_hs = None  # committed (hidden, token) pairs not yet in mtp_cache
    pending_ts: List[int] = []
    if mtp_state_tracker is not None:
        mtp_state_tracker.update(
            cache=cache,
            mtp_cache=mtp_cache,
            pending_hs=pending_hs,
            pending_ts=pending_ts,
            seed_h=seed_h,
            # The lane object is carried by reference, so its key travels with
            # the request through every rollback and rewind path.
            rng=rng,
        )
    gated_off = False
    gate_cycles = 0
    spec_secs = 0.0  # wall-clock over measured spec cycles
    spec_toks = 0  # tokens those cycles delivered
    router_plain = False
    # Tokens committed before ``cur``. Processor calls may temporarily walk
    # speculative prefixes; stateful processors are required to support the
    # same rewind-on-shorter-history contract used by speculative_generate_step.
    token_prefix = (
        token_prefix.astype(mx.uint32)
        if token_prefix is not None
        else mx.array([], mx.uint32)
    )

    if lever_ple and token_prefix.size:
        recent_tokens.extend(int(t) for t in token_prefix[-8:].tolist())

    def _discard_prebuilt():
        nonlocal prebuilt
        if prebuilt is None:
            return
        _discard_hedge(model, mtp_cache, prebuilt)
        prebuilt = None
        lever_state["prebuilt"] = None

    def _build_hedge(k, cur, seed_h, vhidden, targets, draft_tokens):
        # Lever (c): assume every draft lands. Rewind this round's k head
        # entries now (host offsets only) and queue the NEXT round's first
        # draft call -- the pending pairs (seed_h, cur), (h_1, d_1)..(h_k, d_k)
        # plus (h_{k+1}, bonus) -- and its k-1 chained steps behind the verify.
        # A miss rewinds ``appended`` entries; a hit hands the chain over.
        trim_prompt_cache(mtp_cache, k)
        if end_cycle_hook is not None:
            end_cycle_hook(mtp_cache)
        if hasattr(model, "mtp_start_cycle"):
            model.mtp_start_cycle(mtp_cache, share_qsa_indices and k > 1)
        hs = mx.concatenate([seed_h, vhidden[:, : k + 1, :]], axis=1)
        ts = mx.concatenate(
            [mx.array([[cur]], mx.uint32)]
            + [mx.reshape(t, (1, 1)) for t in draft_tokens]
            + [mx.reshape(targets[k], (1, 1))],
            axis=1,
        )
        tokens, lps, hh = [], [], None
        for _ in range(k):
            d_logits, post = model.mtp_step(hs, ts, mtp_cache)
            hh = post[:, -1:, :]
            d_lp = _logprobs(d_logits[0, -1])
            t_i = mx.argmax(d_lp).astype(mx.uint32)
            tokens.append(t_i)
            lps.append(d_lp)
            mx.async_eval(t_i, hh)
            hs, ts = hh, mx.reshape(t_i, (1, 1))
        _lv.bump("hedge_built")
        return (tokens, lps, hh, k, (k + 2) + (k - 1))

    def _plain_step():
        # One width-1 trunk forward: commits `cur`, samples the next token.
        # Keeps the pending-pair protocol intact so persistent drafting can
        # resume seamlessly after a probe.
        nonlocal cur, seed_h, pending_hs, pending_ts, token_prefix
        _discard_prebuilt()
        with mx.stream(generation_stream):
            logit_h, h = _mtp_backbone(
                model, mx.array([[cur]], mx.uint32), cache
            )
            proc_tokens = mx.concatenate(
                [token_prefix, mx.array([cur], mx.uint32)]
            )
            logits = _apply_logits_processors(
                logits_processors, proc_tokens, model.logits(logit_h)[0, -1]
            )
            lp = _logprobs(logits)
            nxt = _sample_from_logprobs(lp, sampling_temp, rng=rng)
        if persistent and (not gated_off or mtp_state_tracker is not None):
            # Pairs only matter if drafting can resume; after a permanent
            # de-latch they would just accumulate unused memory.
            pending_hs = (
                seed_h if pending_hs is None
                else mx.concatenate([pending_hs, seed_h], axis=1)
            )
            pending_ts.append(cur)
        token_prefix = proc_tokens
        if lever_ple:
            recent_tokens.append(cur)
        seed_h, cur = h[:, -1:, :], nxt
        if mtp_state_tracker is not None:
            mtp_state_tracker.update(
                pending_hs=pending_hs,
                pending_ts=pending_ts,
                seed_h=seed_h,
            )
        stats.cycles += 1
        stats.plain_cycles += 1
        stats.plain_tokens += 1
        return nxt, lp

    while ntoks < max_tokens:
        if gated_off:
            tok_, lp = _plain_step()
            ntoks += 1
            yield tok_, lp, False
            continue
        routed_k = None
        if speculation_router is not None:
            decision = speculation_router.decide(
                max_draft=num_draft,
                remaining=max_tokens - ntoks,
            )
            routed_k = decision.num_draft
            stats.router_last_num_draft = routed_k
            stats.router_accept_prob = decision.accept_prob
            stats.router_reengagements = speculation_router.reengagements
            if routed_k == 0:
                if not router_plain:
                    _stop_all_speculation(cache)
                    router_plain = True
                tok_, lp = _plain_step()
                ntoks += 1
                stats.router_plain_cycles += 1
                yield tok_, lp, False
                continue
            if router_plain:
                _start_speculation_or_cleanup(
                    cache,
                    cache,
                    "routed MTP re-entry needs a trimmable prompt cache.",
                )
                router_plain = False
        if rate_gate and not stats.rate_gate_probed and gate_cycles >= _RATE_GATE_WARMUP_CYCLES:
            stats.rate_gate_probed = True
            # Honest plain reference: rollback recording (GDN/rotating-cache
            # speculation bookkeeping) is spec-only overhead, so switch it off
            # for the probe — and leave it off after a de-latch, which is also
            # what makes the plain fallback run at true baseline speed. All
            # tokens are committed at this point, so there is nothing to lose.
            _stop_all_speculation(cache)
            probe_t0 = time.perf_counter()
            n_probe = 0
            while n_probe < _RATE_GATE_PROBE_TOKENS and ntoks < max_tokens:
                tok_, lp = _plain_step()
                ntoks += 1
                n_probe += 1
                yield tok_, lp, False
            plain_rate = (time.perf_counter() - probe_t0) * 1000.0 / max(n_probe, 1)
            spec_rate = spec_secs * 1000.0 / max(spec_toks, 1)
            stats.rate_gate_spec_ms_per_tok = spec_rate
            stats.rate_gate_plain_ms_per_tok = plain_rate
            if spec_rate > plain_rate * (1.0 - _RATE_GATE_MARGIN):
                gated_off = True
                stats.rate_gate_delatched = True
            else:
                _start_speculation_or_cleanup(
                    cache,
                    cache,
                    "MTP rate-gate resume needs a trimmable prompt cache.",
                )
            continue
        cycle_t0 = time.perf_counter()
        _ht.start_round()  # host-stall attribution: open per-round accumulator
        ntoks_at_cycle_start = ntoks
        configured_k = num_draft if routed_k is None else routed_k
        k = draft_tokens_for_budget(configured_k, max_tokens - ntoks)
        if k == 0:
            tok_, lp = _plain_step()
            ntoks += 1
            yield tok_, lp, False
            continue
        stats.cycles += 1

        with verify_sync_round():
            # ---- draft k tokens with the MTP head (chained) ------------------
            if not persistent:
                mtp_cache = model.make_mtp_cache()
            greedy_device = not sampling_temp and not logits_processors
            # Lever (b): temperature-only drafts stay on device too -- the same
            # categorical draw ``_sample_from_logprobs`` makes, minus its
            # ``.item()`` -- so the chain has no per-draft host sync.
            device_sampled = (
                not greedy_device
                and _lv.DEVICE_SAMPLING
                and bool(sampling_temp and sampling_temp > 0)
                and not logits_processors
            )
            device_draft = greedy_device or device_sampled
            drafts: List[int] = []
            draft_tokens: List[mx.array] = []
            draft_logprobs: List[mx.array] = []
            h, tok = seed_h, mx.array([[cur]], mx.uint32)
            _ht_draft_t0 = _ht.tic() if _ht.ENABLED else 0.0  # mtp_draft span
            landed_drafts: List[int] = []  # lever (a): draft ids as they land
            if prebuilt is not None and prebuilt[3] == k and greedy_device:
                # Lever (c) hit: the chain was built and queued under the
                # previous verify and its cycle is still armed (no start_cycle).
                draft_tokens, draft_logprobs, h = prebuilt[0], prebuilt[1], prebuilt[2]
                prebuilt = None
                lever_state["prebuilt"] = None
                _lv.bump("hedge_consumed")
            else:
                _discard_prebuilt()
                if hasattr(model, "mtp_start_cycle"):
                    model.mtp_start_cycle(mtp_cache, share_qsa_indices and k > 1)
                if lever_ple:
                    # Slab position 0 hashes host-known tokens only.
                    model.ple_prefetch_verify(list(recent_tokens), [cur])
                with mx.stream(generation_stream):
                    for i in range(k):
                        if i == 0 and pending_hs is not None:
                            hs = mx.concatenate([pending_hs, h], axis=1)
                            ts = mx.array([pending_ts + [cur]], mx.uint32)
                        else:
                            hs, ts = h, tok
                        d_logits, post = model.mtp_step(hs, ts, mtp_cache)
                        h = post[:, -1:, :]
                        d_lp = _logprobs(d_logits[0, -1])
                        if greedy_device:
                            draft_token = mx.argmax(d_lp).astype(mx.uint32)
                            draft_tokens.append(draft_token)
                            tok = mx.reshape(draft_token, (1, 1))
                            mx.async_eval(draft_token, h)
                        elif device_sampled:
                            draft_token = mx.random.categorical(
                                d_lp, key=draw_key(rng)
                            ).astype(mx.uint32)
                            draft_tokens.append(draft_token)
                            tok = mx.reshape(draft_token, (1, 1))
                            mx.async_eval(draft_token, h)
                            _lv.bump("device_sampled_drafts")
                        else:
                            draft = _sample_from_logprobs(
                                d_lp, sampling_temp, rng=rng
                            )
                            drafts.append(draft)
                            tok = mx.array([[draft]], mx.uint32)
                        draft_logprobs.append(d_lp)
                        if lever_ple and device_draft and i >= 1:
                            # CLOSED 2026-09-08 (measured -1.5%..+2.1%, sign
                            # flipping by cell: the lever's own ceiling is
                            # ~1.0-1.2 ms of a ~46 ms round, smaller than this
                            # lane's run-to-run spread). The block is kept for
                            # the record but its timing counter was removed: it
                            # wrapped an ``.item()``, i.e. a device sync on the
                            # hot path, measuring the thing it perturbs.
                            landed_drafts.append(int(draft_tokens[i - 1].item()))
                            model.ple_prefetch_verify(
                                list(recent_tokens), [cur] + landed_drafts
                            )
            if _ht.ENABLED:
                _ht.toc("mtp_draft", _ht_draft_t0)

            # ---- verify: trunk over [cur, drafts...] in one forward ----------
            hedge = None
            verify_in = (
                mx.concatenate(
                    [mx.array([[cur]], mx.uint32)]
                    + [mx.reshape(token, (1, 1)) for token in draft_tokens],
                    axis=1,
                )
                if device_draft
                else mx.array([[cur] + drafts], mx.uint32)
            )
            with mx.stream(generation_stream):
                vlogit_hidden, vhidden = _mtp_backbone(model, verify_in, cache)
                vlogits = model.logits(vlogit_hidden)
                if device_draft:
                    logprobs = _logprobs(vlogits[0])
                else:
                    processed_logits = []
                    for i in range(k + 1):
                        proc_tokens = mx.concatenate(
                            [
                                token_prefix,
                                mx.array([cur] + drafts[:i], mx.uint32),
                            ]
                        )
                        processed_logits.append(
                            _apply_logits_processors(
                                logits_processors, proc_tokens, vlogits[0, i]
                            )
                        )
                    logprobs = _logprobs(mx.stack(processed_logits))
                targets = mx.argmax(logprobs, axis=-1).astype(mx.uint32)
                if (
                    lever_hedge
                    and greedy_device
                    and speculation_router is None
                    and (not rate_gate or stats.rate_gate_probed)
                ):
                    # Lever (c): only when the next round would draft the same
                    # width (a budget-clamped tail would mis-size the chain).
                    k_next = draft_tokens_for_budget(
                        configured_k, max_tokens - (ntoks + k + 1)
                    )
                    if k_next == k:
                        hedge = _build_hedge(
                            k, cur, seed_h, vhidden, targets, draft_tokens
                        )
                    else:
                        _lv.bump("hedge_skipped")

            # accept span: verify-boundary host sync + longest-accepted-prefix
            # scan + commit/rollback trims. The first mx.eval after the async
            # draft dispatch also drains the (GPU-bound) verify forward, so this
            # span is an UPPER BOUND on accept host cost; subtract the forward
            # GPU floor from the launchbound probe for the pure host share.
            _ht_accept_t0 = _ht.tic() if _ht.ENABLED else 0.0
            if greedy_device:
                padded_drafts = mx.concatenate(
                    [mx.stack(draft_tokens), mx.zeros((1,), mx.uint32)]
                )
                accept_payload = mx.stack([targets, padded_drafts])
                record_verify_sync("hybrid.greedy.accept_boundary")
                mx.eval(accept_payload, vhidden)
                if _ht.ENABLED:
                    _ht.toc("verify_wait", _ht_accept_t0)  # GPU verify drain
                    _ht_accept_t0 = _ht.tic()  # accept = pure host from here
                targets, hosted_drafts = accept_payload.tolist()
                drafts = hosted_drafts[:k]
            elif device_sampled:
                sampled_payload = mx.stack(draft_tokens)
                record_verify_sync("hybrid.sampled.accept_boundary")
                mx.eval(sampled_payload, vhidden)
                if _ht.ENABLED:
                    _ht.toc("verify_wait", _ht_accept_t0)  # GPU verify drain
                    _ht_accept_t0 = _ht.tic()  # accept = pure host from here
                drafts = [int(t) for t in sampled_payload.tolist()]
            else:
                mx.eval(targets, vhidden)
                if _ht.ENABLED:
                    _ht.toc("verify_wait", _ht_accept_t0)  # GPU verify drain
                    _ht_accept_t0 = _ht.tic()  # accept = pure host from here

            n_accept = 0
            if sampling_temp and sampling_temp > 0:
                if logprob_transform is not None:
                    # Transformed distributions: batched residual acceptance
                    # (one sync for the whole scan). Kept off the incumbent
                    # paths so their sync pattern and RNG stream are untouched.
                    n_accept, bonus = _batched_residual_verify(
                        logprobs, draft_logprobs, drafts, sampling_temp, rng=rng
                    )
                elif accept_rule == "block":
                    n_accept, bonus = _block_verify(
                        logprobs, draft_logprobs, drafts, sampling_temp, rng=rng
                    )
                elif accept_rule == "exact":
                    # Upstream external-draft semantics: sample the target's own
                    # token at every position, accept while it equals the draft's
                    # sample; the first mismatch commits the target sample.
                    sampled = mx.random.categorical(logprobs, key=draw_key(rng))
                    mx.eval(sampled)
                    sampled = sampled.tolist()
                    while n_accept < k and sampled[n_accept] == drafts[n_accept]:
                        n_accept += 1
                    bonus = int(sampled[n_accept])
                else:  # "residual" — Leviathan/SpecDec rejection sampling
                    while n_accept < k and _accept_sampled_draft(
                        logprobs[n_accept],
                        draft_logprobs[n_accept],
                        drafts[n_accept],
                        rng=rng,
                    ):
                        n_accept += 1
                    if n_accept < k:
                        bonus = _residual_sample(
                            logprobs[n_accept],
                            draft_logprobs[n_accept],
                            sampling_temp,
                            rng=rng,
                        )
                    else:
                        bonus = _sample_from_logprobs(
                            logprobs[n_accept], sampling_temp, rng=rng
                        )
            else:
                if not greedy_device:
                    record_verify_sync("hybrid.greedy.targets_tolist")
                    targets = targets.tolist()
                while n_accept < k and targets[n_accept] == drafts[n_accept]:
                    n_accept += 1
                bonus = int(targets[n_accept])

        # Trunk cache advanced by k+1 (cur + k drafts); keep cur + n_accept.
        trim_prompt_cache(cache, k - n_accept)
        if persistent and hedge is not None and n_accept == k:
            # Lever (c) hit: the head already holds the committed pairs and
            # the next round's k speculative entries; nothing is pending and
            # the queued chain is handed to the next round as ``prebuilt``.
            prebuilt = hedge
            lever_state["prebuilt"] = hedge
            pending_hs, pending_ts = None, []
            _lv.bump("hedge_hit")
        elif persistent:
            if hedge is not None:
                # Lever (c) miss: drop the optimistic first call and its
                # chained steps; this round's k entries were rewound already.
                trim_prompt_cache(mtp_cache, hedge[4])
                _lv.bump("hedge_miss")
            else:
                # Rewind the k speculative entries: (h_p, cur) plus the k-1
                # chained pairs built from MTP (not trunk) hiddens. The committed
                # span — (h_p, cur), (h_{p+1}, d_1) .. (h_{p+n_accept}, d_na) with
                # TRUNK hiddens — is carried as pending pairs and re-fed as the
                # prefix of the next cycle's first draft call. The bonus token
                # stays out: it becomes the next cycle's cur.
                trim_prompt_cache(mtp_cache, k)
            # Pair the cycle's arming with its exit, AFTER the rewind: called
            # before it, the hook reports the drafted span the rewind is about
            # to remove. The loop owns this exit -- a head cache whose own
            # rewind does not release the cycle would otherwise carry a stale
            # shared index set, and a ledger that no longer spans the cursor,
            # into the next forward or into a captured sidecar.
            end_cycle = getattr(model, "mtp_end_cycle", None)
            if end_cycle is not None:
                end_cycle(mtp_cache)
            if n_accept > 0:
                pending_hs = mx.concatenate(
                    [seed_h, vhidden[:, :n_accept, :]], axis=1
                )
            else:
                pending_hs = seed_h
            pending_ts = [cur] + drafts[:n_accept]
        if lever_ple:
            recent_tokens.extend([cur] + drafts[:n_accept])
        seed_h = vhidden[:, n_accept : n_accept + 1, :]  # hidden that predicted bonus
        if mtp_state_tracker is not None:
            mtp_state_tracker.update(
                pending_hs=pending_hs,
                pending_ts=pending_ts,
                seed_h=seed_h,
            )
        if _ht.ENABLED:
            _ht.toc("accept", _ht_accept_t0)
            _ht.end_round()  # flush this round's buckets before the yields
        stats.draft_proposed += k
        stats.draft_cycles += 1
        if speculation_router is not None:
            speculation_router.observe(k, n_accept)
            stats.router_accept_prob = speculation_router.accept_prob

        # Delivered-token telemetry updates exactly at each yield boundary so
        # an early close (e.g. EOS) never overstates accepted/bonus counts.
        for i in range(n_accept):
            ntoks += 1
            stats.draft_accepted += 1
            yield drafts[i], logprobs[i], True
            if ntoks == max_tokens:
                break
        if ntoks < max_tokens:
            ntoks += 1
            stats.bonus_tokens += 1
            yield bonus, logprobs[n_accept], False
        token_prefix = mx.concatenate(
            [token_prefix, mx.array([cur] + drafts[:n_accept], mx.uint32)]
        )
        cur = bonus
        if gate_cycles > 0:
            # The first cycle absorbs one-time costs that are not speculation
            # (kernel warm-up; on a prompt-cache hit, the reused cache's restore
            # on first use) — keep it out of the measured rate, as the
            # prompt-lookup gate does (2026-09-05).
            spec_secs += time.perf_counter() - cycle_t0
            spec_toks += ntoks - ntoks_at_cycle_start
        gate_cycles += 1


def _mtp_draft_verify_loop(*args, mtp_state_out=None, **kwargs):
    """Run the MTP tail and optionally materialize an exact APC sidecar.

    The implementation carries committed pairs lazily for performance.  This
    wrapper owns generator finalization, flushes those pairs once, evaluates
    the resulting MLX state, and exposes it only when target and draft offsets
    are structurally exact.
    """
    tracker = {} if mtp_state_out is not None else None
    lever_state: dict = {}
    try:
        yield from _mtp_draft_verify_loop_impl(
            *args, mtp_state_tracker=tracker, lever_state=lever_state, **kwargs
        )
    finally:
        prebuilt = lever_state.get("prebuilt")
        if prebuilt is not None and lever_state.get("mtp_cache") is not None:
            # Lever (c): a hedge chain queued for a round that never ran must
            # be rewound before the sidecar offsets are validated.
            _discard_hedge(
                args[0] if args else kwargs["model"],
                lever_state["mtp_cache"],
                prebuilt,
            )
            lever_state["prebuilt"] = None
        if tracker is not None and tracker.get("mtp_cache") is not None:
            mtp_cache = tracker["mtp_cache"]
            pending_hs = tracker.get("pending_hs")
            pending_ts = tracker.get("pending_ts") or []
            if pending_hs is not None and pending_ts:
                model = args[0] if args else kwargs["model"]
                model.mtp_step(
                    pending_hs,
                    mx.array([pending_ts], mx.uint32),
                    mtp_cache,
                )
            cache = tracker["cache"]
            seed_h = tracker.get("seed_h")
            lane_rng = tracker.get("rng")
            mx.eval(
                [c.state for c in cache],
                [c.state for c in mtp_cache],
                seed_h,
                *([lane_rng.key] if lane_rng is not None else []),
            )
            covered_tokens = max(
                (getattr(c, "offset", 0) for c in cache), default=0
            )
            mtp_offset = max(
                (getattr(c, "offset", 0) for c in mtp_cache), default=0
            )
            reusable = (
                covered_tokens > 0
                and seed_h is not None
                and mtp_offset == covered_tokens - 1
            )
            mtp_state_out.update(
                state=(mtp_cache, seed_h),
                covered_tokens=covered_tokens,
                reusable=reusable,
                # Where the lane stopped in its own stream. A resume rebuilds
                # from this key (LaneRNG.from_key) so the continuation does not
                # repeat draws; a fork of one snapshot into several lanes must
                # split it (LaneRNG.fork), never copy it.
                rng_key=None if lane_rng is None else lane_rng.key,
                rng_draws=None if lane_rng is None else lane_rng.draws,
            )


def hybrid_stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    draft_model: Optional[nn.Module] = None,
    *,
    max_tokens: int = 256,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """Stream ``GenerationResponse`` objects from hybrid speculative decoding
    — the hybrid twin of ``stream_generate`` with a draft model.

    Args:
        model (nn.Module): The target model.
        tokenizer: The tokenizer (shared by target and draft).
        prompt: The input prompt string or integer tokens.
        draft_model (nn.Module, optional): The draft model; ``None`` for
          retrieval-only (pure PLD) mode.
        max_tokens (int): Maximum number of tokens to generate.
        kwargs: Forwarded to ``hybrid_generate_step`` (``tau``, ``min_match``,
          ``max_span``, ``num_draft_tokens``, ``stats``, ...).
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    token_generator = hybrid_generate_step(
        prompt, model, draft_model, max_tokens=max_tokens, **kwargs
    )
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        n = -1
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

        if n < 0:
            return  # generator yielded nothing (e.g. max_tokens=0): no summary

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )
