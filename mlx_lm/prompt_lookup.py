"""Proposal backends for draft-free (prompt-lookup) speculative decoding.

Two interchangeable proposers feed the shared verify/accept/cache core in
``generate.prompt_lookup_generate_step``:

- ``NgramProposer`` — tail n-gram lookup against the running sequence. Simple,
  stateless, zero dependencies. Good default.
- ``SuffixAutomatonProposer`` — an online suffix automaton that returns the
  longest repeated suffix's continuation. Strictly stronger retrieval (finds
  longer/earlier matches than a fixed n-gram) at ~microseconds/token.

Both expose the same interface:
    observe(token: int)                      # feed each committed token
    propose(seq, max_span, prompt_len) -> list[int]

The ``SuffixAutomaton`` and ``HybridStats`` classes here come from the
hybrid-speculative work in this project (SuffixAutomaton retrieval + per-source
accounting); they are reused verbatim so the two efforts converge on one core.
"""
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


class SuffixAutomaton:
    """Online suffix automaton over token ids.

    Built incrementally (``extend`` per committed token), it answers, in
    O(suffix-link chain) per call: what is the longest suffix of the current
    sequence that also occurs ending at an earlier position, and where does that
    earlier occurrence end? ``first_end`` is the end position of the FIRST
    occurrence of a state's substrings, fixed at creation (clones inherit it).
    """

    __slots__ = ("seq", "_len", "_link", "_next", "_first_end", "_last")

    def __init__(self, tokens: Sequence[int] = ()):
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
                first_end.append(first_end[q])
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def longest_suffix_match(self, max_len: int = 16) -> Tuple[int, int]:
        """Return (match_len, next_pos): the longest suffix (<= max_len) that
        also occurs ending strictly before the current end, and the index right
        after that earlier occurrence (``seq[next_pos:]`` is the continuation).
        Returns (0, -1) when no suffix repeats."""
        n = len(self.seq)
        if n < 2:
            return 0, -1
        v = self._last
        while v != 0 and self._first_end[v] >= n - 1:
            v = self._link[v]
        if v == 0:
            return 0, -1
        return min(self._len[v], max_len), self._first_end[v] + 1


PLD_CORPUS_MODES = ("target", "uncompacted", "hybrid")


class NgramProposer:
    """Tail n-gram lookup. Stateless: proposes the continuation of the rightmost
    earlier occurrence of the tail n-gram (largest n first)."""

    def __init__(
        self,
        ngram_max: int = 3,
        ngram_min: int = 1,
        prompt_only: bool = False,
        max_lookback: int = 4096,
        retrieval_corpus: Sequence[int] | None = None,
        corpus_mode: str = "target",
    ):
        if corpus_mode not in PLD_CORPUS_MODES:
            raise ValueError(
                f"unknown prompt-lookup corpus mode {corpus_mode!r}; "
                f"expected one of {PLD_CORPUS_MODES}"
            )
        if corpus_mode != "target" and retrieval_corpus is None:
            raise ValueError(
                f"prompt-lookup corpus mode {corpus_mode!r} requires an "
                "uncompacted retrieval corpus"
            )
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.prompt_only = prompt_only
        # Bound the backward scan: on novel text the miss case otherwise
        # walks the FULL sequence per verify cycle for every g — quadratic
        # over a long generation. Proposals are verified anyway, so a bounded
        # window is lossless (worst case: fewer proposals). 0 = unbounded.
        self.max_lookback = max_lookback
        self.retrieval_corpus = (
            [int(token) for token in retrieval_corpus]
            if retrieval_corpus is not None
            else None
        )
        self.corpus_mode = corpus_mode

    def observe(self, token: int) -> None:  # stateless
        pass

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        n = len(seq)
        corpora: List[Tuple[List[int], int | None]] = []
        if self.corpus_mode in ("uncompacted", "hybrid"):
            corpora.append((self.retrieval_corpus or [], None))
        if self.corpus_mode in ("target", "hybrid"):
            corpora.append((seq, prompt_len if self.prompt_only else None))
        for g in range(self.ngram_max, self.ngram_min - 1, -1):
            # An external corpus can supply the earlier occurrence even when
            # the compacted target history contains only the live key itself.
            if n < g:
                continue
            key = seq[-g:]
            for corpus, search_len in corpora:
                limit = (
                    len(corpus)
                    if search_len is None
                    else min(search_len, len(corpus))
                )
                last = limit - g
                if corpus is seq:
                    last = min(last, n - g - 1)
                floor = (
                    -1
                    if not self.max_lookback
                    else max(-1, last - self.max_lookback)
                )
                for i in range(last, floor, -1):
                    if corpus[i : i + g] == key:
                        cont = corpus[i + g : i + g + max_span]
                        if cont:
                            return cont
        return []


class SuffixAutomatonProposer:
    """Suffix-automaton retrieval: continuation of the longest repeated suffix."""

    def __init__(self, min_match: int = 3, max_lookback: int = 32,
                 initial_tokens: Sequence[int] = ()):
        self.min_match = min_match
        self.max_lookback = max_lookback
        self.sam = SuffixAutomaton(initial_tokens)

    def observe(self, token: int) -> None:
        self.sam.extend(token)

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        mlen, nxt = self.sam.longest_suffix_match(self.max_lookback)
        if mlen >= self.min_match and 0 <= nxt < len(seq):
            return seq[nxt : nxt + max_span]
        return []


def make_proposer(spec):
    """Build an EMPTY proposer from a `backend` string, or pass through a proposer
    object. The caller seeds it (feeds the prompt via observe()) — do not seed here
    too, or a stateful backend's coordinates desync from the caller's sequence.
    spec: "ngram" | "suffix_automaton" | a proposer instance."""
    if hasattr(spec, "propose"):
        return spec
    if spec in (None, "ngram"):
        return NgramProposer()
    if spec == "suffix_automaton":
        return SuffixAutomatonProposer()
    raise ValueError(f"unknown prompt-lookup backend {spec!r}")


def snap_proposal_around_verify_cliff(
    proposal: Sequence[int], pending_rows: int = 1
) -> List[int]:
    """Avoid the measured M5 verify-batch cliff without inventing tokens.

    Target verification forwards ``pending_rows + len(proposal)`` rows.  Local
    measurements show that rows 9..15 pay the same attention plateau as a much
    longer batch, so a proposal that would land in that band is shortened to
    keep the verify batch at eight rows.  Proposals already large enough to
    reach 16 rows are preserved.  This is deliberately opt-in at the generator
    boundary because the crossover is hardware/model dependent.
    """
    if pending_rows < 1:
        raise ValueError("pending_rows must be >= 1")
    verify_rows = pending_rows + len(proposal)
    if 9 <= verify_rows <= 15:
        return list(proposal[: max(8 - pending_rows, 0)])
    return list(proposal)


def plan_proposal_around_verify_cliff(
    nominal_span: int, available_span: int, pending_rows: int = 1
) -> int:
    """Choose a safe proposal span, preferring the far side of the cliff.

    ``nominal_span`` is the configured span, while ``available_span`` includes
    the continuation and output-budget limits.  If the nominal verify shape
    lands at L=9..15, extend to L=16 when the continuation exists; otherwise
    shrink to L=8.  The opt-in caller owns the decision to exceed its nominal
    span in order to escape the measured plateau.
    """
    if nominal_span < 0 or available_span < 0:
        raise ValueError("proposal spans must be >= 0")
    if pending_rows < 1:
        raise ValueError("pending_rows must be >= 1")
    span = min(nominal_span, available_span)
    verify_rows = pending_rows + span
    if 9 <= verify_rows <= 15:
        long_span = 16 - pending_rows
        if available_span >= long_span:
            return long_span
        return min(span, max(8 - pending_rows, 0))
    return span


@dataclass
class HybridStats:
    """Per-source accounting for one prompt-lookup generation run."""

    cycles: int = 0
    retrieval_cycles: int = 0
    plain_cycles: int = 0
    retrieval_proposed: int = 0
    retrieval_accepted: int = 0
    bonus_tokens: int = 0
    plain_tokens: int = 0
    span_snap_cycles: int = 0
    span_snap_tokens: int = 0
    span_extend_cycles: int = 0
    span_extend_tokens: int = 0
    verify_span_hist: dict[int, int] = field(default_factory=dict)
    latched: bool = False
    # measured-rate gate (rate_gate=True)
    rate_gate_probed: bool = False
    rate_gate_delatched: bool = False
    rate_gate_spec_ms_per_tok: float = 0.0
    rate_gate_plain_ms_per_tok: float = 0.0
    retrieval_corpus_mode: str = "target"
    retrieval_corpus_tokens: int = 0

    @property
    def total_emitted(self) -> int:
        return self.retrieval_accepted + self.bonus_tokens + self.plain_tokens

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        acc = self.retrieval_accepted / max(self.retrieval_proposed, 1)
        return (
            f"cycles {self.cycles} (retrieval {self.retrieval_cycles}, plain {self.plain_cycles}) | "
            f"tokens {self.total_emitted}: retrieval {self.retrieval_accepted} "
            f"({self.retrieval_accepted / tot:.0%}) + bonus {self.bonus_tokens} + plain {self.plain_tokens} | "
            f"retrieval acceptance {acc:.0%} | latched={self.latched}"
        )


# Canonical/upstream name for the per-run prompt-lookup accounting struct.
# Our tree renamed it to ``HybridStats``; keep the original name as an alias so
# upstream code and tests (e.g. the rate-gate suite) resolve against it.
PromptLookupStats = HybridStats
