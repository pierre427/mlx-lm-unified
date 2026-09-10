"""Per-batch-size draft-depth policy for speculative decoding.

Speculative paths in this tree (adaptive PLD's MTP tail, MTP self-spec) use a
fixed draft depth ``num_draft``. The verify cost of a draft chain scales with
concurrency: at batch size 1 a deep chain amortizes well, but when several
lanes decode at once every lane pays the widened verify forward, so a depth
that wins single-stream loses under load (the same observation behind vLLM's
``num_speculative_tokens_per_batch_size`` / DSD and SGLang DSpark's
verify-budget policy).

This module supplies the policy in two composable pieces:

1. A **depth table** mapping batch-size bands to a depth ceiling: the
   effective depth for a lane is ``min(base_num_draft, band_K)``. The table
   only ever *reduces* the configured depth — it never inflates it — so it is
   safe to apply unconditionally. The default table is conservative::

       bs 1    -> base num_draft (no ceiling)
       bs 2-4  -> min(base, 3)
       bs 5+   -> min(base, 2)

   Overridable per call (``table=``) or process-wide via the environment
   variable ``MLX_LM_SPEC_DEPTH_TABLE`` with rows ``lo:K``, ``lo-hi:K`` or
   ``lo+:K`` separated by commas, e.g. ``"1:6,2-4:3,5-8:2"``. Parsing is
   validated loudly: malformed rows, overlapping bands, or negative depths
   raise ``ValueError`` immediately.

2. A **hard M5 cap** on total verify width: draft tokens + 1 bonus must stay
   <= 8. The M5 SDPA decode kernel steps its qL tile at 8 -> 12 and M=12
   costs the same as M=32 (mlx#3826, closed won't-fix upstream), so a 9..15
   row verify pays the full plateau. The cap is enforced AFTER table lookup;
   the first time a configured depth collides with it a one-time warning is
   emitted. (PLD *retrieval* spans are governed separately by the opt-in
   cliff planner in ``prompt_lookup.py``, which may deliberately extend a
   span to the far side of the plateau — this cap covers draft/MTP chains,
   whose acceptance decays too fast for the far side ever to win.)

Note on scope: speculation in this tree is single-stream only (the continuous
``BatchGenerator`` path never speculates, and ``server.py`` routes
prompt-lookup requests to the sequential path). ``batch_size`` here is the
number of concurrent lanes the *caller* (a multi-lane server) is running;
each lane's generator receives it as a hint at entry.
"""

import os
import threading
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

# M5 SDPA verify cliff: qL tiles step 8 -> 12, and M=12 costs the same as
# M=32 (mlx#3826, closed won't-fix). Total verify width (drafts + 1 bonus)
# beyond 8 pays the full plateau for tokens that mostly get rejected.
MAX_VERIFY_WIDTH = 8
MAX_DRAFT_TOKENS = MAX_VERIFY_WIDTH - 1  # the bonus token takes one slot

DEPTH_TABLE_ENV = "MLX_LM_SPEC_DEPTH_TABLE"

# One-time warning latch for user-configured depths colliding with the cap.
# Generators run on caller threads (multi-lane servers), so the test-and-set
# is guarded by a lock to keep the warning truly one-time.
_cap_warning_emitted = False
_cap_warning_lock = threading.Lock()


@dataclass(frozen=True)
class DepthBand:
    """One table row: batch sizes ``lo..hi`` get depth ceiling ``k``.

    ``hi=None`` means open-ended (``lo`` and up). ``k=None`` means no ceiling
    (the caller's base ``num_draft`` passes through).
    """

    lo: int
    hi: Optional[int]
    k: Optional[int]

    def __post_init__(self):
        if self.lo < 1:
            raise ValueError(f"depth-table band lo must be >= 1, got {self.lo}")
        if self.hi is not None and self.hi < self.lo:
            raise ValueError(f"depth-table band has lo > hi ({self.lo} > {self.hi})")
        if self.k is not None and self.k < 0:
            raise ValueError(f"depth-table band K must be >= 0, got {self.k}")

    def covers(self, batch_size: int) -> bool:
        return self.lo <= batch_size and (self.hi is None or batch_size <= self.hi)


class DepthTable:
    """An ordered, non-overlapping set of :class:`DepthBand` rows.

    Lookup semantics (see :meth:`depth_for`): the band covering ``batch_size``
    supplies a ceiling ``min(base, K)``. A batch size *above* every band falls
    back to the nearest band below it (concurrency higher than the table
    anticipated should never speculate *deeper* than the highest configured
    band). A batch size below every band — impossible for tables that start
    at 1, like the default — uses the base depth unchanged.
    """

    def __init__(self, bands: Sequence[DepthBand]):
        if not bands:
            raise ValueError("depth table needs at least one band")
        bands = sorted(bands, key=lambda b: b.lo)
        for prev, cur in zip(bands, bands[1:]):
            if prev.hi is None or cur.lo <= prev.hi:
                raise ValueError(
                    "depth-table bands overlap: "
                    f"{prev.lo}-{'+' if prev.hi is None else prev.hi} and "
                    f"{cur.lo}-{'+' if cur.hi is None else cur.hi}"
                )
        self.bands: Tuple[DepthBand, ...] = tuple(bands)

    @classmethod
    def parse(cls, spec: str) -> "DepthTable":
        """Parse ``"1:6,2-4:3,5+:2"``-style specs, loudly.

        Each row is ``lo:K``, ``lo-hi:K`` or ``lo+:K``. Whitespace around
        rows and separators is ignored. Any malformed row, overlapping bands,
        or negative K raises ``ValueError`` naming the offending row.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(
                f"empty depth-table spec {spec!r}; expected rows like '1:6,2-4:3,5+:2'"
            )
        bands: List[DepthBand] = []
        for row in spec.split(","):
            row = row.strip()
            if not row:
                raise ValueError(f"empty row in depth-table spec {spec!r}")
            parts = row.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"bad depth-table row {row!r} (expected 'lo:K', 'lo-hi:K' or 'lo+:K')"
                )
            rng, k_str = parts[0].strip(), parts[1].strip()
            try:
                k = int(k_str)
            except ValueError:
                raise ValueError(
                    f"bad depth in depth-table row {row!r}: {k_str!r} is not an integer"
                ) from None
            try:
                if rng.endswith("+"):
                    lo, hi = int(rng[:-1]), None
                elif "-" in rng:
                    lo_str, hi_str = rng.split("-", 1)
                    lo, hi = int(lo_str), int(hi_str)
                else:
                    lo = hi = int(rng)
            except ValueError:
                raise ValueError(
                    f"bad batch-size range in depth-table row {row!r}: {rng!r}"
                ) from None
            try:
                bands.append(DepthBand(lo, hi, k))
            except ValueError as e:
                raise ValueError(f"bad depth-table row {row!r}: {e}") from None
        return cls(bands)

    def depth_for(self, batch_size: int, base_num_draft: int) -> int:
        """Depth ceiling applied to ``base_num_draft`` for ``batch_size``.

        Does NOT apply the M5 verify-width cap — that is layered on by
        :func:`draft_depth_for` (cap after table lookup, so the one-time
        collision warning can tell the user which knob lost).
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        base = max(0, int(base_num_draft))
        chosen: Optional[DepthBand] = None
        for band in self.bands:
            if band.covers(batch_size):
                chosen = band
                break
            if band.lo <= batch_size:
                chosen = band  # nearest band below; keep scanning for a cover
        if chosen is None or chosen.k is None:
            return base
        return min(base, chosen.k)


#: Conservative default: single-stream keeps the configured depth; moderate
#: concurrency caps at 3; heavy concurrency caps at 2.
DEFAULT_DEPTH_TABLE = DepthTable(
    [
        DepthBand(1, 1, None),
        DepthBand(2, 4, 3),
        DepthBand(5, None, 2),
    ]
)


def resolve_depth_table(
    table: Optional[Union[DepthTable, str]] = None,
) -> DepthTable:
    """Resolve the active depth table: explicit arg > env var > default.

    ``table`` may be a :class:`DepthTable` or a spec string. The environment
    variable ``MLX_LM_SPEC_DEPTH_TABLE`` is parsed on every call (no caching)
    so a bad value fails loudly at the next generation entry, not at import.
    """
    if table is not None:
        if isinstance(table, str):
            return DepthTable.parse(table)
        return table
    env = os.environ.get(DEPTH_TABLE_ENV)
    if env is not None:
        try:
            return DepthTable.parse(env)
        except ValueError as e:
            raise ValueError(f"invalid {DEPTH_TABLE_ENV}: {e}") from None
    return DEFAULT_DEPTH_TABLE


def cap_draft_tokens(num_draft: int) -> int:
    """Clamp a draft depth so drafts + 1 bonus <= ``MAX_VERIFY_WIDTH``.

    Emits a one-time warning the first time a configured depth collides with
    the cap, so a user who sets K=10 learns why they observe 7.
    """
    global _cap_warning_emitted
    num_draft = max(0, int(num_draft))
    if num_draft > MAX_DRAFT_TOKENS:
        with _cap_warning_lock:
            emit = not _cap_warning_emitted
            _cap_warning_emitted = True
        if emit:
            warnings.warn(
                f"speculative draft depth {num_draft} exceeds the M5 verify-width "
                f"cap ({MAX_DRAFT_TOKENS} drafts + 1 bonus = {MAX_VERIFY_WIDTH} "
                "rows; the SDPA qL 8->12 tile cliff makes wider verifies cost "
                f"like M=32 — mlx#3826). Using {MAX_DRAFT_TOKENS}.",
                RuntimeWarning,
                stacklevel=3,
            )
        return MAX_DRAFT_TOKENS
    return num_draft


def draft_depth_for(
    batch_size: int,
    base_num_draft: int,
    table: Optional[Union[DepthTable, str]] = None,
) -> int:
    """Effective draft depth for one lane at the given concurrency.

    Applies the depth table (explicit ``table`` arg, else the
    ``MLX_LM_SPEC_DEPTH_TABLE`` environment override, else the conservative
    default), then the hard M5 verify-width cap. Composes with — does not
    replace — the measured rate gates in the speculative paths: the gate
    decides IF a lane speculates, this decides HOW DEEP.
    """
    return cap_draft_tokens(
        resolve_depth_table(table).depth_for(batch_size, base_num_draft)
    )
