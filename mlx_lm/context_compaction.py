"""Selection policies for segmented transcript compaction.

These policies choose source-history segments. They do not mutate target KV or
recurrent state; a serving backend must separately provide an exact rebuild or
a model-declared live-surgery implementation.
"""

from dataclasses import dataclass
from typing import Collection, Mapping

from .cache_planes import TranscriptLedgerPlane


COMPACTION_STRATEGIES = (
    "oldest_contiguous",
    "largest_first",
    "lowest_importance",
)


@dataclass(frozen=True)
class CompactionSelection:
    strategy: str
    segment_ids: tuple[str, ...]
    reclaimed_tokens: int
    target_tokens: int


def select_transcript_segments(
    plane: TranscriptLedgerPlane,
    target_tokens: int,
    *,
    protected_segment_ids: Collection[str] = (),
    strategy: str | None = None,
    importance_by_segment: Mapping[str, float] | None = None,
) -> CompactionSelection:
    """Choose whole transcript segments until the requested reclaim target."""

    selected_strategy = strategy or plane.compaction_strategy
    if selected_strategy not in COMPACTION_STRATEGIES:
        raise ValueError(
            f"unknown compaction strategy {selected_strategy!r}; "
            f"expected one of {COMPACTION_STRATEGIES}"
        )
    if target_tokens < 0:
        raise ValueError("compaction target must be non-negative")
    protected = set(protected_segment_ids)

    if selected_strategy == "oldest_contiguous":
        candidates = []
        for segment in plane.segments:
            if segment.segment_id in protected:
                break
            candidates.append(segment)
    elif selected_strategy == "largest_first":
        candidates = sorted(
            (
                segment
                for segment in plane.segments
                if segment.segment_id not in protected
            ),
            key=lambda segment: (-len(segment.token_ids), segment.token_start),
        )
    else:
        if importance_by_segment is None:
            raise ValueError(
                "lowest_importance compaction requires segment importance scores"
            )
        eligible = [
            segment
            for segment in plane.segments
            if segment.segment_id not in protected
        ]
        missing = [
            segment.segment_id
            for segment in eligible
            if segment.segment_id not in importance_by_segment
        ]
        if missing:
            raise ValueError(
                "missing segment importance scores for " + ", ".join(missing)
            )
        candidates = sorted(
            eligible,
            key=lambda segment: (
                float(importance_by_segment[segment.segment_id]),
                segment.token_start,
            ),
        )

    selected = []
    reclaimed = 0
    for segment in candidates:
        if reclaimed >= target_tokens:
            break
        selected.append(segment.segment_id)
        reclaimed += len(segment.token_ids)
    return CompactionSelection(
        strategy=selected_strategy,
        segment_ids=tuple(selected),
        reclaimed_tokens=reclaimed,
        target_tokens=target_tokens,
    )
