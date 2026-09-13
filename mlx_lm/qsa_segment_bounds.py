"""Offline exact segment bounds for QSA hard top-k selection.

This module is a model-free feasibility probe. It does not run in the Qwen4
hot path. The selector uses conservative centroid-radius bounds to avoid
scoring segments that cannot enter the exact top-k set.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class QSASegmentIndex:
    """Centroid-radius metadata for one pooled-key matrix."""

    keys: np.ndarray
    centroids: np.ndarray
    radii: np.ndarray
    starts: np.ndarray
    stops: np.ndarray

    @property
    def segment_size(self) -> int:
        if not len(self.starts):
            return 0
        return int(self.stops[0] - self.starts[0])


@dataclass(frozen=True)
class QSATopKResult:
    """Exact selected ids plus mechanism-engagement counters."""

    block_ids: np.ndarray
    scores: np.ndarray
    blocks_total: int
    blocks_valid: int
    blocks_scored: int
    segments_total: int
    segments_scored: int
    segments_pruned: int

    @property
    def block_prune_fraction(self) -> float:
        if not self.blocks_valid:
            return 0.0
        return 1.0 - self.blocks_scored / self.blocks_valid

    @property
    def estimated_dot_fraction(self) -> float:
        """Centroid and exact-key dot products versus dense key scoring."""
        if not self.blocks_valid:
            return 0.0
        return (self.segments_total + self.blocks_scored) / self.blocks_valid


@dataclass(frozen=True)
class QSAValueProfile:
    """Offline profile of which segments repeatedly supply selected blocks."""

    block_selection_counts: np.ndarray
    query_touch_counts: np.ndarray
    hot_segment_ids: np.ndarray
    selected_memberships: int
    hot_membership_coverage: float
    hot_query_touch_coverage: float


def profile_qsa_segment_value(
    selected_block_rows,
    *,
    segment_size: int,
    segment_count: int,
    hot_fraction: float = 0.1,
) -> QSAValueProfile:
    """Measure concentration in exact QSA selections.

    Selection frequency is evidence of value, not a retention policy. A later
    policy must also account for bytes, recompute cost, fanout, and drift.
    """
    if segment_size < 1 or segment_count < 0:
        raise ValueError("segment_size must be positive and segment_count non-negative")
    if not 0.0 < hot_fraction <= 1.0:
        raise ValueError("hot_fraction must be in (0, 1]")
    memberships = np.zeros(segment_count, dtype=np.int64)
    query_touches = np.zeros(segment_count, dtype=np.int64)
    query_count = 0
    for row in selected_block_rows:
        ids = np.asarray(row, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= segment_size * segment_count):
            raise ValueError("selected block id is outside the segment range")
        segments = ids // segment_size
        np.add.at(memberships, segments, 1)
        if len(segments):
            np.add.at(query_touches, np.unique(segments), 1)
        query_count += 1
    hot_count = min(segment_count, max(1, math.ceil(segment_count * hot_fraction)))
    segment_ids = np.arange(segment_count, dtype=np.int64)
    hot_ids = np.lexsort((segment_ids, -memberships))[:hot_count]
    membership_total = int(memberships.sum())
    touch_total = int(query_touches.sum())
    return QSAValueProfile(
        memberships,
        query_touches,
        hot_ids,
        membership_total,
        0.0 if membership_total == 0 else float(memberships[hot_ids].sum() / membership_total),
        0.0 if touch_total == 0 else float(query_touches[hot_ids].sum() / touch_total),
    )


def _as_finite_float64(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def build_qsa_segment_index(keys, *, segment_size: int) -> QSASegmentIndex:
    """Build conservative Euclidean bounds over contiguous key segments.

    ``keys`` has shape ``(blocks, head_dim)``. Calculations use float64 and
    every radius rounds toward positive infinity.
    """
    keys = _as_finite_float64(keys, name="keys")
    if keys.ndim != 2:
        raise ValueError("keys must have shape (blocks, head_dim)")
    if keys.shape[1] < 1:
        raise ValueError("keys must have a positive head dimension")
    if segment_size < 1:
        raise ValueError("segment_size must be positive")

    starts = np.arange(0, keys.shape[0], segment_size, dtype=np.int64)
    stops = np.minimum(starts + segment_size, keys.shape[0])
    if not len(starts):
        return QSASegmentIndex(
            keys=keys,
            centroids=np.empty((0, keys.shape[1]), dtype=np.float64),
            radii=np.empty((0,), dtype=np.float64),
            starts=starts,
            stops=stops,
        )

    centroids = np.stack(
        [keys[start:stop].mean(axis=0) for start, stop in zip(starts, stops)]
    )
    radii = np.asarray(
        [
            np.linalg.norm(keys[start:stop] - centroid, axis=1).max(initial=0.0)
            for start, stop, centroid in zip(starts, stops, centroids)
        ],
        dtype=np.float64,
    )
    radii = np.nextafter(radii, np.inf)
    return QSASegmentIndex(keys, centroids, radii, starts, stops)


def qsa_scores(query, keys) -> np.ndarray:
    """Return the Qwen4 QSA score for each pooled key in float64."""
    query = _as_finite_float64(query, name="query")
    keys = _as_finite_float64(keys, name="keys")
    if query.ndim != 2:
        raise ValueError("query must have shape (heads, head_dim)")
    if keys.ndim != 2 or query.shape[1] != keys.shape[1]:
        raise ValueError("query and keys must have the same head dimension")
    values = query @ keys.T
    return np.maximum(values, 0.0).sum(axis=0) / math.sqrt(query.shape[1])


def _stable_topk(scores: np.ndarray, block_ids: np.ndarray, k: int):
    if k == 0:
        return block_ids[:0], scores[:0]
    order = np.lexsort((block_ids, -scores))[:k]
    return block_ids[order], scores[order]


def dense_qsa_topk(query, keys, *, k: int, valid_blocks: int | None = None):
    """Dense deterministic reference selection.

    Ties use ascending block id. The production selector does not promise an
    order for ties, so callers should compare sets when validating MLX output.
    """
    keys = _as_finite_float64(keys, name="keys")
    valid = keys.shape[0] if valid_blocks is None else int(valid_blocks)
    if not 0 <= valid <= keys.shape[0]:
        raise ValueError("valid_blocks is outside the key range")
    if k < 0:
        raise ValueError("k must be non-negative")
    count = min(k, valid)
    ids = np.arange(valid, dtype=np.int64)
    return _stable_topk(qsa_scores(query, keys[:valid]), ids, count)


def qsa_segment_upper_bounds(query, index: QSASegmentIndex) -> np.ndarray:
    """Return conservative QSA score bounds for every segment."""
    query = _as_finite_float64(query, name="query")
    if query.ndim != 2 or query.shape[1] != index.keys.shape[1]:
        raise ValueError("query must have shape (heads, key head_dim)")
    if not len(index.starts):
        return np.empty((0,), dtype=np.float64)
    center_scores = query @ index.centroids.T
    query_norms = np.nextafter(np.linalg.norm(query, axis=1), np.inf)
    head_bounds = center_scores + query_norms[:, None] * index.radii[None, :]
    bounds = np.maximum(head_bounds, 0.0).sum(axis=0) / math.sqrt(query.shape[1])
    # Cover accumulated float64 dot, norm, and reduction error. This probe
    # prefers a missed pruning opportunity to an unsafe bound.
    margin = (
        np.finfo(np.float64).eps
        * 16
        * query.shape[1]
        * np.maximum(1.0, np.abs(bounds))
    )
    return np.nextafter(bounds + margin, np.inf)


def exact_segment_qsa_topk(
    query,
    index: QSASegmentIndex,
    *,
    k: int,
    valid_blocks: int | None = None,
) -> QSATopKResult:
    """Select exact QSA top-k blocks with conservative segment pruning."""
    query = _as_finite_float64(query, name="query")
    if query.ndim != 2 or query.shape[1] != index.keys.shape[1]:
        raise ValueError("query must have shape (heads, key head_dim)")
    valid = index.keys.shape[0] if valid_blocks is None else int(valid_blocks)
    if not 0 <= valid <= index.keys.shape[0]:
        raise ValueError("valid_blocks is outside the key range")
    if k < 0:
        raise ValueError("k must be non-negative")
    count = min(k, valid)
    total_segments = int(np.searchsorted(index.starts, valid, side="left"))
    if count == 0:
        return QSATopKResult(
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            index.keys.shape[0],
            valid,
            0,
            total_segments,
            0,
            total_segments,
        )

    bounds = qsa_segment_upper_bounds(query, index)[:total_segments]
    segment_ids = np.arange(total_segments, dtype=np.int64)
    visit_order = np.lexsort((segment_ids, -bounds))
    scored_ids: list[np.ndarray] = []
    scored_values: list[np.ndarray] = []
    blocks_scored = 0
    segments_scored = 0
    threshold = -np.inf

    for segment_id in visit_order:
        # Strict inequality keeps every segment that can tie the kth score.
        if blocks_scored >= count and bounds[segment_id] < threshold:
            break
        start = int(index.starts[segment_id])
        stop = min(int(index.stops[segment_id]), valid)
        if stop <= start:
            continue
        ids = np.arange(start, stop, dtype=np.int64)
        values = qsa_scores(query, index.keys[start:stop])
        scored_ids.append(ids)
        scored_values.append(values)
        blocks_scored += len(ids)
        segments_scored += 1
        if blocks_scored >= count:
            all_values = np.concatenate(scored_values)
            threshold = float(np.partition(all_values, -count)[-count])

    all_ids = np.concatenate(scored_ids)
    all_values = np.concatenate(scored_values)
    selected_ids, selected_scores = _stable_topk(all_values, all_ids, count)
    return QSATopKResult(
        selected_ids,
        selected_scores,
        index.keys.shape[0],
        valid,
        blocks_scored,
        total_segments,
        segments_scored,
        total_segments - segments_scored,
    )
