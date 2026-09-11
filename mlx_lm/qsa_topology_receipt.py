"""Diagnostic receipts for QSA selected-base-page overlap.

This module is absent from normal inference. Exact QSA selection topology lives
in device arrays, so making it host-visible requires a synchronization.
``build_qsa_topology_receipt`` consumes arrays that are already on the host;
``capture_compact_topology_diagnostic`` is the explicitly named adapter used by
offline experiments and the opt-in runtime capture hook.

In this receipt, a "page" is one logical QSA block.  Cohorts are computed per
query index: those are the rows that could share selected base-page work in a
batched MTP validation dispatch.  The all-query aggregate is included only as
a descriptive diagnostic and must not be read as simultaneous reuse.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

SCHEMA_ID = "mlx-lm.qsa-selected-base-page-topology/v1"


QSA_TOPOLOGY_RECEIPT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_ID,
    "type": "object",
    "required": [
        "schema",
        "capture",
        "geometry",
        "ordered_base_page_sets",
        "exact_set_cohort_sizes",
        "query_cohorts",
        "all_queries",
        "adjacent_query_reuse",
    ],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "capture": {
            "type": "object",
            "required": ["mode", "device_readback", "runtime_hot_path"],
        },
        "geometry": {
            "type": "object",
            "required": [
                "batch",
                "queries_per_row",
                "base_page_count",
                "page_size_tokens",
            ],
        },
        "ordered_base_page_sets": {"type": "array"},
        "exact_set_cohort_sizes": {"type": "array"},
        "query_cohorts": {"type": "array"},
        "all_queries": {"type": "object"},
        "adjacent_query_reuse": {"type": "object"},
    },
}


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _cohort_metrics(
    page_sets: Sequence[tuple[int, ...]], member_ids: Sequence[Any]
) -> dict:
    exact: dict[tuple[int, ...], list[Any]] = defaultdict(list)
    activity: dict[int, list[Any]] = defaultdict(list)
    memberships = 0
    for member, page_set in zip(member_ids, page_sets):
        exact[page_set].append(member)
        memberships += len(page_set)
        for page_id in page_set:
            activity[page_id].append(member)

    pairs = []
    share_any = 0
    for left, right in combinations(range(len(page_sets)), 2):
        value = _jaccard(page_sets[left], page_sets[right])
        shared = len(set(page_sets[left]) & set(page_sets[right]))
        share_any += int(shared > 0)
        pairs.append(
            {
                "left": member_ids[left],
                "right": member_ids[right],
                "intersection_pages": shared,
                "jaccard": value,
            }
        )

    union_count = len(activity)
    sizes = [len(page_set) for page_set in page_sets]
    mean_size = _mean(sizes) or 0.0
    max_size = max(sizes, default=0)
    pair_jaccards = [pair["jaccard"] for pair in pairs]
    exact_rows = [
        {
            "ordered_base_page_set": list(page_set),
            "size": len(members),
            "members": members,
        }
        for page_set, members in sorted(
            exact.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]
    return {
        "active_queries": len(page_sets),
        "selected_page_memberships": memberships,
        "unique_selected_pages": union_count,
        "exact_set_cohorts": exact_rows,
        "largest_exact_set_cohort": max((row["size"] for row in exact_rows), default=0),
        "pairwise_jaccard": {
            "pairs": pairs,
            "mean": _mean(pair_jaccards),
            "minimum": min(pair_jaccards, default=None),
            "maximum": max(pair_jaccards, default=None),
        },
        "union_inflation": {
            # One union gather versus the average or widest independent gather.
            "union_pages": union_count,
            "mean_member_pages": mean_size,
            "max_member_pages": max_size,
            "versus_mean_member": (union_count / mean_size if mean_size else 1.0),
            "versus_max_member": (union_count / max_size if max_size else 1.0),
        },
        "active_queries_per_page": [
            {
                "page_id": page_id,
                "active_queries": len(members),
                "members": members,
            }
            for page_id, members in sorted(activity.items())
        ],
        "same_page_reuse": {
            # Each membership after a page's first use is potentially reusable.
            "reused_memberships": max(0, memberships - union_count),
            "reuse_fraction": (
                max(0, memberships - union_count) / memberships if memberships else 0.0
            ),
            "pairs_sharing_any_page": share_any,
            "pair_share_fraction": share_any / len(pairs) if pairs else None,
        },
    }


def _normalize_compact(
    selected_page_ids: Sequence[Sequence[Sequence[int]]],
    selected_page_counts: Sequence[Sequence[int]],
    *,
    base_page_count: int,
) -> list[list[tuple[int, ...]]]:
    ids = np.asarray(selected_page_ids)
    counts = np.asarray(selected_page_counts)
    if ids.ndim != 3:
        raise ValueError("selected page ids must be rank-3 [B, L, K]")
    if counts.shape != ids.shape[:2]:
        raise ValueError("selected page counts must be [B, L]")
    if ids.shape[0] == 0 or ids.shape[1] == 0:
        raise ValueError("topology receipt requires at least one active query")
    if base_page_count < 0:
        raise ValueError("base_page_count must be non-negative")

    result: list[list[tuple[int, ...]]] = []
    width = int(ids.shape[2])
    for row in range(ids.shape[0]):
        row_sets = []
        for query in range(ids.shape[1]):
            count = int(counts[row, query])
            if count < 0 or count > width:
                raise ValueError("selected page count exceeds compact width")
            selected = tuple(int(value) for value in ids[row, query, :count])
            if any(page_id < 0 for page_id in selected):
                raise ValueError("selected page ids must be non-negative")
            if tuple(sorted(set(selected))) != selected:
                raise ValueError(
                    "selected page ids must be unique and ascending per query"
                )
            row_sets.append(
                tuple(page_id for page_id in selected if page_id < base_page_count)
            )
        result.append(row_sets)
    return result


def build_qsa_topology_receipt(
    selected_page_ids: Sequence[Sequence[Sequence[int]]],
    selected_page_counts: Sequence[Sequence[int]],
    *,
    base_page_count: int,
    page_size_tokens: int,
    capture_mode: str = "host_arrays",
    device_readback: bool = False,
    runtime_hot_path: bool = False,
) -> dict:
    """Build a JSON-safe selected-base-page topology receipt on the host."""
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be positive")
    page_sets = _normalize_compact(
        selected_page_ids,
        selected_page_counts,
        base_page_count=base_page_count,
    )
    batch, queries = len(page_sets), len(page_sets[0])

    query_cohorts = []
    cohort_sizes = [[0] * queries for _ in range(batch)]
    for query in range(queries):
        members = list(range(batch))
        sets = [page_sets[row][query] for row in members]
        metrics = _cohort_metrics(sets, members)
        metrics["query_index"] = query
        query_cohorts.append(metrics)
        counts = Counter(sets)
        for row, page_set in enumerate(sets):
            cohort_sizes[row][query] = counts[page_set]

    flattened_sets = [
        page_sets[row][query] for row in range(batch) for query in range(queries)
    ]
    flattened_members = [
        {"row": row, "query": query} for row in range(batch) for query in range(queries)
    ]
    all_queries = _cohort_metrics(flattened_sets, flattened_members)
    all_queries["simultaneous_reuse_claim"] = False

    adjacent = []
    for row in range(batch):
        for query in range(1, queries):
            previous = page_sets[row][query - 1]
            current = page_sets[row][query]
            retained = len(set(previous) & set(current))
            adjacent.append(
                {
                    "row": row,
                    "from_query": query - 1,
                    "to_query": query,
                    "retained_pages": retained,
                    "current_pages": len(current),
                    "retained_fraction_of_current": (
                        retained / len(current) if current else 1.0
                    ),
                    "jaccard": _jaccard(previous, current),
                }
            )

    return {
        "schema": SCHEMA_ID,
        "capture": {
            "mode": capture_mode,
            "device_readback": bool(device_readback),
            "runtime_hot_path": bool(runtime_hot_path),
        },
        "geometry": {
            "batch": batch,
            "queries_per_row": queries,
            "base_page_count": int(base_page_count),
            "page_size_tokens": int(page_size_tokens),
            "base_tokens": int(base_page_count) * int(page_size_tokens),
        },
        "ordered_base_page_sets": [
            [list(page_set) for page_set in row] for row in page_sets
        ],
        "exact_set_cohort_sizes": cohort_sizes,
        "query_cohorts": query_cohorts,
        "all_queries": all_queries,
        "adjacent_query_reuse": {
            "pairs": adjacent,
            "mean_jaccard": _mean([item["jaccard"] for item in adjacent]),
            "mean_retained_fraction_of_current": _mean(
                [item["retained_fraction_of_current"] for item in adjacent]
            ),
        },
    }


def capture_compact_topology_diagnostic(
    compact, *, base_tokens: int, runtime_hot_path: bool = False
) -> dict:
    """Synchronize compact device selections and build a diagnostic receipt.

    This function deliberately performs a device-to-host conversion. Never
    call it from a performance measurement or a production path without an
    explicit diagnostic opt-in.
    """
    block_size = int(compact.block_size)
    if base_tokens < 0 or base_tokens % block_size:
        raise ValueError("base_tokens must align to compact.block_size")
    return build_qsa_topology_receipt(
        np.asarray(compact.block_ids),
        np.asarray(compact.block_counts),
        base_page_count=base_tokens // block_size,
        page_size_tokens=block_size,
        capture_mode="diagnostic_device_readback",
        device_readback=True,
        runtime_hot_path=runtime_hot_path,
    )


def write_qsa_topology_diagnostic(
    output_path: str | Path,
    compact,
    *,
    base_tokens: int,
    metadata: Optional[dict[str, Any]] = None,
) -> dict:
    """Write one runtime topology receipt atomically.

    The caller must opt into the device readback and keep this outside timed
    measurements. ``metadata`` must contain only JSON-safe hosted values.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    topology = capture_compact_topology_diagnostic(
        compact, base_tokens=base_tokens, runtime_hot_path=True
    )
    payload = {
        "source": {
            "kind": "runtime_private_delta",
            "metadata": dict(metadata or {}),
        },
        "topology": topology,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{id(payload):x}")
    try:
        temporary.write_text(rendered)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _unwrap_topology(payload: dict) -> dict:
    topology = payload.get("topology", payload)
    if topology.get("schema") != SCHEMA_ID:
        raise ValueError("unsupported QSA topology receipt schema")
    return topology


def analyze_qsa_topology_receipts(receipts: Sequence[dict]) -> dict:
    """Aggregate simultaneous-query reuse and choose a bounded next probe."""

    if not receipts:
        raise ValueError("at least one topology receipt is required")
    topologies = [_unwrap_topology(payload) for payload in receipts]
    cohorts = [
        cohort for topology in topologies for cohort in topology["query_cohorts"]
    ]
    largest = [int(cohort["largest_exact_set_cohort"]) for cohort in cohorts]
    active_queries = sum(int(cohort["active_queries"]) for cohort in cohorts)
    cohortable_members = sum(
        sum(
            int(exact["size"])
            for exact in cohort["exact_set_cohorts"]
            if int(exact["size"]) >= 2
        )
        for cohort in cohorts
    )
    exact_reused_memberships = sum(
        sum(
            (int(exact["size"]) - 1) * len(exact["ordered_base_page_set"])
            for exact in cohort["exact_set_cohorts"]
            if int(exact["size"]) >= 2
        )
        for cohort in cohorts
    )
    memberships = sum(int(cohort["selected_page_memberships"]) for cohort in cohorts)
    unique_pages = sum(int(cohort["unique_selected_pages"]) for cohort in cohorts)
    reused = max(0, memberships - unique_pages)
    pair_jaccards = [
        float(pair["jaccard"])
        for cohort in cohorts
        for pair in cohort["pairwise_jaccard"]["pairs"]
    ]
    mean_largest = _mean(largest) or 0.0
    queries_per_page = memberships / unique_pages if unique_pages else 1.0
    read_reduction = reused / memberships if memberships else 0.0
    exact_gate = mean_largest >= 2.0 and exact_reused_memberships > 0
    ordered_union_gate = queries_per_page >= 1.5 and read_reduction >= 0.25
    if exact_gate:
        recommendation = "exact_set"
    elif ordered_union_gate:
        recommendation = "ordered_union"
    else:
        recommendation = "independent"
    return {
        "schema": "mlx-lm.qsa-selected-base-page-topology-summary/v1",
        "receipt_count": len(topologies),
        "simultaneous_query_cohort_count": len(cohorts),
        "exact_set": {
            "mean_largest_cohort": mean_largest,
            "maximum_largest_cohort": max(largest, default=0),
            "cohortable_member_fraction": (
                cohortable_members / active_queries if active_queries else 0.0
            ),
            "predicted_base_read_reduction_fraction_upper_bound": (
                exact_reused_memberships / memberships if memberships else 0.0
            ),
            "gate": {
                "threshold_mean_largest_cohort": 2.0,
                "passes": exact_gate,
            },
        },
        "ordered_union": {
            "selected_page_memberships": memberships,
            "unique_pages_per_simultaneous_cohort_sum": unique_pages,
            "mean_queries_per_selected_page": queries_per_page,
            "predicted_base_read_reduction_fraction_upper_bound": (read_reduction),
            "mean_pairwise_jaccard": _mean(pair_jaccards),
            "gate": {
                "threshold_queries_per_selected_page": 1.5,
                "threshold_base_read_reduction_fraction": 0.25,
                "passes": ordered_union_gate,
            },
        },
        "recommendation": recommendation,
        "interpretation": (
            "The ordered-union reduction is a topology-only upper bound; "
            "it excludes remap, masking, dispatch, and merge costs."
        ),
    }


__all__ = [
    "QSA_TOPOLOGY_RECEIPT_SCHEMA",
    "SCHEMA_ID",
    "analyze_qsa_topology_receipts",
    "build_qsa_topology_receipt",
    "capture_compact_topology_diagnostic",
    "write_qsa_topology_diagnostic",
]
