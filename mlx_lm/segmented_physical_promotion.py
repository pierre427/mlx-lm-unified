# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Opt-in cache-only promotion from segmented Qwen4 B1 rows to physical B2.

``begin`` queues the immutable-prefix copies on a caller-owned second Metal
stream.  ``finish`` is called, on the same Python thread, only after the first
segmented proposal has been committed.  It patches the uniformly accepted
target suffix, then publishes a normal :class:`BatchedSelfMTPState`.

The narrow initial contract matches the live-tip gate: every QSA row starts at
the same offset and every row advances by the same positive amount.  Refusing
ragged advances is important: otherwise shorter rows need a full-history move
at publication rather than a suffix-only patch.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable

import mlx.core as mx

from .hybrid_speculative import (
    BatchedSelfMTPState,
    SegmentedSelfMTPState,
    SelfMTPCachePair,
)
from .models.qwen4_exp import BatchQSAKVCache, QSAKVCache, _qsa_merge_summaries
from .segmented_batch_cache import (
    SegmentedBatchArraysCache,
    SegmentedBatchQSAKVCache,
    build_segmented_batch_cache_pair,
)


class SegmentedPhysicalPromotionDeclined(RuntimeError):
    """The state cannot be promoted by the suffix-only exact contract."""


@dataclass(frozen=True)
class SegmentedPhysicalPromotionReceipt:
    queued_ns: int
    finish_ns: int
    stream_wait_ns: int
    reserved_bytes: int
    patched_bytes: int
    recurrent_arrays_reused: int
    cleanup_error_count: int
    rows: int
    layers: int
    advance: int


def _offset(cache: Any) -> int:
    value = getattr(cache, "offset", None)
    if isinstance(value, int):
        return value
    size = getattr(cache, "size", None)
    if callable(size):
        result = size()
        if isinstance(result, int):
            return result
    raise SegmentedPhysicalPromotionDeclined(
        f"{type(cache).__name__} has no host-readable scalar offset"
    )


def _state_signature(state: SegmentedSelfMTPState):
    return (
        state.membership_epoch,
        tuple(lane.uid for lane in state.lanes),
        tuple(id(pair) for pair in state.row_caches),
        tuple(id(transaction) for transaction in state.transactions),
        tuple(
            (transaction.lineage.generation, transaction.position)
            for transaction in state.transactions
        ),
    )


def _validate_segmented_view(state, view):
    if not isinstance(state, SegmentedSelfMTPState):
        raise TypeError("physical promotion requires SegmentedSelfMTPState")
    if state.poisoned or state.proposal_open or not state.lanes:
        raise SegmentedPhysicalPromotionDeclined(
            "promotion begins only on a healthy, proposal-closed cohort"
        )
    if not (
        len(state.lanes) == len(state.row_caches) == len(state.transactions)
    ):
        raise SegmentedPhysicalPromotionDeclined(
            "segmented lane/cache/transaction ownership is misaligned"
        )
    if view is not state._segmented_caches:
        raise SegmentedPhysicalPromotionDeclined("segmented compute view was replaced")
    for name, group in (("target", view.target), ("draft", view.draft)):
        if not group or any(
            not isinstance(c, (SegmentedBatchQSAKVCache, SegmentedBatchArraysCache))
            for c in group
        ):
            raise SegmentedPhysicalPromotionDeclined(
                "promotion supports only Qwen4 segmented QSA/recurrent layers"
            )
        expected_groups = [getattr(pair, name) for pair in state.row_caches]
        if any(len(row) != len(group) for row in expected_groups):
            raise SegmentedPhysicalPromotionDeclined(
                f"segmented {name} layer width changed"
            )
        for layer, cache in enumerate(group):
            expected_rows = tuple(id(row[layer]) for row in expected_groups)
            if tuple(id(row) for row in cache.rows) != expected_rows:
                raise SegmentedPhysicalPromotionDeclined(
                    f"segmented {name} view no longer owns the live rows"
                )


def ensure_segmented_compute_view(
    state: SegmentedSelfMTPState, *, note: Callable[[str, int], None] | None = None
) -> SelfMTPCachePair:
    """Return and retain the exact segmented compute view for ``state``."""

    if state._segmented_caches is None:
        if state.proposal_open:
            raise SegmentedPhysicalPromotionDeclined(
                "cannot build a compute view during an open proposal"
            )
        state._segmented_caches = build_segmented_batch_cache_pair(
            state.row_caches,
            note=note,
            shared_qsa_prefix=(state.shared_qsa_prefix_id is not None),
        )
    _validate_segmented_view(state, state._segmented_caches)
    return state._segmented_caches


def _reserve_qsa(cache: SegmentedBatchQSAKVCache, tail: int):
    rows = cache.rows
    starts = [_offset(row) for row in rows]
    if len(set(starts)) != 1:
        raise SegmentedPhysicalPromotionDeclined(
            "suffix-only promotion requires equal starting QSA offsets"
        )
    start = starts[0]
    if start <= 0 or tail < 0:
        raise SegmentedPhysicalPromotionDeclined("invalid QSA base/tail geometry")
    populated = next((row for row in rows if row.keys is not None), None)
    if populated is None:
        raise SegmentedPhysicalPromotionDeclined("cannot promote an empty QSA cache")
    for row in rows:
        if type(row) is not QSAKVCache or row.keys is None or row.values is None:
            raise SegmentedPhysicalPromotionDeclined(
                "physical promotion requires plain populated QSAKVCache rows"
            )
        if row.index_keys is None or row.index_keys.shape[1] < start:
            raise SegmentedPhysicalPromotionDeclined(
                "a QSA raw-key ledger is shorter than its base offset"
            )
    # Match BatchKVCache's normal geometric contract.  Allocating only the
    # suffix needed by the first commit makes the first post-promotion decode
    # grow and copy the entire long-context KV buffer again.  The raw-key
    # ledger does not have spare-capacity semantics (update_index_keys slices
    # and concatenates), so reserve only the suffix that finish() may patch.
    live_width = start + tail
    step = int(BatchQSAKVCache.step)
    # A normal physical B2 is merged at exactly ``start`` and its first
    # update appends one whole growth slab.  Preserve that post-first-cycle
    # geometry (including batch stride), rather than globally rounding the
    # sequence width: compiled/view kernels can specialize on those strides.
    slabs = max(1, (tail + step - 1) // step)
    kv_width = start + slabs * step
    batch = BatchQSAKVCache([0] * len(rows))
    batch.keys = mx.zeros(
        (len(rows), populated.keys.shape[1], kv_width, populated.keys.shape[3]),
        dtype=populated.keys.dtype,
    )
    batch.values = mx.zeros(
        (len(rows), populated.values.shape[1], kv_width, populated.values.shape[3]),
        dtype=populated.values.dtype,
    )
    index = rows[0].index_keys
    batch.index_keys = mx.zeros(
        (len(rows), live_width, index.shape[2]), dtype=index.dtype
    )
    for i, row in enumerate(rows):
        batch.keys[i : i + 1, :, :start, :] = row.keys[..., :start, :]
        batch.values[i : i + 1, :, :start, :] = row.values[..., :start, :]
        batch.index_keys[i : i + 1, :start] = row.index_keys[:, :start]
    batch.offset = mx.array(starts)
    batch.left_padding = mx.zeros((len(rows),), dtype=mx.int32)
    batch._idx = start
    return batch, start


def _unwrap_recurrent(cache: SegmentedBatchArraysCache):
    row_types = {type(row) for row in cache.rows}
    if len(row_types) != 1:
        raise SegmentedPhysicalPromotionDeclined(
            "segmented recurrent rows changed concrete cache type"
        )
    result = row_types.pop()(len(cache.cache))
    # The arrays themselves are the retained compute state.  Only the Python
    # list is new; no tensor copy or concatenate is performed here.
    result.cache = list(cache.cache)
    result.left_padding = cache.left_padding
    result.lengths = cache.lengths
    result._host_left_padding = cache._host_left_padding
    result._host_lengths = cache._host_lengths
    result._checkpoints = list(cache._checkpoints)
    result._rollbacks = deque()
    result.speculating = False
    return result


@dataclass
class SegmentedPhysicalPromotionTicket:
    state: SegmentedSelfMTPState
    view: SelfMTPCachePair
    stream: Any
    reserve_tail: int
    signature: tuple
    physical: SelfMTPCachePair
    target_bases: tuple[int | None, ...]
    draft_bases: tuple[int | None, ...]
    creator_thread: int
    queued_ns: int
    reserved_bytes: int
    _finished: bool = False

    def cancel_and_drain(self) -> None:
        """Retire an unpublished candidate after all queued work completes."""

        if self._finished:
            return
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion must drain on the thread that queued its Metal work"
            )
        if self.stream is not None:
            mx.synchronize(self.stream)
        self._finished = True
        # Do not let a scheduler/debug reference to the retired ticket pin the
        # unused physical candidate or duplicate ownership of the live B1 view.
        self.physical = None
        self.view = None
        self.state = None

    def finish(self) -> tuple[BatchedSelfMTPState, SegmentedPhysicalPromotionReceipt]:
        """Validate the committed state, patch it, and consume the B1 cohort."""

        started = time.perf_counter_ns()
        if self._finished:
            raise RuntimeError("physical promotion ticket was already finished")
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion must finish on the thread that queued its Metal work"
            )
        state = self.state
        if state.proposal_open or state._open_proposal is not None:
            raise SegmentedPhysicalPromotionDeclined(
                "first segmented proposal is not closed"
            )
        if state._batched_state is not None or state._transaction_branches:
            raise SegmentedPhysicalPromotionDeclined(
                "segmented proposal still owns compute or transaction branches"
            )
        epoch, uids, pairs, transactions, generations = self.signature
        if state.membership_epoch != epoch or tuple(l.uid for l in state.lanes) != uids:
            raise SegmentedPhysicalPromotionDeclined(
                "membership epoch or lane UID order changed"
            )
        if tuple(id(p) for p in state.row_caches) != pairs or tuple(
            id(t) for t in state.transactions
        ) != transactions:
            raise SegmentedPhysicalPromotionDeclined("row ownership changed")
        if state._segmented_caches is not self.view:
            raise SegmentedPhysicalPromotionDeclined(
                "retained segmented compute view was replaced"
            )
        advances = []
        for transaction, (generation, position) in zip(
            state.transactions, generations
        ):
            stats = transaction.lineage.stats()
            advance = transaction.position - position
            if stats["closed"] or stats["generation"] != generation + 1 or advance <= 0:
                raise SegmentedPhysicalPromotionDeclined(
                    "transaction generation did not advance exactly once"
                )
            advances.append(advance)
        if len(set(advances)) != 1 or advances[0] > self.reserve_tail:
            raise SegmentedPhysicalPromotionDeclined(
                "first-commit target advances must be equal and fit the reserved tail"
            )
        advance = advances[0]

        patched_bytes = 0
        recurrent_reused = 0
        target = []
        draft = []
        context = mx.stream(self.stream) if self.stream is not None else nullcontext()
        with context:
            for source, candidate, base in zip(
                self.view.target, self.physical.target, self.target_bases
            ):
                if isinstance(source, SegmentedBatchQSAKVCache):
                    stop = base + advance
                    for i, row in enumerate(source.rows):
                        if _offset(row) != stop:
                            raise SegmentedPhysicalPromotionDeclined(
                                "QSA target offset disagrees with committed advance"
                            )
                        candidate.keys[i : i + 1, :, base:stop, :] = row.keys[
                            ..., base:stop, :
                        ]
                        candidate.values[i : i + 1, :, base:stop, :] = row.values[
                            ..., base:stop, :
                        ]
                        candidate.index_keys[i : i + 1, base:stop] = row.index_keys[
                            :, base:stop
                        ]
                    candidate._idx = stop
                    # Match the ordinary physical path exactly.  Its first
                    # update leaves one spare 256-token KV slab, but
                    # ``update_index_keys`` concatenates a cursor-width raw-key
                    # ledger.  Retaining the reservation tail here changes the
                    # B2 row stride seen by QSA on every later decode cycle.
                    # This is a much smaller copy than re-coalescing K/V and it
                    # remains queued on the promotion stream.
                    candidate.index_keys = mx.contiguous(
                        candidate.index_keys[:, :stop]
                    )
                    candidate.offset = mx.array([stop] * len(source.rows))
                    candidate.left_padding = mx.zeros(
                        (len(source.rows),), dtype=mx.int32
                    )
                    pooled, identity = _qsa_merge_summaries(
                        source.rows, [stop] * len(source.rows)
                    )
                    candidate._qsa_pooled_keys = pooled
                    candidate._qsa_pooled_ratio = (
                        None
                        if pooled is None
                        else int(source.rows[0]._qsa_pooled_ratio)
                    )
                    candidate._qsa_summary_identity = identity
                    patched_bytes += sum(
                        int(row.keys[..., base:stop, :].nbytes)
                        + int(row.values[..., base:stop, :].nbytes)
                        + int(row.index_keys[:, base:stop].nbytes)
                        for row in source.rows
                    )
                    target.append(candidate)
                else:
                    unwrapped = _unwrap_recurrent(source)
                    recurrent_reused += sum(v is not None for v in source.cache)
                    target.append(unwrapped)

            for source, candidate, base in zip(
                self.view.draft, self.physical.draft, self.draft_bases
            ):
                if isinstance(source, SegmentedBatchQSAKVCache):
                    if any(_offset(row) != base for row in source.rows):
                        raise SegmentedPhysicalPromotionDeclined(
                            "draft QSA changed across the first committed cycle"
                        )
                    pooled, identity = _qsa_merge_summaries(
                        source.rows, [base] * len(source.rows)
                    )
                    candidate._qsa_pooled_keys = pooled
                    candidate._qsa_pooled_ratio = (
                        None
                        if pooled is None
                        else int(source.rows[0]._qsa_pooled_ratio)
                    )
                    candidate._qsa_summary_identity = identity
                    draft.append(candidate)
                else:
                    unwrapped = _unwrap_recurrent(source)
                    recurrent_reused += sum(v is not None for v in source.cache)
                    draft.append(unwrapped)
            mx.async_eval(
                *[
                    value
                    for group in (target, draft)
                    for cache in group
                    for value in getattr(cache, "cache", ())
                    if value is not None
                ],
                *[
                    value
                    for group in (target, draft)
                    for cache in group
                    for value in (
                        getattr(cache, "keys", None),
                        getattr(cache, "values", None),
                        getattr(cache, "index_keys", None),
                    )
                    if value is not None
                ],
            )
        wait_started = time.perf_counter_ns()
        if self.stream is not None:
            mx.synchronize(self.stream)
        stream_wait_ns = time.perf_counter_ns() - wait_started

        pair = SelfMTPCachePair(target=target, draft=draft)
        # Prepare every destination layer before crossing the ownership
        # boundary. A failure here leaves the committed B1 cohort authoritative.
        started_destinations = []
        try:
            for cache in pair.target:
                cache.start_speculation()
                started_destinations.append(cache)
        except BaseException:
            for cache in started_destinations:
                try:
                    cache.stop_speculation()
                except BaseException:
                    pass
            raise

        lanes = list(state.lanes)
        result = BatchedSelfMTPState(lanes, pair, state.membership_epoch)

        # Publication is the point of no return. The old segmented rows must
        # never be presented as usable after this point, even if a best-effort
        # retirement hook fails.
        row_pairs_to_close = list(state.row_caches)
        transactions_to_close = list(state.transactions)
        state.lanes.clear()
        state.row_caches.clear()
        state.transactions.clear()
        state._segmented_caches = None
        state._batched_state = None
        state._transaction_branches.clear()
        state.poisoned = True
        state.poison_reason = "consumed by segmented physical promotion"
        self._finished = True

        cleanup_errors = []
        for row_pair in row_pairs_to_close:
            for cache in row_pair.target:
                try:
                    cache.stop_speculation()
                except BaseException as error:  # cleanup must reach every layer
                    cleanup_errors.append(error)
        for transaction in transactions_to_close:
            try:
                transaction.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            logging.warning(
                "physical promotion published with %d old-row cleanup errors; "
                "the replacement physical state remains authoritative",
                len(cleanup_errors),
            )
        receipt = SegmentedPhysicalPromotionReceipt(
            queued_ns=self.queued_ns,
            finish_ns=time.perf_counter_ns() - started,
            stream_wait_ns=stream_wait_ns,
            reserved_bytes=self.reserved_bytes,
            patched_bytes=patched_bytes,
            recurrent_arrays_reused=recurrent_reused,
            cleanup_error_count=len(cleanup_errors),
            rows=len(lanes),
            layers=len(target) + len(draft),
            advance=advance,
        )
        # The result now owns the physical pair and the segmented source was
        # consumed.  Sever the ticket's strong references immediately: callers
        # may retain receipts/tickets for diagnostics, and doing so must not
        # keep either cache representation resident through later decode.
        self.physical = None
        self.view = None
        self.state = None
        return result, receipt


def begin_segmented_physical_promotion(
    state: SegmentedSelfMTPState,
    *,
    reserve_tail: int,
    stream: Any,
    note: Callable[[str, int], None] | None = None,
) -> SegmentedPhysicalPromotionTicket:
    """Queue prefix-only physical formation without touching the B1 owners."""

    reserve_tail = int(reserve_tail)
    if reserve_tail <= 0:
        raise ValueError("reserve_tail must be positive")
    view = ensure_segmented_compute_view(state, note=note)
    signature = _state_signature(state)
    started = time.perf_counter_ns()
    target, draft = [], []
    target_bases, draft_bases = [], []
    reserved_bytes = 0
    with mx.stream(stream):
        for cache in view.target:
            if isinstance(cache, SegmentedBatchQSAKVCache):
                physical, base = _reserve_qsa(cache, reserve_tail)
                target.append(physical)
                target_bases.append(base)
                reserved_bytes += int(physical.keys.nbytes + physical.values.nbytes)
                reserved_bytes += int(physical.index_keys.nbytes)
            else:
                target.append(None)
                target_bases.append(None)
        for cache in view.draft:
            if isinstance(cache, SegmentedBatchQSAKVCache):
                physical, base = _reserve_qsa(cache, 0)
                draft.append(physical)
                draft_bases.append(base)
                reserved_bytes += int(physical.keys.nbytes + physical.values.nbytes)
                reserved_bytes += int(physical.index_keys.nbytes)
            else:
                draft.append(None)
                draft_bases.append(None)
        mx.async_eval(
            *[
                value
                for group in (target, draft)
                for cache in group
                if cache is not None
                for value in (cache.keys, cache.values, cache.index_keys)
            ]
        )
    return SegmentedPhysicalPromotionTicket(
        state=state,
        view=view,
        stream=stream,
        reserve_tail=reserve_tail,
        signature=signature,
        physical=SelfMTPCachePair(target, draft),
        target_bases=tuple(target_bases),
        draft_bases=tuple(draft_bases),
        creator_thread=threading.get_ident(),
        queued_ns=time.perf_counter_ns() - started,
        reserved_bytes=reserved_bytes,
    )


__all__ = [
    "SegmentedPhysicalPromotionDeclined",
    "SegmentedPhysicalPromotionReceipt",
    "SegmentedPhysicalPromotionTicket",
    "begin_segmented_physical_promotion",
    "ensure_segmented_compute_view",
]
