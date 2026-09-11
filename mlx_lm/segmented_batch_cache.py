# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Batched compute views over request-private self-MTP caches.

The scheduler owns one concrete B1 cache lineage per request.  These adapters
present those lineages to Qwen4 as one batch for a model invocation, while all
persistent writes still land in the B1 caches. Historical QSA tensors are
never joined. Recurrent tensors use one coherent B2 compute view retained for
the cohort; every write and divergent rollback is reflected back to the rows.
"""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from .models.cache import ArraysCache, KVCache
from .models.qwen4_exp import BatchQSAKVCache, QSAKVCache, Qwen4ArraysCache


class SegmentedBatchUnsupported(TypeError):
    """The row caches cannot be represented by the first batched consumer."""


def _host_offset(cache: Any) -> int:
    offset = getattr(cache, "offset", None)
    if isinstance(offset, int):
        return offset
    size = getattr(cache, "size", None)
    if callable(size):
        value = size()
        if isinstance(value, int):
            return value
    raise SegmentedBatchUnsupported(
        f"{type(cache).__name__} has no host-readable scalar offset"
    )


def _pad_sequence(value: mx.array, left: int, right: int, axis: int) -> mx.array:
    padding = [(0, 0)] * value.ndim
    padding[axis] = (int(left), int(right))
    return value if not left and not right else mx.pad(value, padding)


def _slice_row_tree(values, row: int):
    return [
        None if value is None else mx.contiguous(value[row : row + 1])
        for value in values
    ]


class SegmentedBatchQSAKVCache(BatchQSAKVCache):
    """QSA batch ABI backed by independent, unquantized B1 QSA caches."""

    def __init__(self, rows: Sequence[QSAKVCache], note=None):
        if not rows or not all(type(row) is QSAKVCache for row in rows):
            names = ", ".join(type(row).__name__ for row in rows)
            raise SegmentedBatchUnsupported(
                "true batched QSA initially requires plain QSAKVCache rows; "
                f"got {names or 'none'}"
            )
        self.rows = list(rows)
        self._note = note
        self._step_lengths = None
        self._right_padding = None
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None
        self._qsa_summary_identity = None
        self._qsa_summary_restored = False
        self._qsa_pending_pooled = None
        self.index_keys = None
        self.keys = None
        self.values = None
        self._configure_attention_backend("sdpa")
        self._refresh_geometry()

    def _bump(self, key, amount=1):
        if self._note is not None:
            self._note(key, amount)

    def _refresh_geometry(self):
        lengths = [_host_offset(row) for row in self.rows]
        width = max(lengths, default=0)
        self._base_lengths = lengths
        self._base_width = width
        self._idx = width
        self.offset = mx.array(lengths)
        self.left_padding = mx.array([width - value for value in lengths])
        self.keys = None
        self.values = None
        self.index_keys = None

    @property
    def batch_size(self):
        return len(self.rows)

    def prepare(self, *, lengths=None, right_padding=None, **_kwargs):
        if lengths is None:
            raise ValueError("segmented QSA prepare requires per-row lengths")
        lengths = [int(value) for value in lengths]
        if len(lengths) != len(self.rows) or any(value < 0 for value in lengths):
            raise ValueError("segmented QSA lengths must cover every row")
        self._refresh_geometry()
        self._step_lengths = lengths
        right_padding = (
            [0] * len(lengths)
            if right_padding is None
            else [int(value) for value in right_padding]
        )
        if len(right_padding) != len(lengths):
            raise ValueError("segmented QSA right padding must cover every row")
        width = max(lengths, default=0)
        if any(width - value != pad for value, pad in zip(lengths, right_padding)):
            raise ValueError("segmented QSA lengths/right padding disagree")
        self._right_padding = mx.array(right_padding) if any(right_padding) else None

    def prepare_self_mtp_step(self, **kwargs):
        share = self._mtp_share_topk
        shared = self._mtp_shared_topk
        self.prepare(**kwargs)
        self._mtp_share_topk = share
        self._mtp_shared_topk = shared

    def segmented_attention(self, attention, hidden: mx.array, _mask):
        """Run QSA against each B1 history without joining historical K/V.

        The surrounding decoder layer remains batch-shaped, so its projection-
        independent trunk, recurrent layer and MoE stream weights once.  This
        first exact consumer deliberately invokes the QSA sublayer per row;
        a future segment-aware attention kernel can fuse those reductions
        without changing the cache contract.
        """
        if self._step_lengths is None:
            raise RuntimeError("segmented attention outside prepare/finalize")
        width = int(hidden.shape[1])
        projected = attention._project_segmented_qsa(hidden)
        outputs = []
        gates = []
        for index, (row, valid) in enumerate(zip(self.rows, self._step_lengths)):
            if valid == 0:
                outputs.append(mx.zeros_like(hidden[index : index + 1]))
                gates.append(mx.zeros_like(hidden[index : index + 1]))
                continue
            row_hidden = hidden[index : index + 1, :valid]
            row_mask = row.make_mask(
                valid, return_array=True, window_size=None
            )
            if row_mask is not None and row_mask.ndim == 2:
                row_mask = row_mask[None, None]
            row_projected = tuple(
                value[index : index + 1, :valid] for value in projected
            )
            output, gate = attention(
                row_hidden,
                row_mask,
                row,
                _projected=row_projected,
                _return_pre_o=True,
            )
            outputs.append(_pad_sequence(output, 0, width - valid, 1))
            gates.append(_pad_sequence(gate, 0, width - valid, 1))
        self._bump("segmented_attention_calls")
        self._bump("independent_lineages_consumed", len(self.rows))
        self._bump("row_state_splits", len(self.rows))
        self._refresh_geometry()
        output = mx.concatenate(outputs, axis=0)
        gate = mx.concatenate(gates, axis=0)
        return attention.o_proj(output * mx.sigmoid(gate))

    def update_index_keys(self, keys: mx.array):
        del keys
        raise RuntimeError(
            "segmented QSA forbids dense index-ledger materialization; "
            "Attention.segmented_attention must consume request-private rows"
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        del keys, values
        raise RuntimeError(
            "segmented QSA forbids dense K/V materialization; "
            "Attention.segmented_attention must consume request-private rows"
        )

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        from .models.base import create_causal_mask

        del return_array
        return create_causal_mask(
            N, offset=self._base_width, left_padding=self.left_padding, **kwargs
        )

    def last_valid_query(self, values: mx.array) -> mx.array:
        if self._right_padding is None:
            return values[:, -1]
        rows = mx.arange(values.shape[0], dtype=mx.int32)
        positions = values.shape[1] - self._right_padding.astype(mx.int32) - 1
        return values[rows, positions]

    def max_left_padding(self) -> int:
        return max(
            (self._base_width - value for value in self._base_lengths),
            default=0,
        )

    def release_qsa_cycle(self, _who: str, *, keep_shared=False, **_kwargs):
        shared = self._mtp_shared_topk if keep_shared else None
        share = self._mtp_share_topk if keep_shared else False
        self._mtp_share_topk = share
        self._mtp_shared_topk = shared
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None

    def finalize(self):
        self._step_lengths = None
        self._right_padding = None
        self.release_qsa_cycle("SegmentedBatchQSAKVCache.finalize")
        self._refresh_geometry()

    def finalize_self_mtp_step(self):
        share = self._mtp_share_topk
        shared = self._mtp_shared_topk
        self._step_lengths = None
        self._right_padding = None
        self._refresh_geometry()
        self._mtp_share_topk = share
        self._mtp_shared_topk = shared

    def supports_ragged_trim(self):
        return True

    def preflight_ragged_trim(self, counts, *, validate=True):
        counts = [int(value) for value in counts]
        if len(counts) != len(self.rows):
            raise ValueError("segmented QSA trim must cover every row")
        for row, count in zip(self.rows, counts):
            if validate and count > _host_offset(row):
                raise ValueError("segmented QSA trim exceeds a row offset")
        return counts

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            if count:
                row.trim(count)
        self.release_qsa_cycle("SegmentedBatchQSAKVCache.trim_ragged")
        self._refresh_geometry()
        return counts

    def trim(self, count):
        count = int(count)
        self.trim_ragged([count] * len(self.rows))
        return count

    def is_trimmable(self):
        return all(row.is_trimmable() for row in self.rows)

    def empty(self):
        return all(row.empty() for row in self.rows)

    @property
    def nbytes(self):
        return 0


class SegmentedBatchArraysCache(Qwen4ArraysCache):
    """Coherent recurrent-state compute view whose writes split into B1 rows."""

    def __init__(self, rows: Sequence[ArraysCache], note=None):
        if not rows or not all(isinstance(row, ArraysCache) for row in rows):
            names = ", ".join(type(row).__name__ for row in rows)
            raise SegmentedBatchUnsupported(
                f"true batched recurrent state requires ArraysCache rows; got {names}"
            )
        sizes = {len(row.cache) for row in rows}
        if len(sizes) != 1:
            raise SegmentedBatchUnsupported("segmented recurrent slot counts differ")
        super().__init__(sizes.pop())
        self.rows = list(rows)
        self._note = note
        self.speculating = True
        self._refresh_state()

    def _bump(self, key, amount=1):
        if self._note is not None:
            self._note(key, amount)

    def _refresh_state(self):
        joined = []
        for slot in range(len(self.cache)):
            values = [row[slot] for row in self.rows]
            present = [value for value in values if value is not None]
            if not present:
                joined.append(None)
                continue
            if len(present) != len(values):
                raise SegmentedBatchUnsupported(
                    f"recurrent slot {slot} is initialized for only some rows"
                )
            shape = present[0].shape[1:]
            if any(value.shape[1:] != shape for value in present):
                raise SegmentedBatchUnsupported(
                    f"recurrent slot {slot} row geometries differ"
                )
            joined_value = mx.concatenate(values, axis=0)
            joined.append(joined_value)
            self._bump("recurrent_state_materializations")
            self._bump("recurrent_state_materialized_bytes", joined_value.nbytes)
        self.cache = joined

    def __setitem__(self, idx, value):
        self.cache[idx] = value
        if value is None:
            for row in self.rows:
                row[idx] = None
        else:
            if value.shape[0] != len(self.rows):
                raise ValueError("segmented recurrent write has wrong batch size")
            for index, row in enumerate(self.rows):
                row[idx] = mx.contiguous(value[index : index + 1])
        self._bump("row_state_splits", len(self.rows))

    @property
    def batch_size(self):
        return len(self.rows)

    def prepare(self, lengths=None, **kwargs):
        del kwargs
        if lengths is None or len(lengths) != len(self.rows):
            raise ValueError("segmented recurrent prepare must cover every row")
        super().prepare(lengths=lengths)
        for row, length in zip(self.rows, lengths):
            row.prepare(lengths=[int(length)])

    def finalize(self):
        first_error = None
        for row in self.rows:
            try:
                row.finalize()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        super().finalize()
        if first_error is not None:
            raise first_error

    def advance(self, N):
        # Row caches own the persistent states; only the joined view's padding
        # metadata needs advancing during this invocation.
        return ArraysCache.advance(self, N)

    def _row_closure(self, fn, row):
        if fn is None:
            return None

        def sliced(value, _fn=fn, _row=row):
            return _slice_row_tree(_fn(value), _row)

        return sliced

    def stage_ple_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        del per_row_fn
        for index, row in enumerate(self.rows):
            stage = getattr(row, "stage_ple_rollback", None)
            if stage is not None:
                stage(
                    num_tokens,
                    self._row_closure(fn, index),
                    _slice_row_tree(snapshot, index),
                )

    def record_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None, **_kwargs):
        del per_row_fn
        for index, row in enumerate(self.rows):
            row.record_rollback(
                num_tokens,
                self._row_closure(fn, index),
                _slice_row_tree(snapshot, index),
            )

    def supports_ragged_trim(self):
        return True

    def preflight_ragged_trim(self, counts, *, validate=True):
        counts = [int(value) for value in counts]
        if len(counts) != len(self.rows):
            raise ValueError("segmented recurrent trim must cover every row")
        for row, count in zip(self.rows, counts):
            row.preflight_ragged_trim([count], validate=validate)
        return counts

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            row.trim_ragged([count], validate=False)
        self._refresh_state()
        return counts

    def trim(self, count):
        count = int(count)
        self.trim_ragged([count] * len(self.rows))
        return count

    def is_trimmable(self):
        return all(row.is_trimmable() for row in self.rows)

    def empty(self):
        return all(row.empty() for row in self.rows)

    @property
    def nbytes(self):
        return sum(
            int(getattr(value, "nbytes", 0))
            for value in self.cache
            if value is not None
        )


def build_segmented_batch_cache_group(groups, *, note=None):
    """Transpose B1 cache groups into per-layer batched compute adapters."""

    groups = [list(group) for group in groups]
    if not groups:
        return []
    widths = {len(group) for group in groups}
    if len(widths) != 1:
        raise SegmentedBatchUnsupported("segmented cache groups have different layers")
    result = []
    for layer_rows in zip(*groups):
        first = layer_rows[0]
        if isinstance(first, ArraysCache):
            result.append(SegmentedBatchArraysCache(layer_rows, note=note))
        elif isinstance(first, QSAKVCache):
            result.append(SegmentedBatchQSAKVCache(layer_rows, note=note))
        elif isinstance(first, KVCache):
            raise SegmentedBatchUnsupported(
                "plain KV segmented batching is not needed by Qwen4 and is not "
                "yet qualified"
            )
        else:
            raise SegmentedBatchUnsupported(
                f"unsupported segmented cache layer {type(first).__name__}"
            )
    return result


def build_segmented_batch_cache_pair(row_pairs, *, note=None):
    from .hybrid_speculative import SelfMTPCachePair

    return SelfMTPCachePair(
        target=build_segmented_batch_cache_group(
            [pair.target for pair in row_pairs], note=note
        ),
        draft=build_segmented_batch_cache_group(
            [pair.draft for pair in row_pairs], note=note
        ),
    )


__all__ = [
    "SegmentedBatchArraysCache",
    "SegmentedBatchQSAKVCache",
    "SegmentedBatchUnsupported",
    "build_segmented_batch_cache_pair",
]
