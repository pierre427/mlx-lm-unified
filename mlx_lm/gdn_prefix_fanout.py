"""Exact N=2 prefix fan-out for hybrid Qwen caches.

The recurrent boundary comes from the public ``ArraysCache`` rollback API.
Attention KV, QSA's raw-key and pooled-key ledgers, and Qwen4's paired PLE
state travel through the same transaction.  The feature stays opt-in until a
real-model wall-clock gate promotes it.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx

from .models.cache import ArraysCache, trim_ragged_prompt_cache


_STAT_KEYS = (
    "requests",
    "declined_disabled",
    "declined_no_record",
    "declined_span_mismatch",
    "declined_unsupported_cache",
    "armed",
    "materializations",
    "replay_tokens",
    "retained_input_bytes",
    "fanout_batches",
    "fanout_rows",
    "copied_state_bytes",
    "committed_rows",
    "accepted_zero",
    "accepted_partial",
    "accepted_all",
    "aborts",
    "cleanups",
    "hybrid_armed",
    "hybrid_materializations",
    "hybrid_fanout_batches",
    "hybrid_fanout_rows",
    "hybrid_tip_fanout_batches",
    "hybrid_tip_fanout_rows",
    "hybrid_committed_rows",
    "hybrid_aborts",
    "serving_requests",
    "serving_engaged",
    "serving_declined_not_n2",
    "serving_declined_cache",
    "serving_declined_error",
    "serving_cleanups",
)
_STATS = {key: 0 for key in _STAT_KEYS}


def gdn_prefix_fanout_stats(reset: bool = False) -> Dict[str, int]:
    """Return bounded host counters without evaluating any device array."""

    stats = dict(_STATS)
    if reset:
        for key in _STAT_KEYS:
            _STATS[key] = 0
    return stats


def _array_bytes(values: Sequence[Any]) -> int:
    return sum(int(getattr(value, "nbytes", 0)) for value in values)


def gdn_prefix_fanout_enabled(value: Optional[bool] = None) -> bool:
    """Resolve the serving gate; absent means default-off."""

    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_GDN_PREFIX_FANOUT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _note_serving_event(event: str) -> None:
    """Increment one host-only serving counter without touching device state."""

    key = f"serving_{event}"
    if key not in _STATS:
        raise ValueError(f"unknown GDN prefix fan-out serving event: {event}")
    _STATS[key] += 1


def _tree_copy(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_tree_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_copy(item) for item in value)
    if isinstance(value, dict):
        return {key: _tree_copy(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _tree_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tree_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tree_arrays(item)


def _clone_cache(cache):
    clone = type(cache).from_state(
        _tree_copy(cache.state), copy.deepcopy(cache.meta_state)
    )
    arrays = list(_tree_arrays(clone.state))
    if arrays:
        mx.eval(*arrays)
    return clone


class GDNPrefixFanout:
    """Freeze one exact rollback record and materialize an N=2 boundary.

    ``ArraysCache.record_rollback`` already owns the exact replay closure for
    GDN and, for Qwen4, its paired PLE state.  Reusing that record prevents a
    second definition of the recurrent boundary.  The record must come from a
    single-row forward; multi-row rollback records have per-row histories and
    are not a shared-prefix producer.
    """

    def __init__(
        self,
        cache_type,
        state_size: int,
        boundary,
        *,
        retained_input_bytes: int,
    ):
        self._cache_type = cache_type
        self._state_size = state_size
        self._boundary = boundary
        self._parent = tuple(boundary.snapshot)
        self._retained_input_bytes = retained_input_bytes
        self._active_lease: Optional[GDNFanoutLease] = None
        self._closed = False

    @classmethod
    def from_latest_record(
        cls,
        cache: ArraysCache,
        *,
        enabled: bool = False,
        retained_input_bytes: int = 0,
    ) -> Optional["GDNPrefixFanout"]:
        """Arm from the newest exact record, or return ``None`` when off."""

        _STATS["requests"] += 1
        if not enabled:
            _STATS["declined_disabled"] += 1
            return None
        if not isinstance(cache, ArraysCache):
            raise TypeError("GDN prefix fan-out needs an ArraysCache")
        if cache.batch_size != 1:
            raise ValueError("the shared-prefix producer must have one row")
        if (
            isinstance(retained_input_bytes, bool)
            or not isinstance(retained_input_bytes, int)
            or retained_input_bytes < 0
        ):
            raise ValueError("retained_input_bytes must be non-negative")
        try:
            boundary = cache.latest_exact_rollback_boundary()
        except (RuntimeError, ValueError):
            _STATS["declined_no_record"] += 1
            raise
        parent = list(boundary.snapshot)
        if any(
            value is not None and value.shape[0] != 1 for value in parent
        ):
            raise ValueError("the shared-prefix parent must have one row")

        _STATS["armed"] += 1
        _STATS["retained_input_bytes"] += int(retained_input_bytes)
        return cls(
            type(cache),
            len(cache.cache),
            boundary,
            retained_input_bytes=int(retained_input_bytes),
        )

    @property
    def span(self) -> int:
        self._require_open()
        return int(self._boundary.num_tokens)

    @property
    def parent_nbytes(self) -> int:
        self._require_open()
        return _array_bytes(self._parent)

    @property
    def retained_input_bytes(self) -> int:
        self._require_open()
        return self._retained_input_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("GDN prefix fan-out is closed")

    def _materialize(self, boundary: int) -> ArraysCache:
        self._require_open()
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise TypeError("boundary must be an integer")
        if not 0 <= boundary <= self.span:
            raise ValueError(
                f"boundary {boundary} is outside the retained span 0..{self.span}"
            )
        cache = self._boundary.materialize(boundary)
        _STATS["materializations"] += 1
        _STATS["replay_tokens"] += boundary
        return cache

    def fork(self, boundary: int, rows: int = 2) -> "GDNFanoutLease":
        """Materialize one exact boundary and copy it into two private rows."""

        self._require_open()
        if rows != 2:
            raise ValueError("this prototype is qualified only for rows=2")
        if self._active_lease is not None and not self._active_lease.closed:
            raise RuntimeError("close the active fan-out lease before another fork")
        boundary_cache = self._materialize(boundary)
        batch = self._cache_type.merge([boundary_cache, boundary_cache])
        arrays = [value for value in batch.cache if value is not None]
        if arrays:
            mx.eval(*arrays)
        copied = _array_bytes(batch.cache)
        _STATS["fanout_batches"] += 1
        _STATS["fanout_rows"] += rows
        _STATS["copied_state_bytes"] += copied
        lease = GDNFanoutLease(self, batch, rows, boundary, copied)
        self._active_lease = lease
        return lease

    def close(self) -> None:
        if self._closed:
            return
        if self._active_lease is not None:
            self._active_lease.abort()
        self._boundary = None
        self._parent = ()
        self._active_lease = None
        self._closed = True
        _STATS["cleanups"] += 1


class GDNFanoutLease:
    """Two descendant rows whose accepted lengths come from the verifier."""

    def __init__(self, owner, cache, rows, boundary, copied_state_bytes):
        self._owner = owner
        self.cache: Optional[ArraysCache] = cache
        self.rows = rows
        self.boundary = boundary
        self.copied_state_bytes = copied_state_bytes
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> ArraysCache:
        if self._closed or self.cache is None:
            raise RuntimeError("GDN fan-out lease is closed")
        return self.cache

    def start_speculation(self, rollback_window: Optional[int] = None) -> None:
        self._require_open().start_speculation(rollback_window)

    def commit(
        self,
        proposed_tokens: int,
        accepted_lengths: Sequence[int],
    ) -> List[ArraysCache]:
        """Rewind each row to the target-accepted prefix and detach it."""

        cache = self._require_open()
        if (
            isinstance(proposed_tokens, bool)
            or not isinstance(proposed_tokens, int)
            or proposed_tokens < 0
        ):
            raise ValueError("proposed_tokens must be non-negative")
        accepted = list(accepted_lengths)
        if len(accepted) != self.rows:
            raise ValueError(
                f"accepted_lengths has {len(accepted)} rows, expected {self.rows}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > proposed_tokens
            for value in accepted
        ):
            raise ValueError(
                f"accepted lengths must be in 0..{proposed_tokens}: {accepted}"
            )
        if not cache.speculating:
            raise RuntimeError("start_speculation() before committing descendants")

        cache.trim_ragged(
            [proposed_tokens - value for value in accepted]
        )
        cache.stop_speculation()
        descendants = [cache.extract(index) for index in range(self.rows)]
        arrays = [
            value
            for descendant in descendants
            for value in descendant.cache
            if value is not None
        ]
        if arrays:
            mx.eval(*arrays)
        _STATS["committed_rows"] += self.rows
        _STATS["accepted_zero"] += sum(value == 0 for value in accepted)
        _STATS["accepted_all"] += sum(
            value == proposed_tokens for value in accepted
        )
        _STATS["accepted_partial"] += sum(
            0 < value < proposed_tokens for value in accepted
        )
        self._cleanup()
        return descendants

    def abort(self) -> None:
        if self._closed:
            return
        _STATS["aborts"] += 1
        self._cleanup()

    def _cleanup(self) -> None:
        self.cache = None
        self._closed = True
        if self._owner._active_lease is self:
            self._owner._active_lease = None
        self._owner = None
        _STATS["cleanups"] += 1


class HybridCachePrefixFanout:
    """One immutable hybrid prefix with an exact two-row transaction.

    Every recurrent layer must export the same uniform rollback span. Regular
    KV and QSA caches are copied at their live end and trimmed to the selected
    boundary. ``Qwen4ArraysCache`` exports PLE and GDN as one recurrent handle,
    while QSA's ``state``/``meta_state`` copy preserves both auxiliary ledgers.
    """

    def __init__(self, caches, recurrent, span: int):
        self._source = tuple(caches)
        self._recurrent = dict(recurrent)
        self._span = int(span)
        self._active_lease: Optional[HybridCacheFanoutLease] = None
        self._closed = False

    @classmethod
    def from_prompt_cache(
        cls,
        caches: Sequence[Any],
        *,
        enabled: Optional[bool] = None,
        strict: bool = False,
    ) -> Optional["HybridCachePrefixFanout"]:
        _STATS["requests"] += 1
        if not gdn_prefix_fanout_enabled(enabled):
            _STATS["declined_disabled"] += 1
            return None
        caches = list(caches)
        recurrent = {}
        spans = set()
        decline_key = None
        try:
            if not caches:
                decline_key = "declined_unsupported_cache"
                raise ValueError("hybrid prefix fan-out needs a non-empty cache")
            for index, cache in enumerate(caches):
                if isinstance(cache, ArraysCache):
                    try:
                        boundary = cache.latest_exact_rollback_boundary()
                    except (ValueError, RuntimeError):
                        decline_key = "declined_no_record"
                        raise
                    recurrent[index] = boundary
                    spans.add(boundary.num_tokens)
                    continue
                if not callable(getattr(cache, "trim", None)) or not callable(
                    getattr(type(cache), "merge", None)
                ):
                    decline_key = "declined_unsupported_cache"
                    raise TypeError(
                        f"cache entry {index} ({type(cache).__name__}) cannot "
                        "copy, trim, and merge"
                    )
            if not recurrent:
                decline_key = "declined_unsupported_cache"
                raise ValueError("hybrid prefix fan-out found no recurrent cache")
            if len(spans) != 1:
                decline_key = "declined_span_mismatch"
                raise ValueError(
                    f"recurrent rollback spans disagree: {sorted(spans)}"
                )
        except (TypeError, ValueError, RuntimeError):
            _STATS[decline_key or "declined_unsupported_cache"] += 1
            if strict:
                raise
            return None

        _STATS["hybrid_armed"] += 1
        return cls(caches, recurrent, spans.pop())

    @property
    def span(self) -> int:
        self._require_open()
        return self._span

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self):
        if self._closed:
            raise RuntimeError("hybrid prefix fan-out is closed")

    def materialize(self, boundary: int) -> List[Any]:
        """Materialize the full KV/QSA/GDN/PLE cache at one boundary."""

        self._require_open()
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise TypeError("boundary must be an integer")
        if not 0 <= boundary <= self._span:
            raise ValueError(f"boundary {boundary} is outside 0..{self._span}")
        drop = self._span - boundary
        result = []
        for index, source in enumerate(self._source):
            if index in self._recurrent:
                cache = self._recurrent[index].materialize(boundary)
            else:
                cache = _clone_cache(source)
                applied = cache.trim(drop)
                if applied != drop:
                    raise RuntimeError(
                        f"cache entry {index} trimmed {applied}, expected {drop}"
                    )
            result.append(cache)
        arrays = [array for cache in result for array in _tree_arrays(cache.state)]
        if arrays:
            mx.eval(*arrays)
        _STATS["hybrid_materializations"] += 1
        return result

    def fork(self, boundary: int, rows: int = 2) -> "HybridCacheFanoutLease":
        self._require_open()
        if rows != 2:
            raise ValueError("hybrid GDN fan-out is qualified only for rows=2")
        if self._active_lease is not None and not self._active_lease.closed:
            raise RuntimeError("close the active hybrid fan-out lease first")
        single = self.materialize(boundary)
        batched = []
        for index, cache in enumerate(single):
            merge = getattr(type(cache), "merge", None)
            if not callable(merge):
                _STATS["declined_unsupported_cache"] += 1
                raise TypeError(
                    f"cache entry {index} ({type(cache).__name__}) cannot merge"
                )
            batched.append(merge([cache, cache]))
        arrays = [array for cache in batched for array in _tree_arrays(cache.state)]
        if arrays:
            mx.eval(*arrays)
        lease = HybridCacheFanoutLease(self, batched, rows, boundary)
        self._active_lease = lease
        _STATS["hybrid_fanout_batches"] += 1
        _STATS["hybrid_fanout_rows"] += rows
        return lease

    def fork_live_tip(self, rows: int = 2) -> "HybridCacheFanoutLease":
        """Build two rows directly from an exclusively owned live tip.

        ``fork`` preserves a reusable immutable parent, which requires cloning
        every non-recurrent cache before the batch merge. Serving has already
        cloned the APC entry into a request-private canonical lane and
        transfers the result immediately, so that second full-prefix clone is
        unnecessary. This one-shot path merges the live tip itself and marks
        the owner consumed. The ordinary immutable path remains the fallback.
        """

        self._require_open()
        if rows != 2:
            raise ValueError("hybrid GDN tip fan-out is qualified only for rows=2")
        if self._active_lease is not None and not self._active_lease.closed:
            raise RuntimeError("close the active hybrid fan-out lease first")
        if not self._source:
            raise RuntimeError("hybrid GDN tip fan-out source was already consumed")

        batched = []
        for index, source in enumerate(self._source):
            merge = getattr(type(source), "merge", None)
            if not callable(merge):
                _STATS["declined_unsupported_cache"] += 1
                raise TypeError(
                    f"cache entry {index} ({type(source).__name__}) cannot merge"
                )
            batched.append(merge([source, source]))
        arrays = [array for cache in batched for array in _tree_arrays(cache.state)]
        if arrays:
            mx.eval(*arrays)
        lease = HybridCacheFanoutLease(self, batched, rows, self._span)
        self._active_lease = lease
        self._source = ()
        self._recurrent = {}
        _STATS["hybrid_fanout_batches"] += 1
        _STATS["hybrid_fanout_rows"] += rows
        _STATS["hybrid_tip_fanout_batches"] += 1
        _STATS["hybrid_tip_fanout_rows"] += rows
        return lease

    def close(self):
        if self._closed:
            return
        if self._active_lease is not None:
            self._active_lease.abort()
        self._source = ()
        self._recurrent = {}
        self._active_lease = None
        self._closed = True
        _STATS["cleanups"] += 1


class HybridCacheFanoutLease:
    """Private batched descendants with exact ragged commit semantics."""

    def __init__(self, owner, caches, rows, boundary):
        self._owner = owner
        self.caches: Optional[List[Any]] = caches
        self.rows = rows
        self.boundary = boundary
        self._speculating = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> List[Any]:
        if self._closed or self.caches is None:
            raise RuntimeError("hybrid GDN fan-out lease is closed")
        return self.caches

    def start_speculation(self, rollback_window: Optional[int] = None):
        caches = self._require_open()
        started = []
        try:
            for cache in caches:
                cache.start_speculation(rollback_window)
                started.append(cache)
        except BaseException:
            for cache in started:
                cache.stop_speculation()
            raise
        self._speculating = True

    def commit(
        self, proposed_tokens: int, accepted_lengths: Sequence[int]
    ) -> List[List[Any]]:
        caches = self._require_open()
        if not self._speculating:
            raise RuntimeError("start_speculation() before committing descendants")
        if (
            isinstance(proposed_tokens, bool)
            or not isinstance(proposed_tokens, int)
            or proposed_tokens < 0
        ):
            raise ValueError("proposed_tokens must be non-negative")
        accepted = list(accepted_lengths)
        if len(accepted) != self.rows or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= proposed_tokens
            for value in accepted
        ):
            raise ValueError(
                f"accepted lengths must have {self.rows} values in "
                f"0..{proposed_tokens}: {accepted}"
            )
        trim_ragged_prompt_cache(
            caches, [proposed_tokens - value for value in accepted]
        )
        for cache in caches:
            cache.stop_speculation()
        descendants = [
            [cache.extract(row) for cache in caches] for row in range(self.rows)
        ]
        arrays = [
            array
            for row in descendants
            for cache in row
            for array in _tree_arrays(cache.state)
        ]
        if arrays:
            mx.eval(*arrays)
        _STATS["hybrid_committed_rows"] += self.rows
        _STATS["committed_rows"] += self.rows
        _STATS["accepted_zero"] += sum(value == 0 for value in accepted)
        _STATS["accepted_all"] += sum(value == proposed_tokens for value in accepted)
        _STATS["accepted_partial"] += sum(
            0 < value < proposed_tokens for value in accepted
        )
        self._cleanup()
        return descendants

    def take_batch(self) -> List[Any]:
        """Transfer an unadvanced batch cache to a serving generator."""

        caches = self._require_open()
        if self._speculating:
            raise RuntimeError("cannot transfer an open speculative transaction")
        self.caches = None
        self._cleanup()
        return caches

    def abort(self):
        if self._closed:
            return
        if self._speculating and self.caches is not None:
            for cache in self.caches:
                cache.stop_speculation()
        _STATS["hybrid_aborts"] += 1
        _STATS["aborts"] += 1
        self._cleanup()

    def _cleanup(self):
        self.caches = None
        self._speculating = False
        self._closed = True
        if self._owner is not None and self._owner._active_lease is self:
            self._owner._active_lease = None
        self._owner = None
        _STATS["cleanups"] += 1


__all__ = [
    "GDNFanoutLease",
    "GDNPrefixFanout",
    "HybridCacheFanoutLease",
    "HybridCachePrefixFanout",
    "gdn_prefix_fanout_enabled",
    "gdn_prefix_fanout_stats",
]
