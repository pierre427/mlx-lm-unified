"""Exact recurrent-prefix fan-out from an ArraysCache rollback record.

This module is an opt-in prototype.  It turns the latest exact GDN rollback
record into a frozen prefix boundary, materializes that boundary once, and
copies it into two independent batch rows.  The target verifier still decides
how many suffix tokens each row accepts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx

from .models.cache import ArraysCache


_STAT_KEYS = (
    "requests",
    "declined_disabled",
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
        record,
        *,
        retained_input_bytes: int,
    ):
        self._cache_type = cache_type
        self._state_size = state_size
        self._record = record
        self._parent = tuple(record.snapshot)
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
        if not cache.speculating or not cache._rollbacks:
            raise ValueError("the cache has no live exact rollback record")
        if (
            isinstance(retained_input_bytes, bool)
            or not isinstance(retained_input_bytes, int)
            or retained_input_bytes < 0
        ):
            raise ValueError("retained_input_bytes must be non-negative")
        record = cache._rollbacks[-1]
        if record.depths is not None:
            raise ValueError("the producer record must have one uniform span")
        parent = list(record.snapshot)
        if len(parent) != len(cache.cache):
            raise RuntimeError("rollback record does not cover the full cache")
        if any(
            value is not None and value.shape[0] != 1 for value in parent
        ):
            raise ValueError("the shared-prefix parent must have one row")

        _STATS["armed"] += 1
        _STATS["retained_input_bytes"] += int(retained_input_bytes)
        return cls(
            type(cache),
            len(cache.cache),
            record,
            retained_input_bytes=int(retained_input_bytes),
        )

    @property
    def span(self) -> int:
        self._require_open()
        return int(self._record.num_tokens)

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
        state = (
            list(self._parent)
            if boundary == 0
            else list(self._record.fn(boundary))
        )
        if len(state) != self._state_size:
            raise RuntimeError(
                f"materialized {len(state)} entries, expected {self._state_size}"
            )
        arrays = [value for value in state if value is not None]
        if arrays:
            mx.eval(*arrays)
        cache = self._cache_type(self._state_size)
        cache.cache = state
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
        self._record = None
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


__all__ = [
    "GDNFanoutLease",
    "GDNPrefixFanout",
    "gdn_prefix_fanout_stats",
]
