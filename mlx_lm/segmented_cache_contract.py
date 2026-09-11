# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""A lineage-safe contract for shared-base plus private-delta caches."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Protocol
import uuid


class CacheContractError(RuntimeError):
    pass


class StockConsumerRefused(CacheContractError):
    pass


@dataclass(frozen=True)
class CacheSegment:
    payload: Any
    start: int
    length: int
    layout_id: str
    generation: int
    mutable: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.length < 0 or self.generation < 0:
            raise ValueError(
                "segment coordinates and generation must be non-negative"
            )
        if not self.layout_id:
            raise ValueError("layout id is required")

    @property
    def stop(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class MutationPermit:
    cache_id: str
    owner_id: str
    expected_generation: int
    nonce: str


@dataclass(frozen=True)
class SegmentedCacheView:
    base: CacheSegment
    private_delta: CacheSegment
    generation: int


@dataclass(frozen=True)
class MaterializationReceipt:
    cache_id: str
    source_generation: int
    base_generation: int
    delta_generation: int
    layout_id: str
    explicit: bool = True


@dataclass(frozen=True)
class MaterializedCacheView:
    payload: Any
    receipt: MaterializationReceipt


class CacheConsumer(Protocol):
    supports_segmented_cache: bool


class _LineageAuthority:
    """One lock and one compare-and-swap tip for a cache lineage."""

    def __init__(self, generation: int) -> None:
        self.cache_id = uuid.uuid4().hex
        self._lock = RLock()
        self._current_generation = generation
        self._retired_generations: set[int] = set()
        self._pending: tuple[str, str, int] | None = None

    def assert_current(self, generation: int) -> None:
        with self._lock:
            if (
                generation != self._current_generation
                or generation in self._retired_generations
            ):
                raise CacheContractError("cache generation is retired or stale")

    def while_current(self, generation: int, callback: Callable[[], Any]) -> Any:
        """Run a consumer handoff while the source generation is pinned.

        Materialization is an authoritative read of both cache segments.  Keep
        the lineage lock through that read so a concurrent mutation cannot
        retire the generation between the validity check and the copy.
        """

        with self._lock:
            self.assert_current(generation)
            return callback()

    def begin(self, *, owner_id: str, generation: int) -> MutationPermit:
        with self._lock:
            self.assert_current(generation)
            if self._pending is not None:
                raise CacheContractError("a mutation is already in progress")
            nonce = uuid.uuid4().hex
            self._pending = (nonce, owner_id, generation)
            return MutationPermit(self.cache_id, owner_id, generation, nonce)

    def advance(self, permit: MutationPermit) -> int:
        """Retire the source and publish its sole successor atomically."""
        with self._lock:
            expected = (
                permit.nonce,
                permit.owner_id,
                permit.expected_generation,
            )
            if permit.cache_id != self.cache_id or self._pending != expected:
                raise CacheContractError(
                    "mutation permit is invalid or already consumed"
                )
            self.assert_current(permit.expected_generation)
            source = permit.expected_generation
            successor = source + 1
            self._retired_generations.add(source)
            self._current_generation = successor
            self._pending = None
            return successor

    def cancel(self, permit: MutationPermit) -> None:
        with self._lock:
            expected = (
                permit.nonce,
                permit.owner_id,
                permit.expected_generation,
            )
            if permit.cache_id != self.cache_id or self._pending != expected:
                raise CacheContractError("mutation permit cannot be cancelled")
            self.assert_current(permit.expected_generation)
            self._pending = None


class SegmentedCache:
    """Share an immutable base while keeping one owner's delta private."""

    def __init__(
        self,
        *,
        base: CacheSegment,
        private_delta: CacheSegment,
        owner_id: str,
        generation: int = 0,
        _lineage: _LineageAuthority | None = None,
    ) -> None:
        if base.mutable:
            raise CacheContractError("the shared base must be immutable")
        if not private_delta.mutable:
            raise CacheContractError("the private delta must be mutable")
        if base.layout_id != private_delta.layout_id:
            raise CacheContractError("base and delta layouts differ")
        if private_delta.start != base.stop:
            raise CacheContractError(
                "private delta must start after the shared base"
            )
        if not owner_id:
            raise CacheContractError("owner id is required")
        if generation < max(base.generation, private_delta.generation):
            raise CacheContractError("cache generation predates a segment")
        if private_delta.generation != generation:
            raise CacheContractError("private delta generation must match cache")
        lineage = _lineage or _LineageAuthority(generation)
        lineage.assert_current(generation)
        self.base = base
        self.private_delta = private_delta
        self.owner_id = owner_id
        self.generation = generation
        self._lineage = lineage

    @property
    def cache_id(self) -> str:
        return self._lineage.cache_id

    def begin_mutation(
        self, *, owner_id: str, expected_generation: int
    ) -> MutationPermit:
        if owner_id != self.owner_id:
            raise CacheContractError(
                "mutation owner does not own this private delta"
            )
        if expected_generation != self.generation:
            raise CacheContractError("stale cache generation")
        return self._lineage.begin(
            owner_id=owner_id, generation=expected_generation
        )

    def cancel_mutation(self, permit: MutationPermit) -> None:
        if permit.cache_id != self.cache_id or permit.owner_id != self.owner_id:
            raise CacheContractError("mutation permit belongs to another cache")
        self._lineage.cancel(permit)

    def replace_delta(
        self, permit: MutationPermit, replacement: CacheSegment
    ) -> "SegmentedCache":
        if permit.cache_id != self.cache_id or permit.owner_id != self.owner_id:
            raise CacheContractError("mutation permit belongs to another cache")
        if permit.expected_generation != self.generation:
            raise CacheContractError("mutation permit generation is stale")
        if not replacement.mutable:
            raise CacheContractError(
                "replacement delta must remain private and mutable"
            )
        if replacement.layout_id != self.base.layout_id:
            raise CacheContractError("replacement layout differs from the base")
        if replacement.start != self.base.stop:
            raise CacheContractError(
                "replacement delta is not contiguous with the base"
            )
        expected_successor = self.generation + 1
        if replacement.generation != expected_successor:
            raise CacheContractError("replacement must carry the next generation")

        next_generation = self._lineage.advance(permit)
        assert next_generation == expected_successor
        return SegmentedCache(
            base=self.base,
            private_delta=replacement,
            owner_id=self.owner_id,
            generation=next_generation,
            _lineage=self._lineage,
        )

    def fork_from_shared_base(self, *, owner_id: str) -> "SegmentedCache":
        """Create an independent lineage that shares only the immutable base."""
        self._lineage.assert_current(self.generation)
        empty_delta = CacheSegment(
            payload=None,
            start=self.base.stop,
            length=0,
            layout_id=self.base.layout_id,
            generation=self.generation,
            mutable=True,
        )
        return SegmentedCache(
            base=self.base,
            private_delta=empty_delta,
            owner_id=owner_id,
            generation=self.generation,
        )

    def for_consumer(
        self,
        consumer: CacheConsumer,
        *,
        materialize: Callable[[Any, Any], Any] | None = None,
    ) -> SegmentedCacheView | MaterializedCacheView:
        if getattr(consumer, "supports_segmented_cache", False):
            return self._lineage.while_current(
                self.generation,
                lambda: SegmentedCacheView(
                    self.base, self.private_delta, self.generation
                ),
            )
        if materialize is None:
            self._lineage.assert_current(self.generation)
            raise StockConsumerRefused(
                "stock cache consumer requires explicit materialization"
            )

        def materialize_current() -> MaterializedCacheView:
            payload = materialize(self.base.payload, self.private_delta.payload)
            return MaterializedCacheView(
                payload,
                MaterializationReceipt(
                    cache_id=self.cache_id,
                    source_generation=self.generation,
                    base_generation=self.base.generation,
                    delta_generation=self.private_delta.generation,
                    layout_id=self.base.layout_id,
                ),
            )

        return self._lineage.while_current(
            self.generation, materialize_current
        )
