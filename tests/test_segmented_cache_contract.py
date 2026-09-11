from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from mlx_lm.segmented_cache_contract import (
    CacheContractError,
    CacheSegment,
    MaterializedCacheView,
    SegmentedCache,
    SegmentedCacheView,
    StockConsumerRefused,
)


def make_cache():
    return SegmentedCache(
        base=CacheSegment("base", 0, 10, "kv-bf16", 0),
        private_delta=CacheSegment("delta", 10, 2, "kv-bf16", 0, mutable=True),
        owner_id="request-a",
    )


def test_shared_base_must_be_immutable_and_delta_private():
    with pytest.raises(CacheContractError, match="base must be immutable"):
        SegmentedCache(
            base=CacheSegment("base", 0, 10, "kv", 0, mutable=True),
            private_delta=CacheSegment("delta", 10, 0, "kv", 0, mutable=True),
            owner_id="request",
        )
    with pytest.raises(CacheContractError, match="delta must be mutable"):
        SegmentedCache(
            base=CacheSegment("base", 0, 10, "kv", 0),
            private_delta=CacheSegment("delta", 10, 0, "kv", 0),
            owner_id="request",
        )


def test_stock_consumer_is_refused_until_explicit_materialization():
    cache = make_cache()
    stock = SimpleNamespace(supports_segmented_cache=False)
    with pytest.raises(StockConsumerRefused, match="explicit materialization"):
        cache.for_consumer(stock)
    view = cache.for_consumer(stock, materialize=lambda base, delta: base + delta)
    assert isinstance(view, MaterializedCacheView)
    assert view.payload == "basedelta"
    assert view.receipt.explicit is True
    assert view.receipt.source_generation == cache.generation


def test_segment_aware_consumer_receives_base_and_delta_without_copy():
    cache = make_cache()
    aware = SimpleNamespace(supports_segmented_cache=True)
    view = cache.for_consumer(aware)
    assert isinstance(view, SegmentedCacheView)
    assert view.base is cache.base
    assert view.private_delta is cache.private_delta


def test_mutation_is_owner_and_generation_checked_and_permit_is_single_use():
    cache = make_cache()
    with pytest.raises(CacheContractError, match="does not own"):
        cache.begin_mutation(owner_id="request-b", expected_generation=0)
    permit = cache.begin_mutation(owner_id="request-a", expected_generation=0)
    with pytest.raises(CacheContractError, match="already in progress"):
        cache.begin_mutation(owner_id="request-a", expected_generation=0)
    replacement = CacheSegment("next", 10, 3, "kv-bf16", 1, mutable=True)
    updated = cache.replace_delta(permit, replacement)
    assert updated.generation == 1
    assert updated.base is cache.base
    with pytest.raises(CacheContractError, match="already consumed"):
        cache.replace_delta(permit, replacement)
    with pytest.raises(CacheContractError, match="retired or stale"):
        cache.begin_mutation(owner_id="request-a", expected_generation=0)
    with pytest.raises(CacheContractError, match="stale"):
        updated.replace_delta(permit, replace_generation(replacement, 2))


def replace_generation(segment, generation):
    return CacheSegment(
        segment.payload,
        segment.start,
        segment.length,
        segment.layout_id,
        generation,
        mutable=segment.mutable,
    )


def test_fork_shares_only_the_immutable_base():
    cache = make_cache()
    fork = cache.fork_from_shared_base(owner_id="request-b")
    assert fork.base is cache.base
    assert fork.private_delta is not cache.private_delta
    assert fork.private_delta.length == 0
    assert fork.private_delta.payload is None
    assert fork.owner_id == "request-b"
    assert fork.cache_id != cache.cache_id


def test_one_parent_cannot_publish_divergent_same_generation_successors():
    cache = make_cache()
    permit = cache.begin_mutation(
        owner_id="request-a", expected_generation=cache.generation
    )
    first = CacheSegment("first", 10, 1, "kv-bf16", 1, mutable=True)
    second = CacheSegment("second", 10, 1, "kv-bf16", 1, mutable=True)
    successor = cache.replace_delta(permit, first)
    assert successor.cache_id == cache.cache_id
    assert successor.generation == 1
    with pytest.raises(CacheContractError, match="already consumed"):
        cache.replace_delta(permit, second)
    with pytest.raises(CacheContractError, match="retired or stale"):
        cache.for_consumer(SimpleNamespace(supports_segmented_cache=True))


def test_concurrent_begin_has_one_lineage_winner():
    cache = make_cache()
    barrier = Barrier(8)

    def begin():
        barrier.wait()
        try:
            return cache.begin_mutation(
                owner_id="request-a", expected_generation=0
            )
        except CacheContractError as error:
            return error

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: begin(), range(8)))
    permits = [value for value in outcomes if not isinstance(value, Exception)]
    errors = [value for value in outcomes if isinstance(value, Exception)]
    assert len(permits) == 1
    assert len(errors) == 7
    assert all("already in progress" in str(error) for error in errors)
    cache.cancel_mutation(permits[0])
    replacement = cache.begin_mutation(
        owner_id="request-a", expected_generation=0
    )
    assert replacement.nonce != permits[0].nonce


def test_concurrent_replace_atomically_retires_source_generation():
    cache = make_cache()
    permit = cache.begin_mutation(owner_id="request-a", expected_generation=0)
    barrier = Barrier(2)

    def replace(payload):
        barrier.wait()
        candidate = CacheSegment(
            payload, 10, 1, "kv-bf16", 1, mutable=True
        )
        try:
            return cache.replace_delta(permit, candidate)
        except CacheContractError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(replace, ("left", "right")))
    successors = [value for value in outcomes if isinstance(value, SegmentedCache)]
    errors = [value for value in outcomes if isinstance(value, Exception)]
    assert len(successors) == 1
    assert len(errors) == 1
    assert "already consumed" in str(errors[0])
    assert successors[0].cache_id == cache.cache_id
    assert successors[0].generation == 1


def test_materialization_pins_generation_until_copy_completes():
    cache = make_cache()
    permit = cache.begin_mutation(owner_id="request-a", expected_generation=0)
    materializing = Event()
    allow_copy_to_finish = Event()
    replacement_started = Event()

    def materialize(base, delta):
        materializing.set()
        assert allow_copy_to_finish.wait(timeout=2)
        return base + delta

    def consume():
        return cache.for_consumer(
            SimpleNamespace(supports_segmented_cache=False),
            materialize=materialize,
        )

    def replace():
        assert materializing.wait(timeout=2)
        replacement_started.set()
        return cache.replace_delta(
            permit,
            CacheSegment("next", 10, 3, "kv-bf16", 1, mutable=True),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        consumed = pool.submit(consume)
        replaced = pool.submit(replace)
        assert replacement_started.wait(timeout=2)
        assert not replaced.done()
        allow_copy_to_finish.set()
        view = consumed.result(timeout=2)
        successor = replaced.result(timeout=2)

    assert view.payload == "basedelta"
    assert view.receipt.source_generation == 0
    assert successor.generation == 1
    with pytest.raises(CacheContractError, match="retired or stale"):
        cache.for_consumer(SimpleNamespace(supports_segmented_cache=True))


def test_layout_contiguity_and_generation_fail_closed():
    base = CacheSegment("base", 0, 10, "kv", 2)
    with pytest.raises(CacheContractError, match="start after"):
        SegmentedCache(
            base=base,
            private_delta=CacheSegment("delta", 11, 1, "kv", 2, mutable=True),
            owner_id="request",
            generation=2,
        )
    with pytest.raises(CacheContractError, match="predates"):
        SegmentedCache(
            base=base,
            private_delta=CacheSegment("delta", 10, 1, "kv", 2, mutable=True),
            owner_id="request",
            generation=1,
        )
