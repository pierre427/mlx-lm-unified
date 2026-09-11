import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_lm.apc as apc_module
import mlx_lm.models.cache as cache_module
from mlx_lm.apc import APCKey, AutomaticPrefixCache, MTPAPCSidecar
from mlx_lm.cache_planes import (
    CachePlaneKind,
    CachePlaneLease,
    CompiledScheduleMetadata,
    PLEResidencyHints,
    PromptCacheKeyProvenance,
    PromptHostPlane,
    PromptPrefixSpan,
)
from mlx_lm.cow_cache import (
    COWCacheError,
    COWCacheStale,
    COWCacheTelemetry,
    COWCacheUnsupported,
    COWPromptCacheBranch,
    freeze_prompt_cache,
    restore_prompt_cache,
)
from mlx_lm.hybrid_speculative import DetachedSelfMTPLane, SelfMTPCachePair
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    RingKVCache,
    RotatingKVCache,
)
from mlx_lm.models.qwen4_exp import (
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ArraysCache,
)


def _values(length, seed=0):
    return mx.arange(seed, seed + length, dtype=mx.float32).reshape(
        1, 1, length, 1
    )


def _kv(length=4, seed=0):
    cache = KVCache()
    values = _values(length, seed)
    cache.update_and_fetch(values, values)
    mx.eval(cache.keys, cache.values)
    return cache


def _as_list(value):
    return value.tolist() if value is not None else None


def test_default_off_preserves_legacy_deepcopy():
    apc = AutomaticPrefixCache(cow_branching=False)
    source = _kv()
    apc.store(APCKey("model"), [1, 2, 3, 4], [source])
    hit = apc.lookup(APCKey("model"), [1, 2, 3, 4, 5])

    assert hit.hit
    assert not isinstance(hit.cache, COWPromptCacheBranch)
    assert apc.apc_stats["cow_enabled"] is False
    assert apc.apc_stats["cow"]["sources"] == 0


def test_apc_store_freezes_source_and_branches_normal_cache_classes():
    apc = AutomaticPrefixCache(cow_branching=True)
    source = _kv()
    expected = _as_list(source.keys)
    key = APCKey("model")
    apc.store(key, [1, 2, 3, 4], [source])

    # Writing the request-owned producer after publication must detach it from
    # the immutable APC descriptor.
    source.keys[..., 0:1, :] = mx.array([[[[999.0]]]])
    hit = apc.lookup(key, [1, 2, 3, 4, 5])
    mx.eval(hit.cache[0].keys)

    assert isinstance(hit.cache, COWPromptCacheBranch)
    assert type(hit.cache[0]) is KVCache
    assert _as_list(hit.cache[0].keys) == expected
    assert hit.cache.cow_metadata.tokens == (1, 2, 3, 4)
    assert hit.cache.cow_metadata.key == key
    assert hit.prep_telemetry["physical_b2_formed"] is False
    assert hit.prep_telemetry["avoided_copy_bytes"] == hit.cache[0].nbytes
    hit.cache.close()


def test_frozen_apc_container_rejects_external_list_mutation():
    frozen, _ = freeze_prompt_cache(
        [_kv(2)], key="immutable", tokens=(1, 2), cache_type="assistant"
    )

    with pytest.raises(TypeError, match="read-only"):
        frozen.append(_kv(1))
    with pytest.raises(TypeError, match="read-only"):
        frozen[0] = _kv(1)


def test_apc_cow_receipt_carries_optional_prompt_host_plane():
    prompt_host = PromptHostPlane(
        input_fingerprint="messages-a",
        rendered_prompt="hello",
        token_ids=(1, 2),
        prefix_spans=(PromptPrefixSpan("user", 0, 2, 0, 5),),
        token_offsets=(0, 5),
        tokenizer_identity="tok",
        tokenizer_version="1",
        chat_template_identity="chat",
        chat_template_version="1",
        cache_key_provenance=PromptCacheKeyProvenance(model="model"),
    )
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("model")
    apc.store(key, [1, 2], [_kv(2)], prompt_host=prompt_host)
    hit = apc.lookup(key, [1, 2, 3])

    assert hit.prompt_host is prompt_host
    assert hit.prompt_host.rendered_prompt == "hello"
    assert hit.prompt_host.token_ids == (1, 2)
    hit.cache.close()


def test_read_branches_use_descriptor_aliases_without_materialization():
    telemetry = COWCacheTelemetry()
    frozen, _ = freeze_prompt_cache(
        [_kv(8)],
        key="model",
        tokens=range(8),
        cache_type="assistant",
        telemetry=telemetry,
    )
    first = restore_prompt_cache(frozen)
    second = restore_prompt_cache(frozen)

    assert first[0].keys is not frozen[0].keys
    assert second[0].keys is not first[0].keys
    assert _as_list(first[0].keys) == _as_list(second[0].keys)
    stats = telemetry.snapshot()
    assert stats["branches"] == 2
    assert stats["descriptor_aliases"] >= 6  # source + two K/V branches
    assert stats["materializations"] == 0
    assert stats["avoided_copy_bytes"] >= 2 * frozen[0].nbytes
    first.close()
    second.close()


def test_multirow_fork_avoids_physical_b2_and_isolates_each_row():
    frozen, _ = freeze_prompt_cache(
        [_kv(4)], key="model", tokens=range(4), cache_type="assistant"
    )
    rows = frozen.branch_rows(3)
    assert len(rows) == 3
    assert rows.physical_batch_formed is False
    assert all(row[0].keys.shape[0] == 1 for row in rows)

    token = mx.full((1, 1, 1, 1), 123, dtype=mx.float32)
    rows[1][0].update_and_fetch(token, token)
    mx.eval(*(row[0].keys for row in rows))
    assert [row[0].offset for row in rows] == [4, 5, 4]
    assert _as_list(rows[0][0].keys) == _as_list(rows[2][0].keys)
    assert rows[1].cow_prep_telemetry["physical_b2_formed"] is False
    rows.close()
    assert len(rows) == 0
    assert frozen.cow_owner.pin_count == 0


@pytest.mark.parametrize("existing", [1, 3, 17])
def test_first_kv_write_is_copy_on_write_and_counted_once(existing):
    telemetry = COWCacheTelemetry()
    frozen, _ = freeze_prompt_cache(
        [_kv(existing)],
        key="model",
        tokens=range(existing),
        cache_type="assistant",
        telemetry=telemetry,
    )
    left = restore_prompt_cache(frozen)
    right = restore_prompt_cache(frozen)
    before = _as_list(right[0].keys)

    token = mx.full((1, 1, 1, 1), 700, dtype=mx.float32)
    left[0].update_and_fetch(token, token)
    left[0].update_and_fetch(token + 1, token + 1)
    mx.eval(left[0].keys, right[0].keys, frozen[0].keys)

    assert left[0].offset == existing + 2
    assert right[0].offset == existing
    assert frozen[0].offset == existing
    assert _as_list(right[0].keys) == before
    stats = telemetry.snapshot()
    assert stats["materializations"] == 1
    assert stats["materialized_bytes"] == right[0].nbytes
    left.close()
    right.close()


def test_bf16_descriptor_branch_preserves_raw_bits():
    cache = KVCache()
    values = mx.array(
        [[[[0.0], [-0.0], [1.5], [-2.25], [3.140625]]]],
        dtype=mx.bfloat16,
    )
    cache.update_and_fetch(values, values)
    mx.eval(cache.keys, cache.values)
    frozen, _ = freeze_prompt_cache(
        [cache], key="bf16", tokens=range(5), cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)

    assert bool(
        mx.array_equal(
            branch[0].keys.view(mx.uint16), frozen[0].keys.view(mx.uint16)
        ).item()
    )
    assert bool(
        mx.array_equal(
            branch[0].values.view(mx.uint16), frozen[0].values.view(mx.uint16)
        ).item()
    )
    assert branch[0].keys.dtype == mx.bfloat16
    branch.close()


def test_ring_offsets_and_fixed_slab_are_branch_private():
    ring = RingKVCache(capacity=8, buckets=(8, 16))
    values = _values(5)
    ring.update_and_fetch(values, values)
    mx.eval(ring.keys, ring.values, ring.offset)
    frozen, _ = freeze_prompt_cache(
        [ring], key="ring", tokens=range(5), cache_type="assistant"
    )
    left = restore_prompt_cache(frozen)
    right = restore_prompt_cache(frozen)

    new = mx.full((1, 1, 2, 1), 55, dtype=mx.float32)
    left[0].update_and_fetch(new, new)
    mx.eval(left[0].keys, right[0].keys, left[0].offset, right[0].offset)

    assert left[0]._host_offset == 7
    assert int(left[0].offset.item()) == 7
    assert right[0]._host_offset == 5
    assert int(right[0].offset.item()) == 5
    assert right[0].capacity == 8
    assert _as_list(right[0].keys[..., :5, :]) == _as_list(values)
    left.close()
    right.close()


def test_rotating_ring_wrap_and_index_are_branch_private():
    ring = RotatingKVCache(max_size=4)
    values = _values(6)
    ring.update_and_fetch(values, values)
    mx.eval(ring.keys, ring.values)
    frozen, _ = freeze_prompt_cache(
        [ring], key="rotating", tokens=range(6), cache_type="assistant"
    )
    left = restore_prompt_cache(frozen)
    right = restore_prompt_cache(frozen)
    right_state = _as_list(right[0].keys)
    right_idx = right[0]._idx

    new = mx.full((1, 1, 1, 1), 77, dtype=mx.float32)
    left[0].update_and_fetch(new, new)
    mx.eval(left[0].keys, right[0].keys)

    assert left[0].offset == 7
    assert right[0].offset == 6
    assert right[0]._idx == right_idx
    assert _as_list(right[0].keys) == right_state
    left.close()
    right.close()


def test_gdn_recurrent_slots_and_metadata_are_independent():
    recurrent = ArraysCache(2)
    recurrent[0] = mx.arange(4, dtype=mx.float32).reshape(1, 4)
    recurrent[1] = mx.ones((1, 2), dtype=mx.float32)
    frozen, _ = freeze_prompt_cache(
        [recurrent], key="gdn", tokens=[1], cache_type="assistant"
    )
    left = restore_prompt_cache(frozen)
    right = restore_prompt_cache(frozen)
    replacement = mx.full((1, 4), 9, dtype=mx.float32)
    left[0][0] = replacement
    left[0].lengths = mx.array([3])
    mx.eval(left[0][0], right[0][0])

    assert _as_list(left[0][0]) == _as_list(replacement)
    assert _as_list(right[0][0]) == [[0.0, 1.0, 2.0, 3.0]]
    assert right[0].lengths is None
    assert left.cow_owner.telemetry.snapshot()["planes"]["gdn_recurrent"][
        "materializations"
    ] == 1
    left.close()
    right.close()


def test_real_qwen4_qsa_and_combined_recurrent_cache_keep_native_types():
    qsa = QSAKVCache()
    values = _values(3)
    qsa.update_and_fetch(values, values)
    qsa.update_index_keys(mx.arange(6, dtype=mx.float32).reshape(1, 3, 2))
    recurrent = Qwen4ArraysCache(4)
    recurrent[0] = mx.ones((1, 2, 2), dtype=mx.float32)
    recurrent[2] = mx.ones((1, 2, 2), dtype=mx.float32) * 2
    mx.eval(qsa.keys, qsa.values, qsa.index_keys, recurrent[0], recurrent[2])

    frozen, _ = freeze_prompt_cache(
        [qsa, recurrent], key="qwen4", tokens=[1, 2, 3], cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)

    assert type(branch[0]) is QSAKVCache
    assert type(branch[1]) is Qwen4ArraysCache
    qsa_token = mx.full((1, 1, 1, 1), 7, dtype=mx.float32)
    branch[0].update_and_fetch(qsa_token, qsa_token)
    branch[0].update_index_keys(mx.full((1, 1, 2), 8, dtype=mx.float32))
    branch[1][2] = mx.zeros((1, 2, 2), dtype=mx.float32)
    mx.eval(branch[0].keys, branch[0].index_keys, branch[1][2])

    assert branch[0].offset == 4
    assert frozen[0].offset == 3
    assert _as_list(frozen[1][2]) == [[[2.0, 2.0], [2.0, 2.0]]]
    planes = branch.cow_owner.telemetry.snapshot()["planes"]
    assert planes["attention_kv"]["materializations"] == 1
    assert planes["gdn_recurrent"]["materializations"] == 1
    branch.close()


def test_qsa_plane_descriptors_are_non_overlapping():
    qsa = QSAKVCache()
    values = mx.ones((1, 1, 3, 32), dtype=mx.float16)
    qsa.update_and_fetch(values, values)
    qsa.update_index_keys(mx.ones((1, 3, 4), dtype=mx.float16))
    mx.eval(qsa.keys, qsa.values, qsa.index_keys)
    frozen, _ = freeze_prompt_cache(
        [qsa], key="qsa-planes", tokens=(1, 2, 3), cache_type="assistant"
    )

    planes = frozen.cow_owner.plane_stats()
    attention = planes[CachePlaneKind.ATTENTION_KV.value]["logical_bytes"]
    summary = planes[CachePlaneKind.QSA_SUMMARY.value]["logical_bytes"]
    assert attention == qsa.keys.nbytes + qsa.values.nbytes
    assert summary == qsa.index_keys.nbytes
    assert attention + summary == qsa.nbytes


def test_quantized_qsa_attention_materialization_counts_packed_bytes():
    qsa = QSAKVCache()
    values = mx.arange(96, dtype=mx.float16).reshape(1, 1, 3, 32)
    qsa.update_and_fetch(values, values)
    qsa.update_index_keys(mx.ones((1, 3, 4), dtype=mx.float16))
    packed = qsa.to_quantized(group_size=32, bits=4)
    assert isinstance(packed, QSAQuantizedKVCache)
    mx.eval(packed.keys, packed.values, packed.index_keys)
    frozen, _ = freeze_prompt_cache(
        [packed], key="qsa-packed", tokens=(1, 2, 3), cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)
    token = mx.ones((1, 1, 1, 32), dtype=mx.float16)
    branch[0].update_and_fetch(token, token)
    mx.eval(branch[0].keys, branch[0].values)

    planes = branch.cow_owner.telemetry.snapshot()["planes"]
    expected = sum(array.nbytes for array in (*packed.keys, *packed.values))
    assert planes["attention_kv"]["materialized_bytes"] == expected
    assert expected > 0
    branch.close()


def test_cache_list_and_checkpoint_state_preserve_nested_topology():
    kv = _kv(3)
    recurrent = ArraysCache(1)
    recurrent[0] = mx.ones((1, 4), dtype=mx.float32)
    recurrent._checkpoints = [[(3, [mx.zeros((1, 4), dtype=mx.float32)])]]
    nested = CacheList(kv, recurrent)
    frozen, _ = freeze_prompt_cache(
        [nested], key="nested", tokens=(1, 2, 3), cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)

    assert type(branch[0]) is CacheList
    assert type(branch[0][0]) is KVCache
    assert type(branch[0][1]) is ArraysCache
    assert branch[0][1]._checkpoints is not recurrent._checkpoints
    branch[0][1][0] = mx.full((1, 4), 8, dtype=mx.float32)
    mx.eval(branch[0][1][0], frozen[0][1][0])
    assert _as_list(frozen[0][1][0]) == [[1.0, 1.0, 1.0, 1.0]]
    branch.close()


def test_deepcopy_self_mtp_lane_forks_owner_without_copying_locks():
    frozen, _ = freeze_prompt_cache(
        [_kv(4)],
        key="self-mtp",
        tokens=range(4),
        cache_type="assistant",
        sidecar=[_kv(2)],
    )
    branch = restore_prompt_cache(frozen)
    caught_up = mx.full((1, 1, 1, 1), 77, dtype=mx.float32)
    branch[0].update_and_fetch(caught_up, caught_up)
    mx.eval(branch[0].keys)
    lane = DetachedSelfMTPLane(
        lane=SimpleNamespace(uid=0),
        caches=SelfMTPCachePair(target=branch, draft=branch.cow_sidecar),
    )

    sibling = copy.deepcopy(lane)
    assert isinstance(sibling.caches.target, COWPromptCacheBranch)
    assert sibling.caches.target is not branch
    assert sibling.caches.target.cow_owner is branch.cow_owner
    assert sibling.caches.target.cow_lineage_id == branch.cow_lineage_id
    assert sibling.caches.draft is sibling.caches.target.cow_sidecar
    assert frozen.cow_owner.pin_count == 2
    assert sibling.caches.target[0].offset == branch[0].offset == 5
    assert frozen[0].offset == 4
    assert _as_list(sibling.caches.target[0].keys) == _as_list(branch[0].keys)

    token = mx.full((1, 1, 1, 1), 99, dtype=mx.float32)
    sibling.caches.target[0].update_and_fetch(token, token)
    mx.eval(sibling.caches.target[0].keys, branch[0].keys)
    assert sibling.caches.target[0].offset == 6
    assert branch[0].offset == 5
    sibling.caches.target.close()
    branch.close()


def test_closed_branch_drops_payload_and_cannot_be_deepcopied():
    frozen, _ = freeze_prompt_cache(
        [_kv(4)], key="close", tokens=range(4), cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)
    branch.close()

    assert branch == []
    assert branch.cow_sidecar is None
    with pytest.raises(COWCacheError, match="closed"):
        copy.deepcopy(branch)


def test_cow_owner_exposes_independent_plane_fingerprints_and_invalidation():
    prompt_host = PromptHostPlane(
        input_fingerprint="messages-a",
        rendered_prompt="hello",
        token_ids=(1, 2, 3),
        prefix_spans=(PromptPrefixSpan("user", 0, 3, 0, 5),),
        token_offsets=(),
        tokenizer_identity="tok",
        tokenizer_version="1",
        chat_template_identity="chat",
        chat_template_version="1",
        cache_key_provenance=PromptCacheKeyProvenance(model="qwen4"),
    )
    qsa = QSAKVCache()
    values = _values(3)
    qsa.update_and_fetch(values, values)
    qsa.update_index_keys(mx.ones((1, 3, 2), dtype=mx.float32))
    recurrent = Qwen4ArraysCache(4)
    recurrent[0] = mx.ones((1, 2), dtype=mx.float32)
    frozen, _ = freeze_prompt_cache(
        [qsa, recurrent],
        key="qwen4",
        tokens=(1, 2, 3),
        cache_type="assistant",
        prompt_host=prompt_host,
        ple_hints=PLEResidencyHints("p1", (1, 3), "nvme-a"),
        compiled_schedule=CompiledScheduleMetadata(
            "impl-a", "schedule-a", "runtime-a", (48, 2560)
        ),
    )
    owner = frozen.cow_owner
    stats = owner.plane_stats()
    assert {
        "prompt_host",
        "qsa_summary",
        "gdn_recurrent",
        "ple_hints",
        "compiled_schedule",
    } <= set(stats)
    assert all(value["fingerprint"] for value in stats.values())

    # Optional schedule invalidation cannot poison the device cache branch.
    owner.invalidate_plane(CachePlaneKind.COMPILED_SCHEDULE, "runtime_changed")
    branch = restore_prompt_cache(frozen)
    branch.close()

    # A required QSA miss blocks the monolithic device consumer, but the
    # independently fingerprinted prompt plane remains reusable.
    owner.invalidate_plane(CachePlaneKind.QSA_SUMMARY, "qsa_layout_changed")
    with pytest.raises(COWCacheStale):
        restore_prompt_cache(frozen)
    prompt = owner.plane_manifest.try_adopt(
        CachePlaneKind.PROMPT_HOST, prompt_host.fingerprint
    )
    assert isinstance(prompt, CachePlaneLease)
    assert prompt.payload.token_ids == (1, 2, 3)
    prompt.close()
    assert owner.plane_stats()["qsa_summary"]["invalidation_reason"] == (
        "qsa_layout_changed"
    )


def test_target_and_mtp_sidecar_share_one_lineage_but_not_mutable_state():
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("qwen4")
    sidecar = MTPAPCSidecar(
        ([_kv(1, seed=100)], mx.ones((1, 1, 4), dtype=mx.float32)),
        covered_tokens=2,
    )
    apc.store(key, [1, 2, 3], [_kv(2)], sidecar=sidecar)
    first = apc.lookup(key, [1, 2, 3, 4])
    second = apc.lookup(key, [1, 2, 3, 5])

    assert isinstance(first.cache, COWPromptCacheBranch)
    assert first.sidecar is first.cache.cow_sidecar
    assert first.cache.cow_lineage_id == second.cache.cow_lineage_id
    new = mx.full((1, 1, 1, 1), 333, dtype=mx.float32)
    first.sidecar.state[0][0].update_and_fetch(new, new)
    mx.eval(first.sidecar.state[0][0].keys, second.sidecar.state[0][0].keys)
    assert first.sidecar.state[0][0].offset == 2
    assert second.sidecar.state[0][0].offset == 1
    first.cache.close()
    second.cache.close()


def test_invalidation_rejects_new_branches_and_pins_live_branch_until_close():
    frozen, _ = freeze_prompt_cache(
        [_kv()], key="model", tokens=range(4), cache_type="assistant"
    )
    live = restore_prompt_cache(frozen)
    owner = frozen.cow_owner
    assert owner.pin_count == 1

    frozen.close()
    assert owner.source_released is False
    with pytest.raises(COWCacheStale):
        restore_prompt_cache(frozen)

    token = mx.full((1, 1, 1, 1), 44, dtype=mx.float32)
    live[0].update_and_fetch(token, token)
    mx.eval(live[0].keys)
    assert live[0].offset == 5
    live.close()
    assert owner.pin_count == 0
    assert owner.source_released is True


def test_concurrent_invalidation_and_first_write_remain_race_safe():
    frozen, _ = freeze_prompt_cache(
        [_kv()], key="model", tokens=range(4), cache_type="assistant"
    )
    live = restore_prompt_cache(frozen)
    barrier = threading.Barrier(2)

    def mutate():
        barrier.wait()
        # MLX streams are thread-local; exercise the host-side first-write CAS
        # here and perform array work in the dedicated isolation tests above.
        return live.note_mutation(0, "concurrent-first-write")

    def invalidate():
        barrier.wait()
        return frozen.cow_owner.invalidate()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(mutate), pool.submit(invalidate)]
        assert results[0].result(timeout=3) is True
        assert results[1].result(timeout=3) == 1
    assert frozen.cow_owner.source_released is False
    live.close()
    assert frozen.cow_owner.source_released is True


def test_required_plane_invalidation_serializes_with_branch_adoption(monkeypatch):
    frozen, _ = freeze_prompt_cache(
        [_kv(4)], key="plane-race", tokens=range(4), cache_type="assistant"
    )
    owner = frozen.cow_owner
    entered = threading.Event()
    release = threading.Event()
    original = owner.plane_manifest.invalidate

    def blocked_invalidate(kind, reason):
        entered.set()
        assert release.wait(timeout=2)
        return original(kind, reason)

    monkeypatch.setattr(owner.plane_manifest, "invalidate", blocked_invalidate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        invalidate_future = executor.submit(
            owner.invalidate_plane,
            CachePlaneKind.ATTENTION_KV,
            "layout_changed",
        )
        assert entered.wait(timeout=2)
        branch_future = executor.submit(restore_prompt_cache, frozen)
        assert not branch_future.done()
        release.set()
        assert invalidate_future.result(timeout=2) == 1
        with pytest.raises(COWCacheStale):
            branch_future.result(timeout=2)


def test_concurrent_close_cannot_clear_live_tip_during_deepcopy(monkeypatch):
    frozen, _ = freeze_prompt_cache(
        [_kv(4)], key="copy-close-race", tokens=range(4), cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)
    owner = frozen.cow_owner
    entered = threading.Event()
    release = threading.Event()
    original = owner.branch_live

    def blocked_branch_live(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "branch_live", blocked_branch_live)
    with ThreadPoolExecutor(max_workers=2) as executor:
        copy_future = executor.submit(copy.deepcopy, branch)
        assert entered.wait(timeout=2)
        close_future = executor.submit(branch.close)
        assert not close_future.done()
        release.set()
        clone = copy_future.result(timeout=2)
        close_future.result(timeout=2)

    assert branch == []
    assert clone[0].offset == 4
    assert owner.pin_count == 1
    clone.close()
    assert owner.pin_count == 0


class _FailSecondCopy:
    copies = 0

    def __init__(self):
        self.keys = mx.ones((1, 1, 1, 1))

    def __copy__(self):
        type(self).copies += 1
        if type(self).copies == 2:
            raise RuntimeError("branch copy failed")
        clone = object.__new__(type(self))
        clone.__dict__ = dict(self.__dict__)
        return clone

    @property
    def nbytes(self):
        return self.keys.nbytes


def test_partial_branch_failure_releases_pin_exactly_once():
    _FailSecondCopy.copies = 0
    frozen, _ = freeze_prompt_cache(
        [_FailSecondCopy()], key="bad", tokens=[1], cache_type="assistant"
    )
    owner = frozen.cow_owner
    with pytest.raises(Exception, match="cannot copy"):
        restore_prompt_cache(frozen)
    assert owner.pin_count == 0
    assert owner.telemetry.snapshot()["active_leases"] == 0
    assert owner.telemetry.snapshot()["branch_failures"] == 1


class _DeepcopyOnlyLeaf:
    __slots__ = ()

    def __deepcopy__(self, memo):
        return type(self)()


class _LegacyFallbackCache:
    def __init__(self):
        self.keys = mx.ones((1, 1, 2, 1))
        self.extra = _DeepcopyOnlyLeaf()
        self.offset = 2

    @property
    def state(self):
        return self.keys

    @property
    def nbytes(self):
        return self.keys.nbytes

    def is_trimmable(self):
        return False

    def snap_trim_position(self, position):
        return None


def test_unsupported_freeze_uses_legacy_apc_fallback():
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("custom")
    apc.store(key, [1, 2], [_LegacyFallbackCache()])
    hit = apc.lookup(key, [1, 2, 3])

    assert hit.hit
    assert not isinstance(hit.cache, COWPromptCacheBranch)
    assert apc.apc_stats["cow"]["freeze_failures"] == 1


def test_freeze_rejects_transient_speculating_mtp_sidecar():
    draft = ArraysCache(1)
    draft[0] = mx.ones((1, 4), dtype=mx.float32)
    draft.start_speculation()
    sidecar = MTPAPCSidecar(
        state=([draft], mx.ones((1, 1, 4), dtype=mx.float32)),
        covered_tokens=2,
    )

    with pytest.raises(COWCacheUnsupported, match="speculation transaction"):
        freeze_prompt_cache(
            [_kv(2)],
            key="mtp-transient",
            tokens=(1, 2),
            cache_type="assistant",
            sidecar=sidecar,
        )


def test_clear_defers_source_release_until_active_branch_closes():
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("model")
    apc.store(key, [1, 2], [_kv(2)])
    branch = apc.lookup(key, [1, 2, 3]).cache
    owner = branch.cow_owner

    report = apc.clear(release_memory=False)
    assert report["entries"] == 1
    assert owner.source_released is False
    assert apc.lookup(key, [1, 2, 3]).hit is False
    branch.close()
    assert owner.source_released is True


def test_clear_and_lookup_observe_one_atomic_trie_generation(monkeypatch):
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("atomic-clear")
    apc.store(key, [1, 2, 3, 4], [_kv(4)])
    entered = threading.Event()
    release = threading.Event()
    original = apc_module._copy_prompt_cache_for_restore

    def blocked_restore(cache):
        entered.set()
        assert release.wait(timeout=2)
        return original(cache)

    monkeypatch.setattr(
        apc_module, "_copy_prompt_cache_for_restore", blocked_restore
    )
    monkeypatch.setattr(cache_module, "_copy_prompt_cache_for_restore", blocked_restore)
    with ThreadPoolExecutor(max_workers=2) as executor:
        lookup_future = executor.submit(apc.lookup, key, [1, 2, 3, 4, 5])
        assert entered.wait(timeout=2)
        clear_future = executor.submit(apc.clear, release_memory=False)
        assert not clear_future.done()
        release.set()
        hit = lookup_future.result(timeout=2)
        report = clear_future.result(timeout=2)

    assert hit.hit
    assert report["entries"] == 1
    assert apc.lookup(key, [1, 2, 3, 4, 5]).hit is False
    hit.cache.close()


def test_required_plane_invalidation_turns_apc_lookup_into_fail_closed_miss():
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("model")
    apc.store(key, [1, 2], [_kv(2)])
    # Traverse the token radix rather than depending on private node depth.
    node = apc._trie._trie[key]
    for token in (1, 2):
        node = node[token]
    frozen = node["__value__"].prompt_cache
    frozen.cow_owner.invalidate_plane(
        CachePlaneKind.ATTENTION_KV, "kv_layout_changed"
    )

    miss = apc.lookup(key, [1, 2, 3])
    assert miss.hit is False
    assert miss.cache is None
    assert miss.remaining_tokens == [1, 2, 3]
    assert miss.miss_reason == "stale_cow_generation"


def test_mtp_plane_invalidation_reuses_target_and_rebuilds_draft_only():
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("mtp-plane-fallback")
    sidecar = MTPAPCSidecar(
        state=([_kv(1)], mx.ones((1, 1, 4), dtype=mx.float32)),
        covered_tokens=2,
    )
    apc.store(key, [1, 2], [_kv(2)], sidecar=sidecar)
    node = apc._trie._trie[key][1][2]
    frozen = node["__value__"].prompt_cache
    frozen.cow_owner.invalidate_plane(CachePlaneKind.MTP_DRAFT, "draft_changed")

    hit = apc.lookup(key, [1, 2, 3])
    assert hit.hit
    assert hit.hit_kind == "prefix"
    assert hit.remaining_tokens == [3]
    assert hit.sidecar is None
    assert hit.cache.cow_sidecar is None
    assert hit.cache.cow_owner.plane_stats()["mtp_draft"][
        "invalidation_reason"
    ] == "draft_changed"
    hit.cache.close()


def test_prompt_plane_invalidation_does_not_poison_device_cache():
    prompt_host = PromptHostPlane(
        input_fingerprint="prompt-plane",
        rendered_prompt="hello",
        token_ids=(1, 2),
        prefix_spans=(),
        token_offsets=(),
        tokenizer_identity="tok",
        tokenizer_version="1",
        chat_template_identity="chat",
        chat_template_version="1",
        cache_key_provenance=PromptCacheKeyProvenance(model="model"),
    )
    apc = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("prompt-plane-fallback")
    apc.store(key, [1, 2], [_kv(2)], prompt_host=prompt_host)
    frozen = apc._trie._trie[key][1][2]["__value__"].prompt_cache
    frozen.cow_owner.invalidate_plane(CachePlaneKind.PROMPT_HOST, "template_changed")

    hit = apc.lookup(key, [1, 2, 3])
    assert hit.hit
    assert hit.prompt_host is None
    assert hit.cache.cow_metadata.prompt_host is None
    hit.cache.close()


def test_lru_eviction_invalidates_source_but_preserves_live_branch():
    apc = AutomaticPrefixCache(max_size=1, cow_branching=True)
    first_key = APCKey("first")
    second_key = APCKey("second")
    apc.store(first_key, [1, 2], [_kv(2)])
    branch = apc.lookup(first_key, [1, 2, 3]).cache
    owner = branch.cow_owner

    apc.store(second_key, [7, 8], [_kv(2, seed=10)])
    assert owner.plane_stats()["attention_kv"]["invalidation_reason"] == (
        "owner_invalidated"
    )
    assert owner.source_released is False
    token = mx.full((1, 1, 1, 1), 6, dtype=mx.float32)
    branch[0].update_and_fetch(token, token)
    mx.eval(branch[0].keys)
    assert branch[0].offset == 3
    branch.close()
    assert owner.source_released is True


def test_concurrent_close_is_idempotent():
    frozen, _ = freeze_prompt_cache(
        [_kv()], key="model", tokens=range(4), cache_type="assistant"
    )
    branch = restore_prompt_cache(frozen)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: branch.close(), range(32)))
    assert frozen.cow_owner.pin_count == 0
    assert frozen.cow_owner.telemetry.snapshot()["releases"] == 1


def test_copy_module_still_deepcopies_ordinary_cache_graphs():
    cache = [_kv()]
    clone = copy.deepcopy(cache)
    assert clone is not cache
    assert clone[0] is not cache[0]


def test_republishing_a_cow_branch_drops_old_hooks_and_lineage():
    first = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("model")
    first.store(key, [1, 2], [_kv(2)])
    branch = first.lookup(key, [1, 2, 3]).cache

    second = AutomaticPrefixCache(cow_branching=True)
    second.store(key, [1, 2], branch)
    restored = second.lookup(key, [1, 2, 4]).cache
    new = mx.full((1, 1, 1, 1), 9, dtype=mx.float32)
    restored[0].update_and_fetch(new, new)
    mx.eval(restored[0].keys)

    assert restored.cow_lineage_id != branch.cow_lineage_id
    assert restored[0].offset == 3
    assert second.apc_stats["cow"]["materializations"] == 1
    branch.close()
    restored.close()
