from itertools import permutations

import pytest

from mlx_lm.cache_planes import CachePlaneKind
from mlx_lm.cache_scheduler import (
    CacheBatchRequest,
    CacheSchedulerMetrics,
    LineageBatchScheduler,
    PlaneCompatibility,
    PlaneRetentionManager,
    PlaneRetentionRecord,
    layered_cache_scheduler_enabled,
)


def compatibility(kind, digest, *, generation=0, hit=True):
    return PlaneCompatibility(kind, digest, generation, hit)


def request(request_id, lineage, sequence, *planes):
    return CacheBatchRequest(
        request_id,
        lineage,
        sequence,
        tuple(sorted(planes, key=lambda item: item.kind.value)),
    )


def record(
    key,
    kind,
    size,
    cost,
    *,
    sequence=0,
    pinned=False,
    mutable=False,
):
    return PlaneRetentionRecord(
        key,
        "lineage",
        kind,
        size,
        cost,
        sequence,
        pinned=pinned,
        contains_mutable_request_data=mutable,
    )


def group_signature(groups):
    return tuple(
        (
            group.lineage_id,
            tuple(item.request_id for item in group.requests),
            tuple(
                (kind.value, digest, generation)
                for kind, digest, generation in group.shared_planes
            ),
        )
        for group in groups
    )


def test_scheduler_gate_is_default_off(monkeypatch):
    monkeypatch.delenv("MLX_LM_LAYERED_CACHE_SCHEDULER", raising=False)
    assert layered_cache_scheduler_enabled() is False
    monkeypatch.setenv("MLX_LM_LAYERED_CACHE_SCHEDULER", "1")
    assert layered_cache_scheduler_enabled() is True


def test_disabled_scheduler_preserves_fifo_singletons():
    scheduler = LineageBatchScheduler()
    requests = (
        request("later", "b", 2),
        request("first", "a", 1),
    )
    groups = scheduler.group(requests)
    assert [group.requests[0].request_id for group in groups] == [
        "first",
        "later",
    ]
    assert all(not group.shared_planes for group in groups)


def test_partial_hit_grouping_is_lineage_scoped():
    prompt = CachePlaneKind.PROMPT_HOST
    kv = CachePlaneKind.ATTENTION_KV
    requests = (
        request(
            "a",
            "lineage-1",
            0,
            compatibility(prompt, "prompt-a"),
            compatibility(kv, "kv-a"),
        ),
        request(
            "b",
            "lineage-1",
            1,
            compatibility(prompt, "prompt-a"),
            compatibility(kv, "kv-b"),
        ),
        request(
            "c",
            "lineage-2",
            2,
            compatibility(prompt, "prompt-a"),
        ),
    )
    groups = LineageBatchScheduler(enabled=True).group(requests)
    assert [tuple(item.request_id for item in group.requests) for group in groups] == [
        ("a", "b"),
        ("c",),
    ]
    assert groups[0].shared_planes == ((prompt, "prompt-a", 0),)
    assert groups[0].is_partial_hit is True


def test_generation_is_part_of_compatibility():
    kind = CachePlaneKind.GDN_RECURRENT
    groups = LineageBatchScheduler(enabled=True).group(
        (
            request("a", "lineage", 0, compatibility(kind, "state", generation=1)),
            request("b", "lineage", 1, compatibility(kind, "state", generation=2)),
        )
    )
    assert len(groups) == 2


def test_grouping_is_deterministic_for_all_input_permutations():
    prompt = CachePlaneKind.PROMPT_HOST
    kv = CachePlaneKind.ATTENTION_KV
    requests = (
        request("a", "l1", 0, compatibility(prompt, "p"), compatibility(kv, "x")),
        request("b", "l1", 1, compatibility(prompt, "p"), compatibility(kv, "y")),
        request("c", "l1", 2, compatibility(kv, "y")),
        request("d", "l2", 3, compatibility(prompt, "p")),
    )
    expected = group_signature(LineageBatchScheduler(enabled=True).group(requests))
    for ordering in permutations(requests):
        actual = group_signature(LineageBatchScheduler(enabled=True).group(ordering))
        assert actual == expected


def test_scheduler_metrics_are_bounded_zero_sync_counters():
    metrics = CacheSchedulerMetrics()
    scheduler = LineageBatchScheduler(enabled=True, metrics=metrics)
    kind = CachePlaneKind.PROMPT_HOST
    scheduler.group(
        (
            request("a", "l", 0, compatibility(kind, "one")),
            request("b", "l", 1, compatibility(kind, "one")),
        )
    )
    snapshot = metrics.snapshot()
    assert snapshot["schedule_calls"] == 1
    assert snapshot["requests"] == 2
    assert snapshot["groups"] == 1
    assert snapshot["device_synchronizations"] == 0
    assert snapshot["timed_hot_path_sections"] == 0


def test_retention_is_default_off_and_rejects_mutable_request_data():
    kind = CachePlaneKind.ATTENTION_KV
    manager = PlaneRetentionManager({kind: 100})
    assert manager.consider(record("stable", kind, 10, 100)).reason == "disabled"
    enabled = PlaneRetentionManager({kind: 100}, enabled=True)
    decision = enabled.consider(record("request", kind, 10, 100, mutable=True))
    assert decision.admitted is False
    assert decision.reason == "mutable_request_data"
    assert enabled.records() == ()


def test_recomputation_value_drives_eviction_within_each_plane():
    prompt = CachePlaneKind.PROMPT_HOST
    kv = CachePlaneKind.ATTENTION_KV
    manager = PlaneRetentionManager({prompt: 100, kv: 100}, enabled=True)
    assert manager.consider(record("low", prompt, 60, 60, sequence=0)).admitted
    assert manager.consider(record("high", prompt, 40, 400, sequence=1)).admitted
    assert manager.consider(record("kv", kv, 100, 1, sequence=0)).admitted
    decision = manager.consider(record("candidate", prompt, 60, 600, sequence=2))
    assert decision.admitted is True
    assert decision.evicted_keys == ("low",)
    assert {item.key for item in manager.records()} == {"candidate", "high", "kv"}
    assert manager.touch("candidate", 9) is True
    assert manager.touch("missing", 9) is False
    assert manager.set_pinned("candidate", True) is True
    retained = {item.key: item for item in manager.records()}
    assert retained["candidate"].last_access_sequence == 9
    assert retained["candidate"].pinned is True


def test_lower_value_or_pinned_residents_block_admission():
    kind = CachePlaneKind.QSA_SUMMARY
    manager = PlaneRetentionManager({kind: 100}, enabled=True)
    manager.consider(record("valuable", kind, 100, 1000, pinned=True))
    decision = manager.consider(record("cheap", kind, 100, 1))
    assert decision.admitted is False
    assert decision.reason == "lower_recompute_value"
    assert [item.key for item in manager.records()] == ["valuable"]
    replacement = manager.consider(record("valuable", kind, 50, 500))
    assert replacement.admitted is False
    assert replacement.reason == "pinned_replacement"


@pytest.mark.parametrize("kind", tuple(CachePlaneKind))
def test_plane_budgets_remain_independent_under_many_admissions(kind):
    budget = 128
    manager = PlaneRetentionManager({kind: budget}, enabled=True)
    for index in range(32):
        manager.consider(
            record(
                f"{kind.value}-{index}",
                kind,
                8 + index % 7,
                10 + index * 17,
                sequence=index,
            )
        )
        assert sum(item.size_bytes for item in manager.records()) <= budget
