import gc
from dataclasses import FrozenInstanceError, fields

import pytest

from mlx_lm.cache_planes import (
    CachePlaneFallback,
    CachePlaneFingerprint,
    CachePlaneKind,
    CachePlaneLease,
    CachePlaneOwner,
    CompiledScheduleMetadata,
    LayeredCacheManifest,
    PLEResidencyHints,
    PromptCacheKeyProvenance,
    PromptHostPlane,
    PromptHostPlaneCache,
    PromptPrefixSpan,
)


def _prompt(input_fingerprint="messages-sha"):
    return PromptHostPlane(
        input_fingerprint=input_fingerprint,
        rendered_prompt="<|user|>hello<|assistant|>",
        token_ids=(1, 2, 3, 4),
        prefix_spans=(PromptPrefixSpan("system+user", 0, 3, 0, 14),),
        token_offsets=(0, 4, 8, 12),
        tokenizer_identity="qwen-tokenizer",
        tokenizer_version="rev-a",
        chat_template_identity="qwen-chat",
        chat_template_version="v4",
        cache_key_provenance=PromptCacheKeyProvenance(
            model="qwen4",
            revision="weights-a",
            adapter="adapter-a",
            semantic_fingerprint="text-only",
            cache_layout_fingerprint="qsa-gdn-bf16",
        ),
    )


def _fingerprint(kind, value):
    return CachePlaneFingerprint.from_fields(kind, identity=value)


def test_prompt_host_plane_is_immutable_and_excludes_mutable_request_state():
    plane = _prompt()
    with pytest.raises(FrozenInstanceError):
        plane.rendered_prompt = "changed"
    assert plane.token_ids == (1, 2, 3, 4)
    names = {field.name for field in fields(PromptHostPlane)}
    assert not names & {
        "request_id",
        "rng",
        "sampler",
        "stop_state",
        "generation_budget",
        "pending_tokens",
    }


def test_prompt_store_reuses_rendering_and_tokens_on_exact_identity():
    cache = PromptHostPlaneCache()
    plane = _prompt()
    owner = cache.store(plane)
    adopted = cache.lookup(plane.input_fingerprint, plane.fingerprint)

    assert isinstance(adopted, CachePlaneLease)
    assert adopted.payload is plane
    assert adopted.payload.rendered_prompt.startswith("<|user|>")
    assert adopted.payload.token_ids == (1, 2, 3, 4)
    adopted.close()
    assert owner.stats()["hits"] == 1
    assert owner.stats()["active_leases"] == 0


def test_tokenizer_or_template_mismatch_falls_back_only_prompt_plane():
    cache = PromptHostPlaneCache()
    plane = _prompt()
    owner = cache.store(plane)
    mismatch = CachePlaneFingerprint.from_fields(
        CachePlaneKind.PROMPT_HOST,
        input=plane.input_fingerprint,
        tokenizer="other-tokenizer",
        tokenizer_version="rev-b",
        chat_template="other-template",
        chat_template_version="v5",
        model="qwen4",
        revision="weights-a",
        adapter="adapter-a",
        semantic="text-only",
    )
    result = cache.lookup(plane.input_fingerprint, mismatch)

    assert isinstance(result, CachePlaneFallback)
    assert result.reason == "fingerprint_mismatch"
    assert result.action == "rebuild_prompt_host"
    assert owner.stats()["fallbacks"] == 1


def test_one_plane_miss_does_not_invalidate_other_reusable_planes():
    kv_fingerprint = _fingerprint(CachePlaneKind.ATTENTION_KV, "kv-a")
    qsa_fingerprint = _fingerprint(CachePlaneKind.QSA_SUMMARY, "qsa-a")
    gdn_fingerprint = _fingerprint(CachePlaneKind.GDN_RECURRENT, "gdn-a")
    manifest = LayeredCacheManifest(
        [
            CachePlaneOwner(
                kind=CachePlaneKind.ATTENTION_KV,
                payload="kv",
                fingerprint=kv_fingerprint,
            ),
            CachePlaneOwner(
                kind=CachePlaneKind.QSA_SUMMARY,
                payload="qsa",
                fingerprint=qsa_fingerprint,
            ),
            CachePlaneOwner(
                kind=CachePlaneKind.GDN_RECURRENT,
                payload="gdn",
                fingerprint=gdn_fingerprint,
            ),
        ]
    )

    qsa_miss = manifest.try_adopt(
        CachePlaneKind.QSA_SUMMARY,
        _fingerprint(CachePlaneKind.QSA_SUMMARY, "qsa-b"),
    )
    kv_hit = manifest.try_adopt(CachePlaneKind.ATTENTION_KV, kv_fingerprint)
    gdn_hit = manifest.try_adopt(CachePlaneKind.GDN_RECURRENT, gdn_fingerprint)

    assert isinstance(qsa_miss, CachePlaneFallback)
    assert isinstance(kv_hit, CachePlaneLease)
    assert isinstance(gdn_hit, CachePlaneLease)
    assert manifest.stats()["attention_kv"]["eligible"] is True
    assert manifest.stats()["gdn_recurrent"]["eligible"] is True
    kv_hit.close()
    gdn_hit.close()


def test_plane_invalidation_reason_and_pinning_are_independent():
    fingerprint = _fingerprint(CachePlaneKind.MTP_DRAFT, "mtp-k2")
    owner = CachePlaneOwner(
        kind=CachePlaneKind.MTP_DRAFT,
        payload={"draft": "state"},
        fingerprint=fingerprint,
    )
    lease = owner.try_adopt(fingerprint)
    assert isinstance(lease, CachePlaneLease)
    owner.invalidate("mtp_policy_changed")

    stats = owner.stats()
    assert stats["invalidation_reason"] == "mtp_policy_changed"
    assert stats["pinned"] == 1
    miss = owner.try_adopt(fingerprint)
    assert isinstance(miss, CachePlaneFallback)
    assert miss.reason == "mtp_policy_changed"
    lease.close()
    assert owner.stats()["pinned"] == 0
    assert owner.stats()["payload_released"] is True


def test_invalidating_unleased_plane_releases_payload_immediately():
    fingerprint = _fingerprint(CachePlaneKind.QSA_SUMMARY, "qsa")
    owner = CachePlaneOwner(
        kind=CachePlaneKind.QSA_SUMMARY,
        payload={"summary": "state"},
        fingerprint=fingerprint,
    )
    owner.invalidate("layout_changed")
    assert owner.stats()["payload_released"] is True


def test_abandoned_plane_lease_releases_its_pin_via_finalizer():
    fingerprint = _fingerprint(CachePlaneKind.ATTENTION_RING, "ring")
    owner = CachePlaneOwner(
        kind=CachePlaneKind.ATTENTION_RING,
        payload="ring-state",
        fingerprint=fingerprint,
    )
    lease = owner.try_adopt(fingerprint)
    assert isinstance(lease, CachePlaneLease)
    assert owner.stats()["pinned"] == 1
    del lease
    gc.collect()
    assert owner.stats()["pinned"] == 0


def test_each_plane_has_own_materialization_counters():
    kv = CachePlaneOwner(
        kind=CachePlaneKind.ATTENTION_KV,
        payload="kv",
        fingerprint=_fingerprint(CachePlaneKind.ATTENTION_KV, "a"),
    )
    gdn = CachePlaneOwner(
        kind=CachePlaneKind.GDN_RECURRENT,
        payload="gdn",
        fingerprint=_fingerprint(CachePlaneKind.GDN_RECURRENT, "a"),
    )
    kv.record_materialization(1024)
    kv.record_materialization(2048)

    assert kv.stats()["materializations"] == 2
    assert kv.stats()["materialized_bytes"] == 3072
    assert gdn.stats()["materializations"] == 0


def test_prompt_lru_eviction_does_not_affect_another_entry():
    cache = PromptHostPlaneCache(max_entries=1)
    first = cache.store(_prompt("first"))
    second_plane = _prompt("second")
    second = cache.store(second_plane)

    assert first.stats()["invalidation_reason"] == "lru_evicted"
    adopted = cache.lookup("second", second_plane.fingerprint)
    assert isinstance(adopted, CachePlaneLease)
    assert second.stats()["eligible"] is True
    adopted.close()


def test_optional_hint_and_schedule_planes_are_metadata_only():
    hints = PLEResidencyHints("v1", (2, 4, 6), "nvme-map-a")
    schedule = CompiledScheduleMetadata(
        "implementation-sha", "schedule-sha", "mlx-host-sha", (48, 2560)
    )
    assert hints.resident_layer_ids == (2, 4, 6)
    assert schedule.geometry == (48, 2560)
    assert not any(callable(getattr(schedule, field.name)) for field in fields(schedule))
