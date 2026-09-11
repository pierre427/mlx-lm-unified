from dataclasses import replace

import pytest

from mlx_lm.cache_planes import (
    CachePlaneFingerprint,
    CachePlaneKind,
    PLEResidencyHints,
)
from mlx_lm.cache_policy import (
    AccuracyEvidence,
    AccuracyGate,
    CachePlanePlacementPolicy,
    CachePlanePlacementRequest,
    CachePolicyMetrics,
    ColdKVCompressionPolicy,
    ContentAddressedPlaneStore,
    IdentityCodec,
    IntegrityError,
    PersistenceRefused,
    StablePlaneRecord,
    ZlibCodec,
    cache_plane_operation,
    layered_cache_policy_enabled,
)
from mlx_lm.heterogeneous_execution import (
    Engine,
    EngineCapability,
    HeterogeneousExecutionProfile,
    HostRuntimeFingerprint,
    OperationGeometry,
    OperationMeasurement,
)


def host_fingerprint():
    return HostRuntimeFingerprint(
        "lab",
        "M5 Ultra",
        "25A1",
        "qwen4",
        "revision",
        "4bit",
        (("mlx", "0.31.2"),),
    )


def geometry(size=32):
    return OperationGeometry(
        size,
        size,
        (1, size),
        (1, size),
        "bfloat16",
        "row-major",
        "cache-pack-v1",
    )


def sample(fp, engine, cost, geom=None):
    return OperationMeasurement(
        cache_plane_operation(CachePlaneKind.ATTENTION_KV, "cache_pack"),
        engine,
        geom or geometry(),
        fp.digest,
        cost * 0.1,
        cost * 0.2,
        cost * 0.4,
        cost * 0.2,
        cost * 0.1,
        cost + 1,
    )


def placement_profile():
    fp = host_fingerprint()
    profile = HeterogeneousExecutionProfile(
        fp,
        {
            engine: EngineCapability(
                engine,
                frozenset(
                    {
                        cache_plane_operation(
                            CachePlaneKind.ATTENTION_KV, "cache_pack"
                        )
                    }
                ),
                4096,
            )
            for engine in (Engine.CPU, Engine.ANE)
        },
    )
    profile.record(sample(fp, Engine.CPU, 1))
    profile.record(sample(fp, Engine.ANE, 4))
    return profile


def plane_fingerprint(kind=CachePlaneKind.PROMPT_HOST, suffix="a"):
    return CachePlaneFingerprint.from_fields(
        kind, model="qwen4", revision="revision", content=suffix
    )


def stable_record(
    kind=CachePlaneKind.PROMPT_HOST,
    *,
    payload=b"stable-plane",
    suffix="a",
    mutable=False,
    stable=True,
):
    return StablePlaneRecord(
        kind,
        plane_fingerprint(kind, suffix),
        "qwen4@revision/4bit",
        0,
        payload,
        stable=stable,
        contains_mutable_request_data=mutable,
    )


class TruncateColdKVCodec:
    codec_id = "truncate-cold-kv-test-v1"
    lossless = False
    experimental_accuracy_gated = True

    def encode(self, payload):
        return payload[::2]

    def decode(self, payload):
        return payload


class BadLosslessCodec:
    codec_id = "bad-lossless-test-v1"
    lossless = True
    experimental_accuracy_gated = False

    def encode(self, payload):
        return payload[:1]

    def decode(self, payload):
        return payload


def test_policy_gate_and_placement_are_default_off(monkeypatch):
    monkeypatch.delenv("MLX_LM_LAYERED_CACHE_POLICY", raising=False)
    assert layered_cache_policy_enabled() is False
    request = CachePlanePlacementRequest(
        "request", "lineage", CachePlaneKind.ATTENTION_KV, "cache_pack", geometry()
    )
    decision = CachePlanePlacementPolicy(placement_profile()).decide(request)
    assert decision.engine is None
    assert decision.reason == "disabled"


def test_placement_uses_measured_host_capability_not_ple_hints():
    policy = CachePlanePlacementPolicy(placement_profile(), enabled=True)
    plain = CachePlanePlacementRequest(
        "plain", "lineage", CachePlaneKind.ATTENTION_KV, "cache_pack", geometry()
    )
    hinted = replace(
        plain,
        request_id="hinted",
        ple_hints=PLEResidencyHints("v1", (1, 2, 3), "nvme-index"),
    )
    plain_decision = policy.decide(plain)
    hinted_decision = policy.decide(hinted)
    assert plain_decision.engine == Engine.CPU
    assert hinted_decision.engine == plain_decision.engine
    assert plain_decision.advisory_ple_hints_seen is False
    assert hinted_decision.advisory_ple_hints_seen is True


def test_placement_refuses_unmeasured_geometry():
    policy = CachePlanePlacementPolicy(placement_profile(), enabled=True)
    request = CachePlanePlacementRequest(
        "large",
        "lineage",
        CachePlaneKind.ATTENTION_KV,
        "cache_pack",
        geometry(512),
    )
    decision = policy.decide(request)
    assert decision.engine is None
    assert decision.reason.startswith("unqualified:")


def test_placement_evidence_is_bound_to_cache_plane_kind():
    policy = CachePlanePlacementPolicy(placement_profile(), enabled=True)
    request = CachePlanePlacementRequest(
        "gdn",
        "lineage",
        CachePlaneKind.GDN_RECURRENT,
        "cache_pack",
        geometry(),
    )
    decision = policy.decide(request)
    assert decision.engine is None
    assert decision.reason.startswith("unqualified:")


def test_policy_metrics_are_zero_sync_counters():
    metrics = CachePolicyMetrics()
    policy = CachePlanePlacementPolicy(
        placement_profile(), enabled=True, metrics=metrics
    )
    policy.decide(
        CachePlanePlacementRequest(
            "request",
            "lineage",
            CachePlaneKind.ATTENTION_KV,
            "cache_pack",
            geometry(),
        )
    )
    snapshot = metrics.snapshot()
    assert snapshot["placement_decisions"] == 1
    assert snapshot["device_synchronizations"] == 0
    assert snapshot["timed_hot_path_sections"] == 0


def test_store_is_default_off_and_refuses_unstable_or_mutable_data(tmp_path):
    disabled = ContentAddressedPlaneStore(tmp_path)
    with pytest.raises(PersistenceRefused, match="disabled"):
        disabled.put(stable_record())
    store = ContentAddressedPlaneStore(tmp_path, enabled=True)
    with pytest.raises(PersistenceRefused, match="unstable_plane"):
        store.put(stable_record(stable=False))
    with pytest.raises(PersistenceRefused, match="mutable_request_data"):
        store.put(stable_record(mutable=True))
    assert not tuple(tmp_path.rglob("*.plane"))


@pytest.mark.parametrize("codec_id", (IdentityCodec.codec_id, ZlibCodec.codec_id))
def test_default_lossless_codecs_round_trip_and_deduplicate(tmp_path, codec_id):
    metrics = CachePolicyMetrics()
    store = ContentAddressedPlaneStore(
        tmp_path, enabled=True, metrics=metrics
    )
    record = stable_record(payload=b"abcdef" * 100)
    content_id = store.put(record, codec_id=codec_id)
    assert store.put(record, codec_id=codec_id) == content_id
    loaded = store.load(content_id, expected_fingerprint=record.fingerprint)
    assert loaded.payload == record.payload
    assert loaded.lossless is True
    assert loaded.codec_id == codec_id
    snapshot = metrics.snapshot()
    assert snapshot["persistence_writes"] == 1
    assert snapshot["persistence_deduplications"] == 1
    assert snapshot["persistence_reads"] == 1


def test_content_address_binds_payload_and_plane_fingerprint(tmp_path):
    store = ContentAddressedPlaneStore(tmp_path, enabled=True)
    first = stable_record(payload=b"one", suffix="a")
    second = stable_record(payload=b"two", suffix="a")
    third = stable_record(payload=b"one", suffix="b")
    assert len({store.put(first), store.put(second), store.put(third)}) == 3


def test_load_rejects_wrong_fingerprint_and_corruption(tmp_path):
    store = ContentAddressedPlaneStore(tmp_path, enabled=True)
    record = stable_record()
    content_id = store.put(record)
    with pytest.raises(IntegrityError, match="fingerprint mismatch"):
        store.load(
            content_id,
            expected_fingerprint=plane_fingerprint(
                CachePlaneKind.PROMPT_HOST, "wrong"
            ),
        )
    path = tmp_path / content_id[:2] / f"{content_id}.plane"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(IntegrityError, match="content digest"):
        store.load(content_id, expected_fingerprint=record.fingerprint)


def test_store_checks_a_codec_that_claims_to_be_lossless(tmp_path):
    codec = BadLosslessCodec()
    store = ContentAddressedPlaneStore(
        tmp_path, enabled=True, codecs={codec.codec_id: codec}
    )
    with pytest.raises(PersistenceRefused, match="changed_payload"):
        store.put(stable_record(), codec_id=codec.codec_id)


def compression_fixture(*, enabled=True):
    record = stable_record(
        CachePlaneKind.ATTENTION_KV, payload=b"0123456789"
    )
    codec = TruncateColdKVCodec()
    gate = AccuracyGate(record.model_fingerprint, 10, 0.01)
    policy = ColdKVCompressionPolicy(gate, enabled=enabled)
    evidence = AccuracyEvidence(
        record.model_fingerprint,
        record.fingerprint.digest,
        codec.codec_id,
        10,
        0.005,
        True,
    )
    return record, codec, policy, evidence


def test_cold_kv_compression_is_experimental_and_accuracy_gated():
    record, codec, disabled, evidence = compression_fixture(enabled=False)
    assert disabled.admit(
        kind=record.kind, codec=codec, evidence=evidence
    ).reason == "disabled"
    _, _, policy, evidence = compression_fixture()
    assert not policy.admit(
        kind=record.kind,
        codec=codec,
        evidence=replace(evidence, samples=9),
    ).accepted
    assert not policy.admit(
        kind=record.kind,
        codec=codec,
        evidence=replace(evidence, max_abs_logit_error=0.02),
    ).accepted
    assert not policy.admit(
        kind=record.kind,
        codec=codec,
        evidence=replace(evidence, exact_tokens=False),
    ).accepted
    assert policy.admit(
        kind=record.kind, codec=codec, evidence=evidence
    ).accepted
    with pytest.raises(ValueError, match="finite"):
        replace(evidence, max_abs_logit_error=float("nan"))


def test_admitted_cold_kv_codec_is_bound_to_model_plane_and_codec(tmp_path):
    record, codec, policy, evidence = compression_fixture()
    admission = policy.admit(kind=record.kind, codec=codec, evidence=evidence)
    store = ContentAddressedPlaneStore(
        tmp_path, enabled=True, codecs={codec.codec_id: codec}
    )
    content_id = store.put(
        record,
        codec_id=codec.codec_id,
        compression_admission=admission,
    )
    with pytest.raises(PersistenceRefused, match="revalidation_required"):
        store.load(content_id, expected_fingerprint=record.fingerprint)
    loaded = store.load(
        content_id,
        expected_fingerprint=record.fingerprint,
        compression_admission=admission,
    )
    assert loaded.lossless is False
    assert loaded.payload == record.payload[::2]
    with pytest.raises(PersistenceRefused, match="compression_model_mismatch"):
        store.put(
            replace(record, model_fingerprint="other-model"),
            codec_id=codec.codec_id,
            compression_admission=admission,
        )


def test_lossy_codec_is_never_admitted_for_non_kv_plane(tmp_path):
    kv_record, codec, policy, evidence = compression_fixture()
    admission = policy.admit(
        kind=kv_record.kind, codec=codec, evidence=evidence
    )
    prompt = stable_record(CachePlaneKind.PROMPT_HOST)
    store = ContentAddressedPlaneStore(
        tmp_path, enabled=True, codecs={codec.codec_id: codec}
    )
    with pytest.raises(PersistenceRefused, match="not_cold_kv"):
        store.put(
            prompt,
            codec_id=codec.codec_id,
            compression_admission=replace(
                admission,
                plane_fingerprint_digest=prompt.fingerprint.digest,
            ),
        )
