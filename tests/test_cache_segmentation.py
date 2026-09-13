from dataclasses import replace

import pytest

from mlx_lm.cache_planes import CachePlaneKind
from mlx_lm.cache_segmentation import (
    CalibrationResult,
    PerPlaneSegmentationPolicy,
    PlaneSegmentCandidate,
    Qwen4CacheGeometry,
    SegmentValueCandidate,
    SegmentationPolicyConfig,
    SyntheticServingTrace,
    TraceCalibrator,
    pareto_front,
    qsa_window,
    recompute_cost_proxy_us,
    select_protected_segment_hotset,
)


def candidate(
    kind=CachePlaneKind.ATTENTION_KV,
    *,
    tokens=2051,
    nbytes=1_000_000,
    entropy=2.0,
    depth=1,
    fanout=2,
    stable=True,
    mutable=False,
):
    return PlaneSegmentCandidate(
        kind,
        0,
        tokens,
        nbytes,
        entropy,
        depth,
        fanout,
        1000.0,
        17,
        stable,
        mutable,
    )


def budgets():
    return {
        CachePlaneKind.PROMPT_HOST: 1 << 20,
        CachePlaneKind.ATTENTION_KV: 256 << 20,
        CachePlaneKind.ATTENTION_RING: 64 << 20,
        CachePlaneKind.QSA_SUMMARY: 64 << 20,
        CachePlaneKind.GDN_RECURRENT: 128 << 20,
        CachePlaneKind.MTP_DRAFT: 64 << 20,
    }


def result(label, *, latency, memory, fragmentation, hits, avoided):
    return CalibrationResult(
        label,
        "trace",
        1,
        latency,
        0.0,
        100,
        50,
        2.0,
        int(hits * 10),
        int((1 - hits) * 10),
        hits,
        0,
        0.0,
        avoided,
        fragmentation,
        0,
        0,
        memory,
        0.0,
        0,
    )


def test_qwen4_geometry_matches_model_configuration():
    geometry = Qwen4CacheGeometry()
    assert geometry.layers == 48
    assert geometry.full_attention_layers == 12
    assert geometry.linear_attention_layers == 36
    assert geometry.qsa_block_tokens == 4
    assert geometry.block_topk == 512
    assert geometry.dense_shortcircuit_max_tokens == 2051
    assert geometry.full_kv_bytes_per_token == 24_576
    assert geometry.mtp_kv_bytes_per_token == 2_048
    assert geometry.gdn_recurrent_bytes == 56_623_104
    assert geometry.qsa_bytes_per_block == 3_072


def test_qsa_dense_boundary_is_2048_through_2051_not_an_arbitrary_page():
    geometry = Qwen4CacheGeometry()
    at_budget = qsa_window(2048, geometry)
    at_dense_max = qsa_window(2051, geometry)
    first_sparse = qsa_window(2052, geometry)
    assert (at_budget.complete_blocks, at_budget.incomplete_tail_tokens) == (512, 0)
    assert (at_dense_max.complete_blocks, at_dense_max.incomplete_tail_tokens) == (
        512,
        3,
    )
    assert at_budget.dense_shortcircuit is True
    assert at_dense_max.dense_shortcircuit is True
    assert first_sparse.complete_blocks == 513
    assert first_sparse.dense_shortcircuit is False


def test_segmentation_is_default_off():
    plan = PerPlaneSegmentationPolicy().decide(candidate())
    assert plan.reason == "disabled"
    assert plan.segments == ()
    assert plan.retention_eligible is False


def test_all_stable_boundaries_align_to_four_tokens_and_keep_2051_tail():
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig.uniform(64, enabled=True)
    )
    plan = policy.decide(candidate(tokens=2051))
    assert plan.incomplete_tail_tokens == 3
    assert plan.segments[-1].token_stop == 2048
    for segment in plan.segments:
        assert segment.token_start % 4 == 0
        assert segment.token_stop % 4 == 0
        assert segment.token_stop - segment.token_start <= 256


@pytest.mark.parametrize("tokens", tuple(range(2044, 2057)))
def test_boundary_property_never_checkpoints_an_incomplete_qsa_block(tokens):
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig.uniform(16, enabled=True)
    )
    plan = policy.decide(candidate(tokens=tokens))
    assert sum(
        segment.token_stop - segment.token_start for segment in plan.segments
    ) + plan.incomplete_tail_tokens == tokens
    assert all(segment.token_stop % 4 == 0 for segment in plan.segments)


def test_tiny_and_high_entropy_states_bypass_compression():
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig(enabled=True, inline_threshold_bytes=2048)
    )
    tiny = policy.decide(candidate(nbytes=2048, entropy=1.0))
    noisy = policy.decide(candidate(nbytes=1_000_000, entropy=7.9))
    compressible = policy.decide(candidate(nbytes=1_000_000, entropy=2.0))
    assert tiny.inline is True and tiny.codec_id is None
    assert len(tiny.segments) == 1
    assert tiny.fragmentation_bytes == 0
    assert noisy.inline is False and noisy.codec_id is None
    assert noisy.reason == "segmented_high_entropy_no_codec"
    assert compressible.codec_id == "zlib-v1"


def test_qsa_tail_does_not_discard_complete_block_storage():
    geometry = Qwen4CacheGeometry()
    logical_bytes = geometry.plane_bytes(CachePlaneKind.QSA_SUMMARY, 2051)
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig.uniform(64, enabled=True), geometry
    )
    plan = policy.decide(
        candidate(
            CachePlaneKind.QSA_SUMMARY,
            tokens=2051,
            nbytes=logical_bytes,
        )
    )
    assert sum(segment.logical_bytes for segment in plan.segments) == logical_bytes


def test_delta_depth_fragmentation_fanout_and_mutability_are_explicit():
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig.uniform(
            16,
            enabled=True,
            max_delta_depth=4,
            max_fragmentation_ratio=1.0,
        )
    )
    plan = policy.decide(candidate(depth=4, fanout=4, mutable=True))
    assert plan.compact_delta is True
    assert plan.fragmentation_bytes > 0
    assert plan.shared_branch_bytes_avoided == 3_000_000
    assert plan.retention_eligible is False


def test_trace_is_deterministic_serializable_and_contains_real_boundaries():
    first = SyntheticServingTrace.generate(seed=9, sessions=8, turns=7)
    second = SyntheticServingTrace.generate(seed=9, sessions=8, turns=7)
    assert first.digest == second.digest
    assert SyntheticServingTrace.from_json(first.to_json()) == first
    assert {event.token_count for event in first.events}.issuperset(
        {2048, 2051, 2052}
    )
    assert {event.kind for event in first.events}.issuperset(
        {
            CachePlaneKind.PROMPT_HOST,
            CachePlaneKind.ATTENTION_KV,
            CachePlaneKind.ATTENTION_RING,
            CachePlaneKind.QSA_SUMMARY,
            CachePlaneKind.GDN_RECURRENT,
            CachePlaneKind.MTP_DRAFT,
        }
    )


def test_trace_payload_recipe_is_replayable_and_entropy_sensitive():
    trace = SyntheticServingTrace.generate(seed=11, sessions=1, turns=1)
    low = next(event for event in trace.events if event.entropy_class == "low")
    assert low.sample_payload() == low.sample_payload()
    high = replace(low, entropy_class="high")
    assert high.sample_payload() == high.sample_payload()
    assert len(set(high.sample_payload())) > len(set(low.sample_payload()))


def test_recompute_proxy_treats_gdn_as_one_state_plus_serial_replay():
    geometry = Qwen4CacheGeometry()
    short = recompute_cost_proxy_us(
        CachePlaneKind.GDN_RECURRENT,
        128,
        geometry.gdn_recurrent_bytes,
    )
    long = recompute_cost_proxy_us(
        CachePlaneKind.GDN_RECURRENT,
        8192,
        geometry.gdn_recurrent_bytes,
    )
    assert 1_000 < short < 2_000
    assert 7_000 < long < 8_000


def test_protected_hotset_uses_saved_work_per_byte_and_rejects_mutable_state():
    candidates = (
        SegmentValueCandidate("large", 8, 100, 2, 800),
        SegmentValueCandidate("dense", 5, 100, 2, 200),
        SegmentValueCandidate(
            "mutable", 100, 100, 4, 100, contains_mutable_request_data=True
        ),
        SegmentValueCandidate("unstable", 100, 100, 4, 100, stable=False),
    )
    hotset = select_protected_segment_hotset(candidates, byte_budget=800)
    assert hotset.segment_ids == ("dense",)
    assert hotset.resident_bytes == 200
    assert hotset.expected_value_us == 1_000


def test_protected_hotset_is_deterministic_and_byte_bounded():
    candidates = (
        SegmentValueCandidate("b", 1, 10, 1, 10),
        SegmentValueCandidate("a", 1, 10, 1, 10),
        SegmentValueCandidate("c", 1, 10, 1, 10),
    )
    hotset = select_protected_segment_hotset(candidates, byte_budget=20)
    assert hotset.segment_ids == ("a", "b")
    assert hotset.resident_bytes == 20


def test_calibration_reports_traffic_memory_and_recomputation_metrics():
    trace = SyntheticServingTrace.generate(seed=21, sessions=8, turns=7)
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig.uniform(64, enabled=True)
    )
    measured = TraceCalibrator(policy, budgets()).run(trace, label="blocks-64")
    assert measured.trace_digest == trace.digest
    assert measured.events == len(trace.events)
    assert measured.host_latency_ms > 0
    assert measured.compression_cost_ms > 0
    assert measured.bytes_before > measured.bytes_after
    assert measured.compression_ratio > 1
    assert measured.segment_hits > 0
    assert measured.segment_misses > 0
    assert 0 < measured.hit_rate < 1
    assert measured.partial_hit_events > 0
    assert measured.recomputation_avoided_us > 0
    assert measured.fragmentation_bytes > 0
    assert measured.compactions > 0
    assert measured.shared_branch_bytes_avoided > 0
    assert measured.retained_bytes <= sum(budgets().values())
    assert measured.p95_reuse_distance >= measured.mean_reuse_distance


def test_pareto_front_drops_strictly_dominated_candidates():
    champion = result(
        "champion",
        latency=1,
        memory=10,
        fragmentation=1,
        hits=0.8,
        avoided=100,
    )
    dominated = result(
        "dominated",
        latency=2,
        memory=20,
        fragmentation=2,
        hits=0.7,
        avoided=90,
    )
    tradeoff = result(
        "tradeoff",
        latency=0.5,
        memory=30,
        fragmentation=3,
        hits=0.6,
        avoided=80,
    )
    assert pareto_front((champion, dominated, tradeoff)) == (
        "champion",
        "tradeoff",
    )
