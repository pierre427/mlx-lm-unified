from mlx_lm.spomin_pressure import (
    AdaptiveSpominPressurePolicy,
    SpominPressureConfig,
    SpominPressureSample,
)


GiB = 1 << 30


def policy(**overrides):
    values = dict(
        max_context_tokens=100_000,
        min_context_tokens=8_000,
        bytes_per_token_per_session=64 * 1024,
        max_total_active_kv_bytes=8 * GiB,
        system_reserve_bytes=16 * GiB,
        critical_available_bytes=4 * GiB,
        queued_lane_weight=0.5,
        pressure_samples=2,
        recovery_samples=3,
    )
    values.update(overrides)
    return AdaptiveSpominPressurePolicy(
        SpominPressureConfig(**values), memory_reader=lambda: None
    )


def sample(**overrides):
    values = dict(
        current_context_tokens=90_000,
        active_sessions=1,
        queued_sessions=0,
        active_kv_bytes=6 * GiB,
        idle_apc_reclaimable_bytes=0,
        available_system_bytes=32 * GiB,
    )
    values.update(overrides)
    return SpominPressureSample(**values)


def test_healthy_single_session_gets_the_maximum_context_budget():
    decision = policy().observe(sample())
    assert decision.context_limit_tokens == 100_000
    assert decision.action == "headroom"
    assert decision.reasons == ("healthy_single_session",)


def test_idle_apc_is_reclaimed_before_active_context_is_compacted():
    controller = policy()
    decision = controller.observe(
        sample(
            available_system_bytes=14 * GiB,
            idle_apc_reclaimable_bytes=4 * GiB,
        )
    )
    assert decision.reclaim_idle_apc_bytes == 2 * GiB
    assert decision.context_limit_tokens == 100_000
    assert decision.action == "evict_idle_apc"
    assert decision.reasons[0] == "idle_apc_first"


def test_sustained_batch_pressure_compacts_but_one_arrival_does_not():
    controller = policy()
    crowded = sample(active_sessions=4, queued_sessions=2)
    first = controller.observe(crowded)
    second = controller.observe(crowded)

    assert first.context_limit_tokens == 100_000
    assert first.action == "headroom"
    assert second.context_limit_tokens == 26_214
    assert second.action == "compact"
    assert "batch" in second.reasons


def test_transient_queue_pressure_clears_without_compaction():
    controller = policy()
    controller.observe(sample(active_sessions=1, queued_sessions=4))
    healthy = controller.observe(sample())
    assert healthy.context_limit_tokens == 100_000
    assert healthy.action == "headroom"


def test_critical_memory_pressure_bypasses_downshift_hysteresis():
    controller = policy()
    decision = controller.observe(
        sample(
            current_context_tokens=50_000,
            available_system_bytes=2 * GiB,
            active_kv_bytes=2 * GiB,
        )
    )
    assert decision.context_limit_tokens == 8_000
    assert decision.action == "compact"
    assert "critical" in decision.reasons


def test_recovery_requires_more_samples_than_pressure():
    controller = policy()
    crowded = sample(active_sessions=4, queued_sessions=2)
    controller.observe(crowded)
    controller.observe(crowded)
    assert controller.context_limit_tokens == 26_214

    assert controller.observe(sample()).context_limit_tokens == 26_214
    assert controller.observe(sample()).context_limit_tokens == 26_214
    assert controller.observe(sample()).context_limit_tokens == 100_000
