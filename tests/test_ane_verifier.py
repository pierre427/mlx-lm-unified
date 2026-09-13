import numpy as np

from mlx_lm.ane_verifier import (
    ANEVerifierConfig,
    ANEVerifierController,
    ANEVerifierStamp,
    ane_verifier_eligibility,
)


class FakeRunner:
    hidden_size = 8
    package_bytes = 1024

    def __init__(self, payload=None):
        self.payload = payload or {
            "ok": True,
            "token_ids": [7, 3, 2, 1],
            "scores": [4.0, 3.0, 2.0, 1.0],
            "part_ms": [1.0],
        }
        self.submitted = {}
        self.abandoned = []
        self.closed = False

    def submit(self, ticket_id, hidden):
        self.submitted[ticket_id] = hidden

    def receive(self, ticket_id, timeout_s):
        if self.payload is None:
            return None
        result = dict(self.payload)
        result.update(ticket_id=ticket_id, started_ns=20, finished_ns=30)
        return result

    def abandon(self, ticket_id):
        self.abandoned.append(ticket_id)

    def close(self):
        self.closed = True


def stamp(generation=1):
    return ANEVerifierStamp(4, (10,), (generation,), (2,))


def test_eligibility_uses_measured_marginal_delay_and_overlap():
    config = ANEVerifierConfig(
        mode="active",
        allow_approximate_commit=True,
        min_gpu_marginal_delay_ms=1.0,
        qualified_service_p95_ms=10.0,
        min_net_savings_ms=0.1,
    )
    common = dict(
        greedy=True,
        has_logits_processors=False,
        needs_full_logprobs=False,
        package_resident=True,
        memory_headroom_gib=4.0,
        package_gib=1.2,
    )
    low_cost = ane_verifier_eligibility(
        config,
        predicted_gpu_marginal_delay_ms=0.5,
        available_overlap_ms=20.0,
        expected_remaining_verifications=64,
        **common,
    )
    assert not low_cost.eligible
    assert low_cost.reason == "gpu_marginal_cost_is_low"
    no_overlap = ane_verifier_eligibility(
        config,
        predicted_gpu_marginal_delay_ms=3.0,
        available_overlap_ms=2.0,
        expected_remaining_verifications=64,
        **common,
    )
    assert not no_overlap.eligible
    assert no_overlap.reason == "predicted_net_savings_too_low"
    assert ane_verifier_eligibility(
        config,
        predicted_gpu_marginal_delay_ms=3.0,
        available_overlap_ms=20.0,
        expected_remaining_verifications=64,
        **common,
    ).eligible


def test_only_plain_greedy_without_full_logprobs_is_eligible():
    config = ANEVerifierConfig(
        mode="active",
        allow_approximate_commit=True,
        qualified_service_p95_ms=10.0,
        min_net_savings_ms=0.1,
    )
    base = dict(
        greedy=True,
        has_logits_processors=False,
        needs_full_logprobs=False,
        package_resident=True,
        memory_headroom_gib=4.0,
        package_gib=1.0,
        predicted_gpu_marginal_delay_ms=3.0,
        available_overlap_ms=10.0,
        expected_remaining_verifications=64,
    )
    for field, reason in (
        ("greedy", "non_greedy"),
        ("has_logits_processors", "logits_processors"),
        ("needs_full_logprobs", "full_logprobs_requested"),
    ):
        changed = dict(base)
        changed[field] = not changed[field]
        assert ane_verifier_eligibility(config, **changed).reason == reason


def test_fixed_pipeline_drain_is_amortized_over_remaining_work():
    config = ANEVerifierConfig(
        mode="active",
        allow_approximate_commit=True,
        qualified_service_p95_ms=10.0,
        host_overhead_ms=0.1,
        min_net_savings_ms=0.2,
    )
    common = dict(
        greedy=True,
        has_logits_processors=False,
        needs_full_logprobs=False,
        package_resident=True,
        memory_headroom_gib=4.0,
        package_gib=1.2,
        predicted_gpu_marginal_delay_ms=0.84,
        predicted_ane_interference_ms=0.2,
        available_overlap_ms=27.0,
    )
    short = ane_verifier_eligibility(
        config, expected_remaining_verifications=12, **common
    )
    assert not short.eligible
    sustained = ane_verifier_eligibility(
        config, expected_remaining_verifications=64, **common
    )
    assert sustained.eligible
    assert sustained.predicted_net_savings_ms > 0.2


def test_same_cycle_overlap_has_no_final_pipeline_drain():
    config = ANEVerifierConfig(
        mode="active",
        allow_approximate_commit=True,
        qualified_service_p95_ms=10.0,
        min_net_savings_ms=0.2,
    )
    decision = ane_verifier_eligibility(
        config,
        greedy=True,
        has_logits_processors=False,
        needs_full_logprobs=False,
        package_resident=True,
        memory_headroom_gib=4.0,
        package_gib=1.2,
        predicted_gpu_marginal_delay_ms=0.84,
        predicted_ane_interference_ms=0.2,
        available_overlap_ms=12.0,
        expected_remaining_verifications=1,
        predicted_final_drain_ms=0.0,
    )
    assert decision.eligible


def test_ticket_resolves_only_for_same_generation_and_sufficient_margin():
    runner = FakeRunner()
    controller = ANEVerifierController(
        ANEVerifierConfig(mode="active", min_margin=0.5, allow_approximate_commit=True),
        runner,
    )
    ticket = controller.submit(stamp(), np.zeros((1, 1, 8), dtype=np.float16))
    assert tuple(runner.submitted[ticket.ticket_id].shape) == (1, 1, 8)
    result = controller.resolve(ticket, stamp())
    assert result.token_ids[0] == 7
    assert result.margin == 1.0
    assert controller.stats.counts["resolved"] == 1

    stale = controller.submit(stamp(), np.zeros((1, 1, 8), dtype=np.float16))
    assert controller.resolve(stale, stamp(2)) is None
    assert stale.ticket_id in runner.abandoned
    assert controller.stats.counts["fallback_stale"] == 1


def test_low_margin_falls_back_and_close_abandons_inflight():
    runner = FakeRunner(
        {
            "ok": True,
            "token_ids": [7, 3, 2, 1],
            "scores": [4.0, 3.95, 2.0, 1.0],
            "part_ms": [1.0],
        }
    )
    controller = ANEVerifierController(
        ANEVerifierConfig(
            mode="active", min_margin=0.125, allow_approximate_commit=True
        ),
        runner,
    )
    ticket = controller.submit(stamp(), np.zeros((1, 1, 8), dtype=np.float16))
    assert controller.resolve(ticket, stamp()) is None
    assert controller.stats.counts["fallback_low_margin"] == 1
    pending = controller.submit(stamp(), np.zeros((1, 1, 8), dtype=np.float16))
    controller.close()
    assert pending.ticket_id in runner.abandoned
    assert runner.closed


def test_active_commit_requires_explicit_approximate_opt_in():
    import pytest

    with pytest.raises(ValueError, match="allow_approximate_commit"):
        ANEVerifierConfig(mode="active").validated()
    assert ANEVerifierConfig(mode="shadow").validated().mode == "shadow"


def test_shadow_mode_records_a_result_without_authorizing_it():
    runner = FakeRunner()
    controller = ANEVerifierController(
        ANEVerifierConfig(mode="shadow", min_margin=0.5), runner
    )
    ticket = controller.submit(stamp(), np.zeros((1, 1, 8), dtype=np.float16))
    assert controller.resolve(ticket, stamp()) is None
    assert controller.stats.counts["shadow_resolved"] == 1
    assert controller.stats.counts["resolved"] == 0
