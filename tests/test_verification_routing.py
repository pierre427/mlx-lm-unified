import pytest

from mlx_lm.verification_routing import PendingVerification, select_ane_verification


def pending(name, probability, age, *, exact=True):
    return PendingVerification(name, probability, age, 1.0, exact)


def test_route_is_default_off_and_exact_only():
    rows = [pending("likely", 0.9, 20), pending("inexact", 1.0, 30, exact=False)]
    assert (
        select_ane_verification(
            rows, enabled=False, memory_pressure=True, gpu_verify_delay_ms=100
        ).selected_batch_id
        is None
    )
    decision = select_ane_verification(
        rows, enabled=True, memory_pressure=True, gpu_verify_delay_ms=100
    )
    assert decision.selected_batch_id == "likely"
    assert decision.ranked_batch_ids == ("likely",)


def test_most_likely_wins_until_starvation_guard_fires():
    decision = select_ane_verification(
        [pending("likely", 0.9, 20), pending("older", 0.5, 100)],
        enabled=True,
        memory_pressure=True,
        gpu_verify_delay_ms=100,
    )
    assert decision.selected_batch_id == "likely"
    starved = select_ane_verification(
        [pending("likely", 0.99, 20), pending("starved", 0.1, 260)],
        enabled=True,
        memory_pressure=True,
        gpu_verify_delay_ms=100,
    )
    assert starved.selected_batch_id == "starved"


def test_timely_gpu_keeps_work_on_gpu_without_memory_pressure():
    decision = select_ane_verification(
        [pending("batch", 0.9, 20)],
        enabled=True,
        memory_pressure=False,
        gpu_verify_delay_ms=5,
    )
    assert decision.reason == "gpu_service_is_timely"
    assert decision.selected_batch_id is None


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError, match="acceptance_probability"):
        pending("bad", 1.1, 0)


def test_memory_pressure_does_not_replace_a_measured_crossover():
    decision = select_ane_verification(
        [pending("batch", 0.9, 20)],
        enabled=True,
        memory_pressure=True,
        gpu_verify_delay_ms=1,
        min_gpu_delay_ms=2,
    )
    assert decision.reason == "gpu_service_is_timely"
    assert decision.selected_batch_id is None


def test_residency_headroom_and_overlap_are_hard_gates():
    common = dict(
        pending=[pending("batch", 0.9, 20)],
        enabled=True,
        memory_pressure=True,
        gpu_verify_delay_ms=3,
        min_gpu_delay_ms=2,
        ane_package_gib=1.2,
        ane_service_p95_ms=11,
    )
    assert (
        select_ane_verification(**common, ane_package_resident=False).reason
        == "ane_package_not_resident"
    )
    assert (
        select_ane_verification(**common, ane_memory_headroom_gib=1.0).reason
        == "ane_package_exceeds_headroom"
    )
    assert (
        select_ane_verification(
            **common, ane_memory_headroom_gib=2.0, available_overlap_ms=8
        ).reason
        == "ane_service_not_hidden"
    )
    assert (
        select_ane_verification(
            **common, ane_memory_headroom_gib=2.0, available_overlap_ms=20
        ).selected_batch_id
        == "batch"
    )
