import argparse
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "benchmarks" / "qwen4_mtp_dynamic_join_gate.py"
SPEC = importlib.util.spec_from_file_location("qwen4_mtp_dynamic_join_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = dict(
        execute=False,
        model="/definitely/missing",
        context=16384,
        lanes=2,
        initial_lanes=1,
        static_cohort=False,
        join_after_cycles=4,
        max_tokens=128,
        num_draft=2,
        reps=3,
        prefill_step_size=512,
        cooldown_seconds=0.0,
        minimum_system_free_percent=25,
        maximum_swap_growth_mb=16.0,
        max_drift=0.05,
        cancel_uid=None,
        cancel_after_tokens=32,
        seed=20260910,
        share_qsa_indices=True,
        capture_logprob_envelopes=False,
        out="unused.json",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plan_is_paired_rotated_and_model_free():
    plan = MODULE.build_plan(args())
    assert plan["arms"] == list(MODULE.ARMS)
    assert plan["orders"] == [
        ["fixed_cohort", "dynamic_join"],
        ["dynamic_join", "fixed_cohort"],
        ["fixed_cohort", "dynamic_join"],
    ]
    assert plan["joining_lanes"] == 1
    assert not plan["execution_authorized"]


def test_static_plan_requires_and_describes_full_initial_cohort():
    plan = MODULE.build_plan(
        args(lanes=4, initial_lanes=4, static_cohort=True, max_tokens=8)
    )
    assert plan["static_cohort"] is True
    assert plan["joining_lanes"] == 0
    assert "static B4" in plan["correctness_gate"]
    with pytest.raises(ValueError, match="initial-lanes == lanes"):
        MODULE.build_plan(args(lanes=4, initial_lanes=2, static_cohort=True))


@pytest.mark.parametrize(
    "overrides",
    [
        {"context": 0},
        {"lanes": 1},
        {"initial_lanes": 0},
        {"initial_lanes": 2},
        {"join_after_cycles": 0},
        {"max_tokens": 12},
        {"num_draft": 1},
        {"num_draft": 0, "share_qsa_indices": False},
        {"reps": 0},
        {"cooldown_seconds": -1.0},
        {"minimum_system_free_percent": -1},
        {"minimum_system_free_percent": 101},
        {"maximum_swap_growth_mb": -1},
        {"max_drift": -0.1},
        {"max_drift": 1.1},
        {"cancel_uid": -1},
        {"cancel_uid": 2},
        {"cancel_uid": 0, "cancel_after_tokens": 1},
        {"cancel_uid": 0, "cancel_after_tokens": 128},
    ],
)
def test_plan_refuses_invalid_schedule(overrides):
    with pytest.raises(ValueError):
        MODULE.build_plan(args(**overrides))


def test_summary_requires_exact_and_post_join_qsa_engagement():
    rows = [
        {
            "arm": "fixed_cohort",
            "transaction_wall_s": 2.0,
            "aggregate_decode_tps": 100.0,
            "qsa_share": {
                "post_join_share_requested": 0,
                "post_join_reuse_observed": 0,
            },
        },
        {
            "arm": "dynamic_join",
            "transaction_wall_s": 2.5,
            "aggregate_decode_tps": 80.0,
            "exact_fixed_match": True,
            "qsa_share": {
                "post_join_share_requested": 3,
                "post_join_reuse_observed": 3,
            },
        },
    ]
    summary = MODULE.summarize(rows)
    assert summary["dynamic_exact_matches"] == 1
    assert summary["dynamic_qsa_engaged"] == 1


def test_summary_excludes_thermally_discarded_pair_from_medians():
    rows = [
        {
            "arm": "fixed_cohort",
            "repetition": 1,
            "transaction_wall_s": 1.0,
            "aggregate_decode_tps": 100.0,
            "drift_accepted": True,
            "qsa_share": {"post_join_share_requested": 0, "post_join_reuse_observed": 0},
        },
        {
            "arm": "dynamic_join",
            "repetition": 1,
            "transaction_wall_s": 1.1,
            "aggregate_decode_tps": 90.0,
            "drift_accepted": True,
            "exact_fixed_match": True,
            "qsa_share": {"post_join_share_requested": 1, "post_join_reuse_observed": 1},
        },
        {
            "arm": "fixed_cohort",
            "repetition": 2,
            "transaction_wall_s": 9.0,
            "aggregate_decode_tps": 10.0,
            "drift_accepted": False,
            "qsa_share": {"post_join_share_requested": 0, "post_join_reuse_observed": 0},
        },
        {
            "arm": "dynamic_join",
            "repetition": 2,
            "transaction_wall_s": 9.0,
            "aggregate_decode_tps": 10.0,
            "drift_accepted": False,
            "exact_fixed_match": True,
            "qsa_share": {"post_join_share_requested": 1, "post_join_reuse_observed": 1},
        },
    ]
    summary = MODULE.summarize(rows)
    assert summary["accepted_repetitions"] == 1
    assert summary["fixed_cohort"]["accepted_samples"] == 1
    assert summary["fixed_cohort"]["discarded_samples"] == 1
    assert summary["fixed_cohort"]["median_aggregate_decode_tps"] == 100.0
    assert summary["dynamic_join"]["median_aggregate_decode_tps"] == 90.0


def test_first_divergence_handles_content_and_length():
    assert MODULE.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert MODULE.first_divergence([1, 2], [1, 2, 3]) == 2
    assert MODULE.first_divergence([1, 2], [1, 2]) is None


def test_first_envelope_flip_requires_top_two_membership_and_bounded_margin():
    reference = [{"top1_token": 4, "top2_token": 9, "top2_margin": 0.02, "scale": 10.0}]
    candidate = [{"top1_token": 9, "top2_token": 4, "top2_margin": 0.01, "scale": 10.0}]
    receipt = MODULE.classify_first_envelope_flip([4], [9], reference, candidate)
    assert receipt["position"] == 0
    assert receipt["near_tie_candidate"]
    candidate[0]["top2_margin"] = 0.04
    assert not MODULE.classify_first_envelope_flip([4], [9], reference, candidate)["near_tie_candidate"]


def test_warmup_compiles_shapes_without_requiring_measured_cancellation():
    measured = args(
        context=16384,
        max_tokens=64,
        cancel_uid=0,
        cancel_after_tokens=32,
    )
    warm = MODULE.build_warmup_args(measured)
    assert warm.context == 256
    assert warm.max_tokens == 32
    assert warm.cancel_uid is None
    assert measured.cancel_uid == 0


def test_token_id_padding_preserves_stable_suffix():
    class FakeTokenizer:
        @staticmethod
        def encode(_text, add_special_tokens=False):
            assert add_special_tokens is False
            return [7]

    result = MODULE.fill_ids_before_stable_suffix(
        FakeTokenizer(), [1, 2, 10, 11], [1, 2, 3, 10, 11], 6
    )
    assert result == [1, 2, 7, 7, 10, 11]


def test_system_guard_parsers_and_thresholds():
    snapshot = {
        "memory_pressure": {"stdout": "System-wide memory free percentage: 42%"},
        "swapusage": {"stdout": "total = 8.00G used = 1.50G free = 6.50G"},
        "pmset_therm": {
            "returncode": 0,
            "stdout": "Note: No thermal warning level has been recorded",
        },
    }
    assert MODULE.free_percent(snapshot) == 42
    assert MODULE.swap_bytes(snapshot) == int(1.5 * 1024**3)
    assert MODULE.thermal_healthy(snapshot)
    MODULE.require_system_guard(snapshot, args())
