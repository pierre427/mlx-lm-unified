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


def test_first_divergence_handles_content_and_length():
    assert MODULE.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert MODULE.first_divergence([1, 2], [1, 2, 3]) == 2
    assert MODULE.first_divergence([1, 2], [1, 2]) is None


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
