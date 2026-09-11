from argparse import Namespace

import pytest

from benchmarks.qwen4_live_tip_branch_gate import (
    ARMS,
    BRANCH_MODES,
    arm_order,
    build_plan,
    _token_digest,
)


def _args(**updates):
    values = {
        "execute": False,
        "model": "/tmp/model",
        "context": 1024,
        "num_draft": 2,
        "warmup_cycles": 8,
        "measured_cycles": 8,
        "branches": 2,
        "reps": 2,
        "idle_seconds": 60.0,
        "cooldown_seconds": 30.0,
        "branch_mode": "physical",
        "qsa_private_delta": "default",
        "qsa_exact_set_fold": "default",
        "qsa_private_delta_min_context": None,
    }
    values.update(updates)
    return Namespace(**values)


def test_plan_keeps_the_live_boundary_warm():
    plan = build_plan(_args())
    assert "run warmup target/MTP cycles" in plan["timing_boundary"]
    assert "no sleep, mx.clear_cache" in plan["warm_invariant"]
    assert "detach canonicalization is measured" in plan["warm_invariant"]
    assert plan["orders"] == [list(ARMS), list(reversed(ARMS))]
    assert plan["branch_mode"] == "physical"


@pytest.mark.parametrize("mode", BRANCH_MODES)
def test_plan_records_composition_mode(mode):
    plan = build_plan(
        _args(
            branch_mode=mode,
            qsa_private_delta="on",
            qsa_exact_set_fold="off",
            qsa_private_delta_min_context=0,
        )
    )
    assert plan["branch_mode"] == mode
    expected = {
        "qsa_private_delta": "on",
        "qsa_exact_set_fold": "off",
        "qsa_private_delta_min_context": 0,
    }
    assert {key: plan["composition"][key] for key in expected} == expected


def test_arm_order_counterbalances():
    assert arm_order(1) == ["warm_live_tip", "idle_live_tip"]
    assert arm_order(2) == ["idle_live_tip", "warm_live_tip"]


def test_prefix_attestation_digest_is_order_sensitive():
    assert _token_digest([[1, 2], [3]]) != _token_digest([[2, 1], [3]])


@pytest.mark.parametrize(
    "updates",
    [
        {"context": 31},
        {"num_draft": 0},
        {"warmup_cycles": 0},
        {"measured_cycles": 0},
        {"branches": 3},
        {"idle_seconds": -1},
        {"branch_mode": "unknown"},
    ],
)
def test_invalid_plans_fail(updates):
    with pytest.raises(ValueError):
        build_plan(_args(**updates))
