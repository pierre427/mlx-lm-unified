from argparse import Namespace

import pytest

from benchmarks.qwen4_live_tip_branch_gate import ARMS, arm_order, build_plan


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
    }
    values.update(updates)
    return Namespace(**values)


def test_plan_keeps_the_live_boundary_warm():
    plan = build_plan(_args())
    assert "run warmup target/MTP cycles" in plan["timing_boundary"]
    assert "no sleep, mx.clear_cache" in plan["warm_invariant"]
    assert "detach canonicalization is measured" in plan["warm_invariant"]
    assert plan["orders"] == [list(ARMS), list(reversed(ARMS))]


def test_arm_order_counterbalances():
    assert arm_order(1) == ["warm_live_tip", "idle_live_tip"]
    assert arm_order(2) == ["idle_live_tip", "warm_live_tip"]


@pytest.mark.parametrize(
    "updates",
    [
        {"context": 31},
        {"num_draft": 0},
        {"warmup_cycles": 0},
        {"measured_cycles": 0},
        {"branches": 3},
        {"idle_seconds": -1},
    ],
)
def test_invalid_plans_fail(updates):
    with pytest.raises(ValueError):
        build_plan(_args(**updates))
