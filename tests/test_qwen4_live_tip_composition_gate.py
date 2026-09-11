from argparse import Namespace

import pytest

from benchmarks.qwen4_live_tip_composition_gate import (
    PROFILES,
    block_order,
    build_plan,
    profile_settings,
)


def _args(**updates):
    values = {
        "execute": False,
        "model": "/tmp/model",
        "context": 16384,
        "num_draft": 2,
        "warmup_cycles": 8,
        "measured_cycles": 32,
        "reps": 1,
        "candidates": list(PROFILES),
        "cooldown_seconds": 60.0,
        "max_closing_drift": 0.05,
    }
    values.update(updates)
    return Namespace(**values)


def test_plan_places_cooldown_outside_live_boundary():
    plan = build_plan(_args())
    assert plan["cooldown_location"] == "before each complete warm-lane arm only"
    assert "detach current tip -> immediately" in plan["timing_boundary"]


@pytest.mark.parametrize("profile", PROFILES)
def test_counterbalanced_order(profile):
    assert block_order(profile, 1) == ["physical", profile, profile, "physical"]
    assert block_order(profile, 2) == [profile, "physical", "physical", profile]


def test_profiles_are_mechanically_distinct():
    assert profile_settings("segmented")["qsa_private_delta"] == "off"
    assert profile_settings("private_delta")["qsa_exact_set_fold"] == "off"
    assert profile_settings("exact_set")["qsa_exact_set_fold"] == "on"
    assert profile_settings("physical")["branch_mode"] == "physical"


@pytest.mark.parametrize(
    "updates",
    [
        {"candidates": []},
        {"candidates": ["bad"]},
        {"reps": 0},
        {"cooldown_seconds": -1},
        {"max_closing_drift": 1.1},
    ],
)
def test_bad_plans_fail(updates):
    with pytest.raises(ValueError):
        build_plan(_args(**updates))
