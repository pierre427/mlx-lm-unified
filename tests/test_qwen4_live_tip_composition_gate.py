from argparse import Namespace

import pytest

from benchmarks.qwen4_live_tip_composition_gate import (
    PROFILES,
    block_order,
    build_plan,
    profile_settings,
    validate_mechanism_receipt,
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
        "prime_candidates": True,
    }
    values.update(updates)
    return Namespace(**values)


def test_plan_places_cooldown_outside_live_boundary():
    plan = build_plan(_args())
    assert plan["cooldown_location"] == "before each complete warm-lane arm only"
    assert "detach current tip -> immediately" in plan["timing_boundary"]
    assert plan["prime_candidates"] is True
    assert "then physical stabilization" in plan["prime_contract"]


@pytest.mark.parametrize("profile", PROFILES)
def test_counterbalanced_order(profile):
    assert block_order(profile, 1) == ["physical", profile, profile, "physical"]
    assert block_order(profile, 2) == [profile, "physical", "physical", profile]


def test_profiles_are_mechanically_distinct():
    assert profile_settings("segmented")["qsa_private_delta"] == "off"
    assert profile_settings("private_delta")["qsa_exact_set_fold"] == "off"
    assert profile_settings("exact_set")["qsa_exact_set_fold"] == "on"
    assert profile_settings("physical")["branch_mode"] == "physical"
    assert profile_settings("physical_live_tip_fanout")["branch_mode"] == "physical_fanout"
    assert profile_settings("segmented_then_physical")["promote_after_first"]
    assert profile_settings("segmented_race_physical")[
        "async_promote_after_first"
    ]
    assert profile_settings("segmented_async_qsa_physical")[
        "async_qsa_promote_after_first"
    ]
    assert profile_settings("private_async_qsa_physical")[
        "async_qsa_promote_after_first"
    ]
    assert profile_settings("private_async_qsa_physical")["qsa_private_delta"] == "on"
    assert profile_settings("exact_async_qsa_physical")["qsa_exact_set_fold"] == "on"
    assert profile_settings("private_then_physical")["promote_after_first"]
    assert profile_settings("private_then_physical")["qsa_private_delta"] == "on"
    assert profile_settings("exact_then_physical")["promote_after_first"]
    assert profile_settings("exact_then_physical")["qsa_exact_set_fold"] == "on"


def _receipt_row(cycles=1):
    return {
        "measured_cycles": cycles,
        "segmented_delta": {
            "engaged": 1,
            "true_batched_engaged": cycles,
            "batched_target_forwards": cycles,
            "batched_draft_forwards": 2 * cycles,
            "b1_target_forwards": 0,
            "committed_cycles": cycles,
            "transaction_branches": 2 * cycles,
            "transaction_promotions": 2 * cycles,
            "transaction_canonicalizations": 2,
            "segmented_attention_calls": 14 * cycles,
            "full_prefix_materialized_bytes": 0,
            "private_delta_attention_calls": 14 * cycles,
            "private_delta_rows": 28 * cycles,
            "private_delta_declines": 0,
            "exact_set_fold_attention_calls": 14 * cycles,
            "exact_set_fold_rows": 28 * cycles,
            "exact_set_fold_device_proofs": 14 * cycles,
            "exact_set_fold_declines": 0,
            "exact_set_fold_private_fallbacks": 0,
        },
    }


@pytest.mark.parametrize(
    "profile", ["segmented_then_physical", "private_then_physical", "exact_then_physical"]
)
def test_first_cycle_promotion_receipt_is_strict(profile):
    row = _receipt_row()
    validate_mechanism_receipt(row, profile)
    assert row["mechanism_receipt"]["validated"]


def test_mechanism_receipt_rejects_a_silent_fallback():
    row = _receipt_row()
    row["segmented_delta"]["true_batched_engaged"] = 0
    with pytest.raises(RuntimeError, match="mechanism receipt mismatch"):
        validate_mechanism_receipt(row, "segmented_then_physical")


@pytest.mark.parametrize(
    "profile",
    [
        "segmented_async_qsa_physical",
        "private_async_qsa_physical",
        "exact_async_qsa_physical",
    ],
)
def test_async_qsa_promotion_receipt_allows_direct_transaction_retirement(profile):
    row = _receipt_row()
    row["segmented_delta"]["transaction_canonicalizations"] = 0
    validate_mechanism_receipt(row, profile)
    assert row["mechanism_receipt"]["validated"]


def test_live_tip_fanout_receipt_is_strict():
    row = _receipt_row(cycles=32)
    row["fanout_delta"] = {
        "hybrid_tip_fanout_batches": 1,
        "hybrid_tip_fanout_rows": 2,
        "declined_disabled": 0,
        "declined_unsupported_cache": 0,
    }
    validate_mechanism_receipt(row, "physical_live_tip_fanout")
    assert row["mechanism_receipt"]["validated"]

    row["fanout_delta"]["hybrid_tip_fanout_batches"] = 0
    with pytest.raises(RuntimeError, match="mechanism receipt mismatch"):
        validate_mechanism_receipt(row, "physical_live_tip_fanout")


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
