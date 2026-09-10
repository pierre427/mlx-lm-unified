import argparse
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "benchmarks" / "qwen4_mtp_policy_gate.py"
SPEC = importlib.util.spec_from_file_location("qwen4_mtp_policy_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = dict(
        execute=False,
        model="/definitely/missing",
        contexts="16K,64K",
        reps=3,
        max_tokens=256,
        cooldown_seconds=0.0,
        prefill_step_size=512,
        share_qsa_indices=True,
        out="unused.json",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plan_is_three_arm_rotated_and_model_free():
    plan = MODULE.build_plan(args())
    assert plan["contexts"] == [16384, 65536]
    assert len(plan["cells"]) == 6
    assert plan["cells"][0]["order"] == list(MODULE.ARMS)
    assert plan["cells"][1]["order"] == [
        "fixed_k2",
        "adaptive_1_3",
        "target_only",
    ]
    assert not plan["execution_authorized"]


@pytest.mark.parametrize("value", ["", "0", "16K,16K", "broken"])
def test_context_parser_refuses_bad_input(value):
    with pytest.raises((ValueError, TypeError)):
        MODULE.parse_contexts(value)


def test_summary_requires_exact_and_adaptive_engagement_receipts():
    rows = []
    for arm, tps, wall in (
        ("target_only", 50.0, 5.0),
        ("fixed_k2", 75.0, 4.0),
        ("adaptive_1_3", 80.0, 3.8),
    ):
        row = {
            "context": 16384,
            "arm": arm,
            "decode_tps_after_first": tps,
            "wall_s": wall,
        }
        if arm != "target_only":
            row["exact_target_match"] = True
        if arm == "adaptive_1_3":
            row["adaptive"] = {"expansions": 2, "observed_max_depth": 3}
        rows.append(row)
    summary = MODULE.summarize(rows, [16384])["16384"]
    assert summary["exact_matches"] == 2
    assert summary["adaptive_engaged_trials"] == 1
