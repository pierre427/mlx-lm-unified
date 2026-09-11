#!/usr/bin/env python3
"""Counterbalanced Qwen4 live-tip composition gate.

Every arm creates and warms a real one-row self-MTP decode lane, then branches
that current tip immediately.  Cooldown is outside the live-tip boundary.
The physical B2 control brackets each candidate B/C/C/B so a cold APC restore
cannot decide whether an in-flight cache consumer composes.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from qwen4_live_tip_branch_gate import (
        DEFAULT_MODEL,
        _run_arm,
        apply_composition_environment,
        atomic_write,
        system_snapshot,
        utc_now,
    )
except ModuleNotFoundError:  # Imported as benchmarks.* by the test suite.
    from benchmarks.qwen4_live_tip_branch_gate import (
        DEFAULT_MODEL,
        _run_arm,
        apply_composition_environment,
        atomic_write,
        system_snapshot,
        utc_now,
    )


SCHEMA = "mlx-uag.qwen4-live-tip-composition-gate.v1"
PROFILES = ("segmented", "private_delta", "exact_set")


def profile_settings(profile: str) -> dict[str, Any]:
    if profile == "physical":
        return {
            "branch_mode": "physical",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
        }
    if profile == "segmented":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
        }
    if profile == "private_delta":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
        }
    if profile == "exact_set":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "on",
            "qsa_private_delta_min_context": 0,
        }
    raise ValueError(f"unknown profile {profile!r}")


def block_order(profile: str, repetition: int) -> list[str]:
    order = ["physical", profile, profile, "physical"]
    return order if repetition % 2 else [profile, "physical", "physical", profile]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.candidates:
        raise ValueError("at least one candidate is required")
    if args.reps < 1:
        raise ValueError("reps must be positive")
    if args.cooldown_seconds < 0:
        raise ValueError("cooldown cannot be negative")
    unknown = sorted(set(args.candidates).difference(PROFILES))
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    return {
        "schema": f"{SCHEMA}.plan",
        "created_at_utc": utc_now(),
        "execution_authorized": bool(args.execute),
        "model": args.model,
        "context": args.context,
        "num_draft": args.num_draft,
        "warmup_cycles": args.warmup_cycles,
        "measured_cycles": args.measured_cycles,
        "repetitions": args.reps,
        "candidates": list(args.candidates),
        "orders": {
            profile: [block_order(profile, rep) for rep in range(1, args.reps + 1)]
            for profile in args.candidates
        },
        "timing_boundary": (
            "warm B1 target/MTP cycles -> detach current tip -> immediately "
            "attach profile B2 -> first proposal/verify -> commit"
        ),
        "cooldown_location": "before each complete warm-lane arm only",
        "profiles": {
            name: profile_settings(name) for name in ("physical", *args.candidates)
        },
    }


def _configured_args(args: argparse.Namespace, profile: str) -> argparse.Namespace:
    configured = copy.copy(args)
    for key, value in profile_settings(profile).items():
        setattr(configured, key, value)
    configured.idle_seconds = 0.0
    return configured


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def summarize(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, Any] = {}
    for candidate in sorted({block["candidate"] for block in blocks}):
        chosen = [block for block in blocks if block["candidate"] == candidate]
        controls = [row for block in chosen for row in block["rows"] if row["profile"] == "physical"]
        trials = [row for block in chosen for row in block["rows"] if row["profile"] == candidate]
        control_first = _median(controls, "branch_to_first_commit_ms")
        trial_first = _median(trials, "branch_to_first_commit_ms")
        control_tps = _median(controls, "aggregate_branch_decode_tps")
        trial_tps = _median(trials, "aggregate_branch_decode_tps")
        by_candidate[candidate] = {
            "samples_per_arm": len(trials),
            "physical_branch_to_first_commit_ms": control_first,
            "candidate_branch_to_first_commit_ms": trial_first,
            "first_commit_speedup": control_first / trial_first,
            "physical_decode_tps": control_tps,
            "candidate_decode_tps": trial_tps,
            "decode_tps_ratio": trial_tps / control_tps,
            "all_sibling_tokens_exact": all(row["branch_tokens"][0] == row["branch_tokens"][1] for row in trials),
            "all_sibling_states_exact": all(row["sibling_state"]["equal"] for row in trials),
            "swap_growth_bytes": sum(max(0, int(row["swap_growth_bytes"])) for row in trials),
        }
    return by_candidate


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.utils import load
    from qwen4_mtp_dynamic_join_gate import exact_prompt

    model, tokenizer = load(args.model)
    model.eval()
    prompt = mx.array(exact_prompt(tokenizer, args.context, "live-tip-compose"), mx.uint32)
    mx.eval(prompt)

    blocks = []
    for candidate in args.candidates:
        for repetition in range(1, args.reps + 1):
            rows = []
            for slot, profile in enumerate(block_order(candidate, repetition)):
                if args.cooldown_seconds:
                    time.sleep(args.cooldown_seconds)
                configured = _configured_args(args, profile)
                apply_composition_environment(configured)
                row = _run_arm(model, prompt, configured, "warm_live_tip")
                row.update(profile=profile, candidate=candidate, repetition=repetition, slot=slot)
                rows.append(row)
                if args.out:
                    atomic_write(args.out, {"metadata": plan, "status": "running", "blocks": [*blocks, {"candidate": candidate, "repetition": repetition, "rows": rows}]})
            control_tokens = [row["branch_tokens"] for row in rows if row["profile"] == "physical"]
            candidate_tokens = [row["branch_tokens"] for row in rows if row["profile"] == candidate]
            if any(tokens != control_tokens[0] for tokens in [*control_tokens[1:], *candidate_tokens]):
                raise AssertionError(f"{candidate} token trace differs from physical control")
            blocks.append({"candidate": candidate, "repetition": repetition, "rows": rows})

    return {
        "metadata": plan,
        "status": "complete",
        "summary": summarize(blocks),
        "blocks": blocks,
        "system_after": system_snapshot(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--warmup-cycles", type=int, default=8)
    parser.add_argument("--measured-cycles", type=int, default=32)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--candidates", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--minimum-system-free-percent", type=int, default=25)
    parser.add_argument("--maximum-swap-growth-mb", type=int, default=16)
    parser.add_argument("--share-qsa-indices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    result = execute(args, plan)
    if args.out:
        atomic_write(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
