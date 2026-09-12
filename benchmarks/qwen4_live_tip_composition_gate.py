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
PROFILES = (
    "physical_live_tip_fanout",
    "segmented",
    "segmented_then_physical",
    "private_then_physical",
    "exact_then_physical",
    "segmented_async_qsa_physical",
    "segmented_prequeued_async_qsa_physical",
    "private_async_qsa_physical",
    "exact_async_qsa_physical",
    "segmented_race_physical",
    "private_delta",
    "shared_suffix",
    "exact_set",
)


def token_traces_prefix_compatible(
    left: list[list[int]], right: list[list[int]]
) -> bool:
    """Compare cycle-limited traces without requiring equal acceptance counts."""
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        common = min(len(left_row), len(right_row))
        if left_row[:common] != right_row[:common]:
            return False
    return True


def profile_settings(profile: str) -> dict[str, Any]:
    if profile == "physical":
        return {
            "branch_mode": "physical",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
        }
    if profile == "physical_live_tip_fanout":
        return {
            "branch_mode": "physical_fanout",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
        }
    if profile == "segmented":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
        }
    if profile == "segmented_then_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": True,
            "async_promote_after_first": False,
        }
    if profile == "segmented_race_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": True,
            "async_qsa_promote_after_first": False,
        }
    if profile == "segmented_async_qsa_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
            "async_qsa_promote_after_first": True,
            "async_qsa_prequeue": False,
        }
    if profile == "segmented_prequeued_async_qsa_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "off",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
            "async_qsa_promote_after_first": True,
            "async_qsa_prequeue": True,
        }
    if profile == "private_async_qsa_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
            "async_qsa_promote_after_first": True,
        }
    if profile == "exact_async_qsa_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "on",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
            "async_qsa_promote_after_first": True,
        }
    if profile == "private_then_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": True,
            "async_promote_after_first": False,
        }
    if profile == "exact_then_physical":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "on",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": True,
            "async_promote_after_first": False,
        }
    if profile == "private_delta":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
        }
    if profile == "shared_suffix":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "off",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
            "shared_qsa_suffix": True,
        }
    if profile == "exact_set":
        return {
            "branch_mode": "segmented",
            "qsa_private_delta": "on",
            "qsa_exact_set_fold": "on",
            "qsa_private_delta_min_context": 0,
            "promote_after_first": False,
            "async_promote_after_first": False,
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
    if getattr(args, "first_repetition", 1) < 1:
        raise ValueError("first-repetition must be positive")
    if args.cooldown_seconds < 0:
        raise ValueError("cooldown cannot be negative")
    if not 0 <= args.max_closing_drift <= 1:
        raise ValueError("max-closing-drift must be between zero and one")
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
        "max_closing_drift": args.max_closing_drift,
        "prime_candidates": bool(args.prime_candidates),
        "prime_contract": (
            "physical, every candidate, then physical stabilization; two "
            "cycles so first-cycle promotion also primes the physical "
            "follow-up shape"
        ),
        "candidates": list(args.candidates),
        "apc_modes": list(getattr(args, "apc_modes", ("none",))),
        "orders": {
            profile: [
                block_order(profile, rep)
                for rep in range(
                    getattr(args, "first_repetition", 1),
                    getattr(args, "first_repetition", 1) + args.reps,
                )
            ]
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
        "asynchronous_scope": (
            "segmented_race_physical races an independently owned complete "
            "physical first cycle; segmented_async_qsa_physical instead "
            "queues only immutable QSA-base formation on a second Metal "
            "stream and patches the accepted suffix after the visible commit"
        ),
    }


def _configured_args(
    args: argparse.Namespace, profile: str, apc_mode: str = "none"
) -> argparse.Namespace:
    configured = copy.copy(args)
    configured.shared_qsa_suffix = False
    for key, value in profile_settings(profile).items():
        setattr(configured, key, value)
    configured.idle_seconds = 0.0
    configured.apc_mode = apc_mode
    return configured


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def validate_mechanism_receipt(row: dict[str, Any], profile: str) -> None:
    """Refuse a candidate arm whose requested cache consumer did not run."""

    if profile == "physical":
        return
    if profile == "physical_live_tip_fanout":
        receipt = row.get("fanout_delta") or {}
        expected = {
            "hybrid_tip_fanout_batches": 1,
            "hybrid_tip_fanout_rows": 2,
            "declined_disabled": 0,
            "declined_unsupported_cache": 0,
        }
        mismatches = {
            key: {"expected": value, "actual": int(receipt.get(key, 0))}
            for key, value in expected.items()
            if int(receipt.get(key, 0)) != value
        }
        row["mechanism_receipt"] = {
            "validated": not mismatches,
            "expected_cycles": int(row["measured_cycles"]),
            "mismatches": mismatches,
        }
        if mismatches:
            raise RuntimeError(
                f"{profile} mechanism receipt mismatch: {mismatches}"
            )
        return
    receipt = row["segmented_delta"]
    first_only = profile in {
        "segmented_then_physical",
        "private_then_physical",
        "exact_then_physical",
        "segmented_async_qsa_physical",
        "segmented_prequeued_async_qsa_physical",
        "private_async_qsa_physical",
        "exact_async_qsa_physical",
        "segmented_race_physical",
    }
    cycles = 1 if first_only else int(row["measured_cycles"])
    expected = {
        "engaged": 1,
        "true_batched_engaged": cycles,
        "batched_target_forwards": cycles,
        "batched_draft_forwards": 2 * cycles,
        "b1_target_forwards": 0,
        "committed_cycles": cycles,
        "transaction_branches": 2 * cycles,
        "transaction_promotions": 2 * cycles,
        "transaction_canonicalizations": (
            0 if profile in {
                "segmented_async_qsa_physical",
                "segmented_prequeued_async_qsa_physical",
                "private_async_qsa_physical",
                "exact_async_qsa_physical",
            } else 2
        ),
        "segmented_attention_calls": 14 * cycles,
        "full_prefix_materialized_bytes": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": int(receipt.get(key, 0))}
        for key, value in expected.items()
        if int(receipt.get(key, 0)) != value
    }
    private = profile in {
        "private_delta",
        "shared_suffix",
        "exact_set",
        "private_then_physical",
        "exact_then_physical",
        "private_async_qsa_physical",
        "exact_async_qsa_physical",
    }
    if private:
        private_expected = {
            "private_delta_attention_calls": 14 * cycles,
            "private_delta_rows": 28 * cycles,
            "private_delta_declines": 0,
        }
        mismatches.update(
            {
                key: {"expected": value, "actual": int(receipt.get(key, 0))}
                for key, value in private_expected.items()
                if int(receipt.get(key, 0)) != value
            }
        )
    if profile == "shared_suffix":
        shared_expected = {
            # Flash-Next has one QSA layer every four decoder layers: 12
            # target layers, each split/materialized for two sibling rows.
            "shared_qsa_rows": 24,
            "shared_qsa_materializations": 24,
            "shared_qsa_batched_selections": 12 * cycles,
        }
        mismatches.update(
            {
                key: {"expected": value, "actual": int(receipt.get(key, 0))}
                for key, value in shared_expected.items()
                if int(receipt.get(key, 0)) != value
            }
        )
        for key in (
            "shared_qsa_base_bytes",
            "shared_qsa_materialized_bytes",
        ):
            if int(receipt.get(key, 0)) <= 0:
                mismatches[key] = {
                    "expected": ">0",
                    "actual": int(receipt.get(key, 0)),
                }
    exact = profile in {
        "exact_set",
        "exact_then_physical",
        "exact_async_qsa_physical",
    }
    if exact:
        exact_expected = {
            "exact_set_fold_attention_calls": 14 * cycles,
            "exact_set_fold_rows": 28 * cycles,
            "exact_set_fold_device_proofs": 14 * cycles,
            "exact_set_fold_declines": 0,
            "exact_set_fold_private_fallbacks": 0,
        }
        mismatches.update(
            {
                key: {"expected": value, "actual": int(receipt.get(key, 0))}
                for key, value in exact_expected.items()
                if int(receipt.get(key, 0)) != value
            }
        )
    row["mechanism_receipt"] = {
        "validated": not mismatches,
        "expected_cycles": cycles,
        "mismatches": mismatches,
    }
    if mismatches:
        raise RuntimeError(f"{profile} mechanism receipt mismatch: {mismatches}")


def summarize(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, Any] = {}
    identities = sorted({(block["apc_mode"], block["candidate"]) for block in blocks})
    for apc_mode, candidate in identities:
        identity = f"{apc_mode}:{candidate}"
        all_blocks = [
            block for block in blocks
            if block["candidate"] == candidate and block["apc_mode"] == apc_mode
        ]
        chosen = [block for block in all_blocks if block["accepted"]]
        if not chosen:
            by_candidate[identity] = {
                "accepted_blocks": 0,
                "discarded_blocks": len(all_blocks),
                "discard_reasons": sorted(
                    {reason for block in all_blocks for reason in block["discard_reasons"]}
                ),
            }
            continue
        controls = [row for block in chosen for row in block["rows"] if row["profile"] == "physical"]
        trials = [row for block in chosen for row in block["rows"] if row["profile"] == candidate]
        control_first = _median(controls, "branch_to_first_commit_ms")
        trial_first = _median(trials, "branch_to_first_commit_ms")
        control_tps = _median(controls, "aggregate_branch_decode_tps")
        trial_tps = _median(trials, "aggregate_branch_decode_tps")
        by_candidate[identity] = {
            "accepted_blocks": len(chosen),
            "discarded_blocks": len(all_blocks) - len(chosen),
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
        if all(
            row.get("branch_to_first_commit_if_prequeued_ms") is not None
            for row in trials
        ):
            prequeued = _median(
                trials, "branch_to_first_commit_if_prequeued_ms"
            )
            by_candidate[identity].update(
                candidate_prequeued_first_commit_ms=prequeued,
                prequeued_first_commit_speedup=control_first / prequeued,
                async_qsa_queue_ms=_median(trials, "async_qsa_queue_ms"),
                async_wait_after_first_ms=_median(
                    trials, "async_wait_after_first_ms"
                ),
            )
    return by_candidate


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.utils import load
    from qwen4_mtp_dynamic_join_gate import exact_prompt

    model, tokenizer = load(args.model)
    model.eval()
    prompt = mx.array(exact_prompt(tokenizer, args.context, "live-tip-compose"), mx.uint32)
    mx.eval(prompt)

    prime_rows = []
    if args.prime_candidates:
        # Prime both the B2 control and every candidate. Two measured cycles
        # are required: first-cycle promotion profiles switch representation
        # after cycle one, so cycle two is the first physical follow-up shape.
        prime_profiles = (
            "physical",
            *tuple(dict.fromkeys(args.candidates)),
            "physical",
        )
        for profile in prime_profiles:
            configured = _configured_args(args, profile)
            configured.measured_cycles = 2
            apply_composition_environment(configured)
            row = _run_arm(model, prompt, configured, "warm_live_tip")
            row.update(profile=profile, phase="profile_prime")
            validate_mechanism_receipt(row, profile)
            prime_rows.append(row)
            if args.out:
                atomic_write(
                    args.out,
                    {
                        "metadata": plan,
                        "status": "priming",
                        "prime_rows": prime_rows,
                        "blocks": [],
                    },
                )

    blocks = []
    for apc_mode in getattr(args, "apc_modes", ("none",)):
        for candidate in args.candidates:
            first_repetition = getattr(args, "first_repetition", 1)
            for repetition in range(first_repetition, first_repetition + args.reps):
                rows = []
                for slot, profile in enumerate(block_order(candidate, repetition)):
                    if args.cooldown_seconds:
                        time.sleep(args.cooldown_seconds)
                    configured = _configured_args(args, profile, apc_mode)
                    apply_composition_environment(configured)
                    row = _run_arm(model, prompt, configured, "warm_live_tip")
                    row.update(
                        profile=profile,
                        candidate=candidate,
                        apc_mode=apc_mode,
                        repetition=repetition,
                        slot=slot,
                    )
                    if apc_mode != "none" and not (row.get("apc") or {}).get(
                        "hit"
                    ):
                        raise RuntimeError(
                            "factorial arm did not exercise a real APC hit"
                        )
                    validate_mechanism_receipt(row, profile)
                    rows.append(row)
                    if args.out:
                        atomic_write(
                            args.out,
                            {
                                "metadata": plan,
                                "status": "running",
                                "prime_rows": prime_rows,
                                "blocks": [
                                    *blocks,
                                    {
                                        "candidate": candidate,
                                        "apc_mode": apc_mode,
                                        "repetition": repetition,
                                        "rows": rows,
                                    },
                                ],
                            },
                        )
                control_tokens = [
                    row["branch_tokens"]
                    for row in rows
                    if row["profile"] == "physical"
                ]
                candidate_tokens = [
                    row["branch_tokens"]
                    for row in rows
                    if row["profile"] == candidate
                ]
                if any(
                    not token_traces_prefix_compatible(
                        tokens, control_tokens[0]
                    )
                    for tokens in [*control_tokens[1:], *candidate_tokens]
                ):
                    raise AssertionError(
                        f"{candidate} token trace differs from physical control"
                    )
                controls = [row for row in rows if row["profile"] == "physical"]
                trials = [row for row in rows if row["profile"] == candidate]
                closing_drift = abs(
                    controls[-1]["aggregate_branch_decode_tps"]
                    - controls[0]["aggregate_branch_decode_tps"]
                ) / controls[0]["aggregate_branch_decode_tps"]
                candidate_drift = abs(
                    trials[-1]["aggregate_branch_decode_tps"]
                    - trials[0]["aggregate_branch_decode_tps"]
                ) / trials[0]["aggregate_branch_decode_tps"]
                discard_reasons = []
                if closing_drift > args.max_closing_drift:
                    discard_reasons.append("closing_control_drift")
                if candidate_drift > args.max_closing_drift:
                    discard_reasons.append("candidate_drift")
                block = {
                    "candidate": candidate,
                    "apc_mode": apc_mode,
                    "repetition": repetition,
                    "accepted": not discard_reasons,
                    "closing_control_drift_fraction": closing_drift,
                    "candidate_drift_fraction": candidate_drift,
                    "discard_reasons": discard_reasons,
                    "rows": rows,
                }
                blocks.append(block)

    return {
        "metadata": plan,
        "status": "complete",
        "prime_rows": prime_rows,
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
    parser.add_argument(
        "--first-repetition",
        type=int,
        default=1,
        help="first counterbalance index (use 2 for candidate/control/control/candidate)",
    )
    parser.add_argument("--candidates", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument(
        "--apc-modes",
        nargs="+",
        choices=("none", "legacy", "apcv2"),
        default=["none"],
    )
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-closing-drift", type=float, default=0.05)
    parser.add_argument(
        "--prime-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="execute untimed two-cycle physical and candidate arms before brackets",
    )
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
