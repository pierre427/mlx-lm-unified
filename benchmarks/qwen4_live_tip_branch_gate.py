#!/usr/bin/env python3
"""Measure Qwen4 self-MTP fan-out from a genuinely live decode tip.

The timed warm arm runs target/MTP cycles immediately before it branches. It
does not clear the MLX cache, sleep, rebuild the prompt, or restore APC state
between the final warm cycle and the branch. An idle arm uses the same sequence
but waits after detaching the live tip, which isolates execution-working-set
decay from branch mechanics.

The default invocation is plan-only and imports no MLX modules. ``--execute``
loads the model and runs an A/B/B/A bracket.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "mlx-uag.qwen4-live-tip-branch-gate.v1"
DEFAULT_MODEL = (
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
ARMS = ("warm_live_tip", "idle_live_tip")
BRANCH_MODES = ("physical", "segmented")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def arm_order(repetition: int) -> list[str]:
    return list(ARMS if repetition % 2 else reversed(ARMS))


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.context < 32:
        raise ValueError("context must be at least 32")
    if args.num_draft < 1:
        raise ValueError("num-draft must be positive")
    if args.warmup_cycles < 1:
        raise ValueError("warmup-cycles must be positive")
    if args.measured_cycles < 1:
        raise ValueError("measured-cycles must be positive")
    if args.branches != 2:
        raise ValueError("this gate is qualified only for two branches")
    if args.reps < 1 or args.idle_seconds < 0 or args.cooldown_seconds < 0:
        raise ValueError("reps must be positive and waits cannot be negative")
    branch_mode = getattr(args, "branch_mode", "physical")
    if branch_mode not in BRANCH_MODES:
        raise ValueError(f"branch-mode must be one of {BRANCH_MODES}")
    return {
        "schema": f"{SCHEMA}.plan",
        "created_at_utc": utc_now(),
        "execution_authorized": bool(args.execute),
        "model": args.model,
        "context": args.context,
        "num_draft": args.num_draft,
        "warmup_cycles": args.warmup_cycles,
        "measured_cycles": args.measured_cycles,
        "branches": args.branches,
        "branch_mode": branch_mode,
        "repetitions": args.reps,
        "idle_seconds": args.idle_seconds,
        "cooldown_before_arm_seconds": args.cooldown_seconds,
        "orders": [arm_order(rep) for rep in range(1, args.reps + 1)],
        "timing_boundary": (
            "prepare B1 -> run warmup target/MTP cycles -> detach live tip -> "
            "optional idle -> start timer -> clone/attach B2 -> first proposal "
            "and target verification -> commit"
        ),
        "warm_invariant": (
            "no sleep, mx.clear_cache, prompt restore, or unrelated warmup "
            "occurs between the last warm cycle and live-tip branch; required "
            "detach canonicalization is measured"
        ),
        "correctness_gate": (
            "greedy sibling token traces match within every arm and complete "
            "warm/idle traces match by bracket slot"
        ),
        "composition": {
            "qsa_private_delta": getattr(args, "qsa_private_delta", "default"),
            "qsa_exact_set_fold": getattr(args, "qsa_exact_set_fold", "default"),
            "qsa_private_delta_min_context": getattr(
                args, "qsa_private_delta_min_context", None
            ),
            "promote_after_first": getattr(args, "promote_after_first", False),
        },
    }


def _run(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def system_snapshot() -> dict[str, Any]:
    return {
        "pmset_therm": _run(["pmset", "-g", "therm"]),
        "memory_pressure": _run(["memory_pressure", "-Q"]),
        "swapusage": _run(["sysctl", "-n", "vm.swapusage"]),
    }


def _free_percent(snapshot: dict[str, Any]) -> int | None:
    match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%",
        snapshot["memory_pressure"]["stdout"],
    )
    return int(match.group(1)) if match else None


def _swap_bytes(snapshot: dict[str, Any]) -> int | None:
    match = re.search(
        r"used\s*=\s*([0-9.]+)([KMG])",
        snapshot["swapusage"]["stdout"],
    )
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _thermal_healthy(snapshot: dict[str, Any]) -> bool:
    therm = snapshot["pmset_therm"]
    if therm["returncode"] != 0:
        return False
    text = therm["stdout"].lower()
    return "warning level has been recorded" not in text or all(
        line.startswith("note: no ")
        for line in text.splitlines()
        if "warning level has been recorded" in line
    )


def _mlx_memory(mx: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _token_digest(rows: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _state_values(pair: Any) -> list[Any]:
    values = [cache.state for cache in pair.target]
    values.extend(cache.state for cache in pair.draft)
    return values


def _batch_state_values(batch: Any) -> list[Any]:
    if hasattr(batch, "caches"):
        return _state_values(batch.caches)
    values = []
    for pair in batch.row_caches:
        values.extend(_state_values(pair))
    return values


def _array_leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _array_leaves(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _array_leaves(value[key])
    elif value is not None:
        yield value


def _same_detached_state(mx: Any, left: Any, right: Any) -> dict[str, Any]:
    left_values = list(_array_leaves(_state_values(left.caches)))
    right_values = list(_array_leaves(_state_values(right.caches)))
    if len(left_values) != len(right_values):
        return {"equal": False, "reason": "leaf_count", "compared": 0}
    tests = []
    for a, b in zip(left_values, right_values):
        if not hasattr(a, "shape") or not hasattr(b, "shape"):
            tests.append(a == b)
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            tests.append(False)
        else:
            tests.append(mx.array_equal(a, b))
    device_tests = [test for test in tests if hasattr(test, "shape")]
    if device_tests:
        mx.eval(device_tests)
    equal = all(bool(test.item()) if hasattr(test, "item") else bool(test) for test in tests)
    return {"equal": equal, "reason": None if equal else "value", "compared": len(tests)}


def _advance_cycle(mx: Any, model: Any, batch: Any, propose: Any, commit: Any):
    proposal = propose(model, batch)
    mx.eval([output.logprobs for row in proposal.outputs for output in row])
    emitted = [len(row) for row in proposal.outputs]
    commit(batch, proposal, emitted_counts=emitted, terminal=[False] * len(emitted))
    return proposal, emitted


def _run_arm(model: Any, prompt: Any, args: argparse.Namespace, arm: str) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.hybrid_speculative import (
        attach_self_mtp_lanes,
        attach_segmented_self_mtp_lanes,
        commit_batched_self_mtp,
        detach_self_mtp_lanes,
        _self_mtp_group_offset,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )
    from mlx_lm.sample_utils import LaneRNG
    from mlx_lm.segmented_self_mtp import segmented_self_mtp_stats

    before = system_snapshot()
    if not _thermal_healthy(before):
        raise RuntimeError("thermal warning before arm")
    free_before = _free_percent(before)
    swap_before = _swap_bytes(before)
    if free_before is None or free_before < args.minimum_system_free_percent:
        raise MemoryError(f"system free memory is {free_before}% before arm")
    if swap_before is None:
        raise RuntimeError("cannot read swap use before arm")

    mx.reset_peak_memory()
    max_tokens = 8 + (args.warmup_cycles + args.measured_cycles + 2) * (
        args.num_draft + 1
    )
    prepared_started = time.perf_counter_ns()
    detached, first = prepare_self_mtp_lane(
        prompt,
        model,
        uid=0,
        max_tokens=max_tokens,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(args.seed),
        num_draft=args.num_draft,
        sampling_temp=0.0,
        sampling_top_p=1.0,
        sampling_top_k=0,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=args.prefill_step_size,
        share_qsa_indices=args.share_qsa_indices,
    )
    if args.branch_mode == "segmented":
        detached.shared_qsa_prefix_id = hashlib.sha256(
            f"live-tip:{args.context}:{args.seed}".encode()
        ).hexdigest()
    batch = attach_self_mtp_lanes(model, None, [detached])
    mx.synchronize()
    prepared_ns = time.perf_counter_ns()

    warmup_rows = [[int(first.token)]]
    for _ in range(args.warmup_cycles):
        proposal, _ = _advance_cycle(
            mx,
            model,
            batch,
            propose_batched_self_mtp,
            commit_batched_self_mtp,
        )
        warmup_rows[0].extend(int(output.token) for output in proposal.outputs[0])
    live_tip_position = _self_mtp_group_offset(batch.caches.target)
    mx.synchronize()
    last_warm_ns = time.perf_counter_ns()

    batch, rows = detach_self_mtp_lanes(model, batch, [0])
    if batch.lanes or len(rows) != 1:
        raise RuntimeError("failed to detach the single live lane")
    canonical = rows[0]
    mx.synchronize()
    detached_ns = time.perf_counter_ns()
    if args.branch_mode == "segmented":
        # Physical B1 detach deliberately drops any earlier cohort attestation.
        # This gate creates the sibling from this exact canonical object, so it
        # can safely attest the new initial cohort at the branch boundary.
        canonical.shared_qsa_prefix_id = _token_digest(warmup_rows)

    idle_wait_ns = 0
    if arm == "idle_live_tip" and args.idle_seconds:
        idle_started = time.perf_counter_ns()
        time.sleep(args.idle_seconds)
        idle_wait_ns = time.perf_counter_ns() - idle_started

    # This is the branch boundary. The live-tip-to-commit metric also includes
    # the required detach/canonicalization immediately before this point.
    branch_started = time.perf_counter_ns()
    sibling = copy.deepcopy(canonical)
    sibling.lane.uid = 1
    sibling.lane.rng = LaneRNG(args.seed + 1)
    segmented_self_mtp_stats(reset=True)
    segmented_before = segmented_self_mtp_stats(reset=False)
    attach = (
        attach_segmented_self_mtp_lanes
        if args.branch_mode == "segmented"
        else attach_self_mtp_lanes
    )
    branch_batch = attach(model, None, [canonical, sibling])
    mx.eval(_batch_state_values(branch_batch))
    mx.synchronize()
    branch_ready_ns = time.perf_counter_ns()

    proposal = propose_batched_self_mtp(model, branch_batch)
    mx.eval([output.logprobs for row in proposal.outputs for output in row])
    mx.synchronize()
    first_output_ns = time.perf_counter_ns()
    emitted = [len(row) for row in proposal.outputs]
    commit_batched_self_mtp(
        branch_batch,
        proposal,
        emitted_counts=emitted,
        terminal=[False, False],
    )
    mx.eval(_batch_state_values(branch_batch))
    mx.synchronize()
    first_commit_ns = time.perf_counter_ns()

    branch_rows = [
        [int(output.token) for output in outputs] for outputs in proposal.outputs
    ]
    promotion_ms = 0.0
    if getattr(args, "promote_after_first", False):
        if args.branch_mode != "segmented":
            raise ValueError("post-first promotion requires a segmented branch")
        promotion_started = time.perf_counter_ns()
        emptied, promoted_rows = detach_self_mtp_lanes(
            model, branch_batch, [0, 1]
        )
        if emptied.lanes or len(promoted_rows) != 2:
            raise RuntimeError("failed to detach rows for physical promotion")
        branch_batch = attach_self_mtp_lanes(model, None, promoted_rows)
        mx.eval(_batch_state_values(branch_batch))
        mx.synchronize()
        promotion_ms = (time.perf_counter_ns() - promotion_started) / 1e6
    followup_tokens = sum(emitted)
    followup_started = time.perf_counter_ns()
    for _ in range(args.measured_cycles - 1):
        next_proposal, next_emitted = _advance_cycle(
            mx,
            model,
            branch_batch,
            propose_batched_self_mtp,
            commit_batched_self_mtp,
        )
        for row, outputs in zip(branch_rows, next_proposal.outputs):
            row.extend(int(output.token) for output in outputs)
        followup_tokens += sum(next_emitted)
    mx.synchronize()
    followup_finished = time.perf_counter_ns()

    branch_batch, final_rows = detach_self_mtp_lanes(
        model, branch_batch, [0, 1]
    )
    if branch_batch.lanes or len(final_rows) != 2:
        raise RuntimeError("failed to detach both branch rows")
    sibling_state = _same_detached_state(mx, final_rows[0], final_rows[1])
    segmented_after = segmented_self_mtp_stats(reset=False)
    segmented_delta = {
        key: int(value) - int(segmented_before.get(key, 0))
        for key, value in segmented_after.items()
        if isinstance(value, int) and isinstance(segmented_before.get(key, 0), int)
    }

    after = system_snapshot()
    swap_after = _swap_bytes(after)
    if not _thermal_healthy(after):
        raise RuntimeError("thermal warning after arm")
    if swap_after is None:
        raise RuntimeError("cannot read swap use after arm")
    swap_growth = swap_after - swap_before
    if swap_growth > args.maximum_swap_growth_mb * 1024**2:
        raise MemoryError(f"swap grew by {swap_growth} bytes")
    if branch_rows[0] != branch_rows[1]:
        raise AssertionError("greedy sibling branches produced different tokens")

    result = {
        "arm": arm,
        "branch_mode": args.branch_mode,
        "prepare_ms": (prepared_ns - prepared_started) / 1e6,
        "warmup_ms": (last_warm_ns - prepared_ns) / 1e6,
        "warmup_cycles": args.warmup_cycles,
        "live_tip_position": live_tip_position,
        "live_tip_detach_ms": (detached_ns - last_warm_ns) / 1e6,
        "warm_to_detach_gap_ms": (detached_ns - last_warm_ns) / 1e6,
        "idle_wait_ms": idle_wait_ns / 1e6,
        "detach_to_branch_gap_ms": (branch_started - detached_ns) / 1e6,
        "branch_ready_ms": (branch_ready_ns - branch_started) / 1e6,
        "first_proposal_verify_ms": (first_output_ns - branch_ready_ns) / 1e6,
        "first_commit_ms": (first_commit_ns - first_output_ns) / 1e6,
        "promotion_after_first_ms": promotion_ms,
        "branch_to_first_output_ms": (first_output_ns - branch_started) / 1e6,
        "branch_to_first_commit_ms": (first_commit_ns - branch_started) / 1e6,
        "live_tip_to_first_commit_ms": (first_commit_ns - last_warm_ns) / 1e6,
        "followup_ms": (followup_finished - followup_started) / 1e6,
        "measured_cycles": args.measured_cycles,
        "emitted_tokens": followup_tokens,
        "aggregate_branch_decode_tps": followup_tokens
        / max((followup_finished - branch_ready_ns) / 1e9, 1e-12),
        "first_cycle_emitted": emitted,
        "warmup_tokens": warmup_rows,
        "branch_tokens": branch_rows,
        "token_digest": _token_digest(branch_rows),
        "sibling_state": sibling_state,
        "segmented_delta": segmented_delta,
        "segmented_receipt": segmented_after,
        "memory": _mlx_memory(mx),
        "system_before": before,
        "system_after": after,
        "swap_growth_bytes": swap_growth,
    }

    batch = branch_batch = None
    rows = final_rows = None
    detached = canonical = sibling = None
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    result["memory_after_cleanup"] = _mlx_memory(mx)
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "samples": len(selected),
            "median_branch_ready_ms": statistics.median(
                row["branch_ready_ms"] for row in selected
            ),
            "median_first_proposal_verify_ms": statistics.median(
                row["first_proposal_verify_ms"] for row in selected
            ),
            "median_branch_to_first_commit_ms": statistics.median(
                row["branch_to_first_commit_ms"] for row in selected
            ),
            "median_decode_tps": statistics.median(
                row["aggregate_branch_decode_tps"] for row in selected
            ),
        }
    warm = summary["warm_live_tip"]
    idle = summary["idle_live_tip"]
    summary["warm_vs_idle_first_commit_speedup"] = (
        idle["median_branch_to_first_commit_ms"]
        / warm["median_branch_to_first_commit_ms"]
    )
    summary["all_sibling_tokens_exact"] = all(
        row["branch_tokens"][0] == row["branch_tokens"][1] for row in rows
    )
    summary["all_sibling_states_exact"] = all(
        row["sibling_state"]["equal"] for row in rows
    )
    return summary


def apply_composition_environment(args: argparse.Namespace) -> None:
    if args.qsa_private_delta != "default":
        os.environ["MLX_LM_QSA_PRIVATE_DELTA"] = str(
            args.qsa_private_delta == "on"
        ).lower()
    if args.qsa_exact_set_fold != "default":
        os.environ["MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD"] = str(
            args.qsa_exact_set_fold == "on"
        ).lower()
    if args.qsa_private_delta_min_context is not None:
        floor = str(args.qsa_private_delta_min_context)
        os.environ["MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_M1"] = floor
        os.environ["MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_MN"] = floor


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.utils import load

    from qwen4_mtp_dynamic_join_gate import exact_prompt

    apply_composition_environment(args)

    model, tokenizer = load(args.model)
    model.eval()
    prompt = mx.array(exact_prompt(tokenizer, args.context, "live-tip"), mx.uint32)
    mx.eval(prompt)

    rows = []
    for repetition in range(1, args.reps + 1):
        for arm in arm_order(repetition):
            if args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            row = _run_arm(model, prompt, args, arm)
            row["repetition"] = repetition
            rows.append(row)
            if args.out:
                atomic_write(
                    args.out,
                    {
                        "metadata": plan,
                        "status": "running",
                        "rows": rows,
                    },
                )

    for index in range(0, len(rows), 2):
        pair = rows[index : index + 2]
        if len(pair) == 2 and pair[0]["branch_tokens"] != pair[1]["branch_tokens"]:
            raise AssertionError(
                f"warm/idle token traces differ in repetition {pair[0]['repetition']}"
            )
    return {
        "metadata": plan,
        "status": "complete",
        "summary": _summarize(rows),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--warmup-cycles", type=int, default=8)
    parser.add_argument("--measured-cycles", type=int, default=8)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--branch-mode", choices=BRANCH_MODES, default="physical")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--minimum-system-free-percent", type=int, default=30)
    parser.add_argument("--maximum-swap-growth-mb", type=int, default=16)
    parser.add_argument(
        "--qsa-private-delta", choices=("default", "on", "off"), default="default"
    )
    parser.add_argument(
        "--qsa-exact-set-fold", choices=("default", "on", "off"), default="default"
    )
    parser.add_argument("--qsa-private-delta-min-context", type=int)
    parser.add_argument("--promote-after-first", action="store_true")
    parser.add_argument(
        "--share-qsa-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
