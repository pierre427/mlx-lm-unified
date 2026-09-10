"""Thermally bracket the cached-prefix preparation alternatives for Qwen4.

This is a focused follow-up to ``qwen4_gdn_prefix_fanout_full_model_ab``.  It
holds the 16K APC snapshot, model, decode settings, and baseline path constant,
then qualifies the two preparation mechanisms independently and together:

* one-shot fan-out from the request-private live tip.

Every candidate is measured in its own A/B/B/A bracket against the ordinary
self-MTP path.  A block is discarded when the closing baseline drifts or a
thermal snapshot is unhealthy.  Greedy output must remain bit-identical and
each enabled mechanism must produce a non-zero receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

try:
    from benchmarks.qwen4_gdn_prefix_fanout_full_model_ab import (
        PMSET_THERM_COMMAND,
        TRACKED_ENV_PREFIXES,
        _clone_cache,
        _counter_delta,
        _prepare_cached_prefix,
        _prompt_tokens,
        _thermal_arm,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/this_file.py``.
    from qwen4_gdn_prefix_fanout_full_model_ab import (
        PMSET_THERM_COMMAND,
        TRACKED_ENV_PREFIXES,
        _clone_cache,
        _counter_delta,
        _prepare_cached_prefix,
        _prompt_tokens,
        _thermal_arm,
    )
from mlx_lm.gdn_prefix_fanout import gdn_prefix_fanout_stats
from mlx_lm.generate import ParallelSampleGenerator, StopSequenceMatcher
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.utils import load


VARIANTS = {
    "ordinary": {},
    "immutable": {"gdn_prefix_fanout": True},
    "consume": {
        "gdn_prefix_fanout": True,
        "gdn_prefix_fanout_consume": True,
    },
}


def _emit(event, **payload):
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def _once(model, cached, tail, args, variant):
    switches = VARIANTS[variant]
    config = {
        "num_draft": args.num_draft,
        "persistent": True,
        "rate_gate": False,
        "share_qsa_indices": args.share_qsa_indices,
        "sampling_temp": 0.0,
        "accept_rule": "residual",
        "gdn_prefix_fanout": False,
        "gdn_prefix_fanout_consume": False,
        **switches,
    }
    fanout_before = gdn_prefix_fanout_stats()
    started = time.perf_counter_ns()
    stages = {}
    target = _clone_cache(cached["target"])
    if args.stage_timing:
        mx.synchronize()
        now = time.perf_counter_ns()
        stages["target_clone_ms"] = (now - started) / 1e6
        stage_started = now
    draft = _clone_cache(cached["draft"])
    if args.stage_timing:
        mx.synchronize()
        now = time.perf_counter_ns()
        stages["draft_clone_ms"] = (now - stage_started) / 1e6
        stage_started = now
    seed_h = mx.array(cached["seed_h"])
    mx.eval(seed_h)
    if args.stage_timing:
        mx.synchronize()
        now = time.perf_counter_ns()
        stages["seed_clone_ms"] = (now - stage_started) / 1e6
        stage_started = now
    parallel = ParallelSampleGenerator(
        model,
        target,
        tail[-1],
        2,
        max_tokens=args.max_tokens,
        stop_matchers=[StopSequenceMatcher(), StopSequenceMatcher()],
        all_tokens=cached["tokens"],
        self_mtp=config,
        mtp_state=(draft, seed_h),
        lane_rng=LaneRNG(args.seed),
        mtp_prompt=tail,
        prefill_step_size=args.prefill_step_size,
    )
    mx.synchronize()
    prepared_ns = time.perf_counter_ns()
    if args.stage_timing:
        stages["generator_prepare_ms"] = (prepared_ns - stage_started) / 1e6
    rows = [[], []]
    try:
        while len(parallel):
            for row, response in parallel.next():
                rows[row].append(int(response.token))
    finally:
        parallel.close()
    mx.synchronize()
    finished_ns = time.perf_counter_ns()
    prepare_ms = (prepared_ns - started) / 1e6
    decode_ms = (finished_ns - prepared_ns) / 1e6
    total_ms = (finished_ns - started) / 1e6
    decode_tokens = sum(map(len, rows))
    return {
        "variant": variant,
        "switches": switches,
        "prepare_ms": prepare_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "decode_tokens": decode_tokens,
        "aggregate_decode_tokens_per_second": decode_tokens / (decode_ms / 1000),
        "aggregate_request_tokens_per_second": decode_tokens / (total_ms / 1000),
        "token_sha256": [
            hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in rows
        ],
        "tokens": rows,
        "diagnostic_stages": stages,
        "fanout_delta": _counter_delta(
            fanout_before, gdn_prefix_fanout_stats()
        ),
    }


def _require_receipts(arm):
    variant = arm["variant"]
    fanout = arm["fanout_delta"]
    switches = VARIANTS[variant]
    wants_fanout = switches.get("gdn_prefix_fanout", False)
    wants_consume = switches.get("gdn_prefix_fanout_consume", False)
    if bool(fanout["serving_engaged"]) != wants_fanout:
        raise AssertionError(f"{variant}: fan-out engagement receipt mismatch")
    if bool(fanout["hybrid_tip_fanout_batches"]) != wants_consume:
        raise AssertionError(f"{variant}: consuming fan-out receipt mismatch")
    if wants_fanout and (
        fanout["serving_declined_not_n2"]
        or fanout["serving_declined_cache"]
        or fanout["serving_declined_error"]
    ):
        raise AssertionError(f"{variant}: fan-out silently declined")


def _summarize(block, candidate, attempt, arms):
    baselines = (arms[0], arms[3])
    candidates = (arms[1], arms[2])

    def mean(rows, key):
        return statistics.mean(row[key] for row in rows)

    baseline_prepare = mean(baselines, "prepare_ms")
    candidate_prepare = mean(candidates, "prepare_ms")
    baseline_decode = mean(baselines, "decode_ms")
    candidate_decode = mean(candidates, "decode_ms")
    baseline_total = mean(baselines, "total_ms")
    candidate_total = mean(candidates, "total_ms")
    baseline_tps = mean(baselines, "aggregate_decode_tokens_per_second")
    candidate_tps = mean(candidates, "aggregate_decode_tokens_per_second")
    drift = abs(arms[3]["total_ms"] - arms[0]["total_ms"]) / max(
        min(arms[0]["total_ms"], arms[3]["total_ms"]), 1e-9
    )
    return {
        "candidate": candidate,
        "block": block,
        "attempt": attempt,
        "baseline_prepare_ms": baseline_prepare,
        "candidate_prepare_ms": candidate_prepare,
        "prepare_wall_speedup": baseline_prepare / candidate_prepare,
        "baseline_decode_ms": baseline_decode,
        "candidate_decode_ms": candidate_decode,
        "decode_wall_speedup": baseline_decode / candidate_decode,
        "baseline_decode_tokens_per_second": baseline_tps,
        "candidate_decode_tokens_per_second": candidate_tps,
        "decode_tps_ratio": candidate_tps / baseline_tps,
        "baseline_total_ms": baseline_total,
        "candidate_total_ms": candidate_total,
        "total_wall_speedup": baseline_total / candidate_total,
        "closing_baseline_drift_fraction": drift,
        "arms": arms,
    }


def _candidate_gate(model, cached, tail, args, candidate):
    accepted = []
    discarded = []
    failure = None
    for block in range(args.reps):
        attempt = 0
        while True:
            _emit(
                "qwen4_gdn_prep_attempt",
                candidate=candidate,
                block=block,
                attempt=attempt,
            )
            arms = []
            reasons = []
            for slot, variant in enumerate(
                ("ordinary", candidate, candidate, "ordinary")
            ):
                arm, thermal = _thermal_arm(
                    args,
                    f"prep_{candidate}",
                    block,
                    attempt,
                    slot,
                    variant,
                    lambda selected: _once(
                        model, cached, tail, args, selected
                    ),
                )
                if arm is None:
                    reasons.append(thermal["abort_reason"])
                    break
                arms.append(arm)
                _require_receipts(arm)
                _emit(
                    "qwen4_gdn_prep_arm",
                    candidate=candidate,
                    block=block,
                    attempt=attempt,
                    slot=slot,
                    variant=variant,
                    prepare_ms=arm["prepare_ms"],
                    decode_ms=arm["decode_ms"],
                    total_ms=arm["total_ms"],
                    decode_tokens_per_second=arm[
                        "aggregate_decode_tokens_per_second"
                    ],
                    fanout_delta=arm["fanout_delta"],
                    thermal_healthy=thermal["after"]["healthy"],
                )
                if not thermal["after"]["healthy"]:
                    reasons.append("thermal_warning_after_arm")
                    break
            if len(arms) == 4:
                mismatched_slots = [
                    slot
                    for slot, arm in enumerate(arms[1:], start=1)
                    if arm["tokens"] != arms[0]["tokens"]
                ]
                if mismatched_slots:
                    reasons.append("greedy_token_trace_mismatch")
                row = _summarize(block, candidate, attempt, arms)
                row["mismatched_token_slots"] = mismatched_slots
                if row["closing_baseline_drift_fraction"] > args.max_bracket_drift:
                    reasons.append("closing_baseline_drift")
            else:
                row = {
                    "candidate": candidate,
                    "block": block,
                    "attempt": attempt,
                    "arms": arms,
                    "closing_baseline_drift_fraction": None,
                }
            row["accepted"] = not reasons
            row["discard_reasons"] = reasons
            _emit(
                "qwen4_gdn_prep_block",
                **{key: value for key, value in row.items() if key != "arms"},
            )
            if not reasons:
                accepted.append(row)
                break
            discarded.append(row)
            if "greedy_token_trace_mismatch" in reasons:
                failure = {
                    "candidate": candidate,
                    "block": block,
                    "attempt": attempt,
                    "discard_reasons": reasons,
                }
                break
            if attempt >= args.max_block_retries:
                failure = {
                    "candidate": candidate,
                    "block": block,
                    "attempt": attempt,
                    "discard_reasons": reasons,
                }
                break
            attempt += 1
        if failure:
            break
    return {
        "passed": failure is None and len(accepted) == args.reps,
        "failure": failure,
        "blocks": accepted,
        "discarded_blocks": discarded,
        "median_prepare_wall_speedup": statistics.median(
            row["prepare_wall_speedup"] for row in accepted
        ) if accepted else None,
        "median_decode_tps_ratio": statistics.median(
            row["decode_tps_ratio"] for row in accepted
        ) if accepted else None,
        "median_total_wall_speedup": statistics.median(
            row["total_wall_speedup"] for row in accepted
        ) if accepted else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Explain how an exact speculative cache transaction works. ",
    )
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--cached-tail-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(name for name in VARIANTS if name != "ordinary"),
        default=tuple(name for name in VARIANTS if name != "ordinary"),
    )
    parser.add_argument("--minimum-cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--thermal-poll-seconds", type=float, default=5.0)
    parser.add_argument("--thermal-max-cooldown-seconds", type=float, default=600.0)
    parser.add_argument("--thermal-stable-snapshots", type=int, default=2)
    parser.add_argument("--max-bracket-drift", type=float, default=0.05)
    parser.add_argument("--max-block-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--stage-timing",
        action="store_true",
        help=(
            "Synchronize after each cache-clone boundary for diagnostic "
            "attribution. This intentionally perturbs wall-clock timing."
        ),
    )
    parser.add_argument(
        "--share-qsa-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prompt_tokens <= args.cached_tail_tokens or args.cached_tail_tokens < 1:
        parser.error("the cached tail must be positive and shorter than the prompt")
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.minimum_cooldown_seconds < 0 or args.thermal_stable_snapshots < 2:
        parser.error("cooldown must be non-negative and require two clear snapshots")

    model, tokenizer = load(args.model)
    model.eval()
    prompt = _prompt_tokens(tokenizer, args.prompt, args.prompt_tokens)
    split = len(prompt) - args.cached_tail_tokens
    cached = _prepare_cached_prefix(model, prompt[:split], args)
    tail = prompt[split:]
    _emit(
        "qwen4_gdn_prep_snapshot",
        prefix_tokens=split,
        tail_tokens=len(tail),
        build_ms=cached["build_ms"],
        build_tokens_per_second=cached["build_tokens_per_second"],
    )

    gdn_prefix_fanout_stats(reset=True)
    candidates = {
        name: _candidate_gate(model, cached, tail, args, name)
        for name in args.candidates
    }
    failures = [name for name, gate in candidates.items() if not gate["passed"]]
    result = {
        "passed": not failures,
        "failures": failures,
        "model": str(Path(args.model).resolve()),
        "geometry": {
            "prompt_tokens": len(prompt),
            "cached_prefix_tokens": split,
            "cached_tail_tokens": len(tail),
            "rows": 2,
            "num_draft": args.num_draft,
            "max_tokens": args.max_tokens,
        },
        "thermal_controls": {
            "command": list(PMSET_THERM_COMMAND),
            "minimum_cooldown_seconds": args.minimum_cooldown_seconds,
            "poll_seconds": args.thermal_poll_seconds,
            "maximum_cooldown_seconds": args.thermal_max_cooldown_seconds,
            "stable_snapshots_required": args.thermal_stable_snapshots,
            "max_bracket_drift": args.max_bracket_drift,
            "max_block_retries": args.max_block_retries,
        },
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(TRACKED_ENV_PREFIXES)
        },
        "candidates": candidates,
        "fanout_counters": gdn_prefix_fanout_stats(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    if failures:
        raise AssertionError(f"prep candidates failed: {failures}")


if __name__ == "__main__":
    main()
