"""Production-model gate for exact Qwen4 N=2 prefix fan-out.

Two gates run against the same loaded model:

1. one retained prompt materialization followed by two serial descendants is
   compared with one two-row KV/QSA/GDN/PLE fan-out;
2. the real ``ParallelSampleGenerator`` serving path is interleaved with the
   incumbent ordinary cache-copy path and reports request wall time and decode
   throughput.

The script never changes optimization environment variables.  It records them
so QSA sharing, PLE, megakernel, and other process-level levers remain visible
in the result.  The candidate must engage and greedy token traces must match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx

from mlx_lm.gdn_prefix_fanout import (
    HybridCachePrefixFanout,
    _clone_cache as _clone_cache_entry,
    gdn_prefix_fanout_stats,
)
from mlx_lm.generate import ParallelSampleGenerator, StopSequenceMatcher
from mlx_lm.hybrid_speculative import prepare_self_mtp_lane
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.utils import load

TRACKED_ENV_PREFIXES = (
    "MLX_LM_",
    "MLX_QWEN",
    "MLX_METAL",
    "MLX_ENABLE_",
)

PMSET_THERM_COMMAND = ("/usr/bin/pmset", "-g", "therm")


def _emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def _parse_pmset_therm(stdout: str, stderr: str = "", returncode: int = 0):
    """Turn rootless ``pmset -g therm`` output into a conservative gate."""

    combined = "\n".join(part for part in (stdout, stderr) if part).strip()
    lower = combined.lower()
    values = {}
    for line in combined.splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z _-]+?)\s*=\s*(-?\d+)\s*$", line)
        if match:
            key = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
            values[key] = int(match.group(2))

    reasons = []
    if returncode != 0:
        reasons.append(f"pmset_exit_{returncode}")

    thermal_clear = "no thermal warning" in lower
    performance_clear = "no performance warning" in lower
    for line in combined.splitlines():
        normalized = line.strip().lower()
        if "thermal warning" in normalized and "no thermal warning" not in normalized:
            reasons.append("thermal_warning_reported")
        if (
            "performance warning" in normalized
            and "no performance warning" not in normalized
        ):
            reasons.append("performance_warning_reported")

    for key, value in values.items():
        if key in {"thermal_level", "performance_limit"} and value > 0:
            reasons.append(f"{key}_{value}")
        if (
            key.endswith("_speed_limit")
            or key.endswith("_scheduler_limit")
            or key == "scheduler_limit"
        ) and value < 100:
            reasons.append(f"{key}_{value}")

    numeric_clear = any(
        key in values
        for key in (
            "thermal_level",
            "performance_limit",
            "cpu_speed_limit",
            "gpu_speed_limit",
            "scheduler_limit",
            "cpu_scheduler_limit",
            "gpu_scheduler_limit",
        )
    )
    if returncode == 0 and not ((thermal_clear and performance_clear) or numeric_clear):
        reasons.append("thermal_state_unverified")
    reasons = list(dict.fromkeys(reasons))
    return {
        "healthy": not reasons,
        "warning": bool(reasons),
        "reasons": reasons,
        "values": values,
        "returncode": int(returncode),
        "stdout": stdout,
        "stderr": stderr,
    }


def _thermal_snapshot(gate, block, attempt, stage, slot=None):
    captured_at = time.time()
    try:
        completed = subprocess.run(
            PMSET_THERM_COMMAND,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        snapshot = _parse_pmset_therm(
            completed.stdout, completed.stderr, completed.returncode
        )
    except (OSError, subprocess.SubprocessError) as error:
        snapshot = _parse_pmset_therm("", str(error), 127)
    snapshot.update(
        {
            "captured_at_epoch_s": captured_at,
            "gate": gate,
            "block": block,
            "attempt": attempt,
            "stage": stage,
            "slot": slot,
        }
    )
    _emit("gdn_prefix_fanout_thermal", **snapshot)
    return snapshot


def _cool_until_stable(
    args,
    gate,
    block,
    attempt,
    slot,
    *,
    snapshot_fn=_thermal_snapshot,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
):
    """Wait through the minimum cooldown and require consecutive clear polls."""

    minimum = float(args.minimum_cooldown_seconds)
    poll = float(args.thermal_poll_seconds)
    maximum = float(args.thermal_max_cooldown_seconds)
    required = int(args.thermal_stable_snapshots)
    started = monotonic_fn()
    snapshots = []
    stable = 0
    while True:
        elapsed = monotonic_fn() - started
        snapshot = snapshot_fn(gate, block, attempt, "cooldown", slot)
        snapshot["cooldown_elapsed_s"] = elapsed
        snapshots.append(snapshot)
        if elapsed >= minimum and snapshot["healthy"]:
            stable += 1
        else:
            stable = 0
        if stable >= required:
            receipt = {
                "recovered": True,
                "elapsed_s": elapsed,
                "stable_snapshots": stable,
                "snapshots": snapshots,
            }
            _emit(
                "gdn_prefix_fanout_cooldown",
                gate=gate,
                block=block,
                attempt=attempt,
                slot=slot,
                recovered=True,
                elapsed_s=elapsed,
                polls=len(snapshots),
            )
            return receipt
        if elapsed >= maximum and stable == 0:
            receipt = {
                "recovered": False,
                "elapsed_s": elapsed,
                "stable_snapshots": stable,
                "snapshots": snapshots,
            }
            _emit(
                "gdn_prefix_fanout_cooldown",
                gate=gate,
                block=block,
                attempt=attempt,
                slot=slot,
                recovered=False,
                elapsed_s=elapsed,
                polls=len(snapshots),
            )
            return receipt
        remaining_minimum = max(0.0, minimum - elapsed)
        delay = poll
        if remaining_minimum:
            delay = min(poll, remaining_minimum) if poll else remaining_minimum
        if delay:
            sleep_fn(delay)


def _thermal_arm(args, gate, block, attempt, slot, enabled, run):
    cooldown = _cool_until_stable(args, gate, block, attempt, slot)
    if not cooldown["recovered"]:
        return None, {
            "cooldown": cooldown,
            "before": None,
            "after": None,
            "abort_reason": "thermal_recovery_timeout",
        }
    before = _thermal_snapshot(gate, block, attempt, "before_arm", slot)
    if not before["healthy"]:
        return None, {
            "cooldown": cooldown,
            "before": before,
            "after": None,
            "abort_reason": "thermal_warning_before_arm",
        }
    result = run(enabled)
    after = _thermal_snapshot(gate, block, attempt, "after_arm", slot)
    result["thermal"] = {
        "cooldown": cooldown,
        "before": before,
        "after": after,
    }
    return result, result["thermal"]


def _closing_baseline_drift(arms):
    return abs(arms[3]["total_ms"] - arms[0]["total_ms"]) / max(
        min(arms[0]["total_ms"], arms[3]["total_ms"]), 1e-9
    )


def _median(rows, key):
    return statistics.median(row[key] for row in rows) if rows else None


def _skipped_serving_gate(reason):
    return {
        "passed": False,
        "failure": {"reason": reason},
        "candidate_arms_executed": 0,
        "token_exact": False,
        "blocks": [],
        "discarded_blocks": [],
    }


def _arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays(item)


def _eval_cache(cache) -> None:
    values = [array for entry in cache for array in _arrays(entry.state)]
    if values:
        mx.eval(*values)


def _clone_cache(cache):
    clones = [_clone_cache_entry(entry) for entry in cache]
    _eval_cache(clones)
    return clones


def _relative_max(left: mx.array, right: mx.array) -> float:
    delta = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    scale = mx.maximum(mx.max(mx.abs(right.astype(mx.float32))), 1e-8)
    return float((delta / scale).item())


def _logical_cache_array(cache, value: mx.array) -> mx.array:
    """Ignore allocator tail capacity while preserving the live KV prefix."""

    offset = getattr(cache, "offset", None)
    if isinstance(offset, int) and value.ndim >= 2 and value.shape[-2] >= offset:
        return value[..., :offset, :]
    return value


def _cache_error(left, right) -> tuple[bool, float]:
    bit_exact = True
    relative_max = 0.0
    if len(left) != len(right):
        raise AssertionError(f"cache widths differ: {len(left)} != {len(right)}")
    for actual, expected in zip(left, right):
        if type(actual) is not type(expected):
            raise AssertionError(
                f"cache types differ: {type(actual).__name__} != "
                f"{type(expected).__name__}"
            )
        actual_offset = getattr(actual, "offset", None)
        expected_offset = getattr(expected, "offset", None)
        if actual_offset != expected_offset:
            raise AssertionError(
                f"cache offsets differ: {actual_offset} != {expected_offset}"
            )
        actual_arrays = [
            _logical_cache_array(actual, value) for value in _arrays(actual.state)
        ]
        expected_arrays = [
            _logical_cache_array(expected, value) for value in _arrays(expected.state)
        ]
        if len(actual_arrays) != len(expected_arrays):
            raise AssertionError("cache state widths differ")
        for lhs, rhs in zip(actual_arrays, expected_arrays):
            if tuple(lhs.shape) != tuple(rhs.shape):
                raise AssertionError(
                    f"cache shapes differ: {tuple(lhs.shape)} != {tuple(rhs.shape)}"
                )
            if lhs.size:
                exact = bool(mx.array_equal(lhs, rhs))
                bit_exact &= exact
                if not exact:
                    relative_max = max(relative_max, _relative_max(lhs, rhs))
    return bit_exact, relative_max


def _prompt_tokens(tokenizer, text: str, count: int) -> list[int]:
    base = list(tokenizer.encode(text, add_special_tokens=False))
    if not base:
        raise ValueError("the benchmark prompt encoded to zero tokens")
    repeated = (base * ((count + len(base) - 1) // len(base)))[:count]
    if len(repeated) < 2:
        raise ValueError("--prompt-tokens must be at least 2")
    return [int(token) for token in repeated]


def _time_ms(fn):
    started = time.perf_counter_ns()
    result = fn()
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6, result


def _counter_delta(before, after):
    return {key: int(after[key] - before.get(key, 0)) for key in after}


def _require_candidate_engaged(result, gate):
    counters = result["counter_delta"]
    if counters["serving_requests"] != 1:
        raise AssertionError(
            f"{gate} candidate request counter was "
            f"{counters['serving_requests']}, expected 1"
        )
    if counters["serving_engaged"] != 1:
        raise AssertionError(
            f"{gate} candidate serving engagement was "
            f"{counters['serving_engaged']}, expected 1"
        )
    if (
        counters["serving_declined_not_n2"]
        or counters["serving_declined_cache"]
        or counters["serving_declined_error"]
    ):
        raise AssertionError(f"{gate} candidate silently declined fan-out")
    if counters["serving_cleanups"] != 1:
        raise AssertionError(
            f"{gate} candidate cleanup counter was "
            f"{counters['serving_cleanups']}, expected 1"
        )


def _print_arm(gate, block, attempt, slot, result):
    print(
        json.dumps(
            {
                "event": "gdn_prefix_fanout_arm",
                "gate": gate,
                "block": block,
                "attempt": attempt,
                "slot": slot,
                "enabled": result["enabled"],
                "prepare_ms": result["prepare_ms"],
                "prepare_tokens_per_second": result["prepare_tokens_per_second"],
                "decode_ms": result["decode_ms"],
                "decode_tokens_per_second": result[
                    "aggregate_decode_tokens_per_second"
                ],
                "total_ms": result["total_ms"],
                "total_tokens_per_second": result["aggregate_total_tokens_per_second"],
                "token_sha256": result["token_sha256"],
                "counter_delta": result["counter_delta"],
                "thermal_healthy": result["thermal"]["after"]["healthy"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _print_block(gate, result):
    print(
        json.dumps(
            {
                "event": "gdn_prefix_fanout_block",
                "gate": gate,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"arms", "thermal"}
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _component_gate(model, prompt, suffix, args):
    detached, _ = prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=0,
        max_tokens=max(2, args.max_tokens),
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
        record_prefix_fanout=True,
    )
    owner = HybridCachePrefixFanout.from_prompt_cache(
        detached.caches.target, enabled=True, strict=True
    )

    def serial():
        common = owner.materialize(owner.span)
        rows = [_clone_cache(common), _clone_cache(common)]
        outputs = []
        for row in range(2):
            output = model(suffix[row : row + 1], cache=rows[row])
            mx.eval(output)
            _eval_cache(rows[row])
            outputs.append(output)
        return mx.concatenate(outputs, axis=0), rows

    def fanout():
        lease = owner.fork(owner.span)
        try:
            output = model(suffix, cache=lease.caches)
            mx.eval(output)
            _eval_cache(lease.caches)
            rows = [[cache.extract(row) for cache in lease.caches] for row in range(2)]
            for cache in rows:
                _eval_cache(cache)
            return output, rows
        finally:
            lease.abort()

    serial_result = serial()
    fanout_result = fanout()
    logits_exact = bool(mx.array_equal(serial_result[0], fanout_result[0]))
    logits_error = (
        0.0 if logits_exact else _relative_max(serial_result[0], fanout_result[0])
    )
    cache_exact = True
    cache_error = 0.0
    for actual, expected in zip(fanout_result[1], serial_result[1]):
        exact, error = _cache_error(actual, expected)
        cache_exact &= exact
        cache_error = max(cache_error, error)

    samples = []
    discarded = []
    failure = None
    for block in range(args.component_reps):
        attempt = 0
        while True:
            _emit(
                "gdn_prefix_fanout_attempt",
                gate="component",
                block=block,
                attempt=attempt,
                status="started",
            )
            arms = []
            discard_reasons = []
            thermal_abort = None
            for slot, enabled in enumerate((False, True, True, False)):

                def run(candidate, serial=serial, fanout=fanout):
                    elapsed, _ = _time_ms(fanout if candidate else serial)
                    return {"enabled": candidate, "total_ms": elapsed}

                arm, thermal = _thermal_arm(
                    args,
                    "component",
                    block,
                    attempt,
                    slot,
                    enabled,
                    run,
                )
                if arm is None:
                    discard_reasons.append(thermal["abort_reason"])
                    thermal_abort = thermal
                    break
                arms.append(arm)
                _emit(
                    "gdn_prefix_fanout_component_arm",
                    gate="component",
                    block=block,
                    attempt=attempt,
                    slot=slot,
                    enabled=enabled,
                    total_ms=arm["total_ms"],
                    thermal_healthy=thermal["after"]["healthy"],
                )
                if not thermal["after"]["healthy"]:
                    discard_reasons.append("thermal_warning_after_arm")
                    break
            if len(arms) == 4:
                baseline = (arms[0]["total_ms"] + arms[3]["total_ms"]) / 2.0
                candidate = (arms[1]["total_ms"] + arms[2]["total_ms"]) / 2.0
                drift = _closing_baseline_drift(arms)
                if drift > args.max_bracket_drift:
                    discard_reasons.append("closing_baseline_drift")
                row = {
                    "block": block,
                    "attempt": attempt,
                    "serial_ms": baseline,
                    "fanout_ms": candidate,
                    "speedup": baseline / candidate,
                    "closing_baseline_drift_fraction": drift,
                    "arms": arms,
                }
            else:
                row = {
                    "block": block,
                    "attempt": attempt,
                    "closing_baseline_drift_fraction": None,
                    "arms": arms,
                }
            row["accepted"] = not discard_reasons
            row["discard_reasons"] = discard_reasons
            if thermal_abort is not None:
                row["thermal_abort"] = thermal_abort
            if not discard_reasons:
                samples.append(row)
                _print_block("component", row)
                break
            discarded.append(row)
            _print_block("component", row)
            if attempt >= args.max_block_retries:
                failure = {
                    "block": block,
                    "attempt": attempt,
                    "reason": "retry_budget_exhausted",
                    "discard_reasons": discard_reasons,
                }
                _emit("gdn_prefix_fanout_gate_failed", gate="component", **failure)
                break
            _emit(
                "gdn_prefix_fanout_retry",
                gate="component",
                block=block,
                attempt=attempt,
                next_attempt=attempt + 1,
                discard_reasons=discard_reasons,
                retries_remaining=args.max_block_retries - attempt,
            )
            attempt += 1
        if failure:
            break
    owner.close()
    for cache in detached.caches.target:
        cache.stop_speculation()
    return {
        "logits_bit_exact": logits_exact,
        "logits_relative_max": logits_error,
        "cache_bit_exact": cache_exact,
        "cache_relative_max": cache_error,
        "passed": failure is None and len(samples) == args.component_reps,
        "failure": failure,
        "median_speedup": _median(samples, "speedup"),
        "samples": samples,
        "discarded_blocks": discarded,
    }


def _serving_once(model, prompt, args, enabled):
    config = {
        "num_draft": args.num_draft,
        "persistent": True,
        "rate_gate": False,
        "share_qsa_indices": args.share_qsa_indices,
        "sampling_temp": 0.0,
        "accept_rule": "residual",
        "gdn_prefix_fanout": enabled,
    }
    counters_before = gdn_prefix_fanout_stats()
    started = time.perf_counter_ns()
    parallel = ParallelSampleGenerator(
        model,
        None,
        prompt[-1],
        2,
        max_tokens=args.max_tokens,
        stop_matchers=[StopSequenceMatcher(), StopSequenceMatcher()],
        all_tokens=prompt[:-1],
        self_mtp=config,
        lane_rng=LaneRNG(args.seed),
        mtp_prompt=prompt,
        prefill_step_size=args.prefill_step_size,
    )
    mx.synchronize()
    prepared_ms = (time.perf_counter_ns() - started) / 1e6
    rows = [[], []]
    decode_started = time.perf_counter_ns()
    try:
        while len(parallel):
            for row, response in parallel.next():
                rows[row].append(int(response.token))
    finally:
        parallel.close()
    mx.synchronize()
    finished = time.perf_counter_ns()
    decode_ms = (finished - decode_started) / 1e6
    total_ms = (finished - started) / 1e6
    count = sum(len(row) for row in rows)
    prepare_tokens = len(prompt)
    return {
        "enabled": enabled,
        "prepare_ms": prepared_ms,
        "prepare_tokens": prepare_tokens,
        "prepare_tokens_per_second": prepare_tokens / (prepared_ms / 1000.0),
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "aggregate_decode_tokens_per_second": count / (decode_ms / 1000.0),
        "aggregate_request_tokens_per_second": count / (total_ms / 1000.0),
        "aggregate_total_tokens_per_second": (prepare_tokens + count)
        / (total_ms / 1000.0),
        "tokens": rows,
        "token_sha256": [
            hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in rows
        ],
        "counter_delta": _counter_delta(counters_before, gdn_prefix_fanout_stats()),
    }


def _serving_row(block, attempt, arms):
    baseline = [arms[0], arms[3]]
    candidate = [arms[1], arms[2]]
    base_total = statistics.mean(item["total_ms"] for item in baseline)
    cand_total = statistics.mean(item["total_ms"] for item in candidate)
    base_decode = statistics.mean(item["decode_ms"] for item in baseline)
    cand_decode = statistics.mean(item["decode_ms"] for item in candidate)
    return {
        "block": block,
        "attempt": attempt,
        "baseline_total_ms": base_total,
        "candidate_total_ms": cand_total,
        "total_wall_speedup": base_total / cand_total,
        "baseline_prepare_tokens_per_second": statistics.mean(
            item["prepare_tokens_per_second"] for item in baseline
        ),
        "candidate_prepare_tokens_per_second": statistics.mean(
            item["prepare_tokens_per_second"] for item in candidate
        ),
        "baseline_decode_ms": base_decode,
        "candidate_decode_ms": cand_decode,
        "decode_speedup": base_decode / cand_decode,
        "baseline_decode_tokens_per_second": statistics.mean(
            item["aggregate_decode_tokens_per_second"] for item in baseline
        ),
        "candidate_decode_tokens_per_second": statistics.mean(
            item["aggregate_decode_tokens_per_second"] for item in candidate
        ),
        "baseline_total_tokens_per_second": statistics.mean(
            item["aggregate_total_tokens_per_second"] for item in baseline
        ),
        "candidate_total_tokens_per_second": statistics.mean(
            item["aggregate_total_tokens_per_second"] for item in candidate
        ),
        "closing_baseline_drift_fraction": _closing_baseline_drift(arms),
        "arms": arms,
    }


def _run_serving_blocks(gate, gate_label, reps, args, run_once, row_builder):
    blocks = []
    discarded = []
    candidate_arms_executed = 0
    failure = None
    for block in range(reps):
        attempt = 0
        while True:
            _emit(
                "gdn_prefix_fanout_attempt",
                gate=gate,
                block=block,
                attempt=attempt,
                status="started",
            )
            arms = []
            discard_reasons = []
            thermal_abort = None
            for slot, enabled in enumerate((False, True, True, False)):
                arm, thermal = _thermal_arm(
                    args,
                    gate,
                    block,
                    attempt,
                    slot,
                    enabled,
                    run_once,
                )
                if arm is None:
                    discard_reasons.append(thermal["abort_reason"])
                    thermal_abort = thermal
                    break
                arms.append(arm)
                candidate_arms_executed += int(enabled)
                _print_arm(gate, block, attempt, slot, arm)
                if enabled:
                    _require_candidate_engaged(arm, gate_label)
                if not thermal["after"]["healthy"]:
                    discard_reasons.append("thermal_warning_after_arm")
                    break

            if len(arms) == 4:
                if any(item["tokens"] != arms[0]["tokens"] for item in arms[1:]):
                    raise AssertionError(
                        f"{gate_label} candidate and baseline token traces differ"
                    )
                row = row_builder(block, attempt, arms)
                if row["closing_baseline_drift_fraction"] > args.max_bracket_drift:
                    discard_reasons.append("closing_baseline_drift")
            else:
                row = {
                    "block": block,
                    "attempt": attempt,
                    "arms": arms,
                    "closing_baseline_drift_fraction": None,
                }

            row["accepted"] = not discard_reasons
            row["discard_reasons"] = discard_reasons
            if thermal_abort is not None:
                row["thermal_abort"] = thermal_abort
            if not discard_reasons:
                blocks.append(row)
                _print_block(gate, row)
                break

            discarded.append(row)
            _print_block(gate, row)
            if attempt >= args.max_block_retries:
                failure = {
                    "block": block,
                    "attempt": attempt,
                    "reason": "retry_budget_exhausted",
                    "discard_reasons": discard_reasons,
                }
                _emit(
                    "gdn_prefix_fanout_gate_failed",
                    gate=gate,
                    **failure,
                )
                break
            _emit(
                "gdn_prefix_fanout_retry",
                gate=gate,
                block=block,
                attempt=attempt,
                next_attempt=attempt + 1,
                discard_reasons=discard_reasons,
                retries_remaining=args.max_block_retries - attempt,
            )
            attempt += 1
        if failure:
            break
    return {
        "passed": failure is None and len(blocks) == reps,
        "failure": failure,
        "candidate_arms_executed": candidate_arms_executed,
        "blocks": blocks,
        "discarded_blocks": discarded,
    }


def _serving_gate(model, prompt, args):
    result = _run_serving_blocks(
        "fresh_prompt",
        "fresh-prompt",
        args.serving_reps,
        args,
        lambda enabled: _serving_once(model, prompt, args, enabled),
        _serving_row,
    )
    result.update(
        {
            "token_exact": result["passed"],
            "median_total_wall_speedup": _median(
                result["blocks"], "total_wall_speedup"
            ),
            "median_decode_speedup": _median(result["blocks"], "decode_speedup"),
        }
    )
    return result


def _prepare_cached_prefix(model, prefix, args):
    """Capture one exact target-cache plus persistent-MTP APC sidecar."""

    started = time.perf_counter_ns()
    detached, _ = prepare_self_mtp_lane(
        mx.array(prefix, mx.uint32),
        model,
        uid=0,
        max_tokens=max(2, args.max_tokens),
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
    _eval_cache(detached.caches.target)
    _eval_cache(detached.caches.draft)
    mx.eval(detached.lane.seed_h)
    mx.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return {
        "target": detached.caches.target,
        "draft": detached.caches.draft,
        "seed_h": detached.lane.seed_h,
        "tokens": list(prefix),
        "build_ms": elapsed_ms,
        "build_tokens_per_second": len(prefix) / (elapsed_ms / 1000.0),
    }


def _cached_serving_once(model, cached, tail, args, enabled):
    """Run one serving arm from a cloned APC-style target and MTP sidecar."""

    config = {
        "num_draft": args.num_draft,
        "persistent": True,
        "rate_gate": False,
        "share_qsa_indices": args.share_qsa_indices,
        "sampling_temp": 0.0,
        "accept_rule": "residual",
        "gdn_prefix_fanout": enabled,
    }
    counters_before = gdn_prefix_fanout_stats()
    started = time.perf_counter_ns()
    target = _clone_cache(cached["target"])
    draft = _clone_cache(cached["draft"])
    seed_h = mx.array(cached["seed_h"])
    mx.eval(seed_h)
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
    prepared_ms = (time.perf_counter_ns() - started) / 1e6
    rows = [[], []]
    decode_started = time.perf_counter_ns()
    try:
        while len(parallel):
            for row, response in parallel.next():
                rows[row].append(int(response.token))
    finally:
        parallel.close()
    mx.synchronize()
    finished = time.perf_counter_ns()
    decode_ms = (finished - decode_started) / 1e6
    total_ms = (finished - started) / 1e6
    count = sum(len(row) for row in rows)
    prepare_tokens = len(tail)
    return {
        "enabled": enabled,
        "prepare_ms": prepared_ms,
        "prepare_tokens": prepare_tokens,
        "prepare_tokens_per_second": prepare_tokens / (prepared_ms / 1000.0),
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "aggregate_decode_tokens_per_second": count / (decode_ms / 1000.0),
        "aggregate_request_tokens_per_second": count / (total_ms / 1000.0),
        "aggregate_total_tokens_per_second": (prepare_tokens + count)
        / (total_ms / 1000.0),
        "tokens": rows,
        "token_sha256": [
            hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in rows
        ],
        "counter_delta": _counter_delta(counters_before, gdn_prefix_fanout_stats()),
    }


def _cached_serving_row(block, attempt, arms):
    baseline = [arms[0], arms[3]]
    candidate = [arms[1], arms[2]]
    base_prepare = statistics.mean(item["prepare_ms"] for item in baseline)
    cand_prepare = statistics.mean(item["prepare_ms"] for item in candidate)
    base_decode = statistics.mean(item["decode_ms"] for item in baseline)
    cand_decode = statistics.mean(item["decode_ms"] for item in candidate)
    base_total = statistics.mean(item["total_ms"] for item in baseline)
    cand_total = statistics.mean(item["total_ms"] for item in candidate)
    return {
        "block": block,
        "attempt": attempt,
        "baseline_prepare_ms": base_prepare,
        "candidate_prepare_ms": cand_prepare,
        "prepare_wall_speedup": base_prepare / cand_prepare,
        "baseline_prepare_tokens_per_second": statistics.mean(
            item["prepare_tokens_per_second"] for item in baseline
        ),
        "candidate_prepare_tokens_per_second": statistics.mean(
            item["prepare_tokens_per_second"] for item in candidate
        ),
        "baseline_decode_ms": base_decode,
        "candidate_decode_ms": cand_decode,
        "decode_wall_speedup": base_decode / cand_decode,
        "baseline_decode_tokens_per_second": statistics.mean(
            item["aggregate_decode_tokens_per_second"] for item in baseline
        ),
        "candidate_decode_tokens_per_second": statistics.mean(
            item["aggregate_decode_tokens_per_second"] for item in candidate
        ),
        "baseline_total_ms": base_total,
        "candidate_total_ms": cand_total,
        "total_wall_speedup": base_total / cand_total,
        "baseline_total_tokens_per_second": statistics.mean(
            item["aggregate_total_tokens_per_second"] for item in baseline
        ),
        "candidate_total_tokens_per_second": statistics.mean(
            item["aggregate_total_tokens_per_second"] for item in candidate
        ),
        "closing_baseline_drift_fraction": _closing_baseline_drift(arms),
        "arms": arms,
    }


def _cached_serving_gate(model, prompt, args):
    prefix_tokens = args.cached_prefix_tokens
    if prefix_tokens == 0:
        prefix_tokens = len(prompt) - args.cached_tail_tokens
    if not 1 <= prefix_tokens < len(prompt):
        raise ValueError(
            "cached prefix must retain at least one token and leave a non-empty tail"
        )
    prefix = prompt[:prefix_tokens]
    tail = prompt[prefix_tokens:]
    cached = _prepare_cached_prefix(model, prefix, args)
    print(
        json.dumps(
            {
                "event": "gdn_prefix_fanout_cached_snapshot",
                "prefix_tokens": len(prefix),
                "tail_tokens": len(tail),
                "build_ms": cached["build_ms"],
                "build_tokens_per_second": cached["build_tokens_per_second"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    result = _run_serving_blocks(
        "cached_prefix",
        "cached-prefix",
        args.cached_serving_reps,
        args,
        lambda enabled: _cached_serving_once(model, cached, tail, args, enabled),
        _cached_serving_row,
    )
    result.update(
        {
            "token_exact": result["passed"],
            "prefix_tokens": len(prefix),
            "tail_tokens": len(tail),
            "snapshot_build_ms": cached["build_ms"],
            "snapshot_build_tokens_per_second": cached["build_tokens_per_second"],
            "median_prepare_wall_speedup": _median(
                result["blocks"], "prepare_wall_speedup"
            ),
            "median_decode_wall_speedup": _median(
                result["blocks"], "decode_wall_speedup"
            ),
            "median_total_wall_speedup": _median(
                result["blocks"], "total_wall_speedup"
            ),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Explain how an exact speculative cache transaction works. ",
    )
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--suffix-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--component-reps", type=int, default=3)
    parser.add_argument("--serving-reps", type=int, default=2)
    parser.add_argument(
        "--cached-prefix-tokens",
        type=int,
        default=0,
        help=(
            "Tokens retained in the target+MTP sidecar snapshot. Zero uses "
            "--prompt-tokens minus --cached-tail-tokens."
        ),
    )
    parser.add_argument("--cached-tail-tokens", type=int, default=8)
    parser.add_argument("--cached-serving-reps", type=int, default=2)
    parser.add_argument(
        "--minimum-cooldown-seconds",
        "--cool-seconds",
        dest="minimum_cooldown_seconds",
        type=float,
        default=60.0,
        help="Minimum cooldown before every measured arm (default: 60s).",
    )
    parser.add_argument("--thermal-poll-seconds", type=float, default=5.0)
    parser.add_argument("--thermal-max-cooldown-seconds", type=float, default=600.0)
    parser.add_argument("--thermal-stable-snapshots", type=int, default=2)
    parser.add_argument(
        "--max-bracket-drift",
        type=float,
        default=0.05,
        help="Maximum A1/A2 total-wall drift before discarding the ABBA block.",
    )
    parser.add_argument("--max-block-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--share-qsa-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=0.0,
        help=(
            "Maximum serial-vs-fanout cache/logit relative error. Default 0 "
            "keeps the promotion gate lossless; any relaxation is explicit."
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prompt_tokens < 2 or args.suffix_tokens < 1:
        parser.error("prompt tokens must be >=2 and suffix tokens must be positive")
    if args.component_reps < 1 or args.serving_reps < 1 or args.cached_serving_reps < 1:
        parser.error("component, serving, and cached-serving reps must be positive")
    if args.cached_prefix_tokens < 0 or args.cached_tail_tokens < 1:
        parser.error("cached prefix must be non-negative and cached tail positive")
    if (
        args.minimum_cooldown_seconds < 0
        or args.thermal_poll_seconds < 0
        or args.thermal_max_cooldown_seconds < args.minimum_cooldown_seconds
    ):
        parser.error(
            "thermal intervals must be non-negative and max cooldown must be "
            ">= minimum cooldown"
        )
    if args.thermal_stable_snapshots < 2:
        parser.error("thermal stability requires at least two clear snapshots")
    if args.max_bracket_drift < 0 or args.max_block_retries < 0:
        parser.error("drift threshold and retry budget must be non-negative")
    resolved_cached_prefix = args.cached_prefix_tokens or (
        args.prompt_tokens - args.cached_tail_tokens
    )
    if not 1 <= resolved_cached_prefix < args.prompt_tokens:
        parser.error(
            "cached prefix must retain at least one token and leave a non-empty tail"
        )

    model, tokenizer = load(args.model)
    model.eval()
    prompt = _prompt_tokens(tokenizer, args.prompt, args.prompt_tokens)
    suffix = mx.array(
        [
            prompt[: args.suffix_tokens],
            list(reversed(prompt[-args.suffix_tokens :])),
        ],
        mx.uint32,
    )
    mx.eval(suffix)

    gdn_prefix_fanout_stats(reset=True)
    component = _component_gate(model, prompt, suffix, args)
    if component["passed"]:
        serving = _serving_gate(model, prompt, args)
    else:
        serving = _skipped_serving_gate("component_gate_failed")
    if serving["passed"]:
        cached_serving = _cached_serving_gate(model, prompt, args)
    else:
        cached_serving = _skipped_serving_gate("fresh_prompt_gate_failed")
    counters = gdn_prefix_fanout_stats()
    result = {
        "model": str(Path(args.model).resolve()),
        "geometry": {
            "prompt_tokens": len(prompt),
            "suffix_tokens": args.suffix_tokens,
            "rows": 2,
            "num_draft": args.num_draft,
            "max_tokens": args.max_tokens,
        },
        "thermal_controls": {
            "command": list(PMSET_THERM_COMMAND),
            "minimum_cooldown_seconds": args.minimum_cooldown_seconds,
            "poll_seconds": args.thermal_poll_seconds,
            "max_cooldown_seconds": args.thermal_max_cooldown_seconds,
            "stable_snapshots_required": args.thermal_stable_snapshots,
            "max_bracket_drift": args.max_bracket_drift,
            "max_block_retries": args.max_block_retries,
        },
        "levers": {
            "self_mtp": True,
            "persistent_mtp": True,
            "share_qsa_indices": args.share_qsa_indices,
            "ple": "model-configured",
            "apc": "target cache plus exact persistent-MTP sidecar exercised",
            "environment": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith(TRACKED_ENV_PREFIXES)
            },
        },
        "component": component,
        "serving": serving,
        "cached_serving": cached_serving,
        "counters": counters,
    }
    failures = []
    if counters["hybrid_fanout_batches"] < 1:
        failures.append("GDN prefix fan-out did not engage")
    expected_serving_engagements = (
        serving["candidate_arms_executed"] + cached_serving["candidate_arms_executed"]
    )
    if counters["serving_engaged"] != expected_serving_engagements:
        failures.append(
            "serving fan-out engagement mismatch: "
            f"{counters['serving_engaged']} != {expected_serving_engagements}"
        )
    if counters["serving_declined_cache"] or counters["serving_declined_error"]:
        failures.append("a serving candidate arm silently declined fan-out")
    if (
        max(component["logits_relative_max"], component["cache_relative_max"])
        > args.max_relative_error
    ):
        failures.append(
            "full-model serial/fan-out exactness exceeded " f"{args.max_relative_error}"
        )
    if not component["passed"]:
        failures.append("component thermal/drift gate did not complete")
    if not serving["passed"]:
        failures.append("fresh-prompt thermal/drift gate did not complete")
    if not cached_serving["passed"]:
        failures.append("cached-prefix thermal/drift gate did not complete")
    result["passed"] = not failures
    result["failures"] = failures
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
