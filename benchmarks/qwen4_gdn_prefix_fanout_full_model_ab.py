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
import statistics
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
            _logical_cache_array(expected, value)
            for value in _arrays(expected.state)
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


def _print_arm(gate, block, slot, result):
    print(
        json.dumps(
            {
                "event": "gdn_prefix_fanout_arm",
                "gate": gate,
                "block": block,
                "slot": slot,
                "enabled": result["enabled"],
                "prepare_ms": result["prepare_ms"],
                "prepare_tokens_per_second": result[
                    "prepare_tokens_per_second"
                ],
                "decode_ms": result["decode_ms"],
                "decode_tokens_per_second": result[
                    "aggregate_decode_tokens_per_second"
                ],
                "total_ms": result["total_ms"],
                "total_tokens_per_second": result[
                    "aggregate_total_tokens_per_second"
                ],
                "token_sha256": result["token_sha256"],
                "counter_delta": result["counter_delta"],
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
                **{key: value for key, value in result.items() if key != "arms"},
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
            rows = [
                [cache.extract(row) for cache in lease.caches] for row in range(2)
            ]
            for cache in rows:
                _eval_cache(cache)
            return output, rows
        finally:
            lease.abort()

    serial_result = serial()
    fanout_result = fanout()
    logits_exact = bool(mx.array_equal(serial_result[0], fanout_result[0]))
    logits_error = (
        0.0
        if logits_exact
        else _relative_max(serial_result[0], fanout_result[0])
    )
    cache_exact = True
    cache_error = 0.0
    for actual, expected in zip(fanout_result[1], serial_result[1]):
        exact, error = _cache_error(actual, expected)
        cache_exact &= exact
        cache_error = max(cache_error, error)

    samples = []
    for block in range(args.component_reps):
        a1, _ = _time_ms(serial)
        b1, _ = _time_ms(fanout)
        b2, _ = _time_ms(fanout)
        a2, _ = _time_ms(serial)
        baseline = (a1 + a2) / 2.0
        candidate = (b1 + b2) / 2.0
        samples.append(
            {
                "block": block,
                "serial_ms": baseline,
                "fanout_ms": candidate,
                "speedup": baseline / candidate,
                "bracket_drift_fraction": abs(a2 - a1) / max(min(a1, a2), 1e-9),
            }
        )
    owner.close()
    for cache in detached.caches.target:
        cache.stop_speculation()
    return {
        "logits_bit_exact": logits_exact,
        "logits_relative_max": logits_error,
        "cache_bit_exact": cache_exact,
        "cache_relative_max": cache_error,
        "median_speedup": statistics.median(row["speedup"] for row in samples),
        "samples": samples,
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
        "counter_delta": _counter_delta(
            counters_before, gdn_prefix_fanout_stats()
        ),
    }


def _serving_gate(model, prompt, args):
    blocks = []
    for block in range(args.serving_reps):
        arms = []
        for slot, enabled in enumerate((False, True, True, False)):
            if args.cool_seconds:
                time.sleep(args.cool_seconds)
            arm = _serving_once(model, prompt, args, enabled)
            arms.append(arm)
            _print_arm("fresh_prompt", block, slot, arm)
        baseline = [arms[0], arms[3]]
        candidate = [arms[1], arms[2]]
        for arm in candidate:
            _require_candidate_engaged(arm, "fresh-prompt")
        if any(item["tokens"] != baseline[0]["tokens"] for item in arms[1:]):
            raise AssertionError("candidate and interleaved baseline token traces differ")
        base_total = statistics.mean(item["total_ms"] for item in baseline)
        cand_total = statistics.mean(item["total_ms"] for item in candidate)
        base_decode = statistics.mean(item["decode_ms"] for item in baseline)
        cand_decode = statistics.mean(item["decode_ms"] for item in candidate)
        row = {
            "block": block,
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
            "arms": arms,
        }
        blocks.append(row)
        _print_block("fresh_prompt", row)
    return {
        "token_exact": True,
        "median_total_wall_speedup": statistics.median(
            row["total_wall_speedup"] for row in blocks
        ),
        "median_decode_speedup": statistics.median(
            row["decode_speedup"] for row in blocks
        ),
        "blocks": blocks,
    }


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
        "counter_delta": _counter_delta(
            counters_before, gdn_prefix_fanout_stats()
        ),
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

    blocks = []
    for block in range(args.cached_serving_reps):
        arms = []
        for slot, enabled in enumerate((False, True, True, False)):
            if args.cool_seconds:
                time.sleep(args.cool_seconds)
            arm = _cached_serving_once(model, cached, tail, args, enabled)
            arms.append(arm)
            _print_arm("cached_prefix", block, slot, arm)
        baseline = [arms[0], arms[3]]
        candidate = [arms[1], arms[2]]
        for arm in candidate:
            _require_candidate_engaged(arm, "cached-prefix")
        if any(item["tokens"] != baseline[0]["tokens"] for item in arms[1:]):
            raise AssertionError("cached-prefix candidate token traces differ")
        base_prepare = statistics.mean(item["prepare_ms"] for item in baseline)
        cand_prepare = statistics.mean(item["prepare_ms"] for item in candidate)
        base_decode = statistics.mean(item["decode_ms"] for item in baseline)
        cand_decode = statistics.mean(item["decode_ms"] for item in candidate)
        base_total = statistics.mean(item["total_ms"] for item in baseline)
        cand_total = statistics.mean(item["total_ms"] for item in candidate)
        row = {
            "block": block,
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
            "arms": arms,
        }
        blocks.append(row)
        _print_block("cached_prefix", row)
    return {
        "token_exact": True,
        "prefix_tokens": len(prefix),
        "tail_tokens": len(tail),
        "snapshot_build_ms": cached["build_ms"],
        "snapshot_build_tokens_per_second": cached["build_tokens_per_second"],
        "median_prepare_wall_speedup": statistics.median(
            row["prepare_wall_speedup"] for row in blocks
        ),
        "median_decode_wall_speedup": statistics.median(
            row["decode_wall_speedup"] for row in blocks
        ),
        "median_total_wall_speedup": statistics.median(
            row["total_wall_speedup"] for row in blocks
        ),
        "blocks": blocks,
    }


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
    parser.add_argument("--cool-seconds", type=float, default=30.0)
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
    if (
        args.component_reps < 1
        or args.serving_reps < 1
        or args.cached_serving_reps < 1
    ):
        parser.error("component, serving, and cached-serving reps must be positive")
    if args.cached_prefix_tokens < 0 or args.cached_tail_tokens < 1:
        parser.error("cached prefix must be non-negative and cached tail positive")
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
    serving = _serving_gate(model, prompt, args)
    cached_serving = _cached_serving_gate(model, prompt, args)
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
    if counters["hybrid_fanout_batches"] < 1:
        raise AssertionError("GDN prefix fan-out did not engage")
    expected_serving_engagements = (
        args.serving_reps + args.cached_serving_reps
    ) * 2
    if counters["serving_engaged"] != expected_serving_engagements:
        raise AssertionError(
            "serving fan-out engagement mismatch: "
            f"{counters['serving_engaged']} != {expected_serving_engagements}"
        )
    if counters["serving_declined_cache"] or counters["serving_declined_error"]:
        raise AssertionError("a serving candidate arm silently declined fan-out")
    if max(
        component["logits_relative_max"], component["cache_relative_max"]
    ) > args.max_relative_error:
        raise AssertionError(
            "full-model serial/fan-out exactness exceeded "
            f"{args.max_relative_error}"
        )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
