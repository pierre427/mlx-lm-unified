#!/usr/bin/env python3
"""Qualify ANE head offload in a dependency-correct two-request pipeline.

Unlike a width sweep, this benchmark measures the marginal GPU head cost and
the overlap available while the next independent request executes its real
Qwen4 trunk.  Both arms start from cloned production cache state.  The ANE arm
falls back to the native q4 GPU head below its configured confidence margin.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm import load
from mlx_lm.ane_verifier import (
    ANEVerifierConfig,
    ANEVerifierStamp,
    load_ane_verifier,
)
from mlx_lm.gdn_prefix_fanout import _clone_cache
from mlx_lm.hybrid_speculative import _mtp_backbone, prepare_self_mtp_lane
from mlx_lm.sample_utils import LaneRNG


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values):
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "all": values,
    }


def thermal_snapshot():
    return subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, text=True, check=False
    ).stdout.strip()


def cache_clone(cache):
    result = [_clone_cache(entry) for entry in cache]
    arrays = []
    for entry in result:
        state = entry.state
        if isinstance(state, mx.array):
            arrays.append(state)
        elif isinstance(state, (list, tuple)):
            arrays.extend(value for value in state if isinstance(value, mx.array))
    if arrays:
        mx.eval(*arrays)
    return result


def prompt_tokens(tokenizer, text, length):
    seed = tokenizer.encode(text)
    if not seed:
        raise ValueError("prompt tokenization returned no tokens")
    return mx.array(
        (seed * ((length + len(seed) - 1) // len(seed)))[:length], mx.uint32
    )


def make_template(model, prompt):
    detached, first = prepare_self_mtp_lane(
        prompt,
        model,
        uid=0,
        max_tokens=1024,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(427),
        num_draft=2,
        sampling_temp=0.0,
        sampling_top_p=1.0,
        sampling_top_k=0,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=512,
        share_qsa_indices=True,
    )
    mx.eval(first.logprobs)
    return detached.caches.target, int(first.token)


def native_head(model, hidden):
    started = time.perf_counter_ns()
    logits = model.logits(hidden[:, -1:, :])
    token = mx.argmax(logits[0, -1]).astype(mx.uint32)
    mx.eval(token)
    return int(token.item()), (time.perf_counter_ns() - started) / 1e6


def trunk(model, token, cache):
    started = time.perf_counter_ns()
    hidden, _seed = _mtp_backbone(model, mx.array([[token]], mx.uint32), cache)
    mx.eval(hidden)
    return hidden[:, -1:, :], (time.perf_counter_ns() - started) / 1e6


def gpu_arm(model, template_cache, first_token, steps):
    caches = [cache_clone(template_cache), cache_clone(template_cache)]
    tokens = [first_token, first_token]
    traces = [[], []]
    trunk_ms, head_ms, request_ms = [], [], []
    started = time.perf_counter_ns()
    for index in range(steps * 2):
        lane = index % 2
        request_started = time.perf_counter_ns()
        hidden, elapsed = trunk(model, tokens[lane], caches[lane])
        token, head_elapsed = native_head(model, hidden)
        tokens[lane] = token
        traces[lane].append(token)
        trunk_ms.append(elapsed)
        head_ms.append(head_elapsed)
        request_ms.append((time.perf_counter_ns() - request_started) / 1e6)
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    return {
        "tokens": traces,
        "wall_ms": wall_ms,
        "throughput_tps": (steps * 2) / (wall_ms / 1000),
        "trunk_ms": trunk_ms,
        "head_ms": head_ms,
        "request_ms": request_ms,
    }


def ane_arm(model, controller, template_cache, first_token, steps, generation):
    caches = [cache_clone(template_cache), cache_clone(template_cache)]
    tokens = [first_token, first_token]
    traces = [[], []]
    raw_ane = [[], []]
    trunk_ms, resolve_wait_ms, request_ms = [], [], []
    fallbacks = 0
    pending = None
    pending_lane = None
    pending_hidden = None
    pending_started = None
    started = time.perf_counter_ns()
    for index in range(steps * 2):
        lane = index % 2
        request_started = time.perf_counter_ns()
        hidden, elapsed = trunk(model, tokens[lane], caches[lane])
        trunk_ms.append(elapsed)

        if pending is not None:
            resolve_started = time.perf_counter_ns()
            result = controller.resolve(pending, pending.stamp)
            resolve_wait_ms.append((time.perf_counter_ns() - resolve_started) / 1e6)
            if result is None:
                chosen, _ = native_head(model, pending_hidden)
                fallbacks += 1
            else:
                chosen = result.token_ids[0]
                raw_ane[pending_lane].append(chosen)
            tokens[pending_lane] = chosen
            traces[pending_lane].append(chosen)
            request_ms.append((time.perf_counter_ns() - pending_started) / 1e6)

        stamp = ANEVerifierStamp(
            membership_epoch=generation,
            lane_uids=(lane,),
            generations=(generation + index,),
            verify_positions=(len(traces[lane]),),
        )
        pending = controller.submit(stamp, hidden)
        pending_lane = lane
        pending_hidden = hidden
        pending_started = request_started

    resolve_started = time.perf_counter_ns()
    result = controller.resolve(pending, pending.stamp)
    resolve_wait_ms.append((time.perf_counter_ns() - resolve_started) / 1e6)
    if result is None:
        chosen, _ = native_head(model, pending_hidden)
        fallbacks += 1
    else:
        chosen = result.token_ids[0]
        raw_ane[pending_lane].append(chosen)
    traces[pending_lane].append(chosen)
    request_ms.append((time.perf_counter_ns() - pending_started) / 1e6)

    wall_ms = (time.perf_counter_ns() - started) / 1e6
    return {
        "tokens": traces,
        "raw_ane_tokens": raw_ane,
        "wall_ms": wall_ms,
        "throughput_tps": (steps * 2) / (wall_ms / 1000),
        "trunk_ms": trunk_ms,
        "resolve_wait_ms": resolve_wait_ms,
        "request_ms": request_ms,
        "fallbacks": fallbacks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Explain why verification state commits at delivery."
    )
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--min-margin", type=float, default=0.125)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.samples < 2 or args.prompt_tokens < 2:
        parser.error("steps, samples, and prompt tokens must be at least two")

    sidecar = args.model / "ple_rows.bin"
    if sidecar.is_file():
        os.environ.setdefault("MLX_QWEN4_PLE_NVME", str(sidecar))
        os.environ.setdefault("MLX_QWEN4_PLE_NVME_LRU_MB", "256")
        os.environ.setdefault("MLX_LM_UBC_EVICT", "1")

    before = thermal_snapshot()
    model, tokenizer = load(str(args.model))
    model.eval()
    prompt = prompt_tokens(tokenizer, args.prompt, args.prompt_tokens)
    template_cache, first_token = make_template(model, prompt)
    config = ANEVerifierConfig(
        mode="active",
        deadline_ms=25.0,
        min_margin=args.min_margin,
        max_inflight=2,
        allow_approximate_commit=True,
    )
    controller = load_ane_verifier(
        args.package_dir, config=config, model_dir=args.model
    )

    # Warm both paths without retaining the warm-up measurements.
    gpu_arm(model, template_cache, first_token, 2)
    ane_arm(model, controller, template_cache, first_token, 2, 1)

    samples = []
    for sample in range(args.samples):
        order = ("gpu", "ane") if sample % 2 == 0 else ("ane", "gpu")
        arms = {}
        for arm in order:
            if arm == "gpu":
                arms[arm] = gpu_arm(model, template_cache, first_token, args.steps)
            else:
                arms[arm] = ane_arm(
                    model,
                    controller,
                    template_cache,
                    first_token,
                    args.steps,
                    1000 + sample * 100,
                )
        samples.append({"order": order, "arms": arms})

    controller.close()
    gpu_walls = [row["arms"]["gpu"]["wall_ms"] for row in samples]
    ane_walls = [row["arms"]["ane"]["wall_ms"] for row in samples]
    gpu_heads = [value for row in samples for value in row["arms"]["gpu"]["head_ms"]]
    gpu_trunks = [value for row in samples for value in row["arms"]["gpu"]["trunk_ms"]]
    overlap_trunks = [
        value for row in samples for value in row["arms"]["ane"]["trunk_ms"]
    ]
    waits = [
        value for row in samples for value in row["arms"]["ane"]["resolve_wait_ms"]
    ]
    requests_gpu = [
        value for row in samples for value in row["arms"]["gpu"]["request_ms"]
    ]
    requests_ane = [
        value for row in samples for value in row["arms"]["ane"]["request_ms"]
    ]
    exact = [
        row["arms"]["gpu"]["tokens"] == row["arms"]["ane"]["tokens"] for row in samples
    ]
    report = {
        "schema": "mlx-lm.qwen4-ane-verifier-pipeline.v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "model": str(args.model.resolve()),
        "geometry": {
            "requests": 2,
            "steps_per_request": args.steps,
            "prompt_tokens": args.prompt_tokens,
            "samples": args.samples,
        },
        "method": {
            "schedule": "round-robin; ANE head for request A overlaps the real GPU trunk for request B",
            "control": "same prompt boundary cloned independently for each arm",
            "ordering": "alternating AB/BA blocks",
            "gpu_cost_axis": "measured marginal q4 head service and observed overlap interference, not batch-size extrapolation",
            "ane_config": asdict(config),
        },
        "thermal": {"before": before, "after": thermal_snapshot()},
        "summary": {
            "gpu_wall_ms": summarize(gpu_walls),
            "ane_wall_ms": summarize(ane_walls),
            "throughput_ratio_gpu_over_ane_wall": statistics.median(gpu_walls)
            / statistics.median(ane_walls),
            "gpu_head_marginal_ms": summarize(gpu_heads),
            "gpu_trunk_idle_ms": summarize(gpu_trunks),
            "gpu_trunk_during_ane_ms": summarize(overlap_trunks),
            "ane_resolve_wait_ms": summarize(waits),
            "gpu_request_latency_ms": summarize(requests_gpu),
            "ane_request_latency_ms": summarize(requests_ane),
            "exact_blocks": sum(exact),
            "total_blocks": len(exact),
            "ane_fallbacks": sum(row["arms"]["ane"]["fallbacks"] for row in samples),
        },
        "controller_stats": controller.stats.snapshot(),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
