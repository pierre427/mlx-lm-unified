#!/usr/bin/env python3
"""Real multi-turn, multi-session qualification for guarded ANE head offload.

Each turn runs one independent target lane beside a real physical GPU batch.
The target role rotates across sessions.  Background reply budgets differ, so
rows leave the physical batch and the amount of useful ANE overlap changes.
Session text is read from private Codex JSONL traces and never written to the
receipt; only source digests, token counts, output token digests, and timings
are retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Keep the qualification path deterministic across MLX releases.  TF32 changes
# the GPU reference arm and can hide small active-ANE output differences.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx

from mlx_lm import load
from mlx_lm.ane_verifier import (
    ANEVerifierConfig,
    ANEVerifierStamp,
    ane_verifier_eligibility,
    load_ane_verifier,
)
from mlx_lm.apc import APCKey, AutomaticPrefixCache, MTPAPCSidecar
from mlx_lm.gdn_prefix_fanout import _clone_cache
from mlx_lm.hybrid_speculative import (
    _mtp_backbone,
    attach_segmented_self_mtp_lanes,
    close_segmented_self_mtp_state,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.segmented_self_mtp import segmented_self_mtp_stats

_INJECTED_PREFIXES = (
    "# agents.md",
    "<agents.md",
    "<app-context",
    "<collaboration_mode",
    "<environment_context",
    "<permissions instructions",
    "<plugins_instructions",
    "<recommended_plugins",
    "<skills_instructions",
    "<system-reminder",
)


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values):
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        "all": values,
    }


def jain(values):
    if not values or not any(values):
        return 0.0
    return sum(values) ** 2 / (len(values) * sum(value * value for value in values))


def host_snapshot():
    pressure = subprocess.run(
        ["/usr/bin/memory_pressure", "-Q"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    swap = subprocess.run(
        ["/usr/sbin/sysctl", "vm.swapusage"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    thermal = subprocess.run(
        ["/usr/bin/pmset", "-g", "therm"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    used = re.search(r"used\s*=\s*([0-9.]+)M", swap)
    return {
        "free_percent": int(free.group(1)) if free else None,
        "swap_used_mb": float(used.group(1)) if used else None,
        "thermal": thermal,
    }


def _message_text(payload, accepted):
    content = payload.get("content", [])
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n\n".join(
        str(item.get("text", "")).strip()
        for item in content
        if isinstance(item, dict)
        and item.get("type") == accepted
        and str(item.get("text", "")).strip()
    )


def load_codex_turns(path):
    """Return completed user/assistant turns, excluding injected instructions."""
    order = []
    rows = {}
    for line in Path(path).open(errors="strict"):
        item = json.loads(line)
        if item.get("type") != "response_item":
            continue
        payload = item.get("payload", {})
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in {"user", "assistant"}:
            continue
        metadata = payload.get("internal_chat_message_metadata_passthrough") or {}
        turn_id = metadata.get("turn_id")
        if not turn_id:
            continue
        if turn_id not in rows:
            rows[turn_id] = {"user": [], "assistant": []}
            order.append(turn_id)
        text = _message_text(payload, "input_text" if role == "user" else "output_text")
        if role == "user" and text.lstrip().casefold().startswith(_INJECTED_PREFIXES):
            continue
        if text:
            rows[turn_id][role].append(text)
    return [
        {
            "user": "\n\n".join(rows[key]["user"]),
            "assistant": "\n\n".join(rows[key]["assistant"]),
        }
        for key in order
        if rows[key]["user"] and rows[key]["assistant"]
    ]


def choose_turns(turns, count):
    if len(turns) < count:
        raise ValueError(f"session has {len(turns)} completed turns, needs {count}")
    if count == 1:
        return [turns[-1]]
    indices = [round(i * (len(turns) - 1) / (count - 1)) for i in range(count)]
    return [turns[index] for index in indices]


def session_prompts(tokenizer, path, turn_count, max_context):
    turns = load_codex_turns(path)
    selected = choose_turns(list(enumerate(turns)), turn_count)
    selected_indices = {index for index, _row in selected}
    running = []
    prompts = []
    for index, row in enumerate(turns):
        user = tokenizer.encode(f"\n\nUser: {row['user']}\n\nAssistant:")
        if index in selected_indices:
            prompts.append((running + user)[-max_context:])
        running = (running + user + tokenizer.encode(f" {row['assistant']}"))[
            -max_context:
        ]
    return prompts


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


def _prepare_lane(
    model,
    prompt,
    uid,
    *,
    prompt_cache=None,
    mtp_state=None,
    rng=None,
    prompt_boundary_out=None,
):
    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=1024,
        prompt_cache=prompt_cache,
        mtp_state=mtp_state,
        lane_rng=rng or LaneRNG(427 + uid),
        num_draft=2,
        sampling_temp=0.0,
        sampling_top_p=1.0,
        sampling_top_k=0,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=512,
        share_qsa_indices=True,
        prompt_boundary_out=prompt_boundary_out,
    )


def make_template(model, prompt, uid, apc_suffix_tokens):
    """Prepare one persistent-MTP lane through a real exact APC sidecar hit."""

    suffix = min(max(2, apc_suffix_tokens), max(2, len(prompt) - 2))
    prefix = list(prompt[:-suffix])
    if len(prefix) < 2:
        prefix = list(prompt[:2])
    boundary = {}
    seed = 427 + uid
    _prepared, _first = _prepare_lane(
        model,
        prefix,
        uid,
        rng=LaneRNG(seed),
        prompt_boundary_out=boundary,
    )
    if not boundary.get("committed_only"):
        raise RuntimeError("APC setup did not capture a committed MTP boundary")
    covered = int(boundary["covered_tokens"])
    apc = AutomaticPrefixCache(max_size=1, cow_branching=False)
    key = APCKey("ane-real-multisession", semantic_fingerprint=uid)
    sidecar = MTPAPCSidecar(
        boundary["mtp_state"],
        covered,
        rng_key=boundary.get("rng_key"),
        rng_draws=int(boundary.get("rng_draws", 0)),
    )
    apc.store(key, prefix[:covered], boundary["target_cache"], sidecar=sidecar)
    lookup = apc.lookup(key, prompt)
    if not lookup.hit or lookup.sidecar is None or lookup.cached_tokens != covered:
        raise RuntimeError(
            "real-session APC+MTP restore failed: "
            f"hit={lookup.hit} cached={lookup.cached_tokens} expected={covered}"
        )
    restored_rng = (
        LaneRNG.from_key(lookup.sidecar.rng_key, lookup.sidecar.rng_draws)
        if lookup.sidecar.rng_key is not None
        else LaneRNG(seed)
    )
    detached, first = _prepare_lane(
        model,
        lookup.remaining_tokens,
        uid,
        prompt_cache=lookup.cache,
        mtp_state=lookup.sidecar.state,
        rng=restored_rng,
    )
    mx.eval(first.logprobs)
    receipt = {
        "hit": True,
        "hit_kind": lookup.hit_kind,
        "cached_tokens": covered,
        "remaining_tokens": len(lookup.remaining_tokens),
        "topology": "checkpointed_hybrid",
    }
    apc.clear(release_memory=False)
    return detached, int(first.token), receipt


def native_head(model, hidden):
    started = time.perf_counter_ns()
    logits = model.logits(hidden[:, -1:, :])
    token = mx.argmax(logits[0, -1]).astype(mx.uint32)
    mx.eval(token)
    return int(token.item()), (time.perf_counter_ns() - started) / 1e6


def trunk(model, token, cache):
    started = time.perf_counter_ns()
    hidden, _ = _mtp_backbone(model, mx.array([[token]], mx.uint32), cache)
    mx.eval(hidden)
    return hidden[:, -1:, :], (time.perf_counter_ns() - started) / 1e6


def _segmented_delta(before, after):
    return {
        key: int(value) - int(before.get(key, 0))
        for key, value in after.items()
        if isinstance(value, int) and isinstance(before.get(key, 0), int)
    }


def run_turn(
    model,
    controller,
    prompts,
    budgets,
    target_index,
    arm,
    generation,
    config,
    apc_suffix_tokens,
):
    prep_started = time.perf_counter_ns()
    templates = [
        make_template(
            model,
            prompt,
            generation * 10 + i,
            apc_suffix_tokens,
        )
        for i, prompt in enumerate(prompts)
    ]
    target_cache = cache_clone(templates[target_index][0].caches.target)
    target_token = templates[target_index][1]
    background_indices = [i for i in range(len(prompts)) if i != target_index]
    initial_count = max(1, len(background_indices) - 1)
    background = attach_segmented_self_mtp_lanes(
        model,
        None,
        [templates[i][0] for i in background_indices[:initial_count]],
    )
    pending = [templates[i][0] for i in background_indices[initial_count:]]
    uid_to_index = {generation * 10 + index: index for index in background_indices}
    preparation_ms = (time.perf_counter_ns() - prep_started) / 1e6
    traces = [[] for _ in prompts]
    completion_ms = [[] for _ in prompts]
    target_trunk_ms = []
    target_head_ms = []
    background_step_ms = []
    resolve_wait_ms = []
    admission_reasons = []
    fallbacks = 0
    engaged = 0
    stale_fallbacks = 0
    join_receipts = []
    detach_receipts = []
    segmented_before = segmented_self_mtp_stats(reset=False)
    free_percent = host_snapshot()["free_percent"]
    memory_headroom_gib = max(0.0, free_percent or 0) * 128.0 / 100.0
    started = time.perf_counter_ns()
    for step in range(budgets[target_index]):
        if pending and step == 2:
            before_epoch = int(background.membership_epoch)
            joining_uids = [int(item.lane.uid) for item in pending]
            background = attach_segmented_self_mtp_lanes(model, background, pending)
            join_receipts.append(
                {
                    "step": step,
                    "uids": joining_uids,
                    "epoch_before": before_epoch,
                    "epoch_after": int(background.membership_epoch),
                }
            )
            pending = []
        hidden, elapsed = trunk(model, target_token, target_cache)
        target_trunk_ms.append(elapsed)
        remaining = budgets[target_index] - step
        predicted_overlap = (
            statistics.median(background_step_ms) if background_step_ms else 27.0
        )
        eligibility = ane_verifier_eligibility(
            config,
            greedy=True,
            has_logits_processors=False,
            needs_full_logprobs=False,
            package_resident=True,
            memory_headroom_gib=memory_headroom_gib,
            package_gib=controller.package_gib,
            predicted_gpu_marginal_delay_ms=0.84,
            available_overlap_ms=(predicted_overlap if background.lanes else 0.0),
            predicted_ane_interference_ms=0.18,
            expected_remaining_verifications=remaining,
            predicted_final_drain_ms=0.0,
        )
        admission_reasons.append(eligibility.reason)
        ticket = None
        stamp = None
        if arm == "ane" and eligibility.eligible and background.lanes:
            stamp = ANEVerifierStamp(
                membership_epoch=int(background.membership_epoch),
                lane_uids=(target_index,),
                generations=(generation + step,),
                verify_positions=(step,),
            )
            ticket = controller.submit(stamp, hidden)

        if background.lanes:
            background_started = time.perf_counter_ns()
            proposal = propose_batched_self_mtp(model, background)
            mx.eval(
                [output.logprobs for outputs in proposal.outputs for output in outputs]
            )
            background_step_ms.append(
                (time.perf_counter_ns() - background_started) / 1e6
            )
            now_ms = (time.perf_counter_ns() - started) / 1e6
            emitted_counts = []
            terminal = []
            for lane, outputs in zip(background.lanes, proposal.outputs):
                index = uid_to_index[int(lane.uid)]
                remaining_budget = budgets[index] - len(traces[index])
                delivered = outputs[:remaining_budget]
                traces[index].extend(int(output.token) for output in delivered)
                completion_ms[index].extend([now_ms] * len(delivered))
                emitted_counts.append(len(delivered))
                terminal.append(len(traces[index]) >= budgets[index])
            commit_batched_self_mtp(
                background,
                proposal,
                emitted_counts=emitted_counts,
                terminal=terminal,
            )
            leaving = [i for i, done in enumerate(terminal) if done]
            if leaving:
                before_epoch = int(background.membership_epoch)
                leaving_uids = [int(background.lanes[i].uid) for i in leaving]
                background, detached = detach_self_mtp_lanes(model, background, leaving)
                detach_receipts.append(
                    {
                        "step": step,
                        "uids": leaving_uids,
                        "epoch_before": before_epoch,
                        "epoch_after": int(background.membership_epoch),
                        "detached": len(detached),
                    }
                )

        if ticket is not None:
            wait_started = time.perf_counter_ns()
            current_stamp = ANEVerifierStamp(
                membership_epoch=int(background.membership_epoch),
                lane_uids=(target_index,),
                generations=(generation + step,),
                verify_positions=(step,),
            )
            result = controller.resolve(ticket, current_stamp)
            resolve_wait_ms.append((time.perf_counter_ns() - wait_started) / 1e6)
            stale_fallbacks += int(result is None and current_stamp != stamp)
        else:
            result = None
        if result is None:
            target_token, head_ms = native_head(model, hidden)
            target_head_ms.append(head_ms)
            if arm == "ane" and ticket is not None:
                fallbacks += 1
        else:
            target_token = result.token_ids[0]
            engaged += 1
        traces[target_index].append(target_token)
        completion_ms[target_index].append((time.perf_counter_ns() - started) / 1e6)
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    if background.lanes:
        close_segmented_self_mtp_state(background)
    segmented_after = segmented_self_mtp_stats(reset=False)
    total_tokens = sum(map(len, traces))
    per_session_tps = []
    for times in completion_ms:
        if not times:
            per_session_tps.append(0.0)
        else:
            per_session_tps.append(len(times) / (max(times[-1], 1e-9) / 1000.0))
    return {
        "arm": arm,
        "target_index": target_index,
        "prompt_tokens": list(map(len, prompts)),
        "reply_budgets": list(budgets),
        "preparation_ms": preparation_ms,
        "wall_ms": wall_ms,
        "total_tokens": total_tokens,
        "throughput_tps": total_tokens / (wall_ms / 1000.0),
        "tokens": traces,
        "token_digests": [
            hashlib.sha256(",".join(map(str, row)).encode()).hexdigest()
            for row in traces
        ],
        "completion_ms": completion_ms,
        "per_session_tps": per_session_tps,
        "jain_fairness": jain(per_session_tps),
        "target_trunk_ms": target_trunk_ms,
        "target_head_ms": target_head_ms,
        "background_step_ms": background_step_ms,
        "resolve_wait_ms": resolve_wait_ms,
        "ane_engaged": engaged,
        "ane_fallbacks": fallbacks,
        "ane_stale_fallbacks": stale_fallbacks,
        "apc": [template[2] for template in templates],
        "joins": join_receipts,
        "detaches": detach_receipts,
        "segmented_delta": _segmented_delta(segmented_before, segmented_after),
        "admission_reasons": dict(
            (reason, admission_reasons.count(reason))
            for reason in sorted(set(admission_reasons))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--target-tokens", type=int, default=32)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-margin", type=float, default=0.125)
    parser.add_argument("--apc-suffix-tokens", type=int, default=64)
    parser.add_argument("--thermal-helper", type=Path)
    parser.add_argument("--thermal-poll-s", type=float, default=10.0)
    parser.add_argument("--thermal-consecutive", type=int, default=2)
    parser.add_argument("--thermal-min-s", type=float, default=20.0)
    parser.add_argument("--thermal-max-s", type=float, default=120.0)
    parser.add_argument("--thermal-baseline-max-s", type=float, default=90.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.session) < 3:
        parser.error("at least three independent sessions are required")
    if args.turns < 2 or args.target_tokens < 8 or args.samples < 1:
        parser.error("turns >= 2, target-tokens >= 8, and samples >= 1 are required")
    if args.apc_suffix_tokens < 2:
        parser.error("apc-suffix-tokens must be at least 2")

    os.environ.setdefault("MLX_LM_SEGMENTED_SELF_MTP", "1")
    os.environ.setdefault("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    os.environ.setdefault("MLX_LM_QSA_PRIVATE_DELTA", "1")
    os.environ.setdefault("MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", "1")

    sidecar = args.model / "ple_rows.bin"
    if sidecar.is_file():
        os.environ.setdefault("MLX_QWEN4_PLE_NVME", str(sidecar))
        os.environ.setdefault("MLX_QWEN4_PLE_NVME_LRU_MB", "256")
        os.environ.setdefault("MLX_LM_UBC_EVICT", "1")

    before = host_snapshot()
    model, tokenizer = load(str(args.model))
    model.eval()
    sessions = [
        session_prompts(tokenizer, path, args.turns, args.max_context)
        for path in args.session
    ]
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

    thermal_module = None
    settle = None
    baseline = None
    thermal_settles = []
    if args.thermal_helper:
        sys.path.insert(0, str(args.thermal_helper.parent))
        thermal_module = __import__(args.thermal_helper.stem)
        settle = thermal_module.settle

    # A short warmup initializes the model, Core ML packages, and cache classes.
    warm_prompts = [session[0] for session in sessions]
    run_turn(
        model,
        controller,
        warm_prompts,
        [8] * len(sessions),
        0,
        "ane",
        1,
        config,
        args.apc_suffix_tokens,
    )
    gc.collect()
    mx.clear_cache()
    active_memory_baseline = int(mx.get_active_memory())
    # Establish the thermal reference only after model weights, caches, and all
    # Core ML packages have been materialized.  A pre-warmup reference can make
    # a stable loaded system look artificially slow.
    if thermal_module is not None:
        baseline = thermal_module.calibrate_baseline(
            poll_s=args.thermal_poll_s,
            consecutive=args.thermal_consecutive,
            max_s=args.thermal_baseline_max_s,
        )

    samples = []
    for sample in range(args.samples):
        if settle is not None:
            thermal_settles.append(
                settle(
                    baseline["tflops"],
                    poll_s=args.thermal_poll_s,
                    consecutive=args.thermal_consecutive,
                    min_s=args.thermal_min_s,
                    max_s=args.thermal_max_s,
                )
            )
        order = ("gpu", "ane") if sample % 2 == 0 else ("ane", "gpu")
        arms = {arm: [] for arm in order}
        for turn_index in range(args.turns):
            prompts = [session[turn_index] for session in sessions]
            target = turn_index % len(sessions)
            budgets = [
                (
                    args.target_tokens
                    if index == target
                    else max(8, args.target_tokens - 8 * (1 + index % 3))
                )
                for index in range(len(sessions))
            ]
            for arm in order:
                row = run_turn(
                    model,
                    controller,
                    prompts,
                    budgets,
                    target,
                    arm,
                    1000 + sample * 100 + turn_index,
                    config,
                    args.apc_suffix_tokens,
                )
                gc.collect()
                mx.clear_cache()
                row["active_memory_after_cleanup_bytes"] = int(mx.get_active_memory())
                arms[arm].append(row)
        samples.append({"order": order, "arms": arms})

    inflight_before_close = len(controller._tickets)
    controller.close()
    worker_alive_after_close = controller.runner.is_alive
    gc.collect()
    mx.clear_cache()
    active_memory_after_close = int(mx.get_active_memory())
    gpu_walls = [
        sum(turn["wall_ms"] for turn in sample["arms"]["gpu"]) for sample in samples
    ]
    ane_walls = [
        sum(turn["wall_ms"] for turn in sample["arms"]["ane"]) for sample in samples
    ]
    exact = []
    for sample in samples:
        exact.append(
            all(
                gpu["tokens"] == ane["tokens"]
                for gpu, ane in zip(sample["arms"]["gpu"], sample["arms"]["ane"])
            )
        )
    gpu_session_rates = [
        value
        for sample in samples
        for turn in sample["arms"]["gpu"]
        for value in turn["per_session_tps"]
    ]
    ane_session_rates = [
        value
        for sample in samples
        for turn in sample["arms"]["ane"]
        for value in turn["per_session_tps"]
    ]
    report = {
        "schema": "mlx-lm.qwen4-ane-verifier-multisession.v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "model": str(args.model.resolve()),
        "sources": [
            {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "completed_turns": len(load_codex_turns(path)),
            }
            for path in args.session
        ],
        "geometry": {
            "sessions": len(sessions),
            "turns_per_session": args.turns,
            "max_context": args.max_context,
            "target_tokens": args.target_tokens,
            "samples": args.samples,
        },
        "method": {
            "topology": "one rotating B1 target beside a physical GPU batch of the other ready sessions",
            "churn": "background reply budgets differ, causing rows to leave during each turn",
            "privacy": "private session text is read locally and omitted from the receipt",
            "ordering": "counterbalanced GPU/ANE arms",
            "routing": "marginal critical-path eligibility; no batch-count threshold",
            "ane_config": asdict(config),
            "feature_switches": {
                name: os.environ.get(name)
                for name in (
                    "MLX_LM_SEGMENTED_SELF_MTP",
                    "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP",
                    "MLX_LM_QSA_PRIVATE_DELTA",
                    "MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD",
                )
            },
        },
        "thermal": {
            "baseline": (
                None
                if baseline is None
                else {key: baseline.get(key) for key in ("tflops", "stable")}
            ),
            "settles": thermal_settles,
            "before": before,
            "after": host_snapshot(),
        },
        "summary": {
            "gpu_decode_wall_ms": summarize(gpu_walls),
            "ane_decode_wall_ms": summarize(ane_walls),
            "throughput_ratio_gpu_over_ane_wall": statistics.median(gpu_walls)
            / statistics.median(ane_walls),
            "gpu_session_tps": summarize(gpu_session_rates),
            "ane_session_tps": summarize(ane_session_rates),
            "gpu_jain_fairness": statistics.median(
                turn["jain_fairness"]
                for sample in samples
                for turn in sample["arms"]["gpu"]
            ),
            "ane_jain_fairness": statistics.median(
                turn["jain_fairness"]
                for sample in samples
                for turn in sample["arms"]["ane"]
            ),
            "exact_blocks": sum(exact),
            "total_blocks": len(exact),
            "thermally_qualified_blocks": sum(
                receipt["settled"] for receipt in thermal_settles
            ),
            "ane_engaged": sum(
                turn["ane_engaged"]
                for sample in samples
                for turn in sample["arms"]["ane"]
            ),
            "ane_fallbacks": sum(
                turn["ane_fallbacks"]
                for sample in samples
                for turn in sample["arms"]["ane"]
            ),
            "ane_stale_fallbacks": sum(
                turn["ane_stale_fallbacks"]
                for sample in samples
                for turn in sample["arms"]["ane"]
            ),
            "apc_hits": sum(
                receipt["hit"]
                for sample in samples
                for turns in sample["arms"].values()
                for turn in turns
                for receipt in turn["apc"]
            ),
            "segmented_true_batched_engagements": sum(
                turn["segmented_delta"].get("true_batched_engaged", 0)
                for sample in samples
                for turns in sample["arms"].values()
                for turn in turns
            ),
            "branch_transactions": sum(
                turn["segmented_delta"].get("transaction_branches", 0)
                for sample in samples
                for turns in sample["arms"].values()
                for turn in turns
            ),
            "branch_promotions": sum(
                turn["segmented_delta"].get("transaction_promotions", 0)
                for sample in samples
                for turns in sample["arms"].values()
                for turn in turns
            ),
            "branch_rejections": sum(
                turn["segmented_delta"].get("transaction_rejections", 0)
                for sample in samples
                for turns in sample["arms"].values()
                for turn in turns
            ),
            "ane_inflight_before_close": inflight_before_close,
            "ane_worker_alive_after_close": worker_alive_after_close,
            "active_memory_baseline_bytes": active_memory_baseline,
            "active_memory_after_close_bytes": active_memory_after_close,
            "active_memory_growth_bytes": (
                active_memory_after_close - active_memory_baseline
            ),
        },
        "controller_stats": controller.stats.snapshot(),
        "samples": samples,
    }
    measured_turns = [
        turn
        for sample in samples
        for turns in sample["arms"].values()
        for turn in turns
    ]
    report["qualification"] = {
        "exact": sum(exact) == len(exact),
        "all_apc_hits": all(
            receipt["hit"] for turn in measured_turns for receipt in turn["apc"]
        ),
        "segmented_true_batch_engaged": all(
            turn["segmented_delta"].get("true_batched_engaged", 0) > 0
            for turn in measured_turns
        ),
        "branch_ownership_closed": all(
            turn["segmented_delta"].get("transaction_branches", 0)
            == turn["segmented_delta"].get("transaction_promotions", 0)
            + turn["segmented_delta"].get("transaction_rejections", 0)
            for turn in measured_turns
        ),
        "membership_churn_observed": all(
            turn["joins"] and turn["detaches"] for turn in measured_turns
        ),
        "no_segmented_failures": all(
            turn["segmented_delta"].get("failures", 0) == 0 for turn in measured_turns
        ),
        "no_ane_ticket_leak": inflight_before_close == 0,
        "ane_worker_closed": not worker_alive_after_close,
        # MLX's allocator counters can retain a few host bookkeeping bytes.
        # One MiB is far below one cache plane or ANE package and avoids
        # treating that accounting noise as a device-memory ownership leak.
        "no_active_memory_growth": (
            active_memory_after_close <= active_memory_baseline + (1 << 20)
        ),
    }
    report["qualification"]["qualified"] = all(report["qualification"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
