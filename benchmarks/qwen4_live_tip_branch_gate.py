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
from dataclasses import asdict
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import threading
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
BRANCH_MODES = ("physical", "physical_fanout", "segmented")


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
            "async_promote_after_first": getattr(
                args, "async_promote_after_first", False
            ),
            "async_qsa_promote_after_first": getattr(
                args, "async_qsa_promote_after_first", False
            ),
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


def _cache_geometry(batch: Any) -> dict[str, Any]:
    """Host-only structural fingerprint of the steady physical cache."""

    def shape(value):
        return None if value is None else list(value.shape)

    result = {}
    for name, caches in (
        ("target", batch.caches.target),
        ("draft", batch.caches.draft),
    ):
        layers = []
        for cache in caches:
            layers.append(
                {
                    "type": type(cache).__name__,
                    "nbytes": int(cache.nbytes),
                    "idx": getattr(cache, "_idx", None),
                    "offset_shape": shape(getattr(cache, "offset", None)),
                    "left_padding_shape": shape(
                        getattr(cache, "left_padding", None)
                    ),
                    "keys_shape": shape(getattr(cache, "keys", None)),
                    "values_shape": shape(getattr(cache, "values", None)),
                    "index_keys_shape": shape(
                        getattr(cache, "index_keys", None)
                    ),
                    "pooled_keys_shape": shape(
                        getattr(cache, "_qsa_pooled_keys", None)
                    ),
                    "pooled_ratio": getattr(cache, "_qsa_pooled_ratio", None),
                    "summary_identity": getattr(
                        cache, "_qsa_summary_identity", None
                    ),
                    "right_padding_shape": shape(
                        getattr(cache, "_right_padding", None)
                    ),
                    "mtp_share_topk": bool(
                        getattr(cache, "_mtp_share_topk", False)
                    ),
                    "shared_topk_shape": shape(
                        getattr(cache, "_mtp_shared_topk", None)
                    ),
                    "state_shapes": [
                        shape(value)
                        for value in getattr(cache, "cache", ())
                    ],
                    "checkpoint_lanes": len(
                        getattr(cache, "_checkpoints", ())
                    ),
                    "rollback_records": len(
                        getattr(cache, "_rollbacks", ())
                    ),
                    "rollback_window": getattr(
                        cache, "_rollback_window", None
                    ),
                    "rollback_invalid_reason": getattr(
                        cache, "_rollback_invalid_reason", None
                    ),
                    "speculating": bool(
                        getattr(cache, "speculating", False)
                    ),
                }
            )
        result[name] = layers
    return result


def _ple_stats(model: Any) -> dict[str, float | int]:
    """Aggregate host-only counters from all file-backed PLE tables."""

    totals = {
        "lookups": 0,
        "rows": 0,
        "unique_rows": 0,
        "bytes_read": 0,
        "elapsed_seconds": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_evictions": 0,
    }
    layers = list(getattr(model, "layers", ()))
    layers.extend(getattr(getattr(model, "mtp", None), "layers", ()))
    for layer in layers:
        ple = getattr(layer, "ple", None)
        embedding = getattr(
            getattr(ple, "ple_embedding", None), "ngram_embedding", None
        )
        stats = getattr(embedding, "stats", None)
        if stats is None:
            continue
        for key in totals:
            totals[key] += getattr(stats, key)
    return totals


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


def _run_physical_first_cycle(
    mx: Any,
    model: Any,
    rows: list[Any],
    stream: Any,
    attach: Any,
    propose: Any,
    commit: Any,
    result: dict[str, Any],
) -> None:
    """Build and advance an independent physical B2 on one unsafe stream."""

    started = time.perf_counter_ns()
    try:
        with mx.stream(stream):
            batch = attach(model, None, rows)
            mx.eval(_batch_state_values(batch))
            ready = time.perf_counter_ns()
            proposal = propose(model, batch)
            mx.eval(
                [output.logprobs for outputs in proposal.outputs for output in outputs]
            )
            emitted = [len(outputs) for outputs in proposal.outputs]
            commit(
                batch,
                proposal,
                emitted_counts=emitted,
                terminal=[False] * len(emitted),
            )
            mx.eval(_batch_state_values(batch))
            mx.synchronize(stream)
        finished = time.perf_counter_ns()
        result.update(
            batch=batch,
            emitted=emitted,
            logprobs=[
                output.logprobs for outputs in proposal.outputs for output in outputs
            ],
            tokens=[
                [int(output.token) for output in outputs]
                for outputs in proposal.outputs
            ],
            ready_ms=(ready - started) / 1e6,
            total_ms=(finished - started) / 1e6,
        )
    except BaseException as error:
        result["error"] = error


def _run_arm(model: Any, prompt: Any, args: argparse.Namespace, arm: str) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.hybrid_speculative import (
        DetachedSelfMTPLane,
        SelfMTPCachePair,
        attach_prebatched_self_mtp_lanes,
        attach_self_mtp_lanes,
        attach_segmented_self_mtp_lanes,
        commit_batched_self_mtp,
        detach_self_mtp_lanes,
        _self_mtp_group_offset,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )
    from mlx_lm.sample_utils import LaneRNG
    from mlx_lm.apc import (
        APCKey,
        AutomaticPrefixCache,
        AutomaticPrefixCacheV2,
        MTPAPCSidecar,
    )
    from mlx_lm.gdn_prefix_fanout import (
        HybridCachePrefixFanout,
        gdn_prefix_fanout_stats,
    )
    from mlx_lm.segmented_self_mtp import (
        note_segmented_self_mtp,
        segmented_self_mtp_stats,
    )

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
    apc_mode = str(getattr(args, "apc_mode", "none"))
    if apc_mode not in {"none", "legacy", "apcv2"}:
        raise ValueError("apc-mode must be none, legacy, or apcv2")
    os.environ["MLX_LM_MTP_BOUNDARY_COW"] = (
        "1" if apc_mode == "apcv2" else "0"
    )
    prompt_boundary = {} if apc_mode != "none" else None
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
        prompt_boundary_out=prompt_boundary,
    )
    apc_receipt = None
    apc = None
    if apc_mode != "none":
        if not prompt_boundary or not prompt_boundary.get("committed_only"):
            raise RuntimeError("APC benchmark did not capture a committed boundary")
        covered = int(prompt_boundary["covered_tokens"])
        key = APCKey(
            "qwen4-live-tip-factorial",
            cache_layout_fingerprint=(
                "qwen4-exp-layer-segments-v1" if apc_mode == "apcv2" else "legacy"
            ),
        )
        apc = (
            AutomaticPrefixCacheV2(
                max_size=4, layout_name="qwen4-exp-layer-segments-v1"
            )
            if apc_mode == "apcv2"
            else AutomaticPrefixCache(max_size=4, cow_branching=False)
        )
        sidecar = MTPAPCSidecar(
            prompt_boundary["mtp_state"],
            covered,
            rng_key=prompt_boundary.get("rng_key"),
            rng_draws=int(prompt_boundary.get("rng_draws", 0)),
        )
        apc.store(
            key,
            [int(token) for token in prompt[:covered].tolist()],
            prompt_boundary["target_cache"],
            sidecar=sidecar,
        )
        lookup = apc.lookup(key, [int(token) for token in prompt.tolist()])
        if not lookup.hit or lookup.cached_tokens != covered or lookup.sidecar is None:
            raise RuntimeError(
                "APC factorial requires a real exact MTP sidecar hit: "
                f"hit={lookup.hit} cached={lookup.cached_tokens} expected={covered}"
            )
        restored_rng = (
            LaneRNG.from_key(lookup.sidecar.rng_key, lookup.sidecar.rng_draws)
            if lookup.sidecar.rng_key is not None
            else LaneRNG(args.seed)
        )
        detached = first = None
        gc.collect()
        detached, first = prepare_self_mtp_lane(
            mx.array(lookup.remaining_tokens, dtype=mx.uint32),
            model,
            uid=0,
            max_tokens=max_tokens,
            prompt_cache=lookup.cache,
            mtp_state=lookup.sidecar.state,
            lane_rng=restored_rng,
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
        apc_receipt = {
            "mode": apc_mode,
            "hit": True,
            "hit_kind": lookup.hit_kind,
            "cached_tokens": int(lookup.cached_tokens),
            "remaining_tokens": len(lookup.remaining_tokens),
            "snapshot_mode": prompt_boundary["snapshot_mode"],
            "lookup_segments": lookup.segment_manifest,
            "stats": apc.apc_stats,
        }
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

    fanout_before = gdn_prefix_fanout_stats()
    fanout_owner = None
    fanout_capture_ms = 0.0
    if args.branch_mode == "physical_fanout":
        capture_started = time.perf_counter_ns()
        fanout_owner = HybridCachePrefixFanout.from_prompt_cache(
            batch.caches.target,
            enabled=True,
            strict=True,
        )
        fanout_capture_ms = (time.perf_counter_ns() - capture_started) / 1e6
        if fanout_owner is None:
            raise RuntimeError("live-tip fanout refused the Qwen4 hybrid cache")

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

    async_promote = bool(getattr(args, "async_promote_after_first", False))
    async_qsa_promote = bool(
        getattr(args, "async_qsa_promote_after_first", False)
    )
    async_qsa_prequeue = bool(getattr(args, "async_qsa_prequeue", False))
    if async_qsa_prequeue and not async_qsa_promote:
        raise ValueError("QSA prequeue requires asynchronous QSA promotion")
    if async_qsa_prequeue and args.branch_mode != "segmented":
        raise ValueError("QSA prequeue requires a segmented branch")
    staged_qsa = None
    prequeue_started_ns = None
    async_qsa_queue_ms = 0.0
    async_qsa_prequeue_lead_ms = 0.0
    async_qsa_prequeue_breakdown = None
    if async_qsa_prequeue:
        from mlx_lm.segmented_physical_promotion import (
            begin_shared_prefix_physical_promotion,
        )

        prequeue_started_ns = time.perf_counter_ns()
        staged_qsa = begin_shared_prefix_physical_promotion(
            canonical,
            rows=args.branches,
            reserve_tail=args.num_draft + 1,
            stream=mx.new_stream(mx.gpu),
            diagnostic_timing=True,
        )
        async_qsa_queue_ms = (
            time.perf_counter_ns() - prequeue_started_ns
        ) / 1e6
        lead = float(getattr(args, "async_qsa_prequeue_lead_ms", 0.0))
        if lead > 0:
            time.sleep(lead / 1e3)
        async_qsa_prequeue_lead_ms = (
            time.perf_counter_ns() - prequeue_started_ns
        ) / 1e6
        async_qsa_prequeue_breakdown = {
            "validation_ms": staged_qsa.validation_ns / 1e6,
            "allocation_graph_ms": staged_qsa.allocation_graph_ns / 1e6,
            "copy_graph_ms": staged_qsa.copy_graph_ns / 1e6,
            "metadata_graph_ms": staged_qsa.metadata_graph_ns / 1e6,
            "reserve_total_ms": staged_qsa.reserve_total_ns / 1e6,
            "submit_ms": staged_qsa.submit_ns / 1e6,
            "other_ms": max(
                0.0,
                async_qsa_queue_ms
                - staged_qsa.reserve_total_ns / 1e6
                - staged_qsa.submit_ns / 1e6,
            ),
        }

    idle_wait_ns = 0
    if arm == "idle_live_tip" and args.idle_seconds:
        idle_started = time.perf_counter_ns()
        time.sleep(args.idle_seconds)
        idle_wait_ns = time.perf_counter_ns() - idle_started

    # This is the branch boundary. The live-tip-to-commit metric also includes
    # the required detach/canonicalization immediately before this point.
    branch_started = time.perf_counter_ns()
    ple_before = _ple_stats(model)
    sibling = (
        DetachedSelfMTPLane(
            lane=copy.deepcopy(canonical.lane),
            caches=canonical.caches,
        )
        if args.branch_mode == "physical_fanout"
        else copy.deepcopy(canonical)
    )
    sibling.lane.uid = 1
    sibling.lane.rng = LaneRNG(args.seed + 1)
    if async_promote and async_qsa_promote:
        raise ValueError("select only one asynchronous promotion strategy")
    if async_promote and args.branch_mode != "segmented":
        raise ValueError("asynchronous promotion requires a segmented branch")
    if async_qsa_promote and args.branch_mode != "segmented":
        raise ValueError("asynchronous QSA promotion requires a segmented branch")
    async_result: dict[str, Any] = {}
    async_thread = None
    async_started_ns = None
    if async_promote:
        # The physical contender owns independent lane/cache objects.  It
        # executes the same authoritative first cycle on a separate Metal
        # stream while the segmented lane supplies the first visible result.
        # We promote only after proving the first-cycle token traces identical.
        async_rows = copy.deepcopy([canonical, sibling])
        async_stream = mx.new_thread_unsafe_stream(mx.gpu)
        async_started_ns = time.perf_counter_ns()
        async_thread = threading.Thread(
            target=_run_physical_first_cycle,
            args=(
                mx,
                model,
                async_rows,
                async_stream,
                attach_self_mtp_lanes,
                propose_batched_self_mtp,
                commit_batched_self_mtp,
                async_result,
            ),
            name="live-tip-physical-race",
        )
        async_thread.start()
    segmented_self_mtp_stats(reset=True)
    segmented_before = segmented_self_mtp_stats(reset=False)
    if args.branch_mode == "physical_fanout":
        try:
            lease = fanout_owner.fork_live_tip()
            target_batch = lease.take_batch()
            draft_batch = [
                type(cache).merge([cache, cache])
                for cache in canonical.caches.draft
            ]
            mx.eval(
                [cache.state for cache in target_batch],
                [cache.state for cache in draft_batch],
            )
            prepared_caches = SelfMTPCachePair(
                target=target_batch,
                draft=draft_batch,
            )
        finally:
            fanout_owner.close()
        branch_batch = attach_prebatched_self_mtp_lanes(
            model,
            [canonical, sibling],
            prepared_caches,
        )
    else:
        attach = (
            attach_segmented_self_mtp_lanes
            if args.branch_mode == "segmented"
            else attach_self_mtp_lanes
        )
        branch_batch = attach(model, None, [canonical, sibling])
    async_qsa_ticket = None
    async_qsa_bind_ms = 0.0
    async_cache_receipt = None
    if async_qsa_promote:
        from mlx_lm.segmented_physical_promotion import (
            begin_segmented_physical_promotion,
        )

        async_started_ns = (
            prequeue_started_ns
            if staged_qsa is not None
            else time.perf_counter_ns()
        )
        if staged_qsa is None:
            async_qsa_ticket = begin_segmented_physical_promotion(
                branch_batch,
                reserve_tail=args.num_draft + 1,
                stream=mx.new_stream(mx.gpu),
                note=note_segmented_self_mtp,
            )
            async_qsa_queue_ms = (
                time.perf_counter_ns() - async_started_ns
            ) / 1e6
        else:
            bind_started_ns = time.perf_counter_ns()
            async_qsa_ticket = staged_qsa.bind(
                branch_batch, note=note_segmented_self_mtp
            )
            async_qsa_bind_ms = (
                time.perf_counter_ns() - bind_started_ns
            ) / 1e6
    # attach() consumes these detached owners.  Retaining the local wrappers
    # through all follow-up cycles pins the original B1 arrays beside the B2
    # cache and does not model serving, where the preparation list goes out of
    # scope as soon as admission publishes the batch.  It particularly biases
    # segmented promotion because its first cycle has just replaced recurrent
    # row state before the physical B2 takes ownership.
    rows = canonical = sibling = None
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
    async_physical_ready_ms = 0.0
    async_physical_total_ms = 0.0
    async_wait_after_first_ms = 0.0
    async_first_logprobs_exact = False
    if async_qsa_promote:
        promotion_started = time.perf_counter_ns()
        branch_batch, receipt = async_qsa_ticket.finish()
        async_wait_after_first_ms = (
            time.perf_counter_ns() - promotion_started
        ) / 1e6
        promotion_ms = async_wait_after_first_ms
        async_physical_ready_ms = async_qsa_queue_ms
        async_physical_total_ms = (
            time.perf_counter_ns() - async_started_ns
        ) / 1e6
        async_cache_receipt = asdict(receipt)
        # The ticket owns the retired segmented view (and therefore its B1
        # row tensors). Production clears the scheduler ticket before the next
        # decode call; retaining this local through the follow-up loop pins the
        # very cache allocation we are trying to measure away.
        async_qsa_ticket = None
    elif async_promote:
        promotion_started = time.perf_counter_ns()
        emptied, segmented_rows = detach_self_mtp_lanes(
            model, branch_batch, [0, 1]
        )
        if emptied.lanes or len(segmented_rows) != 2:
            raise RuntimeError("failed to detach rows from asynchronous promotion")
        async_thread.join()
        if "error" in async_result:
            raise RuntimeError("asynchronous physical contender failed") from async_result[
                "error"
            ]
        if async_result["tokens"] != branch_rows:
            raise AssertionError(
                "asynchronous physical contender differs from segmented first cycle"
            )
        visible_logprobs = [
            output.logprobs for outputs in proposal.outputs for output in outputs
        ]
        if len(visible_logprobs) != len(async_result["logprobs"]):
            raise AssertionError("asynchronous first-cycle logprob count differs")
        logprob_tests = [
            mx.array_equal(left, right)
            for left, right in zip(visible_logprobs, async_result["logprobs"])
        ]
        mx.eval(logprob_tests)
        async_first_logprobs_exact = all(bool(test.item()) for test in logprob_tests)
        if not async_first_logprobs_exact:
            raise AssertionError(
                "asynchronous physical contender logprobs differ from visible lane"
            )
        branch_batch = async_result["batch"]
        async_physical_ready_ms = float(async_result["ready_ms"])
        async_physical_total_ms = float(async_result["total_ms"])
        async_wait_after_first_ms = (
            time.perf_counter_ns() - promotion_started
        ) / 1e6
        promotion_ms = async_wait_after_first_ms
        for row in segmented_rows:
            transaction = getattr(row, "segment_transaction", None)
            if transaction is not None:
                transaction.close()
                row.segment_transaction = None
        segmented_rows = None
    elif getattr(args, "promote_after_first", False):
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
        promoted_rows = None
    steady_cache_geometry = _cache_geometry(branch_batch)
    followup_tokens = sum(emitted)
    followup_cycle_ms = []
    followup_started = time.perf_counter_ns()
    for _ in range(args.measured_cycles - 1):
        cycle_started = time.perf_counter_ns()
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
        followup_cycle_ms.append(
            (time.perf_counter_ns() - cycle_started) / 1e6
        )
    mx.synchronize()
    followup_finished = time.perf_counter_ns()
    ple_after = _ple_stats(model)
    ple_delta = {
        key: ple_after[key] - ple_before[key]
        for key in ple_before
    }

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
    fanout_after = gdn_prefix_fanout_stats()
    fanout_delta = {
        key: int(value) - int(fanout_before.get(key, 0))
        for key, value in fanout_after.items()
        if isinstance(value, int) and isinstance(fanout_before.get(key, 0), int)
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
        "apc": apc_receipt,
        "branch_mode": args.branch_mode,
        "prepare_ms": (prepared_ns - prepared_started) / 1e6,
        "warmup_ms": (last_warm_ns - prepared_ns) / 1e6,
        "warmup_cycles": args.warmup_cycles,
        "live_tip_position": live_tip_position,
        "live_tip_detach_ms": (detached_ns - last_warm_ns) / 1e6,
        "warm_to_detach_gap_ms": (detached_ns - last_warm_ns) / 1e6,
        "idle_wait_ms": idle_wait_ns / 1e6,
        "detach_to_branch_gap_ms": (branch_started - detached_ns) / 1e6,
        "branch_ready_ms": (
            (branch_ready_ns - branch_started) / 1e6 + fanout_capture_ms
        ),
        "fanout_capture_ms": fanout_capture_ms,
        "first_proposal_verify_ms": (first_output_ns - branch_ready_ns) / 1e6,
        "first_commit_ms": (first_commit_ns - first_output_ns) / 1e6,
        "promotion_after_first_ms": promotion_ms,
        "async_physical_ready_ms": async_physical_ready_ms,
        "async_physical_total_ms": async_physical_total_ms,
        "async_wait_after_first_ms": async_wait_after_first_ms,
        "async_qsa_queue_ms": async_qsa_queue_ms,
        "async_qsa_bind_ms": async_qsa_bind_ms,
        "async_qsa_prequeue_lead_ms": async_qsa_prequeue_lead_ms,
        "async_qsa_prequeued": async_qsa_prequeue,
        "async_qsa_prequeue_breakdown": async_qsa_prequeue_breakdown,
        "async_cache_receipt": async_cache_receipt,
        "steady_cache_geometry": steady_cache_geometry,
        "ple_delta": ple_delta,
        "branch_to_first_commit_if_prequeued_ms": (
            (
                (first_commit_ns - branch_started) / 1e6
                if async_qsa_prequeue
                else (first_commit_ns - branch_started) / 1e6 - async_qsa_queue_ms
            )
            if async_qsa_promote
            else None
        ),
        "async_first_logprobs_exact": async_first_logprobs_exact,
        "async_strategy": (
            "qsa_base_only_plus_suffix_patch"
            if async_qsa_promote
            else (
                "independent_physical_first_cycle_race"
                if async_promote
                else None
            )
        ),
        "async_physical_started_before_branch_ready_ms": (
            (branch_ready_ns - async_started_ns) / 1e6
            if async_started_ns is not None
            else 0.0
        ),
        "branch_to_first_output_ms": (
            (first_output_ns - branch_started) / 1e6 + fanout_capture_ms
        ),
        "branch_to_first_commit_ms": (
            (first_commit_ns - branch_started) / 1e6 + fanout_capture_ms
        ),
        "live_tip_to_first_commit_ms": (first_commit_ns - last_warm_ns) / 1e6,
        "followup_ms": (followup_finished - followup_started) / 1e6,
        "followup_cycle_ms": followup_cycle_ms,
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
        "fanout_delta": fanout_delta,
        "segmented_receipt": segmented_after,
        "memory": _mlx_memory(mx),
        "system_before": before,
        "system_after": after,
        "swap_growth_bytes": swap_growth,
    }

    batch = branch_batch = None
    rows = final_rows = None
    detached = canonical = sibling = None
    # The APC lookup result owns the live COW branch.  Drop every request-side
    # reference before clearing the resident source so the telemetry proves
    # that both the branch pin and APC owner were released at the arm boundary.
    lookup = sidecar = restored_rng = prompt_boundary = None
    if apc is not None:
        result["apc_clear"] = apc.clear(release_memory=False)
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    if apc is not None:
        result["apc_after_cleanup"] = apc.apc_stats
        active_leases = int(
            result["apc_after_cleanup"].get("cow", {}).get("active_leases", 0)
        )
        if active_leases:
            raise RuntimeError(
                f"APC arm leaked {active_leases} COW ownership lease(s)"
            )
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
    parser.add_argument("--async-promote-after-first", action="store_true")
    parser.add_argument("--async-qsa-promote-after-first", action="store_true")
    parser.add_argument("--async-qsa-prequeue", action="store_true")
    parser.add_argument("--async-qsa-prequeue-lead-ms", type=float, default=0.0)
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
