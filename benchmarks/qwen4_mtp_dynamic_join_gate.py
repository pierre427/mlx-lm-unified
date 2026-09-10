#!/usr/bin/env python3
"""Exact fixed-cohort versus dynamic-join gate for Flash-Next self-MTP.

The default invocation is plan-only and imports no MLX modules. ``--execute``
loads one model, runs paired fixed and dynamic schedules over the same prepared
requests, and writes an atomic receipt after every arm. The dynamic arm starts
with a subset of lanes and attaches the remainder at a deterministic cycle
boundary. Promotion requires exact per-request token IDs across schedules and
direct evidence that shared QSA was both requested and reused after the join.

This runner does not manage a service, GPU lease, or thermal cooldown. The
operator owns those boundaries. The aggregate throughput clock excludes the
identical sequential prefill phase so it measures the continuous transaction;
the receipt also records prefill and total wall time separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "mlx-uag.qwen4-mtp-dynamic-join-gate.v1"
DEFAULT_MODEL = "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
ARMS = ("fixed_cohort", "dynamic_join")
RECEIPT_ENV = (
    "MLX_QWEN4_PLE_NVME",
    "MLX_QWEN4_FUSED_GDN_DECODE",
    "MLX_QWEN4_FUSED_GDN_VERIFY",
    "MLX_QWEN4_QSA_POOLED_KEY_CACHE",
    "MLX_QWEN4_QSA_STAGE1_KERNEL",
    "MLX_QWEN4_QSA_SCATTER_CHOSEN",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def system_snapshot() -> dict[str, Any]:
    result = {}
    for name, command in {
        "pmset_therm": ["pmset", "-g", "therm"],
        "memory_pressure": ["memory_pressure", "-Q"],
        "swapusage": ["sysctl", "-n", "vm.swapusage"],
    }.items():
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        result[name] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    return result


def arm_order(repetition: int) -> list[str]:
    return list(ARMS if repetition % 2 else reversed(ARMS))


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.context < 1:
        raise ValueError("context must be positive")
    if args.lanes < 2:
        raise ValueError("lanes must be at least 2")
    if not 1 <= args.initial_lanes < args.lanes:
        raise ValueError("initial-lanes must be in [1, lanes)")
    if args.join_after_cycles < 1:
        raise ValueError("join-after-cycles must be positive")
    if args.num_draft < 2 and args.share_qsa_indices:
        raise ValueError("shared-QSA engagement requires num-draft >= 2")
    if args.num_draft < 1:
        raise ValueError("num-draft must be positive")
    minimum_live_join = 1 + (args.num_draft + 1) * args.join_after_cycles
    if args.max_tokens <= minimum_live_join:
        raise ValueError("max-tokens is too small to guarantee a live join")
    if args.reps < 1:
        raise ValueError("reps must be positive")
    if args.cooldown_seconds < 0:
        raise ValueError("cooldown-seconds cannot be negative")
    return {
        "schema": f"{SCHEMA}.plan",
        "created_at_utc": utc_now(),
        "execution_authorized": bool(args.execute),
        "model": args.model,
        "model_path_exists": Path(args.model).is_dir(),
        "context": args.context,
        "lanes": args.lanes,
        "initial_lanes": args.initial_lanes,
        "joining_lanes": args.lanes - args.initial_lanes,
        "join_after_cycles": args.join_after_cycles,
        "max_tokens_per_lane": args.max_tokens,
        "num_draft": args.num_draft,
        "repetitions": args.reps,
        "prefill_step_size": args.prefill_step_size,
        "share_qsa_indices": args.share_qsa_indices,
        "arms": list(ARMS),
        "orders": [arm_order(rep) for rep in range(1, args.reps + 1)],
        "correctness_gate": "exact per-UID token IDs, fixed cohort versus dynamic join",
        "engagement_gate": (
            "dynamic membership epoch advances at join; shared QSA start requests and "
            "second-step reuse are observed after the join"
        ),
        "performance_metrics": (
            "prepare_wall_s, transaction_wall_s, total_wall_s, aggregate decode and "
            "end-to-end t/s, per-lane acceptance"
        ),
        "timing_boundary": (
            "all lanes are prefetched sequentially before transaction timing in both "
            "arms; dynamic lanes are detached until the configured cycle boundary"
        ),
    }


def template_ids(tokenizer: Any, text: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
    )


def fill_ids_before_stable_suffix(
    tokenizer: Any,
    current: list[int],
    one_unit_longer: list[int],
    target: int,
) -> list[int]:
    """Insert inert filler IDs before the unchanged chat suffix.

    Text-level padding is not exact because BPE can merge the last filler with
    the first suffix token. Comparing adjacent rendered fillers identifies the
    stable suffix/generation-prompt tail; inserting already-tokenized ``x`` IDs
    immediately before it leaves every tail ID unchanged by construction.
    """
    if len(current) > target:
        raise ValueError("current token sequence already exceeds target")
    missing = target - len(current)
    if not missing:
        return list(current)
    common_suffix = 0
    limit = min(len(current), len(one_unit_longer))
    while (
        common_suffix < limit
        and current[-common_suffix - 1] == one_unit_longer[-common_suffix - 1]
    ):
        common_suffix += 1
    if common_suffix == 0 or common_suffix == len(current):
        raise RuntimeError("could not isolate a stable chat-template suffix")
    filler = list(tokenizer.encode("x", add_special_tokens=False))
    if not filler:
        raise RuntimeError("tokenizer produced no ordinary filler token")
    insertion = len(current) - common_suffix
    result = current[:insertion] + [int(filler[0])] * missing + current[insertion:]
    if result[-common_suffix:] != current[-common_suffix:]:
        raise RuntimeError("exact padding changed the chat-template suffix")
    return result


def exact_prompt(tokenizer: Any, target: int, marker: str) -> list[int]:
    prefix = f"Benchmark marker {marker}. Context data follows.\n"
    suffix = (
        "\nEnd context. Ignore it and produce a numbered implementation review "
        "until the output limit."
    )

    def render(count: int) -> str:
        unit = "state cache scheduler invariant rollback token x "
        return prefix + (unit * count) + suffix

    if len(template_ids(tokenizer, render(0))) > target:
        raise ValueError(f"context {target} is below chat-template overhead")
    low, high = 0, 1
    while len(template_ids(tokenizer, render(high))) <= target:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if len(template_ids(tokenizer, render(middle))) <= target:
            low = middle
        else:
            high = middle
    ids = template_ids(tokenizer, render(low))
    ids = fill_ids_before_stable_suffix(
        tokenizer,
        ids,
        template_ids(tokenizer, render(low + 1)),
        target,
    )
    if len(ids) != target:
        raise RuntimeError(f"exact prompt construction failed: {len(ids)} != {target}")
    return ids


class QSAShareCounter:
    """Observe actual second-draft-step QSA reuse without changing model math."""

    def __init__(self, model: Any):
        self.model = model
        self.original_start = getattr(model, "mtp_start_cycle", None)
        self.original_step = getattr(model, "mtp_step", None)
        self.start_calls = 0
        self.share_requested = 0
        self.reuse_observed = 0
        self.post_join_share_requested = 0
        self.post_join_reuse_observed = 0
        self.joined = False
        self._draft_call = 0
        self._share_cycle = False

    def mark_join(self) -> None:
        self.joined = True

    def __enter__(self):
        if self.original_start is None or self.original_step is None:
            raise RuntimeError("model does not expose MTP cycle hooks")

        def counted_start(cache, share_qsa_indices=False):
            shared = bool(share_qsa_indices)
            self.start_calls += 1
            self.share_requested += int(shared)
            self.post_join_share_requested += int(shared and self.joined)
            self._draft_call = 0
            self._share_cycle = shared
            return self.original_start(cache, share_qsa_indices)

        def counted_step(hidden, tokens, cache):
            reused = False
            if self._share_cycle and self._draft_call > 0:
                qsa = [item for item in cache if hasattr(item, "_mtp_shared_topk")]
                reused = bool(qsa) and all(item._mtp_shared_topk is not None for item in qsa)
            if reused:
                self.reuse_observed += 1
                self.post_join_reuse_observed += int(self.joined)
            result = self.original_step(hidden, tokens, cache)
            self._draft_call += 1
            return result

        self.model.mtp_start_cycle = counted_start
        self.model.mtp_step = counted_step
        return self

    def __exit__(self, *_exc):
        self.model.mtp_start_cycle = self.original_start
        self.model.mtp_step = self.original_step

    def receipt(self) -> dict[str, int]:
        return {
            "start_calls": self.start_calls,
            "share_requested": self.share_requested,
            "reuse_observed": self.reuse_observed,
            "post_join_share_requested": self.post_join_share_requested,
            "post_join_reuse_observed": self.post_join_reuse_observed,
        }


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def run_schedule(model: Any, prompts: list[Any], args: argparse.Namespace, arm: str):
    import mlx.core as mx
    from mlx_lm.hybrid_speculative import (
        attach_self_mtp_lanes,
        commit_batched_self_mtp,
        detach_self_mtp_lanes,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )
    from mlx_lm.sample_utils import LaneRNG

    mx.reset_peak_memory()
    started = time.perf_counter()
    prepared = []
    traces: dict[int, list[int]] = {}
    for uid, prompt in enumerate(prompts):
        lane, first = prepare_self_mtp_lane(
            prompt,
            model,
            uid=uid,
            max_tokens=args.max_tokens,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(args.seed + uid),
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
        prepared.append(lane)
        traces[uid] = [int(first.token)]
    prepared_at = time.perf_counter()

    if arm == "fixed_cohort":
        batch = attach_self_mtp_lanes(model, None, prepared)
        pending = []
    else:
        batch = attach_self_mtp_lanes(model, None, prepared[: args.initial_lanes])
        pending = prepared[args.initial_lanes :]

    transaction_started = time.perf_counter()
    cycle = 0
    join_receipt = None
    final_stats = {}
    with QSAShareCounter(model) as qsa:
        while batch.lanes or pending:
            if arm == "dynamic_join" and pending and cycle >= args.join_after_cycles:
                epoch_before = int(batch.membership_epoch)
                uids_before = [int(lane.uid) for lane in batch.lanes]
                batch = attach_self_mtp_lanes(model, batch, pending)
                qsa.mark_join()
                join_receipt = {
                    "cycle": cycle,
                    "epoch_before": epoch_before,
                    "epoch_after": int(batch.membership_epoch),
                    "uids_before": uids_before,
                    "uids_after": [int(lane.uid) for lane in batch.lanes],
                    "joined_uids": [int(item.lane.uid) for item in pending],
                }
                pending = []

            proposal = propose_batched_self_mtp(model, batch)
            mx.eval([output.logprobs for row in proposal.outputs for output in row])
            emitted_counts = []
            terminal = []
            for lane, outputs in zip(batch.lanes, proposal.outputs):
                remaining = args.max_tokens - len(traces[lane.uid])
                delivered = outputs[:remaining]
                traces[lane.uid].extend(int(output.token) for output in delivered)
                emitted_counts.append(len(delivered))
                terminal.append(len(traces[lane.uid]) >= args.max_tokens)
            commit_batched_self_mtp(
                batch,
                proposal,
                emitted_counts=emitted_counts,
                terminal=terminal,
            )
            leaving = [index for index, done in enumerate(terminal) if done]
            if leaving:
                batch, detached = detach_self_mtp_lanes(model, batch, leaving)
                for item in detached:
                    stats = item.lane.stats
                    final_stats[item.lane.uid] = {
                        "cycles": int(stats.cycles),
                        "draft_proposed": int(stats.draft_proposed),
                        "draft_accepted": int(stats.draft_accepted),
                        "acceptance_rate": (
                            float(stats.draft_accepted) / stats.draft_proposed
                            if stats.draft_proposed
                            else None
                        ),
                    }
            cycle += 1

    transaction_finished = time.perf_counter()
    if pending:
        raise RuntimeError("dynamic join did not occur")
    if any(len(row) != args.max_tokens for row in traces.values()):
        raise RuntimeError("one or more lanes did not reach max-tokens")
    if len(final_stats) != args.lanes:
        raise RuntimeError("one or more terminal lane receipts are missing")
    transaction_wall = transaction_finished - transaction_started
    total_wall = transaction_finished - started
    total_tokens = sum(len(row) for row in traces.values())
    decode_tokens = total_tokens - args.lanes
    return {
        "arm": arm,
        "prepare_wall_s": prepared_at - started,
        "transaction_wall_s": transaction_wall,
        "total_wall_s": total_wall,
        "completion_tokens": total_tokens,
        "transaction_decode_tokens": decode_tokens,
        "aggregate_decode_tps": decode_tokens / transaction_wall,
        "aggregate_end_to_end_tps": total_tokens / total_wall,
        "cycles": cycle,
        "peak_memory_gib": mx.get_peak_memory() / float(1 << 30),
        "token_ids_by_uid": {str(uid): row for uid, row in traces.items()},
        "token_sha256_by_uid": {str(uid): token_digest(row) for uid, row in traces.items()},
        "lane_mtp": {str(uid): row for uid, row in final_stats.items()},
        "qsa_share": qsa.receipt(),
        "join": join_receipt,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "samples": len(selected),
            "median_transaction_wall_s": statistics.median(
                row["transaction_wall_s"] for row in selected
            ) if selected else None,
            "median_aggregate_decode_tps": statistics.median(
                row["aggregate_decode_tps"] for row in selected
            ) if selected else None,
        }
    dynamic = [row for row in rows if row["arm"] == "dynamic_join"]
    summary["dynamic_exact_matches"] = sum(
        bool(row.get("exact_fixed_match")) for row in dynamic
    )
    summary["dynamic_trials"] = len(dynamic)
    summary["dynamic_qsa_engaged"] = sum(
        row["qsa_share"]["post_join_share_requested"] > 0
        and row["qsa_share"]["post_join_reuse_observed"] > 0
        for row in dynamic
    )
    return summary


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    if not plan["model_path_exists"]:
        raise FileNotFoundError(args.model)
    import mlx.core as mx
    from mlx_lm.utils import load

    artifact = {
        "metadata": {
            **plan,
            "schema": SCHEMA,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": distribution_version("mlx"),
            "mlx_lm": distribution_version("mlx-lm"),
            "source_revision": git_revision(),
            "environment": {name: os.environ.get(name) for name in RECEIPT_ENV},
            "system_before": system_snapshot(),
        },
        "rows": [],
        "summary": {},
    }
    output = Path(args.out)
    atomic_write(output, artifact)
    model, tokenizer = load(args.model)
    prompts = [
        mx.array(exact_prompt(tokenizer, args.context, f"join-lane-{uid}"), mx.uint32)
        for uid in range(args.lanes)
    ]

    # Compile/warm both membership shapes without including them in receipts.
    warm_args = argparse.Namespace(**vars(args))
    warm_args.context = 256
    warm_args.max_tokens = max(32, 2 * args.join_after_cycles + 5)
    warm_prompts = [
        mx.array(exact_prompt(tokenizer, 256, f"join-warm-{uid}"), mx.uint32)
        for uid in range(args.lanes)
    ]
    for arm in ARMS:
        run_schedule(model, warm_prompts, warm_args, arm)
        mx.clear_cache()

    for repetition in range(1, args.reps + 1):
        if repetition > 1 and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        order = arm_order(repetition)
        before = system_snapshot()
        pair = {}
        for ordinal, arm in enumerate(order):
            row = run_schedule(model, prompts, args, arm)
            row.update(
                repetition=repetition,
                order=order,
                order_ordinal=ordinal,
                completed_at_utc=utc_now(),
            )
            artifact["rows"].append(row)
            pair[arm] = row
            atomic_write(output, artifact)
            mx.clear_cache()
        fixed = pair["fixed_cohort"]
        dynamic = pair["dynamic_join"]
        per_uid = {}
        for uid in range(args.lanes):
            key = str(uid)
            left = fixed["token_ids_by_uid"][key]
            right = dynamic["token_ids_by_uid"][key]
            per_uid[key] = {
                "exact": left == right,
                "first_divergence": first_divergence(left, right),
            }
        dynamic["exact_by_uid"] = per_uid
        dynamic["exact_fixed_match"] = all(row["exact"] for row in per_uid.values())
        dynamic["transaction_tps_ratio_vs_fixed"] = (
            dynamic["aggregate_decode_tps"] / fixed["aggregate_decode_tps"]
        )
        after = system_snapshot()
        for row in pair.values():
            row["pair_system_before"] = before
            row["pair_system_after"] = after
        atomic_write(output, artifact)

    artifact["summary"] = summarize(artifact["rows"])
    artifact["metadata"]["system_after"] = system_snapshot()
    atomic_write(output, artifact)
    dynamic_rows = [row for row in artifact["rows"] if row["arm"] == "dynamic_join"]
    exact = bool(dynamic_rows) and all(row["exact_fixed_match"] for row in dynamic_rows)
    engaged = bool(dynamic_rows) and all(
        row["join"] is not None
        and row["join"]["epoch_after"] == row["join"]["epoch_before"] + 1
        and row["qsa_share"]["post_join_share_requested"] > 0
        and row["qsa_share"]["post_join_reuse_observed"] > 0
        for row in dynamic_rows
    )
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    print(f"VERDICT: {'QUALIFIED' if exact and engaged else 'REJECTED'}")
    return 0 if exact and engaged else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--initial-lanes", type=int, default=1)
    parser.add_argument("--join-after-cycles", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--share-qsa-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out", default="results/qwen4-mtp-dynamic-join-gate-20260910.json"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
