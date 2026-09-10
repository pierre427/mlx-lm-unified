#!/usr/bin/env python3
"""Exact target-only/fixed-k2/adaptive-1..3 gate for Flash-Next.

The default invocation is plan-only and imports no MLX modules. ``--execute``
loads one model, rotates all three arms within each 16K/64K triplet, writes an
atomic receipt after every arm, and refuses promotion unless both speculative
arms match the paired target-only greedy token IDs exactly.  It does not
manage a service or a GPU lease; the operator owns those boundaries.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "mlx-uag.qwen4-mtp-policy-gate.v1"
DEFAULT_MODEL = "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
ARMS = ("target_only", "fixed_k2", "adaptive_1_3")
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


def parse_contexts(value: str) -> list[int]:
    contexts = []
    for item in value.split(","):
        item = item.strip().lower().replace("_", "")
        if not item:
            continue
        contexts.append(int(float(item[:-1]) * 1024) if item.endswith("k") else int(item))
    if not contexts or any(value < 1 for value in contexts):
        raise ValueError("contexts must contain positive integers")
    if len(set(contexts)) != len(contexts):
        raise ValueError("contexts must be unique")
    return contexts


def arm_order(context_index: int, repetition: int) -> list[str]:
    shift = (context_index + repetition - 1) % len(ARMS)
    return list(ARMS[shift:] + ARMS[:shift])


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    contexts = parse_contexts(args.contexts)
    if args.reps < 1:
        raise ValueError("reps must be positive")
    if args.max_tokens < 32:
        raise ValueError("max-tokens must be at least 32 so adaptive depth can engage")
    if args.cooldown_seconds < 0:
        raise ValueError("cooldown-seconds cannot be negative")
    cells = []
    for context_index, context in enumerate(contexts):
        for repetition in range(1, args.reps + 1):
            cells.append(
                {
                    "context": context,
                    "repetition": repetition,
                    "order": arm_order(context_index, repetition),
                }
            )
    return {
        "schema": f"{SCHEMA}.plan",
        "created_at_utc": utc_now(),
        "execution_authorized": bool(args.execute),
        "model": args.model,
        "model_path_exists": Path(args.model).is_dir(),
        "contexts": contexts,
        "repetitions": args.reps,
        "max_tokens": args.max_tokens,
        "cooldown_seconds": args.cooldown_seconds,
        "prefill_step_size": args.prefill_step_size,
        "share_qsa_indices": args.share_qsa_indices,
        "arms": list(ARMS),
        "correctness_gate": "exact generated token IDs against paired target_only",
        "engagement_gate": (
            "fixed draft_proposed > 0; adaptive decisions > 0, expansions > 0, "
            "and observed max depth 3"
        ),
        "performance_metrics": "wall_s, TTFT, decode_tps_after_first, end_to_end_tps",
        "cache": "fresh target and MTP caches per arm; APC disabled",
        "cells": cells,
    }


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
        return prefix + ("state cache scheduler invariant rollback token x " * count) + suffix

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


def finish_row(
    arm: str,
    prompt_tokens: int,
    tokens: list[int],
    started: float,
    first_at: float,
    last_at: float,
    finished: float,
    peak_memory: int,
) -> dict[str, Any]:
    decode_s = max(last_at - first_at, 0.0)
    wall_s = finished - started
    return {
        "arm": arm,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(tokens),
        "token_ids": tokens,
        "token_sha256": token_digest(tokens),
        "ttft_s": first_at - started,
        "decode_s_after_first": decode_s,
        "decode_tps_after_first": (
            (len(tokens) - 1) / decode_s if len(tokens) > 1 and decode_s > 0 else None
        ),
        "wall_s": wall_s,
        "end_to_end_tps": len(tokens) / wall_s,
        "peak_memory_gib": peak_memory / float(1 << 30),
    }


def run_arm(model: Any, prompt: Any, args: argparse.Namespace, arm: str) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.speculation_router import DepthCeilingController

    mx.reset_peak_memory()
    tokens: list[int] = []
    started = time.perf_counter()
    first_at = last_at = None
    stats = None
    controller = None
    qsa_cycles = {"start_calls": 0, "share_requested": 0}
    original_start = getattr(model, "mtp_start_cycle", None)

    if arm == "target_only":
        generator = generate_step(
            prompt,
            model,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=0.0),
            prompt_cache=make_prompt_cache(model),
            prefill_step_size=args.prefill_step_size,
        )
    else:
        stats = HybridStats()
        num_draft = 2
        if arm == "adaptive_1_3":
            num_draft = 3
            controller = DepthCeilingController(1, 3)

        if original_start is not None:
            def counted_start(cache, share_qsa_indices=False):
                qsa_cycles["start_calls"] += 1
                qsa_cycles["share_requested"] += int(bool(share_qsa_indices))
                return original_start(cache, share_qsa_indices)

            model.mtp_start_cycle = counted_start
        generator = self_mtp_generate_step(
            prompt,
            model,
            num_draft=num_draft,
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
            sampling_temp=0.0,
            persistent_mtp=True,
            mtp_share_qsa_indices=args.share_qsa_indices,
            rate_gate=False,
            speculation_router=controller,
            stats=stats,
        )

    try:
        for item in generator:
            token, logprobs = item[:2]
            mx.eval(logprobs)
            tokens.append(int(token))
            last_at = time.perf_counter()
            first_at = first_at or last_at
    finally:
        if original_start is not None:
            model.mtp_start_cycle = original_start
    finished = time.perf_counter()
    if first_at is None or last_at is None or len(tokens) != args.max_tokens:
        raise RuntimeError(f"{arm} emitted {len(tokens)} of {args.max_tokens} tokens")
    row = finish_row(
        arm,
        int(prompt.size),
        tokens,
        started,
        first_at,
        last_at,
        finished,
        mx.get_peak_memory(),
    )
    if stats is not None:
        row["mtp"] = {**asdict(stats), "qsa_cycle_receipt": qsa_cycles}
        row["mtp"]["acceptance_rate"] = (
            stats.draft_accepted / stats.draft_proposed
            if stats.draft_proposed
            else None
        )
    if controller is not None:
        row["adaptive"] = controller.snapshot()
        row["adaptive"]["observed_max_depth"] = min(
            controller.ceiling,
            controller.floor + controller.expansions,
        )
    mx.clear_cache()
    return row


def summarize(rows: list[dict[str, Any]], contexts: list[int]) -> dict[str, Any]:
    summary = {}
    for context in contexts:
        selected = [row for row in rows if row["context"] == context]
        record = {}
        for arm in ARMS:
            arm_rows = [row for row in selected if row["arm"] == arm]
            record[arm] = {
                "samples": len(arm_rows),
                "median_decode_tps": statistics.median(
                    row["decode_tps_after_first"] for row in arm_rows
                ) if arm_rows else None,
                "median_wall_s": statistics.median(row["wall_s"] for row in arm_rows)
                if arm_rows else None,
            }
        comparisons = [row for row in selected if row["arm"] != "target_only"]
        record["exact_trials"] = len(comparisons)
        record["exact_matches"] = sum(bool(row.get("exact_target_match")) for row in comparisons)
        adaptive = [row for row in selected if row["arm"] == "adaptive_1_3"]
        record["adaptive_engaged_trials"] = sum(
            (row.get("adaptive") or {}).get("expansions", 0) > 0
            and (row.get("adaptive") or {}).get("observed_max_depth", 0) >= 3
            for row in adaptive
        )
        summary[str(context)] = record
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

    warm_prompt = mx.array(exact_prompt(tokenizer, 256, "policy-warmup"), mx.uint32)
    for arm in ARMS:
        warm_args = argparse.Namespace(**vars(args))
        warm_args.max_tokens = 32
        run_arm(model, warm_prompt, warm_args, arm)

    for context_index, context in enumerate(plan["contexts"]):
        if context_index and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        prompt = mx.array(
            exact_prompt(tokenizer, context, f"policy-{context}"), mx.uint32
        )
        for repetition in range(1, args.reps + 1):
            order = arm_order(context_index, repetition)
            bracket_before = system_snapshot()
            triplet = {}
            for ordinal, arm in enumerate(order):
                row = run_arm(model, prompt, args, arm)
                row.update(
                    context=context,
                    repetition=repetition,
                    order=order,
                    order_ordinal=ordinal,
                    completed_at_utc=utc_now(),
                )
                artifact["rows"].append(row)
                triplet[arm] = row
                atomic_write(output, artifact)
            target = triplet["target_only"]
            for arm in ("fixed_k2", "adaptive_1_3"):
                row = triplet[arm]
                row["exact_target_match"] = row["token_ids"] == target["token_ids"]
                row["first_divergence"] = next(
                    (
                        index
                        for index, pair in enumerate(zip(row["token_ids"], target["token_ids"]))
                        if pair[0] != pair[1]
                    ),
                    None,
                )
                row["decode_speedup_vs_target"] = (
                    row["decode_tps_after_first"] / target["decode_tps_after_first"]
                )
                row["wall_speedup_vs_target"] = target["wall_s"] / row["wall_s"]
            bracket_after = system_snapshot()
            for row in triplet.values():
                row["triplet_system_before"] = bracket_before
                row["triplet_system_after"] = bracket_after
            atomic_write(output, artifact)

    artifact["summary"] = summarize(artifact["rows"], plan["contexts"])
    artifact["metadata"]["system_after"] = system_snapshot()
    atomic_write(output, artifact)
    exact = all(
        row.get("exact_target_match", True)
        for row in artifact["rows"]
    )
    adaptive = [row for row in artifact["rows"] if row["arm"] == "adaptive_1_3"]
    engaged = bool(adaptive) and all(
        row["adaptive"]["expansions"] > 0
        and row["adaptive"]["observed_max_depth"] >= 3
        for row in adaptive
    )
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    print(f"VERDICT: {'QUALIFIED' if exact and engaged else 'REJECTED'}")
    return 0 if exact and engaged else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--contexts", default="16K,64K")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument(
        "--share-qsa-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out", default="results/qwen4-mtp-policy-gate-20260910.json"
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
