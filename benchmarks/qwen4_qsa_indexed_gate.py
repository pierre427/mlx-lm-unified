#!/usr/bin/env python3
"""Five-phase Metal gate for fixed-chunk indexed split-K QSA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
DEFAULT_OUTPUT = Path(
    "/Users/pierrelamy/Desktop/mlx-uag/results/"
    "qwen4-qsa-indexed-gate-v2-20260901.json"
)
CONTEXTS = (16_384, 32_768, 65_536, 131_072)
MODEL_CONTEXTS = CONTEXTS[:3]
SPLITS = (1, 2, 4, 8)
SWAP_LIMIT_MIB = 512.0
MODEL_LOAD_FREE_FLOOR = 45
RUN_FREE_FLOOR = 25


class GateFailure(RuntimeError):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = int(phase)


@contextmanager
def gpu_lock():
    """Take the lab-wide GPU lock and remove only this owner's file."""

    try:
        os.mkdir(GPU_LOCK)
    except FileExistsError as error:
        owner_path = GPU_LOCK / "owner.json"
        try:
            owner = owner_path.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "owner unavailable"
        raise SystemExit(f"GPU busy: {GPU_LOCK} exists ({owner})") from error
    owner_path = GPU_LOCK / "owner.json"
    owner = {
        "agent": "codex-k-indexed-v2",
        "label": "qwen4-qsa-indexed-gate-v2",
        "pid": os.getpid(),
        "purpose": "fixed-chunk indexed split-K QSA Metal gate",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        owner_path.write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        yield owner
    finally:
        owner_path.unlink(missing_ok=True)
        try:
            GPU_LOCK.rmdir()
        except OSError as error:
            raise RuntimeError(f"could not release {GPU_LOCK}: {error}") from error


def _run_text(command):
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def safety_snapshot():
    memory = _run_text(["/usr/bin/memory_pressure", "-Q"])
    match = re.search(r"free percentage:\s*(\d+)%", memory)
    if match is None:
        raise RuntimeError(f"could not parse memory pressure: {memory!r}")
    swap = _run_text(["/usr/sbin/sysctl", "vm.swapusage"])
    match_swap = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap)
    if match_swap is None:
        raise RuntimeError(f"could not parse swap usage: {swap!r}")
    used = float(match_swap.group(1))
    if match_swap.group(2) == "G":
        used *= 1024.0
    thermal = _run_text(["/usr/bin/pmset", "-g", "therm"])
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "free_percent": int(match.group(1)),
        "swap_used_mib": used,
        "thermal": thermal.splitlines(),
    }


def check_safety(
    snapshot, *, swap_baseline=None, before_load=False, phase=5
):
    floor = MODEL_LOAD_FREE_FLOOR if before_load else RUN_FREE_FLOOR
    if snapshot["free_percent"] < floor:
        raise GateFailure(
            int(phase),
            f"free memory {snapshot['free_percent']}% is below {floor}%",
        )
    if (
        swap_baseline is not None
        and snapshot["swap_used_mib"] - swap_baseline > SWAP_LIMIT_MIB
    ):
        growth = snapshot["swap_used_mib"] - swap_baseline
        raise GateFailure(int(phase), f"swap grew {growth:.2f} MiB")


def thermal_clean(lines):
    text = "\n".join(lines).lower()
    return (
        "no thermal warning" in text
        and "no performance warning" in text
        and "no cpu power status" in text
    )


def settle_thermal(timeout=60.0):
    """Require two clean thermal samples before the next timing arm."""

    started = time.monotonic()
    clean = 0
    samples = []
    while time.monotonic() - started < timeout:
        snapshot = safety_snapshot()
        samples.append(snapshot)
        clean = clean + 1 if thermal_clean(snapshot["thermal"]) else 0
        if clean == 2:
            return {
                "settled": True,
                "seconds": time.monotonic() - started,
                "samples": samples[-2:],
            }
        time.sleep(2.0)
    raise GateFailure(5, "thermal state did not settle")


def clone_containers(value):
    if isinstance(value, list):
        return [clone_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_containers(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_containers(item) for key, item in value.items()}
    return value


def clone_cache(cache):
    return [
        type(layer).from_state(
            clone_containers(layer.state), clone_containers(layer.meta_state)
        )
        for layer in cache
    ]


def compact_fixture(mx, compact_type, context, length=3):
    topk = 512
    q_pos = mx.arange(context - length, context, dtype=mx.int32)[None]
    ids = mx.broadcast_to(
        mx.arange(topk, dtype=mx.uint32)[None, None], (1, length, topk)
    )
    counts = mx.full((1, length), topk, dtype=mx.int32)
    tail_stop = q_pos + 1
    tail_start = tail_stop // 4 * 4
    return compact_type(
        block_ids=ids,
        block_counts=counts,
        tail_start=tail_start,
        tail_stop=tail_stop,
        left_padding=None,
        block_size=4,
        physical_width=context,
        causal_mask=None,
    )


def adversarial_fixture(mx, compact_type, *, batch, length, context=4096):
    width = 512
    ids = np.zeros((batch, length, width), dtype=np.uint32)
    counts = np.zeros((batch, length), dtype=np.int32)
    tail_stop = np.zeros((batch, length), dtype=np.int32)
    left = np.arange(batch, dtype=np.int32) % 3
    causal = np.ones((batch, 1, length, context), dtype=bool)
    for b in range(batch):
        for row in range(length):
            count = (512, 511, 257, 65, 1, 0)[(b + row) % 6]
            counts[b, row] = count
            if count:
                ids[b, row, :count] = np.arange(count, dtype=np.uint32)
            if row == 0:
                tail_stop[b, row] = 2048
            else:
                tail_stop[b, row] = context - left[b] - length + row + 1
            if count == 0:
                causal[b, 0, row] = False
    return compact_type(
        block_ids=mx.array(ids),
        block_counts=mx.array(counts),
        tail_start=mx.array(tail_stop // 4 * 4),
        tail_stop=mx.array(tail_stop),
        left_padding=mx.array(left),
        block_size=4,
        physical_width=context,
        causal_mask=mx.array(causal),
    )


def dense_fixture_mask(mx, context, length=3):
    q_pos = mx.arange(context - length, context, dtype=mx.int32)[None]
    token = mx.arange(context, dtype=mx.int32)[None, None]
    chosen = token // 4 < 512
    complete = ((q_pos + 1) // 4) * 4
    tail = (token >= complete[..., None]) & (token <= q_pos[..., None])
    return (chosen | tail)[:, None]


def max_scaled_error(mx, actual, expected):
    mx.eval(actual, expected)
    delta = float(mx.max(mx.abs(actual - expected)).item())
    scale = max(float(mx.max(mx.abs(expected)).item()), 1.0)
    return delta, delta / scale


def timed(mx, function):
    started = time.perf_counter()
    output = function()
    mx.eval(output)
    return time.perf_counter() - started


def phase1_candidate(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks
    from mlx_lm.models.qwen4_qsa_indexed import (
        qsa_indexed_status,
        qwen4_qsa_indexed_attention,
    )

    mx.random.seed(20260901)
    compact = compact_fixture(mx, QSACompactBlocks, CONTEXTS[0])
    q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, CONTEXTS[0], 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, CONTEXTS[0], 256)).astype(mx.bfloat16)
    qsa_indexed_status(reset=True)
    output = qwen4_qsa_indexed_attention(
        q, k, v, compact, scale=256**-0.5, splits=8
    )
    mx.eval(output)
    status = qsa_indexed_status()
    if status["candidate"] is None or status["fallbacks"]:
        raise GateFailure(1, f"candidate probe did not engage cleanly: {status}")
    return {"phase": 1, "status": "PASS", "receipt": status}


def cast_tie_counts(kernel_np, mirror_np, kernel_cast, mirror_cast, limit):
    mismatch = kernel_cast != mirror_cast
    delta = np.abs(kernel_np - mirror_np)
    ties = mismatch & (delta <= limit)
    return {
        "cast_mismatch_count": int(np.count_nonzero(mismatch)),
        "documented_tie_count": int(np.count_nonzero(ties)),
        "non_tie_count": int(np.count_nonzero(mismatch & ~ties)),
    }


def phase2_exactness(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks
    from mlx_lm.models.qwen4_qsa_indexed import (
        qwen4_qsa_indexed_attention,
        qwen4_qsa_indexed_reference,
    )

    rows = []
    worst_relative = 0.0
    all_s_equal = True
    non_ties = 0
    for length in range(1, 9):
        batch = 1 + length % 2
        mx.random.seed(20260901 + length)
        compact = adversarial_fixture(
            mx, QSACompactBlocks, batch=batch, length=length
        )
        q = mx.random.normal((batch, 24, length, 256)).astype(mx.float32)
        k = mx.random.normal((batch, 2, 4096, 256)).astype(mx.float32)
        v = mx.random.normal((batch, 2, 4096, 256)).astype(mx.float32)
        outputs = {
            splits: qwen4_qsa_indexed_attention(
                q, k, v, compact, scale=256**-0.5, splits=splits
            )
            for splits in SPLITS
        }
        mirror = qwen4_qsa_indexed_reference(
            q, k, v, compact, scale=256**-0.5, splits=8
        )
        mx.eval(*outputs.values(), mirror)
        first = np.asarray(outputs[1])
        split_checks = {}
        for splits in SPLITS:
            current = np.asarray(outputs[splits])
            equal = np.array_equal(first, current)
            all_s_equal = all_s_equal and equal
            split_checks[str(splits)] = {
                "bit_equal_to_s1": equal,
                "max_abs": float(np.max(np.abs(first - current))),
            }
        kernel = np.asarray(outputs[8])
        mirror_np = np.asarray(mirror)
        delta = float(np.max(np.abs(kernel - mirror_np)))
        scale = max(float(np.max(np.abs(mirror_np))), 1.0)
        relative = delta / scale
        worst_relative = max(worst_relative, relative)
        kernel_cast = np.asarray(outputs[8].astype(mx.bfloat16).astype(mx.float32))
        mirror_cast = np.asarray(mirror.astype(mx.bfloat16).astype(mx.float32))
        tie_counts = cast_tie_counts(
            kernel, mirror_np, kernel_cast, mirror_cast, 1.0e-4 * scale
        )
        non_ties += tie_counts["non_tie_count"]
        rows.append(
            {
                "batch": batch,
                "length": length,
                "kernel_mirror_max_abs": delta,
                "kernel_mirror_max_relative": relative,
                "splits": split_checks,
                **tie_counts,
            }
        )
        del q, k, v, outputs, mirror
        mx.clear_cache()
    passed = all_s_equal and worst_relative <= 1.0e-4 and non_ties == 0
    result = {
        "phase": 2,
        "status": "PASS" if passed else "FAIL",
        "s_invariant": all_s_equal,
        "worst_kernel_mirror_relative": worst_relative,
        "non_tie_count": non_ties,
        "rows": rows,
    }
    if not passed:
        raise GateFailure(2, json.dumps(result, sort_keys=True))
    return result


def phase3_gather(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import qwen4_qsa_indexed_attention

    rows = []
    for length in range(1, 9):
        batch = 1 + length % 2
        mx.random.seed(20261001 + length)
        compact = adversarial_fixture(
            mx, QSACompactBlocks, batch=batch, length=length
        )
        q = mx.random.normal((batch, 24, length, 256)).astype(mx.bfloat16)
        k = mx.random.normal((batch, 2, 4096, 256)).astype(mx.bfloat16)
        v = mx.random.normal((batch, 2, 4096, 256)).astype(mx.bfloat16)
        kernel = qwen4_qsa_indexed_attention(
            q, k, v, compact, scale=256**-0.5, splits=8
        )
        gather = _gather_qsa_attention(
            q, k, v, compact, scale=256**-0.5, tile_rows=1
        )
        absolute, relative = max_scaled_error(mx, kernel, gather)
        rows.append(
            {
                "batch": batch,
                "length": length,
                "max_abs": absolute,
                "max_relative": relative,
            }
        )
        del q, k, v, kernel, gather
        mx.clear_cache()
    return {"phase": 3, "status": "PASS", "asserted": False, "rows": rows}


def corpus_tokens(tokenizer, context):
    text = (
        "Direct indexed attention keeps selected KV rows in place. "
        "The verify gate checks rollback, token authority, and receipts. "
    )
    seed = tokenizer.encode(text)
    if not seed:
        raise ValueError("tokenizer produced an empty gate prompt")
    return (seed * (context // len(seed) + 1))[:context]


def top_two(mx, logprobs):
    indices = mx.argpartition(logprobs, kth=-2)[-2:]
    mx.eval(indices)
    pairs = [(int(index), float(logprobs[int(index)].item())) for index in indices]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


def run_model_arm(mx, model, token, cache, *, mode, max_tokens):
    from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
    from mlx_lm.models import qwen4_exp
    from mlx_lm.models.qwen4_qsa_indexed import (
        qsa_indexed_enabled,
        qsa_indexed_status,
        set_qwen4_qsa_indexed,
    )

    previous_indexed = qsa_indexed_enabled()
    names = (
        "_QSA_GATHER_KV",
        "_QSA_GATHER_MIN_QUERY",
        "_QSA_GATHER_MAX_QUERY",
        "_QSA_GATHER_MIN_CONTEXT",
        "_QSA_GATHER_MAX_CONTEXT",
        "_QSA_NAX_DECODE",
    )
    previous = {name: getattr(qwen4_exp, name) for name in names}
    try:
        set_qwen4_qsa_indexed(mode == "indexed")
        qwen4_exp._QSA_GATHER_KV = mode in {"indexed", "gather"}
        qwen4_exp._QSA_GATHER_MIN_QUERY = 1
        qwen4_exp._QSA_GATHER_MAX_QUERY = 8
        qwen4_exp._QSA_GATHER_MIN_CONTEXT = 0
        qwen4_exp._QSA_GATHER_MAX_CONTEXT = 0
        qwen4_exp._QSA_NAX_DECODE = False
        qsa_indexed_status(reset=True)
        stats = HybridStats()
        tokens = []
        chosen_logprobs = []
        top2 = []
        started = time.perf_counter()
        for output_token, logprobs, _ in self_mtp_generate_step(
            mx.array([token], dtype=mx.uint32),
            model,
            num_draft=2,
            max_tokens=max_tokens,
            persistent_mtp=True,
            prompt_cache=cache,
            stats=stats,
        ):
            output_token = int(output_token)
            tokens.append(output_token)
            chosen_logprobs.append(float(logprobs[output_token].item()))
            top2.append(top_two(mx, logprobs))
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(
            b"".join(int(item).to_bytes(4, "little") for item in tokens)
        ).hexdigest()
        return {
            "tokens": tokens,
            "digest": digest,
            "chosen_logprobs": chosen_logprobs,
            "top2": top2,
            "stats": stats.__dict__,
            "indexed_status": qsa_indexed_status(),
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(tokens) / elapsed,
        }
    finally:
        set_qwen4_qsa_indexed(previous_indexed)
        for name, value in previous.items():
            setattr(qwen4_exp, name, value)


def first_divergence(gather, indexed):
    width = min(len(gather["tokens"]), len(indexed["tokens"]))
    at = next(
        (
            index
            for index in range(width)
            if gather["tokens"][index] != indexed["tokens"][index]
        ),
        width if len(gather["tokens"]) != len(indexed["tokens"]) else None,
    )
    if at is None:
        return None
    if at >= width:
        return {"index": at, "classification": "STATE_FAULT", "reason": "length"}
    left = gather["top2"][at]
    right = indexed["top2"][at]
    left_gap = left[0][1] - left[1][1]
    right_gap = right[0][1] - right[1][1]
    same_pair = {left[0][0], left[1][0]} == {right[0][0], right[1][0]}
    near = same_pair and max(left_gap, right_gap) <= 0.002
    return {
        "index": at,
        "gather_token": gather["tokens"][at],
        "indexed_token": indexed["tokens"][at],
        "gather_top2": left,
        "indexed_top2": right,
        "gather_gap": left_gap,
        "indexed_gap": right_gap,
        "same_candidate_pair": same_pair,
        "classification": "NEAR_TIE" if near else "STATE_FAULT",
    }


def prefill_base(mx, model, prompt, make_prompt_cache):
    base = make_prompt_cache(model)
    inputs = mx.array(prompt[:-1], dtype=mx.uint32)[None]
    output = model(inputs, cache=base)
    mx.eval(output, [layer.state for layer in base])
    return base, int(prompt[-1])


def phase4_model(mx, model, tokenizer, max_tokens, swap_baseline):
    from mlx_lm.models.cache import make_prompt_cache

    rows = []
    for context in MODEL_CONTEXTS:
        checkpoint = safety_snapshot()
        check_safety(checkpoint, swap_baseline=swap_baseline, phase=4)
        prompt = corpus_tokens(tokenizer, context)
        base, token = prefill_base(mx, model, prompt, make_prompt_cache)
        gather = run_model_arm(
            mx, model, token, clone_cache(base), mode="gather", max_tokens=max_tokens
        )
        indexed = run_model_arm(
            mx, model, token, clone_cache(base), mode="indexed", max_tokens=max_tokens
        )
        divergence = first_divergence(gather, indexed)
        max_logprob_delta = max(
            (
                abs(left - right)
                for left, right in zip(
                    gather["chosen_logprobs"], indexed["chosen_logprobs"]
                )
            ),
            default=0.0,
        )
        status = indexed["indexed_status"]
        reached = status["query_width_counts"].get("2-8", {}).get("engaged", 0)
        passed = (
            max_logprob_delta <= 0.002
            and not status["fallbacks"]
            and bool(reached)
            and (divergence is None or divergence["classification"] == "NEAR_TIE")
        )
        row = {
            "context": context,
            "status": "PASS" if passed else "FAIL",
            "gather_digest": gather["digest"],
            "indexed_digest": indexed["digest"],
            "max_chosen_logprob_delta": max_logprob_delta,
            "divergence": divergence,
            "gather_stats": gather["stats"],
            "indexed_stats": indexed["stats"],
            "indexed_status": status,
            "cache_boundary": "both arms clone one cache prefilled through prompt[-2]",
            "draft_contract": "self-MTP k=2 uses width-3 rollback-recording verify",
            "safety": checkpoint,
        }
        rows.append(row)
        if not passed:
            raise GateFailure(4, json.dumps(row, sort_keys=True))
        del base, gather, indexed
        mx.clear_cache()
    return {"phase": 4, "status": "PASS", "rows": rows}


def isolated_timing(mx, swap_baseline):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import qwen4_qsa_indexed_attention

    rows = []
    for context in CONTEXTS:
        compact = compact_fixture(mx, QSACompactBlocks, context)
        for length in (3, 1):
            mx.random.seed(20262000 + context + length)
            q = mx.random.normal((1, 24, length, 256)).astype(mx.bfloat16)
            k = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
            v = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
            mask = dense_fixture_mask(mx, context, length)
            arms = {
                "indexed": lambda: qwen4_qsa_indexed_attention(
                    q, k, v, compact, scale=256**-0.5, splits=8
                ),
                "gather": lambda: _gather_qsa_attention(
                    q, k, v, compact, scale=256**-0.5, tile_rows=1
                ),
                "dense": lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=256**-0.5, mask=mask
                ),
            }
            samples = {name: [] for name in arms}
            settlements = []
            for function in arms.values():
                timed(mx, function)
            names = list(arms)
            for repeat in range(8):
                order = names[repeat % 3 :] + names[: repeat % 3]
                for name in order:
                    settlements.append(settle_thermal())
                    samples[name].append(timed(mx, arms[name]))
                    check_safety(
                        safety_snapshot(), swap_baseline=swap_baseline
                    )
            rows.append(
                {
                    "context": context,
                    "length": length,
                    "median_ms": {
                        name: statistics.median(values) * 1000.0
                        for name, values in samples.items()
                    },
                    "settlements": settlements,
                }
            )
            del q, k, v
            mx.clear_cache()
    return rows


def model_timing(mx, model, tokenizer, timing_tokens, swap_baseline):
    from mlx_lm.models.cache import make_prompt_cache

    rows = []
    for context in MODEL_CONTEXTS:
        prompt = corpus_tokens(tokenizer, context)
        base, token = prefill_base(mx, model, prompt, make_prompt_cache)
        arms = {}
        for mode in ("indexed", "gather", "dense"):
            settlement = settle_thermal()
            result = run_model_arm(
                mx,
                model,
                token,
                clone_cache(base),
                mode=mode,
                max_tokens=timing_tokens,
            )
            check_safety(safety_snapshot(), swap_baseline=swap_baseline)
            arms[mode] = {
                "tokens_per_second": result["tokens_per_second"],
                "elapsed_seconds": result["elapsed_seconds"],
                "digest": result["digest"],
                "settlement": settlement,
                "indexed_status": result["indexed_status"],
            }
        rows.append({"context": context, "arms": arms})
        del base
        mx.clear_cache()
    return rows


def phase5_timing(mx, model, tokenizer, timing_tokens, swap_baseline):
    return {
        "phase": 5,
        "status": "PASS",
        "isolated": isolated_timing(mx, swap_baseline),
        "end_to_end": model_timing(
            mx, model, tokenizer, timing_tokens, swap_baseline
        ),
    }


def write_artifacts(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jsonl = output.with_suffix(".jsonl")
    rows = [report["manifest"]] + report["phases"]
    if "failure" in report:
        rows.append({"type": "failure", **report["failure"]})
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output, jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timing-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()
    if not args.execute_metal:
        raise SystemExit("refusing to dispatch Metal without --execute-metal")

    os.environ["MLX_QWEN4_QSA_INDEXED"] = "1"
    os.environ["MLX_QWEN4_QSA_GATHER_KV"] = "1"
    os.environ["MLX_QWEN4_QSA_INDEXED_MIN_QUERY"] = "1"
    report = {
        "manifest": {
            "type": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-gate.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent": "codex-k-indexed-v2",
            "model": str(args.model),
            "outcome": "RUNNING",
        },
        "phases": [],
        "safety": [],
    }
    exit_code = 0
    with gpu_lock() as owner:
        report["manifest"]["lock_owner"] = owner
        import mlx.core as mx
        from mlx_lm.utils import load

        mx.set_default_device(mx.gpu)
        baseline = safety_snapshot()
        report["safety"].append(baseline)
        swap_baseline = baseline["swap_used_mib"]
        current_phase = 1
        try:
            report["phases"].append(phase1_candidate(mx))
            current_phase = 2
            report["phases"].append(phase2_exactness(mx))
            current_phase = 3
            report["phases"].append(phase3_gather(mx))
            current_phase = 4
            mx.clear_cache()
            gc.collect()
            before_load = safety_snapshot()
            report["safety"].append(before_load)
            check_safety(
                before_load,
                swap_baseline=swap_baseline,
                before_load=True,
                phase=4,
            )
            model, tokenizer = load(str(args.model))
            model.eval()
            mx.eval(model.parameters())
            report["phases"].append(
                phase4_model(
                    mx, model, tokenizer, args.max_tokens, swap_baseline
                )
            )
            current_phase = 5
            report["phases"].append(
                phase5_timing(
                    mx, model, tokenizer, args.timing_tokens, swap_baseline
                )
            )
            report["manifest"]["outcome"] = "PASS"
        except GateFailure as error:
            report["manifest"]["outcome"] = f"FAIL_PHASE_{error.phase}"
            report["failure"] = {"phase": error.phase, "message": str(error)}
            exit_code = 1
        except Exception as error:
            report["manifest"]["outcome"] = f"ERROR_PHASE_{current_phase}"
            report["failure"] = {
                "phase": current_phase,
                "type": type(error).__name__,
                "message": str(error),
            }
            exit_code = 1
        finally:
            final = safety_snapshot()
            report["safety"].append(final)
            report["manifest"]["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            write_artifacts(report, args.output)
    print(args.output)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
