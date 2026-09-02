#!/usr/bin/env python3
"""Bounded Metal gate for indexed QSA reads from affine quantized K/V."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
CONTEXTS = (16_384, 32_768, 65_536)
SPLITS = (1, 4, 8)


def _text(command):
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def safety(label):
    memory = _text(["/usr/bin/memory_pressure", "-Q"])
    swap = _text(["/usr/sbin/sysctl", "vm.swapusage"])
    thermal = _text(["/usr/bin/pmset", "-g", "therm"])
    free = int(memory.rsplit(" ", 1)[-1].rstrip("%"))
    return {
        "label": label,
        "at": datetime.now(timezone.utc).isoformat(),
        "free_percent": free,
        "swapusage": swap,
        "thermal": thermal.splitlines(),
        "load1": os.getloadavg()[0],
    }


def require_safety(snapshot, *, floor):
    if snapshot["free_percent"] < floor:
        raise RuntimeError(
            f"free memory {snapshot['free_percent']}% is below {floor}%"
        )


def wait_for_load(limit=10.0, timeout=600.0):
    started = time.monotonic()
    rows = []
    while True:
        load1 = os.getloadavg()[0]
        rows.append({"at": time.time(), "load1": load1})
        if load1 < limit:
            return rows
        if time.monotonic() - started >= timeout:
            raise RuntimeError(f"load1 stayed at {load1:.2f}, above {limit}")
        time.sleep(10.0)


@contextmanager
def gpu_lock():
    inherited = os.environ.get("MLXUAG_GPU_LOCK_ALREADY_HELD")
    if inherited:
        owner = json.loads((GPU_LOCK / "owner.json").read_text())
        if owner.get("owner") != inherited:
            raise RuntimeError("inherited GPU lock owner does not match")
        yield owner
        return
    os.mkdir(GPU_LOCK)
    owner = {
        "owner": "codex-r4-qkv",
        "label": "qwen4-qsa-indexed-qkv-gate",
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (GPU_LOCK / "owner.json").write_text(json.dumps(owner) + "\n")
        yield owner
    finally:
        (GPU_LOCK / "owner.json").unlink(missing_ok=True)
        GPU_LOCK.rmdir()


def compact_fixture(mx, compact_type, context, length=3):
    topk = 512
    q_pos = mx.arange(context - length, context, dtype=mx.int32)[None]
    ids = mx.broadcast_to(
        mx.arange(topk, dtype=mx.uint32)[None, None], (1, length, topk)
    )
    counts = mx.full((1, length), topk, dtype=mx.int32)
    tail_stop = q_pos + 1
    return compact_type(
        block_ids=ids,
        block_counts=counts,
        tail_start=tail_stop // 4 * 4,
        tail_stop=tail_stop,
        left_padding=None,
        block_size=4,
        physical_width=context,
        causal_mask=None,
    )


def exact_delta(mx, left, right):
    mx.eval(left, right)
    left_np = np.asarray(left.astype(mx.float32))
    right_np = np.asarray(right.astype(mx.float32))
    return {
        "bit_exact": bool(np.array_equal(left_np, right_np)),
        "mismatch_count": int(np.count_nonzero(left_np != right_np)),
        "max_abs": float(np.max(np.abs(left_np - right_np))),
    }


def timed(mx, function):
    started = time.perf_counter()
    value = function()
    mx.eval(value)
    return time.perf_counter() - started


def phase_exactness(mx):
    from mlx_lm.models.qwen4_exp import (
        QSACompactBlocks,
        _gather_qsa_quantized_attention,
    )
    from mlx_lm.models.qwen4_qsa_indexed import (
        qwen4_qsa_indexed_quantized_attention,
        qwen4_qsa_indexed_quantized_reference,
    )

    context = CONTEXTS[0]
    compact = compact_fixture(mx, QSACompactBlocks, context)
    rows = []
    for bits in (8, 4):
        mx.random.seed(20260910 + bits)
        q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
        q_keys = mx.quantize(k, group_size=64, bits=bits)
        q_values = mx.quantize(v, group_size=64, bits=bits)
        gather = _gather_qsa_quantized_attention(
            q,
            q_keys,
            q_values,
            compact,
            scale=256**-0.5,
            tile_rows=1,
            group_size=64,
            key_bits=bits,
            value_bits=bits,
        )
        mirror = qwen4_qsa_indexed_quantized_reference(
            q,
            q_keys,
            q_values,
            compact,
            scale=256**-0.5,
            splits=8,
            group_size=64,
            key_bits=bits,
            value_bits=bits,
        )
        kernels = {
            split: qwen4_qsa_indexed_quantized_attention(
                q,
                q_keys,
                q_values,
                compact,
                scale=256**-0.5,
                splits=split,
                group_size=64,
                key_bits=bits,
                value_bits=bits,
            )
            for split in SPLITS
        }
        mx.eval(gather, mirror, *kernels.values())
        row = {
            "bits": bits,
            "context": context,
            "kernel_vs_gather": {
                str(split): exact_delta(mx, output, gather)
                for split, output in kernels.items()
            },
            "kernel_vs_mirror": {
                str(split): exact_delta(mx, output, mirror)
                for split, output in kernels.items()
            },
            "mirror_vs_gather": exact_delta(mx, mirror, gather),
            "split_invariance": {
                str(split): exact_delta(mx, output, kernels[1])
                for split, output in kernels.items()
            },
        }
        row["passed"] = all(
            delta["bit_exact"] for delta in row["kernel_vs_gather"].values()
        ) and all(delta["bit_exact"] for delta in row["split_invariance"].values())
        rows.append(row)
        del q, k, v, q_keys, q_values, gather, mirror, kernels
        mx.clear_cache()
        gc.collect()
    return {"phase": "exactness", "rows": rows}


def phase_timing(mx):
    from mlx_lm.models.qwen4_exp import (
        QSACompactBlocks,
        _gather_qsa_quantized_attention,
    )
    from mlx_lm.models.qwen4_qsa_indexed import (
        dequantize_qsa_quantized_kv,
        qwen4_qsa_indexed_attention,
        qwen4_qsa_indexed_quantized_attention,
    )

    load_wait = wait_for_load()
    rows = []
    for context in CONTEXTS:
        compact = compact_fixture(mx, QSACompactBlocks, context)
        mx.random.seed(20261000 + context)
        q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
        q_keys = mx.quantize(k, group_size=64, bits=8)
        q_values = mx.quantize(v, group_size=64, bits=8)
        dense_k, dense_v = dequantize_qsa_quantized_kv(
            q_keys, q_values, group_size=64, key_bits=8, value_bits=8
        )
        arms = {
            "gather_quantized": lambda: _gather_qsa_quantized_attention(
                q,
                q_keys,
                q_values,
                compact,
                scale=256**-0.5,
                tile_rows=1,
                group_size=64,
                key_bits=8,
                value_bits=8,
            ),
            "indexed_quantized": lambda: qwen4_qsa_indexed_quantized_attention(
                q,
                q_keys,
                q_values,
                compact,
                scale=256**-0.5,
                splits=8,
                group_size=64,
                key_bits=8,
                value_bits=8,
            ),
            "indexed_bf16": lambda: qwen4_qsa_indexed_attention(
                q, dense_k, dense_v, compact, scale=256**-0.5, splits=8
            ),
        }
        for function in arms.values():
            timed(mx, function)
        samples = {name: [] for name in arms}
        names = list(arms)
        for repeat in range(8):
            order = names[repeat % len(names) :] + names[: repeat % len(names)]
            for name in order:
                samples[name].append(timed(mx, arms[name]))
        packed_bytes = sum(x.nbytes for x in (*q_keys, *q_values))
        dense_bytes = dense_k.nbytes + dense_v.nbytes
        rows.append(
            {
                "context": context,
                "length": 3,
                "median_ms": {
                    name: statistics.median(values) * 1000.0
                    for name, values in samples.items()
                },
                "samples_ms": {
                    name: [value * 1000.0 for value in values]
                    for name, values in samples.items()
                },
                "capacity": {
                    "quantized_bytes_per_token": packed_bytes / context,
                    "bf16_bytes_per_token": dense_bytes / context,
                    "quantized_total_bytes": packed_bytes,
                },
                "safety_after": safety(f"timing_{context}"),
            }
        )
        require_safety(rows[-1]["safety_after"], floor=15)
        del q, k, v, q_keys, q_values, dense_k, dense_v
        mx.clear_cache()
        gc.collect()
    crossover = next(
        (
            row["context"]
            for row in rows
            if row["median_ms"]["indexed_quantized"]
            < row["median_ms"]["gather_quantized"]
        ),
        None,
    )
    return {"phase": "timing", "load_wait": load_wait, "rows": rows, "crossover": crossover}


def phase_model(mx, model_path, max_tokens):
    from mlx_lm.generate import maybe_quantize_kv_cache
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load
    from benchmarks.qwen4_qsa_indexed_gate import (
        clone_cache,
        corpus_tokens,
        model_qsa_layer_count,
        prefill_base,
        run_model_arm,
    )

    before = safety("before_model_load")
    require_safety(before, floor=45)
    model, tokenizer = load(str(model_path))
    model.eval()
    mx.eval(model.parameters())
    after_load = safety("after_model_load")
    require_safety(after_load, floor=15)
    prompt = corpus_tokens(tokenizer, CONTEXTS[0])
    base, token = prefill_base(mx, model, prompt, make_prompt_cache)
    maybe_quantize_kv_cache(base, 0, 64, 8)
    mx.eval([layer.state for layer in base])
    gather = run_model_arm(
        mx, model, token, clone_cache(base), mode="gather", max_tokens=max_tokens
    )
    indexed = run_model_arm(
        mx, model, token, clone_cache(base), mode="indexed", max_tokens=max_tokens
    )
    deltas = [
        abs(left - right)
        for left, right in zip(
            gather["chosen_logprobs"], indexed["chosen_logprobs"]
        )
    ]
    status = indexed["indexed_status"]
    qsa_layers = model_qsa_layer_count(model)
    rounds = indexed["stats"].get("draft_cycles", 0)
    engaged = status["query_width_counts"].get("2-8", {}).get("engaged", 0)
    result = {
        "phase": "model_16k_int8",
        "context": CONTEXTS[0],
        "max_tokens": max_tokens,
        "gather_digest": gather["digest"],
        "indexed_digest": indexed["digest"],
        "tokens_exact": gather["tokens"] == indexed["tokens"],
        "max_chosen_logprob_delta": max(deltas, default=0.0),
        "gather_stats": gather["stats"],
        "indexed_stats": indexed["stats"],
        "indexed_status": status,
        "receipt": {
            "qsa_layers": qsa_layers,
            "self_mtp_rounds": rounds,
            "expected_calls": qsa_layers * rounds,
            "engaged_calls": engaged,
            "fallbacks": status["fallbacks"],
        },
        "safety": [before, after_load, safety("after_model_16k")],
    }
    result["passed"] = (
        result["tokens_exact"]
        and result["max_chosen_logprob_delta"] == 0.0
        and engaged == qsa_layers * rounds
        and status["fallbacks"] == 0
    )
    del base, gather, indexed, model, tokenizer
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-wall-limit-minutes", type=float, default=30.0)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()
    if not args.execute_metal:
        raise SystemExit("refusing to dispatch Metal without --execute-metal")
    if not args.skip_model and args.model is None:
        raise SystemExit("--model is required unless --skip-model is set")

    os.environ["MLX_QWEN4_QSA_INDEXED"] = "1"
    os.environ["MLX_QWEN4_QSA_GATHER_KV"] = "1"
    os.environ["MLX_QWEN4_QSA_INDEXED_MIN_QUERY"] = "1"
    report = {
        "schema": "mlx-uag.qwen4-qsa-indexed-qkv-gate.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": _text(["git", "rev-parse", "HEAD"]),
        "policy": {
            "gpu_wall_limit_minutes": args.gpu_wall_limit_minutes,
            "memory_floor_percent": 15,
            "timing_load1_limit": 10,
            "timing_repeats": 8,
        },
        "phases": [],
    }
    started = time.monotonic()
    with gpu_lock() as owner:
        report["lock_owner"] = owner
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        baseline = safety("start")
        require_safety(baseline, floor=15)
        report["safety_start"] = baseline
        report["phases"].append(phase_exactness(mx))
        if time.monotonic() - started > args.gpu_wall_limit_minutes * 60:
            raise RuntimeError("GPU wall bound reached after exactness")
        report["phases"].append(phase_timing(mx))
        if not args.skip_model:
            if time.monotonic() - started > args.gpu_wall_limit_minutes * 60:
                raise RuntimeError("GPU wall bound reached before model phase")
            report["phases"].append(phase_model(mx, args.model, args.max_tokens))
        report["safety_end"] = safety("end")
    report["gpu_wall_seconds"] = time.monotonic() - started
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
