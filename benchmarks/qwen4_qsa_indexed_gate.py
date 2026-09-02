#!/usr/bin/env python3
"""Metal correctness and timing gate for indexed split-K QSA attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
CONTEXTS = (16_384, 32_768, 65_536, 131_072)
MODEL_CONTEXTS = CONTEXTS[:3]


@contextmanager
def gpu_lock():
    """Take the lab-wide GPU lock with atomic mkdir and exact cleanup."""

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
        "label": "qwen4-qsa-indexed-gate",
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        owner_path.write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        yield
    finally:
        owner_path.unlink(missing_ok=True)
        GPU_LOCK.rmdir()


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


def isolated_gate(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import (
        indexed_splits_for,
        qsa_indexed_status,
        qwen4_qsa_indexed_attention,
        qwen4_qsa_indexed_reference,
    )

    rows = []
    for context in CONTEXTS:
        compact = compact_fixture(mx, QSACompactBlocks, context)
        q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
        scale = 256**-0.5
        splits = indexed_splits_for(520)
        reference = qwen4_qsa_indexed_reference(
            q, k, v, compact, scale=scale, splits=splits
        )
        indexed_output = qwen4_qsa_indexed_attention(
            q, k, v, compact, scale=scale, splits=splits
        )
        gather = _gather_qsa_attention(
            q, k, v, compact, scale=scale, tile_rows=1
        )
        reference_abs, reference_rel = max_scaled_error(
            mx, indexed_output, reference
        )
        gather_abs, gather_rel = max_scaled_error(mx, indexed_output, gather)
        if reference_rel > 1.0e-5:
            raise AssertionError(
                f"{context}: indexed/reference relative error {reference_rel}"
            )

        dense_mask = dense_fixture_mask(mx, context)
        arms = {
            "indexed": lambda: qwen4_qsa_indexed_attention(
                q, k, v, compact, scale=scale, splits=splits
            ),
            "gather": lambda: _gather_qsa_attention(
                q, k, v, compact, scale=scale, tile_rows=1
            ),
            "dense": lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=dense_mask
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
        medians = {
            name: statistics.median(values) * 1000.0
            for name, values in samples.items()
        }
        rows.append(
            {
                "context": context,
                "splits": splits,
                "reference_max_abs": reference_abs,
                "reference_max_relative": reference_rel,
                "gather_max_abs": gather_abs,
                "gather_max_relative": gather_rel,
                "median_ms": medians,
                "status": qsa_indexed_status(),
            }
        )
    return rows


def corpus_tokens(tokenizer, context):
    text = (
        "Direct indexed attention keeps the selected KV rows in place. "
        "The verify gate checks rollback, exact token authority, and receipts. "
    )
    seed = tokenizer.encode(text)
    if not seed:
        raise ValueError("tokenizer produced an empty gate prompt")
    return (seed * (context // len(seed) + 1))[:context]


def run_model_arm(mx, model, prompt, cache, *, indexed_enabled, max_tokens):
    from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
    from mlx_lm.models import qwen4_exp
    from mlx_lm.models.qwen4_qsa_indexed import (
        qsa_indexed_status,
        set_qwen4_qsa_indexed,
    )

    set_qwen4_qsa_indexed(indexed_enabled)
    qwen4_exp._QSA_GATHER_KV = True
    qwen4_exp._QSA_GATHER_MIN_QUERY = 1
    qwen4_exp._QSA_GATHER_MAX_QUERY = 8
    qwen4_exp._QSA_GATHER_MIN_CONTEXT = 0
    qsa_indexed_status(reset=True)
    stats = HybridStats()
    tokens = []
    chosen_logprobs = []
    for token, logprobs, _ in self_mtp_generate_step(
        mx.array(prompt, dtype=mx.uint32),
        model,
        num_draft=2,
        max_tokens=max_tokens,
        persistent_mtp=True,
        prompt_cache=cache,
        stats=stats,
    ):
        token = int(token)
        tokens.append(token)
        chosen_logprobs.append(float(logprobs[token].item()))
    digest = hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in tokens)
    ).hexdigest()
    return {
        "tokens": tokens,
        "digest": digest,
        "chosen_logprobs": chosen_logprobs,
        "stats": stats.__dict__,
        "indexed_status": qsa_indexed_status(),
    }


def model_gate(mx, model_path, max_tokens):
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load

    model, tokenizer = load(str(model_path))
    model.eval()
    mx.eval(model.parameters())
    rows = []
    for context in MODEL_CONTEXTS:
        prompt = corpus_tokens(tokenizer, context)
        base = make_prompt_cache(model)
        gather = run_model_arm(
            mx,
            model,
            prompt,
            clone_cache(base),
            indexed_enabled=False,
            max_tokens=max_tokens,
        )
        indexed = run_model_arm(
            mx,
            model,
            prompt,
            clone_cache(base),
            indexed_enabled=True,
            max_tokens=max_tokens,
        )
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
        if gather["digest"] != indexed["digest"]:
            raise AssertionError(f"{context}: greedy digest differs")
        if max_logprob_delta > 0.002:
            raise AssertionError(
                f"{context}: chosen-logprob delta {max_logprob_delta} exceeds 0.002"
            )
        if status["fallbacks"] or not reached:
            raise AssertionError(
                f"{context}: indexed verify did not run cleanly: {status}"
            )
        rows.append(
            {
                "context": context,
                "greedy_digest": indexed["digest"],
                "max_chosen_logprob_delta": max_logprob_delta,
                "gather_stats": gather["stats"],
                "indexed_stats": indexed["stats"],
                "indexed_status": status,
                "cache_boundary": "both arms cloned from one pre-run cache",
                "draft_contract": "self-MTP k=2 produces width-3 rollback-recording verify",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--isolated-only", action="store_true")
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()
    if not args.execute_metal:
        raise SystemExit("refusing to dispatch Metal without --execute-metal")
    if not args.isolated_only and args.model is None:
        parser.error("--model is required unless --isolated-only is set")

    os.environ["MLX_QWEN4_QSA_INDEXED"] = "1"
    os.environ["MLX_QWEN4_QSA_GATHER_KV"] = "1"
    os.environ["MLX_QWEN4_QSA_INDEXED_MIN_QUERY"] = "1"
    with gpu_lock():
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "isolated": isolated_gate(mx),
            "model": None,
        }
        if not args.isolated_only:
            report["model"] = model_gate(mx, args.model, args.max_tokens)
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.output is None:
            print(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
            print(args.output)


if __name__ == "__main__":
    main()
