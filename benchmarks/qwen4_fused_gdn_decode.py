#!/usr/bin/env python3
"""Deferred correctness/performance gate for the Qwen4 fused GDN decode path.

The default invocation is plan-only and does not import MLX.  A real run needs
both ``--execute-metal`` and a local model directory so it cannot accidentally
download a checkpoint or launch the GPU while being inspected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PLAN = {
    "scope": "production Qwen4 GDN layer gate, then optional full target decode",
    "correctness_gate": "32 steps; output, conv cache, and recurrent state array_equal",
    "engagement_gate": "fused graph selected on every fused observation",
    "timing": "resident weights; interleaved stock/fused layer observations",
    "not_covered": ["fused prefill", "batch", "mask", "speculative MTP"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--execute-metal", action="store_true")
    parser.add_argument("--correctness-steps", type=int, default=32)
    parser.add_argument("--timing-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--full-model-steps", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="Explain one benefit of unified memory in a single sentence.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def clone_cache(cache, cache_type, mx):
    clone = cache_type(2)
    clone.cache = [mx.array(value) for value in cache.cache]
    return clone


def cache_equal(left, right, mx):
    return all(
        bool(mx.array_equal(a, b).item())
        for a, b in zip(left.cache, right.cache)
    )


def cache_arrays(caches):
    arrays = []
    for cache in caches:
        values = getattr(cache, "cache", None)
        if values is None:
            values = getattr(cache, "state", ())
        arrays.extend(
            value
            for value in values
            if value is not None and hasattr(value, "dtype")
        )
    return arrays


def all_gdn_caches_equal(left, right, mx):
    pairs = [
        (a, b)
        for left_cache, right_cache in zip(left, right)
        for a, b in zip(
            getattr(left_cache, "cache", ()),
            getattr(right_cache, "cache", ()),
        )
        if a is not None and b is not None
    ]
    return all(bool(mx.array_equal(a, b).item()) for a, b in pairs)


def run_full_model(model, tokenizer, args, mx, qwen4_api):
    set_mode, mode_counts, stats = qwen4_api
    tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not tokens:
        raise SystemExit("--prompt encoded to no tokens")
    prompt = mx.array([tokens], dtype=mx.uint32)
    stock_cache = model.make_cache()
    fused_cache = model.make_cache()
    set_mode(model, "stock")
    stock_logits = model(prompt, cache=stock_cache)
    fused_logits = model(prompt, cache=fused_cache)
    mx.eval(
        stock_logits,
        fused_logits,
        *cache_arrays(stock_cache),
        *cache_arrays(fused_cache),
    )
    if not bool(mx.array_equal(stock_logits, fused_logits).item()):
        return {"passed": False, "stage": "stock prefill reproducibility"}

    next_token = mx.argmax(stock_logits[:, -1, :], axis=-1).astype(mx.uint32)
    mx.eval(next_token)
    before = stats(model)
    timings = {"stock": [], "fused": []}
    mismatch = None
    for step in range(args.full_model_steps):
        token = next_token[:, None]
        outputs = {}
        order = ("stock", "fused") if step % 2 == 0 else ("fused", "stock")
        for mode in order:
            cache = stock_cache if mode == "stock" else fused_cache
            set_mode(model, mode)
            start = time.perf_counter()
            outputs[mode] = model(token, cache=cache)
            mx.eval(outputs[mode], *cache_arrays(cache))
            timings[mode].append(time.perf_counter() - start)

        logits_equal = bool(
            mx.array_equal(outputs["stock"], outputs["fused"]).item()
        )
        caches_equal = all_gdn_caches_equal(stock_cache, fused_cache, mx)
        if not logits_equal or not caches_equal:
            mismatch = {
                "step": step,
                "logits_equal": logits_equal,
                "gdn_caches_equal": caches_equal,
                "max_logit_abs": float(
                    mx.max(mx.abs(outputs["stock"] - outputs["fused"])).item()
                ),
            }
            break
        next_token = mx.argmax(
            outputs["stock"][:, -1, :], axis=-1
        ).astype(mx.uint32)
        mx.eval(next_token)

    after = stats(model)
    layer_count = sum(mode_counts(model).values())
    expected_calls = layer_count * args.full_model_steps
    call_delta = after["fused_calls"] - before["fused_calls"]
    fallback_delta = after["fallbacks"] - before["fallbacks"]
    engaged = call_delta == expected_calls and fallback_delta == 0
    medians = {
        mode: statistics.median(values) if values else None
        for mode, values in timings.items()
    }
    median_speedup = (
        100.0 * (medians["stock"] / medians["fused"] - 1.0)
        if medians["stock"] is not None and medians["fused"] is not None
        else None
    )
    aggregate = {mode: sum(values) for mode, values in timings.items()}
    aggregate_speedup = (
        100.0 * (aggregate["stock"] / aggregate["fused"] - 1.0)
        if aggregate["fused"] > 0
        else None
    )
    return {
        "passed": mismatch is None and engaged,
        "steps": args.full_model_steps,
        "prompt_tokens": len(tokens),
        "gdn_layers": layer_count,
        "mismatch": mismatch,
        "engaged": engaged,
        "fused_call_delta": call_delta,
        "fallback_delta": fallback_delta,
        "raw_seconds_per_token": timings,
        "median_seconds_per_token": medians,
        "median_speedup_percent": median_speedup,
        "aggregate_seconds": aggregate,
        "aggregate_speedup_percent": aggregate_speedup,
    }


def run(args):
    if args.model is None or not args.model.is_dir():
        raise SystemExit("--model must name an existing local checkpoint directory")
    if args.correctness_steps < 32:
        raise SystemExit("--correctness-steps may not be below the 32-step gate")

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    from mlx_lm.models.qwen4_exp import (
        GatedDeltaNet,
        Qwen4ArraysCache,
        qwen4_fused_gdn_mode_counts,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
    )
    from mlx_lm.utils import load

    model, tokenizer = load(str(args.model))
    model.eval()
    layers = [
        module
        for _, module in model.named_modules()
        if isinstance(module, GatedDeltaNet)
    ]
    if not layers:
        raise SystemExit("checkpoint did not instantiate Qwen4 GatedDeltaNet layers")
    layer = layers[0]

    key = mx.random.key(2105)
    hidden = mx.random.normal(
        (args.correctness_steps, 1, 1, layer.hidden_size), key=key
    ).astype(layer.dt_bias.dtype)
    warm = Qwen4ArraysCache(2)
    set_qwen4_fused_gdn_mode(layer, "stock")
    warm_out = layer(hidden[0], cache=warm)
    mx.eval(warm_out, *warm.cache)
    stock_cache = clone_cache(warm, Qwen4ArraysCache, mx)
    fused_cache = clone_cache(warm, Qwen4ArraysCache, mx)

    first_mismatch = None
    for step in range(args.correctness_steps):
        x = hidden[step]
        set_qwen4_fused_gdn_mode(layer, "stock")
        stock = layer(x, cache=stock_cache)
        mx.eval(stock, *stock_cache.cache)
        set_qwen4_fused_gdn_mode(layer, "fused")
        fused = layer(x, cache=fused_cache)
        mx.eval(fused, *fused_cache.cache)
        output_equal = bool(mx.array_equal(stock, fused).item())
        states_equal = cache_equal(stock_cache, fused_cache, mx)
        if not output_equal or not states_equal:
            first_mismatch = {
                "step": step,
                "output_equal": output_equal,
                "states_equal": states_equal,
                "max_output_abs": float(mx.max(mx.abs(stock - fused)).item()),
            }
            break

    stats = qwen4_fused_gdn_stats(layer)
    engaged = (
        stats["fused_calls"] == args.correctness_steps
        and stats["fallbacks"] == 0
    )
    correctness = {
        "passed": first_mismatch is None and engaged,
        "steps": args.correctness_steps,
        "first_mismatch": first_mismatch,
        "engaged": engaged,
        "stats": stats,
    }
    if not correctness["passed"]:
        return {
            "plan": PLAN,
            "correctness": correctness,
            "timing": None,
            "full_model": None,
        }

    timing_hidden = hidden[: min(args.timing_steps, args.correctness_steps)]
    timings = {"stock": [], "fused": []}
    orders = (("stock", "fused"), ("fused", "stock"))
    for repeat in range(args.repeats):
        for mode in orders[repeat % 2]:
            cache = clone_cache(warm, Qwen4ArraysCache, mx)
            mx.eval(*cache.cache)
            set_qwen4_fused_gdn_mode(layer, mode)
            start = time.perf_counter()
            out = None
            for step in range(args.timing_steps):
                out = layer(timing_hidden[step % len(timing_hidden)], cache=cache)
            mx.eval(out, *cache.cache)
            timings[mode].append(time.perf_counter() - start)

    medians = {mode: statistics.median(values) for mode, values in timings.items()}
    aggregate = {mode: sum(values) for mode, values in timings.items()}
    timing = {
        "raw_seconds": timings,
        "median_seconds": medians,
        "median_speedup_percent": 100.0
        * (medians["stock"] / medians["fused"] - 1.0),
        "aggregate_seconds": aggregate,
        "aggregate_speedup_percent": 100.0
        * (aggregate["stock"] / aggregate["fused"] - 1.0),
        "steps_per_observation": args.timing_steps,
    }
    full_model = None
    if args.full_model_steps > 0:
        full_model = run_full_model(
            model,
            tokenizer,
            args,
            mx,
            (
                set_qwen4_fused_gdn_mode,
                qwen4_fused_gdn_mode_counts,
                qwen4_fused_gdn_stats,
            ),
        )
    return {
        "plan": PLAN,
        "correctness": correctness,
        "timing": timing,
        "full_model": full_model,
    }


def main():
    args = parse_args()
    if not args.execute_metal:
        print(json.dumps({"status": "plan-only", "plan": PLAN}, indent=2))
        return 0
    result = run(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    passed = result["correctness"]["passed"] and (
        result.get("full_model") is None or result["full_model"]["passed"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
