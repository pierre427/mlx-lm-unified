#!/usr/bin/env python3
"""One-resident-model spot ablation for Qwen4 call-reduction compositions."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from qwen4_fused_gdn_context_ladder import (
    DEFAULT_MODEL,
    append_jsonl,
    build_prompt,
    cache_arrays,
    clone_cache,
    configure_common_stack,
    gdn_cache_arrays,
    gdn_caches_equal,
    memory_snapshot,
    parse_csv_ints,
    source_corpus,
    utc_now,
)


DEFAULT_OUTPUT = Path(
    "/Users/pierrelamy/Desktop/mlx-uag/results/"
    "qwen4-composition-ablation-20260901.jsonl"
)


VARIANTS = {
    "prior": ("fused", "stock", False),
    "router_only": ("fused", "fused", False),
    "gdn_outproj_only": ("fused_outproj", "stock", False),
    "gdn_router": ("fused_outproj", "fused", False),
    "direct_qsa_only": ("fused", "stock", True),
    "all": ("fused_outproj", "fused", True),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contexts", default="1024,8192,65536")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()
    contexts = parse_csv_ints(args.contexts)
    if not args.execute_metal:
        print(json.dumps({"contexts": contexts, "variants": VARIANTS}, indent=2))
        return 0

    environment = configure_common_stack(args.model)
    import mlx.core as mx
    from mlx_lm.generate import prefill_prompt_cache
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen3_5 import fuse_gated_delta_net_projections
    from mlx_lm.models.qwen3_next import (
        qwen4_moe_router_stats,
        set_qwen4_fused_expert_mode,
        set_qwen4_moe_router_mode,
    )
    from mlx_lm.models.qwen4_exp import (
        Qwen4ArraysCache,
        qsa_nax_decode_status,
        qsa_stage1_status,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
        set_qwen4_qsa_nax_decode,
    )
    from mlx_lm.utils import load

    mx.set_default_device(mx.gpu)
    append_jsonl(args.output, {
        "event": "start",
        "created_at_utc": utc_now(),
        "contexts": contexts,
        "steps": args.steps,
        "variants": VARIANTS,
        "environment": environment,
    })
    model, tokenizer = load(str(args.model))
    model.eval()
    fuse_gated_delta_net_projections(model, enabled=True)
    set_qwen4_fused_expert_mode(model, "tile4")
    mx.eval(model.parameters())

    def select(name: str) -> None:
        gdn, router, direct_qsa = VARIANTS[name]
        set_qwen4_fused_gdn_mode(model, gdn)
        set_qwen4_moe_router_mode(model, router)
        set_qwen4_qsa_nax_decode(direct_qsa)

    corpus = source_corpus(Path(__file__).resolve().parents[1])
    for context in contexts:
        prompt = build_prompt(tokenizer, corpus, context)
        prompt_array = mx.array(prompt, dtype=mx.uint32)
        cache = make_prompt_cache(model)
        qsa_stage1_status(reset=True)
        select("prior")
        started = time.perf_counter()
        prefill_prompt_cache(
            model,
            prompt_array[:-1],
            cache,
            prefill_step_size=args.prefill_step_size,
            progress_callback=lambda done, total, c=context: print(
                f"ctx={c} prefill={done}/{total}", flush=True
            ) if done == total or done % 8192 == 0 else None,
        )
        tail = model(prompt_array[-1:][None], cache=cache)
        mx.eval(tail, cache_arrays(cache))
        first = mx.argmax(tail[:, -1], axis=-1).astype(mx.uint32)
        mx.eval(first)
        stage1 = qsa_stage1_status()
        append_jsonl(args.output, {
            "event": "prefill",
            "created_at_utc": utc_now(),
            "context": context,
            "actual_prompt_tokens": len(prompt),
            "seconds": time.perf_counter() - started,
            "stage1": stage1,
            "memory": memory_snapshot(mx),
        })

        # Compile every graph outside timing.
        for name in VARIANTS:
            warm_cache = clone_cache(cache)
            select(name)
            out = model(first[:, None], cache=warm_cache)
            mx.eval(out, gdn_cache_arrays(warm_cache, Qwen4ArraysCache))

        caches = {name: clone_cache(cache) for name in VARIANTS}
        mx.eval([cache_arrays(value) for value in caches.values()])
        timings = {name: [] for name in VARIANTS}
        tokens_equal = {name: True for name in VARIANTS if name != "prior"}
        states_equal = {name: True for name in VARIANTS if name != "prior"}
        max_abs = {name: 0.0 for name in VARIANTS if name != "prior"}
        first_divergence = {name: None for name in VARIANTS if name != "prior"}
        token = first
        before = {
            "gdn": qwen4_fused_gdn_stats(model),
            "router": qwen4_moe_router_stats(model),
            "qsa": qsa_nax_decode_status(),
        }
        names = list(VARIANTS)
        for step in range(args.steps):
            outputs = {}
            order = names[step % len(names):] + names[:step % len(names)]
            for name in order:
                select(name)
                tick = time.perf_counter()
                out = model(token[:, None], cache=caches[name])
                mx.eval(out, gdn_cache_arrays(caches[name], Qwen4ArraysCache))
                timings[name].append(time.perf_counter() - tick)
                outputs[name] = out
            prior = outputs["prior"]
            prior_token = mx.argmax(prior[:, -1], axis=-1).astype(mx.uint32)
            mx.eval(prior_token)
            for name, out in outputs.items():
                if name == "prior":
                    continue
                same = int(mx.argmax(out[:, -1], axis=-1).item()) == int(
                    prior_token.item()
                )
                tokens_equal[name] &= same
                states_equal[name] &= gdn_caches_equal(
                    caches["prior"], caches[name], Qwen4ArraysCache, mx
                )
                diff = float(mx.max(mx.abs(prior - out)).item())
                max_abs[name] = max(max_abs[name], diff)
                if not same and first_divergence[name] is None:
                    first_divergence[name] = step
            token = prior_token

        after = {
            "gdn": qwen4_fused_gdn_stats(model),
            "router": qwen4_moe_router_stats(model),
            "qsa": qsa_nax_decode_status(),
        }
        medians = {name: statistics.median(ts) for name, ts in timings.items()}
        baseline = medians["prior"]
        result = {
            name: {
                "median_ms": medians[name] * 1000.0,
                "speedup_percent_vs_prior": 100.0 * (baseline / medians[name] - 1.0),
                "greedy_equal": True if name == "prior" else tokens_equal[name],
                "gdn_state_equal": True if name == "prior" else states_equal[name],
                "max_logit_abs": 0.0 if name == "prior" else max_abs[name],
                "first_greedy_divergence": None if name == "prior" else first_divergence[name],
            }
            for name in VARIANTS
        }
        append_jsonl(args.output, {
            "event": "ablation",
            "created_at_utc": utc_now(),
            "context": context,
            "steps": args.steps,
            "result": result,
            "stats_before": before,
            "stats_after": after,
            "memory": memory_snapshot(mx),
        })
        print(f"ctx={context}", flush=True)
        for name, values in result.items():
            print(
                f"  {name:18s} {values['median_ms']:.3f} ms "
                f"{values['speedup_percent_vs_prior']:+.2f}% "
                f"greedy={values['greedy_equal']}",
                flush=True,
            )
        del cache, caches, prompt_array, tail, first
        gc.collect()
        mx.clear_cache()

    append_jsonl(args.output, {
        "event": "complete",
        "created_at_utc": utc_now(),
        "memory": memory_snapshot(mx),
    })
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
