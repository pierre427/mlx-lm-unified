"""Attribute the first cached-tail Qwen4 target forward by operator family.

The profiled arm inserts an explicit Metal synchronization after every logical
family boundary.  Those timings are diagnostic attribution, not production
wall-clock measurements.  Ordinary unfenced arms bracket the probe so the
barrier tax and session drift remain visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

try:
    from benchmarks.qwen4_gdn_prefix_fanout_full_model_ab import (
        _cache_error,
        _clone_cache,
        _eval_cache,
        _prepare_cached_prefix,
        _prompt_tokens,
        _thermal_arm,
    )
    from benchmarks.qwen4_gdn_prep_matrix import (
        _mlx_memory,
        _swap_used_bytes,
        _system_free_percent,
    )
except ModuleNotFoundError:
    from qwen4_gdn_prefix_fanout_full_model_ab import (
        _cache_error,
        _clone_cache,
        _eval_cache,
        _prepare_cached_prefix,
        _prompt_tokens,
        _thermal_arm,
    )
    from qwen4_gdn_prep_matrix import (
        _mlx_memory,
        _swap_used_bytes,
        _system_free_percent,
    )

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.qwen4_ple_nvme import has_file_backed_ple
from mlx_lm.models import qwen3_next, qwen4_exp
from mlx_lm.models.precise_ops import gate_sigmoid
from mlx_lm.utils import load


def _eval_stage(*values) -> None:
    mx.eval(*values)
    mx.synchronize()


def _stage(rows, layer, family, fn, eval_fn=None):
    started = time.perf_counter_ns()
    value = fn()
    if eval_fn is None:
        _eval_stage(value)
    else:
        _eval_stage(*eval_fn(value))
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    rows.append(
        {
            "order": len(rows),
            "layer": layer,
            "family": family,
            "elapsed_ms": elapsed_ms,
        }
    )
    return value


def _profile_moe(rows, index, mlp, mixed):
    """Split the active stock-router/separate-shared MoE path.

    This deliberately fences branches that MLX may otherwise overlap. It is
    an attribution cut, never a production timing claim.
    """

    if mlp.sharding_group is not None:
        raise RuntimeError("detailed MoE profile does not support sharding")
    if mlp.shared_folded:
        raise RuntimeError("detailed MoE profile requires a separate shared expert")
    if mlp.moe_router_mode != "stock" or qwen3_next._MOE_GATE_COMPILE:
        raise RuntimeError("detailed MoE profile requires the stock eager router")
    if qwen3_next.compile_glue_enabled():
        raise RuntimeError("detailed MoE profile requires eager combine glue")

    gates = _stage(
        rows, index, "moe_router_projection", lambda: mlp.gate(mixed)
    )

    def select():
        probabilities = mx.softmax(gates, axis=-1, precise=True)
        inds = mx.argpartition(
            probabilities, kth=-mlp.top_k, axis=-1
        )[..., -mlp.top_k :]
        scores = mx.take_along_axis(probabilities, inds, axis=-1)
        if mlp.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    inds, scores = _stage(
        rows, index, "moe_router_select", select, lambda pair: pair
    )
    route_ids = mx.reshape(inds, (-1,)).tolist()
    rows[-1]["expert_assignments"] = len(route_ids)
    rows[-1]["unique_experts"] = len(set(route_ids))
    if mlp.fused_expert_kernel_enabled:
        routed = _stage(
            rows,
            index,
            "moe_routed_experts",
            lambda: mlp.switch_mlp(
                mixed,
                inds,
                scores=scores,
                variant=mlp.fused_expert_kernel_mode,
            ),
        )
    else:
        routed_rows = _stage(
            rows,
            index,
            "moe_routed_experts",
            lambda: mlp.switch_mlp(mixed, inds),
        )
        routed = _stage(
            rows,
            index,
            "moe_router_weight_reduce",
            lambda: (routed_rows * scores[..., None]).sum(axis=-2),
        )
    shared = _stage(
        rows,
        index,
        "moe_shared_expert",
        lambda: mlp.shared_expert(mixed),
    )
    shared_gate = _stage(
        rows,
        index,
        "moe_shared_gate",
        lambda: gate_sigmoid(mlp.shared_expert_gate(mixed)),
    )
    return _stage(
        rows,
        index,
        "moe_combine",
        lambda: routed + shared_gate * shared,
    )


def _profiled_backbone(model, inputs, cache, *, moe_detail=False):
    """Execute the stock target trunk with barriers at family boundaries."""

    body = model.language_model.model
    rows = []
    hidden = _stage(rows, None, "token_embedding", lambda: body.embed_tokens(inputs))
    hidden = _stage(
        rows,
        None,
        "hyper_stream_tile",
        lambda: mx.tile(hidden, (1, 1, body.args.hc_count)),
    )

    def build_masks():
        fa_mask = None
        if body.fa_idx is not None:
            fa_mask = create_attention_mask(
                hidden, cache[body.fa_idx], return_array=True
            )
            if fa_mask is not None and fa_mask.ndim == 2:
                fa_mask = fa_mask[None, None, :, :]
        ssm_mask = (
            create_ssm_mask(hidden, cache[body.ssm_idx])
            if body.ssm_idx is not None
            else None
        )
        return fa_mask, ssm_mask

    fa_mask, ssm_mask = _stage(
        rows,
        None,
        "mask_construction",
        build_masks,
        lambda pair: tuple(value for value in pair if value is not None),
    )

    for index, (layer, layer_cache) in enumerate(zip(body.layers, cache)):
        if layer.ple is not None:
            ple = layer.ple
            embeddings = _stage(
                rows,
                index,
                "ple_hash_nvme_lookup",
                lambda ple=ple: ple.ple_embedding(inputs, layer_cache, ssm_mask),
                lambda value, layer_cache=layer_cache: (value, layer_cache.state),
            )
            previous_conv = layer_cache[2]

            def ple_device():
                outputs = ple._run_device_chain(
                    hidden, embeddings, ssm_mask, previous_conv, True
                )
                layer_cache[2] = outputs[2]
                return hidden + outputs[0], outputs[1]

            hidden, _ = _stage(
                rows,
                index,
                "ple_device_chain_and_residual",
                ple_device,
                lambda pair, layer_cache=layer_cache: (
                    pair[0],
                    pair[1],
                    layer_cache.state,
                ),
            )

        mixed, residual, inject = _stage(
            rows,
            index,
            "attention_hyper_mix",
            lambda layer=layer: layer.attn_hyper_connection(hidden),
        )
        branch_family = "gdn" if layer.is_linear else "qsa_attention"
        branch = _stage(
            rows,
            index,
            branch_family,
            lambda layer=layer, mixed=mixed: (
                layer.linear_attn(mixed, ssm_mask, layer_cache)
                if layer.is_linear
                else layer.self_attn(mixed, fa_mask, layer_cache)
            ),
            lambda value, layer_cache=layer_cache: (value, layer_cache.state),
        )
        hidden = _stage(
            rows,
            index,
            "attention_inject",
            lambda residual=residual, branch=branch, inject=inject: (
                qwen4_exp._apply_inject(residual, branch, inject)
            ),
        )
        mixed, residual, inject = _stage(
            rows,
            index,
            "moe_hyper_mix",
            lambda layer=layer: layer.mlp_hyper_connection(hidden),
        )
        if moe_detail:
            branch = _profile_moe(rows, index, layer.mlp, mixed)
        else:
            branch = _stage(
                rows,
                index,
                "moe_router_experts_shared",
                lambda layer=layer, mixed=mixed: layer.mlp(mixed),
            )
        hidden = _stage(
            rows,
            index,
            "moe_inject",
            lambda residual=residual, branch=branch, inject=inject: (
                qwen4_exp._apply_inject(residual, branch, inject)
            ),
        )

    # The real catch-up caller discards the return_hyper ``mixed`` output and
    # evaluates only the per-stream hidden plus cache state.  The terminal HC
    # mixer graph is constructed by Qwen4ExpTextModel but remains dead here;
    # charging it would profile work the GPU does not execute on this leg.
    return hidden, rows


def _summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row["elapsed_ms"])
    total = sum(row["elapsed_ms"] for row in rows)
    families = []
    for family, values in grouped.items():
        family_total = sum(values)
        families.append(
            {
                "family": family,
                "calls": len(values),
                "total_ms": family_total,
                "share_of_profiled_wall": family_total / total if total else 0.0,
                "median_call_ms": statistics.median(values),
                "minimum_call_ms": min(values),
                "maximum_call_ms": max(values),
            }
        )
    return sorted(families, key=lambda item: item["total_ms"], reverse=True)


def _run_arm(model, cached, target_tokens, profiled, args):
    before_free = _system_free_percent()
    before_swap = _swap_used_bytes()
    if before_free is None or before_free < args.minimum_system_free_percent:
        raise RuntimeError(
            f"system free {before_free}% is below {args.minimum_system_free_percent}%"
        )
    clone_started = time.perf_counter_ns()
    cache = _clone_cache(cached["target"])
    mx.synchronize()
    clone_ms = (time.perf_counter_ns() - clone_started) / 1e6
    started = time.perf_counter_ns()
    if profiled:
        hidden, rows = _profiled_backbone(
            model, target_tokens, cache, moe_detail=args.moe_detail
        )
    else:
        _, hidden = model.mtp_backbone(target_tokens, cache)
        _eval_stage(hidden, [entry.state for entry in cache])
        rows = []
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    _eval_cache(cache)
    _eval_stage(hidden)
    after_swap = _swap_used_bytes()
    swap_growth = (
        None
        if before_swap is None or after_swap is None
        else after_swap - before_swap
    )
    if swap_growth is not None and swap_growth > args.maximum_swap_growth_mb << 20:
        raise RuntimeError(f"arm grew swap by {swap_growth / (1 << 20):.1f} MiB")
    return {
        "profiled": profiled,
        "clone_ms": clone_ms,
        "target_forward_ms": elapsed_ms,
        "ordered_stages": rows,
        "family_summary": _summarize_rows(rows) if rows else [],
        "hidden": hidden,
        "cache": cache,
        "memory": _mlx_memory(),
        "system_free_percent_before": before_free,
        "swap_used_bytes_before": before_swap,
        "swap_growth_bytes": swap_growth,
    }


def _serializable_arm(arm):
    return {
        key: value
        for key, value in arm.items()
        if key not in {"hidden", "cache"}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain exact cache transactions. ")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--cached-tail-tokens", type=int, default=8)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--share-qsa-indices", action="store_true", default=True)
    parser.add_argument("--minimum-system-free-percent", type=int, default=15)
    parser.add_argument("--maximum-swap-growth-mb", type=int, default=16)
    parser.add_argument("--minimum-cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--thermal-poll-seconds", type=float, default=5.0)
    parser.add_argument("--thermal-max-cooldown-seconds", type=float, default=600.0)
    parser.add_argument("--thermal-stable-snapshots", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--moe-detail", action="store_true")
    parser.add_argument(
        "--single-profile",
        action="store_true",
        help="Run one control and one profiled arm for diagnostic receipts.",
    )
    args = parser.parse_args()
    if args.prompt_tokens <= args.cached_tail_tokens:
        parser.error("cached tail must be shorter than the prompt")
    ple_path = os.environ.get("MLX_QWEN4_PLE_NVME")
    if not ple_path or not Path(ple_path).is_file():
        parser.error("MLX_QWEN4_PLE_NVME must name the file-backed PLE sidecar")
    if os.environ.get("MLX_LM_UBC_EVICT") != "1":
        parser.error("MLX_LM_UBC_EVICT=1 is required")

    model, tokenizer = load(args.model)
    model.eval()
    if not has_file_backed_ple(model):
        raise RuntimeError("loaded model did not activate file-backed PLE")
    prompt = _prompt_tokens(tokenizer, args.prompt, args.prompt_tokens)
    split = len(prompt) - args.cached_tail_tokens
    cached = _prepare_cached_prefix(model, prompt[:split], args)
    target_tokens = mx.array(prompt[split:-1], dtype=mx.uint32)[None]
    if target_tokens.shape[1] != args.cached_tail_tokens - 1:
        raise AssertionError("target catch-up width does not match the requested tail")

    arms = []
    pattern = (False, True) if args.single_profile else (False, True, True, False)
    for slot, profiled in enumerate(pattern):
        arm, thermal = _thermal_arm(
            args,
            "qwen4_first_target_family_profile",
            0,
            0,
            slot,
            profiled,
            lambda enabled: _run_arm(
                model, cached, target_tokens, bool(enabled), args
            ),
        )
        if arm is None:
            raise RuntimeError(f"thermal gate failed at slot {slot}: {thermal}")
        arms.append(arm)

    reference = arms[0]
    comparisons = []
    for slot, arm in enumerate(arms[1:], start=1):
        output_exact = bool(mx.array_equal(arm["hidden"], reference["hidden"]))
        cache_exact, cache_relative_max = _cache_error(
            arm["cache"], reference["cache"]
        )
        comparisons.append(
            {
                "slot": slot,
                "profiled": arm["profiled"],
                "output_bit_exact": output_exact,
                "cache_bit_exact": cache_exact,
                "cache_relative_max": cache_relative_max,
            }
        )
    if not all(
        row["output_bit_exact"] and row["cache_bit_exact"]
        for row in comparisons
    ):
        raise AssertionError(f"profiled path changed target state: {comparisons}")

    ordinary = [arm["target_forward_ms"] for arm in arms if not arm["profiled"]]
    profiled = [arm["target_forward_ms"] for arm in arms if arm["profiled"]]
    result = {
        "passed": True,
        "model": str(Path(args.model).resolve()),
        "geometry": {
            "prompt_tokens": len(prompt),
            "cached_prefix_tokens": split,
            "catchup_tokens": int(target_tokens.shape[1]),
            "layers": len(model.layers),
            "gdn_layers": sum(layer.is_linear for layer in model.layers),
            "qsa_layers": sum(not layer.is_linear for layer in model.layers),
            "ple_layers": sum(layer.ple is not None for layer in model.layers),
            "moe_detail": args.moe_detail,
        },
        "prefix_build_ms": cached["build_ms"],
        "ordinary_target_forward_ms": ordinary,
        "ordinary_median_ms": statistics.median(ordinary),
        "ordinary_closing_drift_fraction": abs(ordinary[-1] - ordinary[0])
        / min(ordinary),
        "profiled_target_forward_ms": profiled,
        "profiled_median_ms": statistics.median(profiled),
        "barrier_inflation_ratio": statistics.median(profiled)
        / statistics.median(ordinary),
        "comparisons": comparisons,
        "arms": [_serializable_arm(arm) for arm in arms],
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("MLX_")
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n")
    print(rendered)

    arms.clear()
    cached.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


if __name__ == "__main__":
    main()
