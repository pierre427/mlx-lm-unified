"""Measure whether lagged target-layer routes are useful expert hints.

This is an analysis harness, not a serving implementation.  It patches the
active switch-layer call only for the lifetime of the process, retains the
lazy route arrays, and evaluates them with the already-required target/cache
boundary.  It deliberately never reads a route from the host inside a layer.
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
    from benchmarks.qwen4_gdn_prefix_fanout_full_model_ab import _eval_cache
    from benchmarks.qwen4_gdn_prep_matrix import (
        _mlx_memory,
        _swap_used_bytes,
        _system_free_percent,
    )
except ModuleNotFoundError:
    from qwen4_gdn_prefix_fanout_full_model_ab import _eval_cache
    from qwen4_gdn_prep_matrix import (
        _mlx_memory,
        _swap_used_bytes,
        _system_free_percent,
    )

from mlx_lm.models import qwen3_next
from mlx_lm.models.qwen4_ple_nvme import has_file_backed_ple
from mlx_lm.utils import load


EXPERT_BYTES = 2_764_800


class RouteCapture:
    """Process-local call wrapper that adds no device-to-host dependency."""

    def __init__(self, tags):
        self.tags = tags
        self.rows = []
        self.enabled = False
        self._originals = {}

    def __enter__(self):
        for cls in (qwen3_next.FusedGateUpSwitchGLU, qwen3_next.FusedDownSwitchGLU):
            original = cls.__call__
            self._originals[cls] = original
            capture = self

            def wrapped(layer, x, indices, scores=None, variant="scalar", *, _original=original):
                tag = capture.tags.get(id(layer))
                if capture.enabled and tag is not None:
                    capture.rows.append((tag, indices, scores))
                return _original(layer, x, indices, scores=scores, variant=variant)

            cls.__call__ = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback):
        for cls, original in self._originals.items():
            cls.__call__ = original


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _route_tags(model):
    tags = {}
    for layer_index, layer in enumerate(model.layers):
        tags[id(layer.mlp.switch_mlp)] = f"target:{layer_index}"
    if hasattr(model, "mtp"):
        for layer_index, layer in enumerate(model.mtp.layers):
            tags[id(layer.mlp.switch_mlp)] = f"mtp:{layer_index}"
    return tags


def _eval_boundary(hidden, cache, captured):
    values = [hidden]
    for _, indices, scores in captured:
        values.append(indices)
        if scores is not None:
            values.append(scores)
    mx.eval(*values)
    _eval_cache(cache)
    mx.synchronize()


def _collapse(indices, scores):
    ids = indices.tolist()
    weights = scores.tolist() if scores is not None else None
    totals = defaultdict(float)
    maxima = defaultdict(float)
    for query, row in enumerate(ids[0]):
        for slot, expert in enumerate(row):
            weight = 1.0 if weights is None else float(weights[0][query][slot])
            expert = int(expert)
            totals[expert] += weight
            maxima[expert] = max(maxima[expert], weight)
    ranked = sorted(totals, key=lambda expert: (-totals[expert], expert))
    return {
        "union": set(totals),
        "totals": dict(totals),
        "maxima": dict(maxima),
        "ranked": ranked,
    }


def _prefill(model, tokens, cache, step):
    for start in range(0, len(tokens), step):
        chunk = mx.array(tokens[start : start + step], dtype=mx.uint32)[None]
        _, hidden = model.mtp_backbone(chunk, cache)
        mx.eval(hidden)
        _eval_cache(cache)
        mx.synchronize()


def _collect_document(model, tokenizer, path, args, capture):
    text = path.read_text(errors="replace")
    tokens = _encode(tokenizer, text)
    needed = args.prefix_tokens + args.cycles * args.slab_tokens
    if len(tokens) < needed:
        raise ValueError(f"{path} has {len(tokens)} tokens; {needed} are required")
    cache = model.make_cache()
    capture.enabled = False
    _prefill(model, tokens[: args.prefix_tokens], cache, args.prefill_step_size)
    cycles = []
    cursor = args.prefix_tokens
    for cycle in range(args.cycles):
        slab = tokens[cursor : cursor + args.slab_tokens]
        cursor += args.slab_tokens
        capture.rows.clear()
        capture.enabled = True
        started = time.perf_counter_ns()
        _, hidden = model.mtp_backbone(
            mx.array(slab, dtype=mx.uint32)[None], cache
        )
        rows = list(capture.rows)
        _eval_boundary(hidden, cache, rows)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        capture.enabled = False
        target = {}
        for tag, indices, scores in rows:
            if tag.startswith("target:"):
                target[int(tag.split(":", 1)[1])] = _collapse(indices, scores)
        if len(target) != len(model.layers):
            raise AssertionError(
                f"captured {len(target)} target layers, expected {len(model.layers)}"
            )
        cycles.append({"cycle": cycle, "elapsed_ms": elapsed_ms, "layers": target})
    del cache
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return {
        "path": str(path.resolve()),
        "token_count": len(tokens),
        "cycles": cycles,
    }


def _train_static(documents, layers):
    scores = {layer: defaultdict(float) for layer in range(layers)}
    for document in documents:
        for cycle in document["cycles"]:
            for layer, route in cycle["layers"].items():
                for expert, weight in route["totals"].items():
                    scores[layer][expert] += weight
    return {
        layer: sorted(scores[layer], key=lambda expert: (-scores[layer][expert], expert))
        for layer in range(layers)
    }


def _score(documents, static, budgets, layers):
    accum = {
        name: {budget: defaultdict(float) for budget in budgets}
        for name in ("lag_union", "lag_top10", "static")
    }
    counts = {name: {budget: 0 for budget in budgets} for name in accum}
    jaccards = []
    cardinalities = []
    for document in documents:
        cycles = document["cycles"]
        for index in range(1, len(cycles)):
            previous = cycles[index - 1]
            current = cycles[index]
            for layer in range(layers):
                prior = previous["layers"][layer]
                actual = current["layers"][layer]
                actual_set = actual["union"]
                cardinalities.append(len(actual_set))
                union = actual_set | prior["union"]
                jaccards.append(len(actual_set & prior["union"]) / len(union))
                top_ids = previous["layers"][layer]["ranked"][:10]
                candidates = {
                    "lag_union": prior["ranked"],
                    "lag_top10": top_ids,
                    "static": static[layer],
                }
                total_mass = sum(actual["totals"].values())
                for name, ranked in candidates.items():
                    for budget in budgets:
                        predicted = set(ranked[:budget])
                        useful = predicted & actual_set
                        row = accum[name][budget]
                        row["recall"] += len(useful) / len(actual_set)
                        row["precision"] += len(useful) / max(len(predicted), 1)
                        row["mass_recall"] += sum(
                            actual["totals"][expert] for expert in useful
                        ) / total_mass
                        row["predicted_experts"] += len(predicted)
                        row["useful_experts"] += len(useful)
                        counts[name][budget] += 1
    results = {}
    for name in accum:
        results[name] = {}
        for budget in budgets:
            count = counts[name][budget]
            row = accum[name][budget]
            predicted = row["predicted_experts"]
            useful = row["useful_experts"]
            results[name][str(budget)] = {
                "samples": count,
                "mean_set_recall": row["recall"] / count,
                "mean_precision": row["precision"] / count,
                "mean_score_mass_recall": row["mass_recall"] / count,
                "predicted_bytes": int(predicted * EXPERT_BYTES),
                "useful_bytes": int(useful * EXPERT_BYTES),
                "wasted_bytes": int((predicted - useful) * EXPERT_BYTES),
            }
    return {
        "mean_route_union_cardinality": statistics.mean(cardinalities),
        "mean_lag1_jaccard": statistics.mean(jaccards),
        "predictors": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-document", action="append", type=Path, required=True)
    parser.add_argument("--test-document", action="append", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--slab-tokens", type=int, default=7)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--budgets", default="8,16,24,32,48,64")
    parser.add_argument("--minimum-system-free-percent", type=float, default=15.0)
    parser.add_argument("--maximum-swap-growth-mb", type=int, default=16)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.slab_tokens != 7:
        parser.error("this probe is intentionally fixed to S=7")
    ple_path = os.environ.get("MLX_QWEN4_PLE_NVME")
    if not ple_path or not Path(ple_path).is_file():
        parser.error("MLX_QWEN4_PLE_NVME must name the file-backed PLE sidecar")
    if os.environ.get("MLX_LM_UBC_EVICT") != "1":
        parser.error("MLX_LM_UBC_EVICT=1 is required")
    free_before = _system_free_percent()
    if free_before is None or free_before < args.minimum_system_free_percent:
        raise RuntimeError(f"system free memory is only {free_before}%")
    swap_before = _swap_used_bytes()

    model, tokenizer = load(args.model)
    model.eval()
    if not has_file_backed_ple(model):
        raise RuntimeError("loaded model did not activate file-backed PLE")
    tags = _route_tags(model)
    with RouteCapture(tags) as capture:
        train = [
            _collect_document(model, tokenizer, path, args, capture)
            for path in args.train_document
        ]
        test = [
            _collect_document(model, tokenizer, path, args, capture)
            for path in args.test_document
        ]
    layers = len(model.layers)
    static = _train_static(train, layers)
    budgets = tuple(int(value) for value in args.budgets.split(","))
    metrics = _score(test, static, budgets, layers)
    swap_after = _swap_used_bytes()
    swap_growth = None if swap_before is None or swap_after is None else swap_after - swap_before
    if swap_growth is not None and swap_growth > args.maximum_swap_growth_mb << 20:
        raise RuntimeError(f"swap grew by {swap_growth / (1 << 20):.1f} MiB")
    result = {
        "passed": True,
        "scope": "route-predictor signal only; no prefetch wall-time claim",
        "model": str(Path(args.model).resolve()),
        "geometry": {
            "layers": layers,
            "prefix_tokens": args.prefix_tokens,
            "cycles_per_document": args.cycles,
            "slab_tokens": args.slab_tokens,
            "expert_bytes": EXPERT_BYTES,
        },
        "train_documents": [
            {"path": row["path"], "token_count": row["token_count"]} for row in train
        ],
        "test_documents": [
            {"path": row["path"], "token_count": row["token_count"]} for row in test
        ],
        "metrics": metrics,
        "document_cycle_ms": {
            row["path"]: [cycle["elapsed_ms"] for cycle in row["cycles"]]
            for row in train + test
        },
        "memory": {
            "system_free_percent_before": free_before,
            "system_free_percent_after": _system_free_percent(),
            "swap_growth_bytes": swap_growth,
            "mlx": _mlx_memory(),
        },
        "environment": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("MLX_")
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
