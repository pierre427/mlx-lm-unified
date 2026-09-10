"""The per-layer megakernel prize, measured with zero kernel code.

A decode step on Qwen3.8-Flash-Next issues hundreds of small kernels and runs
at a small fraction of the machine's bandwidth (see
``wiki/docs/experiments/qwen4-flash-next-decode-budget-2026-09-02.md``). The
end state of every fusion lever in this tree is ONE kernel per layer, which
would read each layer's bytes exactly once in one big streaming read.

That end state has a measurable ceiling that needs no kernel written for it:

  prize = (time the real layer takes) / (time ONE quantized matmul takes
           that reads the same number of bytes)

The denominator is achievable today -- it is just a matmul -- so the ratio is
an upper bound on what fusing a layer to a single dispatch could buy, and the
gap between layer types says which layer to fuse first.

Byte accounting is per-layer-call, not per-model: quantized weights plus their
scales and biases, the recurrent/convolution state a GDN layer reads and
writes, and for a MoE layer only the ``top_k`` experts a token actually
gathers. Activations at M<=3 are noise beside those and are counted anyway.

Plan-only by default; ``--execute`` loads the model and runs on the GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


# Synthetic floor geometry. K and the group size match the checkpoint's GDN
# input projections so the floor matmul is drawn from the same kernel family
# the real layer uses, not a differently-shaped one that happens to move the
# same bytes.
FLOOR_K = 2560
FLOOR_GROUP = 32
FLOOR_BITS = 4

# Distinct activations to cycle through. Must be at least ``reps`` or the batch
# contains repeated identical ops that common-subexpression elimination is free
# to collapse, which understates the time exactly like the dead-node bug did.
_DISTINCT_INPUTS = 64


def _bytes_per_output_row(k: int, group: int, bits: int) -> int:
    """Weight + scale + bias bytes one output row of a quantized table costs."""
    return k * bits // 8 + 2 * (k // group) * 2


def _param_bytes(module: nn.Module, *, skip=()) -> int:
    total = 0
    for name, value in module.parameters().items():
        if name in skip:
            continue
        total += _tree_bytes(value)
    return total


def _tree_bytes(value: Any) -> int:
    if isinstance(value, mx.array):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_tree_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_bytes(v) for v in value)
    return 0


def gdn_layer_bytes(layer, cache) -> dict[str, int]:
    """Bytes one GDN layer call reads and writes.

    Every parameter is read once per call. The convolution and recurrent
    states are read AND written, so they count twice -- at M=1 the recurrent
    state is the single largest non-weight term and hiding half of it would
    flatter the prize.
    """
    weights = _param_bytes(layer)
    state = 0
    if cache is not None and cache[0] is not None:
        state = 2 * (cache[0].nbytes + cache[1].nbytes)
    return {"weights": weights, "state": state, "total": weights + state}


def moe_layer_bytes(block) -> dict[str, int]:
    """Bytes one MoE block call reads for a single token.

    Only ``top_k`` of ``num_experts`` routed expert row-blocks are gathered,
    so the routed term is scaled by that fraction; the router, the shared
    expert and its gate are dense reads.
    """
    routed = _param_bytes(block.switch_mlp)
    experts = block.num_experts + (1 if block.shared_folded else 0)
    per_expert = routed / experts
    gathered = per_expert * (block.top_k + (1 if block.shared_folded else 0))
    dense = _param_bytes(block.gate) + _param_bytes(block.shared_expert_gate)
    if not block.shared_folded:
        dense += _param_bytes(block.shared_expert)
    return {
        "routed_gathered": int(gathered),
        "dense": int(dense),
        "total": int(gathered + dense),
    }


def attention_layer_bytes(layer, context: int) -> dict[str, int]:
    """Bytes one attention layer call reads: weights plus the KV it attends.

    The KV term is the DENSE upper bound (every cached position). Sparse QSA
    selection reads less, so a sparse layer's real prize is larger than the
    one this reports.
    """
    weights = _param_bytes(layer)
    per_pos = 2 * layer.num_kv_heads * layer.head_dim * 2
    return {
        "weights": weights,
        "kv": per_pos * context,
        "total": weights + per_pos * context,
    }


def _floor_table(target_bytes: int):
    """A 4-bit quantized table sized to read ``target_bytes`` in one matmul."""
    per_row = _bytes_per_output_row(FLOOR_K, FLOOR_GROUP, FLOOR_BITS)
    rows = max(1, round(target_bytes / per_row))
    weight = mx.random.randint(
        0, 2**31 - 1, (rows, FLOOR_K * FLOOR_BITS // 32), dtype=mx.uint32
    )
    scales = (
        mx.random.normal((rows, FLOOR_K // FLOOR_GROUP)) * 0.01
    ).astype(mx.bfloat16)
    biases = (
        mx.random.normal((rows, FLOOR_K // FLOOR_GROUP)) * 0.01
    ).astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    actual = weight.nbytes + scales.nbytes + biases.nbytes
    return (weight, scales, biases), rows, actual


def _time(fn, *, reps: int, trials: int) -> float:
    """Median seconds per call of ``fn(i)`` over ``trials`` batches of ``reps``.

    One ``mx.eval`` per batch, not per call: the thing being compared is how
    long a stream of these dispatches takes, and syncing every call would
    charge each one a host round trip the real decode step does not pay.

    EVERY output is held and evaluated together. Keeping only the last one --
    the first draft of this function -- makes the other ``reps - 1`` calls dead
    graph nodes that MLX never executes, so the batch measures one call and
    reports it as ``reps``. That produced floor rates of 1,380 and 5,608 GB/s
    on a machine whose DRAM tops out near 600, which is the only reason the bug
    was caught: a stopwatch that cannot be checked against a physical ceiling
    would have reported those numbers as a result.
    """
    for i in range(3):
        mx.eval(fn(i))
    samples = []
    for trial in range(trials):
        mx.synchronize()
        start = time.perf_counter()
        outputs = [fn(trial * reps + i) for i in range(reps)]
        mx.eval(outputs)
        mx.synchronize()
        samples.append((time.perf_counter() - start) / reps)
        del outputs
    return statistics.median(samples)


def _floor_time(target_bytes: int, rows_m: int, *, reps: int, trials: int):
    (weight, scales, biases), rows, actual = _floor_table(target_bytes)
    inputs = [
        (mx.random.normal((1, rows_m, FLOOR_K), key=mx.random.key(i)) * 0.1).astype(
            mx.bfloat16
        )
        for i in range(_DISTINCT_INPUTS)
    ]
    mx.eval(inputs)

    def call(i):
        return mx.quantized_matmul(
            inputs[i % len(inputs)],
            weight,
            scales,
            biases,
            transpose=True,
            group_size=FLOOR_GROUP,
            bits=FLOOR_BITS,
        )

    seconds = _time(call, reps=reps, trials=trials)
    del weight, scales, biases, inputs
    mx.clear_cache()
    return seconds, {"rows": rows, "table_bytes": actual}


def run(args, model=None) -> dict[str, Any]:
    from mlx_lm.models import qwen4_exp

    if model is None:
        from mlx_lm.utils import load

        model, _ = load(args.model)
        model.eval()

    text = model.language_model.model if hasattr(model, "language_model") else model.model
    layers = text.layers

    gdn_layer = moe_block = attn_layer = None
    for layer in layers:
        if gdn_layer is None and isinstance(
            getattr(layer, "linear_attn", None), qwen4_exp.GatedDeltaNet
        ):
            gdn_layer = layer.linear_attn
        if attn_layer is None and getattr(layer, "self_attn", None) is not None:
            attn_layer = layer.self_attn
        block = getattr(layer, "mlp", None)
        if moe_block is None and hasattr(block, "switch_mlp"):
            moe_block = block
    if gdn_layer is None:
        raise SystemExit("no GatedDeltaNet layer found")

    report: dict[str, Any] = {
        "model": args.model,
        "mlx": mx.__version__,
        "reps": args.reps,
        "trials": args.trials,
        "rows": {},
    }

    hidden = gdn_layer.hidden_size
    for rows_m in args.widths:
        entries = {}

        # --- GDN -------------------------------------------------------
        from mlx_lm.models.cache import ArraysCache

        cache = ArraysCache(size=2)
        warm = (mx.random.normal((1, 64, hidden), key=mx.random.key(1)) * 0.3).astype(
            mx.bfloat16
        )
        gdn_layer(warm, None, cache)
        mx.eval(cache[0], cache[1])
        gdn_bytes = gdn_layer_bytes(gdn_layer, cache)
        gdn_inputs = [
            (
                mx.random.normal((1, rows_m, hidden), key=mx.random.key(100 + i)) * 0.3
            ).astype(mx.bfloat16)
            for i in range(_DISTINCT_INPUTS)
        ]
        mx.eval(gdn_inputs)

        def gdn_call(i, cache=cache):
            return gdn_layer(gdn_inputs[i % len(gdn_inputs)], None, cache)

        gdn_real = _time(gdn_call, reps=args.reps, trials=args.trials)
        gdn_floor, gdn_floor_meta = _floor_time(
            gdn_bytes["total"], rows_m, reps=args.reps, trials=args.trials
        )
        entries["gdn"] = {
            "bytes": gdn_bytes,
            "real_us": gdn_real * 1e6,
            "floor_us": gdn_floor * 1e6,
            "prize": gdn_real / gdn_floor,
            "floor_table": gdn_floor_meta,
            "real_achieved_gbs": gdn_bytes["total"] / gdn_real / 1e9,
            "floor_achieved_gbs": gdn_floor_meta["table_bytes"] / gdn_floor / 1e9,
        }

        # --- MoE -------------------------------------------------------
        if moe_block is not None:
            moe_bytes = moe_layer_bytes(moe_block)
            moe_inputs = [
                (
                    mx.random.normal((1, rows_m, hidden), key=mx.random.key(200 + i))
                    * 0.3
                ).astype(mx.bfloat16)
                for i in range(_DISTINCT_INPUTS)
            ]
            mx.eval(moe_inputs)

            def moe_call(i):
                return moe_block(moe_inputs[i % len(moe_inputs)])

            moe_real = _time(moe_call, reps=args.reps, trials=args.trials)
            moe_floor, moe_floor_meta = _floor_time(
                moe_bytes["total"], rows_m, reps=args.reps, trials=args.trials
            )
            entries["moe"] = {
                "bytes": moe_bytes,
                "real_us": moe_real * 1e6,
                "floor_us": moe_floor * 1e6,
                "prize": moe_real / moe_floor,
                "floor_table": moe_floor_meta,
                "real_achieved_gbs": moe_bytes["total"] / moe_real / 1e9,
                "floor_achieved_gbs": moe_floor_meta["table_bytes"] / moe_floor / 1e9,
            }

        # --- Attention (weights only; KV is the caller's context) --------
        if attn_layer is not None and args.attention_context:
            attn_bytes = attention_layer_bytes(attn_layer, args.attention_context)
            attn_floor, attn_floor_meta = _floor_time(
                attn_bytes["total"], rows_m, reps=args.reps, trials=args.trials
            )
            entries["attention_weights_only_floor"] = {
                "bytes": attn_bytes,
                "floor_us": attn_floor * 1e6,
                "floor_table": attn_floor_meta,
                "note": (
                    "floor only: the real QSA layer needs a populated KV cache "
                    "and sparse selection, which this harness does not build"
                ),
            }

        report["rows"][str(rows_m)] = entries
        del gdn_inputs
        mx.clear_cache()

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--reps", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--attention-context", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="load the model and run on the GPU (default is plan-only)",
    )
    args = parser.parse_args()

    if not args.execute:
        print(__doc__)
        print("Plan only. Re-run with --execute to load the model and measure.")
        return

    report = run(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
