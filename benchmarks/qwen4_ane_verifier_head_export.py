#!/usr/bin/env python3
"""Probe an ANE-resident compact vocabulary projection followed by top-k.

This is deliberately a topology/transaction probe, not a model benchmark.  It
uses deterministic synthetic or selected real FP16 weights at the Qwen4
hidden width.  Only compact indices (and, for chunked reduction, their tiny
score table) cross back to the host; a production caller would map them
through the row sidecar and leave the full target head as commit authority.

The script uses Core ML with ``CPU_AND_NE`` and never imports MLX or submits
Metal work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compute_plan_receipt(model):
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=model.compute_unit
    )
    operations = plan.model_structure.program.functions["main"].block.operations
    preferred = Counter()
    operator_counts = Counter()
    rows = []
    for operation in operations:
        operator_counts[operation.operator_name] += 1
        operator_base = operation.operator_name.rsplit(".", 1)[-1]
        if operator_base in {"const", "cast"} or operator_base.startswith("constexpr_"):
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        preferred_name = type(usage.preferred_compute_device).__name__
        preferred[preferred_name] += 1
        rows.append(
            {
                "operator": operation.operator_name,
                "preferred": preferred_name,
                "supported": [
                    type(device).__name__ for device in usage.supported_compute_devices
                ],
            }
        )
    return {
        "preferred_counts": dict(preferred),
        "operator_counts": dict(operator_counts),
        "operations": rows,
    }


def non_ane_compute_operations(plan):
    return [
        row
        for row in plan["operations"]
        if row["operator"].rsplit(".", 1)[-1] not in {"const", "cast"}
        and not row["operator"].rsplit(".", 1)[-1].startswith("constexpr_")
        and row["preferred"] != "MLNeuralEngineComputeDevice"
    ]


def non_ane_required_operations(plan):
    required = {"linear", "topk"}
    return [
        row
        for row in plan["operations"]
        if row["operator"].rsplit(".", 1)[-1] in required
        and row["preferred"] != "MLNeuralEngineComputeDevice"
    ]


def make_weight(rows, hidden_size, seed):
    rng = np.random.default_rng(seed)
    # Bound the activation range and avoid giant float32 intermediates.
    weight = rng.integers(-128, 128, size=(rows, hidden_size), dtype=np.int16)
    return (weight.astype(np.float16) / np.float16(256.0)).astype(np.float16)


def load_real_compact_weight(model_dir, rows_sidecar, row_limit=None):
    """Dequantize selected affine-q4 LM-head rows on CPU for Core ML export."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    quantization = config["quantization"]
    bits = int(quantization["bits"])
    group_size = int(quantization["group_size"])
    mode = quantization.get("mode", "affine")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "language_model.lm_head."
    weight_path = model_dir / index[prefix + "weight"]
    tensors = mx.load(str(weight_path))
    rows = mx.load(str(rows_sidecar))["rows"].astype(mx.uint32)
    if row_limit is not None:
        rows = rows[:row_limit]
    selected = [
        mx.take(tensors[prefix + suffix], rows, axis=0)
        for suffix in ("weight", "scales", "biases")
    ]
    weight = mx.dequantize(
        *selected,
        group_size=group_size,
        bits=bits,
        mode=mode,
    ).astype(mx.float16)
    mx.eval(weight, rows)
    return (
        np.asarray(weight),
        np.asarray(rows, dtype=np.uint32),
        {
            "kind": "selected_real_lm_head",
            "model": str(model_dir.resolve()),
            "weight_shard": str(weight_path),
            "rows_sidecar": str(Path(rows_sidecar).resolve()),
            "quantization": {
                "bits": bits,
                "group_size": group_size,
                "mode": mode,
            },
        },
    )


def make_program(weight, top_k, chunk_rows=None, batch_size=1, valid_rows=None):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    hidden_size = int(weight.shape[1])

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch_size, 1, hidden_size), dtype=types.fp16)
        ],
        opset_version=ct.target.iOS18,
    )
    def program(hidden):
        flat = mb.reshape(x=hidden, shape=(batch_size, hidden_size))
        if chunk_rows is None:
            logits = mb.linear(x=flat, weight=weight, name="verifier_lm_head")
            _values, indices = mb.topk(
                x=logits,
                k=top_k,
                axis=-1,
                ascending=False,
                name="verifier_topk",
            )
            return indices

        chunk_count = weight.shape[0] // chunk_rows
        bias = None
        if valid_rows is not None and valid_rows < weight.shape[0]:
            bias = np.zeros((weight.shape[0],), dtype=np.float16)
            bias[valid_rows:] = np.float16(-65504.0)
        logits = mb.linear(x=flat, weight=weight, bias=bias, name="verifier_lm_head")
        chunked_logits = mb.reshape(
            x=logits,
            shape=(batch_size * chunk_count, chunk_rows),
            name="chunked_logits",
        )
        values, indices = mb.topk(
            x=chunked_logits,
            k=top_k,
            axis=-1,
            ascending=False,
            name="chunk_topk",
        )
        return values, indices

    return program


def package_disk_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def convert_program(program, compression):
    import coremltools as ct

    converted = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    if compression == "q4_block":
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        return linear_quantize_weights(
            converted,
            config=OptimizationConfig(
                global_config=OpLinearQuantizerConfig(
                    mode="linear",
                    dtype="uint4",
                    granularity="per_block",
                    block_size=64,
                    weight_threshold=2048,
                )
            ),
        )
    if compression in {"lut8", "q4_exact_lut"}:
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )

        palette_options = {
            "mode": "unique" if compression == "q4_exact_lut" else "uniform",
            "granularity": "per_grouped_channel",
            "group_size": 64,
            "channel_axis": 1 if compression == "q4_exact_lut" else 0,
            "weight_threshold": 2048,
        }
        if compression != "q4_exact_lut":
            palette_options["nbits"] = 8
        converted = palettize_weights(
            converted,
            config=OptimizationConfig(
                global_config=OpPalettizerConfig(**palette_options)
            ),
        )
    return converted


def predict_outputs(model, input_name, hidden):
    outputs = model.predict({input_name: hidden})
    indices = [
        np.asarray(value, dtype=np.int64).reshape(-1)
        for value in outputs.values()
        if np.issubdtype(np.asarray(value).dtype, np.integer)
    ]
    values = [
        np.asarray(value, dtype=np.float32).reshape(-1)
        for value in outputs.values()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    ]
    if len(indices) != 1 or len(values) > 1:
        raise RuntimeError(f"unexpected compact outputs: {list(outputs)}")
    return indices[0], values[0] if values else None


def merge_chunk_candidates(
    local_indices, values, *, batch_size, chunk_count, chunk_rows, top_k
):
    local = np.asarray(local_indices, dtype=np.int64).reshape(
        batch_size, chunk_count, top_k
    )
    scores = np.asarray(values, dtype=np.float32).reshape(
        batch_size, chunk_count, top_k
    )
    offsets = (
        np.arange(chunk_count, dtype=np.int64).reshape(1, chunk_count, 1) * chunk_rows
    )
    candidates = local + offsets
    merged = []
    for row in range(batch_size):
        selected = np.argsort(scores[row].reshape(-1))[-top_k:][::-1]
        merged.append(candidates[row].reshape(-1)[selected])
    return candidates, np.stack(merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        help="synthetic row count, or optional prefix limit for a real row sidecar",
    )
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        help=(
            "run one ANE-local top-k per row chunk and return the tiny candidate "
            "table for a host merge"
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("fp16", "lut8", "q4_exact_lut", "q4_block"),
        default="fp16",
    )
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        help="pad vocabulary rows with masked zeros so ANE-friendly chunk widths divide",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rows-sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--package",
        type=Path,
        help="retain the package at this new path; otherwise use a temporary path",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="skip NumPy top-k comparison for large geometry",
    )
    args = parser.parse_args()
    if (args.model is None) != (args.rows_sidecar is None):
        parser.error("--model and --rows-sidecar must be supplied together")
    if args.model is None and args.rows is None:
        parser.error("--rows is required for a synthetic probe")
    for name in ("hidden_size", "top_k", "batch_size", "samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.package is not None and args.package.exists():
        parser.error("--package must name a path that does not exist")

    import coremltools as ct

    started = time.perf_counter()
    if args.model is None:
        weight = make_weight(args.rows, args.hidden_size, args.seed)
        row_ids = np.arange(args.rows, dtype=np.uint32)
        source = {"kind": "deterministic_synthetic", "seed": args.seed}
    else:
        weight, row_ids, source = load_real_compact_weight(
            args.model, args.rows_sidecar, args.rows
        )
    valid_rows, hidden_size = map(int, weight.shape)
    source_fingerprint = hashlib.sha256()
    source_fingerprint.update(row_ids.tobytes())
    source_fingerprint.update(weight.tobytes())
    source_fingerprint = source_fingerprint.hexdigest()
    if args.pad_to_multiple:
        if args.pad_to_multiple < 1:
            parser.error("--pad-to-multiple must be positive")
        padded_rows = (
            (valid_rows + args.pad_to_multiple - 1) // args.pad_to_multiple
        ) * args.pad_to_multiple
        if padded_rows != valid_rows:
            weight = np.pad(weight, ((0, padded_rows - valid_rows), (0, 0)))
            row_ids = np.pad(
                row_ids,
                (0, padded_rows - valid_rows),
                constant_values=np.iinfo(np.uint32).max,
            )
    rows = int(weight.shape[0])
    weight_seconds = time.perf_counter() - started
    if args.top_k > rows:
        parser.error("--top-k cannot exceed selected rows")
    if args.chunk_rows is not None and args.chunk_rows < args.top_k:
        parser.error("--chunk-rows must be at least --top-k")
    if args.chunk_rows is not None and rows % args.chunk_rows:
        parser.error("selected rows must be divisible by --chunk-rows")
    input_rng = np.random.default_rng(args.seed + 1)
    hidden = input_rng.standard_normal(
        (args.batch_size, 1, hidden_size), dtype=np.float32
    ).astype(np.float16)
    program = make_program(
        weight, args.top_k, args.chunk_rows, args.batch_size, valid_rows
    )

    temp = None
    if args.package is None:
        temp = tempfile.TemporaryDirectory(prefix="mlx-uag-ane-vocab-topk-")
        package = Path(temp.name) / "verifier-head.mlpackage"
    else:
        package = args.package.resolve()
        package.parent.mkdir(parents=True, exist_ok=True)

    try:
        convert_started = time.perf_counter()
        converted = convert_program(program, args.compression)
        conversion_seconds = time.perf_counter() - convert_started
        save_started = time.perf_counter()
        converted.save(str(package))
        save_seconds = time.perf_counter() - save_started
        load_started = time.perf_counter()
        runtime = ct.models.MLModel(
            str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        load_seconds = time.perf_counter() - load_started
        plan = compute_plan_receipt(runtime)
        non_ane = non_ane_compute_operations(plan)
        non_ane_required = non_ane_required_operations(plan)
        if non_ane_required:
            raise RuntimeError(
                f"linear/top-k path is not ANE-resident: {non_ane_required}"
            )
        if args.compression in {"lut8", "q4_exact_lut"} and not any(
            "constexpr_lut_to_dense" in name for name in plan["operator_counts"]
        ):
            raise RuntimeError("LUT8 package contains no constexpr LUT operation")
        if args.compression == "q4_block" and not any(
            "constexpr_blockwise_shift_scale" in name
            or "constexpr_affine_dequantize" in name
            for name in plan["operator_counts"]
        ):
            raise RuntimeError("Q4 package contains no blockwise dequantization")

        input_name = runtime.get_spec().description.input[0].name
        for _ in range(args.warmup):
            predict_outputs(runtime, input_name, hidden)
        latencies_ms = []
        output_indices = None
        output_values = None
        for _ in range(args.samples):
            sample_started = time.perf_counter_ns()
            output_indices, output_values = predict_outputs(runtime, input_name, hidden)
            latencies_ms.append((time.perf_counter_ns() - sample_started) / 1e6)
        assert output_indices is not None

        reference = {"skipped": True}
        if not args.skip_reference:
            logits = hidden.reshape(args.batch_size, hidden_size).astype(np.float32) @ (
                weight.astype(np.float32).T
            )
            if valid_rows < rows:
                logits[:, valid_rows:] = -65504.0
            expected = np.argsort(logits, axis=-1)[:, -args.top_k :][:, ::-1]
            if args.chunk_rows is not None:
                if output_values is None:
                    raise RuntimeError(
                        "chunked package did not return candidate scores"
                    )
                chunk_count = (rows + args.chunk_rows - 1) // args.chunk_rows
                _candidates, observed = merge_chunk_candidates(
                    output_indices,
                    output_values,
                    batch_size=args.batch_size,
                    chunk_count=chunk_count,
                    chunk_rows=args.chunk_rows,
                    top_k=args.top_k,
                )
            else:
                observed = output_indices.reshape(args.batch_size, args.top_k)
            reference = {
                "skipped": False,
                "expected_indices": expected.astype(int).tolist(),
                "observed_indices": observed.astype(int).tolist(),
                "set_equal": all(
                    set(left.tolist()) == set(right.tolist())
                    for left, right in zip(expected, observed)
                ),
                "ordered_equal": np.array_equal(expected, observed),
            }

        chunk_count = (
            (rows + args.chunk_rows - 1) // args.chunk_rows
            if args.chunk_rows is not None
            else 1
        )
        output_bytes = output_indices.nbytes + (
            output_values.nbytes if output_values is not None else 0
        )
        candidate_compact_indices = output_indices.reshape(
            args.batch_size, chunk_count, args.top_k
        )
        if args.chunk_rows is not None:
            candidate_compact_indices, host_topk_compact_indices = (
                merge_chunk_candidates(
                    output_indices,
                    output_values,
                    batch_size=args.batch_size,
                    chunk_count=chunk_count,
                    chunk_rows=args.chunk_rows,
                    top_k=args.top_k,
                )
            )
        else:
            host_topk_compact_indices = candidate_compact_indices[:, 0, :]

        report = {
            "schema": "mlx-lm.qwen4-ane-verifier-head-export.v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "full LM-head verification topology; CPU plus ANE, no Metal work",
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "coremltools": ct.__version__,
            },
            "geometry": {
                "rows": rows,
                "valid_rows": valid_rows,
                "hidden_size": hidden_size,
                "top_k": args.top_k,
                "batch_size": args.batch_size,
                "chunk_rows": args.chunk_rows,
                "chunk_count": chunk_count,
                "input_fp16_bytes": hidden.nbytes,
                "output_candidate_bytes_python_materialized": output_bytes,
                "output_candidate_bytes_contract": (
                    args.batch_size
                    * chunk_count
                    * args.top_k
                    * (4 + (2 if args.chunk_rows is not None else 0))
                ),
                "source_weight_fp16_bytes": weight.nbytes,
            },
            "source": source,
            "source_fingerprint": source_fingerprint,
            "package": {
                "path": str(package) if args.package is not None else None,
                "retained": args.package is not None,
                "compression": args.compression,
                "disk_bytes": package_disk_bytes(package),
                "weight_generation_seconds": weight_seconds,
                "conversion_seconds": conversion_seconds,
                "save_seconds": save_seconds,
                "runtime_load_seconds": load_seconds,
            },
            "compute_units": "CPU_AND_NE",
            "compute_plan": plan,
            "non_ane_auxiliary_operations": non_ane,
            "latency_ms": {
                "samples": args.samples,
                "warmup": args.warmup,
                "min": min(latencies_ms),
                "median": percentile(latencies_ms, 50),
                "p95": percentile(latencies_ms, 95),
                "max": max(latencies_ms),
                "all": latencies_ms,
            },
            "output_local_indices": output_indices.astype(int).tolist(),
            "candidate_compact_indices": (
                candidate_compact_indices.astype(int).tolist()
            ),
            "host_topk_compact_indices": (
                host_topk_compact_indices.astype(int).tolist()
            ),
            "host_topk_vocab_ids": (
                row_ids[host_topk_compact_indices].astype(int).tolist()
            ),
            "output_values": (
                output_values.astype(float).tolist()
                if output_values is not None
                else None
            ),
            "reference": reference,
            "decision": {
                "ane_resident": True,
                "next_gate": (
                    "Measure fixed-width GPU/ANE composition before server "
                    "integration. The package remains proposal-only and the "
                    "full target head remains commit authority."
                    if source["kind"] == "selected_real_lm_head"
                    else "Build the same compact projection from the real "
                    "selected Qwen4 LM-head rows before server integration."
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.package is not None:
            manifest = {
                "schema": "mlx-lm.ane-verifier-head-package.v1",
                "stage": "full_vocab_greedy_verify",
                "batch_size": args.batch_size,
                "hidden_size": hidden_size,
                "row_count": rows,
                "valid_row_count": valid_rows,
                "top_k": args.top_k,
                "chunk_rows": args.chunk_rows,
                "weight_bytes": weight.nbytes,
                "package_disk_bytes": report["package"]["disk_bytes"],
                "compression": args.compression,
                "source": source,
                "source_fingerprint": source_fingerprint,
                "compute_units": "CPU_AND_NE",
                "compute_plan": plan,
            }
            package.with_suffix(package.suffix + ".json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        print(
            json.dumps(
                {
                    "compute_plan": plan,
                    "latency_ms": report["latency_ms"],
                    "reference": reference,
                },
                indent=2,
            )
        )
        print(f"wrote {args.output}")
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
