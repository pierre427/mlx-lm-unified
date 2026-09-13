#!/usr/bin/env python3
"""Measure a partitioned ANE greedy verifier head against idle/queued GPU work.

The stateful Qwen4 trunk remains on MLX/Metal.  ANE receives one final hidden
row, evaluates every vocabulary row across ANE-sized FP16 partitions, and
returns a global top-k.  This is the narrow authoritative verification substep
that can be moved without transferring mutable recurrent or attention state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import multiprocessing
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summary(values):
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "all": values,
    }


def thermal_snapshot():
    result = subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def load_q4_head(model_dir):
    import mlx.core as mx

    config = json.loads((model_dir / "config.json").read_text())
    quantization = config["quantization"]
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "language_model.lm_head."
    tensors = mx.load(str(model_dir / index[prefix + "weight"]))
    return (
        tensors[prefix + "weight"],
        tensors[prefix + "scales"],
        tensors[prefix + "biases"],
        {
            "bits": int(quantization["bits"]),
            "group_size": int(quantization["group_size"]),
            "mode": quantization.get("mode", "affine"),
            "weight_shard": str(model_dir / index[prefix + "weight"]),
        },
    )


class PartitionedANEHead:
    def __init__(self, package_dir, package_glob):
        import coremltools as ct

        self.parts = []
        package_paths = sorted(package_dir.glob(package_glob))
        if not package_paths:
            raise ValueError(f"no ANE packages matched {package_glob!r}")
        for package in package_paths:
            manifest = json.loads(
                package.with_suffix(package.suffix + ".json").read_text()
            )
            if manifest.get("stage") != "full_vocab_greedy_verify":
                raise ValueError(f"wrong ANE stage in {package}")
            preferred = manifest["compute_plan"]["preferred_counts"]
            if not preferred.get("MLNeuralEngineComputeDevice"):
                raise ValueError(f"package has no ANE-preferred work: {package}")
            model = ct.models.MLModel(
                str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            rows_path = Path(manifest["source"]["rows_sidecar"])
            import mlx.core as mx

            rows = np.asarray(mx.load(str(rows_path))["rows"], dtype=np.uint32)
            self.parts.append((package, manifest, model, rows))
        self.hidden_size = int(self.parts[0][1]["hidden_size"])
        self.vocab_rows = sum(len(part[3]) for part in self.parts)

    @staticmethod
    def _part_candidates(outputs, manifest):
        integers = [
            np.asarray(value, dtype=np.int64)
            for value in outputs.values()
            if np.issubdtype(np.asarray(value).dtype, np.integer)
        ]
        floats = [
            np.asarray(value, dtype=np.float32)
            for value in outputs.values()
            if np.issubdtype(np.asarray(value).dtype, np.floating)
        ]
        if len(integers) != 1 or len(floats) != 1:
            raise RuntimeError(f"unexpected ANE outputs: {list(outputs)}")
        chunks = int(manifest["row_count"]) // int(manifest["chunk_rows"])
        top_k = int(manifest["top_k"])
        local = integers[0].reshape(chunks, top_k)
        scores = floats[0].reshape(chunks, top_k)
        offsets = np.arange(chunks, dtype=np.int64)[:, None] * int(
            manifest["chunk_rows"]
        )
        return (local + offsets).reshape(-1), scores.reshape(-1)

    def verify(self, hidden, top_k=4):
        hidden = np.ascontiguousarray(
            np.asarray(hidden, dtype=np.float16).reshape(1, 1, self.hidden_size)
        )
        candidates = []
        part_ms = []
        started = time.perf_counter_ns()
        for _package, manifest, model, rows in self.parts:
            input_name = model.get_spec().description.input[0].name
            part_started = time.perf_counter_ns()
            outputs = model.predict({input_name: hidden})
            part_ms.append((time.perf_counter_ns() - part_started) / 1e6)
            indices, scores = self._part_candidates(outputs, manifest)
            valid = indices < len(rows)
            for index, score in zip(indices[valid], scores[valid]):
                candidates.append((float(score), int(rows[index])))
        candidates.sort(reverse=True)
        selected = candidates[:top_k]
        return {
            "token_ids": [token for _score, token in selected],
            "scores": [score for score, _token in selected],
            "margin": selected[0][0] - selected[1][0],
            "part_ms": part_ms,
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
        }


def _ane_worker_main(connection, package_dir, package_glob):
    try:
        head = PartitionedANEHead(Path(package_dir), package_glob)
        connection.send(
            {
                "ready": True,
                "packages": [str(part[0]) for part in head.parts],
                "package_bytes": sum(
                    int(part[1]["package_disk_bytes"]) for part in head.parts
                ),
                "hidden_size": head.hidden_size,
            }
        )
        while True:
            command = connection.recv()
            if command[0] == "stop":
                return
            _, hidden, top_k = command
            connection.send(head.verify(hidden, top_k=top_k))
    finally:
        connection.close()


class PartitionedANEProcess:
    """Keep Core ML orchestration outside the MLX process and its Python GIL."""

    def __init__(self, package_dir, package_glob):
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_ane_worker_main,
            args=(child, str(package_dir), package_glob),
            name="qwen4-ane-verifier",
        )
        self._process.start()
        child.close()
        ready = self._connection.recv()
        if not ready.get("ready"):
            raise RuntimeError(f"ANE verifier worker failed: {ready}")
        self.packages = ready["packages"]
        self.package_bytes = ready["package_bytes"]
        self.hidden_size = ready["hidden_size"]

    def verify(self, hidden, top_k=4):
        self._connection.send(("verify", hidden, top_k))
        return self._connection.recv()

    def close(self):
        if self._process.is_alive():
            self._connection.send(("stop",))
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()


class GPUHead:
    def __init__(self, model_dir):
        self.weight, self.scales, self.biases, self.config = load_q4_head(model_dir)

    def logits(self, hidden):
        import mlx.core as mx

        hidden = mx.array(hidden).astype(mx.bfloat16)
        return mx.quantized_matmul(
            hidden,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.config["group_size"],
            bits=self.config["bits"],
            mode=self.config["mode"],
        )

    def run(self, hidden, top_k=4):
        import mlx.core as mx

        started = time.perf_counter_ns()
        logits = self.logits(hidden)
        indices = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
        values = mx.take_along_axis(logits, indices, axis=-1)
        order = mx.argsort(values, axis=-1)[..., ::-1]
        indices = mx.take_along_axis(indices, order, axis=-1)
        values = mx.take_along_axis(values, order, axis=-1).astype(mx.float32)
        mx.eval(values, indices)
        return {
            "token_ids": np.asarray(indices, dtype=np.int64).tolist(),
            "scores": np.asarray(values, dtype=np.float32).tolist(),
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--package-glob", default="part*.mlpackage")
    parser.add_argument("--widths", default="0,1,2,4,8,16,32")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--parity-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=427)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = [int(value) for value in args.widths.split(",")]
    if min(widths) < 0 or args.samples < 1 or args.parity_rows < 1:
        parser.error("widths must be nonnegative; samples/parity-rows positive")

    import mlx.core as mx

    rng = np.random.default_rng(args.seed)
    max_width = max(widths) + 1
    hidden_bank = rng.standard_normal(
        (max(args.parity_rows, max_width), 2560), dtype=np.float32
    ).astype(np.float16)
    ane = PartitionedANEProcess(args.package_dir, args.package_glob)
    gpu = GPUHead(args.model)

    # Warm both engines and prove that every package produces real work.
    for _ in range(args.warmup):
        ane.verify(hidden_bank[0])
        gpu.run(hidden_bank[:max_width])

    parity = []
    for index in range(args.parity_rows):
        ane_result = ane.verify(hidden_bank[index])
        gpu_result = gpu.run(hidden_bank[index : index + 1])
        ane_ids = ane_result["token_ids"]
        gpu_ids = gpu_result["token_ids"][0]
        parity.append(
            {
                "row": index,
                "top1_equal": ane_ids[0] == gpu_ids[0],
                "top4_set_equal": set(ane_ids) == set(gpu_ids),
                "ane_top1": ane_ids[0],
                "gpu_top1": gpu_ids[0],
                "ane_margin": ane_result["margin"],
            }
        )

    cases = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        for width in widths:
            rows = {
                "gpu_pressure_ms": [],
                "gpu_inclusive_ms": [],
                "gpu_target_marginal_ms": [],
                "gpu_idle_target_ms": [],
                "gpu_queued_target_ms": [],
                "hybrid_target_ms": [],
                "hybrid_gpu_ms": [],
                "hybrid_gpu_interference_ms": [],
                "hybrid_makespan_ms": [],
            }
            orders = []
            for sample in range(args.samples):
                target = hidden_bank[sample % len(hidden_bank)]
                pressure = hidden_bank[1 : width + 1]
                inclusive = np.concatenate([pressure, target[None]], axis=0)
                arm_order = ("isolated", "queued", "hybrid")
                shift = sample % len(arm_order)
                arm_order = arm_order[shift:] + arm_order[:shift]
                orders.append(arm_order)
                observed = {}
                for arm in arm_order:
                    if arm == "isolated":
                        pressure_ms = gpu.run(pressure)["wall_ms"] if width else 0.0
                        inclusive_ms = gpu.run(inclusive)["wall_ms"]
                        idle_target_ms = gpu.run(target[None])["wall_ms"]
                        observed.update(
                            pressure_ms=pressure_ms,
                            inclusive_ms=inclusive_ms,
                            idle_target_ms=idle_target_ms,
                        )
                    elif arm == "queued":
                        queued_started = time.perf_counter_ns()
                        if width:
                            gpu.run(pressure)
                        gpu.run(target[None])
                        observed["queued_ms"] = (
                            time.perf_counter_ns() - queued_started
                        ) / 1e6
                    else:
                        hybrid_started = time.perf_counter_ns()
                        future = executor.submit(ane.verify, target)
                        hybrid_gpu_ms = gpu.run(pressure)["wall_ms"] if width else 0.0
                        ane_result = future.result()
                        observed.update(
                            hybrid_target_ms=ane_result["wall_ms"],
                            hybrid_gpu_ms=hybrid_gpu_ms,
                            hybrid_makespan_ms=(time.perf_counter_ns() - hybrid_started)
                            / 1e6,
                        )
                rows["gpu_pressure_ms"].append(observed["pressure_ms"])
                rows["gpu_inclusive_ms"].append(observed["inclusive_ms"])
                rows["gpu_target_marginal_ms"].append(
                    observed["inclusive_ms"] - observed["pressure_ms"]
                )
                rows["gpu_idle_target_ms"].append(observed["idle_target_ms"])
                rows["gpu_queued_target_ms"].append(observed["queued_ms"])
                rows["hybrid_target_ms"].append(observed["hybrid_target_ms"])
                rows["hybrid_gpu_ms"].append(observed["hybrid_gpu_ms"])
                rows["hybrid_gpu_interference_ms"].append(
                    observed["hybrid_gpu_ms"] - observed["pressure_ms"]
                )
                rows["hybrid_makespan_ms"].append(observed["hybrid_makespan_ms"])
            summarized = {name: summary(values) for name, values in rows.items()}
            summarized["ratios"] = {
                "target_vs_gpu_inclusive": (
                    summarized["gpu_inclusive_ms"]["median"]
                    / summarized["hybrid_target_ms"]["median"]
                ),
                "target_vs_gpu_queued": (
                    summarized["gpu_queued_target_ms"]["median"]
                    / summarized["hybrid_target_ms"]["median"]
                ),
                "makespan_vs_gpu_inclusive": (
                    summarized["gpu_inclusive_ms"]["median"]
                    / summarized["hybrid_makespan_ms"]["median"]
                ),
            }
            summarized["arm_orders"] = orders
            cases[str(width)] = summarized
    finally:
        executor.shutdown()
        ane.close()

    result = {
        "schema": "mlx-lm.qwen4-ane-verifier-crossover.v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "model": str(args.model),
        "packages": ane.packages,
        "package_bytes": ane.package_bytes,
        "method": {
            "gpu": "native affine-q4 MLX quantized_matmul plus top-k",
            "ane": "five FP16 ANE-resident vocabulary partitions plus host candidate merge",
            "pressure_width": "other ready verification-head rows executing on GPU; an experimental axis, not a linear predictor",
            "gpu_target_marginal": "paired GPU head(B+1)-head(B), measured directly for every sample",
            "hybrid_gpu_interference": "paired GPU pressure time during ANE minus isolated GPU pressure time",
            "arm_order": "cyclically rotated isolated/queued/hybrid arms",
            "gpu_inclusive": "target joins the current GPU head batch",
            "gpu_queued": "target waits behind the current GPU head batch",
            "hybrid": "target runs on ANE while pressure rows run on GPU",
            "hidden": "seeded normalized-scale synthetic final hidden rows",
        },
        "thermal": {"before": thermal_snapshot()},
        "parity": {
            "rows": args.parity_rows,
            "top1_equal": sum(row["top1_equal"] for row in parity),
            "top4_set_equal": sum(row["top4_set_equal"] for row in parity),
            "details": parity,
        },
        "cases": cases,
    }
    result["thermal"]["after"] = thermal_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parity": result["parity"], "cases": cases}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
