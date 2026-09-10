#!/usr/bin/env python3
"""Measure whether an FP16 cache-shaped ANE job overlaps MLX GPU work.

The ANE side constructs a two-row FP16 capsule from a resident one-row input.
The GPU side performs the matched MLX concatenate.  Persistent worker threads
and barriers keep thread startup outside the timed region.  This is a systems
boundary probe, not a serving-ready Qwen cache or a claim that cache
construction alone accelerates a model.  Qwen's BF16 cache representation and
MLX-to-e5rt input staging require separate gates.

Run with ANEForge on ``PYTHONPATH``.  The script neither installs software nor
changes system security settings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import threading
import time
from pathlib import Path

import aneforge as af
import mlx.core as mx
import numpy as np


def _median_ms(samples_ns: list[int], inner: int) -> float:
    return statistics.median(samples_ns) / inner / 1e6


def _measure_serial(fn, *, repetitions: int, inner: int) -> list[int]:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        samples.append(time.perf_counter_ns() - started)
    return samples


def _measure_concurrent(
    ane_fn, gpu_fn, *, repetitions: int, inner: int
) -> tuple[list[int], list[int], list[int]]:
    start = threading.Barrier(3)
    done = threading.Barrier(3)
    ane_samples = [0] * repetitions
    gpu_samples = [0] * repetitions

    def worker(fn, samples):
        for repetition in range(repetitions):
            start.wait()
            started = time.perf_counter_ns()
            for _ in range(inner):
                fn()
            samples[repetition] = time.perf_counter_ns() - started
            done.wait()

    ane_thread = threading.Thread(target=worker, args=(ane_fn, ane_samples))
    gpu_thread = threading.Thread(target=worker, args=(gpu_fn, gpu_samples))
    ane_thread.start()
    gpu_thread.start()
    wall_samples = []
    try:
        for _ in range(repetitions):
            start.wait()
            started = time.perf_counter_ns()
            done.wait()
            wall_samples.append(time.perf_counter_ns() - started)
    finally:
        ane_thread.join()
        gpu_thread.join()
    return wall_samples, ane_samples, gpu_samples


def _case(size_mib: float, width: int, warmup: int, repetitions: int, inner: int):
    elements = max(1, int(size_mib * (1 << 20) / np.dtype(np.float16).itemsize))
    height = max(1, (elements + width - 1) // width)
    shape = (height, width)
    source = np.arange(height * width, dtype=np.uint16).reshape(shape).view(np.float16)
    source = np.nan_to_num(source, copy=False)

    ane_input = af.input(shape)
    graph = af.concat([ane_input, ane_input], axis=0)
    build_root = Path(tempfile.mkdtemp(prefix="mlxuag-ane-gpu-overlap-"))
    net = None
    try:
        net = af.compile(graph, opt=0, build_dir=build_root)
        np.copyto(net.input_view(), source)
        gpu_source = mx.array(source)
        mx.eval(gpu_source)
        cpu_output = np.empty((height * 2, width), dtype=np.float16)

        def ane_work():
            net.execute()

        def gpu_work():
            output = mx.concatenate([gpu_source, gpu_source], axis=0)
            mx.eval(output)
            mx.synchronize()

        def cpu_work():
            # Fair resident-buffer CPU fallback: no allocation, but the CPU
            # must still issue both physical copies into the B2 capsule.
            np.copyto(cpu_output[:height], source)
            np.copyto(cpu_output[height:], source)

        for _ in range(warmup):
            ane_work()
            gpu_work()
            cpu_work()

        ane_only = _measure_serial(ane_work, repetitions=repetitions, inner=inner)
        gpu_only = _measure_serial(gpu_work, repetitions=repetitions, inner=inner)
        cpu_only = _measure_serial(cpu_work, repetitions=repetitions, inner=inner)
        serial = _measure_serial(
            lambda: (ane_work(), gpu_work()), repetitions=repetitions, inner=inner
        )
        concurrent, concurrent_ane, concurrent_ane_gpu = _measure_concurrent(
            ane_work,
            gpu_work,
            repetitions=repetitions,
            inner=inner,
        )
        cpu_serial = _measure_serial(
            lambda: (cpu_work(), gpu_work()), repetitions=repetitions, inner=inner
        )
        cpu_concurrent, concurrent_cpu, concurrent_cpu_gpu = _measure_concurrent(
            cpu_work,
            gpu_work,
            repetitions=repetitions,
            inner=inner,
        )

        ane_ms = _median_ms(ane_only, inner)
        gpu_ms = _median_ms(gpu_only, inner)
        cpu_ms = _median_ms(cpu_only, inner)
        serial_ms = _median_ms(serial, inner)
        concurrent_ms = _median_ms(concurrent, inner)
        cpu_serial_ms = _median_ms(cpu_serial, inner)
        cpu_concurrent_ms = _median_ms(cpu_concurrent, inner)
        ideal_ms = max(ane_ms, gpu_ms)
        overlap_saving_ms = max(0.0, ane_ms + gpu_ms - concurrent_ms)
        overlap_efficiency = overlap_saving_ms / max(min(ane_ms, gpu_ms), 1e-9)
        cpu_overlap_saving_ms = max(0.0, cpu_ms + gpu_ms - cpu_concurrent_ms)
        cpu_overlap_efficiency = cpu_overlap_saving_ms / max(min(cpu_ms, gpu_ms), 1e-9)
        expected = np.concatenate([source, source], axis=0)
        ane_exact = bool(np.array_equal(net.output_view(), expected))
        cpu_exact = bool(np.array_equal(cpu_output, expected))
        gpu_result = mx.concatenate([gpu_source, gpu_source], axis=0)
        mx.eval(gpu_result)
        gpu_exact = bool(np.array_equal(np.asarray(gpu_result), expected))
        return {
            "status": "ok",
            "shape": list(shape),
            "input_bytes": int(source.nbytes),
            "output_bytes_per_engine": int(expected.nbytes),
            "ane_only_median_ms": ane_ms,
            "gpu_only_median_ms": gpu_ms,
            "cpu_resident_copy_median_ms": cpu_ms,
            "serial_median_ms": serial_ms,
            "concurrent_wall_median_ms": concurrent_ms,
            "concurrent_speedup_over_serial": serial_ms / concurrent_ms,
            "ideal_concurrent_median_ms": ideal_ms,
            "overlap_efficiency": overlap_efficiency,
            "cpu_gpu_serial_median_ms": cpu_serial_ms,
            "cpu_gpu_concurrent_wall_median_ms": cpu_concurrent_ms,
            "cpu_gpu_concurrent_speedup_over_serial": (
                cpu_serial_ms / cpu_concurrent_ms
            ),
            "cpu_gpu_overlap_efficiency": cpu_overlap_efficiency,
            "cpu_slowdown_versus_ane": cpu_ms / ane_ms,
            "ane_exact": ane_exact,
            "gpu_exact": gpu_exact,
            "cpu_exact": cpu_exact,
            "raw_batch_ms": {
                "ane_only": [value / 1e6 for value in ane_only],
                "gpu_only": [value / 1e6 for value in gpu_only],
                "cpu_only": [value / 1e6 for value in cpu_only],
                "serial": [value / 1e6 for value in serial],
                "concurrent_wall": [value / 1e6 for value in concurrent],
                "concurrent_ane": [value / 1e6 for value in concurrent_ane],
                "concurrent_ane_gpu": [value / 1e6 for value in concurrent_ane_gpu],
                "cpu_gpu_serial": [value / 1e6 for value in cpu_serial],
                "cpu_gpu_concurrent_wall": [value / 1e6 for value in cpu_concurrent],
                "concurrent_cpu": [value / 1e6 for value in concurrent_cpu],
                "concurrent_cpu_gpu": [value / 1e6 for value in concurrent_cpu_gpu],
            },
        }
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    finally:
        if net is not None:
            net.release()
        import shutil

        shutil.rmtree(build_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=float, action="append")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--inner", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.inner < 1 or args.warmup < 0:
        parser.error("repetitions and inner must be positive; warmup cannot be negative")

    rows = []
    for size_mib in args.size_mib or [4.0, 16.0, 32.0]:
        print(json.dumps({"event": "ane_gpu_overlap_start", "size_mib": size_mib}), flush=True)
        row = _case(size_mib, args.width, args.warmup, args.repetitions, args.inner)
        row["size_mib"] = size_mib
        rows.append(row)
        print(json.dumps({"event": "ane_gpu_overlap_result", **row}), flush=True)

    result = {
        "schema": "mlx-uag.ane-gpu-cache-overlap.v1",
        "method": (
            "FP16 resident e5rt B1-to-B2 concat concurrent with matched MLX "
            "GPU concat and resident-buffer NumPy CPU control"
        ),
        "limitations": [
            "direct e5rt exposes no supported CPU execution switch; NumPy is the forced CPU control",
            "real Qwen cache/state is BF16 while this e5rt boundary is FP16",
            "MLX-to-e5rt input staging is not zero-copy in this experiment",
        ],
        "repetitions": args.repetitions,
        "inner": args.inner,
        "cases": rows,
        "passed": all(
            row.get("status") == "ok"
            and row.get("ane_exact")
            and row.get("gpu_exact")
            and row.get("cpu_exact")
            for row in rows
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
