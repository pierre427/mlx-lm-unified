#!/usr/bin/env python3
"""Qualify direct-ANE construction and MLX adoption of cache-shaped capsules.

The experiment deliberately isolates the systems boundary before attempting a
model conversion.  A direct e5rt program receives one FP16 prefix tensor and
constructs a two-row capsule by concatenating the prefix with itself.  It then
measures resident execution, input refresh, output readback, and MLX adoption
against an MLX GPU concatenate of the same bytes.

ANEForge must be supplied on ``PYTHONPATH``.  The script does not change system
security settings or install anything.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path

import aneforge as af
import mlx.core as mx
import numpy as np


def _median_ms(fn, repetitions: int) -> tuple[float, list[float]]:
    rows = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        fn()
        rows.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(rows), rows


def _case(
    size_mib: float,
    width: int,
    warmup: int,
    repetitions: int,
    payload_dtype: str,
) -> dict:
    elements = max(1, int(size_mib * (1 << 20) / np.dtype(np.float16).itemsize))
    height = max(1, (elements + width - 1) // width)
    shape = (height, width)
    if payload_dtype == "bfloat16-bits":
        # e5rt exposes FP16 buffers, while the Qwen cache/state is BF16.  A
        # concat is a byte-preserving operation, so carry the MLX BF16 payload
        # through the FP16-shaped e5rt buffer as raw uint16 bits.  The values
        # are deliberately finite BF16 values; bit equality below is the gate.
        finite = np.sin(np.arange(height * width, dtype=np.float32) * 0.001)
        mlx_source = mx.array(finite.reshape(shape), dtype=mx.bfloat16)
        mx.eval(mlx_source)
        mlx_source_bits = mlx_source.view(mx.uint16)
        source_bits = np.asarray(mlx_source_bits)
        source = source_bits.view(np.float16)
    else:
        source = np.arange(height * width, dtype=np.uint16).reshape(shape).view(np.float16)
        source = np.nan_to_num(source, copy=False)
        mlx_source = mx.array(source)
        mx.eval(mlx_source)
        mlx_source_bits = mlx_source.view(mx.uint16)

    tensor = af.input(shape)
    graph = af.concat([tensor, tensor], axis=0)
    build_root = Path(tempfile.mkdtemp(prefix="mlxuag-ane-cache-capsule-"))
    net = None
    try:
        compile_started = time.perf_counter_ns()
        net = af.compile(graph, opt=0, build_dir=build_root)
        compile_ms = (time.perf_counter_ns() - compile_started) / 1e6
        input_view = net.input_view()
        np.copyto(input_view, source)
        for _ in range(warmup):
            net.execute()

        resident_ms, resident_rows = _median_ms(net.execute, repetitions)

        def refresh_execute():
            np.copyto(input_view, source)
            net.execute()

        refresh_ms, refresh_rows = _median_ms(refresh_execute, repetitions)
        def mlx_refresh_execute():
            # Measure the boundary we would actually cross from an MLX-owned
            # cache.  This may synchronize or materialize a host view before
            # copying into ANEForge's separately-owned input buffer.
            if payload_dtype == "bfloat16-bits":
                host_view = np.asarray(mlx_source_bits).view(np.float16)
            else:
                host_view = np.asarray(mlx_source)
            np.copyto(input_view, host_view)
            net.execute()

        mlx_refresh_ms, mlx_refresh_rows = _median_ms(
            mlx_refresh_execute, repetitions
        )
        output_view = net.output_view()
        expected = np.concatenate([source, source], axis=0)
        exact = bool(np.array_equal(output_view, expected))

        readback_ms, readback_rows = _median_ms(
            lambda: np.array(output_view, copy=True), repetitions
        )

        # NumPy implements DLPack.  Require MLX to adopt the e5rt output buffer
        # without copying; ``copy=None`` is insufficient for a production
        # capability gate because it is allowed to fall back to a copy.
        gc.collect()
        mx.clear_cache()
        active_before = int(mx.get_active_memory())

        def adopt():
            value = mx.from_dlpack(output_view, copy=False)
            mx.eval(value)
            mx.synchronize()
            return value

        adopted = adopt()
        active_after = int(mx.get_active_memory())
        adopt_ms, adopt_rows = _median_ms(adopt, repetitions)
        adopted_bits = adopted.view(mx.uint16)
        expected_bits = expected.view(np.uint16)
        adopted_exact = bool(np.array_equal(np.asarray(adopted_bits), expected_bits))

        # Exercise the imported buffer through a real Metal reduction while
        # the owning e5rt Program remains alive.  A production capsule lease
        # must preserve that ownership relation for its entire consumer span.
        reference = mx.array(expected_bits)
        mx.eval(reference)

        def metal_consume():
            value = mx.all(adopted_bits == reference)
            mx.eval(value)
            mx.synchronize()
            return value

        consumed = metal_consume()
        metal_consumer_ms, metal_consumer_rows = _median_ms(
            metal_consume, repetitions
        )
        metal_consumer_exact = bool(consumed.item())
        output_bits = output_view.view(np.uint16).reshape(-1)
        original_first = np.uint16(output_bits[0])
        alias_sentinel = np.uint16(0x3F80)
        output_bits[0] = alias_sentinel
        mx.synchronize()
        dlpack_alias_coherent = np.uint16(adopted_bits.reshape(-1)[0].item()) == alias_sentinel
        output_bits[0] = original_first
        del adopted
        mx.clear_cache()

        gpu_source = mx.array(source)
        mx.eval(gpu_source)

        def gpu_concat():
            value = mx.concatenate([gpu_source, gpu_source], axis=0)
            mx.eval(value)
            mx.synchronize()
            return value

        for _ in range(warmup):
            gpu_concat()
        gpu_ms, gpu_rows = _median_ms(gpu_concat, repetitions)

        return {
            "status": "ok",
            "payload_dtype": payload_dtype,
            "shape": list(shape),
            "input_bytes": int(source.nbytes),
            "output_bytes": int(expected.nbytes),
            "compile_ms": compile_ms,
            "e5rt_resident_execute_median_ms": resident_ms,
            "e5rt_refresh_execute_median_ms": refresh_ms,
            "mlx_to_e5rt_refresh_execute_median_ms": mlx_refresh_ms,
            "e5rt_output_readback_copy_median_ms": readback_ms,
            "mlx_from_dlpack_median_ms": adopt_ms,
            "mlx_from_dlpack_active_delta_bytes": active_after - active_before,
            "mlx_from_dlpack_alias_coherent": bool(dlpack_alias_coherent),
            "mlx_metal_consumer_median_ms": metal_consumer_ms,
            "mlx_metal_consumer_exact": metal_consumer_exact,
            "program_alive_during_consumer": True,
            "mlx_gpu_concat_median_ms": gpu_ms,
            "ane_exact": exact,
            "mlx_adopted_exact": adopted_exact,
            "raw_ms": {
                "e5rt_resident_execute": resident_rows,
                "e5rt_refresh_execute": refresh_rows,
                "mlx_to_e5rt_refresh_execute": mlx_refresh_rows,
                "e5rt_output_readback_copy": readback_rows,
                "mlx_from_dlpack": adopt_rows,
                "mlx_metal_consumer": metal_consumer_rows,
                "mlx_gpu_concat": gpu_rows,
            },
        }
    except Exception as error:
        return {
            "status": "error",
            "size_mib": size_mib,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        if net is not None:
            net.release()
        import shutil

        shutil.rmtree(build_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=float, action="append")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--payload-dtype",
        choices=("float16", "bfloat16-bits"),
        default="float16",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = args.size_mib or [1.0, 4.0, 16.0, 32.0]
    rows = []
    for size_mib in sizes:
        print(json.dumps({"event": "ane_cache_capsule_start", "size_mib": size_mib}), flush=True)
        row = _case(
            size_mib,
            args.width,
            args.warmup,
            args.repetitions,
            args.payload_dtype,
        )
        row["size_mib"] = size_mib
        rows.append(row)
        print(json.dumps({"event": "ane_cache_capsule_result", **row}), flush=True)
    result = {
        "schema": "mlx-uag.ane-cache-capsule.v1",
        "method": "direct e5rt resident B1-to-B2 concatenate versus MLX GPU",
        "payload_dtype": args.payload_dtype,
        "cases": rows,
        "passed": all(
            row.get("status") == "ok"
            and row.get("ane_exact")
            and row.get("mlx_adopted_exact")
            and row.get("mlx_from_dlpack_alias_coherent")
            and row.get("mlx_metal_consumer_exact")
            for row in rows
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
