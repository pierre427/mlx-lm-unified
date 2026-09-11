#!/usr/bin/env python3
"""Qualify real e5rt cache transport while unrelated MLX work is in flight.

This standalone gate stops before serving integration. It uses a real
``AutomaticPrefixCache`` hit and ``KVCache`` plane, precompiles one pinned
ANEForge e5rt program, and compares ordinary GPU, CPU, e5rt-serial, and
e5rt-overlapped construction through a first MLX attention consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.apc import APCKey, AutomaticPrefixCache
from mlx_lm.cache_capsule import (
    CacheCapsuleProduct,
    CacheCapsulePool,
    KVCacheCapsulePayload,
    capture_kv_cache_plane,
)
from mlx_lm.e5rt_cache_capsule import (
    E5RTCacheCapsuleAdapter,
    E5RTCacheCapsuleSpec,
    PINNED_ANEFORGE_REVISION,
)
from mlx_lm.models.cache import BatchKVCache, KVCache

try:
    from benchmarks.qwen4_gdn_prefix_fanout_full_model_ab import _thermal_arm
except ModuleNotFoundError:
    from qwen4_gdn_prefix_fanout_full_model_ab import _thermal_arm


def _digest(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _thermal():
    result = subprocess.run(
        ["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.splitlines(),
        "stderr": result.stderr.splitlines(),
    }


def _host_receipt():
    chip = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "chip": chip.stdout.strip() if chip.returncode == 0 else None,
        "mlx_version": getattr(mx, "__version__", "unknown"),
    }


def _source_cache(tokens: int, heads: int, dim: int) -> KVCache:
    cache = KVCache()
    base = mx.arange(tokens * heads * dim, dtype=mx.uint32).reshape(
        1, heads, tokens, dim
    )
    keys = ((base % 4093).astype(mx.float32) / 64).astype(mx.bfloat16)
    values = (((base + 31) % 4093).astype(mx.float32) / 96).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    mx.synchronize()
    return cache


def _attention_consumer(cache: BatchKVCache, query, dim: int):
    keys, values = cache.keys_and_values()
    output = mx.fast.scaled_dot_product_attention(
        query, keys, values, scale=dim**-0.5
    )
    mx.eval(output)
    mx.synchronize()
    return output


class _IndependentGPUWork:
    def __init__(self, width: int, repeats: int):
        values = mx.arange(width * width, dtype=mx.float32).reshape(width, width)
        self.left = mx.sin(values * 0.001).astype(mx.float16)
        self.right = mx.cos(values * 0.001).astype(mx.float16)
        self.repeats = int(repeats)
        mx.eval(self.left, self.right)
        mx.synchronize()

    def __call__(self):
        output = None
        for _ in range(self.repeats):
            output = mx.matmul(self.left, self.right)
            mx.eval(output)
        mx.synchronize()
        return output


def _delta(after, before):
    return {key: int(after[key] - before.get(key, 0)) for key in after}


def _stage_cpu_overlap(source):
    """Copy device-visible source bytes on the caller before host fan-out."""

    def stage(value):
        if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
            return np.array(np.asarray(value.view(mx.uint16)), copy=True), "bf16"
        return np.array(np.asarray(value), copy=True), str(value.dtype)

    return stage(source.keys), stage(source.values)


def _repeat_staged_cpu(staged, repeats):
    """Pure NumPy worker: it must never acquire an MLX GPU stream."""

    return tuple(
        (np.repeat(value, repeats, axis=0), dtype) for value, dtype in staged
    )


def _adopt_cpu_overlap(pool, source, repeated):
    def adopt(item):
        value, dtype = item
        result = mx.array(value)
        return result.view(mx.bfloat16) if dtype == "bf16" else result

    keys, values = map(adopt, repeated)
    product = CacheCapsuleProduct(
        KVCacheCapsulePayload(
            keys,
            values,
            source.offset,
            source.generation,
            source.source_id,
            source.layout_fingerprint,
            "cpu",
            None,
        )
    )
    return pool._accept_product(product, source, "cpu", None)


def _run_arm(
    arm: str,
    *,
    apc: AutomaticPrefixCache,
    key: APCKey,
    token_ids,
    batch: int,
    query,
    dim: int,
    independent_work,
    pool: CacheCapsulePool,
    adapter: E5RTCacheCapsuleAdapter,
    cpu_executor: ThreadPoolExecutor,
):
    adapter_before = adapter.counters
    pool_before = pool.counters
    started = time.perf_counter_ns()
    lookup = apc.lookup(key, token_ids + [token_ids[-1] + 1])
    if not lookup.hit or lookup.capsule_generation is None:
        raise AssertionError("e5rt gate requires a real APC hit and generation")
    cache = lookup.cache[0]
    source = capture_kv_cache_plane(
        cache,
        generation=lookup.capsule_generation,
        source_id="real-apc-hit:plane:0",
        target_batch=batch,
        verify_raw_bits=False,
    )
    receipt = lease = batch_cache = None
    wait_ns = 0
    try:
        if arm == "ordinary_gpu":
            batch_cache = BatchKVCache.merge([cache] * batch)
            independent_work()
        elif arm in ("cpu", "gpu"):
            receipt = pool.prepare(source, primary=arm, fallback=None)
            lease = receipt.owner.lease()
            batch_cache = lease.restore_batch_kv_cache(
                lambda payload: mx.eval(payload.keys, payload.values)
            )
            independent_work()
        elif arm == "cpu_overlap":
            staged = _stage_cpu_overlap(source)
            future = cpu_executor.submit(
                _repeat_staged_cpu, staged, source.target_batch
            )
            independent_work()
            wait_started = time.perf_counter_ns()
            repeated = future.result(timeout=30)
            wait_ns = time.perf_counter_ns() - wait_started
            receipt = _adopt_cpu_overlap(pool, source, repeated)
            lease = receipt.owner.lease()
            batch_cache = lease.restore_batch_kv_cache(
                lambda payload: mx.eval(payload.keys, payload.values)
            )
        elif arm in ("e5rt_serial", "e5rt_overlap"):
            ticket = pool.submit(source)
            if arm == "e5rt_overlap":
                independent_work()
            wait_started = time.perf_counter_ns()
            receipt = ticket.await_adopt(timeout_s=30, fallback=None)
            wait_ns = time.perf_counter_ns() - wait_started
            if arm == "e5rt_serial":
                independent_work()
            lease = receipt.owner.lease()
            batch_cache = lease.restore_batch_kv_cache(
                lambda payload: mx.eval(payload.keys, payload.values)
            )
        else:
            raise ValueError(f"unknown arm: {arm}")

        output = _attention_consumer(batch_cache, query, dim)
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        return {
            "arm": arm,
            "wall_ms": wall_ms,
            "ready_wait_ms": wait_ns / 1e6,
            "output_digest": _digest(output.view(mx.uint16)),
            "key_digest": _digest(batch_cache.keys.view(mx.uint16)),
            "value_digest": _digest(batch_cache.values.view(mx.uint16)),
            "adapter_counter_delta": _delta(adapter.counters, adapter_before),
            "pool_counter_delta": _delta(pool.counters, pool_before),
        }
    finally:
        mx.synchronize()
        batch_cache = None
        if lease is not None:
            lease.close()
        if receipt is not None:
            receipt.owner.release()


def _summaries(rows, arms):
    return {
        arm: {
            "median_wall_ms": statistics.median(
                row["wall_ms"] for row in rows if row["arm"] == arm
            ),
            "median_ready_wait_ms": statistics.median(
                row["ready_wait_ms"] for row in rows if row["arm"] == arm
            ),
        }
        for arm in arms
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--gpu-work-width", type=int, default=1024)
    parser.add_argument("--gpu-work-repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--minimum-cooldown-seconds", type=float, default=15.0)
    parser.add_argument("--thermal-poll-seconds", type=float, default=5.0)
    parser.add_argument("--thermal-max-cooldown-seconds", type=float, default=300.0)
    parser.add_argument("--thermal-stable-snapshots", type=int, default=2)
    parser.add_argument(
        "--aneforge-revision", default=PINNED_ANEFORGE_REVISION
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.tokens,
        args.heads,
        args.dim,
        args.batch,
        args.gpu_work_width,
        args.gpu_work_repeats,
        args.repetitions,
    ) < 1:
        parser.error("geometry, work, and repetition values must be positive")
    if args.batch < 2 or args.warmup < 0 or args.minimum_speedup < 1:
        parser.error("batch must be >=2, warmup >=0, and minimum speedup >=1")
    if args.minimum_cooldown_seconds < 0 or args.thermal_stable_snapshots < 2:
        parser.error("cooldown must be non-negative and require two clear snapshots")

    token_ids = list(range(args.tokens))
    apc = AutomaticPrefixCache()
    key = APCKey("e5rt-capsule-overlap-gate", cache_layout_fingerprint="plain-bf16")
    apc.store(key, token_ids, [_source_cache(args.tokens, args.heads, args.dim)])
    lookup = apc.lookup(key, token_ids + [args.tokens])
    if not lookup.hit or lookup.capsule_generation is None:
        raise AssertionError("failed to seed APC gate fixture")
    source = capture_kv_cache_plane(
        lookup.cache[0],
        generation=lookup.capsule_generation,
        source_id="compile-geometry",
        target_batch=args.batch,
    )
    spec = E5RTCacheCapsuleSpec.from_source(source)
    build_dir = args.build_dir
    temporary_build = None
    if build_dir is None:
        temporary_build = tempfile.TemporaryDirectory(
            prefix="mlxuag-e5rt-cache-capsule-gate-"
        )
        build_dir = Path(temporary_build.name)

    compile_started = time.perf_counter_ns()
    adapter = E5RTCacheCapsuleAdapter.compile(
        spec,
        build_dir=build_dir,
        expected_revision=args.aneforge_revision,
    )
    compile_ms = (time.perf_counter_ns() - compile_started) / 1e6
    pool = CacheCapsulePool(
        apc.capsule_generation,
        e5rt_adapter=adapter,
        enabled=True,
    )
    independent_work = _IndependentGPUWork(
        args.gpu_work_width, args.gpu_work_repeats
    )
    query = mx.sin(
        mx.arange(args.batch * args.heads * args.dim, dtype=mx.float32)
    ).reshape(args.batch, args.heads, 1, args.dim).astype(mx.bfloat16)
    mx.eval(query)
    mx.synchronize()
    arms = (
        "ordinary_gpu",
        "gpu",
        "cpu",
        "cpu_overlap",
        "e5rt_serial",
        "e5rt_overlap",
    )
    rows = []
    thermal_before = _thermal()
    cpu_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cache-capsule-cpu-overlap"
    )
    try:
        for iteration in range(args.warmup + args.repetitions):
            order = arms if iteration % 2 == 0 else tuple(reversed(arms))
            for slot, arm in enumerate(order):
                row, thermal = _thermal_arm(
                    args,
                    "e5rt_cache_capsule",
                    iteration,
                    0,
                    slot,
                    arm,
                    lambda selected: _run_arm(
                        selected,
                        apc=apc,
                        key=key,
                        token_ids=token_ids,
                        batch=args.batch,
                        query=query,
                        dim=args.dim,
                        independent_work=independent_work,
                        pool=pool,
                        adapter=adapter,
                        cpu_executor=cpu_executor,
                    ),
                )
                if row is None:
                    raise RuntimeError(
                        "thermal gate refused e5rt arm: "
                        f"{thermal.get('abort_reason', 'unknown')}"
                    )
                row["iteration"] = iteration
                row["thermal"] = thermal
                if iteration >= args.warmup:
                    rows.append(row)
                    print(json.dumps({"event": "e5rt_capsule_arm", **row}), flush=True)
    finally:
        cpu_executor.shutdown(wait=True, cancel_futures=True)
        pool.close()
        adapter.close()
        if temporary_build is not None:
            temporary_build.cleanup()

    summaries = _summaries(rows, arms)
    digests = {
        (row["output_digest"], row["key_digest"], row["value_digest"])
        for row in rows
    }
    e5rt_engagements = sum(
        row["adapter_counter_delta"]["executes"]
        for row in rows
        if row["arm"].startswith("e5rt_")
    )
    overlap_ms = summaries["e5rt_overlap"]["median_wall_ms"]
    comparison_ms = min(
        summaries["ordinary_gpu"]["median_wall_ms"],
        summaries["e5rt_serial"]["median_wall_ms"],
        summaries["cpu"]["median_wall_ms"],
        summaries["cpu_overlap"]["median_wall_ms"],
        summaries["gpu"]["median_wall_ms"],
    )
    speedup = comparison_ms / overlap_ms
    exact = len(digests) == 1
    passed = exact and e5rt_engagements > 0 and speedup >= args.minimum_speedup
    receipt = adapter.revision_receipt
    result = {
        "schema": "mlx-uag.e5rt-cache-capsule-overlap-gate.v1",
        "passed": passed,
        "transport_exact": exact,
        "e5rt_engagements": e5rt_engagements,
        "e5rt_overlap_speedup_over_best_serial_constructor": speedup,
        "minimum_speedup": args.minimum_speedup,
        "scope": (
            "standalone real APC KVCache plane, pinned direct-e5rt transport, "
            "independent MLX GPU work, and first MLX attention consumer; no serving"
        ),
        "geometry": {
            "tokens": args.tokens,
            "heads": args.heads,
            "dim": args.dim,
            "batch": args.batch,
            "source_bytes": spec.input_bytes,
            "capsule_bytes": spec.output_bytes,
        },
        "independent_gpu_work": {
            "width": args.gpu_work_width,
            "repeats": args.gpu_work_repeats,
        },
        "compile_ms_excluded_from_trials": compile_ms,
        "aneforge": None
        if receipt is None
        else {
            "root": receipt.root,
            "revision": receipt.revision,
            "tracked_clean": receipt.tracked_clean,
            "module_file": receipt.module_file,
            "requested_device_mask": "0x4 (ANE)",
        },
        "host": _host_receipt(),
        "thermal_before": thermal_before,
        "thermal_after": _thermal(),
        "thermal_controls": {
            "minimum_cooldown_seconds": args.minimum_cooldown_seconds,
            "poll_seconds": args.thermal_poll_seconds,
            "maximum_cooldown_seconds": args.thermal_max_cooldown_seconds,
            "stable_snapshots_required": args.thermal_stable_snapshots,
        },
        "summaries": summaries,
        "adapter_counters": adapter.counters,
        "adapter_timing_ns": adapter.timing_ns,
        "pool_counters": pool.counters,
        "rows": rows,
        "limitations": [
            "This is a standalone cache-plane gate, not live serving evidence.",
            "The cache payload is deterministic BF16 at Qwen-shaped geometry, "
            "not model-produced KV.",
            "The e5rt graph is byte-preserving transport and performs no BF16 arithmetic.",
            "The asynchronous CPU arm uses one persistent host worker and the same "
            "producer/work/wait/consumer schedule as the e5rt overlap arm.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
