# Copyright © 2026 Apple Inc.
"""NVMe-backed Engram PLE tables for Qwen4-Exp (Qwen3.8-Flash-Next).

The 128 q4/g32 PLE shards are ~30% of the release artifact. This module
serves them from a row-interleaved sidecar file (``ple_rows.bin``) instead
of resident memory. Each row is::

    [weight: dims/8 u32 words, 4-bit packed | scales: dims/32 bf16 | biases: dims/32 bf16]

stored shard-major, so ``global_row = shard_index * rows_per_shard + local_row``
— exactly the split ``ShardedEmbedding.lookup_numpy`` computes with
``flat // rows_per_shard``.

Dequantization runs in numpy: unpack the 4-bit values, multiply-add in
float32, round once to bfloat16 (round-to-nearest-even). For q4 the product
``q * scale`` has at most 12 significant bits and, for finite non-overflowing
operands, is exactly representable in float32, so the
single float32 rounding matches MLX's default-stream ``mx.dequantize`` (and
therefore the resident ``nn.QuantizedEmbedding`` gather) bit-for-bit
(nonfinite scales/biases are outside the supported input domain). The
CPU-stream ``mx.dequantize`` kernel rounds through bfloat16 and does NOT
match; the optional "mx" fallback backend therefore dequantizes on the
default stream.

Activated by ``MLX_QWEN4_PLE_NVME=/path/to/ple_rows.bin`` at load time (see
``install_file_backed_ple``). Unset, nothing in this module runs.

The resident-path PLE micro-levers (``MLX_QWEN4_PLE_GATHER_CONCAT``) operate
on ``ShardedEmbedding`` and are superseded here: installing the sidecar
replaces that module, so those flags have no effect in NVMe mode.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MANIFEST_FORMAT = "qwen4-ple-rows"
MANIFEST_VERSION = 1

# Lookups at or above this many row ids (32 tokens x 16 heads) use the
# prefill worker count; smaller (decode / MTP verify) lookups use 16.
PREFILL_ID_THRESHOLD = 512
DECODE_WORKERS = 16
PREFILL_WORKERS = 64
PREFETCH_WORKERS = 16


@dataclass(frozen=True)
class LookupStats:
    """Foreground lookup counters (prefetch reads are not counted).

    ``bytes_read`` counts disk preads only; an LRU hit reads no bytes.
    ``cache_evictions`` includes evictions caused by prefetch inserts.
    """

    lookups: int
    rows: int
    unique_rows: int
    bytes_read: int
    elapsed_seconds: float
    cache_hits: int
    cache_misses: int
    cache_evictions: int


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """View uint16 bfloat16 bits as float32 values."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits with round-to-nearest-even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def dequant_rows_numpy(row_bytes: np.ndarray, dims: int) -> np.ndarray:
    """Dequantize packed q4/g32 rows to bfloat16 bits.

    Args:
        row_bytes: uint8 array of shape ``[n, row_bytes]`` in sidecar layout.
        dims: row width in elements (must be a multiple of 32).

    Returns:
        uint16 array of shape ``[n, dims]`` holding bfloat16 bits.
    """
    n = row_bytes.shape[0]
    weight_bytes = dims // 2
    group_bytes = (dims // 32) * 2
    words = np.ascontiguousarray(row_bytes[:, :weight_bytes]).view(np.uint32)
    scales = np.ascontiguousarray(
        row_bytes[:, weight_bytes : weight_bytes + group_bytes]
    ).view(np.uint16)
    biases = np.ascontiguousarray(
        row_bytes[:, weight_bytes + group_bytes : weight_bytes + 2 * group_bytes]
    ).view(np.uint16)
    shifts = np.uint32(4) * np.arange(8, dtype=np.uint32)
    q = ((words[..., None] >> shifts) & np.uint32(0xF)).astype(np.float32)
    q = q.reshape(n, dims)
    s = np.repeat(bf16_bits_to_f32(scales), 32, axis=1)
    b = np.repeat(bf16_bits_to_f32(biases), 32, axis=1)
    return f32_to_bf16_bits(q * s + b)


def _read_safetensors_header(path):
    """Return (tensor header dict, data section offset) for a safetensors file."""
    import struct

    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def _source_shard_refs(model_path, manifest):
    """Map shard index -> {part: (file, start, row_nbytes)} for the source tensors."""
    model_path = Path(model_path)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = manifest["tensor_prefix"]
    files = sorted(
        {v for k, v in weight_map.items() if k.startswith(prefix + ".shard_")}
    )
    refs = {}
    for file_name in files:
        header, data_offset = _read_safetensors_header(model_path / file_name)
        file_size = os.path.getsize(model_path / file_name)
        for name, info in header.items():
            if not name.startswith(prefix + ".shard_"):
                continue
            shard_index = int(name.split(".shard_")[1].split(".")[0])
            part = name.rsplit(".", 1)[1]
            start, end = info["data_offsets"]
            rows = info["shape"][0]
            if rows <= 0 or end <= start:
                raise ValueError(f"{name} has an empty tensor in {file_name}")
            if start < 0 or data_offset + end > file_size:
                raise ValueError(
                    f"{name} byte range [{start}, {end}) exceeds {file_name} "
                    f"({file_size} bytes)"
                )
            refs.setdefault(shard_index, {})[part] = (
                model_path / file_name,
                data_offset + start,
                (end - start) // rows,
            )
    return refs


def spot_check_sidecar_rows(
    sidecar_path: str, model_path, manifest: dict, num_random: int = 256
) -> int:
    """Compare sampled sidecar rows byte-for-byte against the source tensors.

    The manifest's index digest only proves the sidecar was built against an
    artifact with the same ``model.safetensors.index.json``; this binds the
    check to actual tensor content. Deterministic edge rows (first/last row
    of the first and last shard, plus each shard boundary neighborhood of
    the first shard) and ``num_random`` freshly-drawn random rows are read
    from both the sidecar and the safetensors source. Raises ``ValueError``
    on the first mismatch: a bit-flipped sidecar or a source whose shard
    bytes changed under an unchanged index must both refuse to load.

    Returns the number of rows checked.
    """
    rows_per_shard = manifest["rows_per_shard"]
    total_rows = manifest["total_rows"]
    row_bytes = manifest["row_bytes"]
    refs = _source_shard_refs(model_path, manifest)
    if sorted(refs) != list(range(manifest["num_shards"])):
        raise ValueError(
            f"artifact has shard indices {sorted(refs)[:3]}..., manifest "
            f"expects 0..{manifest['num_shards'] - 1}"
        )

    edges = {
        0,
        rows_per_shard - 1,
        min(rows_per_shard, total_rows - 1),
        total_rows - rows_per_shard,
        total_rows - 1,
    }
    rng = np.random.default_rng()
    picks = sorted(
        edges | {int(r) for r in rng.integers(0, total_rows, size=num_random)}
    )

    handles = {}

    def fh(path):
        if path not in handles:
            handles[path] = open(path, "rb")
        return handles[path]

    try:
        with open(sidecar_path, "rb") as sidecar:
            for global_row in picks:
                shard_index, local = divmod(global_row, rows_per_shard)
                expected = b""
                for part in ("weight", "scales", "biases"):
                    path, start, nbytes = refs[shard_index][part]
                    f = fh(path)
                    f.seek(start + local * nbytes)
                    expected += f.read(nbytes)
                sidecar.seek(manifest["data_offset"] + global_row * row_bytes)
                if sidecar.read(row_bytes) != expected:
                    raise ValueError(
                        f"PLE sidecar row {global_row} (shard {shard_index}, "
                        f"local {local}) does not match the artifact's shard "
                        "tensors: the sidecar is stale or corrupt. Rebuild it "
                        "with scripts/build_qwen4_ple_sidecar.py."
                    )
    finally:
        for f in handles.values():
            f.close()
    return len(picks)


def has_file_backed_ple(model) -> bool:
    """True when any module in ``model`` is an NVMe-backed PLE embedding."""
    return any(
        getattr(module, "is_file_backed", False)
        for _, module in model.named_modules()
    )


def manifest_path(sidecar_path: str) -> str:
    return str(sidecar_path) + ".manifest.json"


def load_manifest(sidecar_path: str) -> dict:
    with open(manifest_path(sidecar_path), "r") as f:
        manifest = json.load(f)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"{manifest_path(sidecar_path)} is not a {MANIFEST_FORMAT} manifest"
        )
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported PLE sidecar manifest version {manifest.get('version')}"
        )
    _validate_manifest_geometry(manifest)
    return manifest


def _validate_manifest_geometry(manifest: dict) -> None:
    """Reject a manifest whose declared layout is internally inconsistent.

    The digest and spot checks bind the sidecar to the artifact; this binds
    the addressing arithmetic (row width, strides, shard split) before any
    field is used to compute a file offset.
    """
    quant = {key: manifest.get(key) for key in ("bits", "group_size", "mode")}
    if quant != {"bits": 4, "group_size": 32, "mode": "affine"}:
        raise ValueError(f"unsupported PLE sidecar quantization: {quant}")
    dims = manifest.get("dims", 0)
    if dims <= 0 or dims % 32:
        raise ValueError(f"PLE sidecar dims={dims} must be a positive multiple of 32")
    groups = dims // 32
    expected = {
        "weight_bytes": dims // 2,
        "scales_bytes": groups * 2,
        "biases_bytes": groups * 2,
        "row_bytes": dims // 2 + 2 * groups * 2,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"PLE sidecar {field}={manifest.get(field)} does not match "
                f"dims={dims} (expected {value})"
            )
    num_shards = manifest.get("num_shards", 0)
    rows_per_shard = manifest.get("rows_per_shard", 0)
    if num_shards <= 0 or rows_per_shard <= 0:
        raise ValueError("PLE sidecar shard counts must be positive")
    if manifest.get("total_rows") != num_shards * rows_per_shard:
        raise ValueError(
            f"PLE sidecar total_rows={manifest.get('total_rows')} != "
            f"{num_shards} shards x {rows_per_shard} rows"
        )
    if manifest.get("data_offset", -1) < 0:
        raise ValueError("PLE sidecar data_offset must be non-negative")
    if len(manifest.get("shard_sha256", [])) != num_shards:
        raise ValueError("PLE sidecar manifest needs one sha256 per shard")


def index_json_sha256(model_path) -> str:
    with open(Path(model_path) / "model.safetensors.index.json", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_sidecar_against_artifact(sidecar_path: str, model_path) -> dict:
    """Check the sidecar manifest and file against the source artifact.

    Verifies the recorded ``model.safetensors.index.json`` digest and the
    sidecar file size. Raises ``ValueError`` on any mismatch. Returns the
    parsed manifest.
    """
    manifest = load_manifest(sidecar_path)
    expected = manifest["source_index_sha256"]
    actual = index_json_sha256(model_path)
    if expected != actual:
        raise ValueError(
            "PLE sidecar was built from a different artifact: manifest "
            f"source_index_sha256={expected} but {model_path} has {actual}. "
            "Rebuild the sidecar with scripts/build_qwen4_ple_sidecar.py."
        )
    expected_size = manifest["data_offset"] + manifest["total_rows"] * manifest[
        "row_bytes"
    ]
    actual_size = os.path.getsize(sidecar_path)
    if actual_size != expected_size:
        raise ValueError(
            f"PLE sidecar {sidecar_path} has size {actual_size}, "
            f"manifest expects {expected_size}"
        )
    return manifest


def assert_sidecar_not_in_weight_files(sidecar_path: str) -> None:
    """The sidecar must never enter ``load_model``'s weight_files glob.

    ``load_model`` globs ``model*.safetensors`` both to load weights and to
    build the ``MLX_LM_UBC_EVICT`` eviction list. The sidecar backs live
    lookups, so evicting its UBC pages would reintroduce cold reads; keeping
    it out of that glob is a hard invariant of the sidecar naming scheme.
    """
    name = os.path.basename(str(sidecar_path))
    if fnmatch.fnmatch(name, "model*.safetensors"):
        raise ValueError(
            f"PLE sidecar name {name!r} matches the model*.safetensors "
            "weight glob; it would be loaded as weights and UBC-evicted. "
            "Name it ple_rows.bin."
        )


class FileBackedShardedEmbedding(nn.Module):
    """Drop-in for ``ShardedEmbedding.lookup_numpy`` reading rows from NVMe.

    Holds no MLX parameters. Per lookup: dedup row ids, ``pread`` only the
    selected 100-byte rows on a thread pool (16 workers for decode-sized
    inputs, 64 for prefill), dequantize them vectorized in numpy, and return
    a bfloat16 ``mx.array`` shaped ``[*ids.shape, dims]``.
    """

    is_file_backed = True

    def __init__(
        self,
        sidecar_path: str,
        vocab_size: int,
        dims: int,
        num_shards: int,
        data_offset: int = 0,
    ):
        super().__init__()
        if vocab_size % num_shards:
            raise ValueError("PLE vocabulary must split evenly across shards")
        if dims % 32:
            raise ValueError("PLE row width must be a multiple of group size 32")
        self.sidecar_path = str(sidecar_path)
        self.vocab_size = vocab_size
        self.dims = dims
        self.num_shards = num_shards
        self.rows_per_shard = vocab_size // num_shards
        self.row_bytes = dims // 2 + 2 * (dims // 32) * 2
        self.data_offset = data_offset
        self.dequant_backend = os.getenv("MLX_QWEN4_PLE_NVME_DEQUANT", "numpy")
        if self.dequant_backend not in {"numpy", "mx"}:
            raise ValueError("MLX_QWEN4_PLE_NVME_DEQUANT must be numpy or mx")
        # Counters are always on (integer increments); wall-clock timing
        # adds two perf_counter calls per foreground lookup, so it is
        # opt-in for benches.
        self.stats_timing = (
            os.getenv("MLX_QWEN4_PLE_NVME_STATS_TIMING") == "1"
        )
        # Optional explicit hot tier: a bytes-capped LRU of packed rows.
        # macOS UBC already caches the sidecar implicitly; this tier's value
        # is immunity to page-cache eviction under memory pressure. 0 = off.
        lru_mb = float(os.getenv("MLX_QWEN4_PLE_NVME_LRU_MB", "0"))
        if lru_mb < 0:
            raise ValueError("MLX_QWEN4_PLE_NVME_LRU_MB must be non-negative")
        self.lru_capacity_rows = int(lru_mb * 2**20) // self.row_bytes
        self.preheated_rows = 0
        self.decode_workers = int(
            os.getenv("MLX_QWEN4_PLE_NVME_DECODE_WORKERS", str(DECODE_WORKERS))
        )
        self.prefill_workers = int(
            os.getenv("MLX_QWEN4_PLE_NVME_PREFILL_WORKERS", str(PREFILL_WORKERS))
        )
        # All lifecycle state (fd, pools, closed flag, owning pid) is
        # guarded by one lock so close() cannot race a submission and a
        # fork cannot inherit dead executor threads unnoticed.
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        # Counters mutate only on the forward (lookup) thread, except
        # cache evictions, which are updated under the LRU lock.
        self._stat_lookups = 0
        self._stat_rows = 0
        self._stat_unique_rows = 0
        self._stat_bytes = 0
        self._stat_elapsed = 0.0
        self._stat_cache_hits = 0
        self._stat_cache_misses = 0
        self._stat_cache_evictions = 0
        self._open_resources()

    def _open_resources(self):
        self._owner_pid = os.getpid()
        # Fork safety: rebuilt (empty) in a forked child alongside the fd
        # and pools, so a lock held by a dead parent thread cannot leak in.
        self._lru = OrderedDict()
        self._lru_lock = threading.Lock()
        self._fd = os.open(self.sidecar_path, os.O_RDONLY)
        self._pool = ThreadPoolExecutor(
            max_workers=max(self.decode_workers, self.prefill_workers),
            thread_name_prefix="ple-nvme",
        )
        # Prefetch runs on its own pool so page-cache warming can never
        # queue behind (or ahead of) a foreground lookup.
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=PREFETCH_WORKERS, thread_name_prefix="ple-nvme-prefetch"
        )

    def _submit(self, use_prefetch_pool: bool, fns, required: bool):
        """Submit ``fns`` atomically with the closed/fork check.

        Fork safety: a forked child inherits executor bookkeeping but none
        of the worker threads, so submissions in the child would hang.
        Detect the pid change and rebuild fd + pools in the child (the
        parent's descriptors stay untouched). Submitting under the
        lifecycle lock makes the closed check atomic with the submission,
        so close() either sees the futures (and drains them before closing
        the fd) or the submission observes the closed flag. ``required``
        submissions raise when closed; optional (prefetch) ones no-op.

        Each fn is called as ``fn(fd)``; the fd stays valid while the
        returned futures may still run because close() drains the pools
        before closing it.
        """
        with self._lifecycle_lock:
            if self._closed:
                if required:
                    raise RuntimeError("FileBackedShardedEmbedding is closed")
                return []
            if os.getpid() != self._owner_pid:
                self._open_resources()
            pool = self._prefetch_pool if use_prefetch_pool else self._pool
            return [pool.submit(fn, self._fd) for fn in fns]

    def close(self):
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            fd, pool, prefetch_pool = self._fd, self._pool, self._prefetch_pool
        # Shut down outside the lock: workers never take the lifecycle
        # lock, and any submission that won the race completes before the
        # fd closes.
        pool.shutdown(wait=True)
        prefetch_pool.shutdown(wait=True)
        os.close(fd)
        with self._lru_lock:
            self._lru.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _workers_for(self, num_ids: int) -> int:
        if num_ids >= PREFILL_ID_THRESHOLD:
            return self.prefill_workers
        return self.decode_workers

    def _pread_rows(self, row_ids: np.ndarray, workers: int) -> np.ndarray:
        """Read ``row_ids`` from disk on the pool; no cache involved."""
        n = int(row_ids.size)
        out = np.empty((n, self.row_bytes), dtype=np.uint8)
        if n == 0:
            return out
        row_bytes = self.row_bytes
        base = self.data_offset

        def read_span(start, stop):
            def task(fd):
                for i in range(start, stop):
                    offset = base + int(row_ids[i]) * row_bytes
                    data = os.pread(fd, row_bytes, offset)
                    if len(data) != row_bytes:
                        raise IOError(
                            f"short pread of PLE row {int(row_ids[i])} "
                            f"({len(data)}/{row_bytes} bytes)"
                        )
                    out[i] = np.frombuffer(data, dtype=np.uint8)

            return task

        workers = max(1, min(workers, n))
        bounds = np.linspace(0, n, workers + 1, dtype=np.int64)
        futures = self._submit(
            False,
            [
                read_span(int(bounds[w]), int(bounds[w + 1]))
                for w in range(workers)
                if bounds[w] < bounds[w + 1]
            ],
            required=True,
        )
        wait(futures)
        for future in futures:
            future.result()
        return out

    def _cache_put(self, row_ids, rows: np.ndarray) -> None:
        """Insert packed rows; evict LRU entries over the byte budget."""
        if not self.lru_capacity_rows:
            return
        with self._lru_lock:
            for row_id, row in zip(np.asarray(row_ids).tolist(), rows):
                self._lru[int(row_id)] = bytes(row)
                self._lru.move_to_end(int(row_id))
            while len(self._lru) > self.lru_capacity_rows:
                self._lru.popitem(last=False)
                self._stat_cache_evictions += 1

    def _read_rows(self, row_ids: np.ndarray, workers: int) -> np.ndarray:
        """Serve ``row_ids`` from the LRU where possible, disk otherwise."""
        n = int(row_ids.size)
        if not self.lru_capacity_rows or n == 0:
            self._stat_bytes += n * self.row_bytes
            return self._pread_rows(row_ids, workers)
        out = np.empty((n, self.row_bytes), dtype=np.uint8)
        missing_positions = []
        with self._lru_lock:
            for i in range(n):
                cached = self._lru.get(int(row_ids[i]))
                if cached is None:
                    missing_positions.append(i)
                else:
                    self._lru.move_to_end(int(row_ids[i]))
                    out[i] = np.frombuffer(cached, dtype=np.uint8)
        self._stat_cache_hits += n - len(missing_positions)
        self._stat_cache_misses += len(missing_positions)
        if missing_positions:
            missing_ids = row_ids[np.asarray(missing_positions, dtype=np.int64)]
            rows = self._pread_rows(missing_ids, workers)
            out[np.asarray(missing_positions, dtype=np.int64)] = rows
            self._cache_put(missing_ids, rows)
            self._stat_bytes += len(missing_positions) * self.row_bytes
        return out

    def _dequant(self, row_bytes: np.ndarray) -> mx.array:
        if self.dequant_backend == "numpy":
            bits = dequant_rows_numpy(row_bytes, self.dims)
            return mx.array(bits).view(mx.bfloat16)
        # Fallback: MLX default-stream dequantize on the packed selection.
        # The default stream is deliberate: the CPU-stream kernel rounds
        # through bfloat16 and is not bit-exact with the resident path.
        weight_bytes = self.dims // 2
        group_bytes = (self.dims // 32) * 2
        w = mx.array(
            np.ascontiguousarray(row_bytes[:, :weight_bytes]).view(np.uint32)
        )
        s = mx.array(
            np.ascontiguousarray(
                row_bytes[:, weight_bytes : weight_bytes + group_bytes]
            ).view(np.uint16)
        ).view(mx.bfloat16)
        b = mx.array(
            np.ascontiguousarray(
                row_bytes[:, weight_bytes + group_bytes :]
            ).view(np.uint16)
        ).view(mx.bfloat16)
        return mx.dequantize(w, s, b, group_size=32, bits=4, mode="affine")

    def lookup_numpy(self, indices: np.ndarray) -> mx.array:
        started = time.perf_counter() if self.stats_timing else None
        shape = indices.shape
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        rows = self._read_rows(unique, self._workers_for(flat.size))
        values = self._dequant(rows)
        result = values[mx.array(inverse.astype(np.int64))].reshape(
            *shape, self.dims
        )
        self._stat_lookups += 1
        self._stat_rows += flat.size
        self._stat_unique_rows += unique.size
        if started is not None:
            self._stat_elapsed += time.perf_counter() - started
        return result

    @property
    def stats(self) -> LookupStats:
        # Hot-path increments may be numpy ints; normalize here (cold path)
        # so the record is JSON-serializable.
        return LookupStats(
            int(self._stat_lookups),
            int(self._stat_rows),
            int(self._stat_unique_rows),
            int(self._stat_bytes),
            float(self._stat_elapsed),
            int(self._stat_cache_hits),
            int(self._stat_cache_misses),
            int(self._stat_cache_evictions),
        )

    def __call__(self, indices: mx.array) -> mx.array:
        mx.eval(indices)
        return self.lookup_numpy(np.asarray(indices, dtype=np.int64))

    def prefetch_rows(self, indices: np.ndarray) -> list:
        """Warm the page cache (and LRU, when enabled) without blocking.

        Fire-and-forget. With the LRU on, already-cached rows are skipped
        and freshly-read rows are inserted, so prefetched rows survive
        page-cache eviction under memory pressure.
        """
        flat = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
        if self.lru_capacity_rows and flat.size:
            with self._lru_lock:
                flat = np.asarray(
                    [i for i in flat.tolist() if i not in self._lru],
                    dtype=np.int64,
                )
        if flat.size == 0:
            return []
        row_bytes = self.row_bytes
        base = self.data_offset
        cache_put = self._cache_put if self.lru_capacity_rows else None
        bounds = np.linspace(
            0, flat.size, min(PREFETCH_WORKERS, flat.size) + 1, dtype=np.int64
        )

        def warm(start, stop):
            def task(fd):
                span = []
                for i in range(start, stop):
                    span.append(
                        os.pread(fd, row_bytes, base + int(flat[i]) * row_bytes)
                    )
                if cache_put is not None:
                    complete = [
                        (int(flat[start + j]), data)
                        for j, data in enumerate(span)
                        if len(data) == row_bytes
                    ]
                    if complete:
                        cache_put(
                            [row_id for row_id, _ in complete],
                            [
                                np.frombuffer(data, dtype=np.uint8)
                                for _, data in complete
                            ],
                        )

            return task

        # Futures are returned for tests/synchronization; production
        # callers ignore them (fire-and-forget).
        return self._submit(
            True,
            [
                warm(int(bounds[w]), int(bounds[w + 1]))
                for w in range(len(bounds) - 1)
                if bounds[w] < bounds[w + 1]
            ],
            required=False,
        )

    def submit_prefetch(self, fn) -> None:
        """Run ``fn`` (id hashing + ``prefetch_rows``) on the prefetch pool.

        ``fn`` takes no arguments; a closed embedding drops it silently.
        """
        self._submit(True, [lambda _fd: fn()], required=False)

    def preheat_from_file(self, path: str) -> int:
        """Load hot-row ids (one per line, ``#`` comments) into the LRU.

        The file is ordered hottest-first (see
        ``scripts/build_qwen4_ple_hot_rows.py``); ids beyond the LRU byte
        budget are dropped. Synchronous and fail-closed: an id outside the
        table refuses. Returns the number of rows cached.
        """
        if not self.lru_capacity_rows:
            raise ValueError(
                "MLX_QWEN4_PLE_NVME_PREHEAT requires MLX_QWEN4_PLE_NVME_LRU_MB "
                "to be set to a positive budget"
            )
        ids = []
        with open(path) as f:
            for line in f:
                text = line.split("#", 1)[0].strip()
                if not text:
                    continue
                row_id = int(text)
                if not 0 <= row_id < self.vocab_size:
                    raise ValueError(
                        f"hot-row id {row_id} in {path} is outside the PLE "
                        f"table [0, {self.vocab_size})"
                    )
                ids.append(row_id)
        unique = np.unique(
            np.asarray(ids[: self.lru_capacity_rows], dtype=np.int64)
        )
        if unique.size == 0:
            return 0
        rows = self._pread_rows(unique, self.prefill_workers)
        self._cache_put(unique, rows)
        return int(unique.size)


def _iter_ple_embeddings(model):
    """Yield ``(weight_key_prefix, NGramEmbedding)`` for every PLE layer."""
    language_model = getattr(model, "language_model", model)
    for index, layer in enumerate(language_model.model.layers):
        ple = getattr(layer, "ple", None)
        if ple is not None:
            prefix = (
                f"language_model.model.layers.{index}"
                ".ple.ple_embedding.ngram_embedding"
            )
            yield prefix, ple.ple_embedding


def install_file_backed_ple(model, weights: dict, sidecar_path: str, model_path):
    """Replace the matching resident ``ShardedEmbedding`` with the sidecar.

    Called from ``load_model`` after ``sanitize`` and before quantization.
    Verifies the manifest against the artifact, swaps in a
    ``FileBackedShardedEmbedding`` (removing the shard modules from the
    module tree so the quantization predicate never visits them), drops the
    ``shard_*`` tensors from ``weights``, and forces the CPU hash backend.

    Returns the pruned weights dict.
    """
    assert_sidecar_not_in_weight_files(sidecar_path)
    manifest = verify_sidecar_against_artifact(sidecar_path, model_path)
    # Bind the sidecar to actual tensor content, not just the index file:
    # sampled + edge rows must match the source bytes (catches both a
    # corrupted sidecar and source shards that changed under an unchanged
    # index). MLX_QWEN4_PLE_NVME_SPOT_CHECK_ROWS sizes the random sample;
    # the deterministic edge rows are always checked.
    spot_check_sidecar_rows(
        sidecar_path,
        model_path,
        manifest,
        num_random=int(os.getenv("MLX_QWEN4_PLE_NVME_SPOT_CHECK_ROWS", "256")),
    )

    installed = False
    for prefix, ngram_embedding in _iter_ple_embeddings(model):
        if prefix != manifest["tensor_prefix"]:
            continue
        resident = ngram_embedding.ngram_embedding
        if getattr(resident, "is_file_backed", False):
            raise ValueError(f"PLE sidecar already installed at {prefix}")
        for field, expected in (
            ("total_rows", resident.vocab_size),
            ("dims", resident.dims),
            ("num_shards", resident.num_shards),
            ("rows_per_shard", resident.rows_per_shard),
        ):
            if manifest[field] != expected:
                raise ValueError(
                    f"PLE sidecar manifest {field}={manifest[field]} does not "
                    f"match the model ({expected}) at {prefix}"
                )
        ngram_embedding.ngram_embedding = FileBackedShardedEmbedding(
            sidecar_path,
            vocab_size=manifest["total_rows"],
            dims=manifest["dims"],
            num_shards=manifest["num_shards"],
            data_offset=manifest["data_offset"],
        )
        # Optional static preheat of the LRU hot tier from a hot-rows
        # manifest (built by scripts/build_qwen4_ple_hot_rows.py). The
        # count lands on the embedding for bench/config observability.
        if preheat := os.environ.get("MLX_QWEN4_PLE_NVME_PREHEAT"):
            ngram_embedding.ngram_embedding.preheated_rows = (
                ngram_embedding.ngram_embedding.preheat_from_file(preheat)
            )
        # NVMe mode hashes and gathers on CPU by construction. The Metal
        # hash backends would put the row ids on the GPU only for the
        # lookup to sync them straight back; refuse the combination.
        if ngram_embedding.hash_backend in ("metal", "metal_prefill"):
            warnings.warn(
                "MLX_QWEN4_PLE_NVME forces CPU n-gram hashing; overriding "
                f"MLX_QWEN4_PLE_HASH_BACKEND={ngram_embedding.hash_backend} "
                "to routed_cpu"
            )
        ngram_embedding.hash_backend = "routed_cpu"
        shard_prefix = f"{prefix}.shard_"
        for key in [k for k in weights if k.startswith(shard_prefix)]:
            del weights[key]
        installed = True

    if not installed:
        raise ValueError(
            f"PLE sidecar tensor_prefix {manifest['tensor_prefix']!r} matches "
            "no PLE layer in this model"
        )
    return weights
