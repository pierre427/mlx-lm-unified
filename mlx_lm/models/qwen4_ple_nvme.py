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
``q * scale`` has at most 12 significant bits and is exact in float32, so the
single float32 rounding matches MLX's default-stream ``mx.dequantize`` (and
therefore the resident ``nn.QuantizedEmbedding`` gather) bit-for-bit. The
CPU-stream ``mx.dequantize`` kernel rounds through bfloat16 and does NOT
match; the optional "mx" fallback backend therefore dequantizes on the
default stream.

Activated by ``MLX_QWEN4_PLE_NVME=/path/to/ple_rows.bin`` at load time (see
``install_file_backed_ple``). Unset, nothing in this module runs.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
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
    return manifest


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
        self.decode_workers = int(
            os.getenv("MLX_QWEN4_PLE_NVME_DECODE_WORKERS", str(DECODE_WORKERS))
        )
        self.prefill_workers = int(
            os.getenv("MLX_QWEN4_PLE_NVME_PREFILL_WORKERS", str(PREFILL_WORKERS))
        )
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
        self._closed = False
        self._close_lock = threading.Lock()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True)
        self._prefetch_pool.shutdown(wait=True)
        os.close(self._fd)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _workers_for(self, num_ids: int) -> int:
        if num_ids >= PREFILL_ID_THRESHOLD:
            return self.prefill_workers
        return self.decode_workers

    def _read_rows(self, row_ids: np.ndarray, workers: int) -> np.ndarray:
        n = int(row_ids.size)
        out = np.empty((n, self.row_bytes), dtype=np.uint8)
        if n == 0:
            return out
        row_bytes = self.row_bytes
        fd = self._fd
        base = self.data_offset

        def read_span(start: int, stop: int):
            for i in range(start, stop):
                offset = base + int(row_ids[i]) * row_bytes
                data = os.pread(fd, row_bytes, offset)
                if len(data) != row_bytes:
                    raise IOError(
                        f"short pread of PLE row {int(row_ids[i])} "
                        f"({len(data)}/{row_bytes} bytes)"
                    )
                out[i] = np.frombuffer(data, dtype=np.uint8)

        workers = max(1, min(workers, n))
        if workers == 1:
            read_span(0, n)
            return out
        bounds = np.linspace(0, n, workers + 1, dtype=np.int64)
        futures = [
            self._pool.submit(read_span, int(bounds[w]), int(bounds[w + 1]))
            for w in range(workers)
            if bounds[w] < bounds[w + 1]
        ]
        wait(futures)
        for future in futures:
            future.result()
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
        shape = indices.shape
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        rows = self._read_rows(unique, self._workers_for(flat.size))
        values = self._dequant(rows)
        return values[mx.array(inverse.astype(np.int64))].reshape(
            *shape, self.dims
        )

    def __call__(self, indices: mx.array) -> mx.array:
        mx.eval(indices)
        return self.lookup_numpy(np.asarray(indices, dtype=np.int64))

    def prefetch_rows(self, indices: np.ndarray) -> None:
        """Warm the page cache for ``indices`` without blocking. Fire-and-forget."""
        flat = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
        if flat.size == 0 or self._closed:
            return
        row_bytes = self.row_bytes
        fd = self._fd
        base = self.data_offset
        bounds = np.linspace(
            0, flat.size, min(PREFETCH_WORKERS, flat.size) + 1, dtype=np.int64
        )

        def warm(start: int, stop: int):
            for i in range(start, stop):
                os.pread(fd, row_bytes, base + int(flat[i]) * row_bytes)

        for w in range(len(bounds) - 1):
            if bounds[w] < bounds[w + 1]:
                self._prefetch_pool.submit(warm, int(bounds[w]), int(bounds[w + 1]))

    def submit_prefetch(self, fn) -> None:
        """Run ``fn`` (id hashing + ``prefetch_rows``) on the prefetch pool."""
        if not self._closed:
            self._prefetch_pool.submit(fn)


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
