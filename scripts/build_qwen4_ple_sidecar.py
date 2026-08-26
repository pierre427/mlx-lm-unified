#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.
"""Build the row-interleaved NVMe PLE sidecar for Qwen4-Exp artifacts.

Streams the ``ple.ple_embedding.ngram_embedding.shard_*`` tensors
(q4/g32: ``weight`` U32 ``[rows, dims/8]``, ``scales``/``biases`` BF16
``[rows, dims/32]``) out of the artifact's safetensors files into
``ple_rows.bin``. Each output row is::

    [dims/2 bytes packed q4 weight | (dims/32)*2 bytes scales | (dims/32)*2 bytes biases]

written shard-major, so ``global_row = shard_index * rows_per_shard +
local_row`` — the same split ``ShardedEmbedding.lookup_numpy`` computes.

A JSON manifest (``<out>.manifest.json``) records the layout, a sha256 per
shard's row bytes, and the sha256 of ``model.safetensors.index.json`` so the
loader can refuse a sidecar built from a different artifact.

Uses only the standard library and numpy: safe to run beside a live model
server (launch under background QoS: ``taskpolicy -B``).

Examples:
    python scripts/build_qwen4_ple_sidecar.py MODEL_DIR --out .../ple_rows.bin
    python scripts/build_qwen4_ple_sidecar.py MODEL_DIR --out .../ple_rows.bin \
        --verify --verify-rows 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

MANIFEST_FORMAT = "qwen4-ple-rows"
MANIFEST_VERSION = 1
SHARD_MARKER = ".ple.ple_embedding.ngram_embedding.shard_"
# Rows per streaming block: 262144 rows x 100 B ~ 26 MB resident per buffer.
DEFAULT_BLOCK_ROWS = 262144


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Return (tensor header dict, data section offset)."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class TensorRef:
    """Byte-range view of one tensor inside a safetensors file."""

    def __init__(self, file: Path, dtype: str, shape, start: int, end: int):
        self.file = file
        self.dtype = dtype
        self.shape = tuple(shape)
        self.start = start
        self.end = end

    @property
    def row_nbytes(self) -> int:
        return (self.end - self.start) // self.shape[0]

    def read_rows(self, f, row_start: int, row_stop: int) -> bytes:
        f.seek(self.start + row_start * self.row_nbytes)
        return f.read((row_stop - row_start) * self.row_nbytes)


def collect_shards(model_dir: Path) -> tuple[str, list[dict]]:
    """Map every PLE shard to its three TensorRefs, in shard order."""
    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = sorted(
        {v for k, v in weight_map.items() if SHARD_MARKER in k}
    )
    refs: dict[str, TensorRef] = {}
    for file_name in shard_files:
        header, data_offset = read_safetensors_header(model_dir / file_name)
        for name, info in header.items():
            if SHARD_MARKER in name:
                start, end = info["data_offsets"]
                refs[name] = TensorRef(
                    model_dir / file_name,
                    info["dtype"],
                    info["shape"],
                    data_offset + start,
                    data_offset + end,
                )

    prefixes = {k[: k.index(".shard_")] for k in refs}
    if len(prefixes) != 1:
        raise ValueError(f"expected one ngram_embedding prefix, found {prefixes}")
    prefix = prefixes.pop()

    indices = sorted(
        {int(k.split(".shard_")[1].split(".")[0]) for k in refs}
    )
    if indices != list(range(len(indices))):
        raise ValueError(f"non-contiguous shard indices: {indices[:5]}...")

    shards = []
    for i in indices:
        entry = {}
        for part, dtype in (("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")):
            ref = refs[f"{prefix}.shard_{i}.{part}"]
            if ref.dtype != dtype:
                raise ValueError(
                    f"shard_{i}.{part} has dtype {ref.dtype}, expected {dtype}"
                )
            entry[part] = ref
        rows = {entry[p].shape[0] for p in ("weight", "scales", "biases")}
        if len(rows) != 1:
            raise ValueError(f"shard_{i} has inconsistent row counts {rows}")
        shards.append(entry)

    rows_per_shard = {s["weight"].shape[0] for s in shards}
    if len(rows_per_shard) != 1:
        raise ValueError(f"shards disagree on rows_per_shard: {rows_per_shard}")
    groups = shards[0]["scales"].shape[1]
    words = shards[0]["weight"].shape[1]
    if words * 8 != groups * 32:
        raise ValueError(
            f"weight words ({words}) and scale groups ({groups}) disagree"
        )
    return prefix, shards


def interleave_block(weight: bytes, scales: bytes, biases: bytes, layout) -> np.ndarray:
    rows = len(weight) // layout["weight_bytes"]
    out = np.empty((rows, layout["row_bytes"]), dtype=np.uint8)
    wb, gb = layout["weight_bytes"], layout["group_bytes"]
    out[:, :wb] = np.frombuffer(weight, dtype=np.uint8).reshape(rows, wb)
    out[:, wb : wb + gb] = np.frombuffer(scales, dtype=np.uint8).reshape(rows, gb)
    out[:, wb + gb :] = np.frombuffer(biases, dtype=np.uint8).reshape(rows, gb)
    return out


def build(model_dir: Path, out_path: Path, block_rows: int) -> dict:
    prefix, shards = collect_shards(model_dir)
    rows_per_shard = shards[0]["weight"].shape[0]
    groups = shards[0]["scales"].shape[1]
    dims = groups * 32
    layout = {
        "weight_bytes": dims // 2,
        "group_bytes": groups * 2,
        "row_bytes": dims // 2 + 2 * groups * 2,
    }
    total_rows = rows_per_shard * len(shards)
    total_bytes = total_rows * layout["row_bytes"]
    print(
        f"[build] {len(shards)} shards x {rows_per_shard} rows, dims={dims}, "
        f"row={layout['row_bytes']} B, total {total_bytes / 2**30:.2f} GiB",
        flush=True,
    )

    partial = out_path.with_name(out_path.name + ".partial")
    shard_digests = []
    started = time.monotonic()
    open_files: dict[Path, object] = {}

    def fh(path: Path):
        if path not in open_files:
            open_files[path] = open(path, "rb")
        return open_files[path]

    try:
        with open(partial, "wb") as out:
            for index, shard in enumerate(shards):
                digest = hashlib.sha256()
                for row_start in range(0, rows_per_shard, block_rows):
                    row_stop = min(row_start + block_rows, rows_per_shard)
                    block = interleave_block(
                        shard["weight"].read_rows(
                            fh(shard["weight"].file), row_start, row_stop
                        ),
                        shard["scales"].read_rows(
                            fh(shard["scales"].file), row_start, row_stop
                        ),
                        shard["biases"].read_rows(
                            fh(shard["biases"].file), row_start, row_stop
                        ),
                        layout,
                    )
                    data = block.tobytes()
                    digest.update(data)
                    out.write(data)
                shard_digests.append(digest.hexdigest())
                if (index + 1) % 8 == 0 or index + 1 == len(shards):
                    done = (index + 1) * rows_per_shard * layout["row_bytes"]
                    rate = done / max(time.monotonic() - started, 1e-9) / 2**20
                    print(
                        f"[build] shard {index + 1}/{len(shards)} "
                        f"({done / 2**30:.2f} GiB, {rate:.0f} MiB/s)",
                        flush=True,
                    )
    finally:
        for f in open_files.values():
            f.close()

    os.replace(partial, out_path)

    with open(model_dir / "model.safetensors.index.json", "rb") as f:
        index_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "tensor_prefix": prefix,
        "dims": dims,
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
        "weight_bytes": layout["weight_bytes"],
        "scales_bytes": layout["group_bytes"],
        "biases_bytes": layout["group_bytes"],
        "row_bytes": layout["row_bytes"],
        "num_shards": len(shards),
        "rows_per_shard": rows_per_shard,
        "total_rows": total_rows,
        "data_offset": 0,
        "source_model_path": str(model_dir),
        "source_index_sha256": index_sha,
        "shard_sha256": shard_digests,
        "created_unix": int(time.time()),
    }
    manifest_file = out_path.with_name(out_path.name + ".manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build] wrote {out_path} and {manifest_file}", flush=True)
    return manifest


def verify(model_dir: Path, out_path: Path, num_rows: int, seed: int) -> int:
    """Re-read random sidecar rows and check them against the safetensors source.

    Returns the number of mismatching rows (0 = pass). Also re-checks the
    manifest's source index digest.
    """
    with open(out_path.with_name(out_path.name + ".manifest.json")) as f:
        manifest = json.load(f)
    with open(model_dir / "model.safetensors.index.json", "rb") as f:
        index_sha = hashlib.sha256(f.read()).hexdigest()
    if index_sha != manifest["source_index_sha256"]:
        print(
            f"[verify] FAIL: index digest {index_sha} != manifest "
            f"{manifest['source_index_sha256']}"
        )
        return 1

    expected_size = manifest["total_rows"] * manifest["row_bytes"]
    actual_size = os.path.getsize(out_path)
    if actual_size != expected_size:
        print(f"[verify] FAIL: size {actual_size} != expected {expected_size}")
        return 1

    prefix, shards = collect_shards(model_dir)
    if prefix != manifest["tensor_prefix"]:
        print(f"[verify] FAIL: prefix {prefix} != manifest {manifest['tensor_prefix']}")
        return 1

    rng = random.Random(seed)
    rows_per_shard = manifest["rows_per_shard"]
    row_bytes = manifest["row_bytes"]
    wb = manifest["weight_bytes"]
    gb = manifest["scales_bytes"]
    picks = sorted(
        rng.randrange(manifest["total_rows"]) for _ in range(num_rows)
    )
    mismatches = 0
    open_files: dict[Path, object] = {}

    def fh(path: Path):
        if path not in open_files:
            open_files[path] = open(path, "rb")
        return open_files[path]

    try:
        with open(out_path, "rb") as sidecar:
            for global_row in picks:
                shard_index, local = divmod(global_row, rows_per_shard)
                shard = shards[shard_index]
                expected = (
                    shard["weight"].read_rows(
                        fh(shard["weight"].file), local, local + 1
                    )
                    + shard["scales"].read_rows(
                        fh(shard["scales"].file), local, local + 1
                    )
                    + shard["biases"].read_rows(
                        fh(shard["biases"].file), local, local + 1
                    )
                )
                sidecar.seek(global_row * row_bytes)
                actual = sidecar.read(row_bytes)
                if actual != expected:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            f"[verify] row {global_row} (shard {shard_index}, "
                            f"local {local}) mismatch"
                        )
    finally:
        for f in open_files.values():
            f.close()

    status = "PASS" if mismatches == 0 else "FAIL"
    print(
        f"[verify] {status}: {num_rows} random rows checked, "
        f"{mismatches} mismatches (weight {wb} B + scales {gb} B + "
        f"biases {gb} B per row)"
    )
    return mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", type=Path, help="MLX artifact directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Sidecar path (default: MODEL_DIR/ple_rows.bin)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing sidecar instead of building",
    )
    parser.add_argument("--verify-rows", type=int, default=4096)
    parser.add_argument("--verify-seed", type=int, default=0)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    args = parser.parse_args(argv)

    out_path = args.out or (args.model_dir / "ple_rows.bin")
    if args.verify:
        return 1 if verify(
            args.model_dir, out_path, args.verify_rows, args.verify_seed
        ) else 0
    build(args.model_dir, out_path, args.block_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
