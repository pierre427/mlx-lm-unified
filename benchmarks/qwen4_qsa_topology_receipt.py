#!/usr/bin/env python3
"""Emit a diagnostic QSA selected-base-page topology receipt.

The default mode is a CPU-only synthetic cohort.  ``--compact-json`` consumes
a previously hosted compact selection with keys ``block_ids``,
``block_counts``, ``base_page_count``, and ``page_size_tokens``.  Dynamic MLX
selections can be emitted by the opt-in runtime hook and summarized with
``--receipt-dir``. A capture run is diagnostic and cannot report performance;
production inference never imports or calls this benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlx_lm.qsa_topology_receipt import (
    analyze_qsa_topology_receipts,
    build_qsa_topology_receipt,
)


def _synthetic(args) -> tuple[np.ndarray, np.ndarray]:
    if args.selected_pages > args.base_pages:
        raise ValueError("selected pages cannot exceed base pages")
    shared = round(args.selected_pages * args.overlap)
    shared = min(args.selected_pages, max(0, shared))
    ids = np.zeros((args.batch, args.queries, args.selected_pages), dtype=np.uint32)
    common = list(range(shared))
    available = list(range(shared, args.base_pages))
    private_width = args.selected_pages - shared
    for row in range(args.batch):
        for query in range(args.queries):
            if private_width:
                start = (row * private_width + query) % len(available)
                private = [
                    available[(start + index) % len(available)]
                    for index in range(private_width)
                ]
            else:
                private = []
            ids[row, query] = sorted(common + private)
    counts = np.full((args.batch, args.queries), args.selected_pages, dtype=np.int32)
    return ids, counts


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--compact-json", type=Path)
    source.add_argument(
        "--receipt-dir",
        type=Path,
        help="analyze qsa-topology-*.json files emitted by the runtime hook",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--base-pages", type=int, default=4096)
    parser.add_argument("--page-size-tokens", type=int, default=4)
    parser.add_argument("--selected-pages", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = _args()
    if not 0.0 <= args.overlap <= 1.0:
        raise ValueError("overlap must be in [0, 1]")
    if args.receipt_dir is not None:
        paths = sorted(args.receipt_dir.glob("qsa-topology-*.json"))
        if not paths:
            raise ValueError("receipt directory contains no topology receipts")
        receipts = [json.loads(path.read_text()) for path in paths]
        payload = {
            "source": {
                "kind": "runtime_receipt_directory",
                "path": str(args.receipt_dir),
                "files": [path.name for path in paths],
            },
            "analysis": analyze_qsa_topology_receipts(receipts),
        }
    elif args.compact_json is not None:
        payload = json.loads(args.compact_json.read_text())
        ids = payload["block_ids"]
        counts = payload["block_counts"]
        base_pages = int(payload["base_page_count"])
        page_size = int(payload["page_size_tokens"])
        source = {"kind": "hosted_compact_json", "path": str(args.compact_json)}
    else:
        ids, counts = _synthetic(args)
        base_pages = args.base_pages
        page_size = args.page_size_tokens
        source = {
            "kind": "synthetic_cpu",
            "batch": args.batch,
            "queries": args.queries,
            "selected_pages": args.selected_pages,
            "requested_overlap": args.overlap,
        }
    if args.receipt_dir is None:
        receipt = build_qsa_topology_receipt(
            ids,
            counts,
            base_page_count=base_pages,
            page_size_tokens=page_size,
        )
        payload = {
            "source": source,
            "topology": receipt,
            "analysis": analyze_qsa_topology_receipts([receipt]),
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)


if __name__ == "__main__":
    main()
