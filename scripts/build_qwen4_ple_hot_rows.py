#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.
"""Build a hot-rows manifest for the Qwen4 PLE NVMe LRU preheat.

Tokenizes a text corpus with the model tokenizer, runs the model's exact
n-gram hash (CPU path, chunked with carried context), counts how often each
global PLE row is selected, and writes the hottest row ids - one per line,
hottest first - for ``MLX_QWEN4_PLE_NVME_PREHEAT``.

CPU-only: the hash needs just the model config's hash constants; the
resident embedding weights stay lazy and are never materialized. Safe to
run beside a live model server.

Example:
    python scripts/build_qwen4_ple_hot_rows.py MODEL_DIR corpus.txt \
        --out MODEL_DIR/ple_hot_rows.txt --budget-mb 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def hash_corpus_row_counts(embedding, tokens, chunk_tokens: int = 8192) -> Counter:
    """Count global PLE row selections over a token stream.

    Hashes in chunks, carrying the previous ``context_len`` tokens so the
    result is identical to hashing the whole stream at once (per-segment
    EOS resets included).
    """
    import mlx.core as mx

    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    counts: Counter = Counter()
    context_len = embedding.context_len
    for start in range(0, tokens.size, chunk_tokens):
        chunk = tokens[start : start + chunk_tokens]
        previous = tokens[max(0, start - context_len) : start]
        if previous.size == 0:
            previous = np.full(context_len, embedding.eos_token_id, dtype=np.int64)
        ids = embedding._ngram_ids_numpy(
            mx.array(chunk[None]), None, previous=previous[None]
        )
        unique, unique_counts = np.unique(ids, return_counts=True)
        counts.update(dict(zip(unique.tolist(), unique_counts.tolist())))
    return counts


def top_rows(counts: Counter, limit: int) -> list[int]:
    """Hottest row ids, count descending, id ascending on ties."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [row_id for row_id, _ in ordered[:limit]]


def write_hot_rows(path, row_ids, header: dict) -> None:
    path = Path(path)
    lines = [f"# qwen4-ple-hot-rows v1 {json.dumps(header, sort_keys=True)}"]
    lines += [str(row_id) for row_id in row_ids]
    path.write_text("\n".join(lines) + "\n")


def build_ngram_embedding(model_dir: Path, ple_layer_index: int):
    """Construct the hash-bearing NGramEmbedding without touching weights."""
    from mlx_lm.models.qwen4_exp import NGramEmbedding, TextModelArgs

    config = json.loads((model_dir / "config.json").read_text())
    args = TextModelArgs.from_dict(config.get("text_config", config))
    if not args.ple_layer_ids:
        raise ValueError(f"{model_dir} has no PLE layers")
    layer_idx = args.ple_layer_ids[ple_layer_index] - 1
    # The ShardedEmbedding parameters stay lazy (never evaluated): only the
    # hash constants are materialized by the CPU id path.
    return NGramEmbedding(args, args.ple_embed_dim, layer_idx, ple_layer_index)


def verify_hash_constants(embedding, model_dir: Path) -> str:
    """Compare derived hash constants against checkpoint overrides.

    ``load_weights`` replaces the constants with checkpoint tensors when
    the artifact carries them; a manifest built from mismatching constants
    would preheat the wrong rows. Raises ``ValueError`` on a mismatch.
    Returns a short status string.
    """
    import mlx.core as mx

    index_file = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_file.read_text())["weight_map"]
    prefix = f"layers.{embedding.layer_idx}.ple.ple_embedding."
    names = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
    keys = {
        name: key
        for name in names
        for key in weight_map
        if key.endswith(prefix + name)
    }
    if not keys:
        return "checkpoint carries no hash-constant overrides; derived values apply"
    for name in names:
        key = keys.get(name)
        if key is None:
            raise ValueError(f"checkpoint stores only some hash constants: missing {name}")
        stored = np.asarray(
            mx.load(str(model_dir / weight_map[key]))[key], dtype=np.int64
        )
        derived = np.asarray(getattr(embedding, name), dtype=np.int64)
        if not np.array_equal(stored, derived):
            raise ValueError(
                f"checkpoint hash constant {key} differs from the value "
                "derived from config.json; rebuild against the right config"
            )
    return f"verified {len(names)} hash constants against the checkpoint"


def rows_for_budget(embedding, budget_mb: float) -> int:
    dims = embedding.ngram_embedding.dims
    row_bytes = dims // 2 + 2 * (dims // 32) * 2
    return int(budget_mb * 2**20) // row_bytes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("corpus", type=Path, help="UTF-8 text corpus")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: MODEL_DIR/ple_hot_rows.txt"
    )
    parser.add_argument("--top-rows", type=int, default=2_000_000)
    parser.add_argument(
        "--budget-mb",
        type=float,
        default=None,
        help="Additional cap: rows that fit the LRU byte budget",
    )
    parser.add_argument("--ple-layer-index", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument(
        "--verify-constants",
        action="store_true",
        help="Check the derived hash constants against checkpoint overrides",
    )
    args = parser.parse_args(argv)

    from mlx_lm.utils import load_tokenizer

    embedding = build_ngram_embedding(args.model_dir, args.ple_layer_index)
    if args.verify_constants:
        print(
            f"[hot-rows] {verify_hash_constants(embedding, args.model_dir)}",
            flush=True,
        )
    else:
        print(
            "[hot-rows] hash constants derived from config.json; pass "
            "--verify-constants to check checkpoint overrides",
            flush=True,
        )
    limit = args.top_rows
    if args.budget_mb is not None:
        limit = min(limit, rows_for_budget(embedding, args.budget_mb))

    tokenizer = load_tokenizer(args.model_dir)
    eos = embedding.eos_token_id
    tokens: list[int] = []
    documents = 0
    # Blank lines separate documents; EOS between documents mirrors the
    # serving-side per-segment hash reset.
    for document in args.corpus.read_text().split("\n\n"):
        if not document.strip():
            continue
        tokens.extend(tokenizer.encode(document))
        tokens.append(eos)
        documents += 1
    if not tokens:
        raise SystemExit(f"corpus {args.corpus} has no text")
    print(
        f"[hot-rows] {documents} documents, {len(tokens)} tokens", flush=True
    )

    started = time.monotonic()
    counts = hash_corpus_row_counts(embedding, tokens, args.chunk_tokens)
    hottest = top_rows(counts, limit)
    out_path = args.out or (args.model_dir / "ple_hot_rows.txt")
    write_hot_rows(
        out_path,
        hottest,
        {
            "model": str(args.model_dir),
            "corpus": str(args.corpus),
            "corpus_tokens": len(tokens),
            "distinct_rows": len(counts),
            "rows_written": len(hottest),
            "ple_layer_index": args.ple_layer_index,
            "created_unix": int(time.time()),
        },
    )
    print(
        f"[hot-rows] wrote {len(hottest)} of {len(counts)} distinct rows to "
        f"{out_path} in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
