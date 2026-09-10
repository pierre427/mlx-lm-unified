"""Interleaved throughput A/B for the hybrid prompt-lookup proposer.

This is a model-loading benchmark and must only be run by the current GPU owner.
It compares the shipped suffix-automaton proposer against an otherwise-identical
primary with a frozen datastore fallback. Results are append-only JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import mlx.core as mx

from mlx_lm import load
from mlx_lm.generate import generate_step, prompt_lookup_generate_step
from mlx_lm.hybrid_proposer import (
    DatastoreProposer,
    HybridProposer,
    HybridProposerStats,
)
from mlx_lm.prompt_lookup import HybridStats, SuffixAutomatonProposer
from mlx_lm.sample_utils import make_sampler


BUILTIN_DOCUMENTS = [
    """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
    """def binary_search(values, target):
    low, high = 0, len(values) - 1
    while low <= high:
        mid = (low + high) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
    """function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}
""",
    """SELECT customer_id, COUNT(*) AS order_count
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY customer_id
ORDER BY order_count DESC;
""",
    """def chunked(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    for index in range(0, len(items), size):
        yield items[index : index + size]
""",
    """#!/usr/bin/env bash
set -euo pipefail
source_dir="${1:?source directory required}"
destination="${2:?destination required}"
timestamp="$(date +%Y%m%d-%H%M%S)"
tar -czf "${destination}/backup-${timestamp}.tar.gz" "${source_dir}"
""",
    """name: test
on:
  pull_request:
  push:
    branches: [main]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: python -m unittest discover
""",
    """<table class="metrics">
  <thead>
    <tr><th>Name</th><th>Value</th></tr>
  </thead>
  <tbody>
    <tr><td>requests</td><td>128</td></tr>
    <tr><td>errors</td><td>0</td></tr>
  </tbody>
</table>
""",
]

BUILTIN_PROMPTS = [
    {
        "id": "python_fibonacci",
        "prompt": "Complete the implementation:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n",
    },
    {
        "id": "python_binary_search",
        "prompt": (
            "Complete the implementation:\n\n```python\n"
            "def binary_search(values, target):\n"
            "    low, high = 0, len(values) - 1\n"
        ),
    },
    {
        "id": "javascript_debounce",
        "prompt": (
            "Complete this JavaScript helper:\n\n```javascript\n"
            "function debounce(fn, delay) {\n  let timer;\n"
        ),
    },
    {
        "id": "sql_orders",
        "prompt": (
            "Complete this SQL query:\n\n```sql\n"
            "SELECT customer_id, COUNT(*) AS order_count\nFROM orders\n"
        ),
    },
    {
        "id": "python_chunked",
        "prompt": (
            "Complete the implementation:\n\n```python\n"
            "def chunked(items, size):\n    if size <= 0:\n"
        ),
    },
    {
        "id": "bash_backup",
        "prompt": (
            "Complete this shell script:\n\n```bash\n#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'source_dir="${1:?source directory required}"\n'
        ),
    },
    {
        "id": "yaml_ci",
        "prompt": (
            "Complete this workflow verbatim:\n\n```yaml\n"
            "name: test\non:\n  pull_request:\n  push:\n"
        ),
    },
    {
        "id": "html_metrics",
        "prompt": "Complete this HTML table:\n\n```html\n<table class=\"metrics\">\n  <thead>\n",
    },
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _load_workload(args) -> tuple[list[str], list[dict]]:
    documents = list(BUILTIN_DOCUMENTS)
    prompts = list(BUILTIN_PROMPTS)
    if args.datastore_jsonl:
        documents = []
        for row in _read_jsonl(args.datastore_jsonl):
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("each datastore JSONL row needs a string 'text'")
            documents.append(text)
    if args.prompts_jsonl:
        prompts = _read_jsonl(args.prompts_jsonl)
        for row in prompts:
            if not isinstance(row.get("id"), str) or not isinstance(
                row.get("prompt"), str
            ):
                raise ValueError("each prompt JSONL row needs string 'id' and 'prompt'")
    if not documents or not prompts:
        raise ValueError("workload requires at least one document and one prompt")
    return documents, prompts


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _bootstrap_documents(model, tokenizer, prompts: list[dict], max_tokens: int):
    """Build a persistent datastore from prior greedy prompt/completion pairs."""

    sampler = make_sampler(temp=0.0)
    documents = []
    for row in prompts:
        prompt_tokens = _encode(tokenizer, row["prompt"])
        completion = []
        generator = generate_step(
            mx.array(prompt_tokens), model, max_tokens=max_tokens, sampler=sampler
        )
        try:
            for token, _ in generator:
                token = int(token)
                completion.append(token)
                if token in tokenizer.eos_token_ids:
                    break
        finally:
            generator.close()
        documents.append(prompt_tokens + completion)
        print(
            f"bootstrapped {row['id']}: {len(prompt_tokens)} prompt + "
            f"{len(completion)} completion tokens",
            flush=True,
        )
    return documents


def _order(pattern: str) -> Iterable[str]:
    for name in pattern:
        yield "baseline" if name == "a" else "hybrid"


def _run_once(
    *,
    model,
    tokenizer,
    prompt: str,
    backend_name: str,
    datastore: DatastoreProposer,
    args,
) -> dict:
    primary = SuffixAutomatonProposer(
        min_match=args.primary_min_match,
        max_lookback=args.primary_max_lookback,
    )
    source_stats = None
    backend = primary
    if backend_name == "hybrid":
        source_stats = HybridProposerStats()
        backend = HybridProposer(
            primary,
            datastore,
            source_stats,
            datastore_cooldown=args.datastore_cooldown,
            datastore_warmup_tokens=args.datastore_warmup_tokens,
        )

    prompt_tokens = _encode(tokenizer, prompt)
    generation_stats = HybridStats()
    generator = prompt_lookup_generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0),
        backend=backend,
        num_draft=args.num_draft,
        adaptive=args.adaptive_latch,
        cliff_aware_span=args.cliff_aware_span,
        warmup=args.warmup,
        gate=args.gate,
        stats=generation_stats,
    )

    output = []
    first_token_at = None
    mx.synchronize()
    started = time.perf_counter()
    try:
        for token, _, _ in generator:
            if first_token_at is None:
                mx.synchronize()
                first_token_at = time.perf_counter()
            token = int(token)
            output.append(token)
            if token in tokenizer.eos_token_ids:
                break
    finally:
        generator.close()
    mx.synchronize()
    finished = time.perf_counter()

    total_seconds = finished - started
    decode_seconds = (
        finished - first_token_at if first_token_at is not None else total_seconds
    )
    decode_token_count = max(len(output) - 1, 0)
    row = {
        "kind": "run",
        "backend": backend_name,
        "prompt_tokens": len(prompt_tokens),
        "output_tokens": len(output),
        "total_seconds": total_seconds,
        "ttft_seconds": (
            first_token_at - started if first_token_at is not None else None
        ),
        "decode_tokens_per_second": (
            decode_token_count / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "output_sha256": hashlib.sha256(
            ",".join(map(str, output)).encode("ascii")
        ).hexdigest(),
        "generation_stats": asdict(generation_stats),
        "source_stats": source_stats.as_dict() if source_stats else None,
    }
    return row


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datastore-jsonl", type=Path)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--bootstrap-datastore", action="store_true")
    parser.add_argument("--bootstrap-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--order", choices=("ab", "ba", "abba", "baab"), default="abba")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-draft", type=int, default=8)
    parser.add_argument("--primary-min-match", type=int, default=3)
    parser.add_argument("--primary-max-lookback", type=int, default=32)
    parser.add_argument("--datastore-min-match", type=int, default=3)
    parser.add_argument("--datastore-window", type=int, default=64)
    parser.add_argument("--datastore-cooldown", type=int, default=0)
    parser.add_argument("--datastore-warmup-tokens", type=int, default=0)
    parser.add_argument("--adaptive-span", action="store_true")
    parser.add_argument("--cliff-aware-span", action="store_true")
    parser.add_argument("--span-scale", type=float, default=1.0)
    parser.add_argument("--min-span", type=int, default=1)
    parser.add_argument("--adaptive-latch", action="store_true")
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--gate", type=float, default=0.12)
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    if args.bootstrap_datastore and args.datastore_jsonl:
        parser.error("--bootstrap-datastore conflicts with --datastore-jsonl")
    if args.bootstrap_tokens < 1:
        parser.error("--bootstrap-tokens must be >= 1")
    documents, prompts = _load_workload(args)
    model, tokenizer = load(args.model)
    datastore_kwargs = {
        "min_match": args.datastore_min_match,
        "window": args.datastore_window,
        "adaptive_span": args.adaptive_span,
        "span_scale": args.span_scale,
        "min_span": args.min_span,
    }
    if args.bootstrap_datastore:
        token_documents = _bootstrap_documents(
            model, tokenizer, prompts, args.bootstrap_tokens
        )
        datastore = DatastoreProposer(token_documents, **datastore_kwargs)
        datastore_source = "prior_model_completions"
    else:
        datastore = DatastoreProposer.from_texts(
            documents, tokenizer, **datastore_kwargs
        )
        datastore_source = "static_text_corpus"
    metadata = {
        "kind": "metadata",
        "model": args.model,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "datastore_stats": asdict(datastore.stats),
        "datastore_footprint_bytes": datastore.footprint_bytes(),
        "datastore_source": datastore_source,
        "prompt_ids": [row["id"] for row in prompts],
    }
    _append_jsonl(args.output, metadata)

    # ABBA/BAAB within each prompt and repetition interleaves thermal drift.
    for repetition in range(args.repetitions):
        for prompt_row in prompts:
            for order_index, backend_name in enumerate(_order(args.order)):
                row = _run_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_row["prompt"],
                    backend_name=backend_name,
                    datastore=datastore,
                    args=args,
                )
                row.update(
                    {
                        "repetition": repetition,
                        "order_index": order_index,
                        "prompt_id": prompt_row["id"],
                    }
                )
                _append_jsonl(args.output, row)
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
