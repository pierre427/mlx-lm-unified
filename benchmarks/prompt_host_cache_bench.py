# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""CPU-only miss -> hit -> invalidation gate for server prompt-host reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from mlx_lm.cache_planes import PromptHostPlaneCache
from mlx_lm.server import CompletionRequest, ResponseGenerator


class CPUBenchmarkTokenizer:
    """Deterministic stand-in with non-trivial render/tokenize host work."""

    has_chat_template = True
    has_tool_calling = True
    has_thinking = False
    chat_template = "cpu-benchmark-template-v1"
    init_kwargs = {"_commit_hash": "cpu-benchmark-tokenizer-v1"}

    def __init__(self):
        self.apply_calls = 0

    def __len__(self):
        return 65536

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tools, tokenize, **kwargs
    ):
        assert add_generation_prompt and tokenize
        self.apply_calls += 1
        rendered = json.dumps(
            {"messages": messages, "tools": tools, "kwargs": kwargs},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        pieces = re.findall(r"\w+|[^\w\s]", rendered)
        return [
            int.from_bytes(
                hashlib.blake2s(piece.encode(), digest_size=4).digest(), "big"
            )
            % 65536
            for piece in pieces
        ]


def _generator(max_entries, model_identity):
    generator = ResponseGenerator.__new__(ResponseGenerator)
    generator.model_provider = SimpleNamespace(
        cli_args=SimpleNamespace(
            prompt_host_cache=True,
            prompt_host_cache_size=max_entries,
            chat_template_args={"enable_thinking": False},
        ),
        model_key=(model_identity, None, None),
    )
    generator._prompt_host_cache = PromptHostPlaneCache(max_entries)
    generator._prompt_host_tokenizer = None
    generator._prompt_host_tokenizer_epoch = 0
    return generator


def _request(prompt_characters):
    phrase = "render and tokenize this exact repeated prompt safely. "
    content = (phrase * (prompt_characters // len(phrase) + 1))[:prompt_characters]
    return CompletionRequest(
        request_type="chat",
        prompt="",
        messages=[{"role": "user", "content": content}],
        tools=None,
        role_mapping=None,
    )


def _timed(call):
    start = time.perf_counter_ns()
    result = call()
    return time.perf_counter_ns() - start, result


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def run(
    repetitions, prompt_characters, max_entries, tokenizer=None, model_identity=None
):
    tokenizer = tokenizer or CPUBenchmarkTokenizer()
    model_identity = model_identity or "cpu-benchmark-model"
    request = _request(prompt_characters)
    request_args = SimpleNamespace(chat_template_kwargs=None)
    generator = _generator(max_entries, model_identity)

    misses = []
    hits = []
    invalidated_misses = []
    reference = None
    for index in range(repetitions):
        if index:
            generator._prompt_host_cache.clear("benchmark_cycle")
        elapsed, first = _timed(
            lambda: generator._tokenize(tokenizer, request, request_args)
        )
        misses.append(elapsed)

        elapsed, second = _timed(
            lambda: generator._tokenize(tokenizer, request, request_args)
        )
        hits.append(elapsed)
        if first != second:
            raise RuntimeError("host-cache hit changed tokenization output")

        removed = generator._prompt_host_cache.clear("benchmark_invalidation")
        if removed != 1:
            raise RuntimeError(f"invalidation removed {removed} entries, expected 1")
        elapsed, third = _timed(
            lambda: generator._tokenize(tokenizer, request, request_args)
        )
        invalidated_misses.append(elapsed)
        if first != third:
            raise RuntimeError("post-invalidation miss changed tokenization output")
        reference = first

    stats = generator._prompt_host_cache.stats()
    expected = {
        "lookups": repetitions * 3,
        "hits": repetitions,
        "misses": repetitions * 2,
        "stores": repetitions * 2,
    }
    for name, value in expected.items():
        if stats[name] != value:
            raise RuntimeError(f"{name} counter is {stats[name]}, expected {value}")
    if stats["invalidations"] < repetitions:
        raise RuntimeError("invalidation counter did not engage")

    median_miss = statistics.median(misses)
    median_hit = statistics.median(hits)
    return {
        "benchmark": "server_prompt_host_cache_cpu",
        "prompt_characters": prompt_characters,
        "prompt_tokens": len(reference[0]),
        "repetitions": repetitions,
        "model_identity": model_identity,
        "timing_us": {
            "miss_median": round(median_miss / 1000, 3),
            "miss_p95": round(_percentile(misses, 0.95) / 1000, 3),
            "hit_median": round(median_hit / 1000, 3),
            "hit_p95": round(_percentile(hits, 0.95) / 1000, 3),
            "post_invalidation_miss_median": round(
                statistics.median(invalidated_misses) / 1000, 3
            ),
            "miss_over_hit": round(median_miss / median_hit, 3),
        },
        "tokenizer": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "tokenizer_apply_calls": getattr(tokenizer, "apply_calls", None),
        "cache_stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--prompt-characters", type=int, default=32768)
    parser.add_argument("--max-entries", type=int, default=8)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Optional local model directory whose tokenizer should be benchmarked.",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.prompt_characters < 1 or args.max_entries < 1:
        parser.error("all numeric arguments must be positive")
    tokenizer = None
    model_identity = None
    if args.tokenizer_path is not None:
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer_path)
        model_identity = str(args.tokenizer_path.resolve())
    print(
        json.dumps(
            run(
                args.repetitions,
                args.prompt_characters,
                args.max_entries,
                tokenizer=tokenizer,
                model_identity=model_identity,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
