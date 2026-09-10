"""Compose PR #1555 (Hadamard-rotated K quantization) with asymmetric K/V bits
and KVarN variance normalization (vLLM RFC #46613 / arXiv 2606.03458).

Teacher-forced perplexity over a real-text pack for a grid of KV configs:
fp16, symmetric affine/rotated at 8/4 bits, the asym cross
(K4/V8, K4rot/V8, K4rot/V4, K8/V4 ...), and the KVarN cells (K4/V2, K4/V3
norm-only vs rotated vs norm+rotated — the paper's int4-K/int2-V headline).
Rotation applies to keys only; normalization applies to both K and V.

    python benchmarks/kv_rot_asym_probe.py --model <path> --text-file <f> \
        [--max-tokens N] [--time]

Long context: --max-tokens accepts up to 32768. --time reports wall-clock
per generated token for measuring the ~2% rotation / KVarN decode overhead.
"""

import argparse
import math
import time

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache

MAX_TOKENS_CAP = 32768


def perplexity(model, ids, cache):
    logits = model(ids[None], cache=cache)[0, :-1].astype(mx.float32)
    targets = ids[1:]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    return math.exp(mx.mean(nll).item())


def quant_caches(model, key_bits, value_bits, group_size, rotate, normalize):
    n = len(make_prompt_cache(model))
    return [
        QuantizedKVCache(
            group_size=group_size,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
            normalize=normalize,
        )
        for _ in range(n)
    ]


def decode_ms_per_token(model, tokenizer, cache, warmup_ids, steps):
    """Measure wall-clock ms/token for greedy decode with the given cache.

    The cache is warmed on ``warmup_ids`` (its per-channel normalization scales,
    if any, freeze here) then ``steps`` single tokens are generated. Returns the
    mean ms/token over the timed steps (first step excluded as warmup)."""
    logits = model(warmup_ids[None], cache=cache)[0, -1:]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        logits = model(tok[None], cache=cache)[0, -1:]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        if i > 0:  # drop the first (allocation/compile) step
            times.append((time.perf_counter() - t0) * 1e3)
    return sum(times) / max(1, len(times))


CONFIGS = [
    # (label, key_bits, value_bits, rotate, normalize)
    ("K8/V8 affine", 8, 8, False, False),
    ("K8/V8 rotated", 8, 8, True, False),
    ("K4/V4 affine", 4, 4, False, False),
    ("K4/V4 rotated", 4, 4, True, False),
    ("K4/V4 normalized", 4, 4, False, True),
    ("K4/V4 norm+rotated", 4, 4, True, True),
    ("K4/V8 affine (asym #1550)", 4, 8, False, False),
    ("K4/V8 rotated (compose)", 4, 8, True, False),
    ("K8/V4 affine (asym #1550)", 8, 4, False, False),
    ("K8/V4 rotated", 8, 4, True, False),
    ("K4/V6 rotated", 4, 6, True, False),
    ("K3/V8 rotated", 3, 8, True, False),
    # value-floor sweep: keys high, push values down (rotation is K-only,
    # affine here isolates pure V damage)
    ("K8/V3 affine", 8, 3, False, False),
    ("K8/V2 affine", 8, 2, False, False),
    ("K6/V4 affine", 6, 4, False, False),
    ("K6/V3 affine", 6, 3, False, False),
    ("K5/V4 affine", 5, 4, False, False),
    ("K6/V4 rotated", 6, 4, True, False),
    ("K5/V4 rotated", 5, 4, True, False),
    ("K6/V3 rotated", 6, 3, True, False),
    # KVarN headline: int4-K / int2-3-V near FP16 (norm-only vs rotated vs both)
    ("K4/V2 affine", 4, 2, False, False),
    ("K4/V2 normalized", 4, 2, False, True),
    ("K4/V2 rotated", 4, 2, True, False),
    ("K4/V2 norm+rotated", 4, 2, True, True),
    ("K4/V3 affine", 4, 3, False, False),
    ("K4/V3 normalized", 4, 3, False, True),
    ("K4/V3 rotated", 4, 3, True, False),
    ("K4/V3 norm+rotated", 4, 3, True, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help=f"prompt length for the perplexity gate (<= {MAX_TOKENS_CAP})",
    )
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--configs",
        default=None,
        help="comma-separated label substrings to select a subset of CONFIGS",
    )
    ap.add_argument(
        "--time",
        action="store_true",
        help="also report wall-clock ms/token for greedy decode per config",
    )
    ap.add_argument(
        "--time-steps",
        type=int,
        default=32,
        help="number of decode steps to time when --time is set",
    )
    args = ap.parse_args()

    if args.max_tokens > MAX_TOKENS_CAP:
        raise SystemExit(
            f"--max-tokens {args.max_tokens} exceeds the {MAX_TOKENS_CAP} cap"
        )

    model, tokenizer = load(args.model)
    text = open(args.text_file).read()
    ids = mx.array(tokenizer.encode(text)[: args.max_tokens])
    print(f"model={args.model}  tokens={len(ids)}  group_size={args.group_size}")

    base = perplexity(model, ids, make_prompt_cache(model))
    header = f"  {'fp16 KV':34s} ppl = {base:10.2f}"
    if args.time:
        base_ms = decode_ms_per_token(
            model, tokenizer, make_prompt_cache(model), ids, args.time_steps
        )
        header += f"   {base_ms:8.2f} ms/tok"
    print(header)

    configs = CONFIGS
    if args.configs:
        wanted = [w.strip() for w in args.configs.split(",")]
        configs = [c for c in CONFIGS if any(w in c[0] for w in wanted)]

    for label, kb, vb, rot, norm in configs:
        cache = quant_caches(model, kb, vb, args.group_size, rot, norm)
        ppl = perplexity(model, ids, cache)
        line = f"  {label:34s} ppl = {ppl:10.2f}"
        if args.time:
            cache = quant_caches(model, kb, vb, args.group_size, rot, norm)
            ms = decode_ms_per_token(
                model, tokenizer, cache, ids, args.time_steps
            )
            overhead = 100.0 * (ms / base_ms - 1.0)
            line += f"   {ms:8.2f} ms/tok ({overhead:+.1f}%)"
        print(line)


if __name__ == "__main__":
    main()
