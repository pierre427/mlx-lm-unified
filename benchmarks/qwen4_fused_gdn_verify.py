#!/usr/bin/env python3
"""Real-weight gate for the fused Qwen4 GDN speculative-verify kernel.

Plan-only by default; ``--execute-metal`` plus a local checkpoint runs three
resident phases on one loaded model, each comparing the stock verify path
against the fused kernel bit for bit:

1. ``layer``: one production GDN layer under a speculating cache runs verify
   blocks of width ``k + 1``; output, both slots and every restore point
   (the stock replay closure versus the fused snapshot closure, replayed at
   every ``m``) must match, then ``trim`` lands on the same state. Then
   interleaved timing.
2. ``rounds``: the full model runs greedy verify rounds with engine-rule
   acceptance and ``trim_prompt_cache`` rollback. Drafts come from a
   width-``k+1`` oracle on the stock cache itself (a full trim rewinds it),
   scheduled to cover every acceptance count. Logits, every GDN slot, every
   restore point and every attention/QSA state array must match.
3. ``e2e``: ``self_mtp_generate_step`` produces the same request in both
   modes; identical token streams, exact fused-call receipts, interleaved
   decode timing.

Host receipts (memory, swap, tenants) are taken before and after every phase
and the run fails closed on any violation or unreadable scan.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

PLAN = {
    "scope": "Qwen4 B=1 speculative verify (width k+1) with resident real weights",
    "correctness": (
        "layer: exact output, slots and restore points through trims; rounds: "
        "exact logits, GDN slots, restore points and attention state through "
        "engine-rule rollbacks; e2e: identical token streams, exact receipts"
    ),
    "timing": "interleaved stock/fused observations without model reload",
    "not_covered": ["prefill", "batch", "mask", "single-token decode"],
}

LOCK_PATH = Path("/tmp/mlx-lm-qwen4-fused-gdn-verify.lock")
LARGE_PROCESS_RSS_KIB = 8 * 1024 * 1024
REQUIRED_PHASES = ("layer", "rounds", "e2e")
_MODEL_MODULE_PATTERN = re.compile(
    r"vllm_mlx|mlx_lm|mlx-lm|mlx_vlm|bench|serve\.sh|(^|/)(rapid-mlx|rmlx|vllm-mlx)(\s|$)",
    re.IGNORECASE,
)
_MODEL_EXECUTABLES = (
    "lmstudio",
    "ollama",
    "llama-server",
    "rapid-mlx",
    "rmlx",
    "vllm-mlx",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--execute-metal", action="store_true")
    parser.add_argument("--draft-k", type=int, default=2)
    parser.add_argument("--layer-blocks", type=int, default=32)
    parser.add_argument("--layer-timing-blocks", type=int, default=64)
    parser.add_argument("--layer-repeats", type=int, default=8)
    parser.add_argument("--verify-rounds", type=int, default=24)
    parser.add_argument(
        "--prompt", default="Write the integers 1 through 200, one per line."
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--e2e-repeats", type=int, default=4)
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument(
        "--prepared-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="layer phase stamps lengths=[k+1] on every verify slab and rolls "
        "back with trim_ragged, the way the ragged self-MTP engine does",
    )
    parser.add_argument("--skip-rounds", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument(
        "--ple-nvme",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use <model>/ple_rows.bin as the PLE NVMe sidecar when present",
    )
    parser.add_argument("--min-free-percent", type=int, default=20)
    parser.add_argument("--max-swap-growth-mib", type=float, default=2048.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def expected_verify_calls(mode: str, fused_layers: int, rounds: int):
    if mode == "stock":
        return 0, 0
    if mode != "fused":
        raise ValueError(f"unknown mode {mode!r}")
    return fused_layers * rounds, 0


def greedy_accept_count(verify_argmax, drafts) -> int:
    accepted = 0
    for predicted, draft in zip(verify_argmax, drafts):
        if int(predicted) != int(draft):
            break
        accepted += 1
    return accepted


def scripted_drafts(next_tokens, k, round_index, vocab):
    """Keep the first ``k - r % (k+1)`` true drafts and corrupt the next one."""
    correct = k - (round_index % (k + 1))
    drafts = [int(t) for t in next_tokens[:k]]
    if correct < k:
        drafts[correct] = (drafts[correct] + 1) % vocab
    return drafts, correct


def parse_free_percent(text: str):
    match = re.search(r"free percentage:\s*(\d+)%", text)
    return float(match.group(1)) if match else None


def parse_swap_used_mib(text: str):
    match = re.search(r"used = ([0-9.]+)M", text)
    return float(match.group(1)) if match else None


def is_model_process(ps_line: str, own_pid: int) -> bool:
    parts = ps_line.split(maxsplit=2)
    if len(parts) < 3 or not parts[0].isdigit() or int(parts[0]) == own_pid:
        return False
    argv = parts[2].split()
    executable = os.path.basename(argv[0]).lower()
    if executable.startswith("python"):
        return _MODEL_MODULE_PATTERN.search(" ".join(argv[1:4])) is not None
    return any(name in executable for name in _MODEL_EXECUTABLES)


def completion_status(phases: dict, executed_passed: bool) -> dict:
    executed = [name for name in REQUIRED_PHASES if name in phases]
    complete = len(executed) == len(REQUIRED_PHASES)
    return {
        "phases_executed": executed,
        "complete": complete,
        "partial_passed": bool(executed_passed),
        "passed": bool(executed_passed and complete),
    }


def checkpoint_fingerprint(model_dir: Path) -> dict:
    fingerprint = {"path": str(model_dir)}
    for name in ("config.json", "model.safetensors.index.json"):
        target = model_dir / name
        if target.is_file():
            fingerprint[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return fingerprint


# ---------------------------------------------------------------------------
# Host receipts
# ---------------------------------------------------------------------------


def _run(command, *, ok_codes=(0,)):
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode in ok_codes else None


def host_receipt(mx=None):
    memory = _run(["memory_pressure"])
    swap = _run(["sysctl", "vm.swapusage"])
    listeners = _run(
        ["/usr/sbin/lsof", "-nP", "-iTCP:8282", "-sTCP:LISTEN"], ok_codes=(0, 1)
    )
    processes = _run(["ps", "-axo", "pid,rss,command"])
    lines = [] if processes is None else processes.splitlines()[1:]
    receipt = {
        "scan_ok": None not in (memory, swap, listeners, processes) and bool(lines),
        "free_percent": parse_free_percent(memory or ""),
        "swap_used_mib": parse_swap_used_mib(swap or ""),
        "port_8282_listeners": len(
            [line for line in (listeners or "").splitlines()[1:] if line.strip()]
        ),
        "model_processes": [
            line.strip()[:160] for line in lines if is_model_process(line, os.getpid())
        ],
        "large_processes": [
            line.strip()[:160]
            for line in lines
            if line.strip()
            and int(line.split()[0]) != os.getpid()
            and int(line.split()[1]) >= LARGE_PROCESS_RSS_KIB
        ],
        "timestamp": time.time(),
    }
    if mx is not None:
        receipt["mlx_active_mib"] = mx.get_active_memory() / 2**20
        receipt["mlx_peak_mib"] = mx.get_peak_memory() / 2**20
    return receipt


class HostAbortError(RuntimeError):
    pass


def host_violation(receipt, baseline, args, phase):
    if not receipt.get("scan_ok"):
        return f"{phase}: host scans unreadable"
    tenants = {
        key: receipt.get(key)
        for key in ("port_8282_listeners", "model_processes", "large_processes")
        if receipt.get(key)
    }
    if tenants:
        return f"{phase}: GPU tenant present {json.dumps(tenants)[:400]}"
    free, before, now = (
        receipt.get("free_percent"),
        baseline.get("swap_used_mib"),
        receipt.get("swap_used_mib"),
    )
    if free is None or now is None or before is None:
        return f"{phase}: host memory receipt unreadable"
    if free < args.min_free_percent:
        return f"{phase}: free memory {free}% below {args.min_free_percent}%"
    if now - before > args.max_swap_growth_mib:
        return f"{phase}: swap grew {now - before:.0f} MiB"
    return None


def acquire_gate_lock():
    handle = open(LOCK_PATH, "w")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise HostAbortError(f"another gate holds {LOCK_PATH}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


# ---------------------------------------------------------------------------
# MLX helpers
# ---------------------------------------------------------------------------


def arrays_equal(a, b, mx) -> bool:
    return bool(mx.array_equal(a, b).item())


def max_abs(a, b, mx) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def restore_points_equal(stock_cache, fused_cache, steps, mx):
    """Replay both arms' newest record at every m and compare bitwise."""
    stock_records = getattr(stock_cache, "_rollbacks", None)
    fused_records = getattr(fused_cache, "_rollbacks", None)
    if not stock_records or not fused_records:
        return False, "missing rollback record"
    stock_record, fused_record = stock_records[-1], fused_records[-1]
    if stock_record.num_tokens != fused_record.num_tokens:
        return False, "record span mismatch"
    for m in range(1, steps):
        left, right = list(stock_record.fn(m)), list(fused_record.fn(m))
        if len(left) != len(right):
            return False, f"entry count at m={m}"
        mx.eval(*left, *right)
        for slot, (x, y) in enumerate(zip(left, right)):
            if x.shape != y.shape or not arrays_equal(x, y, mx):
                return (
                    False,
                    f"restore point m={m} slot {slot} max_abs {max_abs(x, y, mx)}",
                )
    return True, "equal"


def gdn_caches(cache_list, cache_types):
    return [c for c in cache_list if isinstance(c, cache_types)]


def leaf_state_arrays(cache):
    children = getattr(cache, "caches", None)
    if children is not None:
        arrays = []
        for child in children:
            arrays.extend(leaf_state_arrays(child))
        return arrays
    state = getattr(cache, "state", None)
    items = state if isinstance(state, (list, tuple)) else [state]
    return [item for item in items if item is not None and hasattr(item, "shape")]


def attention_states_equal(stock_list, fused_list, cache_types, mx):
    for index, (a, b) in enumerate(zip(stock_list, fused_list)):
        if isinstance(a, cache_types):
            continue
        left, right = leaf_state_arrays(a), leaf_state_arrays(b)
        if len(left) != len(right):
            return False, f"attention cache {index}: state arity"
        for position, (x, y) in enumerate(zip(left, right)):
            if x.shape != y.shape or x.dtype != y.dtype or not arrays_equal(x, y, mx):
                return False, f"attention cache {index}: state array {position}"
    return True, "equal"


def cache_arrays(cache_list):
    arrays = []
    for cache in cache_list:
        values = getattr(cache, "cache", None)
        if values is None:
            values = leaf_state_arrays(cache)
        arrays.extend(v for v in values if v is not None and hasattr(v, "dtype"))
    return arrays


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_layer(args, layer, mx, cache_type, stats_fn, set_verify_mode):
    steps = args.draft_k + 1
    hidden = mx.random.normal(
        (max(args.layer_blocks, args.layer_timing_blocks), 1, steps, layer.hidden_size),
        key=mx.random.key(2105),
    ).astype(layer.dt_bias.dtype)

    def fresh_pair():
        stock, fused = cache_type(2), cache_type(2)
        set_verify_mode(layer, "stock")
        warm = hidden[0][:, :1]
        mx.eval(layer(warm, cache=stock), *stock.cache)
        fused.cache = [mx.array(v) for v in stock.cache]
        for cache in (stock, fused):
            cache.start_speculation()
        return stock, fused

    stock_cache, fused_cache = fresh_pair()
    mismatch = None
    before = stats_fn(layer)
    restores = []

    def prepare(cache):
        if args.prepared_geometry:
            cache.prepare(lengths=[steps])

    def finalize(cache):
        if args.prepared_geometry:
            cache.finalize()

    def rewind(cache, n):
        if args.prepared_geometry:
            return cache.trim_ragged([n])
        return cache.trim(n)

    for block in range(args.layer_blocks):
        set_verify_mode(layer, "stock")
        prepare(stock_cache)
        stock = layer(hidden[block], cache=stock_cache)
        finalize(stock_cache)
        set_verify_mode(layer, "fused")
        prepare(fused_cache)
        fused = layer(hidden[block], cache=fused_cache)
        finalize(fused_cache)
        mx.eval(stock, fused, *stock_cache.cache, *fused_cache.cache)
        output_equal = arrays_equal(stock, fused, mx)
        slots_equal = all(
            arrays_equal(a, b, mx) for a, b in zip(stock_cache.cache, fused_cache.cache)
        )
        points_equal, detail = restore_points_equal(stock_cache, fused_cache, steps, mx)
        if not (output_equal and slots_equal and points_equal):
            mismatch = {
                "block": block,
                "output_equal": output_equal,
                "slots_equal": slots_equal,
                "restore_points_equal": points_equal,
                "restore_detail": detail,
                "max_output_abs": max_abs(stock, fused, mx),
            }
            break
        n_to_drop = block % steps
        restores.append(n_to_drop)
        if n_to_drop:
            rewind(stock_cache, n_to_drop)
            rewind(fused_cache, n_to_drop)
            mx.eval(*stock_cache.cache, *fused_cache.cache)
            if not all(
                arrays_equal(a, b, mx)
                for a, b in zip(stock_cache.cache, fused_cache.cache)
            ):
                mismatch = {"block": block, "after_trim": n_to_drop}
                break
    after = stats_fn(layer)
    verify_calls = after["verify_calls"] - before["verify_calls"]
    fallbacks = after["verify_fallbacks"] - before["verify_fallbacks"]
    correctness = {
        "passed": mismatch is None
        and args.layer_blocks > 0
        and verify_calls == args.layer_blocks
        and fallbacks == 0
        and set(restores) == set(range(steps)),
        "blocks": args.layer_blocks,
        "verify_width": steps,
        "prepared_geometry": bool(args.prepared_geometry),
        "mismatch": mismatch,
        "verify_calls": verify_calls,
        "verify_fallbacks": fallbacks,
        "restores": restores,
        "last_fallbacks": after["verify_last_fallbacks"],
    }
    if not correctness["passed"]:
        return {"correctness": correctness, "timing": None}

    timings = {"stock": [], "fused": []}
    for repeat in range(args.layer_repeats):
        order = ("stock", "fused") if repeat % 2 == 0 else ("fused", "stock")
        for mode in order:
            cache, _ = fresh_pair()
            set_verify_mode(layer, mode)
            started = time.perf_counter()
            output = None
            for block in range(args.layer_timing_blocks):
                cache._rollbacks.clear()
                prepare(cache)
                output = layer(hidden[block % len(hidden)], cache=cache)
                finalize(cache)
            mx.eval(output, *cache.cache)
            timings[mode].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "correctness": correctness,
        "timing": {
            "raw_seconds": timings,
            "median_seconds": medians,
            "median_speedup_percent": 100.0
            * (medians["stock"] / medians["fused"] - 1.0),
            "blocks_per_observation": args.layer_timing_blocks,
        },
    }


def phase_rounds(args, model, prompt, mx, cache_types, stats_fn, set_verify_mode, api):
    start_spec, trim_prompt_cache = api
    steps = args.draft_k + 1
    fused_layers = set_verify_mode(model, "stock")
    caches = {"stock": model.make_cache(), "fused": model.make_cache()}
    logits = {}
    for mode, cache in caches.items():
        logits[mode] = model(prompt[None], cache=cache)
        mx.eval(logits[mode], *cache_arrays(cache))
    if not arrays_equal(logits["stock"], logits["fused"], mx):
        raise RuntimeError("stock prefill is not reproducible across cache copies")
    for cache in caches.values():
        start_spec(cache)
    vocab = int(logits["stock"].shape[-1])
    y = int(mx.argmax(logits["stock"][0, -1]).item())

    def oracle(cache):
        """True next k tokens after y under the verify-width stock forward.

        Probe j feeds ``[y, t1, .., t_j, pad..]`` at the verify width and reads
        position j; the causal prefix matches what the real round sees, so
        the argmax is the one the engine will accept. A full trim rewinds
        every cache to the pre-block state.
        """
        set_verify_mode(model, "stock")
        tokens = [y]
        for j in range(args.draft_k):
            block = tokens + [tokens[-1]] * (steps - len(tokens))
            out = model(mx.array([block], mx.uint32), cache=cache)
            tokens.append(int(mx.argmax(out[0, j]).item()))
            trimmed = trim_prompt_cache(cache, steps)
            if trimmed != steps:
                raise RuntimeError(f"oracle trim returned {trimmed}, expected {steps}")
        return tokens[1:]

    accepted_hist = {str(n): 0 for n in range(args.draft_k + 1)}
    schedule_misses = 0
    mismatch = None
    rounds_done = 0
    before = stats_fn(model)
    for round_index in range(args.verify_rounds):
        next_tokens = oracle(caches["stock"])
        drafts, expected_accept = scripted_drafts(
            next_tokens, args.draft_k, round_index, vocab
        )
        tokens = mx.array([[y, *drafts]], mx.uint32)
        outputs = {}
        for mode, cache in caches.items():
            set_verify_mode(model, mode)
            outputs[mode] = model(tokens, cache=cache)
        mx.eval(*outputs.values())
        mx.eval(*cache_arrays(caches["stock"]), *cache_arrays(caches["fused"]))
        logits_equal = arrays_equal(outputs["stock"], outputs["fused"], mx)
        detail = None
        for index, (a, b) in enumerate(
            zip(
                gdn_caches(caches["stock"], cache_types),
                gdn_caches(caches["fused"], cache_types),
            )
        ):
            for slot, (x, w) in enumerate(zip(a.cache, b.cache)):
                if (x is None) != (w is None) or (
                    x is not None and not arrays_equal(x, w, mx)
                ):
                    detail = f"gdn cache {index} slot {slot}"
                    break
            if detail is None:
                ok, why = restore_points_equal(a, b, steps, mx)
                if not ok:
                    detail = f"gdn cache {index}: {why}"
            if detail is not None:
                break
        states_equal, state_detail = attention_states_equal(
            caches["stock"], caches["fused"], cache_types, mx
        )
        if not logits_equal or detail is not None or not states_equal:
            mismatch = {
                "round": round_index,
                "logits_equal": logits_equal,
                "max_logit_abs": max_abs(outputs["stock"], outputs["fused"], mx),
                "gdn_detail": detail,
                "attention_states_equal": states_equal,
                "attention_detail": state_detail,
            }
            break
        verify_argmax = mx.argmax(outputs["stock"][0], axis=-1).tolist()
        accepted = greedy_accept_count(verify_argmax, drafts)
        accepted_hist[str(accepted)] += 1
        schedule_misses += int(accepted != expected_accept)
        n_to_drop = args.draft_k - accepted
        for cache in caches.values():
            if n_to_drop and trim_prompt_cache(cache, n_to_drop) != n_to_drop:
                raise RuntimeError("rollback trim refused")
        y = int(verify_argmax[accepted])
        rounds_done += 1
    after = stats_fn(model)
    verify_calls = after["verify_calls"] - before["verify_calls"]
    fallbacks = after["verify_fallbacks"] - before["verify_fallbacks"]
    # Every oracle probe is a stock-arm forward; only the real rounds run in
    # fused mode, so the fused-call receipt is fused_layers per real round.
    expected_calls, expected_fallbacks = expected_verify_calls(
        "fused", fused_layers, rounds_done + (1 if mismatch is not None else 0)
    )
    coverage = all(accepted_hist[str(n)] > 0 for n in range(args.draft_k + 1))
    return {
        "correctness": {
            "passed": mismatch is None
            and rounds_done == args.verify_rounds
            and verify_calls == expected_calls
            and fallbacks == expected_fallbacks
            and coverage,
            "rounds": rounds_done,
            "verify_width": steps,
            "prompt_tokens": int(prompt.size),
            "mismatch": mismatch,
            "accepted_histogram": accepted_hist,
            "acceptance_coverage": coverage,
            "schedule_misses": schedule_misses,
            "verify_calls": verify_calls,
            "expected_verify_calls": expected_calls,
            "verify_fallbacks": fallbacks,
            "fused_layers": fused_layers,
            "last_fallbacks": after["verify_last_fallbacks"],
        }
    }


def phase_e2e(args, model, tokenizer, prompt, mx, stats_fn, set_verify_mode, api):
    generate, stats_type = api
    if args.e2e_repeats % 2:
        raise ValueError("--e2e-repeats must be even so both arm orders balance")
    eos_ids = set()
    eos = getattr(tokenizer, "eos_token_ids", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    if isinstance(eos, int):
        eos_ids.add(eos)
    elif eos:
        eos_ids.update(int(e) for e in eos)
    observations = []
    orders = (("stock", "fused"), ("fused", "stock"))
    for repeat in range(args.e2e_repeats):
        for mode in orders[repeat % 2]:
            fused_layers = set_verify_mode(model, mode)
            before = stats_fn(model)
            stats = stats_type()
            tokens, drafted = [], 0
            started = time.perf_counter()
            first_token_at = None
            for token, _, from_draft in generate(
                prompt,
                model,
                num_draft=args.draft_k,
                max_tokens=args.max_tokens,
                sampling_temp=0.0,
                stats=stats,
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                tokens.append(int(token))
                drafted += int(bool(from_draft))
                if int(token) in eos_ids:
                    break
            ended = time.perf_counter()
            after = stats_fn(model)
            if len(tokens) < 2 or first_token_at is None:
                raise RuntimeError(f"generation produced {len(tokens)} tokens")
            rounds = int(getattr(stats, "draft_cycles", 0))
            verify_calls = after["verify_calls"] - before["verify_calls"]
            fallbacks = after["verify_fallbacks"] - before["verify_fallbacks"]
            expected_calls, expected_fallbacks = expected_verify_calls(
                mode, fused_layers, rounds
            )
            decode_seconds = ended - first_token_at
            observations.append(
                {
                    "mode": mode,
                    "repeat": repeat + 1,
                    "tokens": len(tokens),
                    "draft_accepted_tokens": drafted,
                    "verify_rounds": rounds,
                    "draft_proposed": int(getattr(stats, "draft_proposed", 0)),
                    "draft_accepted": int(getattr(stats, "draft_accepted", 0)),
                    "token_sha256": hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "ttft_seconds": first_token_at - started,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds,
                    "verify_calls": verify_calls,
                    "expected_verify_calls": expected_calls,
                    "verify_fallbacks": fallbacks,
                    "path_counts_exact": verify_calls == expected_calls
                    and fallbacks == expected_fallbacks
                    and rounds > 0,
                    "last_fallbacks": after["verify_last_fallbacks"],
                }
            )
            mx.clear_cache()
    hashes = {item["token_sha256"] for item in observations}
    path_counts_exact = all(item["path_counts_exact"] for item in observations)
    medians = {
        mode: statistics.median(
            item["decode_tokens_per_second"]
            for item in observations
            if item["mode"] == mode
        )
        for mode in ("stock", "fused")
    }
    return {
        "correctness": {
            "passed": len(hashes) == 1 and path_counts_exact,
            "token_exact": len(hashes) == 1,
            "path_counts_exact": path_counts_exact,
            "hashes": sorted(hashes),
        },
        "draft_k": args.draft_k,
        "verify_width": args.draft_k + 1,
        "observations": observations,
        "median_decode_tokens_per_second": medians,
        "median_speedup_percent": 100.0 * (medians["fused"] / medians["stock"] - 1.0),
    }


def run(args, partial: dict | None = None):
    if partial is None:
        partial = {}
    if args.model is None or not args.model.is_dir():
        raise SystemExit("--model must name an existing local checkpoint directory")
    for name in (
        "layer_blocks",
        "layer_timing_blocks",
        "layer_repeats",
        "verify_rounds",
        "e2e_repeats",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    if args.layer_blocks < args.draft_k + 1:
        raise SystemExit("--layer-blocks must cover every restore boundary")
    model_dir = args.model.resolve()
    sidecar = model_dir / "ple_rows.bin"
    if args.ple_nvme and sidecar.is_file():
        os.environ.setdefault("MLX_QWEN4_PLE_NVME", str(sidecar))
        os.environ.setdefault("MLX_QWEN4_PLE_NVME_LRU_MB", "256")

    lock = acquire_gate_lock()
    receipts = {"start": host_receipt()}
    violation = host_violation(receipts["start"], receipts["start"], args, "start")
    if violation is not None:
        raise HostAbortError(violation)
    violations = []

    def take_receipt(name, mx_module=None):
        receipts[name] = host_receipt(mx_module)
        found = host_violation(receipts[name], receipts["start"], args, name)
        if found is not None:
            violations.append(found)
        return found

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    from mlx_lm.hybrid_speculative import (
        HybridStats,
        _start_speculation_or_cleanup,
        self_mtp_generate_step,
    )
    from mlx_lm.models.cache import ArraysCache, trim_prompt_cache
    from mlx_lm.models.qwen4_exp import (
        GatedDeltaNet,
        Qwen4ArraysCache,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
        set_qwen4_fused_gdn_verify_mode,
    )
    from mlx_lm.utils import load

    model, tokenizer = load(str(model_dir), lazy=args.layer_only)
    model.eval()
    set_qwen4_fused_gdn_mode(model, "stock")
    fused_layers = set_qwen4_fused_gdn_verify_mode(model, "stock")
    if not fused_layers:
        raise SystemExit("checkpoint did not instantiate Qwen4 GatedDeltaNet layers")
    layer = next(m for _, m in model.named_modules() if isinstance(m, GatedDeltaNet))
    if args.layer_only:
        mx.eval(layer.parameters())
    found = take_receipt("after_load", mx)
    if found is not None:
        raise HostAbortError(found)

    import mlx_lm

    result = {
        "plan": PLAN,
        "fused_layers": fused_layers,
        "draft_k": args.draft_k,
        "layer_only": bool(args.layer_only),
        "mlx_version": mx.__version__,
        "mlx_lm_path": str(Path(mlx_lm.__file__).parent),
        "ple_nvme": os.environ.get("MLX_QWEN4_PLE_NVME"),
        "checkpoint": checkpoint_fingerprint(model_dir),
        "arguments": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "phases": {},
        "host": receipts,
    }
    partial.update(result)
    cache_types = (ArraysCache, Qwen4ArraysCache)

    result["phases"]["layer"] = phase_layer(
        args,
        layer,
        mx,
        ArraysCache,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_verify_mode,
    )
    take_receipt("after_layer", mx)
    passed = result["phases"]["layer"]["correctness"]["passed"]

    if args.layer_only:
        args.skip_rounds = True
        args.skip_e2e = True
    prompt = mx.array(
        tokenizer.encode(args.prompt, add_special_tokens=False), mx.uint32
    )
    result["prompt_tokens"] = int(prompt.size)

    def start_spec(cache):
        _start_speculation_or_cleanup(
            cache, cache, "the verify gate requires a trimmable prompt cache"
        )

    if passed and not args.skip_rounds:
        found = host_violation(
            receipts["after_layer"], receipts["start"], args, "rounds"
        )
        if found is not None:
            raise HostAbortError(found)
        result["phases"]["rounds"] = phase_rounds(
            args,
            model,
            prompt,
            mx,
            cache_types,
            qwen4_fused_gdn_stats,
            set_qwen4_fused_gdn_verify_mode,
            (start_spec, trim_prompt_cache),
        )
        take_receipt("after_rounds", mx)
        passed = result["phases"]["rounds"]["correctness"]["passed"]
        mx.clear_cache()

    if passed and not args.skip_e2e:
        latest = receipts[max(receipts, key=lambda name: receipts[name]["timestamp"])]
        found = host_violation(latest, receipts["start"], args, "e2e")
        if found is not None:
            raise HostAbortError(found)
        result["phases"]["e2e"] = phase_e2e(
            args,
            model,
            tokenizer,
            prompt,
            mx,
            qwen4_fused_gdn_stats,
            set_qwen4_fused_gdn_verify_mode,
            (self_mtp_generate_step, HybridStats),
        )
        take_receipt("after_e2e", mx)
        passed = result["phases"]["e2e"]["correctness"]["passed"]

    take_receipt("end", mx)
    result["host_violations"] = violations
    result.update(completion_status(result["phases"], passed and not violations))
    lock.close()
    return result


def main():
    args = parse_args()
    if not args.execute_metal:
        print(json.dumps({"plan_only": True, "plan": PLAN}, indent=2))
        return 0
    partial: dict = {}
    try:
        result = run(args, partial)
    except HostAbortError as exc:
        result = {**partial, "plan": PLAN, "passed": False, "aborted": str(exc)}
    except Exception as exc:  # noqa: BLE001 - keep the receipts of a crashed run
        import traceback

        result = {
            **partial,
            "plan": PLAN,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if result.get("complete"):
        return 0 if result.get("passed") else 1
    return 0 if result.get("partial_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
