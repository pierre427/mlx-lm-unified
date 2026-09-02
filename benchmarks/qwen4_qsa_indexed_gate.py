#!/usr/bin/env python3
"""Five-phase Metal gate for fixed-chunk indexed split-K QSA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
DEFAULT_OUTPUT = Path(
    "/Users/pierrelamy/Desktop/mlx-uag/results/"
    "qwen4-qsa-indexed-gate-v2-20260901.json"
)
CONTEXTS = (16_384, 32_768, 65_536, 131_072)
MODEL_CONTEXTS = CONTEXTS[:3]
SPLITS = (1, 2, 4, 8)
DEFAULT_SWAP_LIMIT_MIB = 512.0
DEFAULT_MODEL_LOAD_FREE_FLOOR = 45
DEFAULT_RUN_FREE_FLOOR = 25


class GateFailure(RuntimeError):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = int(phase)


class GateStop(RuntimeError):
    pass


@contextmanager
def gpu_lock():
    """Take the lab-wide GPU lock and remove only this owner's file."""

    inherited_owner = os.environ.get("MLXUAG_GPU_LOCK_ALREADY_HELD")
    if inherited_owner:
        owner_path = GPU_LOCK / "owner.json"
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit("inherited GPU lock has no valid owner.json") from error
        if owner.get("owner") != inherited_owner:
            raise SystemExit(
                f"inherited GPU lock belongs to {owner.get('owner')!r}, "
                f"not {inherited_owner!r}"
            )
        yield owner
        return

    try:
        os.mkdir(GPU_LOCK)
    except FileExistsError as error:
        owner_path = GPU_LOCK / "owner.json"
        try:
            owner = owner_path.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "owner unavailable"
        raise SystemExit(f"GPU busy: {GPU_LOCK} exists ({owner})") from error
    owner_path = GPU_LOCK / "owner.json"
    owner = {
        "agent": "codex-l-phase45",
        "label": "qwen4-qsa-indexed-gate-v2-phase45",
        "pid": os.getpid(),
        "purpose": "fixed-chunk indexed split-K QSA Metal gate",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        owner_path.write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        yield owner
    finally:
        owner_path.unlink(missing_ok=True)
        try:
            GPU_LOCK.rmdir()
        except OSError as error:
            raise RuntimeError(f"could not release {GPU_LOCK}: {error}") from error


def _run_text(command):
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def safety_snapshot(label=None):
    memory = _run_text(["/usr/bin/memory_pressure"])
    match = re.search(r"free percentage:\s*(\d+)%", memory)
    if match is None:
        raise RuntimeError(f"could not parse memory pressure: {memory!r}")
    swap = _run_text(["/usr/sbin/sysctl", "vm.swapusage"])
    match_swap = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap)
    if match_swap is None:
        raise RuntimeError(f"could not parse swap usage: {swap!r}")
    used = float(match_swap.group(1))
    if match_swap.group(2) == "G":
        used *= 1024.0
    thermal = _run_text(["/usr/bin/pmset", "-g", "therm"])
    return {
        "label": label,
        "at": datetime.now(timezone.utc).isoformat(),
        "free_percent": int(match.group(1)),
        "swap_used_mib": used,
        "memory_pressure_tail": memory.splitlines()[-1],
        "swapusage": swap,
        "thermal": thermal.splitlines(),
    }


def check_safety(
    snapshot,
    *,
    swap_baseline=None,
    before_load=False,
    phase=5,
    model_load_free_floor=DEFAULT_MODEL_LOAD_FREE_FLOOR,
    run_free_floor=DEFAULT_RUN_FREE_FLOOR,
    swap_growth_abort_mib=DEFAULT_SWAP_LIMIT_MIB,
):
    floor = model_load_free_floor if before_load else run_free_floor
    if snapshot["free_percent"] < floor:
        raise GateFailure(
            int(phase),
            f"free memory {snapshot['free_percent']}% is below {floor}%",
        )
    if (
        swap_baseline is not None
        and swap_growth_abort_mib > 0
        and snapshot["swap_used_mib"] - swap_baseline > swap_growth_abort_mib
    ):
        growth = snapshot["swap_used_mib"] - swap_baseline
        raise GateFailure(int(phase), f"swap grew {growth:.2f} MiB")


def thermal_clean(lines):
    text = "\n".join(lines).lower()
    return (
        "no thermal warning" in text
        and "no performance warning" in text
        and "no cpu power status" in text
    )


def settle_thermal(timeout=60.0):
    """Require two clean thermal samples before the next timing arm."""

    started = time.monotonic()
    clean = 0
    samples = []
    while time.monotonic() - started < timeout:
        snapshot = safety_snapshot()
        samples.append(snapshot)
        clean = clean + 1 if thermal_clean(snapshot["thermal"]) else 0
        if clean == 2:
            return {
                "settled": True,
                "seconds": time.monotonic() - started,
                "samples": samples[-2:],
            }
        time.sleep(2.0)
    raise GateFailure(5, "thermal state did not settle")


def clone_containers(value):
    if isinstance(value, list):
        return [clone_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_containers(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_containers(item) for key, item in value.items()}
    return value


def clone_cache(cache):
    return [
        type(layer).from_state(
            clone_containers(layer.state), clone_containers(layer.meta_state)
        )
        for layer in cache
    ]


def compact_fixture(mx, compact_type, context, length=3):
    topk = 512
    q_pos = mx.arange(context - length, context, dtype=mx.int32)[None]
    ids = mx.broadcast_to(
        mx.arange(topk, dtype=mx.uint32)[None, None], (1, length, topk)
    )
    counts = mx.full((1, length), topk, dtype=mx.int32)
    tail_stop = q_pos + 1
    tail_start = tail_stop // 4 * 4
    return compact_type(
        block_ids=ids,
        block_counts=counts,
        tail_start=tail_start,
        tail_stop=tail_stop,
        left_padding=None,
        block_size=4,
        physical_width=context,
        causal_mask=None,
    )


def adversarial_fixture(mx, compact_type, *, batch, length, context=4096):
    width = 512
    ids = np.zeros((batch, length, width), dtype=np.uint32)
    counts = np.zeros((batch, length), dtype=np.int32)
    tail_stop = np.zeros((batch, length), dtype=np.int32)
    left = np.arange(batch, dtype=np.int32) % 3
    causal = np.ones((batch, 1, length, context), dtype=bool)
    for b in range(batch):
        for row in range(length):
            count = (512, 511, 257, 65, 1, 0)[(b + row) % 6]
            counts[b, row] = count
            if count:
                ids[b, row, :count] = np.arange(count, dtype=np.uint32)
            if row == 0:
                tail_stop[b, row] = 2048
            else:
                tail_stop[b, row] = context - left[b] - length + row + 1
            if count == 0:
                causal[b, 0, row] = False
    return compact_type(
        block_ids=mx.array(ids),
        block_counts=mx.array(counts),
        tail_start=mx.array(tail_stop // 4 * 4),
        tail_stop=mx.array(tail_stop),
        left_padding=mx.array(left),
        block_size=4,
        physical_width=context,
        causal_mask=mx.array(causal),
    )


def dense_fixture_mask(mx, context, length=3):
    q_pos = mx.arange(context - length, context, dtype=mx.int32)[None]
    token = mx.arange(context, dtype=mx.int32)[None, None]
    chosen = token // 4 < 512
    complete = ((q_pos + 1) // 4) * 4
    tail = (token >= complete[..., None]) & (token <= q_pos[..., None])
    return (chosen | tail)[:, None]


def max_scaled_error(mx, actual, expected):
    mx.eval(actual, expected)
    delta = float(mx.max(mx.abs(actual - expected)).item())
    scale = max(float(mx.max(mx.abs(expected)).item()), 1.0)
    return delta, delta / scale


def timed(mx, function):
    started = time.perf_counter()
    output = function()
    mx.eval(output)
    return time.perf_counter() - started


def phase1_candidate(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks
    from mlx_lm.models.qwen4_qsa_indexed import (
        qsa_indexed_status,
        qwen4_qsa_indexed_attention,
    )

    mx.random.seed(20260901)
    compact = compact_fixture(mx, QSACompactBlocks, CONTEXTS[0])
    q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, CONTEXTS[0], 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, CONTEXTS[0], 256)).astype(mx.bfloat16)
    qsa_indexed_status(reset=True)
    output = qwen4_qsa_indexed_attention(
        q, k, v, compact, scale=256**-0.5, splits=8
    )
    mx.eval(output)
    status = qsa_indexed_status()
    if status["candidate"] is None or status["fallbacks"]:
        raise GateFailure(1, f"candidate probe did not engage cleanly: {status}")
    return {"phase": 1, "status": "PASS", "receipt": status}


def cast_tie_counts(kernel_np, mirror_np, kernel_cast, mirror_cast, limit):
    mismatch = kernel_cast != mirror_cast
    delta = np.abs(kernel_np - mirror_np)
    ties = mismatch & (delta <= limit)
    return {
        "cast_mismatch_count": int(np.count_nonzero(mismatch)),
        "documented_tie_count": int(np.count_nonzero(ties)),
        "non_tie_count": int(np.count_nonzero(mismatch & ~ties)),
    }


def phase2_exactness(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import (
        qwen4_qsa_indexed_attention,
        qwen4_qsa_indexed_reference,
    )

    rows = []
    worst_relative = 0.0
    all_s_equal = True
    non_ties = 0
    for length in range(1, 9):
        batch = 1 + length % 2
        mx.random.seed(20260901 + length)
        compact = adversarial_fixture(
            mx, QSACompactBlocks, batch=batch, length=length
        )
        q = mx.random.normal((batch, 24, length, 256)).astype(mx.float32)
        k = mx.random.normal((batch, 2, 4096, 256)).astype(mx.float32)
        v = mx.random.normal((batch, 2, 4096, 256)).astype(mx.float32)
        outputs = {
            splits: qwen4_qsa_indexed_attention(
                q, k, v, compact, scale=256**-0.5, splits=splits
            )
            for splits in SPLITS
        }
        mirror = qwen4_qsa_indexed_reference(
            q, k, v, compact, scale=256**-0.5, splits=8
        )
        mx.eval(*outputs.values(), mirror)
        first = np.asarray(outputs[1])
        split_checks = {}
        for splits in SPLITS:
            current = np.asarray(outputs[splits])
            equal = np.array_equal(first, current)
            all_s_equal = all_s_equal and equal
            split_checks[str(splits)] = {
                "bit_equal_to_s1": equal,
                "max_abs": float(np.max(np.abs(first - current))),
            }
        kernel = np.asarray(outputs[8])
        mirror_np = np.asarray(mirror)
        delta = float(np.max(np.abs(kernel - mirror_np)))
        scale = max(float(np.max(np.abs(mirror_np))), 1.0)
        relative = delta / scale
        worst_relative = max(worst_relative, relative)
        kernel_cast = np.asarray(outputs[8].astype(mx.bfloat16).astype(mx.float32))
        mirror_cast = np.asarray(mirror.astype(mx.bfloat16).astype(mx.float32))
        tie_counts = cast_tie_counts(
            kernel, mirror_np, kernel_cast, mirror_cast, 1.0e-4 * scale
        )
        non_ties += tie_counts["non_tie_count"]
        rows.append(
            {
                "batch": batch,
                "length": length,
                "kernel_mirror_max_abs": delta,
                "kernel_mirror_max_relative": relative,
                "splits": split_checks,
                **tie_counts,
            }
        )
        del q, k, v, outputs, mirror
        mx.clear_cache()
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "qwen4_qsa_indexed_real_bf16_257.safetensors"
    )
    fixture = mx.load(str(fixture_path))
    fixture_width = int(fixture["k"].shape[2])
    fixture_blocks = fixture_width // 4
    fixture_compact = QSACompactBlocks(
        block_ids=mx.arange(fixture_blocks, dtype=mx.uint32)[None, None],
        block_counts=mx.array([[fixture_blocks]], dtype=mx.int32),
        tail_start=mx.array([[fixture_width]], dtype=mx.int32),
        tail_stop=mx.array([[fixture_width]], dtype=mx.int32),
        left_padding=mx.array([0], dtype=mx.int32),
        block_size=4,
        physical_width=fixture_width,
        causal_mask=None,
    )
    fixture_gather = _gather_qsa_attention(
        fixture["q"],
        fixture["k"],
        fixture["v"],
        fixture_compact,
        scale=256**-0.5,
        tile_rows=1,
    )
    fixture_outputs = {
        splits: qwen4_qsa_indexed_attention(
            fixture["q"],
            fixture["k"],
            fixture["v"],
            fixture_compact,
            scale=256**-0.5,
            splits=splits,
        )
        for splits in (1, 4, 8)
    }
    mx.eval(fixture_gather, *fixture_outputs.values())
    fixture_checks = {}
    fixture_exact = True
    for splits, output in fixture_outputs.items():
        exact = bool(mx.array_equal(output, fixture_gather).item())
        fixture_exact = fixture_exact and exact
        fixture_checks[str(splits)] = {
            "bit_equal_to_gather": exact,
            "max_abs": float(
                mx.max(
                    mx.abs(
                        output.astype(mx.float32)
                        - fixture_gather.astype(mx.float32)
                    )
                ).item()
            ),
        }
    passed = (
        all_s_equal
        and worst_relative <= 1.0e-4
        and non_ties == 0
        and fixture_exact
    )
    result = {
        "phase": 2,
        "status": "PASS" if passed else "FAIL",
        "s_invariant": all_s_equal,
        "worst_kernel_mirror_relative": worst_relative,
        "non_tie_count": non_ties,
        "real_capture_fixture": {
            "path": str(fixture_path),
            "source_geometry": list(map(int, fixture["q"].shape)),
            "physical_width": fixture_width,
            "checks": fixture_checks,
        },
        "rows": rows,
    }
    if not passed:
        raise GateFailure(2, json.dumps(result, sort_keys=True))
    return result


def phase3_gather(mx):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import qwen4_qsa_indexed_attention

    rows = []
    for length in range(1, 9):
        batch = 1 + length % 2
        mx.random.seed(20261001 + length)
        compact = adversarial_fixture(
            mx, QSACompactBlocks, batch=batch, length=length
        )
        q = mx.random.normal((batch, 24, length, 256)).astype(mx.bfloat16)
        k = mx.random.normal((batch, 2, 4096, 256)).astype(mx.bfloat16)
        v = mx.random.normal((batch, 2, 4096, 256)).astype(mx.bfloat16)
        kernel = qwen4_qsa_indexed_attention(
            q, k, v, compact, scale=256**-0.5, splits=8
        )
        gather = _gather_qsa_attention(
            q, k, v, compact, scale=256**-0.5, tile_rows=1
        )
        absolute, relative = max_scaled_error(mx, kernel, gather)
        rows.append(
            {
                "batch": batch,
                "length": length,
                "max_abs": absolute,
                "max_relative": relative,
            }
        )
        del q, k, v, kernel, gather
        mx.clear_cache()
    passed = all(row["max_abs"] == 0.0 for row in rows)
    result = {
        "phase": 3,
        "status": "PASS" if passed else "FAIL",
        "asserted": True,
        "rows": rows,
    }
    if not passed:
        raise GateFailure(3, json.dumps(result, sort_keys=True))
    return result


def corpus_tokens(tokenizer, context):
    text = """
Incident review transcript. The operator first confirms that the service is
quiescent, records memory and thermal state, and preserves the exact prompt
cache boundary. The implementation keeps selected key/value rows in place;
the comparison arm gathers the same rows before attention. Both arms start
from clones of one evaluated cache. During review, the team checks every
rollback record, generated token, chosen log probability, kernel candidate,
and fallback receipt. A digest mismatch triggers a first-divergence analysis
with top candidates, prefix numeric noise, and fresh one-row versus three-row
forward controls. Timing begins only after equivalence passes.

The following change request is realistic rather than synthetic: investigate
a slow long-context inference request, explain the evidence, propose the
smallest default-off optimization, write a targeted regression test, and
report measured latency without claiming deployment. Logs show repeated
cache hits, twelve sparse-attention layers, a width-three verification call,
and no server traffic during the offline gate. The acceptance bar is strict:
the target model remains token-authoritative and any unexplained flip closes
the experiment.

Example code under review:
```python
def choose_route(context, query_width, enabled):
    if not enabled or context < 16384:
        return "dense"
    return "indexed" if 1 <= query_width <= 8 else "gather"
```
"""
    seed = tokenizer.encode(text)
    if not seed:
        raise ValueError("tokenizer produced an empty gate prompt")
    return (seed * (context // len(seed) + 1))[:context]


def top_candidates(mx, logprobs, count=8):
    indices = mx.argpartition(logprobs, kth=-count)[-count:]
    mx.eval(indices)
    pairs = [(int(index), float(logprobs[int(index)].item())) for index in indices]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


@contextmanager
def qsa_mode(mode):
    from mlx_lm.models import qwen4_exp
    from mlx_lm.models.qwen4_qsa_indexed import (
        qsa_indexed_enabled,
        set_qwen4_qsa_indexed,
    )

    previous_indexed = qsa_indexed_enabled()
    names = (
        "_QSA_GATHER_KV",
        "_QSA_GATHER_MIN_QUERY",
        "_QSA_GATHER_MAX_QUERY",
        "_QSA_GATHER_MIN_CONTEXT",
        "_QSA_GATHER_MAX_CONTEXT",
        "_QSA_NAX_DECODE",
    )
    previous = {name: getattr(qwen4_exp, name) for name in names}
    try:
        set_qwen4_qsa_indexed(mode == "indexed")
        qwen4_exp._QSA_GATHER_KV = mode in {"indexed", "gather"}
        qwen4_exp._QSA_GATHER_MIN_QUERY = 1
        qwen4_exp._QSA_GATHER_MAX_QUERY = 8
        qwen4_exp._QSA_GATHER_MIN_CONTEXT = 0
        qwen4_exp._QSA_GATHER_MAX_CONTEXT = 0
        qwen4_exp._QSA_NAX_DECODE = False
        yield
    finally:
        set_qwen4_qsa_indexed(previous_indexed)
        for name, value in previous.items():
            setattr(qwen4_exp, name, value)


def run_model_arm(
    mx,
    model,
    token,
    cache,
    *,
    mode,
    max_tokens,
    abort_event=None,
    abort_phase=4,
):
    from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
    from mlx_lm.models.qwen4_qsa_indexed import qsa_indexed_status

    with qsa_mode(mode):
        qsa_indexed_status(reset=True)
        stats = HybridStats()
        tokens = []
        chosen_logprobs = []
        top8 = []
        started = time.perf_counter()
        for output_token, logprobs, _ in self_mtp_generate_step(
            mx.array([token], dtype=mx.uint32),
            model,
            num_draft=2,
            max_tokens=max_tokens,
            persistent_mtp=True,
            prompt_cache=cache,
            stats=stats,
        ):
            if abort_event is not None and abort_event.is_set():
                raise GateFailure(
                    abort_phase, "free memory fell below the 10% abort floor"
                )
            output_token = int(output_token)
            tokens.append(output_token)
            chosen_logprobs.append(float(logprobs[output_token].item()))
            top8.append(top_candidates(mx, logprobs))
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(
            b"".join(int(item).to_bytes(4, "little") for item in tokens)
        ).hexdigest()
        return {
            "tokens": tokens,
            "digest": digest,
            "chosen_logprobs": chosen_logprobs,
            "top8": top8,
            "stats": stats.__dict__,
            "indexed_status": qsa_indexed_status(),
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(tokens) / elapsed,
        }


def candidate_delta(left, right):
    left_map = dict(left)
    right_map = dict(right)
    shared = set(left_map) & set(right_map)
    return {
        "max_shared_delta": max(
            (abs(left_map[token] - right_map[token]) for token in shared),
            default=None,
        ),
        "complete_union": set(left_map) == set(right_map),
        "left_only": sorted(set(left_map) - set(right_map)),
        "right_only": sorted(set(right_map) - set(left_map)),
    }


def percentile95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)]


def run_forward_control(mx, model, token, base, shared, *, mode, width):
    cache = clone_cache(base)
    sequence = [token] + list(shared)
    leading = 0 if width == 1 else len(sequence) % width
    chunks = []
    if leading:
        chunks.append(sequence[:leading])
        sequence = sequence[leading:]
    chunks.extend(
        sequence[index : index + width]
        for index in range(0, len(sequence), width)
    )
    logits = None
    with qsa_mode(mode):
        for chunk in chunks:
            logits = model(mx.array([chunk], dtype=mx.uint32), cache=cache)[0, -1]
            mx.eval(logits, [layer.state for layer in cache])
    logprobs = logits.astype(mx.float32) - mx.logsumexp(
        logits.astype(mx.float32), axis=-1
    )
    candidates = top_candidates(mx, logprobs)
    result = {
        "mode": mode,
        "chunk_width": width,
        "leading_width": leading,
        "forward_widths": [len(chunk) for chunk in chunks],
        "decisive_width": len(chunks[-1]),
        "argmax": candidates[0][0],
        "top8": candidates,
    }
    del cache
    mx.clear_cache()
    return result


def first_divergence(mx, model, token, base, gather, indexed):
    width = min(len(gather["tokens"]), len(indexed["tokens"]))
    at = next(
        (
            index
            for index in range(width)
            if gather["tokens"][index] != indexed["tokens"][index]
        ),
        width if len(gather["tokens"]) != len(indexed["tokens"]) else None,
    )
    if at is None:
        return None
    if at >= width:
        return {"index": at, "classification": "STATE_FAULT", "reason": "length"}
    left = gather["top8"][at]
    right = indexed["top8"][at]
    left_gap = left[0][1] - left[1][1]
    right_gap = right[0][1] - right[1][1]
    same_pair = {left[0][0], left[1][0]} == {right[0][0], right[1][0]}
    near = (
        same_pair
        and left[1][0] == indexed["tokens"][at]
        and right[1][0] == gather["tokens"][at]
        and max(left_gap, right_gap) <= 0.002
    )
    deltas = [
        candidate_delta(gather["top8"][index], indexed["top8"][index])
        for index in range(at + 1)
    ]
    prefix = [
        row["max_shared_delta"]
        for row in deltas[:at]
        if row["max_shared_delta"] is not None
    ]
    delta_at = deltas[at]["max_shared_delta"]
    prefix_p95 = percentile95(prefix)
    prefix_max = max(prefix, default=0.0)
    in_band = delta_at is not None and delta_at <= max(0.002, prefix_max)
    shared = gather["tokens"][:at]
    controls = {
        f"{mode}_t{width}": run_forward_control(
            mx, model, token, base, shared, mode=mode, width=width
        )
        for mode in ("gather", "indexed")
        for width in (1, 3)
    }
    kernel_shape = (
        controls["gather_t1"]["argmax"] == controls["indexed_t1"]["argmax"]
        and controls["gather_t3"]["argmax"] == gather["tokens"][at]
        and controls["indexed_t3"]["argmax"] == indexed["tokens"][at]
        and controls["gather_t3"]["argmax"] != controls["indexed_t3"]["argmax"]
    )
    if near and in_band:
        classification = "NEAR_TIE"
    elif kernel_shape:
        classification = "KERNEL_SHAPE"
    else:
        classification = "STATE_FAULT"
    return {
        "index": at,
        "gather_token": gather["tokens"][at],
        "indexed_token": indexed["tokens"][at],
        "gather_top2": left[:2],
        "indexed_top2": right[:2],
        "gather_gap": left_gap,
        "indexed_gap": right_gap,
        "same_candidate_pair": same_pair,
        "prefix_delta_band": {
            "definition": "max absolute logprob delta over shared top-8 ids",
            "median": statistics.median(prefix) if prefix else 0.0,
            "p95": prefix_p95,
            "max": prefix_max,
            "at_divergence": delta_at,
            "at_divergence_in_band": in_band,
            "incomplete_prefix_rows": sum(
                not row["complete_union"] for row in deltas[:at]
            ),
        },
        "controls": controls,
        "classification": classification,
    }


def prefill_base(
    mx,
    model,
    prompt,
    make_prompt_cache,
    *,
    abort_event=None,
    chunk_size=None,
    abort_phase=4,
):
    base = make_prompt_cache(model)
    tokens = prompt[:-1]
    step = len(tokens) if chunk_size is None else chunk_size
    for start in range(0, len(tokens), step):
        inputs = mx.array(tokens[start : start + step], dtype=mx.uint32)[None]
        output = model(inputs, cache=base)
        mx.eval(output, [layer.state for layer in base])
        if abort_event is not None and abort_event.is_set():
            raise GateFailure(
                abort_phase, "free memory fell below the 10% abort floor"
            )
    return base, int(prompt[-1])


def model_qsa_layer_count(model):
    layers = model.language_model.model.layers
    return sum(not layer.is_linear for layer in layers)


def phase4_model(
    mx,
    model,
    tokenizer,
    max_tokens,
    swap_baseline,
    safety_rows,
    *,
    model_load_free_floor,
    run_free_floor,
    swap_growth_abort_mib,
    wall_deadline,
    contexts=None,
    abort_event=None,
    prefill_chunk_size=None,
):
    from mlx_lm.models.cache import make_prompt_cache

    rows = []
    qsa_layers = model_qsa_layer_count(model)
    for context in MODEL_CONTEXTS if contexts is None else contexts:
        checkpoint = safety_snapshot(f"before_phase4_{context}")
        check_safety(
            checkpoint,
            swap_baseline=swap_baseline,
            phase=4,
            model_load_free_floor=model_load_free_floor,
            run_free_floor=run_free_floor,
            swap_growth_abort_mib=swap_growth_abort_mib,
        )
        prompt = corpus_tokens(tokenizer, context)
        base, token = prefill_base(
            mx,
            model,
            prompt,
            make_prompt_cache,
            abort_event=abort_event,
            chunk_size=prefill_chunk_size,
        )
        gather = run_model_arm(
            mx,
            model,
            token,
            clone_cache(base),
            mode="gather",
            max_tokens=max_tokens,
            abort_event=abort_event,
        )
        indexed = run_model_arm(
            mx,
            model,
            token,
            clone_cache(base),
            mode="indexed",
            max_tokens=max_tokens,
            abort_event=abort_event,
        )
        divergence = first_divergence(mx, model, token, base, gather, indexed)
        max_logprob_delta = max(
            (
                abs(left - right)
                for left, right in zip(
                    gather["chosen_logprobs"], indexed["chosen_logprobs"]
                )
            ),
            default=0.0,
        )
        status = indexed["indexed_status"]
        reached = status["query_width_counts"].get("2-8", {}).get("engaged", 0)
        cycles = indexed["stats"].get(
            "draft_cycles", indexed["stats"].get("cycles", 0)
        )
        expected_verify_calls = qsa_layers * cycles
        receipt = {
            "qsa_layers": qsa_layers,
            "self_mtp_rounds": cycles,
            "engaged_verify_calls": reached,
            "expected_engaged_verify_calls": expected_verify_calls,
            "engaged_calls_per_qsa_layer_per_round": (
                reached / expected_verify_calls if expected_verify_calls else 0.0
            ),
            "candidate": status["candidate"],
            "fallbacks": status["fallbacks"],
        }
        passed = (
            max_logprob_delta <= 0.002
            and not status["fallbacks"]
            and status["candidate"] is not None
            and reached == expected_verify_calls
            and (divergence is None or divergence["classification"] == "NEAR_TIE")
        )
        row = {
            "context": context,
            "status": "PASS" if passed else "FAIL",
            "gather_digest": gather["digest"],
            "indexed_digest": indexed["digest"],
            "max_chosen_logprob_delta": max_logprob_delta,
            "divergence": divergence,
            "gather_stats": gather["stats"],
            "indexed_stats": indexed["stats"],
            "indexed_status": status,
            "receipt_contract": receipt,
            "cache_boundary": "both arms clone one cache prefilled through prompt[-2]",
            "draft_contract": "self-MTP k=2 uses width-3 rollback-recording verify",
            "safety": checkpoint,
        }
        rows.append(row)
        del base, gather, indexed
        mx.clear_cache()
        gc.collect()
        after = safety_snapshot(f"after_phase4_{context}")
        safety_rows.append(after)
        row["safety_after"] = after
        try:
            check_safety(
                after,
                swap_baseline=swap_baseline,
                phase=4,
                model_load_free_floor=model_load_free_floor,
                run_free_floor=run_free_floor,
                swap_growth_abort_mib=swap_growth_abort_mib,
            )
        except GateFailure as error:
            return {
                "phase": 4,
                "status": "ABORTED_MEMORY_FLOOR",
                "rows": rows,
                "stopped_after_context": context,
                "safety_failure": str(error),
            }
        if not passed:
            raise GateFailure(4, json.dumps(row, sort_keys=True))
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            return {
                "phase": 4,
                "status": "PARTIAL_TIME_BOUND",
                "rows": rows,
                "stopped_after_context": context,
            }
    return {"phase": 4, "status": "PASS", "rows": rows}


def isolated_timing(
    mx,
    swap_baseline,
    safety_rows,
    *,
    model_load_free_floor,
    run_free_floor,
    swap_growth_abort_mib,
    wall_deadline,
):
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import qwen4_qsa_indexed_attention

    rows = []
    for context in CONTEXTS:
        compact = compact_fixture(mx, QSACompactBlocks, context)
        for length in (3, 1):
            mx.random.seed(20262000 + context + length)
            q = mx.random.normal((1, 24, length, 256)).astype(mx.bfloat16)
            k = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
            v = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
            mask = dense_fixture_mask(mx, context, length)
            arms = {
                "indexed": lambda: qwen4_qsa_indexed_attention(
                    q, k, v, compact, scale=256**-0.5, splits=8
                ),
                "gather": lambda: _gather_qsa_attention(
                    q, k, v, compact, scale=256**-0.5, tile_rows=1
                ),
                "dense_masked": lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=256**-0.5, mask=mask
                ),
            }
            samples = {name: [] for name in arms}
            settlements = []
            for function in arms.values():
                timed(mx, function)
            names = list(arms)
            for repeat in range(8):
                order = names[repeat % 3 :] + names[: repeat % 3]
                for name in order:
                    settlements.append(settle_thermal())
                    samples[name].append(timed(mx, arms[name]))
                    check_safety(
                        safety_snapshot(),
                        swap_baseline=swap_baseline,
                        model_load_free_floor=model_load_free_floor,
                        run_free_floor=run_free_floor,
                        swap_growth_abort_mib=swap_growth_abort_mib,
                    )
            row = {
                    "context": context,
                    "length": length,
                    "query_contract": "M=3 verify" if length == 3 else "M=1 datum",
                    "indexed_min_query": 1,
                    "median_ms": {
                        name: statistics.median(values) * 1000.0
                        for name, values in samples.items()
                    },
                    "settlements": settlements,
                }
            rows.append(row)
            del q, k, v
            mx.clear_cache()
            after = safety_snapshot(f"after_phase5_isolated_{context}_m{length}")
            safety_rows.append(after)
            check_safety(
                after,
                swap_baseline=swap_baseline,
                model_load_free_floor=model_load_free_floor,
                run_free_floor=run_free_floor,
                swap_growth_abort_mib=swap_growth_abort_mib,
            )
            row["safety_after"] = after
            if wall_deadline is not None and time.monotonic() >= wall_deadline:
                row["time_bound_reached"] = True
                return rows
    return rows


def model_timing(
    mx,
    model,
    tokenizer,
    timing_tokens,
    swap_baseline,
    safety_rows,
    *,
    model_load_free_floor,
    run_free_floor,
    swap_growth_abort_mib,
    wall_deadline,
):
    from mlx_lm.models.cache import make_prompt_cache

    rows = []
    for context in MODEL_CONTEXTS:
        prompt = corpus_tokens(tokenizer, context)
        base, token = prefill_base(mx, model, prompt, make_prompt_cache)
        arms = {}
        for mode in ("indexed", "gather", "dense"):
            settlement = settle_thermal()
            result = run_model_arm(
                mx,
                model,
                token,
                clone_cache(base),
                mode=mode,
                max_tokens=timing_tokens,
            )
            check_safety(
                safety_snapshot(),
                swap_baseline=swap_baseline,
                model_load_free_floor=model_load_free_floor,
                run_free_floor=run_free_floor,
                swap_growth_abort_mib=swap_growth_abort_mib,
            )
            label = "plain_dense" if mode == "dense" else mode
            arms[label] = {
                "tokens_per_second": result["tokens_per_second"],
                "elapsed_seconds": result["elapsed_seconds"],
                "digest": result["digest"],
                "settlement": settlement,
                "indexed_status": result["indexed_status"],
            }
        row = {"context": context, "arms": arms}
        rows.append(row)
        del base
        mx.clear_cache()
        after = safety_snapshot(f"after_phase5_end_to_end_{context}")
        safety_rows.append(after)
        check_safety(
            after,
            swap_baseline=swap_baseline,
            model_load_free_floor=model_load_free_floor,
            run_free_floor=run_free_floor,
            swap_growth_abort_mib=swap_growth_abort_mib,
        )
        row["safety_after"] = after
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            row["time_bound_reached"] = True
            return rows
    return rows


def phase5_timing(
    mx,
    model,
    tokenizer,
    timing_tokens,
    swap_baseline,
    safety_rows,
    *,
    model_load_free_floor,
    run_free_floor,
    swap_growth_abort_mib,
    wall_deadline,
):
    isolated = isolated_timing(
        mx,
        swap_baseline,
        safety_rows,
        model_load_free_floor=model_load_free_floor,
        run_free_floor=run_free_floor,
        swap_growth_abort_mib=swap_growth_abort_mib,
        wall_deadline=wall_deadline,
    )
    if isolated and isolated[-1].get("time_bound_reached"):
        return {
            "phase": 5,
            "status": "PARTIAL_TIME_BOUND",
            "isolated": isolated,
            "end_to_end": [],
        }
    end_to_end = model_timing(
        mx,
        model,
        tokenizer,
        timing_tokens,
        swap_baseline,
        safety_rows,
        model_load_free_floor=model_load_free_floor,
        run_free_floor=run_free_floor,
        swap_growth_abort_mib=swap_growth_abort_mib,
        wall_deadline=wall_deadline,
    )
    return {
        "phase": 5,
        "status": (
            "PARTIAL_TIME_BOUND"
            if end_to_end and end_to_end[-1].get("time_bound_reached")
            else "PASS"
        ),
        "isolated": isolated,
        "end_to_end": end_to_end,
    }


def write_artifacts(report, output):
    import mlx.core as mx

    version = str(getattr(mx, "__version__", "unknown"))
    report["manifest"]["mlx_version"] = version
    report["manifest"]["mlx_build_hash"] = (
        version.rsplit("+", 1)[1] if "+" in version else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jsonl = output.with_suffix(".jsonl")
    rows = [report["manifest"]] + report["phases"]
    if "failure" in report:
        rows.append({"type": "failure", **report["failure"]})
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output, jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timing-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-phase", type=int, choices=(1, 4), default=1)
    parser.add_argument("--stop-after-phase", type=int, choices=(3, 5), default=5)
    parser.add_argument(
        "--swap-growth-abort-mib",
        type=float,
        default=DEFAULT_SWAP_LIMIT_MIB,
        help="Set to 0 to record swap growth without aborting.",
    )
    parser.add_argument(
        "--model-load-free-floor-percent",
        type=int,
        default=DEFAULT_MODEL_LOAD_FREE_FLOOR,
    )
    parser.add_argument(
        "--run-free-floor-percent",
        type=int,
        default=DEFAULT_RUN_FREE_FLOOR,
    )
    parser.add_argument("--gpu-wall-limit-minutes", type=float, default=0.0)
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()
    if not args.execute_metal:
        raise SystemExit("refusing to dispatch Metal without --execute-metal")

    os.environ["MLX_QWEN4_QSA_INDEXED"] = "1"
    os.environ["MLX_QWEN4_QSA_GATHER_KV"] = "1"
    os.environ["MLX_QWEN4_QSA_INDEXED_MIN_QUERY"] = "1"
    report = {
        "manifest": {
            "type": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-gate.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent": "codex-l-phase45",
            "model": str(args.model),
            "outcome": "RUNNING",
            "start_phase": args.start_phase,
            "stop_after_phase": args.stop_after_phase,
            "policy": {
                "swap_growth_abort_mib": args.swap_growth_abort_mib,
                "model_load_free_floor_percent": args.model_load_free_floor_percent,
                "run_free_floor_percent": args.run_free_floor_percent,
                "gpu_wall_limit_minutes": args.gpu_wall_limit_minutes,
            },
            "prior_phases_1_3": {
                "commit": "82f2422aee277f0c630cf52c5973a70e8e379b74",
                "json_sha256": (
                    "739e5b23e035f05b72c3e74e1810f5b44715f822c55cfac64ae8ed0cfed7d64f"
                ),
                "jsonl_sha256": (
                    "10971bc6a8ebe765eea492d87f133193b692f81c381709649072b76059083f23"
                ),
                "status": "PASS",
            },
        },
        "phases": [],
        "safety": [],
    }
    exit_code = 0
    with gpu_lock() as owner:
        report["manifest"]["lock_owner"] = owner
        import mlx.core as mx
        from mlx_lm.utils import load

        mx.set_default_device(mx.gpu)
        wall_started = time.monotonic()
        wall_deadline = (
            wall_started + args.gpu_wall_limit_minutes * 60.0
            if args.gpu_wall_limit_minutes > 0
            else None
        )
        baseline = safety_snapshot("before_load")
        report["safety"].append(baseline)
        swap_baseline = baseline["swap_used_mib"]
        current_phase = args.start_phase
        model = None
        tokenizer = None
        try:
            if args.start_phase == 1:
                report["phases"].append(phase1_candidate(mx))
                current_phase = 2
                report["phases"].append(phase2_exactness(mx))
                current_phase = 3
                report["phases"].append(phase3_gather(mx))
                if args.stop_after_phase == 3:
                    report["manifest"]["outcome"] = "PASS_PHASES_1_3"
                    raise GateStop()
            current_phase = 4
            mx.clear_cache()
            gc.collect()
            check_safety(
                baseline,
                swap_baseline=swap_baseline,
                before_load=True,
                phase=4,
                model_load_free_floor=args.model_load_free_floor_percent,
                run_free_floor=args.run_free_floor_percent,
                swap_growth_abort_mib=args.swap_growth_abort_mib,
            )
            model, tokenizer = load(str(args.model))
            model.eval()
            mx.eval(model.parameters())
            after_load = safety_snapshot("after_load")
            report["safety"].append(after_load)
            check_safety(
                after_load,
                swap_baseline=swap_baseline,
                phase=4,
                model_load_free_floor=args.model_load_free_floor_percent,
                run_free_floor=args.run_free_floor_percent,
                swap_growth_abort_mib=args.swap_growth_abort_mib,
            )
            phase4 = phase4_model(
                mx,
                model,
                tokenizer,
                args.max_tokens,
                swap_baseline,
                report["safety"],
                model_load_free_floor=args.model_load_free_floor_percent,
                run_free_floor=args.run_free_floor_percent,
                swap_growth_abort_mib=args.swap_growth_abort_mib,
                wall_deadline=wall_deadline,
            )
            report["phases"].append(phase4)
            if phase4["status"] == "PASS":
                current_phase = 5
                phase5 = phase5_timing(
                    mx,
                    model,
                    tokenizer,
                    args.timing_tokens,
                    swap_baseline,
                    report["safety"],
                    model_load_free_floor=args.model_load_free_floor_percent,
                    run_free_floor=args.run_free_floor_percent,
                    swap_growth_abort_mib=args.swap_growth_abort_mib,
                    wall_deadline=wall_deadline,
                )
                report["phases"].append(phase5)
                report["manifest"]["outcome"] = (
                    "PASS"
                    if phase5["status"] == "PASS"
                    else "PARTIAL_TIME_BOUND"
                )
            else:
                report["manifest"]["outcome"] = (
                    "PARTIAL_TIME_BOUND"
                    if phase4["status"] == "PARTIAL_TIME_BOUND"
                    else "FAIL_PHASE_4"
                )
                if phase4["status"] != "PARTIAL_TIME_BOUND":
                    report["failure"] = {
                        "phase": 4,
                        "message": phase4.get("safety_failure", phase4["status"]),
                    }
                    exit_code = 1
        except GateStop:
            pass
        except GateFailure as error:
            report["manifest"]["outcome"] = f"FAIL_PHASE_{error.phase}"
            report["failure"] = {"phase": error.phase, "message": str(error)}
            exit_code = 1
        except Exception as error:
            report["manifest"]["outcome"] = f"ERROR_PHASE_{current_phase}"
            report["failure"] = {
                "phase": current_phase,
                "type": type(error).__name__,
                "message": str(error),
            }
            exit_code = 1
        finally:
            model = None
            tokenizer = None
            mx.clear_cache()
            gc.collect()
            final = safety_snapshot("after_unload")
            report["safety"].append(final)
            report["manifest"]["gpu_wall_seconds"] = time.monotonic() - wall_started
            report["manifest"]["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            write_artifacts(report, args.output)
    print(args.output)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
