#!/usr/bin/env python3
"""Real-weight A/B for the fused GDN input-projection table.

One resident model, one prefill per context, two mirrored greedy trajectories
that alternate which arm runs first at every step. The arms differ in exactly
one thing -- ``MLX_QWEN4_GDN_FUSED_INPROJ`` -- and every step asserts the two
arms produced identical logits and identical GDN cache state, so a null result
cannot be an arm that silently never engaged: the fused call counter is checked
against ``36 * steps`` before any timing is reported.

Width 3 is measured beside width 1 because the deployed profile speculates
(self-MTP k=2), so its verify slab is three rows wide and takes the same fused
table as ordinary decode.

Plan-only by default; ``--execute-metal`` loads the model and runs on the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(
    "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
DEFAULT_OUTPUT = Path(
    "/Users/pierrelamy/Desktop/mlx-uag/results/"
    "qwen4-gdn-onekernel-20260902-ab.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def configure_stack(model_path: Path) -> dict[str, str]:
    """Pin the deployed Flash-Next serving profile before importing mlx-lm.

    Taken from the launchd ``serve.sh`` so the A/B runs against the stack that
    actually serves, not an isolated one; the only deviation is the lever under
    test, which both arms toggle after load.
    """
    values = {
        "MLX_ENABLE_TF32": "0",
        "MLX_QWEN4_PLE_NVME": str(model_path / "ple_rows.bin"),
        "MLX_QWEN4_PLE_NVME_LRU_MB": "256",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto",
        "MLX_QWEN4_FUSED_GDN_VERIFY": "1",
        # Under test; both arms set it per-layer after load.
        "MLX_QWEN4_GDN_FUSED_INPROJ": "0",
        # Not deployed for decode; a separate arm switches it on.
        "MLX_QWEN4_FUSED_GDN_DECODE": "0",
        # The in-place rewrite would delete the split projections this lever
        # keeps; pin it off so the two mechanisms cannot overlap.
        "MLX_QWEN35_GDN_PROJ_FUSION_SUBCLASS": "0",
        "MLX_QWEN4_GDN_SHAPE_STABLE_PROJECTIONS": "0",
        "MLX_QWEN4_SHAPE_STABLE_SHORT_FORWARD": "0",
        "MLX_QWEN4_QSA_FUSED_PROJ": "0",
        "MLX_QWEN4_RMSNORM_FAST": "0",
    }
    os.environ.update(values)
    return values


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def clone_state_containers(value):
    if isinstance(value, list):
        return [clone_state_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_containers(item) for item in value)
    if isinstance(value, dict):
        return {
            clone_state_containers(k): clone_state_containers(v)
            for k, v in value.items()
        }
    return value


def clone_cache(cache):
    return [
        type(layer).from_state(
            clone_state_containers(layer.state),
            clone_state_containers(layer.meta_state),
        )
        for layer in cache
    ]


def gdn_cache_arrays(cache, qwen4_cache_type) -> list:
    return [
        value
        for layer in cache
        if isinstance(layer, qwen4_cache_type)
        for value in layer.cache
        if value is not None
    ]


def cache_arrays(cache) -> list:
    arrays = []
    for layer in cache:
        pending = [layer.state]
        while pending:
            value = pending.pop()
            if isinstance(value, (tuple, list)):
                pending.extend(value)
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif value is not None and hasattr(value, "dtype"):
                arrays.append(value)
    return arrays


def run_trajectory(
    *,
    model,
    base_cache,
    first_token,
    steps: int,
    rep: int,
    mx,
    qwen4_cache_type,
    set_arm,
    inproj_stats,
    gdn_layers: int,
) -> dict[str, Any]:
    caches = {"off": clone_cache(base_cache), "on": clone_cache(base_cache)}
    mx.eval([cache_arrays(c) for c in caches.values()])
    token = first_token
    timings: dict[str, list[float]] = {"off": [], "on": []}
    generated: list[int] = []
    # ``reset=True`` returns the count it is about to zero, so the baseline for
    # THIS trajectory is zero, not what came back. Subtracting the returned
    # value instead made every rep after the first report ``calls=0`` and fail
    # its own engagement check while the timings it guarded were fine -- a
    # receipt that cries wolf is as useless as one that never fires.
    inproj_stats(model, reset=True)
    mismatch = None

    for step in range(steps):
        outputs = {}
        order = ("off", "on") if (rep + step) % 2 == 0 else ("on", "off")
        for arm in order:
            set_arm(arm)
            cache = caches[arm]
            started = time.perf_counter()
            output = model(token[:, None], cache=cache)
            mx.eval(output, gdn_cache_arrays(cache, qwen4_cache_type))
            timings[arm].append(time.perf_counter() - started)
            outputs[arm] = output

        logits_equal = bool(mx.array_equal(outputs["off"], outputs["on"]).item())
        state_pairs = [
            mx.array_equal(a, b)
            for a, b in zip(
                gdn_cache_arrays(caches["off"], qwen4_cache_type),
                gdn_cache_arrays(caches["on"], qwen4_cache_type),
            )
        ]
        mx.eval(state_pairs)
        states_equal = all(bool(v.item()) for v in state_pairs)
        if not logits_equal or not states_equal:
            mismatch = {
                "step": step,
                "logits_equal": logits_equal,
                "gdn_caches_equal": states_equal,
                "max_logit_abs": float(
                    mx.max(mx.abs(outputs["off"] - outputs["on"])).item()
                ),
            }
            break
        token = mx.argmax(outputs["off"][:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(token)
        generated.append(int(token.item()))

    after = inproj_stats(model)
    attempted = len(timings["on"])
    calls = after["calls"]
    expected = gdn_layers * attempted
    medians = {
        arm: statistics.median(values) if values else None
        for arm, values in timings.items()
    }
    aggregates = {arm: sum(values) for arm, values in timings.items()}
    return {
        "passed": mismatch is None
        and len(generated) == steps
        and calls == expected,
        "steps": len(generated),
        "mismatch": mismatch,
        "fused_inproj_calls": calls,
        "expected_fused_inproj_calls": expected,
        "median_seconds_per_token": medians,
        "aggregate_seconds": aggregates,
        "median_speedup_percent": (
            100.0 * (medians["off"] / medians["on"] - 1.0) if medians["on"] else None
        ),
        "aggregate_speedup_percent": 100.0 * (aggregates["off"] / aggregates["on"] - 1.0),
        "decode_tps": {
            arm: (1.0 / value if value else None) for arm, value in medians.items()
        },
        "token_sha256": token_digest(generated),
    }


def slab_timing(*, model, base_cache, mx, set_arm, width: int, reps: int, trials: int):
    """Verify-shaped width-``width`` forward, both arms, from a common state.

    Every forward needs its own cache -- a width-3 step advances the recurrent
    state -- so the clones are built OUTSIDE the timed region; cloning inside
    it would charge the arms for host work neither of them does in serving.
    """
    results = {}
    tokens = mx.array([[7 + i for i in range(width)]], dtype=mx.uint32)
    mx.eval(tokens)
    for arm in ("off", "on"):
        set_arm(arm)
        samples = []
        for trial in range(trials + 1):
            caches = [clone_cache(base_cache) for _ in range(reps)]
            mx.eval([cache_arrays(c) for c in caches])
            mx.synchronize()
            started = time.perf_counter()
            out = None
            for cache in caches:
                out = model(tokens, cache=cache)
            mx.eval(out)
            mx.synchronize()
            if trial:  # first trial is the warm-up
                samples.append((time.perf_counter() - started) / reps)
            del caches
        results[arm] = statistics.median(samples)
        mx.clear_cache()
    return {
        "width": width,
        "median_seconds": results,
        "speedup_percent": 100.0 * (results["off"] / results["on"] - 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contexts", default="1024,16384")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--decode-modes",
        default="stock",
        help="comma-separated fused GDN decode modes to sweep the A/B under",
    )
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()

    contexts = [int(p) for p in args.contexts.split(",") if p.strip()]
    decode_modes = [p.strip() for p in args.decode_modes.split(",") if p.strip()]
    if not args.execute_metal:
        print(
            json.dumps(
                {
                    "status": "plan-only",
                    "contexts": contexts,
                    "decode_modes": decode_modes,
                    "steps": args.steps,
                    "reps": args.reps,
                },
                indent=2,
            )
        )
        return 0

    configure_stack(args.model)
    import mlx.core as mx
    from mlx_lm.generate import prefill_prompt_cache
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen4_exp import (
        Qwen4ArraysCache,
        probe_qwen4_gdn_fused_inproj,
        qwen4_gdn_fused_inproj_stats,
        set_qwen4_fused_gdn_mode,
        set_qwen4_gdn_fused_inproj,
    )
    from mlx_lm.utils import load

    import sys

    sys.path.insert(0, str(BENCH_DIR))
    from qwen4_fused_gdn_context_ladder import build_prompt, source_corpus

    mx.set_default_device(mx.gpu)
    append_jsonl(args.output, {"event": "start", "created_at_utc": utc_now()})

    loaded_at = time.perf_counter()
    model, tokenizer = load(str(args.model))
    model.eval()
    mx.eval(model.parameters())
    before_table_bytes = mx.get_active_memory()
    gdn_layers = set_qwen4_gdn_fused_inproj(model, True)
    stats = qwen4_gdn_fused_inproj_stats(model)
    # The tables are lazy until something forces them; measure the resident
    # cost of the second copy this lever keeps, not the graph that promises it.
    for _, module in model.named_modules():
        entry = getattr(module, "_gdn_inproj_fused_cache", None)
        if entry is not None and entry[1] is not None:
            mx.eval([part for part in entry[1][0][:3] if part is not None])
    after_table_bytes = mx.get_active_memory()
    probe = probe_qwen4_gdn_fused_inproj(model)
    append_jsonl(
        args.output,
        {
            "event": "loaded",
            "created_at_utc": utc_now(),
            "load_s": time.perf_counter() - loaded_at,
            "gdn_layers": gdn_layers,
            "inproj_stats": stats,
            "table_bytes": after_table_bytes - before_table_bytes,
            "gpu_parity_probe": probe,
        },
    )
    if stats["eligible"] != gdn_layers:
        raise RuntimeError(
            f"only {stats['eligible']}/{gdn_layers} GDN layers could build a "
            "fused input-projection table"
        )
    if probe["mismatches"]:
        raise RuntimeError(f"GPU byte-parity probe failed: {probe['mismatches']}")

    def set_arm(arm: str) -> None:
        set_qwen4_gdn_fused_inproj(model, arm == "on")

    corpus = source_corpus(BENCH_DIR.parent)
    for context in contexts:
        prompt = build_prompt(tokenizer, corpus, context)
        prompt_array = mx.array(prompt, dtype=mx.uint32)
        set_arm("off")
        cache = make_prompt_cache(model)
        prefill_at = time.perf_counter()
        prefill_prompt_cache(
            model,
            prompt_array[:-1],
            cache,
            prefill_step_size=args.prefill_step_size,
        )
        mx.eval(cache_arrays(cache))
        prefill_s = time.perf_counter() - prefill_at
        first = model(prompt_array[-1][None, None], cache=cache)
        first_token = mx.argmax(first[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(first_token, cache_arrays(cache))

        for mode in decode_modes:
            set_qwen4_fused_gdn_mode(model, mode)
            for rep in range(args.reps):
                row = run_trajectory(
                    model=model,
                    base_cache=cache,
                    first_token=first_token,
                    steps=args.steps,
                    rep=rep,
                    mx=mx,
                    qwen4_cache_type=Qwen4ArraysCache,
                    set_arm=set_arm,
                    inproj_stats=qwen4_gdn_fused_inproj_stats,
                    gdn_layers=gdn_layers,
                )
                row.update(
                    {
                        "event": "trajectory",
                        "created_at_utc": utc_now(),
                        "context": context,
                        "decode_mode": mode,
                        "rep": rep,
                        "prefill_s": prefill_s,
                    }
                )
                append_jsonl(args.output, row)
                print(
                    f"ctx={context} mode={mode} rep={rep} "
                    f"passed={row['passed']} "
                    f"median%={row['median_speedup_percent']:.2f} "
                    f"digest={row['token_sha256'][:12]}"
                )

        set_qwen4_fused_gdn_mode(model, decode_modes[0])
        slab = slab_timing(
            model=model,
            base_cache=cache,
            mx=mx,
            set_arm=set_arm,
            width=3,
            reps=8,
            trials=5,
        )
        slab.update(
            {"event": "slab", "created_at_utc": utc_now(), "context": context}
        )
        append_jsonl(args.output, slab)
        print(f"ctx={context} width3 slab %={slab['speedup_percent']:.2f}")
        del cache
        mx.clear_cache()

    append_jsonl(args.output, {"event": "done", "created_at_utc": utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
