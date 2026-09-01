#!/usr/bin/env python3
"""Real-weight Qwen4 composed call-reduction context ladder.

The benchmark keeps one resident model and one common optimized configuration.
For each context it prefills once, clones the immutable cache boundary, and
runs mirrored prior/composed greedy trajectories.  The composed arm adds the
direct-M1 selected-block QSA path, exact router, and one-dispatch GDN affine-q4
epilogue.  QSA NAX is a tolerance-gated accuracy change; greedy tokens and GDN
state must remain exact while full logits stay inside the declared norm gate.

Self-MTP and prompt-lookup drafting are intentionally absent: their verifier
width is greater than one, while the fused-GDN kernel is admitted only for
ordinary B=1/M=1 decode.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
DEFAULT_CONTEXTS = "1024,4096,8192,16384,32768,65536"
DEFAULT_OUTPUT = Path(
    "/Users/pierrelamy/Desktop/mlx-uag/results/"
    "qwen4-composed-call-reduction-context-ladder-20260901.jsonl"
)
PRESSURE_CRITICAL = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or any(item < 2 for item in result):
        raise ValueError("contexts must contain integers >= 2")
    return result


def command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def swap_used_mb() -> float | None:
    text = command_output(["/usr/sbin/sysctl", "vm.swapusage"])
    if not text or "used =" not in text:
        return None
    try:
        return float(text.split("used =", 1)[1].split("M", 1)[0].strip())
    except ValueError:
        return None


def memory_snapshot(mx) -> dict[str, Any]:
    pressure = command_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
    )
    return {
        "swap_used_mb": swap_used_mb(),
        "pressure_level": int(pressure) if pressure and pressure.isdigit() else None,
        "metal_active_gib": mx.get_active_memory() / (1 << 30),
        "metal_peak_gib": mx.get_peak_memory() / (1 << 30),
        "thermal": command_output(["/usr/bin/pmset", "-g", "therm"]),
    }


def memory_violation(
    snapshot: dict[str, Any], baseline_swap_mb: float | None, max_swap_growth_mb: float
) -> str | None:
    if snapshot.get("pressure_level") == PRESSURE_CRITICAL:
        return "macOS memory pressure is critical"
    used = snapshot.get("swap_used_mb")
    if used is not None and baseline_swap_mb is not None:
        growth = used - baseline_swap_mb
        if growth > max_swap_growth_mb:
            return (
                f"swap grew {growth:.0f} MB over baseline; "
                f"limit is {max_swap_growth_mb:.0f} MB"
            )
    return None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def configure_common_stack(model_path: Path) -> dict[str, str]:
    """Pin the measured/common configuration before importing mlx-lm."""
    values = {
        "MLX_ENABLE_TF32": "0",
        "MLX_QWEN4_PLE_NVME": str(model_path / "ple_rows.bin"),
        "MLX_QWEN4_PLE_NVME_LRU_MB": "256",
        "MLX_QWEN4_QSA_SCATTER_CHOSEN": "1",
        "MLX_QWEN4_QSA_POOLED_KEY_CACHE": "1",
        "MLX_QWEN4_QSA_NAX_KERNEL": "1",
        "MLX_QWEN4_QSA_NAX_MIN_QUERY": "64",
        "MLX_QWEN4_QSA_NAX_DECODE": "0",
        "MLX_QWEN4_QSA_STAGE1_KERNEL": "1",
        "MLX_QWEN4_QSA_STAGE1_MIN_QUERY": "64",
        # The held-out tail token makes the last 64K prefill chunk expose
        # 65,532 closed tokens.  Admit that exact-64K operating point.
        "MLX_QWEN4_QSA_STAGE1_MIN_PHYSICAL_KV": "65000",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
        "MLX_QWEN35_GDN_PROJ_FUSION_SUBCLASS": "1",
        # The benchmark toggles this after load so the prefill/common boundary
        # and every stock arm are explicit.
        "MLX_QWEN4_FUSED_GDN_DECODE": "0",
        # Pin rejected, disqualified, diagnostic, or non-reaching candidates
        # off so a caller's shell cannot silently change the common arm.
        "MLX_QWEN4_RMSNORM_FAST": "0",
        "MLX_QWEN4_PLE_VECTOR_SHIFT": "0",
        "MLX_QWEN4_PLE_GATHER_CONCAT": "0",
        "MLX_QWEN4_GDN_SHAPE_STABLE_PROJECTIONS": "0",
        "MLX_QWEN4_SHAPE_STABLE_SHORT_FORWARD": "0",
        "MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT": "0",
        "MLX_QWEN4_QSA_FUSED_PROJ": "0",
        "MLX_QWEN4_QSA_GATHER_KV": "0",
        "MLX_QWEN4_MOE_GATE_COMPILE": "0",
        "MLX_QWEN4_MOE_ROUTER_KERNEL": "0",
        "MLX_QWEN4_MOE_SHARED_IN_GATHER": "0",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "0",
        "MLX_QWEN4_PLE_HASH_BACKEND": "cpu",
    }
    os.environ.update(values)
    return values


def clone_state_containers(value):
    if isinstance(value, list):
        return [clone_state_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_containers(item) for item in value)
    if isinstance(value, dict):
        return {
            clone_state_containers(key): clone_state_containers(item)
            for key, item in value.items()
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


def cache_arrays(cache) -> list:
    arrays = []
    for layer in cache:
        state = layer.state
        pending = [state]
        while pending:
            value = pending.pop()
            if isinstance(value, (tuple, list)):
                pending.extend(value)
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif value is not None and hasattr(value, "dtype"):
                arrays.append(value)
    return arrays


def gdn_cache_arrays(cache, qwen4_cache_type) -> list:
    return [
        value
        for layer in cache
        if isinstance(layer, qwen4_cache_type)
        for value in layer.cache
        if value is not None
    ]


def gdn_caches_equal(left, right, qwen4_cache_type, mx) -> bool:
    left_arrays = gdn_cache_arrays(left, qwen4_cache_type)
    right_arrays = gdn_cache_arrays(right, qwen4_cache_type)
    if len(left_arrays) != len(right_arrays):
        return False
    checks = [mx.array_equal(a, b) for a, b in zip(left_arrays, right_arrays)]
    mx.eval(checks)
    return all(bool(check.item()) for check in checks)


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def source_corpus(source_root: Path) -> str:
    preferred = [
        source_root / "mlx_lm/models/qwen4_exp.py",
        source_root / "mlx_lm/models/qwen3_5.py",
        source_root / "mlx_lm/generate.py",
        source_root / "mlx_lm/models/cache.py",
        source_root / "mlx_lm/server.py",
    ]
    seen: set[Path] = set()
    chunks: list[str] = []
    for path in preferred + sorted((source_root / "mlx_lm").rglob("*.py")):
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        chunks.append(
            f"\n\n# --- {path.relative_to(source_root.resolve())} ---\n"
            + path.read_text(errors="replace")
        )
    return "".join(chunks)


def build_prompt(tokenizer, corpus: str, target: int) -> list[int]:
    prefix = "Review this real Python source snapshot.\n\n"
    ask = "\n\nIdentify one concrete cache-correctness risk in the final function."
    lo, hi = 1, min(len(corpus), target * 8)
    best: list[int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + corpus[:mid] + ask}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
        if len(tokens) <= target:
            best = list(tokens)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None or len(best) < target * 0.98:
        raise RuntimeError(
            f"could not construct prompt near {target}; got "
            f"{None if best is None else len(best)}"
        )
    return best


def run_trajectory(
    *,
    model,
    base_cache,
    first_token,
    steps: int,
    rep: int,
    mx,
    qwen4_cache_type,
    set_arm_mode,
    gdn_stats,
    router_stats,
    qsa_stats,
    expect_direct_qsa: bool,
) -> dict[str, Any]:
    prior_cache = clone_cache(base_cache)
    composed_cache = clone_cache(base_cache)
    mx.eval(cache_arrays(prior_cache), cache_arrays(composed_cache))
    token = first_token
    timings = {"prior": [], "composed": []}
    generated: list[int] = []
    before_gdn = gdn_stats(model)
    before_router = router_stats(model)
    before_qsa = qsa_stats()
    mismatch = None
    max_logit_abs_observed = 0.0
    max_relative_logit_l2_observed = 0.0

    for step in range(steps):
        outputs = {}
        order = (
            ("prior", "composed")
            if (rep + step) % 2 == 0
            else ("composed", "prior")
        )
        for mode in order:
            cache = prior_cache if mode == "prior" else composed_cache
            set_arm_mode(mode)
            started = time.perf_counter()
            output = model(token[:, None], cache=cache)
            mx.eval(output, gdn_cache_arrays(cache, qwen4_cache_type))
            timings[mode].append(time.perf_counter() - started)
            outputs[mode] = output

        prior = outputs["prior"]
        composed = outputs["composed"]
        logits_equal = bool(mx.array_equal(prior, composed).item())
        max_logit_abs = float(mx.max(mx.abs(prior - composed)).item())
        rel_logit_l2 = float(
            mx.linalg.norm((prior - composed).astype(mx.float32)).item()
            / max(mx.linalg.norm(prior.astype(mx.float32)).item(), 1.0e-12)
        )
        greedy_equal = int(mx.argmax(prior[:, -1], axis=-1).item()) == int(
            mx.argmax(composed[:, -1], axis=-1).item()
        )
        max_logit_abs_observed = max(max_logit_abs_observed, max_logit_abs)
        max_relative_logit_l2_observed = max(
            max_relative_logit_l2_observed, rel_logit_l2
        )
        states_equal = gdn_caches_equal(
            prior_cache, composed_cache, qwen4_cache_type, mx
        )
        if not greedy_equal or not states_equal:
            mismatch = {
                "step": step,
                "logits_equal": logits_equal,
                "greedy_equal": greedy_equal,
                "gdn_caches_equal": states_equal,
                "max_logit_abs": max_logit_abs,
                "relative_logit_l2": rel_logit_l2,
            }
            break
        token = mx.argmax(prior[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(token)
        generated.append(int(token.item()))

    after_gdn = gdn_stats(model)
    after_router = router_stats(model)
    after_qsa = qsa_stats()
    completed = len(generated)
    attempted = len(timings["composed"])
    expected_gdn_calls = 36 * attempted
    gdn_calls = (
        after_gdn["fused_outproj_calls"] - before_gdn["fused_outproj_calls"]
    )
    gdn_fallbacks = after_gdn["fallbacks"] - before_gdn["fallbacks"]
    expected_router_calls = 48 * attempted
    router_calls = after_router["fused_calls"] - before_router["fused_calls"]
    router_fallbacks = after_router["fallbacks"] - before_router["fallbacks"]
    qsa_before = before_qsa["counts"].get("engaged", 0)
    qsa_calls = after_qsa["counts"].get("engaged", 0) - qsa_before
    expected_qsa_calls = 12 * attempted if expect_direct_qsa else 0
    medians = {
        mode: statistics.median(values) if values else None
        for mode, values in timings.items()
    }
    aggregates = {mode: sum(values) for mode, values in timings.items()}
    return {
        "passed": mismatch is None and completed == steps
        and gdn_calls == expected_gdn_calls and gdn_fallbacks == 0
        and router_calls == expected_router_calls and router_fallbacks == 0
        and qsa_calls == expected_qsa_calls,
        "steps": completed,
        "mismatch": mismatch,
        "max_logit_abs_observed": max_logit_abs_observed,
        "max_relative_logit_l2_observed": max_relative_logit_l2_observed,
        "gdn_outproj_calls": gdn_calls,
        "expected_gdn_outproj_calls": expected_gdn_calls,
        "gdn_fallbacks": gdn_fallbacks,
        "router_calls": router_calls,
        "expected_router_calls": expected_router_calls,
        "router_fallbacks": router_fallbacks,
        "direct_qsa_calls": qsa_calls,
        "expected_direct_qsa_calls": expected_qsa_calls,
        "timings_s": timings,
        "median_seconds_per_token": medians,
        "aggregate_seconds": aggregates,
        "median_speedup_percent": 100.0
        * (medians["prior"] / medians["composed"] - 1.0),
        "aggregate_speedup_percent": 100.0
        * (aggregates["prior"] / aggregates["composed"] - 1.0),
        "token_sha256": token_digest(generated),
        "token_ids": generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--max-swap-growth-mb", type=float, default=512.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--direct-qsa",
        action="store_true",
        help="include tolerance-class direct M1 QSA (failed 8K greedy gate)",
    )
    parser.add_argument("--execute-metal", action="store_true")
    args = parser.parse_args()
    contexts = parse_csv_ints(args.contexts)
    if not args.execute_metal:
        print(json.dumps({
            "status": "plan-only",
            "contexts": contexts,
            "steps": args.steps,
            "reps": args.reps,
            "model": str(args.model),
        }, indent=2))
        return 0
    if not args.model.is_dir() or not (args.model / "ple_rows.bin").is_file():
        raise SystemExit("model and its PLE-NVMe sidecar must exist locally")

    common_env = configure_common_stack(args.model)
    import mlx.core as mx
    from mlx_lm.generate import prefill_prompt_cache
    from mlx_lm.models.qwen3_5 import fuse_gated_delta_net_projections
    from mlx_lm.models.qwen3_next import (
        qwen4_fused_expert_mode_counts,
        qwen4_moe_router_stats,
        set_qwen4_fused_expert_mode,
        set_qwen4_moe_router_mode,
    )
    from mlx_lm.models.qwen4_exp import (
        Qwen4ArraysCache,
        qsa_nax_decode_status,
        qsa_stage1_status,
        qwen4_fused_gdn_mode_counts,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
        set_qwen4_qsa_nax_decode,
    )
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load

    mx.set_default_device(mx.gpu)
    baseline_swap = swap_used_mb()
    append_jsonl(args.output, {
        "event": "start",
        "created_at_utc": utc_now(),
        "model": str(args.model),
        "contexts": contexts,
        "steps": args.steps,
        "reps": args.reps,
        "prefill_step_size": args.prefill_step_size,
        "baseline_swap_mb": baseline_swap,
        "max_swap_growth_mb": args.max_swap_growth_mb,
        "common_environment": common_env,
        "arm_difference": {
            "prior": "existing fused GDN; stock router; dense M1 QSA",
            "composed": (
                "one-dispatch GDN+q4; exact router; "
                + ("direct M1 QSA" if args.direct_qsa else "stock M1 QSA")
            ),
        },
        "excluded_conflicts": ["self-MTP", "prompt-lookup drafting", "adaptive PLD"],
    })

    loaded_at = time.perf_counter()
    model, tokenizer = load(str(args.model))
    model.eval()
    projection_layers = fuse_gated_delta_net_projections(model, enabled=True)
    fused_expert_layers = set_qwen4_fused_expert_mode(model, "tile4")
    set_qwen4_fused_gdn_mode(model, "fused")
    set_qwen4_moe_router_mode(model, "stock")
    set_qwen4_qsa_nax_decode(False)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - loaded_at
    loaded_memory = memory_snapshot(mx)
    reason = memory_violation(
        loaded_memory, baseline_swap, args.max_swap_growth_mb
    )
    append_jsonl(args.output, {
        "event": "loaded",
        "created_at_utc": utc_now(),
        "load_s": load_s,
        "gdn_projection_fused_layers": projection_layers,
        "fused_expert_layers": fused_expert_layers,
        "fused_expert_modes": qwen4_fused_expert_mode_counts(model),
        "gdn_modes": qwen4_fused_gdn_mode_counts(model),
        "memory": loaded_memory,
    })
    if reason:
        raise RuntimeError(reason)
    if projection_layers != 36 or fused_expert_layers != 49:
        raise RuntimeError(
            f"optimized stack did not reach expected topology: "
            f"GDN={projection_layers}, MoE={fused_expert_layers}"
        )

    def set_arm_mode(mode: str) -> None:
        if mode == "prior":
            set_qwen4_fused_gdn_mode(model, "fused")
            set_qwen4_moe_router_mode(model, "stock")
            set_qwen4_qsa_nax_decode(False)
        elif mode == "composed":
            set_qwen4_fused_gdn_mode(model, "fused_outproj")
            set_qwen4_moe_router_mode(model, "fused")
            set_qwen4_qsa_nax_decode(args.direct_qsa)
        else:
            raise ValueError(mode)

    corpus = source_corpus(Path(__file__).resolve().parents[1])
    for context in contexts:
        prompt = build_prompt(tokenizer, corpus, context)
        prompt_array = mx.array(prompt, dtype=mx.uint32)
        cache = make_prompt_cache(model)
        qsa_stage1_status(reset=True)
        mx.reset_peak_memory()
        prefill_at = time.perf_counter()
        if len(prompt) > 1:
            prefill_prompt_cache(
                model,
                prompt_array[:-1],
                cache,
                prefill_step_size=args.prefill_step_size,
                progress_callback=lambda done, total, c=context: print(
                    f"ctx={c} prefill={done}/{total}", flush=True
                ) if done == total or done % 8192 == 0 else None,
            )
        set_arm_mode("prior")
        tail_logits = model(prompt_array[-1:][None], cache=cache)
        mx.eval(tail_logits, cache_arrays(cache))
        prefill_s = time.perf_counter() - prefill_at
        first_token = mx.argmax(tail_logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(first_token)
        after_prefill = memory_snapshot(mx)
        reason = memory_violation(
            after_prefill, baseline_swap, args.max_swap_growth_mb
        )
        append_jsonl(args.output, {
            "event": "prefill",
            "created_at_utc": utc_now(),
            "context_target": context,
            "actual_prompt_tokens": len(prompt),
            "prefill_s": prefill_s,
            "prefill_tokens_per_s": len(prompt) / prefill_s,
            "first_token": int(first_token.item()),
            "qsa_stage1": qsa_stage1_status(),
            "memory": after_prefill,
        })
        print(
            f"ctx={context} prompt={len(prompt)} prefill={prefill_s:.2f}s "
            f"peak={after_prefill['metal_peak_gib']:.2f}GiB",
            flush=True,
        )
        if reason:
            raise RuntimeError(reason)

        # Compile both decode graphs outside timed observations.
        for warm_mode in ("prior", "composed"):
            warm_cache = clone_cache(cache)
            set_arm_mode(warm_mode)
            warm_logits = model(first_token[:, None], cache=warm_cache)
            mx.eval(warm_logits, gdn_cache_arrays(warm_cache, Qwen4ArraysCache))
            del warm_cache, warm_logits

        for rep in range(args.reps):
            mx.reset_peak_memory()
            result = run_trajectory(
                model=model,
                base_cache=cache,
                first_token=first_token,
                steps=args.steps,
                rep=rep,
                mx=mx,
                qwen4_cache_type=Qwen4ArraysCache,
                set_arm_mode=set_arm_mode,
                gdn_stats=qwen4_fused_gdn_stats,
                router_stats=qwen4_moe_router_stats,
                qsa_stats=qsa_nax_decode_status,
                expect_direct_qsa=args.direct_qsa and len(prompt) > 2051,
            )
            snapshot = memory_snapshot(mx)
            row = {
                "event": "trajectory",
                "created_at_utc": utc_now(),
                "context_target": context,
                "actual_prompt_tokens": len(prompt),
                "rep": rep,
                "result": result,
                "memory": snapshot,
            }
            append_jsonl(args.output, row)
            print(
                f"ctx={context} rep={rep} pass={result['passed']} "
                f"prior={result['median_seconds_per_token']['prior']*1000:.3f}ms "
                f"composed={result['median_seconds_per_token']['composed']*1000:.3f}ms "
                f"gain={result['median_speedup_percent']:+.2f}% "
                f"calls=gdn:{result['gdn_outproj_calls']}/"
                f"{result['expected_gdn_outproj_calls']} router:"
                f"{result['router_calls']}/{result['expected_router_calls']} "
                f"qsa:{result['direct_qsa_calls']}/"
                f"{result['expected_direct_qsa_calls']}",
                flush=True,
            )
            reason = memory_violation(snapshot, baseline_swap, args.max_swap_growth_mb)
            if not result["passed"]:
                raise RuntimeError(f"parity or engagement failed at context {context}")
            if reason:
                raise RuntimeError(reason)

        del cache, prompt_array, tail_logits, first_token
        gc.collect()
        mx.clear_cache()

    final_memory = memory_snapshot(mx)
    append_jsonl(args.output, {
        "event": "complete",
        "created_at_utc": utc_now(),
        "memory": final_memory,
    })
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
