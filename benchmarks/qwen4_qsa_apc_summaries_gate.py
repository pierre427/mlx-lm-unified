#!/usr/bin/env python3
"""Fresh-process 64K gate for persistent Qwen4 QSA summaries."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
SCHEMA = "qwen4-qsa-apc-summaries-gate-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv, check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_one() -> float | None:
    output = command_output(["/usr/bin/uptime"])
    if output is None:
        return None
    match = re.search(r"load averages?:\s*([0-9.]+)", output)
    return None if match is None else float(match.group(1))


def free_percent() -> int | None:
    output = command_output(["/usr/bin/memory_pressure"])
    if output is None:
        return None
    match = re.search(r"free percentage:\s*(\d+)%", output)
    return None if match is None else int(match.group(1))


def swap_used_mb() -> float | None:
    output = command_output(["/usr/sbin/sysctl", "vm.swapusage"])
    if output is None:
        return None
    match = re.search(r"used =\s*([0-9.]+)M", output)
    return None if match is None else float(match.group(1))


def memory_snapshot(mx=None) -> dict[str, Any]:
    snapshot = {
        "free_percent": free_percent(),
        "swap_used_mb": swap_used_mb(),
        "load_one": load_one(),
    }
    if mx is not None:
        snapshot.update(
            metal_active_gib=mx.get_active_memory() / (1 << 30),
            metal_peak_gib=mx.get_peak_memory() / (1 << 30),
        )
    return snapshot


def require_entry_gates(min_free: int, max_load: float) -> dict[str, Any]:
    snapshot = memory_snapshot()
    if snapshot["free_percent"] is None or snapshot["free_percent"] < min_free:
        raise RuntimeError(
            f"free memory {snapshot['free_percent']}% is below {min_free}%"
        )
    if snapshot["load_one"] is None or snapshot["load_one"] >= max_load:
        raise RuntimeError(
            f"load1 {snapshot['load_one']} is not below {max_load}"
        )
    return snapshot


def require_runtime_gates(
    *, floor: int, baseline_swap: float | None, where: str
) -> dict[str, Any]:
    snapshot = memory_snapshot()
    if snapshot["free_percent"] is None or snapshot["free_percent"] < floor:
        raise RuntimeError(
            f"{where}: free memory {snapshot['free_percent']}% is below {floor}%"
        )
    used = snapshot["swap_used_mb"]
    snapshot["swap_growth_mb"] = (
        None if used is None or baseline_swap is None else used - baseline_swap
    )
    return snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_receipt(array) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

    dtype = str(array.dtype)
    value = np.asarray(array.astype(mx.float32) if array.dtype == mx.bfloat16 else array)
    digest = hashlib.sha256()
    digest.update(dtype.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return {
        "shape": list(value.shape),
        "dtype": dtype,
        "sha256": digest.hexdigest(),
    }


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little"))
    return digest.hexdigest()


def source_corpus() -> str:
    preferred = [
        REPO / "mlx_lm/models/qwen4_exp.py",
        REPO / "mlx_lm/models/qwen3_5.py",
        REPO / "mlx_lm/generate.py",
        REPO / "mlx_lm/models/cache.py",
        REPO / "mlx_lm/server.py",
    ]
    seen: set[Path] = set()
    chunks = []
    for path in preferred + sorted((REPO / "mlx_lm").rglob("*.py")):
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        chunks.append(
            f"\n\n# --- {path.relative_to(REPO.resolve())} ---\n"
            + path.read_text(errors="replace")
        )
    return "".join(chunks)


def build_tokens(tokenizer, total: int) -> list[int]:
    corpus = source_corpus()
    prefix = "Review this real Python source snapshot.\n\n"
    suffix = "\n\nIdentify one concrete cache-correctness risk in the final function."
    target = total + 1024
    lo, hi = 1, len(corpus)
    best: list[int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + corpus[:mid] + suffix}],
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
    if best is None or len(best) < total:
        raise RuntimeError(
            f"could not build {total} realistic tokens; got "
            f"{None if best is None else len(best)}"
        )
    return best[:total]


def configure(model: Path, summaries: bool) -> dict[str, str]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_LM_UBC_EVICT": "1",
        "MLX_QWEN4_PLE_NVME": str(model / "ple_rows.bin"),
        "MLX_QWEN4_PLE_NVME_LRU_MB": "256",
        "MLX_QWEN4_QSA_APC_SUMMARIES": "1" if summaries else "0",
        "MLX_QWEN4_QSA_POOLED_KEY_CACHE": "1",
        "MLX_QWEN4_QSA_SCATTER_CHOSEN": "1",
        "MLX_QWEN4_QSA_NAX_KERNEL": "1",
        "MLX_QWEN4_QSA_NAX_MIN_QUERY": "64",
        "MLX_QWEN4_QSA_NAX_DECODE": "0",
        "MLX_QWEN4_QSA_STAGE1_KERNEL": "1",
        "MLX_QWEN4_QSA_STAGE1_MIN_QUERY": "64",
        "MLX_QWEN4_QSA_STAGE1_MIN_PHYSICAL_KV": "65000",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
        "MLX_QWEN4_FUSED_GDN_VERIFY": "1",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto",
        "MLX_QWEN4_PLE_HASH_BACKEND": "cpu",
    }
    os.environ.update(values)
    return values


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Instrumentation:
    def __init__(self, mx, qwen4_exp):
        self.mx = mx
        self.module = qwen4_exp
        self.selection_records: list[dict[str, Any]] = []
        self.pool_seconds: dict[str, float] = defaultdict(float)
        self.pool_blocks: dict[str, int] = defaultdict(int)
        self.stage1_seconds = 0.0
        self.stage1_calls = 0
        self.capture_selections = True
        self._original_call = qwen4_exp.QSAIndexer.__call__
        self._original_pool = qwen4_exp.QSAIndexer._pool_blocks
        self._original_stage1 = qwen4_exp.qsa_stage1_select

    def install(self) -> None:
        owner = self

        def timed_pool(indexer, raw, starts):
            started = time.perf_counter()
            output = owner._original_pool(indexer, raw, starts)
            owner.mx.eval(output)
            layer = str(indexer.summary_identity["layer_id"])
            owner.pool_seconds[layer] += time.perf_counter() - started
            owner.pool_blocks[layer] += int(starts.shape[0])
            return output

        def timed_stage1(*args, **kwargs):
            started = time.perf_counter()
            output = owner._original_stage1(*args, **kwargs)
            owner.mx.eval(output)
            owner.stage1_seconds += time.perf_counter() - started
            owner.stage1_calls += 1
            return output

        def captured_call(indexer, *args, **kwargs):
            selection = owner._original_call(indexer, *args, **kwargs)
            if not owner.capture_selections:
                return selection
            compact = selection.compact_blocks()
            if compact is None:
                record = {"kind": selection.kind}
            else:
                owner.mx.eval(compact.block_ids, compact.block_counts)
                record = {
                    "kind": selection.kind,
                    "layer_id": str(indexer.summary_identity["layer_id"]),
                    "query_width": int(selection.length),
                    "physical_width": int(selection.physical_width),
                    "block_ids": array_receipt(compact.block_ids),
                    "block_counts": array_receipt(compact.block_counts),
                }
            owner.selection_records.append(record)
            return selection

        self.module.QSAIndexer._pool_blocks = timed_pool
        self.module.qsa_stage1_select = timed_stage1
        self.module.QSAIndexer.__call__ = captured_call

    def reset(self) -> None:
        self.selection_records.clear()
        self.pool_seconds.clear()
        self.pool_blocks.clear()
        self.stage1_seconds = 0.0
        self.stage1_calls = 0

    def report(self) -> dict[str, Any]:
        return {
            "selections": list(self.selection_records),
            "pool_seconds_by_layer": dict(self.pool_seconds),
            "pool_blocks_by_layer": dict(self.pool_blocks),
            "pool_seconds_total": sum(self.pool_seconds.values()),
            "stage1_seconds": self.stage1_seconds,
            "stage1_calls": self.stage1_calls,
        }


def seed(args) -> int:
    entry = require_entry_gates(args.min_free, args.max_load)
    environment = configure(args.model, True)
    import mlx.core as mx
    from mlx_lm.generate import prefill_prompt_cache
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache
    from mlx_lm.models.qwen4_exp import qsa_stage1_status
    from mlx_lm.utils import load

    mx.set_default_device(mx.gpu)
    baseline_swap = swap_used_mb()
    loaded_at = time.perf_counter()
    model, tokenizer = load(str(args.model))
    load_seconds = time.perf_counter() - loaded_at
    require_runtime_gates(
        floor=args.memory_floor, baseline_swap=baseline_swap, where="model load"
    )
    tokens = build_tokens(tokenizer, args.context + args.suffix)
    base, extension = tokens[: args.context], tokens[args.context :]
    cache = make_prompt_cache(model)
    qsa_stage1_status(reset=True)
    started = time.perf_counter()

    def progress(done, total):
        if done == total or (done and done % 8192 == 0):
            snapshot = require_runtime_gates(
                floor=args.memory_floor,
                baseline_swap=baseline_swap,
                where=f"seed prefill {done}/{total}",
            )
            print(
                f"seed {done}/{total} free={snapshot['free_percent']}% "
                f"swap={snapshot['swap_used_mb']}MiB",
                flush=True,
            )

    prefill_prompt_cache(
        model,
        mx.array(base, dtype=mx.uint32),
        cache,
        prefill_step_size=args.prefill_step,
        progress_callback=progress,
    )
    prefill_seconds = time.perf_counter() - started
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    save_prompt_cache(
        str(args.cache),
        cache,
        {
            "schema": SCHEMA,
            "base_token_sha256": token_digest(base),
            "extension_token_sha256": token_digest(extension),
        },
    )
    save_seconds = time.perf_counter() - save_started
    saved_arrays, raw_metadata = mx.load(str(args.cache), return_metadata=True)
    provenance = next(
        (
            json.loads(value)
            for value in raw_metadata.values()
            if '"format": "qsa_apc_summaries"' in value
        ),
        None,
    )
    if provenance is None or not provenance.get("entries"):
        raise RuntimeError("saved cache has no QSA summary provenance")
    del saved_arrays
    final = require_runtime_gates(
        floor=args.memory_floor, baseline_swap=baseline_swap, where="seed save"
    )
    write_json(
        args.result,
        {
            "schema": SCHEMA,
            "phase": "seed",
            "created_at_utc": utc_now(),
            "model": str(args.model),
            "repo_commit": command_output(["/usr/bin/git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "context_tokens": len(base),
            "extension_tokens": len(extension),
            "base_token_sha256": token_digest(base),
            "extension_token_sha256": token_digest(extension),
            "cache": {
                "path": str(args.cache),
                "bytes": args.cache.stat().st_size,
                "sha256": sha256_file(args.cache),
                "summary_provenance": provenance,
            },
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "save_seconds": save_seconds,
            "qsa_stage1": qsa_stage1_status(),
            "environment": environment,
            "memory": {"entry": entry, "final": final},
        },
    )
    return 0


def arm(args) -> int:
    entry = require_entry_gates(args.min_free, args.max_load)
    summaries = args.arm == "warm"
    environment = configure(args.model, summaries)
    import mlx.core as mx
    from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
    from mlx_lm.models import qwen4_exp
    from mlx_lm.models.cache import load_prompt_cache
    from mlx_lm.utils import load

    mx.set_default_device(mx.gpu)
    baseline_swap = swap_used_mb()
    loaded_at = time.perf_counter()
    model, tokenizer = load(str(args.model))
    load_seconds = time.perf_counter() - loaded_at
    require_runtime_gates(
        floor=args.memory_floor, baseline_swap=baseline_swap, where="model load"
    )
    tokens = build_tokens(tokenizer, args.context + args.suffix)
    base, extension = tokens[: args.context], tokens[args.context :]
    seed_receipt = json.loads(args.seed_result.read_text())
    if token_digest(base) != seed_receipt["base_token_sha256"]:
        raise RuntimeError("base token digest differs from seed process")
    if token_digest(extension) != seed_receipt["extension_token_sha256"]:
        raise RuntimeError("extension token digest differs from seed process")

    instrument = Instrumentation(mx, qwen4_exp)
    instrument.install()
    qwen4_exp.qsa_stage1_status(reset=True)
    cache = load_prompt_cache(str(args.cache))
    exact_started = time.perf_counter()
    logit_hidden, _ = model.mtp_backbone(
        mx.array(extension, dtype=mx.uint32)[None], cache
    )
    logits = model.logits(logit_hidden[:, -1:, :])
    mx.eval(logits, [item.state for item in cache])
    exact_seconds = time.perf_counter() - exact_started
    exact_instrumentation = instrument.report()

    instrument.reset()
    instrument.capture_selections = False
    qwen4_exp.qsa_stage1_status(reset=True)
    timed_cache = load_prompt_cache(str(args.cache))
    started = time.perf_counter()
    timed_hidden, _ = model.mtp_backbone(
        mx.array(extension, dtype=mx.uint32)[None], timed_cache
    )
    timed_logits = model.logits(timed_hidden[:, -1:, :])
    mx.eval(timed_logits, [item.state for item in timed_cache])
    ttft_seconds = time.perf_counter() - started
    extension_receipt = {
        "ttft_seconds": ttft_seconds,
        "exactness_seconds": exact_seconds,
        "final_logits": array_receipt(logits),
        "timing_final_logits": array_receipt(timed_logits),
        "exactness_instrumentation": exact_instrumentation,
        "timing_instrumentation": instrument.report(),
        "qsa_stage1": qwen4_exp.qsa_stage1_status(),
        "memory": require_runtime_gates(
            floor=args.memory_floor,
            baseline_swap=baseline_swap,
            where=f"{args.arm} extension",
        ),
    }

    instrument.reset()
    instrument.capture_selections = True
    qwen4_exp.qsa_stage1_status(reset=True)
    mtp_cache = load_prompt_cache(str(args.cache))
    stats = HybridStats()
    mtp_started = time.perf_counter()
    output_tokens = []
    logprob_receipts = []
    for token, logprobs, from_draft in self_mtp_generate_step(
        mx.array(extension, dtype=mx.uint32),
        model,
        num_draft=2,
        max_tokens=args.mtp_tokens,
        prefill_step_size=args.prefill_step,
        persistent_mtp=False,
        prompt_cache=mtp_cache,
        stats=stats,
    ):
        mx.eval(logprobs)
        output_tokens.append({"token": int(token), "from_draft": bool(from_draft)})
        logprob_receipts.append(array_receipt(logprobs))
    mtp_seconds = time.perf_counter() - mtp_started
    mtp_receipt = {
        "seconds": mtp_seconds,
        "tokens": output_tokens,
        "logprobs": logprob_receipts,
        "instrumentation": instrument.report(),
        "qsa_stage1": qwen4_exp.qsa_stage1_status(),
        "stats": dataclasses.asdict(stats),
        "memory": require_runtime_gates(
            floor=args.memory_floor,
            baseline_swap=baseline_swap,
            where=f"{args.arm} self-MTP",
        ),
    }
    write_json(
        args.result,
        {
            "schema": SCHEMA,
            "phase": "arm",
            "arm": args.arm,
            "created_at_utc": utc_now(),
            "model": str(args.model),
            "repo_commit": command_output(["/usr/bin/git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "load_seconds": load_seconds,
            "environment": environment,
            "memory_entry": entry,
            "extension": extension_receipt,
            "self_mtp": mtp_receipt,
        },
    )
    return 0


def compare(args) -> int:
    seed_receipt = json.loads(args.seed_result.read_text())
    warm = json.loads(args.warm_result.read_text())
    cold = json.loads(args.cold_result.read_text())
    extension_checks = {
        "selection_sets_bit_identical": warm["extension"]["exactness_instrumentation"]["selections"]
        == cold["extension"]["exactness_instrumentation"]["selections"],
        "logits_bit_identical": warm["extension"]["final_logits"]
        == cold["extension"]["final_logits"],
        "timing_logits_bit_identical": warm["extension"]["timing_final_logits"]
        == cold["extension"]["timing_final_logits"],
    }
    mtp_checks = {
        "tokens_bit_identical": warm["self_mtp"]["tokens"]
        == cold["self_mtp"]["tokens"],
        "logprobs_bit_identical": warm["self_mtp"]["logprobs"]
        == cold["self_mtp"]["logprobs"],
        "selection_sets_bit_identical": warm["self_mtp"]["instrumentation"]["selections"]
        == cold["self_mtp"]["instrumentation"]["selections"],
    }
    warm_time = warm["extension"]["ttft_seconds"]
    cold_time = cold["extension"]["ttft_seconds"]
    saving = cold_time - warm_time
    passed = all(extension_checks.values()) and all(mtp_checks.values())
    result = {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "status": "passed" if passed else "failed",
        "decision": "admit" if passed and saving > 0 else "hold",
        "seed": seed_receipt,
        "warm": warm,
        "cold": cold,
        "exactness": {"extension": extension_checks, "self_mtp": mtp_checks},
        "timing": {
            "warm_ttft_seconds": warm_time,
            "cold_ttft_seconds": cold_time,
            "ttft_saving_seconds": saving,
            "ttft_saving_percent": 100 * saving / cold_time,
            "warm_pool_seconds": warm["extension"]["timing_instrumentation"]["pool_seconds_total"],
            "cold_pool_seconds": cold["extension"]["timing_instrumentation"]["pool_seconds_total"],
            "warm_stage1_seconds": warm["extension"]["timing_instrumentation"]["stage1_seconds"],
            "cold_stage1_seconds": cold["extension"]["timing_instrumentation"]["stage1_seconds"],
        },
    }
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "timing": result["timing"]}, indent=2))
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    def metal_phase(name: str):
        phase = subparsers.add_parser(name)
        phase.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        phase.add_argument("--cache", type=Path, required=True)
        phase.add_argument("--result", type=Path, required=True)
        phase.add_argument("--context", type=int, default=65536)
        phase.add_argument("--suffix", type=int, default=2048)
        phase.add_argument("--prefill-step", type=int, default=2048)
        phase.add_argument("--min-free", type=int, default=45)
        phase.add_argument("--memory-floor", type=int, default=15)
        phase.add_argument("--max-load", type=float, default=10.0)
        phase.add_argument("--execute-metal", action="store_true")
        return phase

    seed_parser = metal_phase("seed")
    seed_parser.set_defaults(handler=seed)
    arm_parser = metal_phase("arm")
    arm_parser.add_argument("--arm", choices=("warm", "cold"), required=True)
    arm_parser.add_argument("--seed-result", type=Path, required=True)
    arm_parser.add_argument("--mtp-tokens", type=int, default=64)
    arm_parser.set_defaults(handler=arm)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--seed-result", type=Path, required=True)
    compare_parser.add_argument("--warm-result", type=Path, required=True)
    compare_parser.add_argument("--cold-result", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(handler=compare)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase in ("seed", "arm"):
        if not args.execute_metal:
            print(json.dumps({"status": "plan-only", "phase": args.phase}, indent=2))
            return 0
        if not args.model.is_dir() or not (args.model / "ple_rows.bin").is_file():
            raise SystemExit("the pinned model and PLE sidecar must exist")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
