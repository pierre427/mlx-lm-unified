#!/usr/bin/env python3
"""Run the bounded indexed-QSA 32K gate and M=1 timing ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


os.environ["MLX_QWEN4_QSA_INDEXED"] = "1"
os.environ["MLX_QWEN4_QSA_GATHER_KV"] = "1"
os.environ["MLX_QWEN4_QSA_INDEXED_MIN_QUERY"] = "1"

LAB_ROOT = Path("/Users/pierrelamy/Desktop/mlx-uag")
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from benchmarks import qwen4_qsa_indexed_gate as gate


GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
DEFAULT_MODEL = Path(
    "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
DEFAULT_OUTPUT_DIR = LAB_ROOT / "results"
PREFIX = "qwen4-qsa-indexed-timing-20260901"
CONTEXTS = (16_384, 32_768, 65_536, 131_072)
START_FREE_FLOOR = 15
ABORT_FREE_FLOOR = 10


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jsonl = output.with_suffix(".jsonl")
    rows = [report["manifest"]]
    rows.extend(report.get("records", []))
    if report.get("failure"):
        rows.append({"record": "failure", **report["failure"]})
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output, jsonl


@contextmanager
def owned_gpu_lock():
    os.mkdir(GPU_LOCK)
    owner_path = GPU_LOCK / "owner.json"
    owner = {
        "owner": "codex-n-timing",
        "agent": "codex-n-timing",
        "label": "qwen4-qsa-indexed-32k-and-timing",
        "pid": os.getpid(),
        "purpose": "phase 4 at 32K and phase 5 M=1/M=3 timing",
        "started_at": utc_now(),
    }
    try:
        owner_path.write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        yield owner
    finally:
        owner_path.unlink(missing_ok=True)
        GPU_LOCK.rmdir()


def validate_inherited_lock():
    owner = json.loads((GPU_LOCK / "owner.json").read_text(encoding="utf-8"))
    if owner.get("owner") != "codex-n-timing":
        raise RuntimeError(f"GPU lock belongs to {owner.get('owner')!r}")
    return owner


class MemoryGuard:
    """Set an event when model-cell free memory falls below 10%."""

    def __init__(self, poll_seconds=0.5):
        self.event = threading.Event()
        self._stop = threading.Event()
        self._poll_seconds = poll_seconds
        self.breach = None
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self._poll_seconds):
            try:
                snapshot = gate.safety_snapshot("memory_guard")
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
                self.event.set()
                return
            if snapshot["free_percent"] < ABORT_FREE_FLOOR:
                self.breach = snapshot
                self.event.set()
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=2.0)


def check_cell_start(snapshot, phase):
    if snapshot["free_percent"] < START_FREE_FLOOR:
        raise gate.GateFailure(
            phase,
            f"cell start free memory {snapshot['free_percent']}% is below 15%",
        )


def check_abort_floor(snapshot, phase):
    if snapshot["free_percent"] < ABORT_FREE_FLOOR:
        raise gate.GateFailure(
            phase,
            f"free memory {snapshot['free_percent']}% is below the 10% abort floor",
        )


def wait_for_cpu_load(limit=10.0, timeout=600.0):
    started = time.monotonic()
    samples = []
    while True:
        output = subprocess.run(
            ["/usr/bin/uptime"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        match = re.search(r"load averages?:\s*([0-9.]+)", output)
        if match is None:
            raise gate.GateFailure(5, f"could not parse uptime: {output!r}")
        load = float(match.group(1))
        samples.append(
            {"at": utc_now(), "one_minute_load": load, "uptime": output}
        )
        if load < limit:
            return {
                "limit": limit,
                "wait_seconds": time.monotonic() - started,
                "samples": samples,
            }
        if time.monotonic() - started >= timeout:
            raise gate.GateFailure(5, "one-minute CPU load did not fall below 10")
        time.sleep(5.0)


def thermal_baseline():
    import thermal_settle

    record = thermal_settle.calibrate_baseline(
        duration_s=1.0,
        poll_s=1.0,
        band_pct=5.0,
        consecutive=3,
        max_s=30.0,
    )
    if not record["stable"]:
        raise gate.GateFailure(5, "GPU calibration baseline did not stabilize")
    return record


def settle_to_baseline(baseline):
    import thermal_settle

    record = thermal_settle.settle(
        baseline["tflops"],
        duration_s=1.0,
        poll_s=1.0,
        band_pct=5.0,
        consecutive=2,
        min_s=0.0,
        max_s=60.0,
        read_signals=thermal_settle.read_thermal_signals,
    )
    if not record["settled"]:
        raise gate.GateFailure(5, "GPU calibration did not return to baseline")
    return record


def load_model(mx, model_path):
    from mlx_lm.utils import load

    model, tokenizer = load(str(model_path))
    model.eval()
    mx.eval(model.parameters())
    return model, tokenizer


def run_phase4_cell(model_path, context, output):
    validate_inherited_lock()
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    report = {
        "manifest": {
            "record": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-timing.v1",
            "agent": "codex-n-timing",
            "cell": "phase4",
            "context": context,
            "model": str(model_path),
            "started_at": utc_now(),
            "policy": {
                "start_free_floor_percent": START_FREE_FLOOR,
                "abort_free_floor_percent": ABORT_FREE_FLOOR,
                "swap_growth_is_fatal": False,
                "max_tokens": 256,
                "one_model_load_per_process": True,
            },
            "outcome": "RUNNING",
        },
        "records": [],
        "safety": [],
    }
    model = tokenizer = None
    exit_code = 0
    baseline = gate.safety_snapshot("before_load")
    report["safety"].append(baseline)
    try:
        check_cell_start(baseline, 4)
        model, tokenizer = load_model(mx, model_path)
        after_load = gate.safety_snapshot("after_load")
        report["safety"].append(after_load)
        check_cell_start(after_load, 4)
        with MemoryGuard() as guard:
            phase = gate.phase4_model(
                mx,
                model,
                tokenizer,
                256,
                baseline["swap_used_mib"],
                report["safety"],
                model_load_free_floor=START_FREE_FLOOR,
                run_free_floor=START_FREE_FLOOR,
                swap_growth_abort_mib=0,
                wall_deadline=None,
                contexts=(context,),
                abort_event=guard.event,
                prefill_chunk_size=2048,
            )
        report["records"].append(phase)
        report["memory_guard"] = {
            "breach": guard.breach,
            "error": guard.error,
        }
        row = phase.get("rows", [{}])[-1]
        after = row.get("safety_after", {})
        check_abort_floor(after, 4)
        if guard.error is not None:
            raise gate.GateFailure(4, f"memory guard failed: {guard.error}")
        passed = row.get("status") == "PASS" and guard.breach is None
        report["manifest"]["outcome"] = (
            "PASS" if passed else phase.get("status", "FAIL")
        )
        if not passed:
            exit_code = 1
    except gate.GateFailure as error:
        report["manifest"]["outcome"] = (
            "SKIPPED_MEMORY_START"
            if "cell start free memory" in str(error)
            else "FAIL"
        )
        report["failure"] = {"phase": error.phase, "message": str(error)}
        exit_code = 0 if "cell start free memory" in str(error) else 1
    except Exception as error:
        report["manifest"]["outcome"] = "ERROR"
        report["failure"] = {
            "phase": 4,
            "type": type(error).__name__,
            "message": str(error),
        }
        exit_code = 1
    finally:
        model = tokenizer = None
        mx.clear_cache()
        gc.collect()
        report["safety"].append(gate.safety_snapshot("after_unload"))
        report["manifest"]["finished_at"] = utc_now()
        write_report(report, output)
    return exit_code


def run_isolated_cell(output):
    validate_inherited_lock()
    import mlx.core as mx
    from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
    from mlx_lm.models.qwen4_qsa_indexed import qwen4_qsa_indexed_attention

    mx.set_default_device(mx.gpu)
    report = {
        "manifest": {
            "record": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-timing.v1",
            "agent": "codex-n-timing",
            "cell": "isolated",
            "started_at": utc_now(),
            "contexts": list(CONTEXTS),
            "measurement": "discarded warm-up; median of 8; rotated arm order",
            "compact_source": (
                "captured-fixture geometry: 512 sorted four-token blocks plus "
                "the causal tail at the requested physical context width"
            ),
            "outcome": "RUNNING",
        },
        "records": [],
        "safety": [],
    }
    exit_code = 0
    try:
        baseline_safety = gate.safety_snapshot("before_isolated")
        report["safety"].append(baseline_safety)
        check_cell_start(baseline_safety, 5)
        report["cpu_load_gate"] = wait_for_cpu_load()
        for context in CONTEXTS:
            for length in (3, 1):
                before = gate.safety_snapshot(
                    f"before_isolated_{context}_m{length}"
                )
                check_cell_start(before, 5)
                compact = gate.compact_fixture(mx, QSACompactBlocks, context, length)
                mx.random.seed(20262000 + context + length)
                q = mx.random.normal((1, 24, length, 256)).astype(mx.bfloat16)
                k = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
                v = mx.random.normal((1, 2, context, 256)).astype(mx.bfloat16)
                mask = gate.dense_fixture_mask(mx, context, length)
                arms = {
                    "indexed": lambda: qwen4_qsa_indexed_attention(
                        q, k, v, compact, scale=256**-0.5, splits=8
                    ),
                    "dense_masked": lambda: mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=256**-0.5, mask=mask
                    ),
                }
                if length == 3:
                    arms["gather"] = lambda: _gather_qsa_attention(
                        q, k, v, compact, scale=256**-0.5, tile_rows=1
                    )
                for function in arms.values():
                    gate.timed(mx, function)
                cell_settlement = gate.settle_thermal(timeout=30.0)
                names = list(arms)
                samples = {name: [] for name in names}
                controls = []
                for repeat in range(8):
                    offset = repeat % len(names)
                    order = names[offset:] + names[:offset]
                    for name in order:
                        controls.append(
                            {
                                "repeat": repeat,
                                "arm": name,
                                "order": list(order),
                            }
                        )
                        samples[name].append(gate.timed(mx, arms[name]) * 1000.0)
                medians = {
                    name: statistics.median(values)
                    for name, values in samples.items()
                }
                after = gate.safety_snapshot(
                    f"after_isolated_{context}_m{length}"
                )
                check_abort_floor(after, 5)
                report["safety"].append(after)
                report["records"].append(
                    {
                        "record": "isolated_timing",
                        "context": context,
                        "query_width": length,
                        "query_contract": "M=3" if length == 3 else "M=1",
                        "median_ms": medians,
                        "samples_ms": samples,
                        "arm_order": controls,
                        "cell_thermal_settle": cell_settlement,
                        "indexed_speedup_vs_dense": (
                            medians["dense_masked"] / medians["indexed"]
                        ),
                        "indexed_speedup_vs_gather": (
                            medians["gather"] / medians["indexed"]
                            if "gather" in medians
                            else None
                        ),
                        "safety_before": before,
                        "safety_after": after,
                    }
                )
                del arms, q, k, v, mask, compact
                mx.clear_cache()
                gc.collect()
        report["manifest"]["outcome"] = "PASS"
    except Exception as error:
        report["manifest"]["outcome"] = "FAIL"
        report["failure"] = {
            "phase": 5,
            "type": type(error).__name__,
            "message": str(error),
        }
        exit_code = 1
    finally:
        mx.clear_cache()
        gc.collect()
        report["safety"].append(gate.safety_snapshot("after_isolated_unload"))
        report["manifest"]["finished_at"] = utc_now()
        write_report(report, output)
    return exit_code


def run_end_to_end_cell(model_path, context, output):
    validate_inherited_lock()
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    report = {
        "manifest": {
            "record": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-timing.v1",
            "agent": "codex-n-timing",
            "cell": "end_to_end",
            "context": context,
            "model": str(model_path),
            "started_at": utc_now(),
            "measurement": (
                "B1 self-MTP k=2; 3 rotated arm cycles; 128 greedy tokens per run"
            ),
            "outcome": "RUNNING",
        },
        "records": [],
        "safety": [],
    }
    model = tokenizer = base = None
    exit_code = 0
    baseline_safety = gate.safety_snapshot("before_load")
    report["safety"].append(baseline_safety)
    try:
        check_cell_start(baseline_safety, 5)
        model, tokenizer = load_model(mx, model_path)
        after_load = gate.safety_snapshot("after_load")
        report["safety"].append(after_load)
        check_cell_start(after_load, 5)
        prompt = gate.corpus_tokens(tokenizer, context)
        with MemoryGuard() as guard:
            base, token = gate.prefill_base(
                mx,
                model,
                prompt,
                make_prompt_cache,
                abort_event=guard.event,
                chunk_size=2048,
                abort_phase=5,
            )
            for mode in ("indexed", "gather", "dense"):
                gate.run_model_arm(
                    mx,
                    model,
                    token,
                    gate.clone_cache(base),
                    mode=mode,
                    max_tokens=8,
                    abort_event=guard.event,
                    abort_phase=5,
                )
            report["cpu_load_gate"] = wait_for_cpu_load()
            before_timing = gate.safety_snapshot(
                f"before_end_to_end_{context}"
            )
            report["safety"].append(before_timing)
            check_cell_start(before_timing, 5)
            baseline = thermal_baseline()
            samples = {name: [] for name in ("indexed", "gather", "plain_dense")}
            controls = []
            modes = ("indexed", "gather", "dense")
            for repeat in range(3):
                order = modes[repeat:] + modes[:repeat]
                for mode in order:
                    label = "plain_dense" if mode == "dense" else mode
                    settle = (
                        {
                            "settled": True,
                            "reason": "initial stable calibration",
                            "baseline_tflops": baseline["tflops"],
                            "elapsed_s": baseline["elapsed_s"],
                        }
                        if not controls
                        else settle_to_baseline(baseline)
                    )
                    result = gate.run_model_arm(
                        mx,
                        model,
                        token,
                        gate.clone_cache(base),
                        mode=mode,
                        max_tokens=128,
                        abort_event=guard.event,
                        abort_phase=5,
                    )
                    samples[label].append(
                        {
                            "repeat": repeat,
                            "order": list(order),
                            "elapsed_seconds": result["elapsed_seconds"],
                            "tokens_per_second": result["tokens_per_second"],
                            "digest": result["digest"],
                            "indexed_status": result["indexed_status"],
                        }
                    )
                    controls.append(
                        {"repeat": repeat, "arm": label, "settle": settle}
                    )
        if guard.breach is not None:
            raise gate.GateFailure(5, "free memory fell below the 10% abort floor")
        if guard.error is not None:
            raise gate.GateFailure(5, f"memory guard failed: {guard.error}")
        medians = {
            name: statistics.median(
                sample["tokens_per_second"] for sample in arm_samples
            )
            for name, arm_samples in samples.items()
        }
        after = gate.safety_snapshot(f"after_end_to_end_{context}")
        check_abort_floor(after, 5)
        report["safety"].append(after)
        record = {
            "record": "end_to_end_timing",
            "context": context,
            "median_tokens_per_second": medians,
            "samples": samples,
            "thermal_baseline": baseline,
            "arm_order_and_settle": controls,
            "indexed_delta_vs_gather_percent": (
                (medians["indexed"] / medians["gather"] - 1.0) * 100.0
            ),
            "indexed_delta_vs_plain_dense_percent": (
                (medians["indexed"] / medians["plain_dense"] - 1.0) * 100.0
            ),
            "memory_guard": {"breach": guard.breach, "error": guard.error},
            "safety_before": before_timing,
            "safety_after": after,
        }
        report["records"].append(record)
        report["manifest"]["outcome"] = "PASS"
    except gate.GateFailure as error:
        report["manifest"]["outcome"] = (
            "SKIPPED_MEMORY_START"
            if "cell start free memory" in str(error)
            else "FAIL"
        )
        report["failure"] = {"phase": error.phase, "message": str(error)}
        exit_code = 0 if "cell start free memory" in str(error) else 1
    except Exception as error:
        report["manifest"]["outcome"] = "ERROR"
        report["failure"] = {
            "phase": 5,
            "type": type(error).__name__,
            "message": str(error),
        }
        exit_code = 1
    finally:
        base = model = tokenizer = None
        mx.clear_cache()
        gc.collect()
        report["safety"].append(gate.safety_snapshot("after_unload"))
        report["manifest"]["finished_at"] = utc_now()
        write_report(report, output)
    return exit_code


def load_report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def phase4_passed(report):
    rows = [row for phase in report.get("records", []) for row in phase.get("rows", [])]
    return bool(rows and rows[-1].get("status") == "PASS")


def phase4_after_free(report):
    rows = [row for phase in report.get("records", []) for row in phase.get("rows", [])]
    if not rows:
        return None
    return rows[-1].get("safety_after", {}).get("free_percent")


def run_child(args, cell, *, context=None, deadline, attempt=None):
    suffix = cell if context is None else f"{cell}-{context}"
    if attempt is not None:
        suffix = f"{suffix}-{attempt}"
    output = args.output_dir / f"{PREFIX}-{suffix}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        cell,
        "--model",
        str(args.model),
        "--output",
        str(output),
    ]
    if context is not None:
        command.extend(("--context", str(context)))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {"status": "SKIPPED_TIME_BOUND", "path": str(output)}
    try:
        completed = subprocess.run(command, timeout=remaining, check=False)
    except subprocess.TimeoutExpired:
        report = load_report(output) if output.exists() else None
        return {
            "status": "TIME_BOUND",
            "path": str(output),
            "report": report,
        }
    report = load_report(output) if output.exists() else None
    return {
        "status": "DONE" if completed.returncode == 0 else "FAILED",
        "returncode": completed.returncode,
        "path": str(output),
        "json_sha256": sha256(output) if output.exists() else None,
        "jsonl_sha256": (
            sha256(output.with_suffix(".jsonl"))
            if output.with_suffix(".jsonl").exists()
            else None
        ),
        "report": report,
    }


def run_all(args):
    started = time.monotonic()
    deadline = started + args.wall_limit_minutes * 60.0
    combined_path = args.output_dir / f"{PREFIX}.json"
    combined = {
        "manifest": {
            "record": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-timing.v1",
            "agent": "codex-n-timing",
            "started_at": utc_now(),
            "model": str(args.model),
            "gpu_wall_limit_minutes": args.wall_limit_minutes,
            "outcome": "RUNNING",
        },
        "records": [],
    }
    with owned_gpu_lock() as owner:
        combined["manifest"]["lock_owner"] = owner
        phase32 = run_child(args, "phase4", context=32_768, deadline=deadline)
        combined["records"].append({"record": "child", **phase32})
        passed32 = bool(
            phase32.get("report") and phase4_passed(phase32["report"])
        )
        free32 = (
            phase4_after_free(phase32["report"])
            if phase32.get("report")
            else None
        )
        passed64 = False
        if passed32 and free32 is not None and free32 >= 25:
            phase64 = run_child(args, "phase4", context=65_536, deadline=deadline)
            passed64 = bool(
                phase64.get("report") and phase4_passed(phase64["report"])
            )
        else:
            phase64 = {
                "status": "SKIPPED_32K_FREE_BELOW_25_PERCENT",
                "post_32k_free_percent": free32,
            }
        combined["records"].append({"record": "child", **phase64})
        isolated = run_child(args, "isolated", deadline=deadline)
        combined["records"].append({"record": "child", **isolated})
        end16 = run_child(args, "end-to-end", context=16_384, deadline=deadline)
        combined["records"].append({"record": "child", **end16})
        if passed32:
            end32 = run_child(
                args, "end-to-end", context=32_768, deadline=deadline
            )
        else:
            end32 = {"status": "SKIPPED_PHASE4_NOT_PASSED", "context": 32_768}
        combined["records"].append({"record": "child", **end32})
        if passed64:
            end64 = run_child(
                args, "end-to-end", context=65_536, deadline=deadline
            )
        else:
            end64 = {"status": "SKIPPED_PHASE4_NOT_PASSED", "context": 65_536}
        combined["records"].append({"record": "child", **end64})
        combined["manifest"]["outcome"] = "COMPLETE"
        combined["manifest"]["gpu_wall_seconds"] = time.monotonic() - started
        combined["manifest"]["finished_at"] = utc_now()
        write_report(combined, combined_path)
    return 0


def run_remaining(args):
    started = time.monotonic()
    deadline = started + args.wall_limit_minutes * 60.0
    combined_path = args.output_dir / f"{PREFIX}-resume.json"
    combined = {
        "manifest": {
            "record": "manifest",
            "schema": "mlx-uag.qwen4-qsa-indexed-timing.v1",
            "agent": "codex-n-timing",
            "started_at": utc_now(),
            "model": str(args.model),
            "gpu_wall_limit_minutes": args.wall_limit_minutes,
            "prior_32k_contract": (
                "digest/logprob exact; 2064 engaged calls equal "
                "172 draft cycles x 12 QSA layers; zero fallbacks"
            ),
            "outcome": "RUNNING",
        },
        "records": [],
    }
    with owned_gpu_lock() as owner:
        combined["manifest"]["lock_owner"] = owner
        phase64 = run_child(args, "phase4", context=65_536, deadline=deadline)
        combined["records"].append({"record": "child", **phase64})
        passed64 = bool(
            phase64.get("report") and phase4_passed(phase64["report"])
        )
        isolated = run_child(
            args, "isolated", deadline=deadline, attempt="r2"
        )
        combined["records"].append({"record": "child", **isolated})
        for context in (16_384, 32_768):
            result = run_child(
                args,
                "end-to-end",
                context=context,
                deadline=deadline,
                attempt="r2",
            )
            combined["records"].append({"record": "child", **result})
        if passed64:
            end64 = run_child(
                args,
                "end-to-end",
                context=65_536,
                deadline=deadline,
                attempt="r2",
            )
        else:
            end64 = {"status": "SKIPPED_PHASE4_NOT_PASSED", "context": 65_536}
        combined["records"].append({"record": "child", **end64})
        combined["manifest"]["outcome"] = "COMPLETE"
        combined["manifest"]["gpu_wall_seconds"] = time.monotonic() - started
        combined["manifest"]["finished_at"] = utc_now()
        write_report(combined, combined_path)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-remaining", action="store_true")
    parser.add_argument(
        "--cell", choices=("phase4", "isolated", "end-to-end")
    )
    parser.add_argument("--context", type=int, choices=CONTEXTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wall-limit-minutes", type=float, default=60.0)
    args = parser.parse_args()
    if args.run_all:
        return run_all(args)
    if args.run_remaining:
        return run_remaining(args)
    if args.cell is None or args.output is None:
        parser.error("a child run requires --cell and --output")
    if args.cell in {"phase4", "end-to-end"} and args.context is None:
        parser.error(f"{args.cell} requires --context")
    if args.cell == "phase4":
        return run_phase4_cell(args.model, args.context, args.output)
    if args.cell == "isolated":
        return run_isolated_cell(args.output)
    return run_end_to_end_cell(args.model, args.context, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
