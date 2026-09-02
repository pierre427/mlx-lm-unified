#!/usr/bin/env python3
"""Where a decode step's time actually goes: host graph build vs GPU execute.

The compiled-glue arm removed 338 of ~2,078 MLX primitives per token and came
out a NET LOSS. That is only consistent with the per-token cost being paid on
the HOST -- building and encoding the graph -- rather than on the GPU running
it, so this measures the two halves directly instead of inferring them.

Three numbers per step kind, on a warm 16K cache:

  host_build_ms   wall from calling the model to the Python call returning,
                  with the GPU synchronized first. MLX is lazy, so nothing has
                  executed yet: this is graph construction alone.
  gpu_ms          wall of the ``mx.eval`` that immediately follows: encode,
                  execute, and wait.
  pipelined_ms    steady-state wall per step over 32 ``async_eval``-chained
                  steps. If this is ~max(host, gpu) the two halves overlap and
                  only the larger one matters; if it is ~host + gpu they do not
                  and both do.

Primitive counts come from ``mx.export_to_dot`` over the unevaluated graph of
exactly one step (the cache is evaluated first, so nothing older is included);
each ``[label ="Primitive"]`` node is one primitive.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import qwen4_gdn_inproj_ab as AB

MODEL = Path("/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
OUT = Path("/Users/pierrelamy/Desktop/mlx-uag/results/qwen4-gdn-onekernel-20260902-hostbudget.json")
CONTEXT = 16384
TRIALS = 20
PIPELINE_STEPS = 32


def count_primitives(mx, outputs) -> int:
    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as handle:
        path = handle.name
    mx.export_to_dot(path, *outputs)
    text = Path(path).read_text()
    Path(path).unlink()
    return len(re.findall(r'\[label\s*="', text))


def measure(mx, name, step, live_arrays, *, trials=TRIALS, pipeline=PIPELINE_STEPS):
    """host_build / gpu / pipelined for one step kind."""
    for _ in range(3):
        mx.eval(step(), live_arrays())
    mx.synchronize()

    primitives = None
    build, gpu = [], []
    for i in range(trials):
        mx.synchronize()
        t0 = time.perf_counter()
        out = step()
        t1 = time.perf_counter()
        if primitives is None:
            primitives = count_primitives(mx, list(out) + list(live_arrays()))
            # Counting walked the graph but did not evaluate it; the timings
            # below must not include that walk, so this trial's gpu sample is
            # taken after it and the build sample already ended above.
        mx.eval(out, live_arrays())
        t2 = time.perf_counter()
        build.append((t1 - t0) * 1e3)
        gpu.append((t2 - t1) * 1e3)

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(pipeline):
        out = step()
        mx.async_eval(out, live_arrays())
    mx.eval(out, live_arrays())
    mx.synchronize()
    pipelined = (time.perf_counter() - t0) * 1e3 / pipeline

    host_ms = statistics.median(build)
    gpu_ms = statistics.median(gpu)
    return {
        "kind": name,
        "host_build_ms": host_ms,
        "gpu_ms": gpu_ms,
        "serial_sum_ms": host_ms + gpu_ms,
        "pipelined_wall_ms": pipelined,
        "primitives": primitives,
        "host_build_p25_p75": [
            statistics.quantiles(build, n=4)[0],
            statistics.quantiles(build, n=4)[2],
        ],
        "gpu_p25_p75": [
            statistics.quantiles(gpu, n=4)[0],
            statistics.quantiles(gpu, n=4)[2],
        ],
        "overlap": "max" if pipelined < 0.5 * (host_ms + gpu_ms) + 0.5 * max(host_ms, gpu_ms) else "sum",
    }


def main() -> int:
    AB.configure_stack(MODEL)
    import mlx.core as mx
    from mlx_lm.generate import prefill_prompt_cache
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load
    from qwen4_fused_gdn_context_ladder import build_prompt, source_corpus

    mx.set_default_device(mx.gpu)
    started = time.perf_counter()
    model, tokenizer = load(str(MODEL))
    model.eval()
    mx.eval(model.parameters())
    print(f"[{time.perf_counter()-started:6.1f}s] loaded", flush=True)

    corpus = source_corpus(BENCH.parent)
    prompt = mx.array(build_prompt(tokenizer, corpus, CONTEXT), dtype=mx.uint32)
    cache = make_prompt_cache(model)
    prefill_prompt_cache(model, prompt[:-1], cache, prefill_step_size=2048)
    mx.eval(AB.cache_arrays(cache))
    print(f"[{time.perf_counter()-started:6.1f}s] prefilled {CONTEXT}", flush=True)

    token = mx.array([[7]], dtype=mx.uint32)
    tokens3 = mx.array([[7, 11, 13]], dtype=mx.uint32)
    report = {"model": str(MODEL), "mlx": mx.__version__, "context": CONTEXT,
              "trials": TRIALS, "pipeline_steps": PIPELINE_STEPS, "rows": []}

    rows = [
        ("dense_M1", lambda: (model(token, cache=cache),)),
        ("verify_M3_trunk", lambda: (model(tokens3, cache=cache),)),
    ]
    for name, step in rows:
        row = measure(mx, name, step, lambda: AB.cache_arrays(cache))
        report["rows"].append(row)
        print(f"[host] {row['kind']:16s} host {row['host_build_ms']:7.3f} ms  "
              f"gpu {row['gpu_ms']:7.3f} ms  sum {row['serial_sum_ms']:7.3f}  "
              f"pipelined {row['pipelined_wall_ms']:7.3f}  prims {row['primitives']:6d}  "
              f"-> {row['overlap']}", flush=True)

    # The MTP draft head, driven the way hybrid_speculative.py drives it.
    if getattr(model, "mtp", None) is not None:
        mtp_cache = model.make_mtp_cache()
        logits, hidden = model.mtp_backbone(token, cache=cache)
        mx.eval(logits, hidden, AB.cache_arrays(cache))
        draft_tokens = mx.array([[7]], dtype=mx.uint32)

        def draft():
            return model.mtp_step(hidden, draft_tokens, mtp_cache)

        row = measure(mx, "mtp_draft", draft, lambda: AB.cache_arrays(mtp_cache))
        report["rows"].append(row)
        print(f"[host] {row['kind']:16s} host {row['host_build_ms']:7.3f} ms  "
              f"gpu {row['gpu_ms']:7.3f} ms  sum {row['serial_sum_ms']:7.3f}  "
              f"pipelined {row['pipelined_wall_ms']:7.3f}  prims {row['primitives']:6d}  "
              f"-> {row['overlap']}", flush=True)

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[{time.perf_counter()-started:6.1f}s] HOST BUDGET DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
