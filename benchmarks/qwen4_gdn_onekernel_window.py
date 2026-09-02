#!/usr/bin/env python3
"""One GPU window: the megakernel prize, then the fused in-proj A/B.

One model load serves both measurements. The prize needs the real layers and
the A/B needs the real decode loop, and reloading 67 GiB between them would
change nothing except the risk of doing it twice.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import qwen4_gdn_inproj_ab as AB
import qwen4_layer_onekernel_prize as PRIZE

RESULTS = Path("/Users/pierrelamy/Desktop/mlx-uag/results")
MODEL = Path("/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP")


class PrizeArgs:
    model = str(MODEL)
    widths = [1, 3]
    reps = 32
    trials = 5
    attention_context = 16384


def main() -> int:
    AB.configure_stack(MODEL)
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
    from qwen4_fused_gdn_context_ladder import build_prompt, source_corpus

    mx.set_default_device(mx.gpu)
    started = time.perf_counter()
    AB.append_jsonl(AB.DEFAULT_OUTPUT, {"event": "window_start", "created_at_utc": AB.utc_now()})

    model, tokenizer = load(str(MODEL))
    model.eval()
    mx.eval(model.parameters())
    print(f"[{time.perf_counter()-started:6.1f}s] loaded, active "
          f"{mx.get_active_memory()/(1<<30):.1f} GiB", flush=True)

    # --- 1. the prize, with the lever OFF so the numerator is today's layer ---
    set_qwen4_gdn_fused_inproj(model, False)
    prize = PRIZE.run(PrizeArgs(), model=model)
    (RESULTS / "qwen4-gdn-onekernel-20260902-prize.json").write_text(
        json.dumps(prize, indent=2) + "\n"
    )
    for width, entries in prize["rows"].items():
        for kind, row in entries.items():
            if "prize" in row:
                print(f"[prize] M={width} {kind:5s} {row['bytes']['total']/1e6:7.2f} MB "
                      f"real {row['real_us']:8.2f} us  floor {row['floor_us']:8.2f} us  "
                      f"prize {row['prize']:5.2f}x  ({row['real_achieved_gbs']:5.0f} vs "
                      f"{row['floor_achieved_gbs']:5.0f} GB/s)", flush=True)
    print(f"[{time.perf_counter()-started:6.1f}s] prize done", flush=True)

    # --- 2. arm the lever, prove exactness on the real weights ---
    before = mx.get_active_memory()
    gdn_layers = set_qwen4_gdn_fused_inproj(model, True)
    stats = qwen4_gdn_fused_inproj_stats(model)
    for _, module in model.named_modules():
        entry = getattr(module, "_gdn_inproj_fused_cache", None)
        if entry is not None and entry[1] is not None:
            mx.eval([part for part in entry[1][0][:3] if part is not None])
    table_bytes = mx.get_active_memory() - before
    probe = probe_qwen4_gdn_fused_inproj(model)
    print(f"[gate] layers={gdn_layers} eligible={stats['eligible']} "
          f"table={table_bytes/1e6:.0f} MB probe_checked={probe['checked']} "
          f"mismatches={probe['mismatches']}", flush=True)
    AB.append_jsonl(AB.DEFAULT_OUTPUT, {
        "event": "gate", "created_at_utc": AB.utc_now(), "gdn_layers": gdn_layers,
        "inproj_stats": stats, "table_bytes": table_bytes, "gpu_parity_probe": probe,
    })
    if stats["eligible"] != gdn_layers or probe["mismatches"]:
        raise RuntimeError(f"gate failed: {stats} {probe['mismatches']}")

    # Collect the layers ONCE. ``set_qwen4_gdn_fused_inproj`` walks
    # ``named_modules()``, and this model carries 512 experts per MoE block, so
    # a full walk between every arm switch would cost more host time than the
    # difference being measured -- 3,072 walks over the A/B.
    from mlx_lm.models.qwen4_exp import GatedDeltaNet as _GDN

    gdn_modules = [m for _, m in model.named_modules() if isinstance(m, _GDN)]
    assert len(gdn_modules) == gdn_layers

    def set_arm(arm: str) -> None:
        armed = arm == "on"
        for layer in gdn_modules:
            layer.gdn_fused_inproj = armed

    def cheap_stats(_model, *, reset: bool = False) -> dict:
        """``calls`` receipt over the pre-collected layers, no module walk."""
        total = sum(layer.gdn_fused_inproj_calls for layer in gdn_modules)
        if reset:
            for layer in gdn_modules:
                layer.gdn_fused_inproj_calls = 0
        return {"calls": total}

    # --- 3. the A/B ---
    corpus = source_corpus(BENCH.parent)
    for context in (1024, 16384):
        prompt = build_prompt(tokenizer, corpus, context)
        prompt_array = mx.array(prompt, dtype=mx.uint32)
        set_arm("off")
        cache = make_prompt_cache(model)
        prefill_at = time.perf_counter()
        AB.prefill = None
        prefill_prompt_cache(model, prompt_array[:-1], cache, prefill_step_size=2048)
        mx.eval(AB.cache_arrays(cache))
        prefill_s = time.perf_counter() - prefill_at
        first = model(prompt_array[-1][None, None], cache=cache)
        first_token = mx.argmax(first[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(first_token, AB.cache_arrays(cache))

        for mode in ("stock", "fused"):
            set_qwen4_fused_gdn_mode(model, mode)
            for rep in range(5):
                row = AB.run_trajectory(
                    model=model, base_cache=cache, first_token=first_token,
                    steps=128, rep=rep, mx=mx, qwen4_cache_type=Qwen4ArraysCache,
                    set_arm=set_arm, inproj_stats=cheap_stats,
                    gdn_layers=gdn_layers,
                )
                row.update({"event": "trajectory", "created_at_utc": AB.utc_now(),
                            "context": context, "decode_mode": mode, "rep": rep,
                            "prefill_s": prefill_s})
                AB.append_jsonl(AB.DEFAULT_OUTPUT, row)
                print(f"[ab] ctx={context} gdn={mode} rep={rep} pass={row['passed']} "
                      f"calls={row['fused_inproj_calls']}/{row['expected_fused_inproj_calls']} "
                      f"off={row['median_seconds_per_token']['off']*1e3:.3f} ms "
                      f"on={row['median_seconds_per_token']['on']*1e3:.3f} ms "
                      f"median%={row['median_speedup_percent']:+.2f} "
                      f"agg%={row['aggregate_speedup_percent']:+.2f} "
                      f"d={row['token_sha256'][:10]}", flush=True)

        set_qwen4_fused_gdn_mode(model, "stock")
        slab = AB.slab_timing(model=model, base_cache=cache, mx=mx, set_arm=set_arm,
                              width=3, reps=8, trials=5)
        slab.update({"event": "slab", "created_at_utc": AB.utc_now(), "context": context})
        AB.append_jsonl(AB.DEFAULT_OUTPUT, slab)
        print(f"[slab] ctx={context} width3 off={slab['median_seconds']['off']*1e3:.3f} ms "
              f"on={slab['median_seconds']['on']*1e3:.3f} ms "
              f"%={slab['speedup_percent']:+.2f}", flush=True)
        del cache
        mx.clear_cache()
        print(f"[{time.perf_counter()-started:6.1f}s] context {context} done", flush=True)

    AB.append_jsonl(AB.DEFAULT_OUTPUT, {"event": "window_done", "created_at_utc": AB.utc_now(),
                                        "wall_s": time.perf_counter() - started})
    print(f"[{time.perf_counter()-started:6.1f}s] WINDOW DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
