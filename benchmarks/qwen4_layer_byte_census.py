#!/usr/bin/env python3
"""Per-layer-call byte census straight from the checkpoint header.

The denominator of the megakernel prize is "how long would ONE big read of
this layer's bytes take", so the bytes have to be right before any timing
means anything. Reading them out of the safetensors headers -- shapes and
storage sizes, no tensor data -- makes the accounting auditable and costs no
GPU: it is the same number whether or not a model is resident.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct


def header_sizes(root: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in sorted(glob.glob(os.path.join(root, "model*.safetensors"))):
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            sizes[name] = end - start
    return sizes


def census(root: str) -> dict:
    sizes = header_sizes(root)
    config = json.load(open(os.path.join(root, "config.json")))
    text = config.get("text_config", config)

    kinds = collections.defaultdict(set)
    for name in sizes:
        parts = name.split("layers.")
        if len(parts) > 1:
            index = int(parts[1].split(".")[0])
            kinds[index].add(name.split(f"layers.{index}.")[1].split(".")[0])
    gdn = sorted(i for i, s in kinds.items() if "linear_attn" in s)
    attn = sorted(i for i, s in kinds.items() if "self_attn" in s)

    def total(prefix: str) -> int:
        return sum(v for k, v in sizes.items() if k.startswith(prefix))

    layer = gdn[1]
    prefix = f"language_model.model.layers.{layer}."
    gdn_weights = total(prefix + "linear_attn")
    in_proj = sum(
        total(prefix + f"linear_attn.in_proj_{p}") for p in ("qkv", "z", "b", "a")
    )
    n_v = text["linear_num_value_heads"]
    n_k = text["linear_num_key_heads"]
    head_k = text["linear_key_head_dim"]
    head_v = text["linear_value_head_dim"]
    conv_dim = 2 * n_k * head_k + n_v * head_v
    conv_state = (text["linear_conv_kernel_dim"] - 1) * conv_dim * 2
    recurrent_state = n_v * head_k * head_v * 4
    gdn_total = gdn_weights + 2 * (conv_state + recurrent_state)

    moe_all = total(prefix + "mlp.")
    routed = sum(
        total(prefix + f"mlp.switch_mlp.{p}")
        for p in ("gate_proj", "up_proj", "down_proj")
    )
    dense = moe_all - routed
    top_k, experts = text["num_experts_per_tok"], text["num_experts"]
    moe_total = routed * top_k / experts + dense

    attn_prefix = f"language_model.model.layers.{attn[-1]}."
    attn_weights = total(attn_prefix + "self_attn")
    kv_per_pos = 2 * text["num_key_value_heads"] * text["head_dim"] * 2

    return {
        "model": root,
        "gdn_layers": len(gdn),
        "attention_layers": len(attn),
        "gdn": {
            "weights": gdn_weights,
            "in_proj_quartet": in_proj,
            "in_proj_share_of_layer": in_proj / gdn_total,
            "conv_state": conv_state,
            "recurrent_state": recurrent_state,
            "per_call_total": gdn_total,
        },
        "moe": {
            "routed_pool": routed,
            "gathered": int(routed * top_k / experts),
            "dense": dense,
            "per_call_total": int(moe_total),
            "top_k": top_k,
            "num_experts": experts,
        },
        "attention": {"weights": attn_weights, "kv_bytes_per_position": kv_per_pos},
        "per_token": {
            "gdn": len(gdn) * gdn_total,
            "moe": (len(gdn) + len(attn)) * moe_total,
            "attention_weights": len(attn) * attn_weights,
        },
        "fused_inproj_table_bytes": len(gdn) * in_proj,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = census(args.model)
    per_token = sum(report["per_token"].values())
    report["per_token"]["total"] = per_token
    report["per_token"]["floor_ms_at_600GBs"] = per_token / 600e9 * 1e3
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
