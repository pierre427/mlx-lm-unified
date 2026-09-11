#!/usr/bin/env python3
"""Prepare a bounded real-Qwen4 phase-family artifact for an ICB P1 gate.

The default ``inspect`` mode reads checkpoint metadata only. ``prepare`` uses
the current pack ABI and MirrorExecutor, so it is GPU work and requires the
same explicit acknowledgement as the native P0 benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


DEFAULT_MODEL = Path(
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
SCHEMA = "mlx-uag.qwen4-phase-family-icb-p1.v1"
GPU_OWNER = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(model: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    return config, index["weight_map"]


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config", config)


def choose_linear_layer(config: dict[str, Any], requested: int | None) -> int:
    layer_types = text_config(config).get("layer_types")
    if not layer_types:
        raise ValueError("checkpoint config has no layer_types")
    if requested is not None:
        if requested < 0 or requested >= len(layer_types):
            raise ValueError(f"layer {requested} outside 0..{len(layer_types) - 1}")
        if layer_types[requested] != "linear_attention":
            raise ValueError(f"layer {requested} is {layer_types[requested]!r}, not linear_attention")
        return requested
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            return index
    raise ValueError("checkpoint has no linear_attention layer")


def required_keys(layer: int) -> list[str]:
    base = f"language_model.model.layers.{layer}"
    hyper = f"{base}.attn_hyper_connection"
    return [
        f"{hyper}.hc_norm.weight",
        f"{hyper}.input_mix_weight_down",
        f"{hyper}.input_mix_weight_up",
        f"{hyper}.block_inject_weight",
        f"{base}.linear_attn.in_proj_qkv",
    ]


def key_available(weight_map: dict[str, str], key: str) -> bool:
    return key in weight_map or f"{key}.weight" in weight_map


def inspect(model: Path, layer: int | None) -> dict[str, Any]:
    config, weight_map = load_metadata(model)
    chosen = choose_linear_layer(config, layer)
    keys = required_keys(chosen)
    availability = {key: key_available(weight_map, key) for key in keys}
    return {
        "schema": SCHEMA,
        "mode": "metadata-only",
        "model": str(model),
        "layer": chosen,
        "layer_type": "linear_attention",
        "keys": availability,
        "ready": all(availability.values()),
        "gpu_touched": False,
    }


def write_array(path: Path, value: Any, dtype: np.dtype) -> dict[str, Any]:
    array = np.asarray(value, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)
    return {
        "file": path.name,
        "dtype": np.dtype(dtype).name,
        "shape": list(contiguous.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_blob(path: Path, data: bytes, dtype: str) -> dict[str, Any]:
    path.write_bytes(data)
    return {
        "file": path.name,
        "dtype": dtype,
        "shape": [len(data)],
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def emit_source(path: Path) -> dict[str, Any]:
    from mlx_lm.models import qwen4_megakernel as MK
    from mlx_lm.models import qwen4_megakernel_body as MB

    from qwen4_phase_family_msl import build_source

    source, provenance = build_source(MK, MB)
    metadata = write_blob(path, source.encode(), "metal-source-utf8")
    return {
        "schema": SCHEMA,
        "mode": "source-only",
        "gpu_touched": False,
        "source": metadata,
        "provenance": provenance,
    }


def prepare(model: Path, layer: int | None, out_dir: Path) -> dict[str, Any]:
    if os.environ.get("MLX_UAG_GPU_LEASE_ACK") != "1":
        raise RuntimeError("prepare mode requires MLX_UAG_GPU_LEASE_ACK=1")
    lease_agent = os.environ.get("MLX_UAG_GPU_LEASE_AGENT")
    if not lease_agent or not GPU_OWNER.is_file():
        raise RuntimeError("prepare mode requires the active filesystem GPU lease")
    owner = json.loads(GPU_OWNER.read_text())
    if (
        not owner.get("claimed")
        or owner.get("agent_id") != lease_agent
        or int(owner.get("lease_expires_at", 0)) <= int(time.time())
    ):
        raise RuntimeError("acknowledged agent does not own a live GPU lease")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {out_dir}")

    import mlx.core as mx

    from mlx_lm.models import qwen4_megakernel as MK
    from mlx_lm.models import qwen4_megakernel_body as MB
    from mlx_lm.models import qwen4_megakernel_pack as MP
    from mlx_lm.models import qwen4_megakernel_schedule as MS

    from qwen4_phase_family_msl import build_source

    config, _ = load_metadata(model)
    chosen = choose_linear_layer(config, layer)
    keys = required_keys(chosen)
    source = MP.SafetensorsSource(str(model))
    missing = [key for key in keys if not source.has(key)]
    if missing:
        raise RuntimeError(f"required checkpoint tensors are missing: {missing}")

    pack = MP.build_pack(source, [(key, "main") for key in keys], validate=True)
    if len(pack.buffers) != 1:
        raise RuntimeError(
            f"P1 native loader is intentionally one-group; pack produced {len(pack.buffers)} groups"
        )
    if any(pack.entries[key].bits not in (0, 4) for key in keys):
        specs = {key: pack.entries[key].bits for key in keys}
        raise RuntimeError(f"P1 only admits dense gains plus Q4 projections: {specs}")

    base = f"language_model.model.layers.{chosen}"
    plan = MS.LayerPlan(index=chosen, is_linear=True, prefix=base)
    schedule = MK.Schedule()
    # The non-fused spelling is defined by Schedule/MirrorExecutor and exposes
    # each affine/glue boundary as its own ICB command. The current persistent
    # body uses fused HC_DOWN/HC_UP and no longer has a standalone opcode-5
    # branch; the native P1 wrapper implements that mirror-defined boundary.
    MS._hyper_block(
        schedule,
        pack,
        plan,
        "attn_hyper_connection",
        MK.SCRATCH["RESID_A"],
        fused=False,
    )
    schedule.add(
        MK.Step(
            op=MK.OP_QMV,
            entry=pack.entries[f"{base}.linear_attn.in_proj_qkv"].index,
            src=MK.SCRATCH["MIXED"],
            dst=MK.SCRATCH["GDN_QKV"],
            arg0=MK.PHASE_ROWS["gdn_in_proj"],
            barrier=MK.BAR_NONE,
        )
    )
    if len(schedule.steps) != 6:
        raise RuntimeError(f"expected six P1 commands, got {len(schedule.steps)}")

    rng = np.random.default_rng(20260910)
    residual = (rng.standard_normal(MK.HC_HIDDEN) * 0.5).astype(np.float32)
    scratch_input = np.zeros(MK.SCRATCH_FLOATS, dtype=np.float32)
    start = MK.SCRATCH["RESID_A"]
    scratch_input[start : start + MK.HC_HIDDEN] = residual

    mirror = MS.MirrorExecutor(pack)
    mirror.scratch = mx.array(scratch_input)
    mirror.run(schedule)
    mx.eval(mirror.scratch, pack.table, schedule.to_array(), pack.buffers)
    scratch_expected = np.asarray(mirror.scratch, dtype=np.float32)

    out_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    artifacts["weight_group_0"] = write_array(
        out_dir / "weight-group-0.u32", pack.buffers[0], np.uint32
    )
    artifacts["table"] = write_array(out_dir / "table.u32", pack.table, np.uint32)
    artifacts["schedule"] = write_array(
        out_dir / "schedule.u32", schedule.to_array(), np.uint32
    )
    artifacts["scratch_input"] = write_array(
        out_dir / "scratch-input.f32", scratch_input, np.float32
    )
    artifacts["scratch_expected"] = write_array(
        out_dir / "scratch-expected.f32", scratch_expected, np.float32
    )
    phase_source, phase_source_provenance = build_source(MK, MB)
    artifacts["phase_source"] = write_blob(
        out_dir / "qwen4-phase-family-p1.metal",
        phase_source.encode(),
        "metal-source-utf8",
    )

    source_root = Path(MB.__file__).resolve().parents[2]
    implementation_files = {
        "megakernel": Path(MK.__file__).resolve(),
        "body": Path(MB.__file__).resolve(),
        "pack": Path(MP.__file__).resolve(),
        "schedule": Path(MS.__file__).resolve(),
        "phase_source_generator": Path(__file__).with_name(
            "qwen4_phase_family_msl.py"
        ).resolve(),
        "native_runner": Path(__file__).with_name(
            "qwen4_phase_family_icb.mm"
        ).resolve(),
        "artifact_preparer": Path(__file__).resolve(),
        "receipt_verifier": Path(__file__).with_name(
            "verify_qwen4_phase_family_p1.py"
        ).resolve(),
    }
    implementation = {
        name: {
            "path": str(path.relative_to(source_root)),
            "sha256": sha256_file(path),
        }
        for name, path in implementation_files.items()
    }
    implementation["body_helpers"] = {
        "sha256": sha256_bytes(MB.BODY_HELPERS.encode()),
        "bytes": len(MB.BODY_HELPERS.encode()),
    }
    implementation["kernel_header"] = {
        "sha256": sha256_bytes(MK.kernel_header().encode()),
        "bytes": len(MK.kernel_header().encode()),
    }
    implementation["phase_source"] = {
        "sha256": sha256_bytes(phase_source.encode()),
        "bytes": len(phase_source.encode()),
        "provenance": phase_source_provenance,
    }

    scratch_contract = {
        "inputs": [
            {"name": "RESID_A", "offset": start, "floats": MK.HC_HIDDEN}
        ],
        "outputs": [
            {"name": "NORMED", "offset": MK.SCRATCH["NORMED"], "floats": MK.HC_HIDDEN},
            {"name": "HC_LR", "offset": MK.SCRATCH["HC_LR"], "floats": MK.HC_LOWRANK},
            {"name": "HC_W", "offset": MK.SCRATCH["HC_W"], "floats": MK.HC_HIDDEN},
            {"name": "MIXED", "offset": MK.SCRATCH["MIXED"], "floats": MK.HIDDEN},
            {"name": "INJECT", "offset": MK.SCRATCH["INJECT"], "floats": MK.HC_COUNT},
            {"name": "GDN_QKV", "offset": MK.SCRATCH["GDN_QKV"], "floats": MK.CONV_DIM},
        ],
        "full_scratch_floats": MK.SCRATCH_FLOATS,
        "comparison": {
            "direct_vs_icb": "bitwise_uint32",
            "oracle": "abs(candidate-oracle) <= atol + rtol*abs(oracle)",
            "oracle_rtol": 1e-4,
            "oracle_atol": 1e-5,
            "record": ["max_abs", "mean_abs", "outside_tolerance"],
        },
    }

    manifest = {
        "schema": SCHEMA,
        "model": str(model),
        "layer": chosen,
        "layer_type": "linear_attention",
        "oracle": "qwen4_megakernel_schedule.MirrorExecutor",
        "oracle_arithmetic": "fp32 phase-definition path",
        "candidate": "standalone native Metal direct and reusable-ICB lanes",
        "native_geometry": {
            "threadgroups": 40,
            "threads_per_threadgroup": 512,
            "qmv_threadgroup_bytes": MK.HIDDEN * 4,
            "norm_threadgroup_bytes": (16 + MK.HC_COUNT) * 4,
        },
        "pack_abi": {
            "table_stride_words": MP.TABLE_STRIDE,
            "table_fields": list(MP.TABLE_FIELDS),
            "sb_layout": pack.sb_layout,
            "groups": len(pack.buffers),
            "entries": [
                {"key": key, "row": pack.entries[key].row()}
                for key in pack.order
            ],
        },
        "schedule_abi": {
            "step_stride_words": MK.STEP_STRIDE,
            "commands": len(schedule.steps),
            "rows": [list(step.row()) for step in schedule.steps],
            "op_names": [MK.OP_NAMES.get(step.op, str(step.op)) for step in schedule.steps],
            "original_device_barriers": schedule.device_barriers,
            "icb_barriers": len(schedule.steps) - 1,
        },
        "scratch_contract": scratch_contract,
        "implementation": implementation,
        "artifacts": artifacts,
        "mechanism_gate": {
            "required_direct_commands": len(schedule.steps),
            "required_icb_commands": len(schedule.steps),
            "required_icb_execute_calls": 1,
            "required_bitwise_outputs": len(scratch_contract["outputs"]),
            "fail_closed": True,
        },
        "pack_summary": pack.summary(),
        "gpu_touched": True,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int)
    parser.add_argument(
        "--mode", choices=("inspect", "emit-source", "prepare"), default="inspect"
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "inspect":
        result = inspect(args.model, args.layer)
    elif args.mode == "emit-source":
        if args.source_out is None or not args.source_out.is_absolute():
            raise ValueError("emit-source mode requires an absolute --source-out")
        result = emit_source(args.source_out)
    else:
        if args.out_dir is None or not args.out_dir.is_absolute():
            raise ValueError("prepare mode requires an absolute --out-dir")
        result = prepare(args.model, args.layer, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
