#!/usr/bin/env python3
"""Fail-closed CPU verifier for a Qwen4 phase-family ICB P1 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RECEIPT_SCHEMA = "mlx-uag.qwen4-phase-family-icb-p1-receipt.v1"
RESULT_SCHEMA = "mlx-uag.qwen4-phase-family-icb-p1-verification.v1"
MANIFEST_SCHEMA = "mlx-uag.qwen4-phase-family-icb-p1.v1"
MAX_MISMATCH_OFFSETS = 8
ORACLE_RTOL = 1e-4
ORACLE_ATOL = 1e-5
EXPECTED_OP_NAMES = (
    "OP_GROUP_RMSNORM",
    "OP_QMV",
    "OP_QMV",
    "OP_HC_MIX",
    "OP_QMV",
    "OP_QMV",
)
EXPECTED_INPUTS = (("RESID_A", 0, 10240),)
EXPECTED_OUTPUTS = (
    ("NORMED", 20480, 10240),
    ("HC_LR", 30720, 320),
    ("HC_W", 31040, 10240),
    ("MIXED", 41280, 2560),
    ("INJECT", 43840, 4),
    ("GDN_QKV", 46404, 10240),
)
EXPECTED_SCHEDULE_ROWS = (
    (4, 0, 0, 20480, 10240, 2560, 0, 2),
    (1, 1, 20480, 30720, 4, 3, 0, 2),
    (1, 2, 30720, 31040, 4, 2, 0, 2),
    (5, 0xFFFFFFFF, 31040, 41280, 4, 2560, 0, 2),
    (1, 3, 20480, 43840, 1, 4, 2, 1),
    (1, 4, 41280, 46404, 2, 0, 0, 0),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_manifest(artifact: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    manifest_path = artifact / "manifest.json"
    manifest = load_json(manifest_path)

    if manifest.get("schema") != MANIFEST_SCHEMA:
        failures.append(f"manifest schema must be {MANIFEST_SCHEMA}")

    stamp_path = artifact / "MANIFEST.sha256"
    fields = stamp_path.read_text().strip().split()
    if len(fields) != 2 or fields[1] != "manifest.json":
        failures.append("MANIFEST.sha256 has an invalid format")
    elif fields[0] != sha256_file(manifest_path):
        failures.append("manifest.json digest does not match MANIFEST.sha256")

    for name, metadata in manifest.get("artifacts", {}).items():
        path = artifact / metadata["file"]
        if not path.is_file():
            failures.append(f"artifact {name} is missing: {path.name}")
            continue
        if path.stat().st_size != metadata["bytes"]:
            failures.append(f"artifact {name} byte count differs from manifest")
        if sha256_file(path) != metadata["sha256"]:
            failures.append(f"artifact {name} digest differs from manifest")

    repo_root = Path(__file__).resolve().parents[2]
    for name, metadata in manifest.get("implementation", {}).items():
        relative = metadata.get("path")
        if not relative:
            continue
        path = repo_root / relative
        if not path.is_file():
            failures.append(f"implementation source {name} is missing: {relative}")
        elif sha256_file(path) != metadata["sha256"]:
            failures.append(f"implementation source {name} drifted: {relative}")

    pack_abi = manifest.get("pack_abi", {})
    if pack_abi.get("groups") != 1 or pack_abi.get("table_stride_words") != 12:
        failures.append("manifest must bind the one-group table-stride-12 pack ABI")
    schedule_abi = manifest.get("schedule_abi", {})
    rows = schedule_abi.get("rows", [])
    normalized_rows = tuple(
        tuple(row) for row in rows if isinstance(row, list)
    )
    if (
        schedule_abi.get("commands") != 6
        or schedule_abi.get("step_stride_words") != 8
        or schedule_abi.get("icb_barriers") != 5
        or tuple(schedule_abi.get("op_names", ())) != EXPECTED_OP_NAMES
        or normalized_rows != EXPECTED_SCHEDULE_ROWS
    ):
        failures.append("manifest does not contain the exact six-command P1 schedule ABI")
    inputs = tuple(
        (item.get("name"), item.get("offset"), item.get("floats"))
        for item in manifest.get("scratch_contract", {}).get("inputs", ())
        if isinstance(item, dict)
    )
    if inputs != EXPECTED_INPUTS:
        failures.append("manifest input range differs from the fixed P1 contract")
    outputs = tuple(
        (item.get("name"), item.get("offset"), item.get("floats"))
        for item in manifest.get("scratch_contract", {}).get("outputs", ())
        if isinstance(item, dict)
    )
    if outputs != EXPECTED_OUTPUTS:
        failures.append("manifest output ranges differ from the fixed P1 coverage contract")
    if manifest.get("scratch_contract", {}).get("full_scratch_floats") != 112698:
        failures.append("manifest full scratch size differs from the fixed P1 contract")
    comparison = manifest.get("scratch_contract", {}).get("comparison", {})
    if comparison != {
        "direct_vs_icb": "bitwise_uint32",
        "oracle": "abs(candidate-oracle) <= atol + rtol*abs(oracle)",
        "oracle_rtol": ORACLE_RTOL,
        "oracle_atol": ORACLE_ATOL,
        "record": ["max_abs", "mean_abs", "outside_tolerance"],
    }:
        failures.append("manifest accuracy policy differs from the fixed P1 contract")
    mechanism_gate = manifest.get("mechanism_gate", {})
    if mechanism_gate != {
        "required_direct_commands": 6,
        "required_icb_commands": 6,
        "required_icb_execute_calls": 1,
        "required_bitwise_outputs": 6,
        "fail_closed": True,
    }:
        failures.append("manifest mechanism gate differs from the fixed P1 contract")
    return manifest, failures


def bitwise_report(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    expected_bits = expected.view(np.uint32)
    observed_bits = observed.view(np.uint32)
    differing = np.flatnonzero(expected_bits != observed_bits)
    return {
        "mismatch_words": int(differing.size),
        "first_relative_word_offsets": [
            int(value) for value in differing[:MAX_MISMATCH_OFFSETS]
        ],
    }


def numeric_report(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    expected64 = expected.astype(np.float64)
    observed64 = observed.astype(np.float64)
    difference = np.abs(expected64 - observed64)
    exact = expected.view(np.uint32) == observed.view(np.uint32)
    finite = np.isfinite(expected64) & np.isfinite(observed64)
    allowed = ORACLE_ATOL + ORACLE_RTOL * np.abs(expected64)
    accepted = exact | (finite & (difference <= allowed))
    outside = np.flatnonzero(~accepted)
    finite_difference = difference[finite]
    return {
        "rtol": ORACLE_RTOL,
        "atol": ORACLE_ATOL,
        "max_abs": float(finite_difference.max()) if finite_difference.size else None,
        "mean_abs": float(finite_difference.mean()) if finite_difference.size else None,
        "outside_tolerance": int(outside.size),
        "first_relative_word_offsets": [
            int(value) for value in outside[:MAX_MISMATCH_OFFSETS]
        ],
    }


def load_scratch(path: Path, floats: int) -> np.ndarray:
    value = np.fromfile(path, dtype=np.float32)
    if value.size != floats:
        raise ValueError(f"{path} has {value.size} float32 words; expected {floats}")
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact.resolve()
    manifest, failures = verify_manifest(artifact)
    manifest_digest = sha256_file(artifact / "manifest.json")
    receipt = load_json(args.receipt)

    if receipt.get("schema") != RECEIPT_SCHEMA:
        failures.append(f"receipt schema must be {RECEIPT_SCHEMA}")
    if receipt.get("manifest_sha256") != manifest_digest:
        failures.append("receipt is not bound to this manifest digest")
    if receipt.get("implementation") != manifest.get("implementation"):
        failures.append("receipt implementation binding differs from manifest")
    receipt_accuracy = receipt.get("accuracy", {})
    if (
        receipt_accuracy.get("oracle_rtol") != ORACLE_RTOL
        or receipt_accuracy.get("oracle_atol") != ORACLE_ATOL
        or receipt_accuracy.get("oracle_comparison")
        != "abs(candidate-oracle) <= atol + rtol*abs(oracle)"
        or receipt_accuracy.get("direct_icb_comparison") != "bitwise_uint32"
    ):
        failures.append("receipt does not declare the fixed numeric/bitwise accuracy policy")

    gate = manifest["mechanism_gate"]
    expected_mechanism = {
        "direct_commands": gate["required_direct_commands"],
        "direct_barriers": gate["required_direct_commands"] - 1,
        "direct_submissions": 1,
        "direct_device_receipts": gate["required_direct_commands"],
        "icb_encoded_commands": gate["required_icb_commands"],
        "icb_barriers": gate["required_icb_commands"] - 1,
        "icb_execute_calls": gate["required_icb_execute_calls"],
        "icb_submissions": 1,
        "icb_device_receipts": gate["required_icb_commands"],
    }
    observed_mechanism = receipt.get("mechanism", {})
    mechanism_checks: dict[str, dict[str, Any]] = {}
    for name, expected in expected_mechanism.items():
        observed = observed_mechanism.get(name)
        passed = observed == expected
        mechanism_checks[name] = {
            "expected": expected,
            "observed": observed,
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"mechanism counter {name}: expected {expected}, observed {observed}"
            )

    full_floats = manifest["scratch_contract"]["full_scratch_floats"]
    expected = load_scratch(
        artifact / manifest["artifacts"]["scratch_expected"]["file"], full_floats
    )
    direct = load_scratch(args.direct_scratch, full_floats)
    icb = load_scratch(args.icb_scratch, full_floats)

    output_checks: list[dict[str, Any]] = []
    for output in manifest["scratch_contract"]["outputs"]:
        offset = int(output["offset"])
        end = offset + int(output["floats"])
        oracle_direct = numeric_report(expected[offset:end], direct[offset:end])
        oracle_icb = numeric_report(expected[offset:end], icb[offset:end])
        direct_icb = bitwise_report(direct[offset:end], icb[offset:end])
        passed = not (
            oracle_direct["outside_tolerance"]
            or oracle_icb["outside_tolerance"]
            or direct_icb["mismatch_words"]
        )
        output_checks.append(
            {
                **output,
                "oracle_vs_direct": oracle_direct,
                "oracle_vs_icb": oracle_icb,
                "direct_vs_icb": direct_icb,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"output acceptance failed: {output['name']}")

    required_outputs = gate["required_bitwise_outputs"]
    if len(output_checks) != required_outputs:
        failures.append(
            f"coverage count: expected {required_outputs} outputs, checked {len(output_checks)}"
        )

    return {
        "schema": RESULT_SCHEMA,
        "passed": not failures,
        "artifact": str(artifact),
        "receipt": str(args.receipt.resolve()),
        "manifest_sha256": manifest_digest,
        "receipt_manifest_bound": receipt.get("manifest_sha256") == manifest_digest,
        "implementation_digest_bound": (
            receipt.get("implementation") == manifest.get("implementation")
        ),
        "mechanism": mechanism_checks,
        "coverage": {
            "required_outputs": required_outputs,
            "checked_outputs": len(output_checks),
            "outputs": output_checks,
        },
        "accuracy_policy": {
            "direct_vs_icb": "bitwise_uint32",
            "oracle": "numpy isclose: abs(a-b) <= atol + rtol*abs(oracle)",
            "oracle_rtol": ORACLE_RTOL,
            "oracle_atol": ORACLE_ATOL,
        },
        "timing": receipt.get("timing"),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--direct-scratch", type=Path, required=True)
    parser.add_argument("--icb-scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify(args)
    except Exception as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "passed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
