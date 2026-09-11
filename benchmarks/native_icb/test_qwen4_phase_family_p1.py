#!/usr/bin/env python3
"""CPU-only contract tests for the native Qwen4 phase-family P1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import verify_qwen4_phase_family_p1 as verifier


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_manifest() -> dict:
    return {
        "schema": verifier.MANIFEST_SCHEMA,
        "pack_abi": {"groups": 1, "table_stride_words": 12},
        "schedule_abi": {
            "commands": 6,
            "step_stride_words": 8,
            "icb_barriers": 5,
            "op_names": list(verifier.EXPECTED_OP_NAMES),
            "rows": [list(row) for row in verifier.EXPECTED_SCHEDULE_ROWS],
        },
        "scratch_contract": {
            "inputs": [
                {"name": name, "offset": offset, "floats": floats}
                for name, offset, floats in verifier.EXPECTED_INPUTS
            ],
            "outputs": [
                {"name": name, "offset": offset, "floats": floats}
                for name, offset, floats in verifier.EXPECTED_OUTPUTS
            ],
            "full_scratch_floats": 112698,
            "comparison": {
                "direct_vs_icb": "bitwise_uint32",
                "oracle": "abs(candidate-oracle) <= atol + rtol*abs(oracle)",
                "oracle_rtol": verifier.ORACLE_RTOL,
                "oracle_atol": verifier.ORACLE_ATOL,
                "record": ["max_abs", "mean_abs", "outside_tolerance"],
            },
        },
        "implementation": {},
        "artifacts": {},
        "mechanism_gate": {
            "required_direct_commands": 6,
            "required_icb_commands": 6,
            "required_icb_execute_calls": 1,
            "required_bitwise_outputs": 6,
            "fail_closed": True,
        },
    }


def write_manifest(directory: Path, manifest: dict) -> None:
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (directory / "MANIFEST.sha256").write_text(
        f"{digest(path)}  manifest.json\n"
    )


class TestManifestContract(unittest.TestCase):
    def verify(self, mutate=None) -> list[str]:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            manifest = base_manifest()
            if mutate is not None:
                mutate(manifest)
            write_manifest(directory, manifest)
            _, failures = verifier.verify_manifest(directory)
            return failures

    def test_exact_contract_passes(self):
        self.assertEqual(self.verify(), [])

    def test_schema_drift_fails(self):
        failures = self.verify(lambda value: value.update(schema="wrong"))
        self.assertTrue(any("schema" in item for item in failures), failures)

    def test_schedule_row_drift_fails(self):
        def mutate(value):
            value["schedule_abi"]["rows"][2][4] = 2

        failures = self.verify(mutate)
        self.assertTrue(any("six-command" in item for item in failures), failures)

    def test_output_range_drift_fails(self):
        def mutate(value):
            value["scratch_contract"]["outputs"][5]["offset"] += 1

        failures = self.verify(mutate)
        self.assertTrue(any("output ranges" in item for item in failures), failures)


class TestReceiptBinding(unittest.TestCase):
    def test_manifest_digest_binding_passes_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            zeros = np.zeros(112698, dtype=np.float32)
            expected = directory / "scratch-expected.f32"
            direct = directory / "direct.f32"
            icb = directory / "icb.f32"
            zeros.tofile(expected)
            zeros.tofile(direct)
            zeros.tofile(icb)
            manifest = base_manifest()
            manifest["artifacts"]["scratch_expected"] = {
                "file": expected.name,
                "bytes": expected.stat().st_size,
                "sha256": digest(expected),
            }
            write_manifest(directory, manifest)
            receipt = directory / "receipt.json"
            receipt_data = {
                "schema": verifier.RECEIPT_SCHEMA,
                "manifest_sha256": digest(directory / "manifest.json"),
                "implementation": {},
                "accuracy": {
                    "oracle_rtol": verifier.ORACLE_RTOL,
                    "oracle_atol": verifier.ORACLE_ATOL,
                    "oracle_comparison": (
                        "abs(candidate-oracle) <= atol + rtol*abs(oracle)"
                    ),
                    "direct_icb_comparison": "bitwise_uint32",
                },
                "mechanism": {
                    "direct_commands": 6,
                    "direct_barriers": 5,
                    "direct_submissions": 1,
                    "direct_device_receipts": 6,
                    "icb_encoded_commands": 6,
                    "icb_barriers": 5,
                    "icb_execute_calls": 1,
                    "icb_submissions": 1,
                    "icb_device_receipts": 6,
                },
            }
            receipt.write_text(json.dumps(receipt_data))
            arguments = argparse.Namespace(
                artifact=directory,
                receipt=receipt,
                direct_scratch=direct,
                icb_scratch=icb,
            )
            accepted = verifier.verify(arguments)
            self.assertTrue(accepted["passed"], accepted)
            self.assertTrue(accepted["receipt_manifest_bound"])

            receipt_data["accuracy"]["oracle_comparison"] = "unchecked"
            receipt.write_text(json.dumps(receipt_data))
            policy_rejected = verifier.verify(arguments)
            self.assertFalse(policy_rejected["passed"], policy_rejected)
            self.assertTrue(
                any(
                    "accuracy policy" in item
                    for item in policy_rejected["failures"]
                ),
                policy_rejected,
            )
            receipt_data["accuracy"]["oracle_comparison"] = (
                "abs(candidate-oracle) <= atol + rtol*abs(oracle)"
            )
            receipt.write_text(json.dumps(receipt_data))

            divergent = zeros.copy()
            divergent[verifier.EXPECTED_OUTPUTS[0][1]] = 0.1
            divergent.tofile(direct)
            divergent.tofile(icb)
            rejected = verifier.verify(arguments)
            self.assertFalse(rejected["passed"], rejected)
            first = rejected["coverage"]["outputs"][0]
            self.assertEqual(first["direct_vs_icb"]["mismatch_words"], 0)
            self.assertEqual(first["oracle_vs_direct"]["outside_tolerance"], 1)
            self.assertTrue(
                any("output acceptance failed" in item for item in rejected["failures"])
            )

            zeros.tofile(direct)
            near = zeros.copy()
            near[verifier.EXPECTED_OUTPUTS[0][1]] = verifier.ORACLE_ATOL * 0.5
            near.tofile(icb)
            bitwise_rejected = verifier.verify(arguments)
            self.assertFalse(bitwise_rejected["passed"], bitwise_rejected)
            first = bitwise_rejected["coverage"]["outputs"][0]
            self.assertEqual(first["oracle_vs_icb"]["outside_tolerance"], 0)
            self.assertEqual(first["direct_vs_icb"]["mismatch_words"], 1)

            zeros.tofile(icb)

            receipt_data["manifest_sha256"] = "0" * 64
            receipt.write_text(json.dumps(receipt_data))
            result = verifier.verify(
                arguments
            )
            self.assertFalse(result["passed"])
            self.assertFalse(result["receipt_manifest_bound"])
            self.assertTrue(
                any("not bound" in item for item in result["failures"]), result
            )


class TestNumericOraclePolicy(unittest.TestCase):
    def test_material_oracle_divergence_is_rejected(self):
        expected = np.zeros(4, dtype=np.float32)
        observed = expected.copy()
        observed[2] = 0.1

        report = verifier.numeric_report(expected, observed)

        self.assertEqual(report["outside_tolerance"], 1)
        self.assertEqual(report["first_relative_word_offsets"], [2])
        self.assertGreater(report["max_abs"], verifier.ORACLE_ATOL)


if __name__ == "__main__":
    unittest.main()
