import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mlx_lm.models import qwen4_qsa_indexed
from mlx_lm.qsa_topology_receipt import (
    QSA_TOPOLOGY_RECEIPT_SCHEMA,
    SCHEMA_ID,
    analyze_qsa_topology_receipts,
    build_qsa_topology_receipt,
    capture_compact_topology_diagnostic,
    write_qsa_topology_diagnostic,
)


class TestQSATopologyReceipt(unittest.TestCase):
    def test_exact_and_partial_cohorts_are_host_visible(self):
        ids = np.array(
            [
                [[0, 2, 5, 9], [0, 2, 6, 9]],
                [[0, 2, 5, 8], [0, 2, 6, 8]],
                [[0, 2, 5, 7], [0, 3, 6, 7]],
            ],
            dtype=np.uint32,
        )
        # Page 9 is a private-delta page and is excluded from the base sets.
        receipt = build_qsa_topology_receipt(
            ids,
            np.full((3, 2), 4, dtype=np.int32),
            base_page_count=9,
            page_size_tokens=4,
        )

        self.assertEqual(receipt["schema"], SCHEMA_ID)
        self.assertFalse(receipt["capture"]["device_readback"])
        self.assertFalse(receipt["capture"]["runtime_hot_path"])
        self.assertEqual(receipt["ordered_base_page_sets"][0], [[0, 2, 5], [0, 2, 6]])
        first = receipt["query_cohorts"][0]
        self.assertEqual(first["largest_exact_set_cohort"], 1)
        self.assertEqual(first["union_inflation"]["union_pages"], 5)
        self.assertAlmostEqual(first["union_inflation"]["versus_mean_member"], 15 / 11)
        activity = {
            row["page_id"]: row["active_queries"]
            for row in first["active_queries_per_page"]
        }
        self.assertEqual(activity[0], 3)
        self.assertEqual(activity[2], 3)
        self.assertEqual(first["same_page_reuse"]["reused_memberships"], 6)
        self.assertEqual(receipt["exact_set_cohort_sizes"], [[1, 1]] * 3)

    def test_exact_set_cohort_size_and_pairwise_jaccard(self):
        ids = np.array(
            [
                [[0, 1, 2]],
                [[0, 1, 2]],
                [[0, 1, 3]],
            ],
            dtype=np.uint32,
        )
        receipt = build_qsa_topology_receipt(
            ids,
            np.full((3, 1), 3, dtype=np.int32),
            base_page_count=4,
            page_size_tokens=4,
        )
        cohort = receipt["query_cohorts"][0]
        self.assertEqual(receipt["exact_set_cohort_sizes"], [[2], [2], [1]])
        self.assertEqual(cohort["largest_exact_set_cohort"], 2)
        self.assertEqual(
            [pair["jaccard"] for pair in cohort["pairwise_jaccard"]["pairs"]],
            [1.0, 0.5, 0.5],
        )
        self.assertAlmostEqual(cohort["same_page_reuse"]["reuse_fraction"], 5 / 9)

    def test_adjacent_query_reuse_is_separate_from_batch_reuse(self):
        ids = np.array([[[0, 1, 2], [0, 1, 3], [0, 1, 3]]], np.uint32)
        receipt = build_qsa_topology_receipt(
            ids,
            np.full((1, 3), 3, np.int32),
            base_page_count=4,
            page_size_tokens=4,
        )
        adjacent = receipt["adjacent_query_reuse"]
        self.assertEqual([pair["jaccard"] for pair in adjacent["pairs"]], [0.5, 1.0])
        self.assertFalse(receipt["all_queries"]["simultaneous_reuse_claim"])

    def test_counts_filter_compact_suffix_and_validate_order(self):
        ids = np.array([[[1, 3, 0, 0]]], dtype=np.uint32)
        receipt = build_qsa_topology_receipt(
            ids,
            [[2]],
            base_page_count=4,
            page_size_tokens=4,
        )
        self.assertEqual(receipt["ordered_base_page_sets"], [[[1, 3]]])
        with self.assertRaisesRegex(ValueError, "unique and ascending"):
            build_qsa_topology_receipt(
                [[[2, 1]]],
                [[2]],
                base_page_count=4,
                page_size_tokens=4,
            )

    def test_explicit_compact_adapter_marks_the_device_readback(self):
        compact = SimpleNamespace(
            block_ids=np.array([[[0, 2, 4]]], dtype=np.uint32),
            block_counts=np.array([[3]], dtype=np.int32),
            block_size=4,
        )
        receipt = capture_compact_topology_diagnostic(compact, base_tokens=16)
        self.assertEqual(
            receipt["capture"],
            {
                "mode": "diagnostic_device_readback",
                "device_readback": True,
                "runtime_hot_path": False,
            },
        )
        self.assertEqual(receipt["ordered_base_page_sets"], [[[0, 2]]])

    def test_receipt_keys_match_the_published_schema(self):
        receipt = build_qsa_topology_receipt(
            [[[0]]], [[1]], base_page_count=1, page_size_tokens=4
        )
        required = set(QSA_TOPOLOGY_RECEIPT_SCHEMA["required"])
        self.assertTrue(required.issubset(receipt))

    def test_production_segmented_cache_does_not_import_the_syncing_adapter(self):
        source = (
            Path(__file__).parents[1] / "mlx_lm" / "segmented_batch_cache.py"
        ).read_text()
        self.assertNotIn("qsa_topology_receipt", source)
        self.assertNotIn("capture_compact_topology_diagnostic", source)

    def test_analyzer_prefers_exact_set_cohorts(self):
        receipt = build_qsa_topology_receipt(
            [[[0, 1]], [[0, 1]], [[0, 2]]],
            [[2], [2], [2]],
            base_page_count=3,
            page_size_tokens=4,
        )
        analysis = analyze_qsa_topology_receipts([receipt])
        self.assertEqual(analysis["recommendation"], "exact_set")
        self.assertTrue(analysis["exact_set"]["gate"]["passes"])
        self.assertAlmostEqual(
            analysis["exact_set"]["predicted_base_read_reduction_fraction_upper_bound"],
            2 / 6,
        )

    def test_analyzer_uses_ordered_union_for_partial_overlap(self):
        receipt = build_qsa_topology_receipt(
            [[[0, 1, 2]], [[0, 1, 3]], [[0, 1, 4]]],
            [[3], [3], [3]],
            base_page_count=5,
            page_size_tokens=4,
        )
        analysis = analyze_qsa_topology_receipts([receipt])
        self.assertEqual(analysis["recommendation"], "ordered_union")
        self.assertFalse(analysis["exact_set"]["gate"]["passes"])
        self.assertTrue(analysis["ordered_union"]["gate"]["passes"])
        self.assertAlmostEqual(
            analysis["ordered_union"][
                "predicted_base_read_reduction_fraction_upper_bound"
            ],
            4 / 9,
        )

    def test_atomic_writer_includes_runtime_metadata(self):
        compact = SimpleNamespace(
            block_ids=np.array([[[0, 2, 4]]], dtype=np.uint32),
            block_counts=np.array([[3]], dtype=np.int32),
            block_size=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qsa-topology-test.json"
            write_qsa_topology_diagnostic(
                output,
                compact,
                base_tokens=16,
                metadata={"capture_index": 7},
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["source"]["metadata"]["capture_index"], 7)
            self.assertTrue(payload["topology"]["capture"]["device_readback"])
            self.assertTrue(payload["topology"]["capture"]["runtime_hot_path"])
            self.assertEqual(list(Path(directory).glob(".*.tmp-*")), [])

    def test_runtime_capture_is_opt_in_and_bounded(self):
        compact = SimpleNamespace(
            block_ids=np.array([[[0, 2, 4]]], dtype=np.uint32),
            block_counts=np.array([[3]], dtype=np.int32),
            block_size=4,
            physical_width=20,
            causal_mask=None,
        )
        q = np.zeros((1, 12, 1, 256), dtype=np.float16)
        delta_lengths = np.array([4], dtype=np.uint32)
        qwen4_qsa_indexed._TOPOLOGY_CAPTURE_COUNT = 0
        with patch.object(
            qwen4_qsa_indexed, "_TOPOLOGY_CAPTURE_DIR", None
        ), patch.object(qwen4_qsa_indexed.mx, "eval") as evaluate:
            self.assertIsNone(
                qwen4_qsa_indexed._capture_private_delta_topology_diagnostic(
                    compact,
                    q=q,
                    base_tokens=16,
                    delta_width=4,
                    delta_lengths=delta_lengths,
                    requested_splits=None,
                    requested_hpt=None,
                )
            )
            evaluate.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            qwen4_qsa_indexed, "_TOPOLOGY_CAPTURE_DIR", directory
        ), patch.object(qwen4_qsa_indexed, "_TOPOLOGY_CAPTURE_LIMIT", 2):
            paths = [
                qwen4_qsa_indexed._capture_private_delta_topology_diagnostic(
                    compact,
                    q=q,
                    base_tokens=16,
                    delta_width=4,
                    delta_lengths=delta_lengths,
                    requested_splits=None,
                    requested_hpt=None,
                )
                for _ in range(3)
            ]
            self.assertIsNotNone(paths[0])
            self.assertIsNotNone(paths[1])
            self.assertIsNone(paths[2])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_private_delta_attention_contains_the_opt_in_hook(self):
        source = Path(qwen4_qsa_indexed.__file__).read_text()
        function = source.split("def qwen4_qsa_indexed_private_delta_attention", 1)[
            1
        ].split("\ndef qwen4_qsa_indexed_attention", 1)[0]
        self.assertIn("_capture_private_delta_topology_diagnostic", function)


if __name__ == "__main__":
    unittest.main()
