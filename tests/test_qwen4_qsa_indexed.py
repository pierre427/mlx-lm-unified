# Copyright © 2026 Apple Inc.

import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm.models import qwen4_qsa_indexed as indexed
from mlx_lm.models import qwen4_exp as qwen4_exp
from mlx_lm.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention
from mlx_lm.server import (
    APIHandler,
    SOFT_RELOAD_KEYS,
    apply_soft_reload,
    plan_soft_reload,
)


def _compact(batch, length, *, total=32, selected_width=7):
    left = np.arange(batch, dtype=np.int32) % 3
    ids = np.zeros((batch, length, selected_width), dtype=np.uint32)
    counts = np.zeros((batch, length), dtype=np.int32)
    tail_stop = np.zeros((batch, length), dtype=np.int32)
    for b in range(batch):
        logical_total = total - int(left[b])
        for row in range(length):
            q_pos = min(logical_total - 1, 7 + 2 * row)
            tail_stop[b, row] = q_pos + 1
            closed = (q_pos + 1) // 4
            count = min((b + row) % 5, closed)
            counts[b, row] = count
            if count:
                # At a block boundary this can end in the tail block. The
                # converter must deduplicate that appended slot.
                choices = np.arange(count, dtype=np.uint32)
                ids[b, row, :count] = choices
    tail_start = tail_stop // 4 * 4
    token_logical = np.arange(total)[None, :] - left[:, None]
    q_pos = tail_stop - 1
    causal = (
        (token_logical[:, None, :] >= 0)
        & (token_logical[:, None, :] <= q_pos[..., None])
    )[:, None]
    return QSACompactBlocks(
        block_ids=mx.array(ids),
        block_counts=mx.array(counts),
        tail_start=mx.array(tail_start),
        tail_stop=mx.array(tail_stop),
        left_padding=mx.array(left),
        block_size=4,
        physical_width=total,
        causal_mask=mx.array(causal),
    )


def _arrays(batch, length, *, total=32, dtype=mx.float32):
    heads, kv_heads, dim = 4, 2, 8
    q = mx.random.normal((batch, heads, length, dim)).astype(dtype)
    k = mx.random.normal((batch, kv_heads, total, dim)).astype(dtype)
    v = mx.random.normal((batch, kv_heads, total, dim)).astype(dtype)
    return q, k, v


class TestQSAIndexedReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def test_reference_matches_gather_random_and_adversarial_rows(self):
        mx.random.seed(17)
        for length in range(2, 9):
            batch = 1 + (length - 2) % 3
            with self.subTest(batch=batch, length=length):
                compact = _compact(batch, length)
                q, k, v = _arrays(batch, length)
                actual = indexed.qwen4_qsa_indexed_reference(
                    q, k, v, compact, scale=8**-0.5, splits=4
                )
                expected = _gather_qsa_attention(
                    q, k, v, compact, scale=8**-0.5, tile_rows=2
                )
                mx.eval(actual, expected)
                np.testing.assert_allclose(
                    np.asarray(actual),
                    np.asarray(expected),
                    rtol=1.0e-5,
                    atol=1.0e-5,
                )

    def test_split_count_is_bit_exact_at_attention_dtype(self):
        mx.random.seed(23)
        compact = _compact(2, 4)
        q, k, v = _arrays(2, 4, dtype=mx.bfloat16)
        outputs = [
            indexed.qwen4_qsa_indexed_reference(
                q, k, v, compact, scale=8**-0.5, splits=splits
            )
            for splits in (1, 2, 4, 8)
        ]
        mx.eval(*outputs)
        first = np.asarray(outputs[0].astype(mx.float32))
        for splits, output in zip((1, 2, 4, 8), outputs):
            self.assertTrue(
                np.array_equal(first, np.asarray(output.astype(mx.float32))),
                f"split count {splits} changed the final attention dtype",
            )

    def test_query_rows_are_independent(self):
        mx.random.seed(29)
        compact = _compact(2, 3)
        q, k, v = _arrays(2, 3)
        baseline = indexed.qwen4_qsa_indexed_reference(
            q, k, v, compact, scale=8**-0.5, splits=4
        )
        changed_q = mx.concatenate(
            [q[:, :, :1], q[:, :, 1:] * -31.0 + 19.0], axis=2
        )
        changed = indexed.qwen4_qsa_indexed_reference(
            changed_q,
            k,
            v,
            compact,
            scale=8**-0.5,
            splits=4,
        )
        mx.eval(baseline, changed)
        np.testing.assert_array_equal(
            np.asarray(baseline[:, :, 0]), np.asarray(changed[:, :, 0])
        )

    def test_duplicate_block_ids_fail_closed(self):
        compact = QSACompactBlocks(
            block_ids=mx.array([[[1, 1]]], dtype=mx.uint32),
            block_counts=mx.array([[2]], dtype=mx.int32),
            tail_start=mx.array([[4]], dtype=mx.int32),
            tail_stop=mx.array([[4]], dtype=mx.int32),
            left_padding=None,
            block_size=4,
            physical_width=16,
            causal_mask=None,
        )
        q, k, v = _arrays(1, 1, total=16)
        with self.assertRaisesRegex(ValueError, "unique"):
            indexed.qwen4_qsa_indexed_reference(
                q, k, v, compact, scale=8**-0.5, splits=1
            )

    def test_fully_masked_row_returns_zero_without_nan(self):
        compact = QSACompactBlocks(
            block_ids=mx.zeros((1, 2, 1), dtype=mx.uint32),
            block_counts=mx.zeros((1, 2), dtype=mx.int32),
            tail_start=mx.array([[4, 8]], dtype=mx.int32),
            tail_stop=mx.array([[4, 8]], dtype=mx.int32),
            left_padding=None,
            block_size=4,
            physical_width=16,
            causal_mask=None,
        )
        q, k, v = _arrays(1, 2, total=16)
        output = indexed.qwen4_qsa_indexed_reference(
            q, k, v, compact, scale=8**-0.5, splits=4
        )
        mx.eval(output)
        np.testing.assert_array_equal(np.asarray(output), np.zeros(output.shape))

    def test_metal_kernel_object_construction_does_not_dispatch(self):
        self.assertIsNotNone(indexed._partition_kernel())

    def test_synchronous_dispatch_failure_reuses_fetched_kv_for_gather(self):
        mx.random.seed(31)
        compact = _compact(1, 3)
        q, k, v = _arrays(1, 3)
        expected = _gather_qsa_attention(
            q, k, v, compact, scale=8**-0.5, tile_rows=1
        )
        seen = []

        def gather_spy(gq, gk, gv, *args, **kwargs):
            seen.append((gk is k, gv is v))
            return _gather_qsa_attention(gq, gk, gv, *args, **kwargs)

        indexed.qsa_indexed_status(reset=True)
        with (
            mock.patch.object(
                qwen4_exp,
                "qwen4_qsa_indexed_attention",
                side_effect=RuntimeError("synthetic dispatch failure"),
            ),
            mock.patch.object(qwen4_exp, "_gather_qsa_attention", gather_spy),
        ):
            actual = qwen4_exp._indexed_qsa_attention_or_gather(
                q,
                k,
                v,
                compact,
                scale=8**-0.5,
                splits=4,
                tile_rows=1,
            )
        mx.eval(actual, expected)
        self.assertEqual(seen, [(True, True)])
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        self.assertEqual(
            indexed.qsa_indexed_status()["counts"]["dispatch_raised"], 1
        )


class TestQSAIndexedAdmission(unittest.TestCase):
    def selection(self, **overrides):
        values = dict(
            kind="explicit",
            n_blocks=600,
            raw_block_ids=SimpleNamespace(shape=(1, 3, 512)),
            physical_width=16_384,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def decide(self, selection=None, **kwargs):
        values = dict(length=3, training=False, layout_ok=True)
        values.update(kwargs)
        return indexed.decide_qsa_indexed_admission(
            self.selection() if selection is None else selection, **values
        )

    def test_admission_reason_matrix(self):
        with mock.patch.object(indexed, "_QSA_INDEXED_ENABLED", False):
            self.assertEqual(self.decide(), (False, "disabled"))
        with (
            mock.patch.object(indexed, "_QSA_INDEXED_ENABLED", True),
            mock.patch.object(indexed, "indexed_kernel_available", return_value=True),
        ):
            cases = [
                ({"training": True}, self.selection(), "training"),
                ({}, self.selection(kind="implicit_all"), "selection_not_explicit"),
                ({}, self.selection(n_blocks=512), "dense_by_construction"),
                ({"length": 1}, self.selection(), "width_out_of_range"),
                ({}, self.selection(physical_width=8192), "context_out_of_range"),
                ({"layout_ok": False}, self.selection(), "unsupported_layout"),
            ]
            for kwargs, selection, reason in cases:
                with self.subTest(reason=reason):
                    self.assertEqual(
                        self.decide(selection=selection, **kwargs), (False, reason)
                    )
            self.assertEqual(self.decide(), (True, "engaged"))
        with (
            mock.patch.object(indexed, "_QSA_INDEXED_ENABLED", True),
            mock.patch.object(indexed, "indexed_kernel_available", return_value=False),
        ):
            self.assertEqual(self.decide(), (False, "kernel_unavailable"))

    def test_runtime_refusal_reasons_and_width_buckets_are_receipted(self):
        indexed.qsa_indexed_status(reset=True)
        for reason, width in (
            ("nax_engaged", 1),
            ("probe_declined", 3),
            ("dispatch_raised", 12),
        ):
            indexed.record_qsa_indexed_receipt(
                engaged=False,
                reason=reason,
                length=width,
                context=32_768,
                splits=4,
            )
        status = indexed.qsa_indexed_status()
        self.assertEqual(status["counts"]["nax_engaged"], 1)
        self.assertEqual(status["fallbacks"], 2)
        self.assertEqual(status["query_width_counts"]["1"]["declined"], 1)
        self.assertEqual(status["query_width_counts"]["2-8"]["declined"], 1)
        self.assertEqual(status["query_width_counts"][">8"]["declined"], 1)

    def test_env_unset_keeps_dispatch_decisions_identical(self):
        with mock.patch.object(indexed, "_QSA_INDEXED_ENABLED", False):
            for use_nax in (False, True):
                for gather_enabled in (False, True):
                    engage, reason = self.decide()
                    self.assertFalse(engage)
                    self.assertEqual(reason, "disabled")
                    old = (
                        use_nax,
                        gather_enabled and not use_nax,
                        not (use_nax or (gather_enabled and not use_nax)),
                    )
                    new_gather = gather_enabled and not use_nax and not engage
                    new = (use_nax, new_gather, not (use_nax or engage or new_gather))
                    self.assertEqual(repr(old).encode(), repr(new).encode())

    def test_split_table_and_override(self):
        expected = {
            8: 1,
            64: 1,
            65: 2,
            127: 2,
            128: 4,
            256: 4,
            512: 8,
            520: 8,
        }
        with mock.patch.object(indexed, "_SPLITS_OVERRIDE", 0):
            self.assertEqual(
                {width: indexed.indexed_splits_for(width) for width in expected},
                expected,
            )
        with mock.patch.object(indexed, "_SPLITS_OVERRIDE", 6):
            self.assertEqual(indexed.indexed_splits_for(520), 6)
        with self.assertRaises(ValueError):
            indexed.indexed_splits_for(0)


class TestQSAIndexedServer(unittest.TestCase):
    def test_soft_reload_key_and_status_endpoint_round_trip(self):
        self.assertIn("qwen4_qsa_indexed", SOFT_RELOAD_KEYS)
        original = indexed.qsa_indexed_enabled()
        try:
            indexed.set_qwen4_qsa_indexed(False)
            changes = apply_soft_reload(
                SimpleNamespace(),
                plan_soft_reload(SimpleNamespace(), {"qwen4_qsa_indexed": True}),
            )
            self.assertEqual(
                changes,
                {"qwen4_qsa_indexed": {"old": False, "new": True}},
            )
            handler = APIHandler.__new__(APIHandler)
            handler.path = "/v1/status/qwen4-qsa-indexed"
            handler.wfile = io.BytesIO()
            handler._set_completion_headers = lambda code=200: setattr(
                handler, "status", code
            )
            handler.end_headers = lambda: None
            handler.do_GET()
            body = json.loads(handler.wfile.getvalue())
            self.assertEqual(handler.status, 200)
            self.assertTrue(body["enabled"])
            self.assertIn("query_width_counts", body)
        finally:
            indexed.set_qwen4_qsa_indexed(original)


if __name__ == "__main__":
    unittest.main()
