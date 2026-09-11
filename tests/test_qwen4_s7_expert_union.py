import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import qwen4_s7_expert_union as s7_union


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def reshape(self, shape):
        return FakeArray(shape, self.dtype)


def production_inputs():
    return (
        FakeArray((1, 7, 2560), mx.bfloat16),
        FakeArray((1, 7, 10), mx.uint32),
        FakeArray((1, 7, 10), mx.bfloat16),
        FakeArray((512, 1280, 320), mx.uint32),
        FakeArray((512, 1280, 40), mx.bfloat16),
        FakeArray((512, 1280, 40), mx.bfloat16),
        FakeArray((512, 2560, 80), mx.uint32),
        FakeArray((512, 2560, 10), mx.bfloat16),
        FakeArray((512, 2560, 10), mx.bfloat16),
    )


class TestS7UnionPlan(unittest.TestCase):
    def test_union_is_query_major_and_maps_every_original_slot(self):
        routes = [
            list(range(0, 10)),
            list(range(5, 15)),
            list(range(10, 20)),
            list(range(15, 25)),
            list(range(20, 30)),
            list(range(25, 35)),
            list(range(30, 40)),
        ]
        plan = s7_union.build_expert_union(routes)
        self.assertEqual(plan.experts, tuple(range(40)))
        self.assertEqual(len(plan.slot_words), 40)

        for union_index, expert in enumerate(plan.experts):
            for query, row in enumerate(routes):
                expected = row.index(expert) if expert in row else None
                self.assertEqual(plan.slot(union_index, query), expected)

        # Expert 30 belongs to queries 5 and 6, so it must use the second
        # uint32 word. Packing seven bytes into one word would lose this map.
        at = plan.experts.index(30)
        low, high = plan.slot_words[at]
        self.assertEqual(low, 0)
        self.assertNotEqual(high, 0)
        self.assertEqual(plan.slot(at, 5), 5)
        self.assertEqual(plan.slot(at, 6), 0)

    def test_disjoint_routes_reach_the_seventy_expert_bound(self):
        routes = [list(range(query * 10, query * 10 + 10)) for query in range(7)]
        plan = s7_union.build_expert_union(routes)
        self.assertEqual(len(plan.experts), s7_union.MAX_UNION)
        self.assertEqual(plan.experts, tuple(range(70)))
        for query in range(7):
            for slot in range(10):
                self.assertEqual(plan.slot(query * 10 + slot, query), slot)

    def test_malformed_routes_fail_before_encoding(self):
        routes = [list(range(query * 10, query * 10 + 10)) for query in range(7)]
        with self.assertRaisesRegex(ValueError, "shape"):
            s7_union.build_expert_union(routes[:6])
        routes[3][9] = routes[3][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            s7_union.build_expert_union(routes)
        routes[3][9] = 512
        with self.assertRaisesRegex(ValueError, "outside"):
            s7_union.build_expert_union(routes)


class TestS7AdmissionAndReceipts(unittest.TestCase):
    def setUp(self):
        s7_union.set_qwen4_s7_expert_union(False)
        s7_union.qwen4_s7_expert_union_status(reset=True)

    def tearDown(self):
        s7_union.set_qwen4_s7_expert_union(False)
        s7_union.qwen4_s7_expert_union_status(reset=True)

    def test_default_gate_is_hard_off_and_counted(self):
        result = s7_union.admit_qwen4_s7_expert_union(*production_inputs())
        self.assertFalse(result.accepted)
        self.assertIn("disabled", result.reason)
        status = s7_union.qwen4_s7_expert_union_status()
        self.assertFalse(status["enabled"])
        self.assertEqual(status["counts"]["admission_calls"], 1)
        self.assertEqual(status["counts"]["disabled"], 1)
        self.assertEqual(status["counts"]["e1_dispatches"], 0)

    def test_exact_production_geometry_is_reachable_when_explicitly_enabled(self):
        with patch.object(
            s7_union.mx.metal, "is_available", return_value=True
        ), patch.object(s7_union.mx, "default_device", return_value=mx.gpu):
            result = s7_union.admit_qwen4_s7_expert_union(
                *production_inputs(), enabled=True
            )
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.tokens, 7)
        status = s7_union.qwen4_s7_expert_union_status()
        self.assertEqual(status["counts"]["admitted"], 1)

    def test_nearby_width_and_table_layout_are_refused_by_name(self):
        values = list(production_inputs())
        values[0] = FakeArray((1, 6, 2560), mx.bfloat16)
        values[1] = FakeArray((1, 6, 10), mx.uint32)
        values[2] = FakeArray((1, 6, 10), mx.bfloat16)
        result = s7_union.admit_qwen4_s7_expert_union(*values, enabled=True)
        self.assertFalse(result.accepted)
        self.assertIn("S=6 is not S=7", result.reason)

        values = list(production_inputs())
        values[3] = FakeArray((512, 640, 320), mx.uint32)
        result = s7_union.admit_qwen4_s7_expert_union(*values, enabled=True)
        self.assertFalse(result.accepted)
        self.assertIn("gate_up_weight shape", result.reason)

    def test_dispatch_receipts_prove_both_component_halves_ran(self):
        calls = []

        def fake_e1(**kwargs):
            calls.append(("e1", kwargs))
            return [FakeArray((7, 10, 640), mx.bfloat16)]

        def fake_e2(**kwargs):
            calls.append(("e2", kwargs))
            return [FakeArray((7, 2560), mx.bfloat16)]

        accepted = s7_union.S7ExpertUnionAdmission(True, "eligible", 7)
        with patch.object(
            s7_union, "admit_qwen4_s7_expert_union", return_value=accepted
        ), patch.object(
            s7_union, "_get_e1_kernel", return_value=fake_e1
        ), patch.object(s7_union, "_get_e2_kernel", return_value=fake_e2):
            result = s7_union.qwen4_s7_expert_union(*production_inputs())

        self.assertEqual(result.shape, (1, 7, 2560))
        self.assertEqual(calls[0][1]["grid"], (32, 640, 1))
        self.assertEqual(calls[1][1]["grid"], (32, 2560, 1))
        status = s7_union.qwen4_s7_expert_union_status()
        self.assertEqual(status["counts"]["component_calls"], 1)
        self.assertEqual(status["counts"]["e1_dispatches"], 1)
        self.assertEqual(status["counts"]["e2_dispatches"], 1)


class TestS7MetalSourceContract(unittest.TestCase):
    def test_both_kernels_use_two_slot_words_and_strided_route_views(self):
        for source in (s7_union._E1_SOURCE, s7_union._E2_SOURCE):
            self.assertIn("union_slot_lo", source)
            self.assertIn("union_slot_hi", source)
            self.assertIn("indices_strides", source)
            self.assertIn("8 * (query - 4)", source)
            self.assertIn("query < 4", source)
            self.assertNotIn("8 * query)) & 0xFFu);\n          if", source)

    def test_e1_keeps_weight_reads_outside_query_loop(self):
        source = s7_union._E1_SOURCE
        block_loop = source.index("for (uint block = lane")
        query_loop = source.index("for (uint query = 0", block_loop)
        self.assertLess(source.index("gu2[gate_row", block_loop), query_loop)
        active = source[block_loop : source.index("simd_sum", block_loop)]
        self.assertNotIn("slot1", active)

    def test_qmv_matches_stock_fast_sixteen_input_lane_partition(self):
        self.assertIn("constexpr uint IN_BLOCKS = (H / 8) / 2", s7_union._E1_SOURCE)
        self.assertIn("constexpr uint DOWN_BLOCKS = (EH / 8) / 2", s7_union._E2_SOURCE)
        self.assertIn("for (uint block = lane", s7_union._E1_SOURCE)
        self.assertIn("for (uint block = lane", s7_union._E2_SOURCE)
        self.assertIn("const device uint2* gu2", s7_union._E1_SOURCE)
        self.assertIn("const device uint2* down2", s7_union._E2_SOURCE)
        self.assertIn("qwen4_s7_qdot4", s7_union._QMV_HEADER)
        self.assertIn("a.y / 16.0f", s7_union._QMV_HEADER)

    def test_e2_stages_by_original_slot_then_reduces_slot_order(self):
        source = s7_union._E2_SOURCE
        stage = source.index("slot_values[elem] =")
        ordered = source.index("for (uint slot = 0; slot < TOPK", stage)
        reduction = source.index(
            "routed = T(float(routed) + slot_values[query * TOPK + slot])",
            ordered,
        )
        self.assertLess(stage, ordered)
        self.assertLess(ordered, reduction)
        union_loop = source.index("for (uint u = 0; u < union_n[0]")
        union_end = source.index(
            "threadgroup_barrier(mem_flags::mem_threadgroup);", union_loop
        )
        self.assertNotIn("routed", source[union_loop:union_end])

    def test_component_never_materializes_or_relayouts_weight_tables(self):
        source = open(s7_union.__file__, encoding="utf-8").read()
        for forbidden in (
            "mx.concatenate",
            "mx.dequantize",
            "mx.take(",
            "qmm_rhs",
            "Copyright ©",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
