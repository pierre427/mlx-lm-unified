"""Contract tests for the megakernel weight pack.

Synthetic tensors only: the point is the layout, the offset table and the
bit-exact round trip, none of which depend on the checkpoint.  The real-model
run is ``results/qwen4-megakernel-pack-check-20260903.py`` in the lab repo.
"""

import unittest

import mlx.core as mx

from mlx_lm.models.qwen4_megakernel_pack import (
    KIND_DENSE,
    KIND_QUANT,
    TABLE_FIELDS,
    TABLE_STRIDE,
    MegaWeightPack,
    PackError,
    build_pack,
    decode_path_keys,
    fuse_scales_biases,
)


class DictSource:
    """A source backed by a plain dict, with rebinding."""

    def __init__(self, tensors, quant):
        self.tensors = tensors
        self.quant = quant
        self.rebound = {}

    def has(self, key):
        return key in self.tensors

    def quant_spec(self, key):
        return self.quant[key]

    def fetch(self, key):
        return self.tensors[key]

    def rebind(self, key, parts):
        self.rebound[key] = parts

    def release(self):
        return None


def _quantized(rows, cols, bits=4, group_size=64, experts=0):
    lead = (experts,) if experts else ()
    words = cols * bits // 32
    ngroups = cols // group_size
    mx.random.seed(rows * 131 + cols)
    return {
        "weight": mx.random.randint(
            0, 2**31 - 1, (*lead, rows, words), dtype=mx.uint32
        ),
        "scales": mx.random.normal((*lead, rows, ngroups)).astype(mx.bfloat16),
        "biases": mx.random.normal((*lead, rows, ngroups)).astype(mx.bfloat16),
    }


def _dense(*shape):
    mx.random.seed(sum(shape))
    return {"weight": mx.random.normal(shape).astype(mx.bfloat16)}


class TestPackLayout(unittest.TestCase):
    def _source(self):
        tensors = {
            "a.proj": _quantized(256, 128),
            "a.norm.weight": _dense(128),
            "b.experts": _quantized(64, 128, experts=8),
            "b.gate": _quantized(32, 128, bits=8),
        }
        quant = {
            "a.proj": (4, 64),
            "a.norm.weight": (0, 0),
            "b.experts": (4, 64),
            "b.gate": (8, 64),
        }
        return DictSource(tensors, quant)

    def test_round_trip_is_bit_exact(self):
        source = self._source()
        plan = [
            ("a.proj", "main"),
            ("a.norm.weight", "main"),
            ("b.gate", "main"),
            ("b.experts", "experts"),
        ]
        pack = build_pack(source, plan, validate=True)
        self.assertEqual(pack.stats["validated_entries"], 4)
        for key, _ in plan:
            got = pack.views(key)
            want = source.fetch(key)
            for part, value in want.items():
                self.assertEqual(tuple(got[part].shape), tuple(value.shape), key)
                self.assertTrue(
                    mx.array_equal(got[part], value).item(), f"{key}.{part}"
                )

    def test_roles_do_not_share_a_buffer(self):
        source = self._source()
        plan = [
            ("a.proj", "main"),
            ("b.experts", "experts"),
            ("a.norm.weight", "main"),
        ]
        pack = build_pack(source, plan, validate=True)
        roles = {
            pack.group_roles[pack.entries[key].group] for key, _ in plan
        }
        self.assertEqual(roles, {"main", "experts"})
        self.assertNotEqual(
            pack.entries["a.proj"].group, pack.entries["b.experts"].group
        )
        # main entries stay together even though the plan interleaves roles
        self.assertEqual(
            pack.entries["a.proj"].group, pack.entries["a.norm.weight"].group
        )

    def test_group_cap_splits_buffers(self):
        source = self._source()
        plan = [("a.proj", "main"), ("b.gate", "main")]
        pack = build_pack(source, plan, max_group_bytes=8 * 1024, validate=True)
        self.assertGreater(len(pack.buffers), 1)
        # every group is either under the cap or a single oversize entry
        per_group = {}
        for key in pack.order:
            per_group.setdefault(pack.entries[key].group, []).append(key)
        for index, buf in enumerate(pack.buffers):
            if buf.size * 4 > 8 * 1024:
                self.assertEqual(len(per_group[index]), 1)

    def test_entries_are_aligned(self):
        source = self._source()
        plan = [("a.proj", "main"), ("a.norm.weight", "main"), ("b.gate", "main")]
        pack = build_pack(source, plan, validate=True)
        for key, _ in plan:
            entry = pack.entries[key]
            self.assertEqual(entry.w_off % 16, 0, key)
            self.assertEqual(entry.sb_off % 16, 0, key)

    def test_table_rows_describe_the_entries(self):
        source = self._source()
        plan = [("a.proj", "main"), ("a.norm.weight", "main"),
                ("b.experts", "experts")]
        pack = build_pack(source, plan, validate=True)
        self.assertEqual(pack.table.size, len(plan) * TABLE_STRIDE)
        row = pack.table_row("a.proj")
        self.assertEqual(row["kind"], KIND_QUANT)
        self.assertEqual(row["rows"], 256)
        self.assertEqual(row["cols"], 128)
        self.assertEqual(row["bits"], 4)
        self.assertEqual(row["group_size"], 64)
        self.assertEqual(row["experts"], 0)
        self.assertEqual(pack.table_row("b.experts")["experts"], 8)
        self.assertEqual(pack.table_row("a.norm.weight")["kind"], KIND_DENSE)
        flat = pack.table.tolist()
        for key in pack.order:
            entry = pack.entries[key]
            base = entry.index * TABLE_STRIDE
            self.assertEqual(flat[base: base + TABLE_STRIDE], entry.row())

    def test_rebind_hands_back_views(self):
        source = self._source()
        plan = [("a.proj", "main"), ("a.norm.weight", "main")]
        pack = build_pack(source, plan, validate=True, rebind=True)
        self.assertTrue(pack.stats["rebound"])
        for key, _ in plan:
            for part, value in source.rebound[key].items():
                self.assertTrue(
                    mx.array_equal(value, source.fetch(key)[part]).item(),
                    f"{key}.{part}",
                )

    def test_missing_tensor_fails_closed(self):
        source = self._source()
        with self.assertRaises(PackError):
            build_pack(source, [("nope", "main")])

    def test_fuse_layouts(self):
        scales = mx.arange(6).reshape(2, 3).astype(mx.bfloat16)
        biases = (mx.arange(6).reshape(2, 3) + 100).astype(mx.bfloat16)
        # shipped layout: a row's scales and biases adjacent
        fused = fuse_scales_biases(scales, biases)
        self.assertEqual(tuple(fused.shape), (2, 2, 3))
        self.assertTrue(mx.array_equal(fused[:, 0, :], scales).item())
        self.assertTrue(mx.array_equal(fused[:, 1, :], biases).item())
        split = fuse_scales_biases(scales, biases, "split")
        self.assertTrue(mx.array_equal(split[0], scales).item())
        self.assertTrue(mx.array_equal(split[1], biases).item())
        with self.assertRaises(PackError):
            fuse_scales_biases(scales, biases, "nope")

    def test_both_layouts_round_trip(self):
        plan = [("a.proj", "main"), ("b.experts", "experts"),
                ("b.gate", "main"), ("a.norm.weight", "main")]
        for layout in ("interleaved", "split"):
            source = self._source()
            pack = build_pack(source, plan, validate=True, sb_layout=layout)
            self.assertEqual(pack.stats["sb_layout"], layout)
            for key, _ in plan:
                got, want = pack.views(key), source.fetch(key)
                for part, value in want.items():
                    self.assertTrue(
                        mx.array_equal(got[part], value).item(),
                        f"{layout}/{key}.{part}",
                    )


class TestDecodePathPlan(unittest.TestCase):
    LAYER_TYPES = [
        "linear_attention" if (i + 1) % 4 else "full_attention" for i in range(48)
    ]

    def test_plan_covers_both_layer_kinds(self):
        plan = decode_path_keys(
            num_layers=48, layer_types=self.LAYER_TYPES, ple_layer_ids=[2]
        )
        keys = [key for key, _ in plan]
        self.assertIn("language_model.model.layers.0.linear_attn.in_proj_qkv", keys)
        self.assertIn("language_model.model.layers.3.self_attn.q_proj", keys)
        self.assertIn(
            "language_model.model.layers.3.self_attn.indexer.index_qk_proj", keys
        )
        self.assertIn("language_model.model.layers.1.ple.key_proj", keys)
        self.assertIn("language_model.lm_head", keys)
        self.assertIn("mtp.layers.0.self_attn.q_proj", keys)
        self.assertIn("mtp.fc_hidden", keys)
        # the embedding is hoisted on the host, never packed
        self.assertNotIn("language_model.model.embed_tokens", keys)

    def test_plan_has_no_duplicates(self):
        plan = decode_path_keys(
            num_layers=48, layer_types=self.LAYER_TYPES, ple_layer_ids=[2]
        )
        keys = [key for key, _ in plan]
        self.assertEqual(len(keys), len(set(keys)))

    def test_layer_filter_and_expert_toggle(self):
        plan = decode_path_keys(
            num_layers=48,
            layer_types=self.LAYER_TYPES,
            ple_layer_ids=[2],
            layers=[0, 3],
            include_mtp=False,
            include_experts=False,
        )
        keys = [key for key, _ in plan]
        self.assertNotIn("language_model.model.layers.1.ple.key_proj", keys)
        self.assertFalse(any("switch_mlp" in key for key in keys))
        self.assertFalse(any(key.startswith("mtp.") for key in keys))
        self.assertTrue(any(".layers.0." in key for key in keys))
        self.assertTrue(any(".layers.3." in key for key in keys))

    def test_expert_role_is_separate(self):
        plan = decode_path_keys(
            num_layers=48, layer_types=self.LAYER_TYPES, ple_layer_ids=[2]
        )
        roles = {role for key, role in plan if "switch_mlp" in key}
        self.assertEqual(roles, {"experts"})
        self.assertEqual(
            {role for key, role in plan if key == "language_model.lm_head"},
            {"lm_head"},
        )


if __name__ == "__main__":
    unittest.main()
