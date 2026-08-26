# Copyright © 2026 Apple Inc.

import importlib.util
import json
import shutil
import unittest
from contextlib import contextmanager
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import utils
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.models.qwen4_exp import Model, ModelArgs, ShardedEmbedding, TextModelArgs
from mlx_lm.models.qwen4_ple_nvme import (
    FileBackedShardedEmbedding,
    assert_sidecar_not_in_weight_files,
    bf16_bits_to_f32,
    dequant_rows_numpy,
    f32_to_bf16_bits,
    verify_sidecar_against_artifact,
)

_REPO_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_qwen4_ple_sidecar",
    _REPO_ROOT / "scripts" / "build_qwen4_ple_sidecar.py",
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def tiny_nvme_args(**overrides):
    # dims per n-gram head = 640 / 4 = 160: five g32 groups per row, the
    # release geometry, so every test row straddles group boundaries.
    values = dict(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[2],
        ple_embed_dim=640,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )
    values.update(overrides)
    return TextModelArgs(**values)


@contextmanager
def env_var(name, value):
    previous = environ.get(name)
    if value is None:
        environ.pop(name, None)
    else:
        environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            environ.pop(name, None)
        else:
            environ[name] = previous


class TestQwen4PleNvme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        cls.sidecar = cls.model_dir / "ple_rows.bin"

        mx.random.seed(7)
        args = tiny_nvme_args()
        config = {"model_type": "qwen4_exp", "text_config": dict(args.__dict__)}
        model = Model(ModelArgs.from_dict(config))
        model.set_dtype(mx.bfloat16)
        model, config = utils.quantize_model(model, config, 64, 4)
        utils.save_model(cls.model_dir, model)
        utils.save_config(config, cls.model_dir / "config.json")
        assert builder.main(
            [str(cls.model_dir), "--out", str(cls.sidecar)]
        ) == 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def load_resident(self):
        with env_var("MLX_QWEN4_PLE_NVME", None):
            model, _ = utils.load_model(self.model_dir)
        return model

    def load_nvme(self):
        with env_var("MLX_QWEN4_PLE_NVME", str(self.sidecar)):
            model, _ = utils.load_model(self.model_dir)
        return model

    @staticmethod
    def ngram_embedding(model):
        return model.language_model.model.layers[1].ple.ple_embedding

    # ------------------------------------------------------------------
    # Sidecar builder
    # ------------------------------------------------------------------

    def test_builder_row_layout_matches_source_tensors(self):
        manifest = json.loads((Path(str(self.sidecar) + ".manifest.json")).read_text())
        self.assertEqual(manifest["dims"], 160)
        self.assertEqual(manifest["row_bytes"], 100)
        self.assertEqual(manifest["num_shards"], 4)
        self.assertEqual(manifest["total_rows"], 88)
        self.assertEqual(len(manifest["shard_sha256"]), 4)

        resident = self.ngram_embedding(self.load_resident()).ngram_embedding
        rows_per_shard = resident.rows_per_shard
        with open(self.sidecar, "rb") as f:
            data = f.read()
        rng = np.random.default_rng(3)
        for global_row in rng.integers(0, manifest["total_rows"], size=32):
            shard_index, local = divmod(int(global_row), rows_per_shard)
            shard = getattr(resident, f"shard_{shard_index}")
            expected = (
                np.asarray(shard.weight[local]).tobytes()
                + np.asarray(shard.scales[local].view(mx.uint16)).tobytes()
                + np.asarray(shard.biases[local].view(mx.uint16)).tobytes()
            )
            offset = int(global_row) * 100
            self.assertEqual(data[offset : offset + 100], expected)

    def test_builder_verify_passes_and_catches_corruption(self):
        self.assertEqual(
            builder.verify(self.model_dir, self.sidecar, num_rows=64, seed=0), 0
        )
        with TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "ple_rows.bin"
            shutil.copy(self.sidecar, corrupt)
            shutil.copy(
                str(self.sidecar) + ".manifest.json",
                str(corrupt) + ".manifest.json",
            )
            raw = bytearray(corrupt.read_bytes())
            for i in range(len(raw)):
                raw[i] ^= 0xFF
            corrupt.write_bytes(bytes(raw))
            self.assertGreater(
                builder.verify(self.model_dir, corrupt, num_rows=16, seed=0), 0
            )

    def test_manifest_digest_and_size_are_enforced(self):
        verify_sidecar_against_artifact(str(self.sidecar), self.model_dir)
        with TemporaryDirectory() as tmp:
            copied = Path(tmp) / "ple_rows.bin"
            shutil.copy(self.sidecar, copied)
            manifest_file = Path(str(copied) + ".manifest.json")
            shutil.copy(str(self.sidecar) + ".manifest.json", manifest_file)

            manifest = json.loads(manifest_file.read_text())
            manifest["source_index_sha256"] = "0" * 64
            manifest_file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different artifact"):
                verify_sidecar_against_artifact(str(copied), self.model_dir)

            shutil.copy(str(self.sidecar) + ".manifest.json", manifest_file)
            with open(copied, "ab") as f:
                f.write(b"\x00")
            with self.assertRaisesRegex(ValueError, "size"):
                verify_sidecar_against_artifact(str(copied), self.model_dir)

    # ------------------------------------------------------------------
    # Dequantization parity
    # ------------------------------------------------------------------

    def test_numpy_dequant_bit_exact_vs_default_stream_and_embedding(self):
        rng = np.random.default_rng(0)
        n, dims = 2048, 160
        scale = np.exp(rng.uniform(-8, 8, (n, 1))).astype(np.float32)
        base = rng.standard_normal((n, dims)).astype(np.float32) * scale
        source = mx.array(base).astype(mx.bfloat16)
        w, s, b = mx.quantize(source, group_size=32, bits=4)
        mx.eval(w, s, b)
        s_bits = np.asarray(s.view(mx.uint16))
        b_bits = np.asarray(b.view(mx.uint16))
        # The random battery must include negative scales.
        self.assertTrue((bf16_bits_to_f32(s_bits) < 0).any())

        rows = np.concatenate(
            [
                np.asarray(w).view(np.uint8).reshape(n, 80),
                s_bits.view(np.uint8).reshape(n, 10),
                b_bits.view(np.uint8).reshape(n, 10),
            ],
            axis=1,
        )
        ours = dequant_rows_numpy(rows, dims)

        reference = mx.dequantize(w, s, b, group_size=32, bits=4, mode="affine")
        mx.eval(reference)
        np.testing.assert_array_equal(
            ours, np.asarray(reference.view(mx.uint16))
        )

        embedding = nn.QuantizedEmbedding(n, dims, group_size=32, bits=4)
        embedding.weight, embedding.scales, embedding.biases = w, s, b
        ids = mx.array(rng.integers(0, n, size=(97,)))
        gathered = embedding(ids)
        mx.eval(gathered)
        np.testing.assert_array_equal(
            ours[np.asarray(ids)], np.asarray(gathered.view(mx.uint16))
        )

    def test_numpy_dequant_bit_exact_on_bf16_rounding_edge(self):
        # q=9, scale=169.0 (0x4329), bias=-1520.0 (0xc4be): the exact product
        # 9 * 169 - 1520 = 1.0 requires the multiply-add to round through
        # float32 exactly once. The CPU-stream mx.dequantize kernel rounds
        # through bfloat16 and returns 0.0 here, which is why the runtime
        # never uses it.
        dims = 160
        word = np.uint32(0)
        for i in range(8):
            word |= np.uint32(9) << np.uint32(4 * i)
        w = mx.array(np.full((1, 20), word, dtype=np.uint32))
        s = mx.array(np.full((1, 5), 0x4329, dtype=np.uint16)).view(mx.bfloat16)
        b = mx.array(np.full((1, 5), 0xC4BE, dtype=np.uint16)).view(mx.bfloat16)
        reference = mx.dequantize(w, s, b, group_size=32, bits=4, mode="affine")
        mx.eval(reference)

        rows = np.concatenate(
            [
                np.asarray(w).view(np.uint8).reshape(1, 80),
                np.asarray(s.view(mx.uint16)).view(np.uint8).reshape(1, 10),
                np.asarray(b.view(mx.uint16)).view(np.uint8).reshape(1, 10),
            ],
            axis=1,
        )
        ours = dequant_rows_numpy(rows, dims)
        np.testing.assert_array_equal(ours, np.asarray(reference.view(mx.uint16)))
        self.assertEqual(bf16_bits_to_f32(ours)[0, 0], 1.0)

    def test_file_backed_lookup_matches_resident_all_pool_sizes(self):
        resident = self.ngram_embedding(self.load_resident()).ngram_embedding
        file_backed = self.ngram_embedding(self.load_nvme()).ngram_embedding
        self.assertIsInstance(file_backed, FileBackedShardedEmbedding)
        rng = np.random.default_rng(11)
        for shape in ((1, 1, 16), (2, 5, 16), (2, 40, 16)):
            ids = rng.integers(0, 88, size=shape)
            expected = resident.lookup_numpy(ids)
            actual = file_backed.lookup_numpy(ids)
            mx.eval(expected, actual)
            self.assertEqual(actual.dtype, mx.bfloat16)
            self.assertTrue(
                mx.array_equal(
                    expected.view(mx.uint16), actual.view(mx.uint16)
                ).item()
            )

    def test_mx_fallback_dequant_backend_matches_numpy(self):
        file_backed = self.ngram_embedding(self.load_nvme()).ngram_embedding
        ids = np.arange(88).reshape(1, -1)
        expected = file_backed.lookup_numpy(ids)
        file_backed.dequant_backend = "mx"
        actual = file_backed.lookup_numpy(ids)
        file_backed.dequant_backend = "numpy"
        mx.eval(expected, actual)
        self.assertTrue(
            mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()
        )

    # ------------------------------------------------------------------
    # Load path
    # ------------------------------------------------------------------

    def test_env_unset_keeps_resident_shards(self):
        model = self.load_resident()
        embedding = self.ngram_embedding(model).ngram_embedding
        self.assertIsInstance(embedding, ShardedEmbedding)
        self.assertFalse(getattr(embedding, "is_file_backed", False))
        names = [name for name, _ in tree_flatten(model.parameters())]
        self.assertTrue(any(".ngram_embedding.shard_0.weight" in n for n in names))

    def test_env_set_installs_file_backed_and_drops_shards(self):
        model = self.load_nvme()
        embedding = self.ngram_embedding(model)
        self.assertIsInstance(
            embedding.ngram_embedding, FileBackedShardedEmbedding
        )
        self.assertEqual(embedding.hash_backend, "routed_cpu")
        names = [name for name, _ in tree_flatten(model.parameters())]
        self.assertFalse(any(".ngram_embedding.shard_" in n for n in names))

    def test_metal_hash_backend_falls_back_with_warning(self):
        import warnings

        with env_var("MLX_QWEN4_PLE_HASH_BACKEND", "metal"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = self.load_nvme()
        self.assertTrue(any("routed_cpu" in str(w.message) for w in caught))
        self.assertEqual(self.ngram_embedding(model).hash_backend, "routed_cpu")

    def test_wrong_artifact_is_refused_at_load(self):
        with TemporaryDirectory() as tmp:
            copied = Path(tmp) / "ple_rows.bin"
            shutil.copy(self.sidecar, copied)
            manifest_file = Path(str(copied) + ".manifest.json")
            manifest = json.loads(
                Path(str(self.sidecar) + ".manifest.json").read_text()
            )
            manifest["source_index_sha256"] = "0" * 64
            manifest_file.write_text(json.dumps(manifest))
            with env_var("MLX_QWEN4_PLE_NVME", str(copied)):
                with self.assertRaisesRegex(ValueError, "different artifact"):
                    utils.load_model(self.model_dir)

    def test_sidecar_never_enters_weight_files_or_ubc_evict_list(self):
        # load_model builds both the weight list and the MLX_LM_UBC_EVICT
        # eviction list from this exact glob; the sidecar must stay outside.
        import glob

        weight_files = glob.glob(str(self.model_dir / "model*.safetensors"))
        self.assertTrue(weight_files)
        self.assertTrue(self.sidecar.exists())
        self.assertNotIn(str(self.sidecar), weight_files)
        assert_sidecar_not_in_weight_files(str(self.sidecar))
        with self.assertRaisesRegex(ValueError, "weight glob"):
            assert_sidecar_not_in_weight_files("model_rows.safetensors")

    # ------------------------------------------------------------------
    # End to end
    # ------------------------------------------------------------------

    def assert_bit_identical(self, expected, actual):
        mx.eval(expected, actual)
        self.assertEqual(expected.dtype, actual.dtype)
        if expected.dtype == mx.bfloat16:
            expected = expected.view(mx.uint16)
            actual = actual.view(mx.uint16)
        self.assertTrue(mx.array_equal(expected, actual).item())

    def test_prefill_decode_and_multi_token_forwards_bit_identical(self):
        resident = self.load_resident()
        nvme = self.load_nvme()
        resident_cache = resident.make_cache()
        nvme_cache = nvme.make_cache()

        steps = [
            [[1, 2, 3, 4, 5]],  # prefill
            [[6]],  # decode
            [[7]],  # decode
            [[8, 9, 10]],  # MTP-verify style multi-token forward
        ]
        for step in steps:
            tokens = mx.array(step, dtype=mx.int64)
            self.assert_bit_identical(
                resident(tokens, cache=resident_cache),
                nvme(tokens, cache=nvme_cache),
            )
        # PLE token history and ShortConv state stay in lockstep.
        for index in (1, 2):
            self.assert_bit_identical(
                resident_cache[1][index], nvme_cache[1][index]
            )

    def test_speculative_rollback_stays_bit_identical(self):
        resident = self.load_resident()
        nvme = self.load_nvme()
        resident_cache = resident.make_cache()
        nvme_cache = nvme.make_cache()

        prompt = mx.array([[1, 2, 3]], dtype=mx.int64)
        self.assert_bit_identical(
            resident(prompt, cache=resident_cache),
            nvme(prompt, cache=nvme_cache),
        )
        for cache in (*resident_cache, *nvme_cache):
            cache.start_speculation()
        draft = mx.array([[4, 5, 6]], dtype=mx.int64)
        self.assert_bit_identical(
            resident(draft, cache=resident_cache),
            nvme(draft, cache=nvme_cache),
        )
        trim_prompt_cache(resident_cache, 2)
        trim_prompt_cache(nvme_cache, 2)
        for cache in (*resident_cache, *nvme_cache):
            cache.stop_speculation()
        follow = mx.array([[7]], dtype=mx.int64)
        self.assert_bit_identical(
            resident(follow, cache=resident_cache),
            nvme(follow, cache=nvme_cache),
        )
        self.assert_bit_identical(resident_cache[1][3], nvme_cache[1][3])

    def test_mtp_backbone_and_step_bit_identical(self):
        resident = self.load_resident()
        nvme = self.load_nvme()
        resident_cache = resident.make_cache()
        nvme_cache = nvme.make_cache()
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
        r_sample, r_hyper = resident.mtp_backbone(tokens, resident_cache)
        n_sample, n_hyper = nvme.mtp_backbone(tokens, nvme_cache)
        self.assert_bit_identical(r_sample, n_sample)
        self.assert_bit_identical(r_hyper, n_hyper)

        r_logits, r_next = resident.mtp_step(
            r_hyper[:, :2], mx.array([[2, 3]], dtype=mx.int64), resident.make_mtp_cache()
        )
        n_logits, n_next = nvme.mtp_step(
            n_hyper[:, :2], mx.array([[2, 3]], dtype=mx.int64), nvme.make_mtp_cache()
        )
        self.assert_bit_identical(r_logits, n_logits)
        self.assert_bit_identical(r_next, n_next)

    def test_generate_step_with_chunked_prefill_and_prefetch_matches(self):
        resident = self.load_resident()
        nvme = self.load_nvme()
        self.assertIsNone(resident.prefill_prefetch_hook())
        self.assertIsNotNone(nvme.prefill_prefetch_hook())

        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=mx.int64)
        outputs = {}
        for name, model in (("resident", resident), ("nvme", nvme)):
            cache = make_prompt_cache(model)
            outputs[name] = [
                token
                for token, _ in generate_step(
                    prompt,
                    model,
                    max_tokens=8,
                    prompt_cache=cache,
                    prefill_step_size=3,
                )
            ]
        self.assertEqual(outputs["resident"], outputs["nvme"])

    def test_prefetch_prompt_chunk_does_not_disturb_cache_or_results(self):
        nvme = self.load_nvme()
        embedding = self.ngram_embedding(nvme)
        cache = nvme.make_cache()
        prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
        first = nvme(prompt, cache=cache)
        mx.eval(first)
        history_before = np.asarray(cache[1][3]).copy()
        embedding.prefetch_prompt_chunk(
            np.array([[5, 6, 7]]), np.array([[3, 4]])
        )
        embedding.ngram_embedding._prefetch_pool.shutdown(wait=True)
        np.testing.assert_array_equal(np.asarray(cache[1][3]), history_before)


if __name__ == "__main__":
    unittest.main()
