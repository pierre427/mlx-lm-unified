# Copyright © 2026 Apple Inc.

import importlib.util
import json
import shutil
import unittest
from contextlib import contextmanager
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
    load_manifest,
    spot_check_sidecar_rows,
    verify_sidecar_against_artifact,
)

_REPO_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_qwen4_ple_sidecar",
    _REPO_ROOT / "scripts" / "build_qwen4_ple_sidecar.py",
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

_spec_hot = importlib.util.spec_from_file_location(
    "build_qwen4_ple_hot_rows",
    _REPO_ROOT / "scripts" / "build_qwen4_ple_hot_rows.py",
)
hot_rows = importlib.util.module_from_spec(_spec_hot)
_spec_hot.loader.exec_module(hot_rows)


def tiny_nvme_args(**overrides):
    # dims per n-gram head = 640 / 4 = 160: five g32 groups per row - the
    # release ROW geometry, so every test row straddles group boundaries.
    # Shard count/rows/offsets stay tiny; production-scale addressing is
    # covered by the sparse-sidecar and hash-range tests below.
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

    def test_builder_refuses_published_fp8_layout(self):
        with TemporaryDirectory() as tmp:
            fp8_dir = Path(tmp)
            prefix = (
                "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
            )
            tensors = {
                f"{prefix}.shard_0.weight": mx.to_fp8(mx.ones((4, 160))),
                f"{prefix}.weight_scale": mx.array([0.5], dtype=mx.bfloat16),
            }
            mx.save_safetensors(str(fp8_dir / "model.safetensors"), tensors)
            (fp8_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {k: "model.safetensors" for k in tensors},
                    }
                )
            )
            # The dedicated "FP8 layout" diagnostic ships with the FP8
            # checkpoint-ingestion commit, which is intentionally not part of
            # the q4-only serving port. The q4 builder still refuses the FP8
            # layout: it sees U8 shard weights where it requires U32-packed
            # 4-bit words and raises before building a sidecar.
            with self.assertRaisesRegex(ValueError, "FP8 layout|dtype U8"):
                builder.collect_shards(fp8_dir)

    def test_manifest_geometry_is_validated(self):
        cases = (
            ({"bits": 8}, "quantization"),
            ({"mode": "mxfp4"}, "quantization"),
            ({"dims": 168}, "multiple of 32"),
            ({"row_bytes": 99}, "does not match"),
            ({"weight_bytes": 84}, "does not match"),
            ({"total_rows": 89}, "shards x"),
            ({"num_shards": 0}, "positive"),
            ({"data_offset": -1}, "non-negative"),
            ({"shard_sha256": ["0" * 64]}, "one sha256 per shard"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with TemporaryDirectory() as tmp:
                    copied = Path(tmp) / "ple_rows.bin"
                    shutil.copy(self.sidecar, copied)
                    manifest_file = Path(str(copied) + ".manifest.json")
                    manifest = json.loads(
                        Path(str(self.sidecar) + ".manifest.json").read_text()
                    )
                    manifest.update(overrides)
                    manifest_file.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, message):
                        verify_sidecar_against_artifact(
                            str(copied), self.model_dir
                        )

    def test_truncated_source_file_is_refused_by_range_check(self):
        manifest = load_manifest(str(self.sidecar))
        with TemporaryDirectory() as tmp:
            broken_dir = Path(tmp) / "model"
            shutil.copytree(self.model_dir, broken_dir)
            for weights_file in broken_dir.glob("*.safetensors"):
                _, data_offset = builder.read_safetensors_header(weights_file)
                with open(weights_file, "r+b") as f:
                    f.truncate(data_offset + 10)
            with self.assertRaisesRegex(ValueError, r"byte range .* exceeds"):
                spot_check_sidecar_rows(
                    str(self.sidecar), broken_dir, manifest, num_random=4
                )

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

    def test_private_verify_table_matches_resident_bits_on_cpu(self):
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            resident = self.ngram_embedding(self.load_resident()).ngram_embedding
            file_backed = self.ngram_embedding(self.load_nvme()).ngram_embedding
            ids = np.arange(48, dtype=np.int64).reshape(1, 3, 16) % 88
            expected = resident.lookup_numpy(ids)
            actual = file_backed.lookup_verify_device(mx.array(ids))
            mx.eval(expected, actual)
            np.testing.assert_array_equal(
                np.asarray(expected.view(mx.uint16)),
                np.asarray(actual.view(mx.uint16)),
            )
        finally:
            mx.set_default_device(previous)

    @unittest.skipUnless(
        mx.metal.is_available()
        and environ.get("MLX_QWEN4_PLE_VERIFY_METAL_TESTS") == "1",
        "requires the explicit PLE Metal test gate",
    )
    def test_width_three_verify_uses_no_numpy_hot_path(self):
        from mlx_lm.verify_sync import (
            trace_verify_syncs,
            verify_sync_round,
            verify_sync_status,
        )

        resident = self.load_resident()
        with env_var("MLX_QWEN4_PLE_VERIFY_DEVICE", "1"):
            nvme = self.load_nvme()
        resident_embedding = self.ngram_embedding(resident)
        embedding = self.ngram_embedding(nvme)
        table = embedding.ngram_embedding
        resident_cache = resident.make_cache()[1]
        cache = nvme.make_cache()[1]
        tokens = mx.array([[1, 2, 3]], dtype=mx.int64)
        mask = mx.ones((1, 3), dtype=mx.bool_)
        expected = resident_embedding(
            tokens, cache=resident_cache, mask=mask
        )
        mx.eval(expected, resident_cache[3])
        with (
            patch.object(
                embedding,
                "_ngram_ids_numpy",
                side_effect=AssertionError("NumPy hash used"),
            ),
            patch.object(
                table,
                "lookup_numpy",
                side_effect=AssertionError("NumPy lookup used"),
            ),
            trace_verify_syncs(),
        ):
            with verify_sync_round():
                output = embedding(tokens, cache=cache, mask=mask)
                mx.eval(output, cache[3])
        status = verify_sync_status()
        np.testing.assert_array_equal(
            np.asarray(expected.view(mx.uint16)),
            np.asarray(output.view(mx.uint16)),
        )
        np.testing.assert_array_equal(
            np.asarray(resident_cache[3]), np.asarray(cache[3])
        )
        self.assertEqual(status["total"], 0)
        self.assertTrue(table.verify_status["device_prepared"])
        self.assertEqual(table.verify_status["device_lookups"], 1)
        self.assertEqual(table.verify_status["fallback_lookups"], 0)

    def test_non_verify_shape_uses_sidecar_fallback_receipt(self):
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            nvme = self.load_nvme()
            embedding = self.ngram_embedding(nvme)
            table = embedding.ngram_embedding
            cache = nvme.make_cache()[1]
            with patch.object(
                table, "lookup_numpy", wraps=table.lookup_numpy
            ) as fallback:
                output = embedding(
                    mx.array([[1, 2, 3, 4]], dtype=mx.int64),
                    cache=cache,
                    mask=mx.ones((1, 4), dtype=mx.bool_),
                )
                mx.eval(output)
            self.assertEqual(fallback.call_count, 1)
            self.assertEqual(table.verify_status["device_lookups"], 0)
            self.assertEqual(table.verify_status["fallback_lookups"], 1)
        finally:
            mx.set_default_device(previous)

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

    def test_lookup_stats_count_and_timing_is_opt_in(self):
        file_backed = self.ngram_embedding(self.load_nvme()).ngram_embedding
        self.assertFalse(file_backed.stats_timing)
        ids = np.array([[0, 5, 5, 87]])
        file_backed.lookup_numpy(ids)
        stats = file_backed.stats
        self.assertEqual(stats.lookups, 1)
        self.assertEqual(stats.rows, 4)
        self.assertEqual(stats.unique_rows, 3)
        self.assertEqual(stats.bytes_read, 3 * file_backed.row_bytes)
        self.assertEqual(stats.elapsed_seconds, 0.0)
        self.assertEqual(
            {k: type(v) for k, v in vars(stats).items()},
            {
                "lookups": int,
                "rows": int,
                "unique_rows": int,
                "bytes_read": int,
                "elapsed_seconds": float,
                "cache_hits": int,
                "cache_misses": int,
                "cache_evictions": int,
            },
        )

        with env_var("MLX_QWEN4_PLE_NVME_STATS_TIMING", "1"):
            timed = self.ngram_embedding(self.load_nvme()).ngram_embedding
        self.assertTrue(timed.stats_timing)
        timed.lookup_numpy(ids)
        self.assertGreater(timed.stats.elapsed_seconds, 0.0)

    # ------------------------------------------------------------------
    # Hot tier: LRU row cache + preheat
    # ------------------------------------------------------------------

    def load_nvme_hot(self, lru_mb="1", preheat=None):
        with env_var("MLX_QWEN4_PLE_NVME_LRU_MB", lru_mb):
            with env_var("MLX_QWEN4_PLE_NVME_PREHEAT", preheat):
                return self.load_nvme()

    def test_lru_cache_is_exact_and_counted(self):
        plain = self.ngram_embedding(self.load_nvme()).ngram_embedding
        hot = self.ngram_embedding(self.load_nvme_hot()).ngram_embedding
        self.assertEqual(plain.lru_capacity_rows, 0)
        self.assertGreater(hot.lru_capacity_rows, 88)
        rng = np.random.default_rng(23)
        ids = rng.integers(0, 88, size=(2, 7, 16))
        first = hot.lookup_numpy(ids)
        expected = plain.lookup_numpy(ids)
        mx.eval(first, expected)
        self.assertTrue(
            mx.array_equal(
                first.view(mx.uint16), expected.view(mx.uint16)
            ).item()
        )
        stats = hot.stats
        unique = np.unique(ids).size
        self.assertEqual(stats.cache_misses, unique)
        self.assertEqual(stats.bytes_read, unique * hot.row_bytes)

        # The second lookup is served fully from the cache, byte-identical.
        second = hot.lookup_numpy(ids)
        mx.eval(second)
        self.assertTrue(
            mx.array_equal(
                second.view(mx.uint16), expected.view(mx.uint16)
            ).item()
        )
        stats = hot.stats
        self.assertEqual(stats.cache_hits, unique)
        self.assertEqual(stats.cache_misses, unique)
        self.assertEqual(stats.bytes_read, unique * hot.row_bytes)

    def test_lru_budget_bounds_entries_and_counts_evictions(self):
        # 0.001 MB = 1048 bytes = 10 rows of 100 B.
        hot = self.ngram_embedding(
            self.load_nvme_hot(lru_mb="0.001")
        ).ngram_embedding
        self.assertEqual(hot.lru_capacity_rows, 10)
        hot.lookup_numpy(np.arange(88))
        self.assertLessEqual(len(hot._lru), 10)
        self.assertEqual(hot.stats.cache_evictions, 88 - 10)
        # Cached survivors still serve exact bytes.
        survivors = np.asarray(sorted(hot._lru), dtype=np.int64)
        expected = self.ngram_embedding(self.load_nvme()).ngram_embedding
        actual = hot.lookup_numpy(survivors)
        reference = expected.lookup_numpy(survivors)
        mx.eval(actual, reference)
        self.assertTrue(
            mx.array_equal(
                actual.view(mx.uint16), reference.view(mx.uint16)
            ).item()
        )

    def test_lru_is_cleared_on_close_and_rebuilt_on_fork_lifecycle(self):
        hot = self.ngram_embedding(self.load_nvme_hot()).ngram_embedding
        hot.lookup_numpy(np.arange(16))
        self.assertEqual(len(hot._lru), 16)
        hot.close()
        self.assertEqual(len(hot._lru), 0)

    def test_prefetch_populates_the_lru(self):
        hot = self.ngram_embedding(self.load_nvme_hot()).ngram_embedding
        ids = np.arange(12)
        for future in hot.prefetch_rows(ids):
            future.result()
        self.assertEqual(len(hot._lru), 12)
        plain = self.ngram_embedding(self.load_nvme()).ngram_embedding
        actual = hot.lookup_numpy(ids)
        reference = plain.lookup_numpy(ids)
        mx.eval(actual, reference)
        self.assertTrue(
            mx.array_equal(
                actual.view(mx.uint16), reference.view(mx.uint16)
            ).item()
        )
        stats = hot.stats
        self.assertEqual(stats.cache_hits, 12)
        self.assertEqual(stats.bytes_read, 0)

    def test_cache_put_after_close_is_a_no_op(self):
        hot = self.ngram_embedding(self.load_nvme_hot()).ngram_embedding
        ids = np.arange(4)
        rows = hot._pread_rows(ids, 4)
        hot.close()
        # A foreground miss returning from its preads after close() must
        # not repopulate the cleared cache.
        hot._cache_put(ids, rows)
        self.assertEqual(len(hot._lru), 0)

    def test_stale_pid_rebuilds_before_cache_hit_and_prefetch(self):
        from os import getpid

        hot = self.ngram_embedding(self.load_nvme_hot()).ngram_embedding
        ids = np.arange(8)
        expected = hot.lookup_numpy(ids)
        self.assertEqual(len(hot._lru), 8)

        # Simulate a forked child: the pid no longer matches, so the next
        # complete cache hit must rebuild (empty LRU, fresh fd/pools) and
        # re-read from disk instead of serving inherited state.
        hot._owner_pid = getpid() + 1
        bytes_before = hot.stats.bytes_read
        again = hot.lookup_numpy(ids)
        mx.eval(expected, again)
        self.assertEqual(hot._owner_pid, getpid())
        self.assertTrue(
            mx.array_equal(
                again.view(mx.uint16), expected.view(mx.uint16)
            ).item()
        )
        self.assertGreater(hot.stats.bytes_read, bytes_before)

        # Same for the prefetch membership filter.
        hot._owner_pid = getpid() + 1
        for future in hot.prefetch_rows(ids):
            future.result()
        self.assertEqual(hot._owner_pid, getpid())
        self.assertEqual(len(hot._lru), 8)

    def test_preheat_eviction_priority_and_dedupe_before_cap(self):
        with TemporaryDirectory() as tmp:
            # Hottest-first manifest, capacity 10 (0.001 MB): eviction must
            # discard the COLDEST preheated rows first.
            ranked = Path(tmp) / "ranked.txt"
            ranked.write_text("\n".join(str(i) for i in range(20)) + "\n")
            hot = self.ngram_embedding(
                self.load_nvme_hot(lru_mb="0.001", preheat=str(ranked))
            ).ngram_embedding
            self.assertEqual(hot.preheated_rows, 10)
            self.assertEqual(sorted(hot._lru), list(range(10)))
            hot.lookup_numpy(np.array([30, 31, 32]))
            self.assertEqual(
                sorted(hot._lru), [0, 1, 2, 3, 4, 5, 6, 30, 31, 32]
            )

            # Duplicates dedupe BEFORE the cap, so they cannot underfill
            # the budget: capacity 3 still gets three distinct hot rows.
            dup = Path(tmp) / "dup.txt"
            dup.write_text("5\n5\n7\n5\n2\n7\n1\n")
            capped = self.ngram_embedding(
                self.load_nvme_hot(lru_mb="0.0003", preheat=str(dup))
            ).ngram_embedding
            self.assertEqual(capped.lru_capacity_rows, 3)
            self.assertEqual(capped.preheated_rows, 3)
            self.assertEqual(sorted(capped._lru), [2, 5, 7])

    def test_hot_rows_verify_constants_against_checkpoint(self):
        embedding = hot_rows.build_ngram_embedding(self.model_dir, 0)
        status = hot_rows.verify_hash_constants(embedding, self.model_dir)
        self.assertIn("verified", status)
        embedding.layer_multipliers = mx.array(
            np.asarray(embedding.layer_multipliers, dtype=np.int64) + 1
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            hot_rows.verify_hash_constants(embedding, self.model_dir)

    def test_preheat_manifest_loads_caps_and_fails_closed(self):
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "hot_rows.txt"
            manifest.write_text(
                "# qwen4-ple-hot-rows v1 {}\n5\n7\n5\n2\n# comment\n\n"
            )
            hot = self.ngram_embedding(
                self.load_nvme_hot(preheat=str(manifest))
            ).ngram_embedding
            self.assertEqual(hot.preheated_rows, 3)
            hot.lookup_numpy(np.array([2, 5, 7]))
            self.assertEqual(hot.stats.cache_hits, 3)
            self.assertEqual(hot.stats.bytes_read, 0)

            # Budget cap: 10 rows keeps only the first (hottest) ids.
            many = Path(tmp) / "many.txt"
            many.write_text("\n".join(str(i) for i in range(40)) + "\n")
            capped = self.ngram_embedding(
                self.load_nvme_hot(lru_mb="0.001", preheat=str(many))
            ).ngram_embedding
            self.assertEqual(capped.preheated_rows, 10)
            self.assertEqual(sorted(capped._lru), list(range(10)))

            # Fail closed: id outside the table, and preheat without LRU.
            bad = Path(tmp) / "bad.txt"
            bad.write_text("88\n")
            with self.assertRaisesRegex(ValueError, "outside the PLE table"):
                self.load_nvme_hot(preheat=str(bad))
            with self.assertRaisesRegex(ValueError, "requires"):
                self.load_nvme_hot(lru_mb=None, preheat=str(manifest))

    def test_hot_rows_script_matches_unchunked_hash_and_preheats(self):
        embedding = hot_rows.build_ngram_embedding(self.model_dir, 0)
        self.assertEqual(embedding.layer_idx, 1)
        rng = np.random.default_rng(5)
        tokens = rng.integers(0, 64, size=300)
        tokens[::37] = 63  # sprinkle EOS segment resets

        chunked = hot_rows.hash_corpus_row_counts(embedding, tokens, 7)
        whole = hot_rows.hash_corpus_row_counts(embedding, tokens, 10_000)
        self.assertEqual(chunked, whole)
        self.assertTrue(all(0 <= i < 88 for i in chunked))
        self.assertEqual(
            sum(chunked.values()), tokens.size * embedding.ngram_heads
        )

        hottest = hot_rows.top_rows(chunked, 5)
        self.assertEqual(len(hottest), 5)
        counts = [chunked[i] for i in hottest]
        self.assertEqual(counts, sorted(counts, reverse=True))

        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "hot.txt"
            hot_rows.write_hot_rows(manifest, hottest, {"test": True})
            hot = self.ngram_embedding(
                self.load_nvme_hot(preheat=str(manifest))
            ).ngram_embedding
            self.assertEqual(hot.preheated_rows, 5)

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
        self.assertEqual(len(embedding.ngram_embedding._verify_shards), 4)
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

    def test_save_and_fuse_refused_while_file_backed(self):
        # save_model serializes model.parameters(); with the shards pruned
        # it would silently write an artifact without PLE tables. fuse()
        # saves through the same path, so this guard covers both.
        nvme = self.load_nvme()
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "NVMe sidecar"):
                utils.save_model(Path(tmp) / "export", nvme)

    def test_stale_sidecar_same_index_different_source_refused(self):
        # Corrupt the source shard tensors while keeping
        # model.safetensors.index.json byte-identical: the index digest
        # check passes, the content spot check must refuse.
        with TemporaryDirectory() as tmp:
            stale_dir = Path(tmp) / "stale"
            shutil.copytree(self.model_dir, stale_dir)
            _, shards = builder.collect_shards(stale_dir)
            ref = shards[0]["weight"]
            with open(ref.file, "r+b") as f:
                f.seek(ref.start)
                length = ref.end - ref.start
                data = bytearray(f.read(length))
                for i in range(length):
                    data[i] ^= 0xFF
                f.seek(ref.start)
                f.write(bytes(data))
            self.assertEqual(
                (stale_dir / "model.safetensors.index.json").read_bytes(),
                (self.model_dir / "model.safetensors.index.json").read_bytes(),
            )
            with env_var(
                "MLX_QWEN4_PLE_NVME", str(stale_dir / "ple_rows.bin")
            ):
                with self.assertRaisesRegex(ValueError, "stale or corrupt"):
                    utils.load_model(stale_dir)

    def test_bitflipped_same_size_sidecar_fails_load_preflight(self):
        with TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "ple_rows.bin"
            raw = bytearray(self.sidecar.read_bytes())
            for i in range(len(raw)):
                raw[i] ^= 0xFF
            corrupt.write_bytes(bytes(raw))
            shutil.copy(
                str(self.sidecar) + ".manifest.json",
                str(corrupt) + ".manifest.json",
            )
            with env_var("MLX_QWEN4_PLE_NVME", str(corrupt)):
                with self.assertRaisesRegex(ValueError, "stale or corrupt"):
                    utils.load_model(self.model_dir)

    def test_ubc_eviction_call_never_receives_the_sidecar(self):
        from unittest import mock

        with env_var("MLX_QWEN4_PLE_NVME", str(self.sidecar)):
            with env_var("MLX_LM_UBC_EVICT", "1"):
                with mock.patch(
                    "mlx_lm.ubc_evict.ubc_evict_paths", return_value=0
                ) as evict:
                    utils.load_model(self.model_dir)
        evict.assert_called_once()
        (paths,) = evict.call_args.args
        real_paths = {Path(p).resolve() for p in paths}
        self.assertIn((self.model_dir / "model.safetensors").resolve(), real_paths)
        self.assertNotIn(self.sidecar.resolve(), real_paths)

    def test_linear_only_lora_applies_and_ple_adapter_is_refused(self):
        from mlx.utils import tree_flatten as flatten

        from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

        lora_config = {"rank": 2, "scale": 1.0, "dropout": 0.0,
                       "keys": ["self_attn.q_proj"]}
        donor = self.load_nvme()
        linear_to_lora_layers(donor, 1, lora_config)
        adapter_weights = {
            name: value
            for name, value in flatten(donor.parameters())
            if name.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
        }
        self.assertTrue(adapter_weights)

        with TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "fine_tune_type": "lora",
                        "num_layers": 1,
                        "lora_parameters": lora_config,
                    }
                )
            )
            mx.save_safetensors(
                str(adapter_dir / "adapters.safetensors"), adapter_weights
            )
            model = load_adapters(self.load_nvme(), str(adapter_dir))
            logits = model(mx.array([[1, 2, 3]], dtype=mx.int64))
            mx.eval(logits)

        shard_key = (
            "language_model.model.layers.1.ple.ple_embedding"
            ".ngram_embedding.shard_0.weight"
        )
        with TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"fine_tune_type": "full"})
            )
            mx.save_safetensors(
                str(adapter_dir / "adapters.safetensors"),
                {shard_key: mx.zeros((22, 20), dtype=mx.uint32)},
            )
            with self.assertRaisesRegex(ValueError, "NVMe sidecar"):
                load_adapters(self.load_nvme(), str(adapter_dir))

    def test_fork_recovery_and_close_semantics(self):
        table = self.ngram_embedding(self.load_nvme()).ngram_embedding
        ids = np.arange(16).reshape(1, 1, 16)
        before = np.asarray(table.lookup_numpy(ids).view(mx.uint16))

        # Simulate a fork: pretend another process created the pools. The
        # next lookup must rebuild fd + executors and still read correctly.
        # (In a real fork the parent keeps its own resources; here both
        # live in one process, so release the pre-"fork" ones afterwards.)
        stale_pool, stale_prefetch, stale_fd = (
            table._pool,
            table._prefetch_pool,
            table._fd,
        )
        table._owner_pid = -1
        after = np.asarray(table.lookup_numpy(ids).view(mx.uint16))
        np.testing.assert_array_equal(before, after)
        self.assertIsNot(table._pool, stale_pool)
        import os as _os

        self.assertEqual(table._owner_pid, _os.getpid())
        stale_pool.shutdown(wait=True)
        stale_prefetch.shutdown(wait=True)
        _os.close(stale_fd)

        table.close()
        table.close()  # idempotent
        with self.assertRaisesRegex(RuntimeError, "closed"):
            table.lookup_numpy(ids)
        table.prefetch_rows(ids)  # optional path: silent no-op when closed
        table.submit_prefetch(lambda: None)

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

    def test_short_history_prefetch_is_wrong_row_but_harmless(self):
        # An APC restore can hand the prefetcher fewer than context_len
        # prior tokens. The EOS-padded hash then warms DIFFERENT rows than
        # the real cache-backed lookup will use - a cold read, never a
        # wrong result. Pin both halves of that contract.
        resident = self.load_resident()
        nvme = self.load_nvme()
        embedding = self.ngram_embedding(nvme)

        # (a) the prefetched ids differ from the true-history ids
        chunk = mx.array([[5, 6, 7]], dtype=mx.int64)
        true_ids = embedding._ngram_ids_numpy(
            chunk, None, previous=np.array([[3, 4]])
        )
        short_ids = embedding._ngram_ids_numpy(
            chunk, None, previous=np.array([[4]])  # one-token APC suffix
        )
        self.assertFalse(np.array_equal(true_ids, short_ids))

        # (b) after a short-suffix prefetch, real forwards stay
        # bit-identical to the resident model
        embedding.prefetch_prompt_chunk(np.array([[5, 6, 7]]), np.array([[4]]))
        embedding.ngram_embedding._prefetch_pool.shutdown(wait=True)
        resident_cache = resident.make_cache()
        nvme_cache = nvme.make_cache()
        for step in ([[1, 2, 3, 4]], [[5]], [[6, 7]]):
            tokens = mx.array(step, dtype=mx.int64)
            self.assert_bit_identical(
                resident(tokens, cache=resident_cache),
                nvme(tokens, cache=nvme_cache),
            )


class TestQwen4PleNvmeProductionGeometry(unittest.TestCase):
    """Production-scale addressing and numerics without a 32 GB sidecar."""

    ROWS_PER_SHARD = 2_500_012
    NUM_SHARDS = 128
    TOTAL_ROWS = ROWS_PER_SHARD * NUM_SHARDS  # 320,001,536
    REACHABLE_ROWS = 320_001_446  # sum of the 16 per-head prime vocabularies

    def test_sparse_sidecar_production_shard_edges(self):
        # An APFS-sparse file gives the production 32 GB offset space with
        # near-zero allocation: only the probed rows are written. Covers
        # shard boundaries, the last reachable row, and both ends of the
        # 90-row padding tail.
        probe_rows = [
            0,
            self.ROWS_PER_SHARD - 1,  # last row of shard 0
            self.ROWS_PER_SHARD,  # first row of shard 1
            64 * self.ROWS_PER_SHARD + 12345,  # mid-file
            self.REACHABLE_ROWS - 1,  # last hash-reachable row
            self.REACHABLE_ROWS,  # first padded row
            self.TOTAL_ROWS - 1,  # last padded row (file end - 100 B)
        ]
        rng = np.random.default_rng(5)
        payload = {
            row: rng.integers(0, 256, size=100, dtype=np.uint8).tobytes()
            for row in probe_rows
        }
        with TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "ple_rows.bin"
            with open(sparse, "wb") as f:
                f.truncate(self.TOTAL_ROWS * 100)
            self.assertEqual(
                sparse.stat().st_size, 32_000_153_600
            )
            with open(sparse, "r+b") as f:
                for row, data in payload.items():
                    f.seek(row * 100)
                    f.write(data)
            table = FileBackedShardedEmbedding(
                str(sparse),
                vocab_size=self.TOTAL_ROWS,
                dims=160,
                num_shards=self.NUM_SHARDS,
            )
            try:
                ids = np.array(probe_rows, dtype=np.int64).reshape(1, -1)
                actual = np.asarray(table.lookup_numpy(ids).view(mx.uint16))
                expected = dequant_rows_numpy(
                    np.stack(
                        [
                            np.frombuffer(payload[row], dtype=np.uint8)
                            for row in probe_rows
                        ]
                    ),
                    160,
                )
                np.testing.assert_array_equal(actual[0], expected)
                # The shard/local split matches the resident computation.
                for row in probe_rows:
                    self.assertEqual(
                        divmod(row, self.ROWS_PER_SHARD),
                        (row // self.ROWS_PER_SHARD,
                         row - (row // self.ROWS_PER_SHARD)
                         * self.ROWS_PER_SHARD),
                    )
            finally:
                table.close()

    def test_release_hash_range_never_reaches_padding(self):
        from mlx_lm.models.qwen4_exp import NGramEmbedding

        args = TextModelArgs(ple_layer_ids=[2])
        embedding = NGramEmbedding(
            args, args.ple_embed_dim, layer_idx=1, ple_layer_index=0
        )
        sizes = np.asarray(embedding.ngram_heads_vocab_sizes)
        offsets = np.asarray(embedding.ngram_heads_offsets)
        self.assertEqual(int(sizes.sum()), self.REACHABLE_ROWS)
        self.assertEqual(
            embedding.ngram_embedding.vocab_size, self.TOTAL_ROWS
        )
        self.assertEqual(self.TOTAL_ROWS - self.REACHABLE_ROWS, 90)
        # Per-head ranges tile [0, reachable) exactly; the maximum
        # emittable id is reachable - 1, so the 90 padded rows can never
        # be gathered.
        self.assertEqual(int(offsets[-1] + sizes[-1]), self.REACHABLE_ROWS)
        rng = np.random.default_rng(1)
        tokens = mx.array(
            rng.integers(0, args.vocab_size, size=(2, 257)), dtype=mx.int64
        )
        ids = embedding._ngram_ids_numpy(tokens, None)
        self.assertGreaterEqual(int(ids.min()), 0)
        self.assertLess(int(ids.max()), self.REACHABLE_ROWS)


class TestQwen4PleNvmeDequantEdgeMatrix(unittest.TestCase):
    """q x scale x bias edge matrix against the default-stream kernel.

    The numpy path is IEEE float32 with one RTNE rounding to bfloat16. The
    GPU kernel diverges only outside the checkpoint's value domain:
      - results that are bfloat16-subnormal (GPU flushes to signed zero),
      - q*scale overflowing float32 (the GPU fma avoids the intermediate
        overflow that numpy's separate multiply hits).
    Real q4/g32 checkpoint tables contain neither (the adversarial review
    sampled real shards and found no subnormal or nonfinite scales/biases),
    and nonfinite inputs are documented as out of domain. This test pins
    exact equality on the supported domain and pins the divergence to
    exactly those two windows so a kernel change cannot silently widen it.
    """

    SCALES = (0x3F80, 0xBF80, 0x4329, 0x0080, 0x0001, 0x8001, 0x7F00,
              0xFF00, 0x7F7F)
    BIASES = (0x0000, 0x8000, 0x3B80, 0xBB80, 0xC4BE, 0x7F7F, 0xFF7F)
    SUBNORMAL_SCALES = {0x0001, 0x8001}
    OVERFLOW_PAIRS = {(0x7F00, 0xFF7F), (0xFF00, 0x7F7F), (0x7F7F, 0xFF7F)}

    def test_edge_matrix_matches_default_stream_on_supported_domain(self):
        pairs = [(s, b) for s in self.SCALES for b in self.BIASES]
        n = len(pairs)
        # Two alternating words put every q in 0..15 in every group.
        words = np.tile(
            np.array([0x76543210, 0xFEDCBA98] * 10, dtype=np.uint32), (n, 1)
        )
        s_bits = np.array([[s] * 5 for s, _ in pairs], dtype=np.uint16)
        b_bits = np.array([[b] * 5 for _, b in pairs], dtype=np.uint16)
        rows = np.concatenate(
            [
                words.view(np.uint8).reshape(n, 80),
                s_bits.view(np.uint8).reshape(n, 10),
                b_bits.view(np.uint8).reshape(n, 10),
            ],
            axis=1,
        )
        with np.errstate(over="ignore"):
            ours = dequant_rows_numpy(rows, 160)
        reference = mx.dequantize(
            mx.array(words),
            mx.array(s_bits).view(mx.bfloat16),
            mx.array(b_bits).view(mx.bfloat16),
            group_size=32,
            bits=4,
            mode="affine",
        )
        mx.eval(reference)
        reference_bits = np.asarray(reference.view(mx.uint16))

        divergent = np.zeros(n, dtype=bool)
        for index, (s, b) in enumerate(pairs):
            divergent[index] = (
                s in self.SUBNORMAL_SCALES or (s, b) in self.OVERFLOW_PAIRS
            )
        np.testing.assert_array_equal(
            reference_bits[~divergent], ours[~divergent]
        )
        # Window rows MAY diverge, and any divergence must follow the
        # documented pattern: for subnormal scales the GPU value on a
        # mismatched element is a flush to signed zero. (Overflow-window
        # rows differ via fma vs separate multiply; no element pattern is
        # asserted beyond membership in the window.)
        for index in np.flatnonzero(divergent):
            mismatch = reference_bits[index] != ours[index]
            if pairs[index][0] in self.SUBNORMAL_SCALES and mismatch.any():
                gpu = reference_bits[index][mismatch]
                self.assertTrue(
                    np.isin(gpu, (0x0000, 0x8000)).all(),
                    "GPU subnormal handling changed: no longer a flush to "
                    "signed zero",
                )

    def test_bf16_halfway_ties_round_to_even_both_parities_and_signs(self):
        # q=1 rows: value = scale + bias. 1 + 2^-8 and 1 + 3*2^-8 are exact
        # halfway points between bfloat16 neighbours; RTNE keeps the even
        # mantissa (0x3F80) and rounds up to it (0x3F82) respectively.
        cases = [
            (0x3F80, 0x3B80, 0x3F80),  # 1.00390625 -> 1.0 (down to even)
            (0x3F80, 0x3C40, 0x3F82),  # 1.01171875 -> 1.015625 (up to even)
            (0xBF80, 0xBB80, 0xBF80),  # -1.00390625 -> -1.0
            (0xBF80, 0xBC40, 0xBF82),  # -1.01171875 -> -1.015625
        ]
        word = np.uint32(0x11111111)  # q = 1 everywhere
        for s_val, b_val, expected in cases:
            w = mx.array(np.full((1, 20), word, dtype=np.uint32))
            s = mx.array(np.full((1, 5), s_val, dtype=np.uint16)).view(
                mx.bfloat16
            )
            b = mx.array(np.full((1, 5), b_val, dtype=np.uint16)).view(
                mx.bfloat16
            )
            reference = mx.dequantize(
                w, s, b, group_size=32, bits=4, mode="affine"
            )
            mx.eval(reference)
            rows = np.concatenate(
                [
                    np.asarray(w).view(np.uint8).reshape(1, 80),
                    np.full((1, 5), s_val, dtype=np.uint16)
                    .view(np.uint8)
                    .reshape(1, 10),
                    np.full((1, 5), b_val, dtype=np.uint16)
                    .view(np.uint8)
                    .reshape(1, 10),
                ],
                axis=1,
            )
            ours = dequant_rows_numpy(rows, 160)
            self.assertTrue((ours == expected).all(), hex(s_val))
            np.testing.assert_array_equal(
                ours, np.asarray(reference.view(mx.uint16))
            )


if __name__ == "__main__":
    unittest.main()
