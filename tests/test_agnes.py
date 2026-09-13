import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Generation owns a module-level execution stream, so select CPU before
# importing it or any model code used by these tiny correctness tests.
mx.set_default_device(mx.cpu)

from mlx_lm.apc import APCKey, AutomaticPrefixCache, AutomaticPrefixCacheV2
from mlx_lm.cache_planes import CachePlaneKind
from mlx_lm.generate import generate_step, prompt_lookup_generate_step
from mlx_lm.models import agnes, qwen4_fused_gdn
from mlx_lm.models.cache import KVCache, record_state_checkpoints
from mlx_lm.models.qwen3_5 import fuse_gated_delta_net_projections
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated
from mlx_lm.prompt_lookup import HybridStats
from mlx_lm.sample_utils import make_sampler
from mlx_lm.server import ResponseGenerator


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class FakeGdnCache:
    def __init__(self, conv_state, recurrent_state, *, speculating=False):
        self.cache = [conv_state, recurrent_state]
        self.speculating = speculating
        self.advanced = 0

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    def rollback_spans(self, width, mask):
        return ()

    def advance(self, amount):
        self.advanced += amount


class TestAgnes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.cpu)

    def make_args(self, **overrides):
        text_config = {
            "model_type": "agnes_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 24,
            "parallel_ffn_intermediate_size": 8,
            "vocab_size": 32,
            "layer_types": [agnes.LAYER_DELTA, agnes.LAYER_GLOBAL],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 64,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
        }
        text_config.update(overrides)
        return agnes.ModelArgs.from_dict(
            {"model_type": "agnes", "text_config": text_config}
        )

    def test_exact_layer_plan_parallel_ffn_and_cached_decode(self):
        model = agnes.Model(self.make_args())
        self.assertIsInstance(model.layers[0].delta_attn, agnes.GatedDeltaNet)
        self.assertIsInstance(model.layers[1].global_attn, agnes.Qwen3NextAttention)
        self.assertIsNotNone(model.layers[0].mlp.parallel_ffn)

        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        full = model(tokens)
        cache = model.make_cache()
        steps = [model(tokens[:, i : i + 1], cache=cache) for i in range(4)]
        decoded = mx.concatenate(steps, axis=1)
        mx.eval(full, decoded)
        self.assertTrue(mx.allclose(full, decoded, rtol=2e-4, atol=2e-4))

    def test_apcv2_prefix_hit_miss_fork_and_cold_reuse_logits(self):
        model = agnes.Model(self.make_args())
        self.assertEqual(model.apc_v2_layout, "agnes-hybrid-layer-segments-v1")
        self.assertEqual(
            model.language_model.apc_v2_layout,
            "agnes-hybrid-layer-segments-v1",
        )
        apc = AutomaticPrefixCacheV2(max_size=2, layout_name=model.apc_v2_layout)
        key = APCKey(
            "agnes-tiny",
            revision="test-revision",
            cache_layout_fingerprint=model.apc_v2_layout,
        )
        prefix = mx.array([[1, 2, 3]], dtype=mx.int32)

        miss = apc.lookup(key, prefix[0].tolist())
        self.assertFalse(miss.hit)
        self.assertEqual(miss.miss_reason, "no_compatible_prefix")

        source = model.make_cache()
        source_logits = model(prefix, cache=source)
        mx.eval(source_logits, *[cache.state for cache in source])
        capabilities = apc.store(key, prefix[0].tolist(), source)
        self.assertEqual(capabilities.topology, "checkpointed_hybrid")

        incompatible = apc.lookup(
            APCKey(
                "agnes-tiny",
                revision="other-revision",
                cache_layout_fingerprint=model.apc_v2_layout,
            ),
            [1, 2, 3, 4],
        )
        self.assertFalse(incompatible.hit)

        suffix_a = mx.array([[4, 5]], dtype=mx.int32)
        suffix_b = mx.array([[6, 7]], dtype=mx.int32)
        hit = apc.lookup(key, [1, 2, 3, 4, 5])
        self.assertTrue(hit.hit)
        self.assertEqual(hit.cached_tokens, 3)
        self.assertEqual(hit.remaining_tokens, [4, 5])
        sibling = hit.cache.fork()

        reused_a = model(suffix_a, cache=hit.cache)
        reused_b = model(suffix_b, cache=sibling)
        cold_a_cache = model.make_cache()
        cold_b_cache = model.make_cache()
        model(prefix, cache=cold_a_cache)
        model(prefix, cache=cold_b_cache)
        cold_a = model(suffix_a, cache=cold_a_cache)
        cold_b = model(suffix_b, cache=cold_b_cache)
        mx.eval(reused_a, reused_b, cold_a, cold_b)

        # Compare the same prefill/decode split. A monolithic forward can use
        # different kernels and is not the cache-reuse fidelity authority.
        self.assertTrue(mx.array_equal(reused_a, cold_a))
        self.assertTrue(mx.array_equal(reused_b, cold_b))

        stats = apc.apc_stats
        self.assertEqual(stats["version"], 2)
        self.assertEqual(stats["layout_name"], model.apc_v2_layout)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)
        self.assertEqual(stats["cached_tokens"], 3)
        self.assertEqual(stats["layer_segments"]["fallback_entries"], 0)
        self.assertEqual(
            stats["layer_segments"]["by_plane"]["gdn_recurrent"]["layers"],
            1,
        )
        self.assertEqual(
            stats["layer_segments"]["by_plane"]["attention_kv"]["layers"],
            1,
        )
        self.assertEqual(
            stats["cow"]["planes"]["gdn_recurrent"]["materializations"],
            2,
        )
        self.assertEqual(
            stats["cow"]["planes"]["attention_kv"]["materializations"],
            2,
        )
        hit.cache.close()
        sibling.close()

    def test_server_selects_apcv2_for_native_agnes(self):
        response = ResponseGenerator.__new__(ResponseGenerator)
        response.prompt_cache = AutomaticPrefixCache(max_size=3, max_bytes=123456)
        response._cache_capsule_pool = None
        model = agnes.Model(self.make_args())

        response._configure_apc_for_model(model)

        self.assertIsInstance(response.prompt_cache, AutomaticPrefixCacheV2)
        self.assertEqual(response.prompt_cache.layout_name, model.apc_v2_layout)
        self.assertEqual(response.prompt_cache.max_size, 3)
        self.assertEqual(response.prompt_cache.max_bytes, 123456)
        response._cache_capsule_pool.close()

    def test_existing_gdn_projection_fusion_accepts_agnes_explicitly(self):
        model = agnes.Model(
            self.make_args(
                hidden_size=32,
                intermediate_size=64,
                parallel_ffn_intermediate_size=32,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
            )
        )
        projection_names = (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        )
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda path, module: any(
                path.endswith(name) for name in projection_names
            ),
        )
        mx.eval(model.parameters())

        # The inherited optimization remains default-off and reports the
        # number of real Agnes delta layers it verified and rewrote.
        self.assertEqual(fuse_gated_delta_net_projections(model), 0)
        self.assertFalse(hasattr(model.layers[0].delta_attn, "in_proj_fused"))
        self.assertEqual(fuse_gated_delta_net_projections(model, enabled=True), 1)
        self.assertTrue(hasattr(model.layers[0].delta_attn, "in_proj_fused"))

    def test_fused_gdn_admission_keeps_architecture_gate_contracts_separate(self):
        dtype = mx.bfloat16
        values = {
            "qkv": FakeArray((1, 1, 10240), dtype),
            "z": FakeArray((1, 1, 6144), dtype),
            "b": FakeArray((1, 1, 48), dtype),
            "a": FakeArray((1, 1, 48), dtype),
            "conv_state": FakeArray((1, 3, 10240), dtype),
            "recurrent_state": FakeArray((1, 48, 128, 128), mx.float32),
            "conv_weight": FakeArray((10240, 4, 1), dtype),
            "A_log": FakeArray((48,), mx.float32),
            "dt_bias": FakeArray((48,), dtype),
            "norm_weight": FakeArray((128,), dtype),
            "mask": None,
            "spans": (),
            "speculating": False,
            "training": False,
            "sharded": False,
            "num_key_heads": 16,
            "num_value_heads": 48,
            "key_head_dim": 128,
            "value_head_dim": 128,
            "conv_kernel": 4,
        }
        agnes_gate = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values, gate_activation="swish", architecture="agnes"
        )
        self.assertTrue(agnes_gate.accepted, agnes_gate.reason)
        self.assertFalse(
            qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
                **values, gate_activation="swish"
            ).accepted
        )
        self.assertFalse(
            qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
                **values, gate_activation="sigmoid", architecture="agnes"
            ).accepted
        )

    def test_fused_gdn_kernel_selects_agnes_numerical_contract_without_dispatch(self):
        calls = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return [
                FakeArray(shape, dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"]
                )
            ]

        dtype = mx.bfloat16
        values = [
            FakeArray((1, 1, 10240), dtype),
            FakeArray((1, 1, 6144), dtype),
            FakeArray((1, 1, 48), dtype),
            FakeArray((1, 1, 48), dtype),
            FakeArray((1, 3, 10240), dtype),
            FakeArray((10240, 4, 1), dtype),
            FakeArray((48,), mx.float32),
            FakeArray((48,), dtype),
            FakeArray((1, 48, 128, 128), mx.float32),
            FakeArray((128,), dtype),
            1.0e-6,
        ]
        with patch.object(qwen4_fused_gdn, "_kernel", return_value=fake_kernel):
            qwen4_fused_gdn.qwen4_fused_gdn_decode(
                *values, threadgroup_y=8, architecture="agnes"
            )
            qwen4_fused_gdn.qwen4_fused_gdn_decode(*values, threadgroup_y=8)

        self.assertIn(("AGNES_NUMERICS", 1), calls[0]["template"])
        self.assertIn(("AGNES_NUMERICS", 0), calls[1]["template"])
        self.assertIn("mlx_sigmoid_fast<float>(zv)", qwen4_fused_gdn._SOURCE)
        self.assertIn("1.0e-6f / float(DK)", qwen4_fused_gdn._SOURCE)

    def test_fused_gdn_integration_defaults_on_and_updates_cache_after_success(self):
        layer = agnes.Model(self.make_args()).layers[0].delta_attn
        layer.eval()
        self.assertTrue(layer.fused_gdn_decode)
        self.assertFalse(layer.set_fused_gdn_decode(False))
        self.assertTrue(layer.set_fused_gdn_decode(True))

        qkv = mx.zeros((1, 1, 16), dtype=mx.bfloat16)
        z = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        gates = mx.zeros((1, 1, 2), dtype=mx.bfloat16)
        original_conv = object()
        original_state = object()
        cache = FakeGdnCache(original_conv, original_state)
        next_conv = object()
        next_state = object()
        fused_out = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")

        with (
            patch.object(agnes, "admit_qwen4_fused_gdn_decode", return_value=accepted),
            patch.object(agnes, "fused_gdn_runtime_supported", return_value=True),
            patch.object(agnes, "probe_agnes_fused_gdn_decode", return_value=8),
            patch.object(
                agnes,
                "qwen4_fused_gdn_decode",
                return_value=(fused_out, next_conv, next_state),
            ) as execute,
        ):
            output = layer._try_fused_decode(qkv, z, gates, gates, None, cache)

        mx.eval(output)
        self.assertEqual(output.shape, (1, 1, 16))
        self.assertIs(cache[0], next_conv)
        self.assertIs(cache[1], next_state)
        self.assertEqual(cache.advanced, 1)
        self.assertEqual(layer.fused_gdn_decode_calls, 1)
        self.assertEqual(execute.call_args.kwargs["architecture"], "agnes")

    def test_fused_gdn_declines_before_runtime_and_bounds_reason_counters(self):
        layer = agnes.Model(self.make_args()).layers[0].delta_attn
        layer.eval()
        layer.set_fused_gdn_decode(True)
        qkv = mx.zeros((1, 1, 16), dtype=mx.bfloat16)
        z = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        gates = mx.zeros((1, 1, 2), dtype=mx.bfloat16)
        cache = FakeGdnCache(None, None)
        with patch.object(agnes, "fused_gdn_runtime_supported") as runtime:
            self.assertIsNone(
                layer._try_fused_decode(qkv, z, gates, gates, None, cache)
            )
        runtime.assert_not_called()
        self.assertEqual(layer.fused_gdn_decode_last_fallback, "uninitialized cache")

        for index in range(32):
            layer._fused_gdn_fallback(f"reason {index}")
        self.assertLessEqual(
            len(layer.fused_gdn_decode_fallback_reasons),
            agnes._FUSED_GDN_FALLBACK_REASON_LIMIT + 1,
        )

    def test_fused_gdn_cpu_runtime_decline_is_output_exact(self):
        model = agnes.Model(self.make_args())
        model.eval()
        prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
        token = mx.array([[4]], dtype=mx.int32)
        stock_cache = model.make_cache()
        fused_cache = model.make_cache()
        self.assertEqual(agnes.set_agnes_fused_gdn_decode(model, False), 1)
        model(prefix, cache=stock_cache)
        model(prefix, cache=fused_cache)

        stock = model(token, cache=stock_cache)
        self.assertEqual(agnes.set_agnes_fused_gdn_decode(model, True), 1)
        accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")
        with patch.object(agnes, "admit_qwen4_fused_gdn_decode", return_value=accepted):
            candidate = model(token, cache=fused_cache)
        mx.eval(stock, candidate)

        self.assertTrue(mx.array_equal(stock, candidate))
        stats = agnes.agnes_fused_gdn_stats(model)
        self.assertEqual(stats["enabled_layers"], 1)
        self.assertEqual(stats["fused_calls"], 0)
        self.assertEqual(stats["fallbacks"], 1)
        self.assertEqual(stats["reasons"], {"Metal runtime unavailable": 1})

    def test_apcv2_checkpoint_trim_and_atomic_invalidation(self):
        model = agnes.Model(self.make_args())
        prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
        continuation = mx.array([[4, 5]], dtype=mx.int32)
        source = model.make_cache()
        prefix_logits = model(prefix, cache=source)
        mx.eval(prefix_logits, *[cache.state for cache in source])
        record_state_checkpoints(source, [3], force=True)
        long_logits = model(continuation, cache=source)
        mx.eval(long_logits, *[cache.state for cache in source])

        apc = AutomaticPrefixCacheV2(max_size=2, layout_name=model.apc_v2_layout)
        key = APCKey("agnes-trim")
        stored_tokens = [1, 2, 3, 4, 5]
        apc.store(key, stored_tokens, source)

        # The requested branch diverges inside the stored entry. APCv2 must
        # restore the GDN checkpoint and trim attention K/V to the same point.
        hit = apc.lookup(key, [1, 2, 3, 9])
        self.assertTrue(hit.hit)
        self.assertEqual(hit.hit_kind, "prefix")
        self.assertEqual(hit.cached_tokens, 3)
        self.assertEqual(hit.remaining_tokens, [9])
        self.assertEqual(
            hit.segment_manifest["by_plane"]["gdn_recurrent"]["segments"],
            1,
        )
        self.assertEqual(
            hit.segment_manifest["by_plane"]["attention_kv"]["segments"],
            1,
        )

        reused = model(
            mx.array([hit.remaining_tokens], dtype=mx.int32), cache=hit.cache
        )
        cold_cache = model.make_cache()
        model(prefix, cache=cold_cache)
        cold = model(mx.array([[9]], dtype=mx.int32), cache=cold_cache)
        mx.eval(reused, cold)
        self.assertTrue(mx.array_equal(reused, cold))
        hit.cache.close()

        entry = apc._trie.get(key, stored_tokens)
        owner = entry.prompt_cache.cow_owner
        recurrent_segment = next(
            segment
            for segment in owner.segment_manifest.segments
            if segment.key.kind == CachePlaneKind.GDN_RECURRENT
        )
        self.assertTrue(
            owner.invalidate_segment(recurrent_segment.key, "test-stale-gdn")
        )
        invalidated = apc.lookup(key, [1, 2, 3, 10])
        self.assertFalse(invalidated.hit)
        self.assertEqual(invalidated.miss_reason, "stale_cow_generation")

    def test_apcv2_reused_prefix_composes_with_prompt_lookup(self):
        model = agnes.Model(self.make_args(max_position_embeddings=256))
        model.eval()
        mx.eval(model.parameters())
        history = list(range(16)) * 4
        sampler = make_sampler(temp=0.0)

        plain_cache = model.make_cache()
        plain = list(
            generate_step(
                mx.array(history),
                model,
                max_tokens=8,
                sampler=sampler,
                prompt_cache=plain_cache,
                prefill_step_size=2048,
            )
        )
        plain_tokens = [int(item[0]) for item in plain]

        class ReplayProposer:
            def observe(self, token):
                pass

            def propose(self, sequence, max_span, prompt_length):
                generated = len(sequence) - prompt_length
                return plain_tokens[generated : generated + max_span]

        cached_prefix = history[:48]
        source = model.make_cache()
        prefix_logits = model(mx.array(cached_prefix)[None], cache=source)
        mx.eval(prefix_logits, *[cache.state for cache in source])
        apc = AutomaticPrefixCacheV2(max_size=2, layout_name=model.apc_v2_layout)
        key = APCKey("agnes-pld-composition")
        apc.store(key, cached_prefix, source)
        hit = apc.lookup(key, history)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.cached_tokens, 48)

        pld_stats = HybridStats()
        pld = list(
            prompt_lookup_generate_step(
                mx.array(hit.remaining_tokens),
                model,
                max_tokens=8,
                sampler=sampler,
                prompt_cache=hit.cache,
                history_prompt=mx.array(history),
                backend=ReplayProposer(),
                num_draft=4,
                ngram_max=3,
                ngram_min=1,
                adaptive=False,
                warmup=48,
                gate=0.12,
                rate_gate=False,
                stats=pld_stats,
            )
        )
        self.assertEqual([int(item[0]) for item in pld], plain_tokens)
        self.assertGreater(pld_stats.retrieval_cycles, 0)
        self.assertGreater(pld_stats.retrieval_accepted, 0)

        # Batched PLD verification can differ from sequential decode by small
        # floating-point ties, so state fidelity uses the same tolerance as
        # the model reference harness while token equivalence stays exact.
        for reused_cache, cold_cache in zip(hit.cache, plain_cache):
            if hasattr(reused_cache, "keys"):
                self.assertEqual(reused_cache.offset, cold_cache.offset)
                reused_state = reused_cache.keys_and_values()
                cold_state = cold_cache.keys_and_values()
            else:
                reused_state = reused_cache.cache
                cold_state = cold_cache.cache
            for reused_array, cold_array in zip(reused_state, cold_state):
                mx.eval(reused_array, cold_array)
                self.assertTrue(
                    mx.allclose(reused_array, cold_array, rtol=2e-4, atol=2e-4)
                )

        counters = apc.apc_stats
        self.assertEqual(counters["hits"], 1)
        self.assertEqual(counters["cached_tokens"], 48)
        self.assertEqual(
            counters["cow"]["planes"]["gdn_recurrent"]["materializations"],
            1,
        )
        self.assertEqual(
            counters["cow"]["planes"]["attention_kv"]["materializations"],
            1,
        )
        hit.cache.close()

    def test_prompt_lookup_rejections_remain_exact_beyond_rollback_window(self):
        model = agnes.Model(self.make_args(max_position_embeddings=512))
        model.eval()
        mx.eval(model.parameters())
        prompt = list(range(16)) * 4
        sampler = make_sampler(temp=0.0)

        plain_cache = model.make_cache()
        plain = list(
            generate_step(
                mx.array(prompt),
                model,
                max_tokens=160,
                sampler=sampler,
                prompt_cache=plain_cache,
                prefill_step_size=2048,
            )
        )
        plain_tokens = [int(item[0]) for item in plain]

        class PartialRejectProposer:
            def observe(self, token):
                pass

            def propose(self, sequence, max_span, prompt_length):
                generated = len(sequence) - prompt_length
                proposal = plain_tokens[generated : generated + max_span]
                if len(proposal) > 1:
                    proposal[1] = (proposal[1] + 1) % 32
                return proposal

        pld_cache = model.make_cache()
        pld_stats = HybridStats()
        pld = list(
            prompt_lookup_generate_step(
                mx.array(prompt),
                model,
                max_tokens=160,
                sampler=sampler,
                prompt_cache=pld_cache,
                backend=PartialRejectProposer(),
                num_draft=4,
                adaptive=False,
                rate_gate=False,
                stats=pld_stats,
            )
        )
        self.assertEqual([int(item[0]) for item in pld], plain_tokens)
        self.assertGreater(pld_stats.retrieval_accepted, 0)
        self.assertGreater(
            pld_stats.retrieval_proposed, pld_stats.retrieval_accepted
        )

        for layer, (actual, reference) in enumerate(zip(pld_cache, plain_cache)):
            if isinstance(actual, KVCache):
                self.assertEqual(actual.offset, reference.offset, f"layer {layer}")
                actual_state = actual.keys_and_values()
                reference_state = reference.keys_and_values()
            else:
                actual_state = actual.cache
                reference_state = reference.cache
            for slot, (a, b) in enumerate(zip(actual_state, reference_state)):
                mx.eval(a, b)
                self.assertTrue(
                    mx.allclose(a, b, rtol=2e-4, atol=2e-4),
                    f"layer {layer} slot {slot}",
                )

    def test_sanitize_raw_then_native_does_not_shift_norm_twice(self):
        raw_model = agnes.Model(self.make_args(mtp_num_hidden_layers=1))
        norm_key = "model.language_model.layers.0.input_layernorm.weight"
        conv_key = "model.language_model.layers.0.delta_attn.conv1d.weight"
        base = mx.arange(16, dtype=mx.float32)
        raw_conv = mx.zeros((16, 1, 4), dtype=mx.float32)
        converted = raw_model.sanitize(
            {
                norm_key: base,
                conv_key: raw_conv,
                "mtp.fc.weight": mx.zeros((16, 32)),
                "model.visual.stub": mx.zeros((1,)),
            }
        )

        mlx_norm_key = "language_model.model.layers.0.input_layernorm.weight"
        mlx_conv_key = "language_model.model.layers.0.delta_attn.conv1d.weight"
        self.assertTrue(mx.array_equal(converted[mlx_norm_key], base + 1.0))
        self.assertEqual(converted[mlx_conv_key].shape, (16, 4, 1))
        self.assertFalse(any("mtp." in key for key in converted))
        self.assertFalse(any("visual" in key for key in converted))

        native_model = agnes.Model(self.make_args(mtp_num_hidden_layers=1))
        loaded = native_model.sanitize(converted.copy())
        self.assertTrue(mx.array_equal(loaded[mlx_norm_key], base + 1.0))

    def test_bfloat16_gated_norm_preserves_reference_operation_order(self):
        norm = Qwen3NextRMSNormGated(8, 1e-6)
        norm.weight = (mx.arange(8) / 16 + 0.75).astype(mx.bfloat16)
        hidden = (mx.arange(16).reshape(1, 2, 8) / 13 - 0.4).astype(mx.bfloat16)
        gate = (mx.arange(16).reshape(1, 2, 8) / 17 - 0.3).astype(mx.bfloat16)
        actual = norm(hidden, gate)
        expected = (
            mx.fast.rms_norm(hidden, norm.weight, norm.eps).astype(mx.float32)
            * nn.silu(gate.astype(mx.float32))
        ).astype(mx.bfloat16)
        mx.eval(actual, expected)
        self.assertTrue(mx.array_equal(actual, expected))

    def test_native_vlm_weight_names_load_strictly(self):
        model = agnes.Model(self.make_args())
        native = dict(tree_flatten(model.parameters()))
        native["vision_tower.stub"] = mx.zeros((1,))

        reloaded = agnes.Model(self.make_args())
        sanitized = reloaded.sanitize(native)
        reloaded.load_weights(list(sanitized.items()), strict=True)
        mx.eval(reloaded.parameters())

    def test_invalid_layer_plan_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "layer_types"):
            agnes.Model(
                self.make_args(layer_types=["full_attention", agnes.LAYER_DELTA])
            )

    def test_pinned_public_config_parses_without_constructing_full_model(self):
        layer_types = [
            agnes.LAYER_GLOBAL if (i + 1) % 4 == 0 else agnes.LAYER_DELTA
            for i in range(72)
        ]
        args = agnes.ModelArgs.from_dict(
            {
                "model_type": "agnes",
                "text_config": {
                    "model_type": "agnes_text",
                    "hidden_size": 5120,
                    "num_hidden_layers": 72,
                    "intermediate_size": 17408,
                    "parallel_ffn_intermediate_size": 2048,
                    "vocab_size": 248320,
                    "layer_types": layer_types,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                },
            }
        )
        text = agnes.TextModelArgs.from_dict(args.text_config)
        self.assertEqual(args.model_type, "agnes")
        self.assertEqual(text.num_hidden_layers, 72)
        self.assertEqual(text.parallel_ffn_intermediate_size, 2048)
        self.assertEqual(text.layer_types.count(agnes.LAYER_GLOBAL), 18)
        self.assertEqual(text.layer_types.count(agnes.LAYER_DELTA), 54)


if __name__ == "__main__":
    unittest.main()
