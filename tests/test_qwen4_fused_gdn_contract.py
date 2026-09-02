import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import qwen4_exp, qwen4_fused_gdn


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeCache:
    def __init__(self, conv_state=None, recurrent_state=None, *, speculating=False):
        self.cache = [conv_state, recurrent_state]
        self.lengths = None
        self.speculating = speculating
        self.advanced = 0

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    def advance(self, amount):
        self.advanced += amount


def production_values(dtype=mx.bfloat16):
    return dict(
        qkv=FakeArray((1, 1, 10240), dtype),
        z=FakeArray((1, 1, 6144), dtype),
        b=FakeArray((1, 1, 48), dtype),
        a=FakeArray((1, 1, 48), dtype),
        conv_state=FakeArray((1, 3, 10240), dtype),
        recurrent_state=FakeArray((1, 48, 128, 128), mx.float32),
        conv_weight=FakeArray((10240, 4, 1), dtype),
        A_log=FakeArray((48,), mx.float32),
        dt_bias=FakeArray((48,), dtype),
        norm_weight=FakeArray((128,), dtype),
    )


def admission(**overrides):
    values = production_values()
    values.update(overrides)
    return qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        mask=None,
        spans=(),
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="sigmoid",
    )


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )


class Identity:
    def __call__(self, value):
        return value


class TestFusedGdnAdmission(unittest.TestCase):
    def test_concurrent_probe_publishes_only_after_initialization(self):
        entered = Event()
        release = Event()
        calls = 0

        def blocking_kernel(*args, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return (object(), object(), object())

        with patch.object(
            qwen4_fused_gdn, "_PROBE_COMPLETE", False
        ), patch.object(
            qwen4_fused_gdn, "_PROBED_THREADGROUP_Y", None
        ), patch.object(
            qwen4_fused_gdn, "_PROBE_LOCK", Lock()
        ), patch.object(
            qwen4_fused_gdn, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_fused_gdn,
            "qwen4_fused_gdn_decode",
            side_effect=blocking_kernel,
        ), patch.object(qwen4_fused_gdn.mx, "eval"), ThreadPoolExecutor(
            max_workers=2
        ) as pool:
            first = pool.submit(
                qwen4_fused_gdn.probe_qwen4_fused_gdn_decode, mx.bfloat16
            )
            self.assertTrue(entered.wait(timeout=5))
            second = pool.submit(
                qwen4_fused_gdn.probe_qwen4_fused_gdn_decode, mx.bfloat16
            )
            release.set()
            self.assertEqual(first.result(timeout=5), 32)
            self.assertEqual(second.result(timeout=5), 32)

        self.assertEqual(calls, 1)

    def test_probe_tries_smaller_threadgroup_after_runtime_error(self):
        successful_outputs = (object(), object(), object())
        with patch.object(
            qwen4_fused_gdn, "_PROBE_COMPLETE", False
        ), patch.object(
            qwen4_fused_gdn, "_PROBED_THREADGROUP_Y", None
        ), patch.object(
            qwen4_fused_gdn, "_PROBE_LOCK", Lock()
        ), patch.object(
            qwen4_fused_gdn, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_fused_gdn,
            "qwen4_fused_gdn_decode",
            side_effect=[RuntimeError("threadgroup resources"), successful_outputs],
        ) as execute, patch.object(qwen4_fused_gdn.mx, "eval"):
            selected = qwen4_fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16)

        self.assertEqual(selected, 16)
        self.assertEqual(
            [call.kwargs["threadgroup_y"] for call in execute.call_args_list],
            [32, 16],
        )

    def test_production_single_token_decode_is_admitted(self):
        result = admission()
        self.assertTrue(result.accepted, result.reason)

    def test_batch_prefill_mask_and_speculation_fall_back(self):
        result = admission(qkv=FakeArray((2, 1, 10240), mx.bfloat16))
        self.assertFalse(result.accepted)
        self.assertIn("qkv shape", result.reason)
        result = admission(qkv=FakeArray((1, 2, 10240), mx.bfloat16))
        self.assertFalse(result.accepted)
        self.assertIn("qkv shape", result.reason)

        values = production_values()
        result = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=object(),
            spans=(),
            speculating=False,
            training=False,
            sharded=False,
            num_key_heads=16,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "masked decode")

        result = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=None,
            spans=(),
            speculating=True,
            training=False,
            sharded=False,
            num_key_heads=16,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "speculative rollback")

    def test_dtype_and_geometry_are_strict(self):
        result = admission(A_log=FakeArray((48,), mx.float16))
        self.assertFalse(result.accepted)
        self.assertIn("A_log", result.reason)

        result = admission(A_log=FakeArray((48,), mx.bfloat16))
        self.assertTrue(result.accepted, result.reason)

        values = production_values()
        result = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=None,
            spans=(),
            speculating=False,
            training=False,
            sharded=False,
            num_key_heads=24,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )
        self.assertFalse(result.accepted)
        self.assertIn("unsupported geometry", result.reason)

    def test_kernel_dispatch_geometry_is_one_threadgroup_per_value_head(self):
        calls = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return [
                FakeArray(shape, dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"]
                )
            ]

        values = production_values()
        with patch.object(qwen4_fused_gdn, "_kernel", return_value=fake_kernel):
            outputs = qwen4_fused_gdn.qwen4_fused_gdn_decode(
                values["qkv"],
                values["z"],
                values["b"],
                values["a"],
                values["conv_state"],
                values["conv_weight"],
                values["A_log"],
                values["dt_bias"],
                values["recurrent_state"],
                values["norm_weight"],
                1.0e-6,
                threadgroup_y=16,
            )
        self.assertEqual(calls[0]["grid"], (32, 16, 48))
        self.assertEqual(calls[0]["threadgroup"], (32, 16, 1))
        self.assertEqual(outputs[0].shape, (1, 1, 6144))
        self.assertEqual(outputs[1].shape, (1, 3, 10240))
        self.assertEqual(outputs[2].shape, (1, 48, 128, 128))
        self.assertIn(("RATIO", 3), calls[0]["template"])
        with self.assertRaisesRegex(ValueError, "unsupported threadgroup_y"):
            qwen4_fused_gdn.qwen4_fused_gdn_decode(
                values["qkv"],
                values["z"],
                values["b"],
                values["a"],
                values["conv_state"],
                values["conv_weight"],
                values["A_log"],
                values["dt_bias"],
                values["recurrent_state"],
                values["norm_weight"],
                1.0e-6,
                threadgroup_y=3,
            )


class TestFusedGdnIntegration(unittest.TestCase):
    def test_resident_switch_preserves_weights(self):
        with patch.object(qwen4_exp, "_FUSED_GDN_DECODE", False):
            layer = qwen4_exp.GatedDeltaNet(tiny_args())
        weight = layer.conv1d.weight
        self.assertEqual(
            qwen4_exp.qwen4_fused_gdn_mode_counts(layer),
            {"stock": 1, "fused": 0, "fused_outproj": 0},
        )
        self.assertEqual(qwen4_exp.set_qwen4_fused_gdn_mode(layer, "fused"), 1)
        self.assertIs(layer.conv1d.weight, weight)
        self.assertEqual(layer.fused_gdn_decode_mode, "fused")
        self.assertEqual(
            qwen4_exp.set_qwen4_fused_gdn_mode(layer, "fused_outproj"), 1
        )
        self.assertIs(layer.conv1d.weight, weight)
        self.assertEqual(layer.fused_gdn_decode_mode, "fused_outproj")
        self.assertEqual(qwen4_exp.set_qwen4_fused_gdn_mode(layer, "stock"), 1)
        self.assertIs(layer.conv1d.weight, weight)
        with self.assertRaisesRegex(ValueError, "unknown fused GDN decode mode"):
            qwen4_exp.set_qwen4_fused_gdn_mode(layer, "bogus")

    def test_uninitialized_and_speculative_cache_do_not_reach_runtime(self):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        values = production_values()
        with patch.object(qwen4_exp, "fused_gdn_runtime_supported") as runtime:
            result = layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, FakeCache()
            )
        self.assertIsNone(result)
        runtime.assert_not_called()
        self.assertEqual(layer.fused_gdn_decode_last_fallback, "uninitialized cache")

        cache = FakeCache(
            values["conv_state"], values["recurrent_state"], speculating=True
        )
        result = layer._try_fused_decode(
            values["qkv"], values["z"], values["b"], values["a"], None, cache
        )
        self.assertIsNone(result)
        self.assertEqual(layer.fused_gdn_decode_last_fallback, "speculative rollback")

    def test_admitted_path_updates_cache_and_counter_without_real_kernel(self):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        layer.out_proj = Identity()
        values = production_values()
        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        fused_out = FakeArray((1, 1, 6144), mx.bfloat16)
        next_conv = object()
        next_state = object()
        accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")
        with patch.object(
            qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted
        ), patch.object(
            qwen4_exp, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8
        ), patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            return_value=(fused_out, next_conv, next_state),
        ) as execute:
            result = layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, cache
            )
        self.assertIs(result, fused_out)
        self.assertIs(cache[0], next_conv)
        self.assertIs(cache[1], next_state)
        self.assertEqual(cache.advanced, 1)
        self.assertEqual(layer.fused_gdn_decode_calls, 1)
        self.assertEqual(execute.call_args.kwargs["threadgroup_y"], 8)

    def test_synchronous_dispatch_failure_preserves_cache(self):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        values = production_values()
        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")
        with patch.object(
            qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted
        ), patch.object(
            qwen4_exp, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8
        ), patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            side_effect=RuntimeError("dispatch rejected"),
        ):
            result = layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, cache
            )

        self.assertIsNone(result)
        self.assertIs(cache[0], values["conv_state"])
        self.assertIs(cache[1], values["recurrent_state"])
        self.assertEqual(cache.advanced, 0)
        self.assertEqual(
            layer.fused_gdn_decode_last_fallback,
            "Metal kernel dispatch failed: RuntimeError",
        )


def ragged_cache(values, lengths, *, left_padding=None, speculating=False):
    """A Qwen4 state cache prepared the way the production ragged engine does.

    ``mlx_backend._prepare_group`` calls ``prepare(lengths=..., ...)`` on every
    state cache of every step -- a one-lane plain decode step included -- so
    ``lengths`` is stamped on the slab that reaches ``_try_fused_decode``.
    """
    cache = qwen4_exp.Qwen4ArraysCache(4, left_padding=left_padding)
    cache.cache[0] = values["conv_state"]
    cache.cache[1] = values["recurrent_state"]
    if speculating:
        cache.start_speculation()
    cache.prepare(lengths=lengths)
    return cache


class TestRaggedEngineDecodeAdmission(unittest.TestCase):
    """The deployed decline of 2026-09-02, reproduced and fixed on CPU."""

    def _try(self, cache, values=None):
        values = values or production_values()
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        layer.out_proj = Identity()
        fused_out = FakeArray((1, 1, 6144), mx.bfloat16)
        with patch.object(
            qwen4_exp, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8
        ), patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            return_value=(fused_out, object(), object()),
        ):
            result = layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, cache
            )
        return layer, result

    def _admit(self, cache, values=None):
        """Run the real admission on spans the prepared cache itself derived."""
        values = values or production_values()
        return qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=None,
            spans=cache.rollback_spans(1, None),
            speculating=bool(cache.speculating),
            training=False,
            sharded=False,
            num_key_heads=16,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )

    def test_fully_valid_single_lane_with_lengths_is_admitted(self):
        values = production_values()
        cache = ragged_cache(values, [1])
        # Before this fix the engine refused every such slab outright with
        # "ragged cache lengths", which is what the deployed run measured.
        self.assertIsNotNone(cache.lengths)
        result = self._admit(cache, values)
        self.assertTrue(result.accepted, result.reason)
        # ... and the layer's span gate now passes it through to geometry.
        layer, _ = self._try(cache, values)
        self.assertEqual(
            layer.fused_gdn_decode_fallback_reasons,
            {"unsupported geometry (1, 2, 64, 64, 4)": 1},
        )

    def test_a_deep_lane_is_still_one_step_of_span(self):
        # ``rollback_spans`` clamps the row's length to the slab width, so a
        # lane 4,000 tokens deep still describes a fully valid one-token step.
        values = production_values()
        result = self._admit(ragged_cache(values, [4000]), values)
        self.assertTrue(result.accepted, result.reason)

    def test_unprepared_cache_is_still_admitted(self):
        values = production_values()
        cache = qwen4_exp.Qwen4ArraysCache(4)
        cache.cache[0] = values["conv_state"]
        cache.cache[1] = values["recurrent_state"]
        self.assertEqual(cache.rollback_spans(1, None), ())
        self.assertTrue(self._admit(cache, values).accepted)

    def test_admitted_span_means_an_all_ones_mask(self):
        """The safety argument for ignoring the mask, checked not asserted.

        The kernel reads no mask. Admitting a slab whose length metadata is
        present is only exact if the mask the cache derives from that same
        metadata is all ones -- which is what a span equal to the slab width
        means, and what a shorter span does not.
        """
        values = production_values()
        for lengths, admitted in (([1], True), ([4000], True), ([0], False)):
            cache = ragged_cache(values, lengths)
            mask = cache.make_mask(1)
            self.assertEqual(
                bool(mx.all(mask).item()), admitted, f"lengths={lengths}"
            )
            self.assertEqual(
                self._admit(cache, values).accepted, admitted, f"lengths={lengths}"
            )

    def test_right_padded_lane_declines(self):
        values = production_values()
        layer, result = self._try(ragged_cache(values, [0]), values)
        self.assertIsNone(result)
        self.assertEqual(
            layer.fused_gdn_decode_fallback_reasons,
            {"padded rollback geometry": 1},
        )

    def test_batched_lengths_decline(self):
        values = production_values()
        layer, result = self._try(ragged_cache(values, [1, 1]), values)
        self.assertIsNone(result)
        self.assertEqual(
            layer.fused_gdn_decode_fallback_reasons,
            {"padded rollback geometry": 1},
        )

    def test_left_padded_lane_declines_as_undescribable(self):
        values = production_values()
        cache = ragged_cache(values, [1], left_padding=[2])
        layer, result = self._try(cache, values)
        self.assertIsNone(result)
        self.assertEqual(
            layer.fused_gdn_decode_fallback_reasons,
            {"rollback geometry not describable": 1},
        )

    def test_masked_step_without_metadata_declines(self):
        values = production_values()
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        result = layer._try_fused_decode(
            values["qkv"], values["z"], values["b"], values["a"], object(), cache
        )
        self.assertIsNone(result)
        self.assertEqual(layer.fused_gdn_decode_last_fallback, "masked decode")

    def test_speculating_single_token_step_still_declines(self):
        values = production_values()
        cache = ragged_cache(values, [1], speculating=True)
        layer, result = self._try(cache, values)
        self.assertIsNone(result)
        self.assertEqual(
            layer.fused_gdn_decode_fallback_reasons, {"speculative rollback": 1}
        )

    def test_decode_and_verify_share_one_span_predicate(self):
        for spans, mask, width, expected in (
            (None, None, 1, "rollback geometry not describable"),
            ((), object(), 1, None),
            ([2], None, 1, "padded rollback geometry"),
            ([1, 1], None, 1, "padded rollback geometry"),
            ((), None, 1, None),
            ([3], None, 3, None),
        ):
            decode = qwen4_fused_gdn.admit_rollback_span(
                spans, mask, width, masked_reason="masked decode"
            )
            verify = qwen4_fused_gdn.admit_rollback_span(
                spans, mask, width, masked_reason="masked verify"
            )
            if expected is None and mask is None:
                self.assertIsNone(decode, spans)
                self.assertIsNone(verify, spans)
            elif expected is None:
                self.assertEqual(decode.reason, "masked decode")
                self.assertEqual(verify.reason, "masked verify")
            else:
                self.assertEqual(decode.reason, expected, spans)
                self.assertEqual(verify.reason, expected, spans)


class TestDecodeFallbackReasonHistogram(unittest.TestCase):
    def test_histogram_is_durable_across_a_later_success(self):
        values = production_values()
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        layer.out_proj = Identity()
        layer._try_fused_decode(
            values["qkv"], values["z"], values["b"], values["a"], None, FakeCache()
        )
        accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")
        with patch.object(
            qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted
        ), patch.object(
            qwen4_exp, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8
        ), patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            return_value=(FakeArray((1, 1, 6144), mx.bfloat16), object(), object()),
        ):
            layer._try_fused_decode(
                values["qkv"],
                values["z"],
                values["b"],
                values["a"],
                None,
                FakeCache(values["conv_state"], values["recurrent_state"]),
            )
        # The deployed smoke reported 2,592 fallbacks with an EMPTY
        # last_fallbacks map, because a later success clears the snapshot.
        self.assertIsNone(layer.fused_gdn_decode_last_fallback)
        self.assertEqual(
            layer.fused_gdn_decode_fallback_reasons, {"uninitialized cache": 1}
        )

    def test_key_set_is_bounded(self):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        for index in range(qwen4_exp._DECODE_FALLBACK_REASON_LIMIT + 8):
            layer._fused_gdn_fallback(f"reason {index}")
        reasons = layer.fused_gdn_decode_fallback_reasons
        self.assertEqual(len(reasons), qwen4_exp._DECODE_FALLBACK_REASON_LIMIT + 1)
        self.assertEqual(reasons["other"], 8)

    def test_stats_aggregate_and_reset_the_histogram(self):
        model = SimpleNamespace()
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        other = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        other.eval()
        layer._fused_gdn_fallback("padded rollback geometry")
        other._fused_gdn_fallback("padded rollback geometry")
        other._fused_gdn_fallback("masked decode")
        model.named_modules = lambda: [("a", layer), ("b", other)]
        stats = qwen4_exp.qwen4_fused_gdn_stats(model)
        self.assertEqual(
            stats["decode_fallback_reasons"],
            {"padded rollback geometry": 2, "masked decode": 1},
        )
        self.assertEqual(stats["fallbacks"], 3)
        qwen4_exp.qwen4_fused_gdn_stats(model, reset=True)
        self.assertEqual(
            qwen4_exp.qwen4_fused_gdn_stats(model)["decode_fallback_reasons"], {}
        )


if __name__ == "__main__":
    unittest.main()
