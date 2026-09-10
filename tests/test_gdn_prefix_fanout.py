"""CPU proof for exact N=2 recurrent-prefix fan-out."""

import copy
import types
import unittest
from unittest.mock import patch

import mlx.core as mx

from benchmarks.qwen4_gdn_prefix_fanout_full_model_ab import (
    _cached_serving_once,
    _cool_until_stable,
    _parse_pmset_therm,
    _prepare_cached_prefix,
    _run_serving_blocks,
    _serving_row,
)

from mlx_lm.gdn_prefix_fanout import (
    GDNPrefixFanout,
    HybridCachePrefixFanout,
    gdn_prefix_fanout_stats,
)
from mlx_lm.generate import ParallelSampleGenerator, StopSequenceMatcher
from mlx_lm.hybrid_speculative import prepare_self_mtp_lane
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
from mlx_lm.models.qwen4_exp import (
    Model,
    ModelArgs,
    QSAKVCache,
    Qwen4ArraysCache,
)
from mlx_lm.sample_utils import LaneRNG

_DEVICE = None
HIDDEN = 128
WINDOW = 3


class TestThermalBenchmarkGate(unittest.TestCase):
    def test_pmset_parser_requires_explicit_clear_state(self):
        healthy = _parse_pmset_therm(
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
        )
        self.assertTrue(healthy["healthy"])

        limited = _parse_pmset_therm("CPU_Speed_Limit = 80\nGPU_Speed_Limit = 100\n")
        self.assertFalse(limited["healthy"])
        self.assertIn("cpu_speed_limit_80", limited["reasons"])

        warned = _parse_pmset_therm("Performance warning level: 1\n")
        self.assertFalse(warned["healthy"])
        self.assertIn("performance_warning_reported", warned["reasons"])

        unknown = _parse_pmset_therm("No useful thermal fields here\n")
        self.assertFalse(unknown["healthy"])
        self.assertIn("thermal_state_unverified", unknown["reasons"])

    def test_cooldown_counts_clear_snapshots_only_after_minimum(self):
        args = types.SimpleNamespace(
            minimum_cooldown_seconds=2.0,
            thermal_poll_seconds=1.0,
            thermal_max_cooldown_seconds=10.0,
            thermal_stable_snapshots=2,
        )
        clock = [0.0]
        health = iter((True, False, True, True))

        def monotonic():
            return clock[0]

        def sleep(seconds):
            clock[0] += seconds

        def snapshot(*_args):
            clear = next(health)
            return {"healthy": clear, "warning": not clear, "reasons": []}

        result = _cool_until_stable(
            args,
            "test",
            0,
            0,
            0,
            snapshot_fn=snapshot,
            sleep_fn=sleep,
            monotonic_fn=monotonic,
        )
        self.assertTrue(result["recovered"])
        self.assertEqual(result["stable_snapshots"], 2)
        self.assertEqual(result["elapsed_s"], 3.0)
        self.assertEqual(len(result["snapshots"]), 4)

    def test_cooldown_times_out_while_warning_persists(self):
        args = types.SimpleNamespace(
            minimum_cooldown_seconds=1.0,
            thermal_poll_seconds=1.0,
            thermal_max_cooldown_seconds=2.0,
            thermal_stable_snapshots=2,
        )
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        result = _cool_until_stable(
            args,
            "test",
            0,
            0,
            0,
            snapshot_fn=lambda *_args: {
                "healthy": False,
                "warning": True,
                "reasons": ["thermal_warning_reported"],
            },
            sleep_fn=sleep,
            monotonic_fn=lambda: clock[0],
        )
        self.assertFalse(result["recovered"])
        self.assertEqual(result["elapsed_s"], 2.0)
        self.assertEqual(len(result["snapshots"]), 3)

    def test_drifted_block_is_retained_and_retried(self):
        args = types.SimpleNamespace(max_bracket_drift=0.05, max_block_retries=1)
        timings = iter((100.0, 80.0, 80.0, 110.0, 100.0, 80.0, 80.0, 104.0))

        def fake_thermal_arm(_args, _gate, _block, _attempt, _slot, enabled, _run):
            total_ms = next(timings)
            counters = {
                "serving_requests": int(enabled),
                "serving_engaged": int(enabled),
                "serving_declined_not_n2": 0,
                "serving_declined_cache": 0,
                "serving_declined_error": 0,
                "serving_cleanups": int(enabled),
            }
            arm = {
                "enabled": enabled,
                "prepare_ms": 10.0,
                "prepare_tokens_per_second": 100.0,
                "decode_ms": total_ms - 10.0,
                "total_ms": total_ms,
                "aggregate_decode_tokens_per_second": 10.0,
                "aggregate_total_tokens_per_second": 11.0,
                "tokens": [[1, 2], [1, 2]],
                "token_sha256": ["a", "a"],
                "counter_delta": counters,
            }
            thermal = {
                "cooldown": {"recovered": True, "snapshots": []},
                "before": {"healthy": True},
                "after": {"healthy": True},
            }
            arm["thermal"] = thermal
            return arm, thermal

        with patch(
            "benchmarks.qwen4_gdn_prefix_fanout_full_model_ab._thermal_arm",
            side_effect=fake_thermal_arm,
        ):
            result = _run_serving_blocks(
                "test",
                "test",
                1,
                args,
                lambda _enabled: None,
                _serving_row,
            )
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["discarded_blocks"]), 1)
        self.assertEqual(
            result["discarded_blocks"][0]["discard_reasons"],
            ["closing_baseline_drift"],
        )
        self.assertEqual(result["blocks"][0]["attempt"], 1)
        self.assertLess(result["blocks"][0]["closing_baseline_drift_fraction"], 0.05)


def setUpModule():
    global _DEVICE
    _DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_DEVICE)


def _layer():
    mx.random.seed(31)
    layer = GatedDeltaNet(
        TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=HIDDEN,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            vocab_size=128,
        )
    )
    mx.eval(layer.parameters())
    return layer


def _inputs(batch, steps, seed):
    return (
        mx.random.normal((batch, steps, HIDDEN), key=mx.random.key(seed)).astype(
            mx.float32
        )
        * 0.25
    )


def _state_copy(cache):
    values = [None if value is None else mx.array(value) for value in cache.cache]
    mx.eval(*(value for value in values if value is not None))
    return values


def _assert_state_equal(test, left, right):
    test.assertEqual(len(left), len(right))
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            test.assertIs(lhs, rhs)
        else:
            test.assertTrue(bool(mx.array_equal(lhs, rhs)))


def _assert_state_close(test, left, right, relative_max=3e-3):
    test.assertEqual(len(left), len(right))
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            test.assertIs(lhs, rhs)
            continue
        delta = mx.max(mx.abs(lhs.astype(mx.float32) - rhs.astype(mx.float32)))
        scale = mx.maximum(mx.max(mx.abs(rhs.astype(mx.float32))), 1e-8)
        test.assertLess(float((delta / scale).item()), relative_max)


def _tree_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tree_arrays(item)


def _assert_cache_groups_close(test, left, right, relative_max=3e-3):
    test.assertEqual(len(left), len(right))
    for actual, expected in zip(left, right):
        test.assertIs(type(actual), type(expected))
        if callable(getattr(actual, "keys_and_values", None)):
            actual_arrays = list(_tree_arrays(actual.keys_and_values()))
            expected_arrays = list(_tree_arrays(expected.keys_and_values()))
            for name in ("index_keys", "_qsa_pooled_keys"):
                lhs = getattr(actual, name, None)
                rhs = getattr(expected, name, None)
                if lhs is not None or rhs is not None:
                    test.assertIsNotNone(lhs)
                    test.assertIsNotNone(rhs)
                    actual_arrays.append(lhs)
                    expected_arrays.append(rhs)
        else:
            actual_arrays = list(_tree_arrays(actual.state))
            expected_arrays = list(_tree_arrays(expected.state))
        test.assertEqual(len(actual_arrays), len(expected_arrays))
        for lhs, rhs in zip(actual_arrays, expected_arrays):
            if lhs.size == 0 or rhs.size == 0:
                test.assertEqual(tuple(lhs.shape), tuple(rhs.shape))
                continue
            delta = mx.max(mx.abs(lhs.astype(mx.float32) - rhs.astype(mx.float32)))
            scale = mx.maximum(mx.max(mx.abs(rhs.astype(mx.float32))), 1e-8)
            test.assertLess(float((delta / scale).item()), relative_max)


def _tiny_hybrid_model():
    text_config = dict(
        model_type="qwen4_exp_text",
        hidden_size=32,
        intermediate_size=0,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    mx.eval(model.parameters())
    return model


def _reference(layer, prefix, window, boundary, suffix=None):
    cache = ArraysCache(2)
    mx.eval(layer(prefix, cache=cache))
    if boundary:
        mx.eval(layer(window[:, :boundary], cache=cache))
    if suffix is not None and suffix.shape[1]:
        mx.eval(layer(suffix, cache=cache))
    return cache


class TestGDNPrefixFanout(unittest.TestCase):
    def setUp(self):
        gdn_prefix_fanout_stats(reset=True)
        self.layer = _layer()
        self.prefix = _inputs(1, 5, 41)
        self.window = _inputs(1, WINDOW, 43)
        self.source = ArraysCache(2)
        mx.eval(self.layer(self.prefix, cache=self.source))
        self.source.start_speculation()
        mx.eval(self.layer(self.window, cache=self.source))
        boundary = self.source.latest_exact_rollback_boundary()
        self.parent_before = [
            None if value is None else mx.array(value) for value in boundary.snapshot
        ]
        mx.eval(*(value for value in self.parent_before if value is not None))

    def _owner(self):
        retained = self.source.latest_exact_rollback_boundary().nbytes
        return GDNPrefixFanout.from_latest_record(
            self.source,
            enabled=True,
            retained_input_bytes=retained,
        )

    def test_default_off_is_inert(self):
        owner = GDNPrefixFanout.from_latest_record(self.source)
        self.assertIsNone(owner)
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["declined_disabled"], 1)
        self.assertEqual(stats["materializations"], 0)

    def test_zero_partial_all_boundaries_match_serial_oracle(self):
        owner = self._owner()
        for boundary in (0, 1, WINDOW):
            with self.subTest(boundary=boundary):
                lease = owner.fork(boundary)
                reference = _reference(self.layer, self.prefix, self.window, boundary)
                for row in range(2):
                    _assert_state_close(
                        self,
                        lease.cache.extract(row).cache,
                        reference.cache,
                    )
                lease.abort()
        for before, parent in zip(self.parent_before, owner._parent):
            if before is not None:
                self.assertTrue(bool(mx.array_equal(before, parent)))
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["materializations"], 3)
        self.assertEqual(stats["replay_tokens"], 0 + 1 + WINDOW)
        self.assertEqual(stats["fanout_rows"], 6)
        self.assertGreater(stats["copied_state_bytes"], 0)
        self.assertEqual(stats["retained_input_bytes"], owner.retained_input_bytes)
        owner.close()

    def test_supported_boundary_api_replaces_private_stack_access(self):
        boundary = self.source.latest_exact_rollback_boundary()
        self.assertEqual(boundary.num_tokens, WINDOW)
        zero = boundary.materialize(0)
        full = boundary.materialize(WINDOW)
        _assert_state_equal(self, zero.cache, self.parent_before)
        _assert_state_equal(self, full.cache, self.source.cache)

    def test_descendants_accept_zero_partial_all_and_match_solo(self):
        cases = ((0, 0), (1, 2), (WINDOW, WINDOW))
        owner = self._owner()
        parent = _state_copy(_reference(self.layer, self.prefix, self.window, 1))

        for case_index, accepted in enumerate(cases):
            with self.subTest(accepted=accepted):
                lease = owner.fork(1)
                suffix = _inputs(2, WINDOW, 100 + case_index)
                lease.start_speculation()
                mx.eval(self.layer(suffix, cache=lease.cache))
                descendants = lease.commit(WINDOW, accepted)
                self.assertTrue(lease.closed)
                self.assertIsNone(lease.cache)

                for row, keep in enumerate(accepted):
                    reference = _reference(
                        self.layer,
                        self.prefix,
                        self.window,
                        1,
                        suffix[row : row + 1, :keep],
                    )
                    _assert_state_close(self, descendants[row].cache, reference.cache)
                _assert_state_equal(self, list(owner._parent), self.parent_before)
                _assert_state_equal(
                    self,
                    parent,
                    _state_copy(_reference(self.layer, self.prefix, self.window, 1)),
                )

        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["committed_rows"], 6)
        self.assertEqual(stats["accepted_zero"], 2)
        self.assertEqual(stats["accepted_partial"], 2)
        self.assertEqual(stats["accepted_all"], 2)
        owner.close()

    def test_abort_and_validation_cleanup(self):
        owner = self._owner()
        with self.assertRaisesRegex(ValueError, "rows=2"):
            owner.fork(0, rows=3)
        with self.assertRaisesRegex(ValueError, "outside"):
            owner.fork(WINDOW + 1)
        lease = owner.fork(0)
        copied = lease.copied_state_bytes
        self.assertGreater(copied, 0)
        self.assertEqual(
            copied,
            sum(value.nbytes for value in lease.cache.cache if value is not None),
        )
        lease.abort()
        lease.abort()
        self.assertTrue(lease.closed)
        owner.close()
        owner.close()
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["aborts"], 1)
        self.assertEqual(stats["cleanups"], 2)

    def test_qwen4_record_materializes_paired_ple_state(self):
        cache = Qwen4ArraysCache(4)
        cache.cache = [mx.full((1, 2), index, dtype=mx.float32) for index in range(4)]
        cache.start_speculation()

        def ple_replay(m):
            return [
                mx.full((1, 2), 20 + m, dtype=mx.float32),
                mx.full((1, 2), 30 + m, dtype=mx.float32),
            ]

        def gdn_replay(m):
            return [
                mx.full((1, 2), m, dtype=mx.float32),
                mx.full((1, 2), 10 + m, dtype=mx.float32),
            ]

        cache.stage_ple_rollback(3, ple_replay, cache.cache[2:])
        cache.record_rollback(3, gdn_replay, cache.cache[:2])
        owner = GDNPrefixFanout.from_latest_record(
            cache, enabled=True, retained_input_bytes=96
        )
        lease = owner.fork(2)
        expected = (2, 12, 22, 32)
        for slot, value in enumerate(expected):
            self.assertTrue(
                bool(
                    mx.array_equal(
                        lease.cache[slot],
                        mx.full((2, 2), value, dtype=mx.float32),
                    )
                )
            )
        lease.abort()
        owner.close()


class TestHybridCachePrefixFanout(unittest.TestCase):
    def setUp(self):
        gdn_prefix_fanout_stats(reset=True)
        mx.random.seed(7301)
        self.model = _tiny_hybrid_model()
        self.prefix = mx.array([[1, 2, 3, 4, 5]], mx.uint32)
        self.ring = mx.array([[6, 7, 8]], mx.uint32)
        self.source = self.model.make_cache()
        mx.eval(self.model(self.prefix, cache=self.source))
        for cache in self.source:
            cache.start_speculation()
        mx.eval(self.model(self.ring, cache=self.source))

    def tearDown(self):
        for cache in self.source:
            cache.stop_speculation()

    def _reference(self, boundary, suffix=None):
        cache = self.model.make_cache()
        tokens = mx.concatenate([self.prefix, self.ring[:, :boundary]], axis=1)
        mx.eval(self.model(tokens, cache=cache))
        output = None
        if suffix is not None and suffix.shape[1]:
            output = self.model(suffix, cache=cache)
            mx.eval(
                output, *[array for item in cache for array in _tree_arrays(item.state)]
            )
        return output, cache

    def test_full_boundary_carries_qsa_gdn_and_paired_ple(self):
        owner = HybridCachePrefixFanout.from_prompt_cache(
            self.source, enabled=True, strict=True
        )
        self.assertEqual(owner.span, 3)
        for boundary in (0, 1, 3):
            with self.subTest(boundary=boundary):
                materialized = owner.materialize(boundary)
                _, reference = self._reference(boundary)
                _assert_cache_groups_close(self, materialized, reference)
                self.assertTrue(any(isinstance(c, QSAKVCache) for c in materialized))
                self.assertTrue(
                    any(
                        isinstance(c, Qwen4ArraysCache) and len(c.cache) == 4
                        for c in materialized
                    )
                )
        owner.close()

    def test_live_tip_fanout_matches_immutable_tip_and_is_one_shot(self):
        immutable = HybridCachePrefixFanout.from_prompt_cache(
            self.source, enabled=True, strict=True
        )
        consuming = HybridCachePrefixFanout.from_prompt_cache(
            self.source, enabled=True, strict=True
        )
        expected = immutable.fork(immutable.span)
        actual = consuming.fork_live_tip()
        _assert_cache_groups_close(self, actual.caches, expected.caches)
        actual.abort()
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            consuming.fork_live_tip()
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["hybrid_tip_fanout_batches"], 1)
        self.assertEqual(stats["hybrid_tip_fanout_rows"], 2)
        expected.abort()
        consuming.close()
        immutable.close()

    def test_full_transaction_zero_partial_all_matches_solo(self):
        owner = HybridCachePrefixFanout.from_prompt_cache(
            self.source, enabled=True, strict=True
        )
        for accepted in ((0, 3), (1, 2), (3, 3)):
            with self.subTest(accepted=accepted):
                lease = owner.fork(2)
                suffix = mx.array([[11, 12, 13], [21, 22, 23]], mx.uint32)
                lease.start_speculation()
                output = self.model(suffix, cache=lease.caches)
                mx.eval(output)
                descendants = lease.commit(3, accepted)
                for row, keep in enumerate(accepted):
                    _, reference = self._reference(2, suffix[row : row + 1, :keep])
                    _assert_cache_groups_close(self, descendants[row], reference)
                self.assertTrue(lease.closed)
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["hybrid_fanout_batches"], 3)
        self.assertEqual(stats["hybrid_fanout_rows"], 6)
        self.assertEqual(stats["hybrid_committed_rows"], 6)
        self.assertEqual(stats["accepted_zero"], 1)
        self.assertEqual(stats["accepted_partial"], 2)
        self.assertEqual(stats["accepted_all"], 3)
        owner.close()

    def test_hybrid_abort_stops_every_open_cache(self):
        owner = HybridCachePrefixFanout.from_prompt_cache(
            self.source, enabled=True, strict=True
        )
        lease = owner.fork(2)
        batch = list(lease.caches)
        lease.start_speculation()
        recurrent = [cache for cache in batch if isinstance(cache, ArraysCache)]
        self.assertTrue(recurrent)
        self.assertTrue(all(cache.speculating for cache in recurrent))
        lease.abort()
        self.assertTrue(lease.closed)
        self.assertTrue(all(not cache.speculating for cache in recurrent))
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["hybrid_aborts"], 1)
        owner.close()

    def test_disabled_declines_without_touching_cache(self):
        before = [list(_tree_arrays(cache.state)) for cache in self.source]
        self.assertIsNone(
            HybridCachePrefixFanout.from_prompt_cache(self.source, enabled=False)
        )
        after = [list(_tree_arrays(cache.state)) for cache in self.source]
        for left, right in zip(before, after):
            self.assertEqual(len(left), len(right))
            self.assertTrue(
                all(bool(mx.array_equal(a, b)) for a, b in zip(left, right))
            )
        self.assertEqual(gdn_prefix_fanout_stats()["declined_disabled"], 1)

    def test_decline_reasons_are_host_only_and_specific(self):
        self.assertIsNone(
            HybridCachePrefixFanout.from_prompt_cache([object()], enabled=True)
        )
        recurrent = ArraysCache(2)
        self.assertIsNone(
            HybridCachePrefixFanout.from_prompt_cache([recurrent], enabled=True)
        )
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["declined_unsupported_cache"], 1)
        self.assertEqual(stats["declined_no_record"], 1)


class TestServingComposition(unittest.TestCase):
    def setUp(self):
        mx.random.seed(59)
        self.model = _tiny_hybrid_model()

    def _run(self, enabled, *, rows=2, consume=False):
        parallel = ParallelSampleGenerator(
            self.model,
            None,
            5,
            rows,
            max_tokens=8,
            stop_matchers=[StopSequenceMatcher() for _ in range(rows)],
            all_tokens=[1, 2, 3, 4],
            self_mtp={
                "num_draft": 2,
                "persistent": True,
                "rate_gate": False,
                "share_qsa_indices": True,
                "sampling_temp": 0.0,
                "accept_rule": "residual",
                "gdn_prefix_fanout": enabled,
                "gdn_prefix_fanout_consume": consume,
            },
            lane_rng=LaneRNG(20260910),
            mtp_prompt=[1, 2, 3, 4, 5],
            prefill_step_size=4,
        )
        tokens = [[] for _ in range(rows)]
        terminal = [None] * rows
        try:
            while len(parallel):
                for row, response in parallel.next():
                    tokens[row].append(response.token)
                    if response.finish_reason is not None:
                        terminal[row] = response
        finally:
            parallel.close()
        return tokens, terminal

    def test_n2_serving_path_is_exact_and_engages(self):
        baseline, baseline_terminal = self._run(False)
        gdn_prefix_fanout_stats(reset=True)
        candidate, candidate_terminal = self._run(True)

        self.assertEqual(candidate, baseline)
        self.assertEqual(
            [list(item.all_tokens) for item in candidate_terminal],
            [list(item.all_tokens) for item in baseline_terminal],
        )
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["hybrid_armed"], 1)
        self.assertEqual(stats["hybrid_fanout_batches"], 1)
        self.assertEqual(stats["hybrid_fanout_rows"], 2)
        self.assertEqual(stats["hybrid_aborts"], 0)
        self.assertEqual(stats["serving_requests"], 1)
        self.assertEqual(stats["serving_engaged"], 1)
        self.assertEqual(stats["serving_declined_error"], 0)
        self.assertEqual(stats["serving_cleanups"], 1)
        self.assertGreaterEqual(stats["cleanups"], 2)

    def test_consuming_serving_path_is_exact_and_engages(self):
        baseline, _ = self._run(False)
        gdn_prefix_fanout_stats(reset=True)
        candidate, _ = self._run(True, consume=True)
        self.assertEqual(candidate, baseline)
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["hybrid_tip_fanout_batches"], 1)
        self.assertEqual(stats["hybrid_tip_fanout_rows"], 2)
        self.assertEqual(stats["serving_engaged"], 1)
        self.assertEqual(stats["serving_declined_error"], 0)

    def test_non_n2_declines_without_recording_or_mutating(self):
        gdn_prefix_fanout_stats(reset=True)
        tokens, terminal = self._run(True, rows=3)
        self.assertEqual([len(row) for row in tokens], [8, 8, 8])
        self.assertTrue(all(item.finish_reason == "length" for item in terminal))
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["requests"], 0)
        self.assertEqual(stats["hybrid_armed"], 0)
        self.assertEqual(stats["serving_requests"], 1)
        self.assertEqual(stats["serving_declined_not_n2"], 1)

    def test_prebatched_attach_failure_falls_back_closed(self):
        gdn_prefix_fanout_stats(reset=True)
        with patch(
            "mlx_lm.hybrid_speculative.attach_prebatched_self_mtp_lanes",
            side_effect=RuntimeError("synthetic attach refusal"),
        ):
            tokens, terminal = self._run(True)
        self.assertEqual([len(row) for row in tokens], [8, 8])
        self.assertTrue(all(item.finish_reason == "length" for item in terminal))
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["serving_engaged"], 0)
        self.assertEqual(stats["serving_declined_error"], 1)

    def test_apc_style_target_and_mtp_sidecar_engage(self):
        cached, _ = prepare_self_mtp_lane(
            mx.array([1, 2, 3, 4, 5], mx.uint32),
            self.model,
            uid=0,
            max_tokens=4,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(71),
            num_draft=2,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=4,
            share_qsa_indices=True,
        )
        gdn_prefix_fanout_stats(reset=True)
        parallel = ParallelSampleGenerator(
            self.model,
            copy.deepcopy(cached.caches.target),
            7,
            2,
            max_tokens=4,
            stop_matchers=[StopSequenceMatcher(), StopSequenceMatcher()],
            all_tokens=[1, 2, 3, 4, 5],
            self_mtp={
                "num_draft": 2,
                "persistent": True,
                "rate_gate": False,
                "share_qsa_indices": True,
                "sampling_temp": 0.0,
                "accept_rule": "residual",
                "gdn_prefix_fanout": True,
            },
            mtp_state=(
                copy.deepcopy(cached.caches.draft),
                mx.array(cached.lane.seed_h),
            ),
            lane_rng=LaneRNG(73),
            mtp_prompt=[6, 7],
            prefill_step_size=4,
        )
        rows = [[], []]
        try:
            while len(parallel):
                for row, response in parallel.next():
                    rows[row].append(response.token)
        finally:
            parallel.close()
        self.assertEqual([len(row) for row in rows], [4, 4])
        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["serving_engaged"], 1)
        self.assertEqual(stats["serving_declined_error"], 0)

    def test_cached_full_model_harness_clones_sidecar_per_arm(self):
        args = types.SimpleNamespace(
            max_tokens=4,
            seed=20260910,
            num_draft=2,
            prefill_step_size=4,
            share_qsa_indices=True,
        )
        cached = _prepare_cached_prefix(self.model, [1, 2, 3, 4, 5], args)
        gdn_prefix_fanout_stats(reset=True)
        baseline = _cached_serving_once(self.model, cached, [6, 7], args, False)
        candidate = _cached_serving_once(self.model, cached, [6, 7], args, True)
        self.assertEqual(candidate["tokens"], baseline["tokens"])
        self.assertEqual(candidate["counter_delta"]["serving_engaged"], 1)
        self.assertEqual(candidate["counter_delta"]["serving_declined_error"], 0)
        self.assertEqual(cached["tokens"], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
