"""CPU proof for exact N=2 recurrent-prefix fan-out."""

import unittest

import mlx.core as mx

from mlx_lm.gdn_prefix_fanout import (
    GDNPrefixFanout,
    gdn_prefix_fanout_stats,
)
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
from mlx_lm.models.qwen4_exp import Qwen4ArraysCache


_DEVICE = None
HIDDEN = 128
WINDOW = 3


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
    return mx.random.normal(
        (batch, steps, HIDDEN), key=mx.random.key(seed)
    ).astype(mx.float32) * 0.25


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
        self.parent_before = [
            None if value is None else mx.array(value)
            for value in self.source._rollbacks[-1].snapshot
        ]
        mx.eval(*(value for value in self.parent_before if value is not None))

    def _owner(self):
        record = self.source._rollbacks[-1]
        # q, k, v, a, b and conv_input are the arrays retained by the exact
        # rollback closure. S0 is the frozen parent, not a ring-input byte.
        defaults = record.fn.__defaults__
        retained_values = list(defaults[:5]) + [defaults[6]]
        retained = sum(
            int(getattr(value, "nbytes", 0))
            for value in retained_values
        )
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
                reference = _reference(
                    self.layer, self.prefix, self.window, boundary
                )
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

    def test_descendants_accept_zero_partial_all_and_match_solo(self):
        cases = ((0, 0), (1, 2), (WINDOW, WINDOW))
        owner = self._owner()
        parent = _state_copy(
            _reference(self.layer, self.prefix, self.window, 1)
        )

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
                    _assert_state_close(
                        self, descendants[row].cache, reference.cache
                    )
                _assert_state_equal(self, list(owner._parent), self.parent_before)
                _assert_state_equal(self, parent, _state_copy(
                    _reference(self.layer, self.prefix, self.window, 1)
                ))

        stats = gdn_prefix_fanout_stats()
        self.assertEqual(stats["committed_rows"], 6)
        self.assertEqual(stats["accepted_zero"], 2)
        self.assertEqual(stats["accepted_partial"], 2)
        self.assertEqual(stats["accepted_all"], 2)
        owner.close()
        self.assertTrue(owner.closed)
        self.assertEqual(owner._parent, ())

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
        cache.cache = [
            mx.full((1, 2), index, dtype=mx.float32) for index in range(4)
        ]
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


if __name__ == "__main__":
    unittest.main()
