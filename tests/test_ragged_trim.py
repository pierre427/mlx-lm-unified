# Copyright © 2026 Apple Inc.

"""Per-row (ragged) speculative rollback on the batch caches.

CPU-only by construction: the shapes are tiny and the assertions are on
integer/bookkeeping state, never on float logits (which are shape dependent
on this stack).
"""

import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    BatchRotatingQuantizedKVCache,
    KVCache,
    RaggedTrimUnsupported,
    can_trim_ragged,
    trim_ragged_prompt_cache,
)

_DEVICE = None


def setUpModule():
    global _DEVICE
    _DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_DEVICE)


def _tokens(batch, length, heads=2, dim=4, base=0):
    """Distinct per (row, position) values so a misplaced cell is visible."""
    rows = mx.arange(batch, dtype=mx.float32).reshape(batch, 1, 1, 1) * 1000
    positions = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    return mx.broadcast_to(rows + positions + base, (batch, heads, length, dim))


class _LedgerKVCache(BatchKVCache):
    """Stand-in for BatchQSAKVCache: K/V plus a [B, T, D] side ledger."""

    _RAGGED_TRIM_AUX_ARRAYS = (("index_keys", 1),)

    def __init__(self, left_padding, **kwargs):
        super().__init__(left_padding, **kwargs)
        self.index_keys = None

    def update_index_keys(self, keys):
        self.index_keys = (
            keys
            if self.index_keys is None
            else mx.concatenate([self.index_keys[:, : self._idx], keys], axis=1)
        )
        return self.index_keys


class _HandRolledKVCache(BatchKVCache):
    """A subclass that takes the ``_trim_ragged_aux`` escape hatch."""

    def __init__(self, left_padding, **kwargs):
        super().__init__(left_padding, **kwargs)
        self.aux_calls = []

    def _trim_ragged_aux(self, shifts, lo, hi, spec):
        self.aux_calls.append((shifts.tolist(), lo, hi, spec))


class _UndeclaredKVCache(BatchKVCache):
    """A subclass that forgot to say what it keeps beside K/V."""


class TestRaggedTrimBatchKV(unittest.TestCase):
    def _fill(self, cls=BatchKVCache, left_padding=(0, 0, 0), length=6, **kwargs):
        cache = cls(list(left_padding), **kwargs)
        batch = len(left_padding)
        values = _tokens(batch, length)
        cache.update_and_fetch(values, values)
        return cache, values

    def _valid_rows(self, cache):
        """The live cells of each row, oldest first."""
        left = cache.left_padding.tolist()
        return [
            cache.keys[i, 0, int(left[i]) : cache._idx, 0].tolist()
            for i in range(cache.keys.shape[0])
        ]

    def test_ragged_trim_drops_per_row_counts(self):
        cache, _ = self._fill(length=6)
        applied = cache.trim_ragged([0, 1, 3])

        self.assertEqual(applied, [0, 1, 3])
        self.assertEqual(cache.offset.tolist(), [6, 5, 3])
        self.assertEqual(cache.left_padding.tolist(), [0, 1, 3])
        self.assertEqual(cache._idx, 6)
        self.assertEqual(
            self._valid_rows(cache),
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
                [2000.0, 2001.0, 2002.0],
            ],
        )

    def test_uniform_vector_costs_no_padding(self):
        cache, _ = self._fill(length=6)
        cache.trim_ragged([2, 2, 2])

        self.assertEqual(cache._idx, 4)
        self.assertEqual(cache.offset.tolist(), [4, 4, 4])
        self.assertEqual(cache.left_padding.tolist(), [0, 0, 0])

    def test_all_zero_vector_is_a_noop(self):
        cache, _ = self._fill(length=6)
        before = cache.keys.tolist()
        self.assertEqual(cache.trim_ragged([0, 0, 0]), [0, 0, 0])
        self.assertEqual(cache._idx, 6)
        self.assertEqual(cache.offset.tolist(), [6, 6, 6])
        self.assertEqual(cache.keys.tolist(), before)

    def test_unequal_left_padding_is_preserved(self):
        cache, _ = self._fill(left_padding=(2, 0, 1), length=6)
        # Rows start with 4, 6 and 5 live tokens respectively.
        self.assertEqual(cache.offset.tolist(), [4, 6, 5])
        cache.trim_ragged([1, 0, 2])

        self.assertEqual(cache.offset.tolist(), [3, 6, 3])
        self.assertEqual(cache.left_padding.tolist(), [3, 0, 3])
        self.assertEqual(
            self._valid_rows(cache),
            [
                [2.0, 3.0, 4.0],
                [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0],
                [2001.0, 2002.0, 2003.0],
            ],
        )

    def test_next_append_lands_after_every_row(self):
        cache, _ = self._fill(length=6)
        cache.trim_ragged([0, 1, 3])
        nxt = _tokens(3, 1, base=90)
        cache.update_and_fetch(nxt, nxt)

        self.assertEqual(cache.offset.tolist(), [7, 6, 4])
        self.assertEqual(
            self._valid_rows(cache),
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 90.0],
                [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1090.0],
                [2000.0, 2001.0, 2002.0, 2090.0],
            ],
        )

    def test_matches_per_row_independent_decode(self):
        """A merged, raggedly trimmed batch equals per-row single caches."""
        lengths = [7, 5, 6]
        drops = [2, 0, 3]
        singles = []
        for row, length in enumerate(lengths):
            single = KVCache()
            values = _tokens(1, length) + row * 1000
            single.update_and_fetch(values, values)
            singles.append(single)
        batch = BatchKVCache.merge(singles)
        batch.trim_ragged(drops)
        for single, drop in zip(singles, drops):
            single.trim(drop)

        rows = self._valid_rows(batch)
        for row, single in enumerate(singles):
            self.assertEqual(
                rows[row], single.keys[0, 0, : single.offset, 0].tolist()
            )
            self.assertEqual(int(batch.offset[row].item()), single.offset)

    def test_extract_after_ragged_trim(self):
        cache, _ = self._fill(length=6)
        cache.trim_ragged([0, 1, 3])
        for row, expected in enumerate(([0, 1, 2, 3, 4, 5], [1000, 1001, 1002, 1003, 1004], [2000, 2001, 2002])):
            single = cache.extract(row)
            self.assertEqual(single.offset, len(expected))
            self.assertEqual(
                single.keys[0, 0, :, 0].tolist(), [float(v) for v in expected]
            )

    def test_filter_reclaims_the_padding(self):
        cache, _ = self._fill(length=6)
        cache.trim_ragged([1, 2, 3])
        self.assertEqual(cache.left_padding.tolist(), [0, 1, 2])
        cache.filter(mx.array([1, 2]))
        self.assertEqual(cache.left_padding.tolist(), [0, 1])
        self.assertEqual(cache._idx, 4)
        self.assertEqual(cache.offset.tolist(), [4, 3])

    def test_over_trim_fails_loud(self):
        cache, _ = self._fill(left_padding=(3, 0, 0), length=6)
        with self.assertRaises(ValueError):
            cache.trim_ragged([4, 0, 0])
        with self.assertRaises(ValueError):
            cache.trim_ragged([0, -1, 0])
        with self.assertRaises(ValueError):
            cache.trim_ragged([1, 1])
        with self.assertRaises(TypeError):
            cache.trim_ragged(2)

    def test_pending_right_padding_fails_loud(self):
        cache = BatchKVCache([0, 0, 0])
        cache.prepare(right_padding=[0, 1, 2], lengths=[6, 5, 4])
        values = _tokens(3, 6)
        cache.update_and_fetch(values, values)
        with self.assertRaises(RaggedTrimUnsupported):
            cache.trim_ragged([1, 0, 0])
        cache.finalize()
        cache.trim_ragged([1, 0, 0])
        self.assertEqual(cache.offset.tolist(), [5, 5, 4])


class TestRaggedTrimAuxLedger(unittest.TestCase):
    def _fill(self, length=6, batch=3):
        cache = _LedgerKVCache([0] * batch)
        values = _tokens(batch, length)
        cache.update_and_fetch(values, values)
        ledger = mx.broadcast_to(
            mx.arange(batch, dtype=mx.float32).reshape(batch, 1, 1) * 1000
            + mx.arange(length, dtype=mx.float32).reshape(1, length, 1),
            (batch, length, 2),
        )
        cache.update_index_keys(ledger)
        return cache

    def test_ledger_stays_in_step(self):
        cache = self._fill(length=6)
        cache.trim_ragged([0, 1, 3])

        self.assertEqual(cache.index_keys.shape[1], cache._idx)
        left = cache.left_padding.tolist()
        for row, offset in enumerate(cache.offset.tolist()):
            live = cache.index_keys[row, int(left[row]) : cache._idx, 0].tolist()
            # len(index_keys) == offset, per row.
            self.assertEqual(len(live), int(offset))
            self.assertEqual(live, [row * 1000.0 + p for p in range(int(offset))])

    def test_ledger_survives_the_next_append(self):
        cache = self._fill(length=6)
        cache.trim_ragged([0, 1, 3])
        nxt = _tokens(3, 1, base=90)
        cache.update_and_fetch(nxt, nxt)
        cache.update_index_keys(mx.full((3, 1, 2), 42.0))

        self.assertEqual(cache.index_keys.shape[1], cache._idx)
        left = cache.left_padding.tolist()
        for row, offset in enumerate(cache.offset.tolist()):
            live = cache.index_keys[row, int(left[row]) : cache._idx, 0].tolist()
            self.assertEqual(len(live), int(offset))
            self.assertEqual(live[-1], 42.0)

    def test_short_ledger_fails_loud_and_changes_nothing(self):
        cache = self._fill(length=6)
        cache.index_keys = cache.index_keys[:, :3]
        before = {
            "keys": cache.keys.tolist(),
            "values": cache.values.tolist(),
            "index_keys": cache.index_keys.tolist(),
            "idx": cache._idx,
            "offset": cache.offset.tolist(),
            "left_padding": cache.left_padding.tolist(),
        }
        with self.assertRaises(RuntimeError):
            cache.trim_ragged([0, 1, 3])
        # The roll is in place, so a late raise must not have moved K/V.
        self.assertEqual(cache.keys.tolist(), before["keys"])
        self.assertEqual(cache.values.tolist(), before["values"])
        self.assertEqual(cache.index_keys.tolist(), before["index_keys"])
        self.assertEqual(cache._idx, before["idx"])
        self.assertEqual(cache.offset.tolist(), before["offset"])
        self.assertEqual(cache.left_padding.tolist(), before["left_padding"])

    def test_override_hook_receives_the_shifts(self):
        cache = _HandRolledKVCache([0, 0, 0])
        values = _tokens(3, 6)
        cache.update_and_fetch(values, values)
        cache.trim_ragged([1, 2, 4])
        self.assertEqual(cache.aux_calls, [([0, 1, 3], 0, 5, None)])

    def test_undeclared_subclass_fails_loud(self):
        cache = _UndeclaredKVCache([0, 0])
        values = _tokens(2, 4)
        cache.update_and_fetch(values, values)
        with self.assertRaises(RaggedTrimUnsupported):
            cache.trim_ragged([1, 0])
        # A uniform trim is unaffected.
        self.assertEqual(cache.trim(1), 1)


class TestRaggedTrimUnsupportedClasses(unittest.TestCase):
    def test_rotating_batch_caches_fail_loud(self):
        for cls, kwargs in (
            (BatchRotatingKVCache, {"max_size": 32}),
            (BatchRotatingQuantizedKVCache, {"max_size": 32}),
        ):
            with self.subTest(cls=cls.__name__):
                cache = cls(left_padding=[0, 0], **kwargs)
                self.assertFalse(cache.supports_ragged_trim())
                with self.assertRaises(RaggedTrimUnsupported):
                    cache.trim_ragged([1, 0])

    def test_single_sequence_caches_fail_loud(self):
        cache = KVCache()
        self.assertFalse(cache.supports_ragged_trim())
        with self.assertRaises(RaggedTrimUnsupported):
            cache.trim_ragged([1])

    def test_mixed_cache_list_never_degrades_to_uniform(self):
        supported = BatchKVCache([0, 0])
        unsupported = BatchRotatingKVCache(max_size=32, left_padding=[0, 0])
        values = _tokens(2, 4)
        supported.update_and_fetch(values, values)
        unsupported.update_and_fetch(values, values)

        self.assertFalse(can_trim_ragged([supported, unsupported]))
        with self.assertRaises(RaggedTrimUnsupported):
            trim_ragged_prompt_cache([supported, unsupported], [1, 0])
        # The supported entry was not touched.
        self.assertEqual(supported.offset.tolist(), [4, 4])

    def test_trim_ragged_prompt_cache_applies_to_every_entry(self):
        caches = [BatchKVCache([0, 0]) for _ in range(3)]
        values = _tokens(2, 5)
        for cache in caches:
            cache.update_and_fetch(values, values)
        self.assertTrue(can_trim_ragged(caches))
        self.assertEqual(trim_ragged_prompt_cache(caches, [2, 0]), [2, 0])
        for cache in caches:
            self.assertEqual(cache.offset.tolist(), [3, 5])

    def test_group_preflights_before_any_entry_moves(self):
        """A late per-row rejection must not leave earlier entries rewound."""
        kv = BatchKVCache([0, 0])
        values = _tokens(2, 4)
        kv.update_and_fetch(values, values)
        arrays = ArraysCache(1)
        arrays.start_speculation()
        layer = _Recurrence(arrays)
        layer(mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32))
        arrays.trim_ragged([0, 4])  # row 1 spends its whole record

        state_before = arrays[0].tolist()
        with self.assertRaises(RuntimeError):
            # The KV could do this; the recurrent entry cannot.
            trim_ragged_prompt_cache([kv, arrays], [0, 1])
        self.assertEqual(kv._idx, 4)
        self.assertEqual(kv.offset.tolist(), [4, 4])
        self.assertEqual(kv.left_padding.tolist(), [0, 0])
        self.assertEqual(arrays[0].tolist(), state_before)

    def test_group_preflights_a_short_aux_ledger(self):
        first = BatchKVCache([0, 0])
        values = _tokens(2, 4)
        first.update_and_fetch(values, values)
        second = _LedgerKVCache([0, 0])
        second.update_and_fetch(values, values)
        second.update_index_keys(mx.zeros((2, 4, 2)))
        second.index_keys = second.index_keys[:, :2]

        with self.assertRaises(RuntimeError):
            trim_ragged_prompt_cache([first, second], [0, 1])
        self.assertEqual(first.offset.tolist(), [4, 4])
        self.assertEqual(second.offset.tolist(), [4, 4])

    def test_mixed_kv_and_recurrent_cache_group(self):
        """The real trunk shape: attention rows plus recurrent-state rows."""
        kv = BatchKVCache([0, 0])
        values = _tokens(2, 4)
        kv.update_and_fetch(values, values)
        arrays = ArraysCache(1)
        arrays.start_speculation()
        layer = _Recurrence(arrays)
        layer(mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32))

        group = [kv, arrays]
        self.assertTrue(can_trim_ragged(group))
        self.assertEqual(trim_ragged_prompt_cache(group, [0, 2]), [0, 2])
        self.assertEqual(kv.offset.tolist(), [4, 2])
        # state <- state * 2 + token: row 0 keeps all four, row 1 keeps two.
        self.assertEqual(arrays[0].reshape(-1).tolist(), [26, 16])


class TestRaggedTrimQuantized(unittest.TestCase):
    def test_quantized_batch_cache_bookkeeping(self):
        cache = BatchQuantizedKVCache([0, 0, 0], group_size=32, bits=8)
        values = mx.random.normal((3, 2, 96, 64))
        cache.update_and_fetch(values, values)
        self.assertTrue(cache.supports_ragged_trim())
        cache.trim_ragged([0, 8, 24])

        self.assertEqual(cache._idx, 96)
        self.assertEqual(cache.offset.tolist(), [96, 88, 72])
        self.assertEqual(cache.left_padding.tolist(), [0, 8, 24])
        for row, length in enumerate((96, 88, 72)):
            single = cache.extract(row)
            self.assertEqual(single.offset, length)

    def test_quantized_rows_keep_their_own_prefix(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=32, bits=8)
        values = mx.broadcast_to(
            mx.arange(64, dtype=mx.float32).reshape(1, 1, 64, 1), (2, 1, 64, 32)
        )
        cache.update_and_fetch(values, values)
        cache.trim_ragged([0, 5])

        for row, length in enumerate((64, 59)):
            single = cache.extract(row)
            keys = mx.dequantize(
                *single.keys, group_size=cache.group_size, bits=cache.key_bits
            )
            self.assertEqual(keys.shape[2], length)
            self.assertEqual(
                [round(v) for v in keys[0, 0, :, 0].tolist()], list(range(length))
            )


class _Recurrence:
    """Tiny exact-replay recurrent layer: state <- state * 2 + token."""

    def __init__(self, cache):
        self.cache = cache
        self.replays = 0

    def __call__(self, tokens):
        """tokens: [B, S] int32."""
        before = None if self.cache[0] is None else mx.array(self.cache[0])
        batch, steps = tokens.shape
        zero = mx.zeros((batch, 1), dtype=mx.int32)
        state = zero if before is None else before

        def advance(start, count, rows=slice(None)):
            out = start
            for step in range(count):
                out = out * 2 + tokens[rows, step : step + 1]
            return out

        def rollback(m):
            self.replays += 1
            return [advance(zero if before is None else before, m)]

        def per_row_rollback(ms):
            self.replays += 1
            base = zero if before is None else before
            rows = [
                advance(base[i : i + 1], ms[i], slice(i, i + 1))
                for i in range(batch)
            ]
            return [mx.concatenate(rows)]

        snapshot = [None if before is None else mx.array(before)]
        self.cache.record_rollback(
            steps, rollback, snapshot, per_row_fn=per_row_rollback
        )
        self.cache[0] = advance(state, steps)
        return self.cache[0]


class TestArraysCacheRaggedTrim(unittest.TestCase):
    def _run(self, per_row: bool, batch: int = 0):
        cache = ArraysCache(1)
        cache.start_speculation()
        if batch:
            # Speculation starts after prefill, so the first record's snapshot
            # is real state rather than None.
            cache.cache = [mx.zeros((batch, 1), dtype=mx.int32)]
        layer = _Recurrence(cache)
        if not per_row:
            # Drop the vectorized form so the a0 path (replay per distinct
            # length) is exercised instead.
            original = cache.record_rollback

            def record(num_tokens, fn, snapshot, per_row_fn=None):
                return original(num_tokens, fn, snapshot)

            cache.record_rollback = record
            layer = _Recurrence(cache)
        return cache, layer

    def _reference(self, prompt, tail, keep):
        """State of one row after prompt + the first ``keep`` tail tokens."""
        state = 0
        for token in list(prompt) + list(tail[:keep]):
            state = state * 2 + token
        return state

    def test_ragged_rollback_is_exact(self):
        for per_row in (False, True):
            with self.subTest(per_row=per_row):
                cache, layer = self._run(per_row)
                prompt = [[3, 1], [4, 2], [5, 9]]
                layer(mx.array(prompt, dtype=mx.int32))
                tail = [[7, 6, 2], [8, 1, 3], [9, 4, 5]]
                layer(mx.array(tail, dtype=mx.int32))

                drops = [0, 1, 3]
                self.assertEqual(cache.trim_ragged(drops), drops)
                got = cache[0].reshape(-1).tolist()
                want = [
                    self._reference(prompt[i], tail[i], 3 - drops[i])
                    for i in range(3)
                ]
                self.assertEqual(got, want)

    def test_a0_replays_once_per_distinct_length(self):
        cache, layer = self._run(per_row=False)
        layer(mx.array([[1], [2], [3], [4]], dtype=mx.int32))
        layer(mx.array([[5, 6, 7], [8, 9, 1], [2, 3, 4], [5, 6, 7]], dtype=mx.int32))
        layer.replays = 0
        # Rows 1 and 2 share a drop, so only two lengths need a replay; the
        # untouched row 0 is free.
        cache.trim_ragged([0, 1, 1, 2])
        self.assertEqual(layer.replays, 2)

    def test_per_row_form_builds_one_graph(self):
        cache, layer = self._run(per_row=True)
        layer(mx.array([[1], [2], [3]], dtype=mx.int32))
        layer(mx.array([[5, 6, 7], [8, 9, 1], [2, 3, 4]], dtype=mx.int32))
        layer.replays = 0
        cache.trim_ragged([0, 1, 3])
        self.assertEqual(layer.replays, 1)

    def test_rollback_spans_several_records(self):
        cache, layer = self._run(per_row=False)
        prompt = [[3], [1], [4]]
        layer(mx.array(prompt, dtype=mx.int32))
        tail = [[7, 6], [8, 1], [9, 4]]
        layer(mx.array([[t[0]] for t in tail], dtype=mx.int32))
        layer(mx.array([[t[1]] for t in tail], dtype=mx.int32))

        drops = [0, 1, 2]
        cache.trim_ragged(drops)
        got = cache[0].reshape(-1).tolist()
        want = [self._reference(prompt[i], tail[i], 2 - drops[i]) for i in range(3)]
        self.assertEqual(got, want)

    def test_zero_vector_leaves_state_and_records_alone(self):
        cache, layer = self._run(per_row=False)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        before = cache[0].tolist()
        records = len(cache._rollbacks)
        self.assertEqual(cache.trim_ragged([0, 0]), [0, 0])
        self.assertEqual(cache[0].tolist(), before)
        self.assertEqual(len(cache._rollbacks), records)

    def test_over_trim_fails_loud(self):
        cache, layer = self._run(per_row=False)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        with self.assertRaises(RuntimeError):
            cache.trim_ragged([0, 5])

    def test_membership_change_invalidates_records(self):
        cache, layer = self._run(per_row=False)
        layer(mx.array([[1], [2], [3]], dtype=mx.int32))
        cache.filter(mx.array([0, 2]))
        self.assertEqual(len(cache._rollbacks), 0)
        with self.assertRaises(RuntimeError) as raised:
            cache.trim_ragged([1, 1])
        self.assertIn("filter()", str(raised.exception))

    def test_single_sequence_paths_agree(self):
        """A one-row ragged trim matches the untouched uniform path exactly."""
        results = []
        for use_ragged_api in (False, True):
            cache = ArraysCache(1)
            cache.start_speculation()
            layer = _Recurrence(cache)
            layer(mx.array([[3, 1, 4]], dtype=mx.int32))
            layer(mx.array([[1, 5, 9]], dtype=mx.int32))
            if use_ragged_api:
                self.assertEqual(cache.trim_ragged([4]), [4])
            else:
                self.assertEqual(cache.trim(4), 4)
            results.append(cache[0].reshape(-1).tolist())
        # Six tokens seen, four dropped: the state after [3, 1].
        self.assertEqual(results[0], [7])
        self.assertEqual(results[1], results[0])

    def test_uniform_and_ragged_agree_within_a_record(self):
        cache, layer = self._run(per_row=False)
        prompt = [[3, 1], [4, 2]]
        layer(mx.array(prompt, dtype=mx.int32))
        tail = [[7, 6, 2], [8, 1, 3]]
        layer(mx.array(tail, dtype=mx.int32))
        cache.trim_ragged([2, 2])
        self.assertEqual(
            cache[0].reshape(-1).tolist(),
            [self._reference(prompt[i], tail[i], 1) for i in range(2)],
        )

    def test_batched_rows_match_independent_single_row_caches(self):
        """The merged, raggedly trimmed state equals per-row solo decode."""
        prompt = [[3, 1], [4, 2], [5, 9]]
        tail = [[7, 6, 2], [8, 1, 3], [9, 4, 5]]
        drops = [0, 1, 3]

        batch, batch_layer = self._run(per_row=False)
        batch_layer(mx.array(prompt, dtype=mx.int32))
        batch_layer(mx.array(tail, dtype=mx.int32))
        batch.trim_ragged(drops)

        for row in range(3):
            solo = ArraysCache(1)
            solo.start_speculation()
            layer = _Recurrence(solo)
            layer(mx.array([prompt[row]], dtype=mx.int32))
            layer(mx.array([tail[row]], dtype=mx.int32))
            if drops[row]:
                solo.trim(drops[row])
            self.assertEqual(
                batch[0][row : row + 1].tolist(), solo[0].tolist()
            )

    def test_untouched_row_is_bit_identical(self):
        cache, layer = self._run(per_row=False)
        layer(mx.array([[3, 1], [4, 2]], dtype=mx.int32))
        layer(mx.array([[7, 6, 2], [8, 1, 3]], dtype=mx.int32))
        before = cache[0][0:1].tolist()
        cache.trim_ragged([0, 3])
        self.assertEqual(cache[0][0:1].tolist(), before)

    def test_divergent_then_uniform_rewind(self):
        """The reported blocker: row 0 must keep the record row 1 used up."""
        cache = ArraysCache(1)
        cache.start_speculation()
        cache.cache = [mx.array([[0.0], [0.0]])]

        def push(count):
            before = mx.array(cache.cache[0])
            cache.record_rollback(count, lambda m: [before + m], [before])
            cache.cache = [before + count]

        push(1)  # record A
        push(1)  # record B
        self.assertEqual(cache[0].reshape(-1).tolist(), [2.0, 2.0])

        cache.trim_ragged([0, 1])
        self.assertEqual(cache[0].reshape(-1).tolist(), [2.0, 1.0])
        # Row 0 still has both records; row 1 only has A.
        self.assertEqual(cache._row_capacity(2), [2, 1])

        cache.trim_ragged([1, 1])
        self.assertEqual(cache[0].reshape(-1).tolist(), [1.0, 0.0])

    def test_rewind_after_divergence_matches_solo_decode(self):
        """Every row, after any split of its rewind, matches decoding alone."""
        prompt = [[3, 1], [4, 2]]
        chunks = [[[7], [8]], [[6, 2], [1, 3]]]  # records of 1 and 2 tokens
        tail = [[7, 6, 2], [8, 1, 3]]

        for first in ((0, 1), (1, 0), (0, 3), (2, 1), (1, 1), (3, 0)):
            for second in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)):
                if any(f + s > 3 for f, s in zip(first, second)):
                    continue
                with self.subTest(first=first, second=second):
                    cache, layer = self._run(per_row=False, batch=2)
                    layer(mx.array(prompt, dtype=mx.int32))
                    for chunk in chunks:
                        layer(mx.array(chunk, dtype=mx.int32))
                    cache.trim_ragged(list(first))
                    cache.trim_ragged(list(second))
                    want = [
                        self._reference(
                            prompt[i], tail[i], 3 - first[i] - second[i]
                        )
                        for i in range(2)
                    ]
                    self.assertEqual(cache[0].reshape(-1).tolist(), want)

    def test_over_trim_after_divergence_fails_loud(self):
        cache, layer = self._run(per_row=False, batch=2)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        layer(mx.array([[3], [4]], dtype=mx.int32))
        cache.trim_ragged([0, 2])
        self.assertEqual(cache._row_capacity(2), [2, 0])
        with self.assertRaises(RuntimeError) as raised:
            cache.trim_ragged([0, 1])
        self.assertIn("[1]", str(raised.exception))
        # The rejected call left the state alone.
        self.assertEqual(cache._row_capacity(2), [2, 0])

    def test_record_budget_stays_conservative_for_scalar_readers(self):
        """``r[0]`` answers the uniform question, so it reports the min depth."""
        cache, layer = self._run(per_row=False, batch=2)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        layer(mx.array([[3, 5], [4, 6]], dtype=mx.int32))
        self.assertEqual(sum(r[0] for r in cache._rollbacks), 3)
        cache.trim_ragged([0, 2])
        # Row 1 used up the two-token record; the scalar view must not claim
        # tokens row 1 no longer has.
        self.assertEqual(sum(r[0] for r in cache._rollbacks), 1)
        self.assertEqual(cache._row_capacity(2), [3, 1])
        # A uniform trim of 1 is what that scalar promises, and it works.
        self.assertEqual(cache.trim(1), 1)

    def test_uniform_trim_refuses_past_the_shallowest_row(self):
        cache, layer = self._run(per_row=False, batch=2)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        layer(mx.array([[3], [4]], dtype=mx.int32))
        cache.trim_ragged([0, 2])
        with self.assertRaises(RuntimeError):
            cache.trim(1)

    def test_non_integral_counts_are_rejected(self):
        cache, layer = self._run(per_row=False, batch=2)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        with self.assertRaises(ValueError):
            cache.trim_ragged([-0.5, 1.9])
        with self.assertRaises(ValueError):
            cache.trim_ragged([0.0, 1.5])
        # Integral floats still describe a real rewind.
        self.assertEqual(cache.trim_ragged([0.0, 1.0]), [0, 1])

    def test_staged_rollback_hook_runs_on_invalidation(self):
        class _Staged(ArraysCache):
            cleared = 0

            def _clear_staged_rollback(self):
                type(self).cleared += 1

        cache = _Staged(1)
        cache.start_speculation()
        cache.cache = [mx.array([[1.0], [2.0]])]
        cache.filter(mx.array([0, 1]))
        self.assertEqual(_Staged.cleared, 1)

    def test_record_shape_stays_a_three_tuple(self):
        cache, layer = self._run(per_row=True)
        layer(mx.array([[1], [2]], dtype=mx.int32))
        record = cache._rollbacks[-1]
        self.assertEqual(len(record), 3)
        num_tokens, fn, snapshot = record
        self.assertEqual(num_tokens, 1)
        self.assertTrue(callable(fn))
        self.assertEqual(sum(r[0] for r in cache._rollbacks), 1)
        self.assertIsNotNone(record.per_row_fn)


class _MaskedRecurrence:
    """Stand-in for GatedDeltaNet on a bare ArraysCache.

    Live steps honour the mask (masked steps are recurrence no-ops, exactly
    like the GDN kernel); the replay closure is mask-free and indexed from
    slab position 0, exactly like GDN's ``_rollback``.
    """

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, tokens):
        steps = tokens.shape[1]
        mask = self.cache.make_mask(steps)
        before = self.cache[0]

        def rollback(m):
            out = before
            for step in range(m):
                out = out * 2 + tokens[:, step : step + 1]
            return [out]

        spans = self.cache.rollback_spans(steps, mask)
        if self.cache.speculating and spans is not None:
            self.cache.record_rollback(steps, rollback, [before])

        out = before
        for step in range(steps):
            advanced = out * 2 + tokens[:, step : step + 1]
            out = advanced if mask is None else mx.where(
                mask[:, step : step + 1], advanced, out
            )
        self.cache[0] = out
        self.cache.advance(steps)
        return out


class TestRollbackSpans(unittest.TestCase):
    """Per-row spans on a bare ArraysCache: Qwen3.5 and Qwen3-Next."""

    def test_spans_describe_the_geometry(self):
        cache = ArraysCache(1)
        self.assertEqual(cache.rollback_spans(4), ())

        padded = ArraysCache(1)
        padded.prepare(lengths=[4, 2, 3])
        self.assertEqual(padded.rollback_spans(4), [4, 2, 3])
        # A span never exceeds the slab width.
        self.assertEqual(padded.rollback_spans(2), [2, 2, 2])

        left = ArraysCache(1, left_padding=[1, 0])
        self.assertIsNone(left.rollback_spans(3))

    def test_mask_without_metadata_refuses(self):
        cache = ArraysCache(1)
        self.assertIsNone(cache.rollback_spans(3, mask=mx.ones((2, 3), mx.bool_)))

    def test_advance_carries_the_spans(self):
        cache = ArraysCache(1)
        cache.prepare(lengths=[5, 2])
        cache.advance(2)
        self.assertEqual(cache.rollback_spans(3), [3, 0])
        cache.finalize()
        self.assertEqual(cache.rollback_spans(3), ())

    def _padded_run(self):
        """Record A unpadded, then record B with per-row spans 2 and 1."""
        cache = ArraysCache(1)
        cache.start_speculation()
        cache.cache = [mx.zeros((2, 1), dtype=mx.int32)]
        layer = _MaskedRecurrence(cache)
        layer(mx.array([[3], [5]], dtype=mx.int32))
        cache.prepare(lengths=[2, 1])
        layer(mx.array([[7, 9], [11, 0]], dtype=mx.int32))
        cache.finalize()
        return cache

    def test_padded_forward_credits_each_row_its_own_span(self):
        cache = self._padded_run()
        self.assertEqual(cache[0].reshape(-1).tolist(), [35, 21])
        # Row 1 advanced one token through a two-wide slab.
        self.assertEqual(cache._row_capacity(2), [3, 2])

    def test_rewind_spanning_records_lands_on_the_right_one(self):
        cache = self._padded_run()
        cache.trim_ragged([0, 2])
        # Row 1 owns one token in B and one in A, so rewinding two puts it
        # back before A. Crediting it the slab width would stop inside B.
        self.assertEqual(cache[0].reshape(-1).tolist(), [35, 0])

    def test_partial_rewind_uses_the_rows_own_depth(self):
        cache = self._padded_run()
        cache.trim_ragged([0, 1])
        self.assertEqual(cache[0].reshape(-1).tolist(), [35, 5])

    def test_replay_of_a_wider_row_leaves_the_narrow_row_alone(self):
        cache = self._padded_run()
        cache.trim_ragged([1, 0])
        self.assertEqual(cache[0].reshape(-1).tolist(), [13, 21])

    def test_uniform_trim_respects_the_shortest_row(self):
        cache = self._padded_run()
        self.assertEqual(cache.trim(2), 2)
        self.assertEqual(cache[0].reshape(-1).tolist(), [3, 0])
        with self.assertRaises(RuntimeError):
            cache.trim(2)

    def test_unpadded_forward_keeps_the_uniform_record(self):
        cache = ArraysCache(1)
        cache.start_speculation()
        cache.cache = [mx.zeros((2, 1), dtype=mx.int32)]
        _MaskedRecurrence(cache)(mx.array([[3, 4], [5, 6]], dtype=mx.int32))
        self.assertIsNone(cache._rollbacks[-1].depths)
        self.assertEqual(cache._row_capacity(2), [2, 2])

    def test_recording_under_leading_pads_fails_loud(self):
        cache = ArraysCache(1, left_padding=[1, 0])
        cache.start_speculation()
        cache.cache = [mx.zeros((2, 1), dtype=mx.int32)]
        with self.assertRaises(RuntimeError) as raised:
            cache.record_rollback(2, lambda m: [mx.zeros((2, 1))], [cache[0]])
        self.assertIn("rollback_spans", str(raised.exception))

    def test_span_row_count_must_match_the_batch(self):
        cache = ArraysCache(1)
        cache.start_speculation()
        cache.cache = [mx.zeros((2, 1), dtype=mx.int32)]
        with self.assertRaises(RuntimeError):
            cache.record_rollback(
                2, lambda m: [cache[0]], [cache[0]], depths=[2, 2, 2]
            )


class TestQwen4CacheContracts(unittest.TestCase):
    """The production Qwen4 cache types, exercised through this file's API only."""

    def test_qwen4_arrays_cache_ragged_trim(self):
        from mlx_lm.models.qwen4_exp import Qwen4ArraysCache

        # Two state slots, staged the way the PLE+GDN pair combines them.
        cache = Qwen4ArraysCache(2)
        cache.start_speculation()
        before = [mx.array([[1], [2]]), mx.array([[10], [20]])]
        cache.cache = [mx.array(a) for a in before]

        def gdn(m):
            return [before[0] + m]

        def ple(m):
            return [before[1] + 100 * m]

        cache.stage_ple_rollback(3, ple, [before[1]])
        cache.record_rollback(3, gdn, [before[0]])
        cache.cache = [before[0] + 3, before[1] + 300]

        cache.trim_ragged([0, 2])
        # Row 0 keeps all three tokens, row 1 rewinds to one.
        self.assertEqual(cache[0].reshape(-1).tolist(), [4, 3])
        self.assertEqual(cache[1].reshape(-1).tolist(), [310, 120])

    def test_batch_qsa_cache_declares_its_ledger_or_fails_loud(self):
        from mlx_lm.models.qwen4_exp import BatchQSAKVCache

        cache = BatchQSAKVCache([0, 0])
        values = _tokens(2, 4)
        cache.update_and_fetch(values, values)
        cache.update_index_keys(
            mx.broadcast_to(
                mx.arange(4, dtype=mx.float32).reshape(1, 4, 1), (2, 4, 2)
            )
        )
        try:
            cache.trim_ragged([0, 2])
        except RaggedTrimUnsupported:
            # Not wired yet: the ledger contract must be declared in
            # qwen4_exp.py before batched MTP can reject a draft.
            return
        # Once declared, the raw index keys must stay in step with the KV.
        self.assertEqual(cache.index_keys.shape[1], cache._idx)
        left = cache.left_padding.tolist()
        for row, offset in enumerate(cache.offset.tolist()):
            live = cache.index_keys[row, int(left[row]) : cache._idx, 0].tolist()
            self.assertEqual(len(live), int(offset))
            self.assertEqual(live, [float(p) for p in range(int(offset))])


if __name__ == "__main__":
    unittest.main()
