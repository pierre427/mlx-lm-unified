"""Contracts a merged batch cache owes its rows.

1. ``make_mask`` must exclude both the left pad prefix and the right pad tail.
2. ``merge`` must give every row its own storage, so n-way replication of one
   prefilled cache cannot make the samples share mutable state.

CPU-only: tiny arrays, no model loads.
"""

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    CacheList,
    KVCache,
)

_DEVICE = None


def setUpModule():
    global _DEVICE
    _DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_DEVICE)


def _rows_have_own_storage(array) -> bool:
    """False when the rows are a broadcast view of one row's buffer."""
    mx.eval(array)
    view = np.array(array, copy=False)
    return view.shape[0] < 2 or view.strides[0] != 0


def _kv_values(batch, length, heads=2, dim=4, base=0.0):
    rows = mx.arange(batch, dtype=mx.float32).reshape(batch, 1, 1, 1) * 1000
    positions = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    return mx.contiguous(
        mx.broadcast_to(rows + positions + base, (batch, heads, length, dim))
    )


class TestArraysCacheMask(unittest.TestCase):
    def test_merged_from_empty_masks_the_right_padding(self):
        """The fresh ragged batch path: merge empties, then prepare(lengths)."""
        cache = ArraysCache.merge([ArraysCache(4), ArraysCache(4)])
        self.assertEqual(cache.batch_size, 2)
        cache.prepare(lengths=[7, 3], right_padding=[0, 4])
        mask = cache.make_mask(7)

        self.assertEqual(mask.shape, (2, 7))
        self.assertEqual(mask[0].tolist(), [True] * 7)
        self.assertEqual(mask[1].tolist(), [True] * 3 + [False] * 4)

    def test_left_padding_only(self):
        cache = ArraysCache(4, left_padding=[2, 0])
        mask = cache.make_mask(5)
        self.assertEqual(mask[0].tolist(), [False, False, True, True, True])
        self.assertEqual(mask[1].tolist(), [True] * 5)

    def test_lengths_only(self):
        cache = ArraysCache(4)
        cache.prepare(lengths=[5, 2])
        mask = cache.make_mask(5)
        self.assertEqual(mask[0].tolist(), [True] * 5)
        self.assertEqual(mask[1].tolist(), [True, True, False, False, False])

    def test_both_bounds_apply(self):
        cache = ArraysCache(4, left_padding=[2, 0])
        cache.prepare(lengths=[5, 3])
        mask = cache.make_mask(5)
        self.assertEqual(mask[0].tolist(), [False, False, True, True, True])
        self.assertEqual(mask[1].tolist(), [True, True, True, False, False])

    def test_no_metadata_has_no_mask(self):
        self.assertIsNone(ArraysCache(4).make_mask(3))

    def test_bounds_track_chunked_prefill(self):
        cache = ArraysCache.merge([ArraysCache(2), ArraysCache(2)])
        cache.prepare(lengths=[7, 3])
        self.assertEqual(cache.make_mask(4)[1].tolist(), [True, True, True, False])
        cache.advance(4)
        # Row 1's prompt is exhausted: the whole second chunk is filler.
        self.assertEqual(cache.make_mask(3)[0].tolist(), [True, True, True])
        self.assertEqual(cache.make_mask(3)[1].tolist(), [False, False, False])

    def test_finalize_clears_both_bounds(self):
        cache = ArraysCache.merge([ArraysCache(2), ArraysCache(2)])
        cache.prepare(lengths=[7, 3])
        cache.finalize()
        self.assertIsNone(cache.make_mask(3))

    def test_merge_leaves_an_unreached_slot_empty(self):
        one = ArraysCache(2)
        one.cache = [mx.arange(2.0).reshape(1, 2), None]
        merged = ArraysCache.merge([one, one])
        self.assertEqual(merged[0].shape, (2, 2))
        self.assertIsNone(merged[1])


class TestMergeRowIsolation(unittest.TestCase):
    """n-way replication: merge([cache] * n) must not share row storage."""

    N = 3

    def _prefilled_kv(self, cls=KVCache, length=6):
        cache = cls()
        values = _kv_values(1, length)
        cache.update_and_fetch(values, values)
        return cache

    def _prefilled_qsa(self, length=6):
        from mlx_lm.models.qwen4_exp import QSAKVCache

        cache = QSAKVCache()
        values = _kv_values(1, length)
        cache.update_and_fetch(values, values)
        cache.update_index_keys(
            mx.contiguous(
                mx.broadcast_to(
                    mx.arange(length, dtype=mx.float32).reshape(1, length, 1),
                    (1, length, 2),
                )
            )
        )
        return cache

    def _prefilled_arrays(self, cls=ArraysCache, slots=2):
        cache = cls(slots)
        cache.cache = [
            mx.arange(4.0).reshape(1, 4) + 10 * slot for slot in range(slots)
        ]
        return cache

    def test_kv_rows_diverge_independently(self):
        source = self._prefilled_kv()
        merged = source.merge([source] * self.N)

        self.assertIsNot(merged.keys, source.keys)
        self.assertTrue(_rows_have_own_storage(merged.keys))
        self.assertTrue(_rows_have_own_storage(merged.values))

        source_before = source.keys[0, 0, :6, 0].tolist()
        others_before = [
            merged.keys[r, 0, :6, 0].tolist() for r in range(1, self.N)
        ]
        # Diverge row 0 the way a batched decode does: a row-varying append.
        merged.update_and_fetch(_kv_values(self.N, 1, base=90), _kv_values(self.N, 1, base=90))
        merged.trim_ragged([1] + [0] * (self.N - 1))

        self.assertEqual(source.keys[0, 0, :6, 0].tolist(), source_before)
        for offset, before in enumerate(others_before):
            row = offset + 1
            live = merged.keys[row, 0, : merged._idx, 0].tolist()
            self.assertEqual(live[:6], before)

    def test_qsa_index_keys_are_not_shared(self):
        source = self._prefilled_qsa()
        merged = source.merge([source] * self.N)

        self.assertIsNot(merged.index_keys, source.index_keys)
        self.assertTrue(_rows_have_own_storage(merged.keys))
        self.assertTrue(_rows_have_own_storage(merged.index_keys))

        source_before = source.index_keys[0, :, 0].tolist()
        others_before = [
            merged.index_keys[r, :, 0].tolist() for r in range(1, self.N)
        ]
        merged.index_keys[0:1] = mx.full(merged.index_keys[0:1].shape, -99.0)

        self.assertEqual(source.index_keys[0, :, 0].tolist(), source_before)
        for offset, before in enumerate(others_before):
            self.assertEqual(
                merged.index_keys[offset + 1, :, 0].tolist(), before
            )

    def test_arrays_cache_rows_are_not_shared(self):
        source = self._prefilled_arrays()
        merged = source.merge([source] * self.N)

        for slot, entry in enumerate(merged.cache):
            self.assertIsNot(entry, source.cache[slot])
            self.assertTrue(_rows_have_own_storage(entry))

        source_before = source.cache[0].tolist()
        others_before = [merged.cache[0][r].tolist() for r in range(1, self.N)]
        merged.cache[0][0:1] = mx.full((1, 4), -99.0)

        self.assertEqual(source.cache[0].tolist(), source_before)
        for offset, before in enumerate(others_before):
            self.assertEqual(merged.cache[0][offset + 1].tolist(), before)

    def test_qwen4_arrays_cache_keeps_its_type_and_rows(self):
        from mlx_lm.models.qwen4_exp import Qwen4ArraysCache

        source = self._prefilled_arrays(Qwen4ArraysCache, slots=4)
        merged = source.merge([source] * self.N)
        self.assertIsInstance(merged, Qwen4ArraysCache)
        for slot, entry in enumerate(merged.cache):
            self.assertIsNot(entry, source.cache[slot])
            self.assertTrue(_rows_have_own_storage(entry))

    def test_cache_list_delegates_row_isolation(self):
        source = CacheList(self._prefilled_kv(), self._prefilled_arrays())
        merged = source.merge([source] * self.N)

        kv, arrays = merged.caches
        self.assertTrue(_rows_have_own_storage(kv.keys))
        self.assertTrue(_rows_have_own_storage(arrays.cache[0]))
        self.assertIsNot(kv.keys, source.caches[0].keys)
        self.assertIsNot(arrays.cache[0], source.caches[1].cache[0])

        arrays_before = source.caches[1].cache[0].tolist()
        arrays.cache[0][0:1] = mx.full((1, 4), -99.0)
        self.assertEqual(source.caches[1].cache[0].tolist(), arrays_before)

    def test_single_row_merge_is_write_isolated(self):
        """B=1 merge can share the source buffer; writes must still not leak.

        ``mx.zeros`` fully overwritten by one slice assign can hand back the
        assigned buffer, so the merged array and the source can share
        storage. MLX copies on a slice write while that storage is shared,
        which is what keeps the source cache intact — pin it here so a change
        in that behaviour is caught rather than silently corrupting a stored
        prompt cache.
        """
        source = self._prefilled_kv()
        merged = source.merge([source])
        before = source.keys[0, 0, :6, 0].tolist()
        merged.keys[0:1, :, 0:1] = mx.full((1, 2, 1, 4), -99.0)
        self.assertEqual(source.keys[0, 0, :6, 0].tolist(), before)

        arrays = self._prefilled_arrays()
        merged_arrays = arrays.merge([arrays])
        arrays_before = arrays.cache[0].tolist()
        merged_arrays.cache[0][0:1] = mx.full((1, 4), -99.0)
        self.assertEqual(arrays.cache[0].tolist(), arrays_before)

    def test_replicated_rows_start_equal(self):
        """Replication is only useful if the rows begin identical."""
        source = self._prefilled_qsa()
        merged = source.merge([source] * self.N)
        first_kv = merged.keys[0].tolist()
        first_index = merged.index_keys[0].tolist()
        for row in range(1, self.N):
            self.assertEqual(merged.keys[row].tolist(), first_kv)
            self.assertEqual(merged.index_keys[row].tolist(), first_index)
        self.assertEqual(merged.offset.tolist(), [6] * self.N)
        self.assertEqual(merged.left_padding.tolist(), [0] * self.N)


if __name__ == "__main__":
    unittest.main()
