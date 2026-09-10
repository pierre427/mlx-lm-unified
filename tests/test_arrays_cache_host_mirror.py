"""Host-mirror contracts for recurrent-cache batch membership changes.

CPU-only and model-free.  These tests treat a call through ``_host_vector``
with no identity-valid mirror as a fallback device read; on GPU that path's
``tolist()`` is the synchronization this lifecycle optimization avoids.
"""

import unittest

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache


_DEVICE = None


def setUpModule():
    global _DEVICE
    _DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_DEVICE)


class _FallbackSpy(ArraysCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_reads = {"lengths": 0, "left_padding": 0}

    def _host_vector(self, field, cached):
        value = getattr(self, field)
        if value is not None and (cached is None or cached[0] is not value):
            self.fallback_reads[field] += 1
        return super()._host_vector(field, cached)


def _cache(rows, *, cls=_FallbackSpy, padding=None, lengths=None, base=0):
    cache = cls(2, left_padding=padding)
    values = mx.arange(rows * 3, dtype=mx.float32).reshape(rows, 3) + base
    cache.cache = [values, values + 100]
    if lengths is not None:
        cache.prepare(lengths=lengths)
    return cache


class TestArraysCacheHostMirrors(unittest.TestCase):
    def assertMirror(self, cache, field, expected):
        mirror = getattr(cache, f"_host_{field}")
        value = getattr(cache, field)
        self.assertIsNotNone(mirror)
        self.assertIs(mirror[0], value)
        self.assertEqual(mirror[1], expected)

    def test_constructor_and_fresh_merge_need_no_fallback(self):
        direct = _cache(2, padding=[3, 1])
        direct.advance(1)
        self.assertEqual(direct.fallback_reads, {"lengths": 0, "left_padding": 0})
        self.assertMirror(direct, "left_padding", [2, 0])

        fresh = _FallbackSpy.merge([_FallbackSpy(2), _FallbackSpy(2)])
        fresh.prepare(lengths=[4, 2])
        fresh.cache = [mx.zeros((2, 1)), mx.zeros((2, 1))]
        fresh.advance(1)
        self.assertEqual(fresh.fallback_reads, {"lengths": 0, "left_padding": 0})
        self.assertMirror(fresh, "left_padding", [-1, -1])
        self.assertMirror(fresh, "lengths", [3, 1])

    def test_filter_transforms_both_valid_mirrors(self):
        cache = _cache(3, padding=[3, 0, 1], lengths=[5, 4, 2])
        original = [entry.tolist() for entry in cache.cache]

        cache.filter([2, 0])
        self.assertMirror(cache, "left_padding", [1, 3])
        self.assertMirror(cache, "lengths", [2, 5])
        cache.advance(1)

        self.assertEqual(cache.fallback_reads, {"lengths": 0, "left_padding": 0})
        self.assertEqual(cache.left_padding.tolist(), [0, 2])
        self.assertEqual(cache.lengths.tolist(), [1, 4])
        for before, after in zip(original, cache.cache):
            self.assertEqual(after.tolist(), [before[2], before[0]])

    def test_filter_fails_closed_for_device_indices(self):
        cache = _cache(3, padding=[3, 0, 1], lengths=[5, 4, 2])
        cache.filter(mx.array([2, 0]))

        self.assertIsNone(cache._host_left_padding)
        self.assertIsNone(cache._host_lengths)
        cache.advance(1)
        self.assertEqual(cache.fallback_reads, {"lengths": 1, "left_padding": 1})
        self.assertEqual(cache.left_padding.tolist(), [0, 2])
        self.assertEqual(cache.lengths.tolist(), [1, 4])

    def test_filter_fails_closed_for_stale_source_mirror(self):
        cache = _cache(3, padding=[0, 0, 0], lengths=[5, 4, 2])
        # Identity mismatch: the cached values no longer prove this array.
        cache.lengths = mx.array([9, 8, 7])
        cache.filter([1, 2])

        self.assertIsNone(cache._host_lengths)
        self.assertMirror(cache, "left_padding", [0, 0])
        cache.advance(1)
        self.assertEqual(cache.fallback_reads, {"lengths": 1, "left_padding": 0})
        self.assertEqual(cache.lengths.tolist(), [7, 6])

    def test_extend_concatenates_valid_mirrors(self):
        left = _cache(2, padding=[2, 0], lengths=[5, 3], base=0)
        right = _cache(1, padding=[4], lengths=[7], base=1000)

        left.extend(right)
        self.assertMirror(left, "left_padding", [2, 0, 4])
        self.assertMirror(left, "lengths", [5, 3, 7])
        left.advance(1)

        self.assertEqual(left.fallback_reads, {"lengths": 0, "left_padding": 0})
        self.assertEqual(left.left_padding.tolist(), [1, -1, 3])
        self.assertEqual(left.lengths.tolist(), [4, 2, 6])
        self.assertEqual(left.cache[0][-1].tolist(), right.cache[0][0].tolist())

    def test_extend_knows_zero_rows_synthesized_for_missing_side(self):
        left = _cache(2)
        right = _cache(1, padding=[3], lengths=[6], base=1000)

        left.extend(right)
        self.assertMirror(left, "left_padding", [0, 0, 3])
        self.assertMirror(left, "lengths", [0, 0, 6])
        left.advance(1)

        self.assertEqual(left.fallback_reads, {"lengths": 0, "left_padding": 0})
        self.assertEqual(left.left_padding.tolist(), [-1, -1, 2])
        self.assertEqual(left.lengths.tolist(), [-1, -1, 5])

    def test_extend_fails_closed_when_either_source_is_unproven(self):
        left = _cache(2, padding=[0, 1], lengths=[5, 3])
        right = _cache(1, padding=[2], lengths=[7], base=1000)
        right._host_lengths = None

        left.extend(right)
        self.assertIsNone(left._host_lengths)
        self.assertMirror(left, "left_padding", [0, 1, 2])
        left.advance(1)

        self.assertEqual(left.fallback_reads, {"lengths": 1, "left_padding": 0})
        self.assertEqual(left.lengths.tolist(), [4, 2, 6])

    def test_lifecycle_matches_fallback_path_and_preserves_subclass(self):
        class DerivedArraysCache(_FallbackSpy):
            pass

        fast = _cache(
            3,
            cls=DerivedArraysCache,
            padding=[2, 0, 1],
            lengths=[6, 3, 5],
        )
        fallback = _cache(
            3,
            cls=DerivedArraysCache,
            padding=[2, 0, 1],
            lengths=[6, 3, 5],
        )
        addition_fast = _cache(
            1, cls=DerivedArraysCache, padding=[0], lengths=[4], base=1000
        )
        addition_fallback = _cache(
            1, cls=DerivedArraysCache, padding=[0], lengths=[4], base=1000
        )

        for cache in (fast, fallback):
            cache.filter([2, 0])
        # Force the old fallback-read lifecycle only in the reference arm.
        fallback._host_left_padding = None
        fallback._host_lengths = None
        for cache, addition in (
            (fast, addition_fast),
            (fallback, addition_fallback),
        ):
            cache.advance(1)
            cache.extend(addition)
            cache.advance(2)
            mx.eval(cache.state)

        self.assertIsInstance(fast, DerivedArraysCache)
        self.assertEqual(fast.fallback_reads, {"lengths": 0, "left_padding": 0})
        self.assertEqual(fallback.fallback_reads, {"lengths": 1, "left_padding": 1})
        self.assertEqual(fast.left_padding.tolist(), fallback.left_padding.tolist())
        self.assertEqual(fast.lengths.tolist(), fallback.lengths.tolist())
        self.assertTrue(mx.array_equal(fast.make_mask(8), fallback.make_mask(8)).item())
        for fast_entry, fallback_entry in zip(fast.cache, fallback.cache):
            self.assertTrue(mx.array_equal(fast_entry, fallback_entry).item())


if __name__ == "__main__":
    unittest.main()
