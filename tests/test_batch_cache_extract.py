import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    BatchRotatingQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    RotatingQuantizedKVCache,
)


class TestBatchCacheExtract(unittest.TestCase):
    def _prefix_cache(self, cls, **kwargs):
        cache = cls(**kwargs)
        values = mx.broadcast_to(
            mx.arange(92, dtype=mx.float32).reshape(1, 1, 92, 1),
            (1, 2, 92, 4),
        )
        cache.update_and_fetch(values, values)
        cache.trim(40)
        return cache

    def test_extract_excludes_pending_right_padding(self):
        for batch_cls, single_cls, kwargs in (
            (BatchKVCache, KVCache, {}),
            (BatchRotatingKVCache, RotatingKVCache, {"max_size": 1024}),
        ):
            with self.subTest(batch_cls=batch_cls.__name__):
                caches = [self._prefix_cache(single_cls, **kwargs) for _ in range(4)]
                batch = batch_cls.merge(caches)

                lengths = [270, 280, 272, 271]
                width = max(lengths)
                batch.prepare(
                    lengths=lengths,
                    right_padding=[width - length for length in lengths],
                )
                values = mx.broadcast_to(
                    mx.arange(width, dtype=mx.float32).reshape(1, 1, width, 1),
                    (4, 2, width, 4),
                )
                batch.update_and_fetch(values, values)

                before_finalize = batch.extract(0)
                self.assertEqual(before_finalize.keys.shape[2], 52 + 270)
                self.assertEqual(before_finalize.offset, 52 + 270)

                batch.finalize()
                after_finalize = batch.extract(0)
                self.assertEqual(after_finalize.keys.shape[2], 52 + 270)
                self.assertEqual(after_finalize.offset, 52 + 270)
                self.assertTrue(
                    mx.array_equal(before_finalize.keys, after_finalize.keys)
                )
                self.assertTrue(
                    mx.array_equal(before_finalize.values, after_finalize.values)
                )

    def test_quantized_extract_excludes_pending_right_padding(self):
        # mlx-lm#1746 class: an extract in the prepare()..finalize() window
        # must not save right-padding filler cells into a prompt cache.
        def prefix_cache(cls, **kwargs):
            cache = cls(**kwargs)
            values = mx.broadcast_to(
                mx.arange(92, dtype=mx.float32).reshape(1, 1, 92, 1),
                (1, 2, 92, 32),
            )
            cache.update_and_fetch(values, values)
            cache.trim(40)
            return cache

        for batch_cls, single_cls, kwargs in (
            (
                BatchQuantizedKVCache,
                QuantizedKVCache,
                {"group_size": 32, "bits": 8},
            ),
            (
                BatchRotatingQuantizedKVCache,
                RotatingQuantizedKVCache,
                {"max_size": 1024, "group_size": 32, "bits": 8},
            ),
        ):
            with self.subTest(batch_cls=batch_cls.__name__):
                caches = [prefix_cache(single_cls, **kwargs) for _ in range(4)]
                batch = batch_cls.merge(caches)

                lengths = [270, 280, 272, 271]
                width = max(lengths)
                batch.prepare(
                    lengths=lengths,
                    right_padding=[width - length for length in lengths],
                )
                values = mx.broadcast_to(
                    mx.arange(width, dtype=mx.float32).reshape(1, 1, width, 1),
                    (4, 2, width, 32),
                )
                batch.update_and_fetch(values, values)

                before_finalize = batch.extract(0)
                self.assertEqual(before_finalize.keys[0].shape[2], 52 + 270)
                self.assertEqual(before_finalize.offset, 52 + 270)

                batch.finalize()
                after_finalize = batch.extract(0)
                self.assertEqual(after_finalize.keys[0].shape[2], 52 + 270)
                self.assertEqual(after_finalize.offset, 52 + 270)
                for before, after in zip(
                    (*before_finalize.keys, *before_finalize.values),
                    (*after_finalize.keys, *after_finalize.values),
                ):
                    self.assertTrue(mx.array_equal(before, after))


if __name__ == "__main__":
    unittest.main()
