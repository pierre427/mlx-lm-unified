import os
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import BatchKVCache
from mlx_lm.models import qwen4_qsa_m1 as direct_m1
from mlx_lm.models import qwen4_qsa_nax as nax


def _cache_prefix(total=32, dim=256):
    cache = BatchKVCache([0])
    values = mx.arange(2 * total * dim, dtype=mx.float32).reshape(
        1, 2, total, dim
    )
    keys = (values / 1024).astype(mx.bfloat16)
    vals = (values / 2048).astype(mx.bfloat16)
    cache.update_and_fetch(keys, vals)
    return cache.keys_and_values()


def _inputs(length):
    ids = mx.zeros((1, length, 8), dtype=mx.uint32)
    counts = mx.ones((1, length), dtype=mx.uint32)
    n_sel = mx.zeros((1, length), dtype=mx.uint32)
    q_pos = mx.arange(length, dtype=mx.int32).reshape(1, length)
    left_pad = mx.zeros((1,), dtype=mx.int32)
    return ids, counts, n_sel, q_pos, left_pad


class TestQSAKVStrideDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def test_nax_passes_cache_prefix_views_without_copy(self):
        k_view, v_view = _cache_prefix()
        q = mx.zeros((1, 24, 3, 256), dtype=mx.bfloat16)
        ids, counts, n_sel, q_pos, left_pad = _inputs(3)
        captured = {}

        def dispatch(**kwargs):
            captured["inputs"] = kwargs["inputs"]
            return (mx.zeros((1, 24, 3, 256), dtype=mx.float32),)

        with mock.patch.object(nax, "_KERNEL", side_effect=dispatch):
            nax.nax_qsa_attention(
                q,
                k_view,
                v_view,
                ids,
                counts,
                n_sel,
                q_pos,
                left_pad,
                scale=256**-0.5,
                u_width=8,
                total=32,
                n_kv_heads=2,
            )
        self.assertIs(captured["inputs"][1], k_view)
        self.assertIs(captured["inputs"][2], v_view)

    def test_direct_m1_passes_cache_prefix_views_without_copy(self):
        k_view, v_view = _cache_prefix()
        q = mx.zeros((1, 24, 1, 256), dtype=mx.bfloat16)
        ids, counts, n_sel, q_pos, left_pad = _inputs(1)
        captured = {}

        def dispatch(**kwargs):
            captured["inputs"] = kwargs["inputs"]
            return (mx.zeros((1, 24, 1, 256), dtype=mx.bfloat16),)

        with mock.patch.object(direct_m1, "_kernel", return_value=dispatch):
            direct_m1.qwen4_qsa_direct_m1(
                q,
                k_view,
                v_view,
                ids,
                counts,
                n_sel,
                q_pos,
                left_pad,
                scale=256**-0.5,
                u_width=8,
                total=32,
                n_kv_heads=2,
            )
        self.assertIs(captured["inputs"][1], k_view)
        self.assertIs(captured["inputs"][2], v_view)


@unittest.skipUnless(
    mx.metal.is_available()
    and os.getenv("MLX_QWEN4_QSA_STRIDE_METAL_TESTS") == "1",
    "requires the explicit QSA stride Metal test gate",
)
class TestQSAKVStrideMetal(unittest.TestCase):
    def test_cache_prefix_views_match_contiguous_inputs(self):
        previous = mx.default_device()
        mx.set_default_device(mx.gpu)
        try:
            k_view, v_view = _cache_prefix()
            k_contiguous = mx.contiguous(k_view)
            v_contiguous = mx.contiguous(v_view)

            q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
            ids, counts, n_sel, q_pos, left_pad = _inputs(3)
            nax_view = nax.nax_qsa_attention(
                q,
                k_view,
                v_view,
                ids,
                counts,
                n_sel,
                q_pos,
                left_pad,
                scale=256**-0.5,
                u_width=8,
                total=32,
                n_kv_heads=2,
            )
            nax_contiguous = nax.nax_qsa_attention(
                q,
                k_contiguous,
                v_contiguous,
                ids,
                counts,
                n_sel,
                q_pos,
                left_pad,
                scale=256**-0.5,
                u_width=8,
                total=32,
                n_kv_heads=2,
            )

            q_m1 = q[:, :, :1]
            m1_inputs = _inputs(1)
            m1_view = direct_m1.qwen4_qsa_direct_m1(
                q_m1,
                k_view,
                v_view,
                *m1_inputs,
                scale=256**-0.5,
                u_width=8,
                total=32,
                n_kv_heads=2,
            )
            m1_contiguous = direct_m1.qwen4_qsa_direct_m1(
                q_m1,
                k_contiguous,
                v_contiguous,
                *m1_inputs,
                scale=256**-0.5,
                u_width=8,
                total=32,
                n_kv_heads=2,
            )
            mx.eval(nax_view, nax_contiguous, m1_view, m1_contiguous)
            np.testing.assert_array_equal(
                np.asarray(nax_view), np.asarray(nax_contiguous)
            )
            np.testing.assert_array_equal(
                np.asarray(m1_view.view(mx.uint16)),
                np.asarray(m1_contiguous.view(mx.uint16)),
            )
        finally:
            mx.set_default_device(previous)
