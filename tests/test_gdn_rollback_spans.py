# Copyright © 2026 Apple Inc.

"""GDN rollback staging under a right-padded speculative slab.

Qwen3.5 and Qwen3-Next run GatedDeltaNet on a bare ``ArraysCache``. The old
staging predicate disarmed whenever a mask or padding metadata was present,
which is exactly the uniform-width verify forward that has to roll back. It
now gates on ``ArraysCache.rollback_spans``, and the cache credits each row
its own span rather than the slab width.

CPU-only: tiny synthetic layers, no model loads.
"""

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
from mlx_lm.models.qwen3_next import ModelArgs as NextArgs
from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet

_DEVICE = None
HIDDEN = 128


def setUpModule():
    global _DEVICE
    _DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_DEVICE)


def _qwen35_layer():
    mx.random.seed(3)
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


def _qwen3next_layer():
    mx.random.seed(5)
    layer = Qwen3NextGatedDeltaNet(
        NextArgs(
            model_type="qwen3_next",
            hidden_size=HIDDEN,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            num_experts=2,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[],
            moe_intermediate_size=64,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            max_position_embeddings=512,
            head_dim=32,
        )
    )
    mx.eval(layer.parameters())
    return layer


def _inputs(batch, steps, seed):
    return mx.random.normal(
        (batch, steps, HIDDEN), key=mx.random.key(seed)
    ).astype(mx.float32) * 0.3


LAYERS = (("qwen3_5", _qwen35_layer), ("qwen3_next", _qwen3next_layer))


class TestGDNRollbackSpans(unittest.TestCase):
    PROMPT = 5
    SPANS = [3, 1]

    def _prime(self, layer, batch):
        """An unpadded prefill, the way speculation is entered."""
        cache = ArraysCache(2)
        prompt = _inputs(batch, self.PROMPT, seed=7)
        mx.eval(layer(prompt, mask=None, cache=cache))
        return cache, prompt

    def _slab(self, layer, cache, drafts):
        """One right-padded verify forward of width max(SPANS)."""
        width = drafts.shape[1]
        cache.prepare(
            lengths=self.SPANS,
            right_padding=[width - s for s in self.SPANS],
        )
        mask = cache.make_mask(width)
        mx.eval(layer(drafts, mask=mask, cache=cache))
        cache.finalize()

    def test_padded_slab_records_per_row_spans(self):
        for name, build in LAYERS:
            with self.subTest(model=name):
                layer = build()
                cache, _ = self._prime(layer, batch=2)
                cache.start_speculation()
                drafts = _inputs(2, max(self.SPANS), seed=11)
                self._slab(layer, cache, drafts)

                self.assertEqual(len(cache._rollbacks), 1)
                record = cache._rollbacks[-1]
                self.assertEqual(record.num_tokens, max(self.SPANS))
                # Row 1 advanced one token through a three-wide slab.
                self.assertEqual(record.depths, self.SPANS)
                self.assertEqual(cache._row_capacity(2), self.SPANS)

    def test_unpadded_forward_still_records_uniformly(self):
        for name, build in LAYERS:
            with self.subTest(model=name):
                layer = build()
                cache, _ = self._prime(layer, batch=2)
                cache.start_speculation()
                mx.eval(layer(_inputs(2, 3, seed=13), mask=None, cache=cache))
                record = cache._rollbacks[-1]
                self.assertIsNone(record.depths)
                self.assertEqual(cache._row_capacity(2), [3, 3])

    def test_leading_pads_still_disarm(self):
        for name, build in LAYERS:
            with self.subTest(model=name):
                layer = build()
                cache, _ = self._prime(layer, batch=2)
                cache.start_speculation()
                cache.left_padding = mx.array([0, 2])
                mx.eval(layer(_inputs(2, 3, seed=17), mask=None, cache=cache))
                # A row whose tokens start later cannot be replayed by a
                # scalar depth, so nothing is recorded rather than lying.
                self.assertEqual(len(cache._rollbacks), 0)

    def test_ragged_rewind_matches_per_row_solo_decode(self):
        """Reject a different draft count per row and land where solo does."""
        for name, build in LAYERS:
            for accepted in ((0, 0), (2, 0), (1, 1), (3, 1)):
                with self.subTest(model=name, accepted=accepted):
                    layer = build()
                    cache, prompt = self._prime(layer, batch=2)
                    cache.start_speculation()
                    drafts = _inputs(2, max(self.SPANS), seed=11)
                    self._slab(layer, cache, drafts)
                    cache.trim_ragged(
                        [s - a for s, a in zip(self.SPANS, accepted)]
                    )

                    for row, take in enumerate(accepted):
                        solo_layer = build()
                        solo = ArraysCache(2)
                        mx.eval(
                            solo_layer(
                                prompt[row : row + 1], mask=None, cache=solo
                            )
                        )
                        if take:
                            mx.eval(
                                solo_layer(
                                    drafts[row : row + 1, :take],
                                    mask=None,
                                    cache=solo,
                                )
                            )
                        for slot, what in ((0, "conv"), (1, "recurrent state")):
                            got = np.asarray(cache[slot][row : row + 1])
                            ref = np.asarray(solo[slot])
                            scale = float(np.abs(ref).max()) or 1.0
                            self.assertLess(
                                float(np.abs(got - ref).max()) / scale,
                                3e-3,
                                f"{name} row {row} {what}",
                            )


if __name__ == "__main__":
    unittest.main()
