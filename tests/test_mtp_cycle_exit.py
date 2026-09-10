"""The MTP draft cycle has one exit, and the loop takes it (CPU-only).

``Model.mtp_start_cycle`` arms QSA top-k sharing for one draft cycle;
``Model.mtp_end_cycle`` is its paired exit. The draft/verify loop calls the
exit AFTER it rewinds the drafted span, and the ordering is load-bearing in
both directions:

  * called BEFORE the rewind, the hook sees the drafted KV the rewind is about
    to remove and reports it -- the 13-vs-12 ledger desync;
  * not called at all, a head cache whose own rewind does not release the
    cycle carries a live shared index set into the next forward, or into the
    sidecar a request stores.

Runs on the tiny ``qwen4_exp`` model from ``test_qwen4_exp``.
"""

import unittest

import mlx.core as mx

from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import KVCache


class LooseQSAKVCache(qwen4_exp.QSAKVCache):
    """A head cache whose rewind does NOT end the cycle.

    The stock cache releases from ``trim`` as well, which would hide whether
    the LOOP takes the exit. Ending the cycle is the loop's obligation, so it
    is tested against a cache that does not do it for free.
    """

    def trim(self, n):
        return KVCache.trim(self, n)


class TestDraftCycleExit(unittest.TestCase):
    K = 2  # sharing engages only for k > 1
    PROMPT = list(range(1, 13))

    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)
        from test_qwen4_exp import tiny_args

        cls._args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def _model(self, cache_cls=None):
        mx.random.seed(3)
        model = qwen4_exp.Model(
            qwen4_exp.ModelArgs(
                model_type="qwen4_exp", text_config=self._args.__dict__
            )
        )
        if cache_cls is not None:
            model.make_mtp_cache = lambda window_size=None, sink_size=4: [
                cache_cls() for _ in model.mtp.layers
            ]
        return model

    def _generate(self, model, *, max_tokens, stop_after=None, state_out=None):
        gen = self_mtp_generate_step(
            mx.array(self.PROMPT, dtype=mx.uint32),
            model,
            num_draft=self.K,
            max_tokens=max_tokens,
            sampling_temp=0.0,
            persistent_mtp=True,
            mtp_share_qsa_indices=True,
            stats=HybridStats(),
            mtp_state_out=state_out,
        )
        tokens = []
        try:
            for token, _lp, _from_draft in gen:
                tokens.append(int(token))
                if stop_after is not None and len(tokens) == stop_after:
                    break
        finally:
            gen.close()
        return tokens

    def _head(self, state_out):
        mtp_cache, _tail_hidden = state_out["state"]
        return mtp_cache[0]

    def test_a_shared_cycle_generates_without_reporting_un_ledgered_kv(self):
        # The ordering guard: ending the cycle before the rewind raises here,
        # naming the drafted span the rewind was about to remove.
        tokens = self._generate(self._model(), max_tokens=8)
        self.assertEqual(len(tokens), 8)

    def test_an_aborted_run_leaves_the_head_cache_disarmed(self):
        out = {}
        model = self._model(LooseQSAKVCache)
        self._generate(model, max_tokens=64, stop_after=3, state_out=out)
        head = self._head(out)
        self.assertIsInstance(head, LooseQSAKVCache)
        self.assertFalse(head._mtp_share_topk, "the cycle stayed armed")
        self.assertIsNone(head._mtp_shared_topk, "a stale index set survived")

    def test_an_aborted_run_leaves_the_ledger_spanning_the_cursor(self):
        out = {}
        model = self._model(LooseQSAKVCache)
        self._generate(model, max_tokens=64, stop_after=3, state_out=out)
        head = self._head(out)
        self.assertIsNotNone(head.index_keys)
        self.assertEqual(head.index_keys.shape[1], head.offset)

    def test_the_stock_cache_ends_clean_too(self):
        # The live path: the same post-condition with the production cache.
        out = {}
        self._generate(self._model(), max_tokens=64, stop_after=3, state_out=out)
        head = self._head(out)
        self.assertFalse(head._mtp_share_topk)
        self.assertIsNone(head._mtp_shared_topk)
        self.assertEqual(head.index_keys.shape[1], head.offset)

    def test_the_exit_does_not_change_the_tokens(self):
        # Sharing is a speed lever: ending its cycle must not move the stream.
        shared = self._generate(self._model(), max_tokens=8)
        model = self._model()
        model.mtp_start_cycle = lambda cache, share=False: (
            qwen4_exp.Model.mtp_start_cycle(model, cache, False)
        )
        plain = self._generate(model, max_tokens=8)
        self.assertEqual(shared, plain)

    def test_a_model_without_the_hook_still_runs(self):
        # Most MTP models define no cycle hooks at all; the call is a no-op
        # for them.
        from test_qwen3_5_mtp import tiny_args as qwen3_5_args

        from mlx_lm.models.qwen3_5 import TextModel

        mx.random.seed(0)
        model = TextModel(qwen3_5_args())
        mx.eval(model.parameters())
        self.assertIsNone(getattr(model, "mtp_end_cycle", None))
        self.assertEqual(len(self._generate(model, max_tokens=6)), 6)


if __name__ == "__main__":
    unittest.main()
