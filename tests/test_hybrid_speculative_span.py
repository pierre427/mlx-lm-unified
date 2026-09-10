"""Focused lifecycle checks for adaptive-PLD verify-cliff span routing."""

import unittest

import mlx.core as mx

from mlx_lm.hybrid_speculative import HybridStats, adaptive_pld_generate_step


class _Cache:
    def __init__(self):
        self.offset = 0
        self.speculating = False

    def start_speculation(self):
        self.speculating = True

    def stop_speculation(self):
        self.speculating = False

    def is_trimmable(self):
        return self.speculating

    def trim(self, n):
        self.offset -= n
        return n


class _Model:
    def __init__(self):
        self.input_lengths = []

    def __call__(self, x, cache=None):
        self.input_lengths.append(x.shape[-1])
        cache[0].offset += x.shape[-1]
        return mx.zeros((x.shape[0], x.shape[1], 64))


def _first_verify(history, *, cliff_aware_span):
    cache = _Cache()
    model = _Model()
    stats = HybridStats()
    generator = adaptive_pld_generate_step(
        mx.array(history[-1:]),
        model,
        prompt_cache=[cache],
        history_prompt=mx.array(history),
        max_tokens=16,
        min_match=3,
        max_span=10,
        cliff_aware_span=cliff_aware_span,
        stats=stats,
    )
    next(generator)
    generator.close()
    return model.input_lengths[0], stats, cache


class TestAdaptivePLDCliffAwareSpan(unittest.TestCase):
    def test_default_off_and_extend_to_sixteen(self):
        history = [1, 2, 3] + list(range(10, 30)) + [1, 2, 3]

        rows, stats, cache = _first_verify(history, cliff_aware_span=False)
        self.assertEqual(rows, 11)
        self.assertEqual(stats.verify_span_hist, {11: 1})
        self.assertEqual(stats.span_extend_cycles, 0)
        self.assertFalse(cache.speculating)

        rows, stats, cache = _first_verify(history, cliff_aware_span=True)
        self.assertEqual(rows, 16)
        self.assertEqual(stats.verify_span_hist, {16: 1})
        self.assertEqual(stats.span_extend_cycles, 1)
        self.assertEqual(stats.span_extend_tokens, 5)
        self.assertFalse(cache.speculating)

    def test_short_continuation_shrinks_to_eight(self):
        history = [1, 2, 3] + list(range(10, 20)) + [1, 2, 3]
        rows, stats, cache = _first_verify(history, cliff_aware_span=True)
        self.assertEqual(rows, 8)
        self.assertEqual(stats.verify_span_hist, {8: 1})
        self.assertEqual(stats.span_snap_cycles, 1)
        self.assertEqual(stats.span_snap_tokens, 3)
        self.assertFalse(cache.speculating)


class TestAdaptivePLDLifecycleSemantics(unittest.TestCase):
    def test_max_tokens_zero_yields_nothing_and_does_no_work(self):
        cache = _Cache()
        model = _Model()
        generator = adaptive_pld_generate_step(
            mx.array([1, 2, 3]),
            model,
            prompt_cache=[cache],
            max_tokens=0,
        )
        self.assertEqual(list(generator), [])
        self.assertEqual(model.input_lengths, [])
        self.assertFalse(cache.speculating)
        self.assertEqual(cache.offset, 0)

    def test_early_close_counts_only_delivered_tokens(self):
        # The suffix [1,2,3] recurs, so retrieval proposes the zeros that
        # followed its first occurrence — which the zero-predicting model
        # accepts. The consumer closes after the FIRST delivered token
        # (e.g. EOS): telemetry must count exactly one accepted token and
        # no bonus.
        history = [1, 2, 3] + [0] * 6 + [1, 2, 3]
        cache = _Cache()
        model = _Model()
        stats = HybridStats()
        generator = adaptive_pld_generate_step(
            mx.array(history[-1:]),
            model,
            prompt_cache=[cache],
            history_prompt=mx.array(history),
            max_tokens=16,
            min_match=2,
            max_span=8,
            stats=stats,
        )
        tok, _logprobs, from_retrieval = next(generator)
        self.assertEqual(tok, 0)
        self.assertTrue(from_retrieval)
        generator.close()

        self.assertGreaterEqual(stats.retrieval_proposed, 2)
        self.assertEqual(stats.retrieval_accepted, 1)
        self.assertEqual(stats.bonus_tokens, 0)
        self.assertEqual(stats.plain_tokens, 0)
        self.assertEqual(stats.total_emitted, 1)
        self.assertFalse(cache.speculating)


if __name__ == "__main__":
    unittest.main()
