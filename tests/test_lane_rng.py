"""Per-lane RNG keys for the self-MTP loop.

Batched decode shares one process. With every draw taken from the global
``mx.random`` stream, a lane that joins or leaves reorders every other lane's
draws, so a request's tokens depend on the traffic scheduled beside it — a
composition dependence created in the sampling layer itself, which the batch
plan's P1 contract cannot absorb (``wiki/docs/plans/qwen38-mtp-batch-
composition.md`` §3). A lane carries a ``LaneRNG`` and splits it per draw
instead.

The gates here are the contract:

* reproducibility — same lane key, same draws, whatever the global stream does;
* independence — different lane keys draw independently, and keyed draws leave
  the global stream untouched;
* join/leave — interleaving another lane's generation does not change a lane's
  tokens (with the power check that the global-stream path DOES change them);
* rollback — the key advances once per draw made and is never rewound, so the
  draw count depends on the schedule, not on acceptance;
* off-path identity — with no lane key, every helper draws from the global
  stream in the same order and shape as before lane keys existed.

CPU-only and tiny fixtures on purpose.
"""

import unittest

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    HybridStats,
    _accept_sampled_draft,
    _batched_residual_verify,
    _block_verify,
    _residual_sample,
    _sample_from_logprobs,
    self_mtp_generate_step,
)
from mlx_lm.sample_utils import LaneRNG, draw_key, transformed_logprobs


def _key_list(key):
    return tuple(key.tolist())


def _probe():
    """A draw from the global stream: its value pins the stream's position."""
    x = mx.random.uniform(shape=(3,))
    mx.eval(x)
    return tuple(x.tolist())


class TestLaneRNGPrimitive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def test_same_seed_reproduces_the_draw_sequence(self):
        a, b = LaneRNG(7), LaneRNG(7)
        for _ in range(16):
            self.assertEqual(_key_list(a.next_key()), _key_list(b.next_key()))
        self.assertEqual(a.draws, b.draws)

    def test_different_seeds_draw_independently(self):
        a, b = LaneRNG(1), LaneRNG(2)

        def draws(lane):
            return [
                mx.random.uniform(shape=(4,), key=lane.next_key()).tolist()
                for _ in range(32)
            ]

        da, db = draws(a), draws(b)
        self.assertNotEqual(da, db)
        # No positionwise collisions either: a shared or mirrored stream would
        # show them immediately.
        for x, y in zip(da, db):
            self.assertNotEqual(x, y)

    def test_subkeys_never_repeat(self):
        rng = LaneRNG(11)
        keys = {_key_list(rng.next_key()) for _ in range(500)}
        self.assertEqual(len(keys), 500)

    def test_re_deriving_the_seed_repeats_draws(self):
        # The footgun the design forbids, pinned as a power check: re-deriving
        # key(seed) per draw returns the same value every time.
        naive = [
            mx.random.uniform(shape=(2,), key=mx.random.key(5)).tolist()
            for _ in range(4)
        ]
        self.assertEqual(len(set(map(tuple, naive))), 1)
        rng = LaneRNG(5)
        carried = [
            mx.random.uniform(shape=(2,), key=rng.next_key()).tolist()
            for _ in range(4)
        ]
        self.assertEqual(len(set(map(tuple, carried))), 4)

    def test_no_rewind_api_and_monotone_draw_count(self):
        rng = LaneRNG(3)
        self.assertFalse(hasattr(rng, "rewind"))
        self.assertFalse(hasattr(rng, "reset"))
        counts = []
        for _ in range(5):
            rng.next_key()
            counts.append(rng.draws)
        self.assertEqual(counts, [1, 2, 3, 4, 5])

    def test_from_key_resumes_instead_of_repeating(self):
        # Snapshot/restore (the APC sidecar path): a resumed lane must continue
        # its stream. Rebuilding from the CARRIED key does; rebuilding from the
        # seed would replay the first draws.
        rng = LaneRNG(21)
        first = [_key_list(rng.next_key()) for _ in range(3)]
        snapshot, drawn = rng.key, rng.draws
        rest = [_key_list(rng.next_key()) for _ in range(3)]

        resumed = LaneRNG.from_key(snapshot, drawn)
        self.assertEqual(resumed.draws, drawn)
        self.assertEqual([_key_list(resumed.next_key()) for _ in range(3)], rest)
        self.assertNotEqual(rest, first)

        replayed = LaneRNG(21)
        self.assertEqual([_key_list(replayed.next_key()) for _ in range(3)], first)

    def test_fork_makes_independent_lanes_and_advances_the_parent(self):
        rng = LaneRNG(31)
        before = _key_list(rng.key)
        lanes = rng.fork(3)
        self.assertEqual(len(lanes), 3)
        self.assertNotEqual(_key_list(rng.key), before)
        keys = {_key_list(lane.key) for lane in lanes} | {_key_list(rng.key)}
        self.assertEqual(len(keys), 4)
        for lane in lanes:
            self.assertEqual(lane.draws, 0)
        with self.assertRaises(ValueError):
            rng.fork(0)

    def test_keyed_draws_leave_the_global_stream_untouched(self):
        mx.random.seed(17)
        solo = _probe()
        mx.random.seed(17)
        rng = LaneRNG(99)
        for _ in range(10):
            mx.eval(mx.random.uniform(shape=(5,), key=rng.next_key()))
        self.assertEqual(_probe(), solo)

    def test_draw_key_of_none_is_the_global_stream(self):
        self.assertIsNone(draw_key(None))


class TestOffPathIdentity(unittest.TestCase):
    """With no lane key every helper draws from the global stream in the same
    order and shape as before lane keys existed."""

    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def setUp(self):
        # Disjoint transformed supports: every draft is rejected, so the
        # residual branch (not the fallback) takes the correction.
        t = mx.array([3.0, 2.5, 2.0, -1.0, -1.5, -2.0])
        d = mx.array([-1.0, -1.5, -2.0, 3.0, 2.5, 2.0])
        self.tlp = transformed_logprobs(t, 0.8, top_k=3)
        self.dlp = transformed_logprobs(d, 0.8, top_k=3)

    def _stream_after(self, fn, seed=404):
        mx.random.seed(seed)
        fn()
        return _probe()

    def test_batched_residual_verify_consumes_one_uniform_then_one_sample(self):
        k = 2
        logprobs = mx.stack([self.tlp] * (k + 1))
        drafts = [3, 4]
        got = self._stream_after(
            lambda: _batched_residual_verify(
                logprobs, [self.dlp] * k, drafts, 0.8
            )
        )
        expected = self._stream_after(
            lambda: (
                mx.eval(mx.random.uniform(shape=(k,))),
                mx.eval(mx.random.categorical(self.tlp)),
            )
        )
        self.assertEqual(got, expected)

    def test_block_verify_consumes_one_uniform_then_one_sample(self):
        k = 2
        logprobs = mx.stack([self.tlp] * (k + 1))
        got = self._stream_after(
            lambda: _block_verify(logprobs, [self.dlp] * k, [3, 4], 0.8)
        )
        expected = self._stream_after(
            lambda: (
                mx.eval(mx.random.uniform(shape=(k,))),
                mx.eval(mx.random.categorical(self.tlp)),
            )
        )
        self.assertEqual(got, expected)

    def test_accept_sampled_draft_consumes_one_scalar_uniform(self):
        got = self._stream_after(
            lambda: _accept_sampled_draft(self.tlp, self.dlp, 3)
        )
        expected = self._stream_after(lambda: mx.eval(mx.random.uniform(shape=())))
        self.assertEqual(got, expected)

    def test_sample_from_logprobs_consumes_one_categorical(self):
        got = self._stream_after(lambda: _sample_from_logprobs(self.tlp, 0.8))
        expected = self._stream_after(
            lambda: mx.eval(mx.random.categorical(self.tlp))
        )
        self.assertEqual(got, expected)

    def test_residual_sample_consumes_one_categorical(self):
        got = self._stream_after(
            lambda: _residual_sample(self.tlp, self.dlp, 0.8)
        )
        expected = self._stream_after(
            lambda: mx.eval(mx.random.categorical(self.tlp))
        )
        self.assertEqual(got, expected)

    def test_unkeyed_helpers_are_seed_reproducible(self):
        k = 2
        logprobs = mx.stack([self.tlp] * (k + 1))
        runs = []
        for _ in range(2):
            mx.random.seed(2026)
            runs.append(
                _batched_residual_verify(logprobs, [self.dlp] * k, [3, 4], 0.8)
            )
        self.assertEqual(runs[0], runs[1])

    def test_keyed_helpers_ignore_the_global_seed(self):
        k = 2
        logprobs = mx.stack([self.tlp] * (k + 1))
        results = []
        for seed in (1, 999_999):
            mx.random.seed(seed)
            rng = LaneRNG(64)
            results.append(
                _batched_residual_verify(
                    logprobs, [self.dlp] * k, [3, 4], 0.8, rng=rng
                )
            )
        self.assertEqual(results[0], results[1])

    def test_greedy_consumes_no_lane_key(self):
        rng = LaneRNG(8)
        tok = _sample_from_logprobs(self.tlp, 0.0, rng=rng)
        self.assertEqual(tok, int(mx.argmax(self.tlp).item()))
        self.assertEqual(rng.draws, 0)


class TestLaneRNGInGeneration(unittest.TestCase):
    """The lane key through ``self_mtp_generate_step`` on a tiny CPU model."""

    TEMP, TOP_P, TOP_K = 0.8, 0.85, 8
    K = 2

    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)
        from mlx_lm.models.qwen3_5 import TextModel

        from test_qwen3_5_mtp import tiny_args

        mx.random.seed(0)
        cls.model = TextModel(tiny_args())
        mx.eval(cls.model.parameters())
        cls.prompt = mx.random.randint(0, 64, (16,)).astype(mx.uint32)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def _gen(self, *, lane_rng=None, max_tokens=12, temp=None, **kwargs):
        stats = HybridStats()
        gen = self_mtp_generate_step(
            self.prompt,
            self.model,
            num_draft=self.K,
            max_tokens=max_tokens,
            sampling_temp=self.TEMP if temp is None else temp,
            sampling_top_p=self.TOP_P,
            sampling_top_k=self.TOP_K,
            persistent_mtp=True,
            stats=stats,
            lane_rng=lane_rng,
            **kwargs,
        )
        return gen, stats

    def _tokens(self, **kwargs):
        gen, stats = self._gen(**kwargs)
        return [int(t) for t, _lp, _fd in gen], stats

    def test_lane_key_reproduces_the_stream_under_any_global_seed(self):
        streams = []
        for seed in (1, 424_242):
            mx.random.seed(seed)
            toks, stats = self._tokens(lane_rng=LaneRNG(55))
            streams.append(toks)
            self.assertGreater(stats.draft_proposed, 0)
        self.assertEqual(streams[0], streams[1])

    def test_different_lane_keys_give_different_streams(self):
        mx.random.seed(3)
        a, _ = self._tokens(lane_rng=LaneRNG(101))
        mx.random.seed(3)
        b, _ = self._tokens(lane_rng=LaneRNG(202))
        self.assertNotEqual(a, b)

    def test_forked_lanes_diverge(self):
        # The n>1 shape: one request becomes several lanes. Forking gives them
        # independent streams; copying the lane object would repeat one.
        parent = LaneRNG(77)
        left, right = parent.fork(2)
        mx.random.seed(9)
        a, _ = self._tokens(lane_rng=left)
        mx.random.seed(9)
        b, _ = self._tokens(lane_rng=right)
        self.assertNotEqual(a, b)

    def _interleaved(self, lane_a, lane_b):
        """Run two generators turn by turn: B joins, draws, and leaves inside
        A's lifetime. Returns A's tokens."""
        gen_a, _ = self._gen(lane_rng=lane_a)
        gen_b, _ = self._gen(lane_rng=lane_b, max_tokens=6)
        out_a, done_b = [], False
        for tok, _lp, _fd in gen_a:
            out_a.append(int(tok))
            if not done_b:
                try:
                    next(gen_b)
                except StopIteration:
                    done_b = True
        gen_b.close()
        return out_a

    def test_join_and_leave_do_not_perturb_a_lane(self):
        mx.random.seed(31)
        solo, _ = self._tokens(lane_rng=LaneRNG(1234))
        mx.random.seed(31)
        with_neighbour = self._interleaved(LaneRNG(1234), LaneRNG(5678))
        self.assertEqual(solo, with_neighbour)

    def test_global_stream_path_is_perturbed_by_a_neighbour(self):
        # Power check for the test above: without lane keys, the neighbour's
        # draws reorder this lane's stream. This is the coupling lane keys fix.
        mx.random.seed(31)
        solo, _ = self._tokens(lane_rng=None)
        mx.random.seed(31)
        with_neighbour = self._interleaved(None, None)
        self.assertNotEqual(solo, with_neighbour)

    def test_draw_count_is_acceptance_independent(self):
        # The rollback contract, measured: the key advances once per draw MADE
        # — k draft draws, one acceptance uniform and one correction per verify
        # cycle, one per plain step, one for the first token — never rewound on
        # rejection and never reused. So the count follows the schedule, not
        # the acceptance outcome.
        rng = LaneRNG(4242)
        mx.random.seed(5)
        toks, stats = self._tokens(lane_rng=rng, max_tokens=24)
        self.assertEqual(len(toks), 24)
        self.assertGreater(stats.draft_cycles, 0)
        expected = (
            1  # the first token, sampled before the loop
            + stats.draft_proposed  # one draw per drafted token
            + 2 * stats.draft_cycles  # acceptance uniforms + correction
            + stats.plain_cycles  # plain steps inside the loop
        )
        self.assertEqual(rng.draws, expected)

    def test_rejections_happen_and_the_lane_still_advances_monotonically(self):
        rng = LaneRNG(31_337)
        mx.random.seed(2)
        seen = [0]
        gen, stats = self._gen(lane_rng=rng, max_tokens=24)
        for _ in gen:
            self.assertGreaterEqual(rng.draws, seen[-1])
            seen.append(rng.draws)
        # The fixture must actually reject drafts, or the gate proves nothing.
        self.assertGreater(stats.draft_proposed, stats.draft_accepted)
        self.assertEqual(seen, sorted(seen))

    def test_greedy_draws_nothing_and_is_unchanged_by_a_lane_key(self):
        rng = LaneRNG(64)
        mx.random.seed(12)
        keyed, _ = self._tokens(lane_rng=rng, temp=0.0)
        mx.random.seed(12)
        plain, _ = self._tokens(lane_rng=None, temp=0.0)
        self.assertEqual(keyed, plain)
        self.assertEqual(rng.draws, 0)

    def test_sidecar_carries_the_lane_key_and_resumes_the_stream(self):
        rng = LaneRNG(808)
        out = {}
        mx.random.seed(6)
        toks, _ = self._tokens(lane_rng=rng, max_tokens=10, mtp_state_out=out)
        self.assertEqual(_key_list(out["rng_key"]), _key_list(rng.key))
        self.assertEqual(out["rng_draws"], rng.draws)

        # A lane rebuilt from the sidecar continues; one rebuilt from the seed
        # would replay the draws the first run already made.
        resumed = LaneRNG.from_key(out["rng_key"], out["rng_draws"])
        twin = LaneRNG.from_key(out["rng_key"])
        self.assertEqual(_key_list(resumed.next_key()), _key_list(twin.next_key()))
        self.assertNotEqual(
            _key_list(LaneRNG(808).next_key()), _key_list(resumed.key)
        )
        self.assertGreater(len(toks), 0)

    def test_lane_rng_rejects_a_bare_seed(self):
        gen = self_mtp_generate_step(
            self.prompt, self.model, max_tokens=4, sampling_temp=self.TEMP,
            lane_rng=17,
        )
        with self.assertRaises(TypeError):
            next(gen)


if __name__ == "__main__":
    unittest.main()
