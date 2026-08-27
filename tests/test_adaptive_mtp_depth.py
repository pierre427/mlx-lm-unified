# Copyright © 2026 Apple Inc.

"""Adaptive MTP draft-depth ceiling (--self-mtp-adaptive-depth-ceiling).

Three layers, all on tiny synthetic models (no checkpoints):

1. DepthCeilingController unit tests — the #2046-style policy: start at the
   native floor, expand only after sustained full-native-prefix acceptance,
   back off on decline, clamp to [floor, ceiling], hysteretic dwell.
2. Server admission — the flag creates a fresh per-request controller with
   ``num_draft`` promoted to the ceiling; unset leaves the config (and thus
   the whole off path) byte-identical to today's.
3. Composition through the ``speculation_router=`` seam of
   ``self_mtp_generate_step`` — off-path stream identity, every-k rollback
   exactness (trim spy + offset invariant), and the live features the depth
   now varies under: transformed verifier, rate gate, windowed MTP at the
   ceiling, QSA index sharing, APC sidecar capture.
"""

import types
import unittest
from unittest import mock

import mlx.core as mx

import mlx_lm.hybrid_speculative as hybrid_speculative
from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel
from mlx_lm.server import _self_mtp_config, _validate_adaptive_depth_ceiling
from mlx_lm.spec_policy import MAX_DRAFT_TOKENS
from mlx_lm.speculation_router import DepthCeilingController, SpeculationDecision

from test_qwen3_5_mtp import tiny_args


class _ScheduleRouter:
    """Seam-compatible stub forcing a deterministic per-cycle depth."""

    reengagements = 0

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.index = 0
        self.accept_prob = 0.0
        self.observed = []

    def decide(self, *, max_draft=None, remaining=None):
        k = self.schedule[self.index % len(self.schedule)]
        self.index += 1
        if max_draft is not None:
            k = min(k, int(max_draft))
        if remaining is not None:
            k = min(k, max(0, int(remaining)))
        return SpeculationDecision(k, "schedule", 0.0, False, 0)

    def observe(self, proposed, accepted):
        self.observed.append((proposed, accepted))


class TestDepthCeilingController(unittest.TestCase):
    def test_starts_at_floor_with_seam_contract(self):
        controller = DepthCeilingController(2, 3)
        decision = controller.decide(max_draft=3, remaining=100)
        self.assertEqual(decision.num_draft, 2)
        self.assertFalse(decision.latched_plain)
        self.assertEqual(decision.cooldown_remaining, 0)
        self.assertEqual(controller.reengagements, 0)
        self.assertIsInstance(controller.accept_prob, float)

    def test_expands_after_sustained_full_native_acceptance(self):
        controller = DepthCeilingController(2, 3)
        for _ in range(8):
            controller.observe(2, 2)
        self.assertEqual(controller.decide(max_draft=3).num_draft, 3)
        self.assertEqual(controller.expansions, 1)

    def test_no_expansion_below_65_percent(self):
        controller = DepthCeilingController(2, 3)
        # 5/8 = 0.625 < 0.65: stays at the floor.
        for accepted in (2, 2, 2, 2, 2, 0, 1, 0):
            controller.observe(2, accepted)
        self.assertEqual(controller.decide(max_draft=3).num_draft, 2)
        # Full rounds slide in but the three failures only leave the window
        # after eight more rounds; 5/8 persists until the first failure
        # falls out (6/8 = 0.75 >= 0.65), then it expands.
        for _ in range(5):
            controller.observe(2, 2)
        self.assertEqual(controller.decide(max_draft=3).num_draft, 2)
        controller.observe(2, 2)
        self.assertEqual(controller.decide(max_draft=3).num_draft, 3)

    def test_backoff_on_decline(self):
        controller = DepthCeilingController(2, 3)
        for _ in range(8):
            controller.observe(2, 2)
        self.assertEqual(controller.depth, 3)
        # Sustained sub-native acceptance: 0/8 < 0.50 backs off to the floor.
        for _ in range(8):
            controller.observe(3, 1)
        self.assertEqual(controller.decide(max_draft=3).num_draft, 2)
        self.assertEqual(controller.backoffs, 1)

    def test_floor_and_ceiling_clamps(self):
        controller = DepthCeilingController(2, 3)
        for _ in range(64):
            controller.observe(2, 0)
        self.assertEqual(controller.depth, 2)  # never below the floor
        for _ in range(64):
            controller.observe(controller.depth, controller.depth)
        self.assertEqual(controller.depth, 3)  # never above the ceiling

    def test_hysteretic_dwell_after_change(self):
        controller = DepthCeilingController(2, 3)
        for _ in range(8):
            controller.observe(2, 2)
        self.assertEqual(controller.depth, 3)
        # The window was cleared: 7 failing rounds are not yet evidence.
        for _ in range(7):
            controller.observe(3, 0)
        self.assertEqual(controller.depth, 3)
        controller.observe(3, 0)
        self.assertEqual(controller.depth, 2)

    def test_truncated_round_accepted_in_full_counts_as_full(self):
        controller = DepthCeilingController(2, 3)
        # Budget-truncated cycles (proposed < floor) accepted in full carry
        # no evidence against expansion.
        for _ in range(8):
            controller.observe(1, 1)
        self.assertEqual(controller.depth, 3)

    def test_decide_respects_caps(self):
        controller = DepthCeilingController(2, 5)
        self.assertEqual(controller.decide(max_draft=1).num_draft, 1)
        self.assertEqual(controller.decide(max_draft=5, remaining=1).num_draft, 1)
        self.assertEqual(controller.decide(max_draft=5, remaining=0).num_draft, 0)
        self.assertEqual(controller.decide().num_draft, 2)

    def test_observe_validates_counts(self):
        controller = DepthCeilingController(2, 3)
        for proposed, accepted in ((0, 0), (-1, 0), (2, 3), (2, -1)):
            with self.assertRaises(ValueError):
                controller.observe(proposed, accepted)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            DepthCeilingController(0, 3)
        with self.assertRaises(ValueError):
            DepthCeilingController(3, 2)
        with self.assertRaises(ValueError):
            DepthCeilingController(2, 3, window=0)
        with self.assertRaises(ValueError):
            DepthCeilingController(2, 3, expand_threshold=0.4, backoff_threshold=0.5)

    def test_snapshot_reports_depth_trajectory(self):
        controller = DepthCeilingController(2, 3)
        for _ in range(8):
            controller.observe(2, 2)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["floor"], 2)
        self.assertEqual(snapshot["ceiling"], 3)
        self.assertEqual(snapshot["depth"], 3)
        self.assertEqual(snapshot["expansions"], 1)


class TestServerAdmission(unittest.TestCase):
    """_self_mtp_config wiring for --self-mtp-adaptive-depth-ceiling."""

    def cli(self, **overrides):
        ns = types.SimpleNamespace(
            self_mtp=True,
            self_mtp_num_draft=2,
            self_mtp_persistent=True,
            self_mtp_rate_gate=True,
            self_mtp_share_qsa_indices=False,
            self_mtp_share_qsa_indices_min_prompt_tokens=0,
            self_mtp_window_size=0,
            self_mtp_window_sink_size=4,
            self_mtp_window_min_prompt_tokens=0,
            self_mtp_transformed_verifier=False,
            self_mtp_adaptive_depth_ceiling=None,
            kv_bits=None,
        )
        for key, value in overrides.items():
            setattr(ns, key, value)
        return ns

    @staticmethod
    def args():
        return types.SimpleNamespace(
            model=types.SimpleNamespace(draft="default_model"),
            prompt_lookup_ngram=0,
            sampling=types.SimpleNamespace(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
            ),
        )

    def setUp(self):
        self.model = types.SimpleNamespace(mtp=object())

    def test_flag_unset_leaves_config_byte_identical(self):
        config = _self_mtp_config(self.args(), self.cli(), self.model)
        self.assertNotIn("speculation_router", config)
        self.assertEqual(config["num_draft"], 2)
        # Exactly the pre-change key set: generate.py then passes
        # speculation_router=None and the engine path is HEAD's.
        self.assertEqual(
            set(config),
            {
                "num_draft",
                "persistent",
                "rate_gate",
                "share_qsa_indices",
                "sampling_temp",
                "accept_rule",
                "state_out",
            },
        )

    def test_flag_set_promotes_ceiling_and_installs_controller(self):
        cli = self.cli(self_mtp_adaptive_depth_ceiling=3)
        config = _self_mtp_config(self.args(), cli, self.model)
        self.assertEqual(config["num_draft"], 3)
        controller = config["speculation_router"]
        self.assertIsInstance(controller, DepthCeilingController)
        self.assertEqual(controller.floor, 2)
        self.assertEqual(controller.ceiling, 3)
        self.assertEqual(controller.depth, 2)

    def test_controller_is_fresh_per_request(self):
        cli = self.cli(self_mtp_adaptive_depth_ceiling=3)
        first = _self_mtp_config(self.args(), cli, self.model)
        second = _self_mtp_config(self.args(), cli, self.model)
        self.assertIsNot(
            first["speculation_router"], second["speculation_router"]
        )

    def test_flag_validation_bounds(self):
        ok = types.SimpleNamespace(
            self_mtp_num_draft=2, self_mtp_adaptive_depth_ceiling=3
        )
        _validate_adaptive_depth_ceiling(ok)
        unset = types.SimpleNamespace(
            self_mtp_num_draft=2, self_mtp_adaptive_depth_ceiling=None
        )
        _validate_adaptive_depth_ceiling(unset)
        equal = types.SimpleNamespace(
            self_mtp_num_draft=2, self_mtp_adaptive_depth_ceiling=2
        )
        _validate_adaptive_depth_ceiling(equal)
        top = types.SimpleNamespace(
            self_mtp_num_draft=1,
            self_mtp_adaptive_depth_ceiling=MAX_DRAFT_TOKENS,
        )
        _validate_adaptive_depth_ceiling(top)
        for num_draft, ceiling in ((3, 2), (2, MAX_DRAFT_TOKENS + 1), (2, 0)):
            bad = types.SimpleNamespace(
                self_mtp_num_draft=num_draft,
                self_mtp_adaptive_depth_ceiling=ceiling,
            )
            with self.assertRaises(ValueError):
                _validate_adaptive_depth_ceiling(bad)


class _TinyMTPBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(0)
        cls.model = TextModel(tiny_args())
        mx.eval(cls.model.parameters())
        cls.prompt = mx.random.randint(0, 64, (16,)).astype(mx.uint32)

    def _run(self, seed=7, max_tokens=24, **kwargs):
        mx.random.seed(seed)
        stats = kwargs.pop("stats", None) or HybridStats()
        tokens = [
            int(token)
            for token, _lp, _fd in self_mtp_generate_step(
                self.prompt,
                self.model,
                max_tokens=max_tokens,
                stats=stats,
                **kwargs,
            )
        ]
        return tokens, stats


class TestOffPathAndSeamIdentity(_TinyMTPBase):
    def test_off_path_stream_identity_greedy(self):
        # HEAD's exact call shape (no speculation_router kwarg at all) vs the
        # new off path (explicit None) vs the seam at a constant depth: all
        # three must be byte-identical for greedy.
        head, _ = self._run(num_draft=2)
        off, _ = self._run(num_draft=2, speculation_router=None)
        constant, stats = self._run(
            num_draft=2, speculation_router=DepthCeilingController(2, 2)
        )
        self.assertEqual(head, off)
        self.assertEqual(head, constant)
        # The final cycle may be budget-capped below the constant depth.
        self.assertIn(stats.router_last_num_draft, (1, 2))

    def test_off_path_stream_identity_seeded_sampling(self):
        # temp > 0: the controller consumes no RNG, so identical cycle
        # shapes must give an identical sampled stream under the same seed.
        kwargs = dict(num_draft=2, sampling_temp=0.8, max_tokens=20)
        head, _ = self._run(seed=41, **kwargs)
        off, _ = self._run(seed=41, speculation_router=None, **kwargs)
        constant, _ = self._run(
            seed=41,
            speculation_router=DepthCeilingController(2, 2),
            **kwargs,
        )
        self.assertEqual(head, off)
        self.assertEqual(head, constant)


class TestRollbackExactness(_TinyMTPBase):
    """Every-k rollback exactness with a trim spy.

    Per draft cycle the loop must trim exactly k - n_accept rows from the
    target cache and rewind exactly k speculative entries from the persistent
    MTP cache; after the run the target cache must hold exactly the committed
    sequence (prompt + delivered - 1 pending verify token).
    """

    def _spied_run(self, router, num_draft, max_tokens=24, **kwargs):
        target_cache = make_prompt_cache(self.model)
        records = []
        real_trim = hybrid_speculative.trim_prompt_cache

        def spy(cache_list, n):
            records.append((cache_list is target_cache, int(n)))
            return real_trim(cache_list, n)

        with mock.patch.object(hybrid_speculative, "trim_prompt_cache", spy):
            tokens, stats = self._run(
                num_draft=num_draft,
                persistent_mtp=True,
                prompt_cache=target_cache,
                speculation_router=router,
                max_tokens=max_tokens,
                **kwargs,
            )
        return tokens, stats, records, target_cache

    def _check_invariants(self, tokens, stats, records, cache, max_k, max_tokens):
        self.assertEqual(len(tokens), max_tokens)
        offset = max(getattr(c, "offset", 0) for c in cache)
        # prompt + delivered, minus the pending bonus/cur not yet verified.
        self.assertEqual(offset, int(self.prompt.size) + max_tokens - 1)
        # One (target, mtp) trim pair per draft cycle, in call order: the
        # target trim drops the k - n_accept rejected rows, then the MTP
        # rewind drops the full k speculative entries.
        cycle_pairs = []
        index = 0
        while index < len(records):
            is_target, rejected = records[index]
            self.assertTrue(is_target, "cycle must trim the target cache first")
            self.assertLessEqual(0, rejected)
            index += 1
            self.assertLess(index, len(records), "missing MTP rewind")
            is_target_2, k_cycle = records[index]
            self.assertFalse(is_target_2, "MTP rewind must follow the verify trim")
            index += 1
            self.assertLessEqual(rejected, k_cycle)  # rejected <= proposed
            self.assertLessEqual(k_cycle, max_k)
            self.assertGreaterEqual(k_cycle, 1)
            cycle_pairs.append((rejected, k_cycle))
        self.assertEqual(len(cycle_pairs), stats.draft_cycles)
        self.assertEqual(sum(k for _r, k in cycle_pairs), stats.draft_proposed)
        self.assertEqual(
            sum(r for r, _k in cycle_pairs),
            stats.draft_proposed - stats.draft_accepted,
        )
        # The exactness claim is vacuous unless some rejection occurred.
        self.assertGreater(stats.draft_proposed - stats.draft_accepted, 0)

    def test_every_constant_k_from_floor_to_verify_cap(self):
        for k in range(1, MAX_DRAFT_TOKENS + 1):
            with self.subTest(k=k):
                tokens, stats, records, cache = self._spied_run(
                    DepthCeilingController(k, k), num_draft=k
                )
                self._check_invariants(tokens, stats, records, cache, k, 24)

    def test_varying_k_schedule(self):
        router = _ScheduleRouter([1, 2, 3, 2, 1, 3])
        tokens, stats, records, cache = self._spied_run(router, num_draft=3)
        self._check_invariants(tokens, stats, records, cache, 3, 24)
        # The schedule really varied the per-cycle depth.
        self.assertGreater(len({k for _n, k in router.observed}), 1)


class TestLiveFeatureComposition(_TinyMTPBase):
    def test_transformed_verifier_with_variable_k(self):
        # Batched residual acceptance must handle a different k every cycle.
        tokens, stats = self._run(
            seed=11,
            num_draft=3,
            sampling_temp=0.8,
            sampling_top_p=0.85,
            sampling_top_k=8,
            persistent_mtp=True,
            speculation_router=_ScheduleRouter([1, 2, 3]),
            max_tokens=24,
        )
        self.assertEqual(len(tokens), 24)
        self.assertGreater(stats.draft_proposed, 0)
        self.assertIn(stats.router_last_num_draft, (1, 2, 3))

    def test_transformed_verifier_with_real_controller(self):
        tokens, stats = self._run(
            seed=13,
            num_draft=3,
            sampling_temp=0.8,
            sampling_top_p=0.85,
            sampling_top_k=8,
            persistent_mtp=True,
            speculation_router=DepthCeilingController(2, 3),
            max_tokens=24,
        )
        self.assertEqual(len(tokens), 24)
        self.assertGreater(stats.draft_proposed, 0)

    def test_rate_gate_delatch_with_adaptive_depth(self):
        # Force the one-shot probe to de-latch; the plain tail must still
        # deliver every token and the controller must simply stop being
        # consulted (the gate keeps sole authority over IF).
        with mock.patch.object(hybrid_speculative, "_RATE_GATE_MARGIN", 1.0):
            tokens, stats = self._run(
                num_draft=3,
                persistent_mtp=True,
                rate_gate=True,
                speculation_router=DepthCeilingController(2, 3),
                max_tokens=48,
            )
        self.assertEqual(len(tokens), 48)
        self.assertTrue(stats.rate_gate_probed)
        self.assertTrue(stats.rate_gate_delatched)
        self.assertGreater(stats.plain_tokens, 0)

    def test_rate_gate_keep_with_adaptive_depth(self):
        # Force the probe to keep speculating; drafting must resume with the
        # controller still deciding the depth after the probe.
        with mock.patch.object(hybrid_speculative, "_RATE_GATE_MARGIN", -100.0):
            tokens, stats = self._run(
                num_draft=3,
                persistent_mtp=True,
                rate_gate=True,
                speculation_router=DepthCeilingController(2, 3),
                max_tokens=48,
            )
        self.assertEqual(len(tokens), 48)
        self.assertTrue(stats.rate_gate_probed)
        self.assertFalse(stats.rate_gate_delatched)
        self.assertGreater(stats.draft_cycles, hybrid_speculative._RATE_GATE_WARMUP_CYCLES)

    def test_apc_sidecar_capture_with_varying_k(self):
        # The sidecar must stay structurally exact (mtp offset == target
        # offset - 1) when every cycle may use a different depth, and the
        # controller itself must never leak into the captured state.
        state_out = {}
        tokens, _stats = self._run(
            num_draft=3,
            persistent_mtp=True,
            speculation_router=_ScheduleRouter([2, 3, 1]),
            mtp_state_out=state_out,
            max_tokens=24,
        )
        self.assertEqual(len(tokens), 24)
        self.assertTrue(state_out["reusable"])
        self.assertEqual(
            state_out["covered_tokens"], int(self.prompt.size) + 24 - 1
        )
        mtp_cache, seed_h = state_out["state"]
        self.assertIsNotNone(seed_h)
        for layer_cache in mtp_cache:
            self.assertFalse(
                isinstance(layer_cache, DepthCeilingController)
            )


class TestQwen4Composition(unittest.TestCase):
    """Windowed MTP and QSA index sharing under a varying depth."""

    @classmethod
    def setUpClass(cls):
        from mlx_lm.models.qwen4_exp import Model, ModelArgs
        from test_qwen4_exp import tiny_args as qwen4_tiny_args

        mx.random.seed(4)
        args = qwen4_tiny_args(mtp_num_hidden_layers=1)
        cls.model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=args.__dict__)
        )
        mx.eval(cls.model.parameters())
        cls.prompt = mx.random.randint(0, 60, (16,)).astype(mx.uint32)

    def _run(self, seed=88, max_tokens=24, **kwargs):
        mx.random.seed(seed)
        stats = HybridStats()
        tokens = [
            int(token)
            for token, _lp, _fd in self_mtp_generate_step(
                self.prompt,
                self.model,
                max_tokens=max_tokens,
                stats=stats,
                **kwargs,
            )
        ]
        return tokens, stats

    def test_windowed_mtp_at_the_verify_width_ceiling(self):
        # Ceiling k=7 (drafts + bonus == the M5 verify width) against a
        # bounded sink-window draft cache: every k-token rewind must stay
        # inside the retained rollback tail (K+1 <= rollback tail), which
        # SinkWindowKVCache enforces by raising on violation.
        tokens, stats = self._run(
            num_draft=MAX_DRAFT_TOKENS,
            persistent_mtp=True,
            mtp_window_size=8,
            mtp_sink_size=2,
            speculation_router=_ScheduleRouter([MAX_DRAFT_TOKENS]),
            max_tokens=24,
        )
        self.assertEqual(len(tokens), 24)
        self.assertGreater(stats.draft_proposed, 0)

    def test_windowed_mtp_with_real_controller(self):
        tokens, stats = self._run(
            num_draft=3,
            persistent_mtp=True,
            mtp_window_size=8,
            mtp_sink_size=2,
            speculation_router=DepthCeilingController(2, 3),
            max_tokens=24,
        )
        self.assertEqual(len(tokens), 24)
        self.assertGreater(stats.draft_proposed, 0)

    def test_qsa_index_sharing_with_varying_k(self):
        # Sharing is armed per cycle with THAT cycle's k (> 1); a k=1 cycle
        # must run unshared, and the end-of-cycle rewind must always clear
        # the shared top-k so no indices leak across cycles of different k.
        state_out = {}
        tokens, stats = self._run(
            num_draft=3,
            persistent_mtp=True,
            mtp_share_qsa_indices=True,
            speculation_router=_ScheduleRouter([1, 3, 2]),
            mtp_state_out=state_out,
            max_tokens=24,
        )
        self.assertEqual(len(tokens), 24)
        self.assertGreater(stats.draft_proposed, 0)
        mtp_cache, _seed_h = state_out["state"]
        for layer_cache in mtp_cache:
            self.assertIsNone(getattr(layer_cache, "_mtp_shared_topk", None))


if __name__ == "__main__":
    unittest.main()
