# Copyright © 2026 Apple Inc.

"""Exact self-MTP speculative verification under transformed sampling.

``transformed_logprobs`` must induce exactly the distribution
``make_sampler(temp, top_p, top_k, min_p)`` samples from — including the
sampler's native-dtype normalization (bf16/fp16 logits round differently
than fp32; see the bf16 near-tie test). The batched residual verifier must
accept only inside the target's transformed support and preserve the
transformed target distribution; end-to-end self-MTP decoding with the
shared transform must match plain sampling from the same transformed
distribution.

Statistical gates here are SMOKE GATES, not exactness proofs: the TV checks
(N=6000, TV<0.05) pass any discrepancy below ~5% TV, and the chi-square
threshold (32 at df<=8) is a deliberately conservative bound with bins picked
from the reference arm and sparse cells pooled into "other". The exactness
burden is carried by the deterministic cases: support agreement on
filter-boundary ties, the native-dtype near-tie, forced-rejection kernels,
and the -inf/underflow edges. Fixed-seed stream comparison is invalid across
sampling strategies, so the end-to-end gate is a many-sample marginal
comparison (the ``test_mtp_accept_rules`` protocol).
"""

import types
import unittest
from collections import Counter

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    HybridStats,
    _batched_residual_verify,
    _make_sampling_transform,
    _residual_sample,
    _sample_from_logprobs,
    self_mtp_generate_step,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel
from mlx_lm.sample_utils import (
    make_logits_processors,
    make_sampler,
    transformed_logprobs,
)
from mlx_lm.server import _self_mtp_config

from test_qwen3_5_mtp import tiny_args


def _tv(counts, ref_probs):
    total = sum(counts.values())
    return 0.5 * sum(
        abs(counts.get(i, 0) / total - p) for i, p in enumerate(ref_probs)
    )


def _support(logprobs):
    return {i for i, v in enumerate(mx.exp(logprobs).tolist()) if v > 0.0}


def _sampler_draws(profile, base_logprobs, n, seed):
    sampler = make_sampler(
        profile["temp"],
        top_p=profile["top_p"],
        top_k=profile["top_k"],
        min_p=profile["min_p"],
    )
    mx.random.seed(seed)
    draws = sampler(mx.broadcast_to(base_logprobs, (n,) + base_logprobs.shape))
    mx.eval(draws)
    return Counter(draws.tolist())


# name -> (params, fixed fixture seed). hash() is process-randomized, so
# seeds are explicit integers to keep fixtures deterministic between runs.
PROFILES = {
    "thinking": (dict(temp=1.0, top_p=0.95, top_k=20, min_p=0.0), 101),
    "nonthinking": (dict(temp=0.7, top_p=0.8, top_k=20, min_p=0.0), 102),
    "top_p_only": (dict(temp=0.9, top_p=0.6, top_k=0, min_p=0.0), 103),
    "top_k_only": (dict(temp=1.3, top_p=1.0, top_k=5, min_p=0.0), 104),
    "min_p": (dict(temp=0.8, top_p=1.0, top_k=0, min_p=0.05), 105),
    "all_filters": (dict(temp=0.7, top_p=0.85, top_k=12, min_p=0.02), 106),
}


class TestTransformEquivalence(unittest.TestCase):
    """transformed_logprobs vs the empirical make_sampler distribution."""

    VOCAB = 32
    N = 6000

    def _check_profile(self, prof, logits):
        # Mirror generate_step: normalize in the logits' NATIVE dtype and
        # hand that to the sampler; the transform gets the raw logits.
        base_lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        lp = transformed_logprobs(
            logits,
            prof["temp"],
            top_p=prof["top_p"],
            top_k=prof["top_k"],
            min_p=prof["min_p"],
        )
        probs = mx.exp(lp)
        mx.eval(probs)
        probs = probs.tolist()
        self.assertAlmostEqual(sum(probs), 1.0, places=4)
        counts = _sampler_draws(prof, base_lp, self.N, 1234)
        # Support agreement: the sampler never emits a filtered token.
        for tok in counts:
            self.assertGreater(probs[tok], 0.0, f"tok={tok}")
        # Smoke gate, not exactness evidence (see module docstring).
        self.assertLess(_tv(counts, probs), 0.05)

    def test_transform_matches_make_sampler_distribution(self):
        for name, (prof, seed) in PROFILES.items():
            with self.subTest(profile=name):
                mx.random.seed(seed)
                self._check_profile(prof, mx.random.normal((self.VOCAB,)) * 2.0)

    def test_transform_matches_make_sampler_distribution_bf16(self):
        for name, (prof, seed) in PROFILES.items():
            with self.subTest(profile=name):
                mx.random.seed(seed)
                logits = (mx.random.normal((self.VOCAB,)) * 2.0).astype(
                    mx.bfloat16
                )
                self._check_profile(prof, logits)

    def test_native_dtype_normalization_matches_sampler(self):
        # The dtype blocker's counterexample: in bf16 the tiny logit gap
        # rounds away during native normalization, so the sampler's
        # distribution is exactly 50/50. An fp32-normalizing transform
        # would instead claim ~0.731/0.269 at temp=0.001.
        gap = -0.0010004044
        logits16 = mx.array([0.0, gap], dtype=mx.bfloat16)
        prof = dict(temp=0.001, top_p=1.0, top_k=0, min_p=0.0)
        p = mx.exp(transformed_logprobs(logits16, prof["temp"])).tolist()
        self.assertLess(abs(p[0] - 0.5), 1e-3)
        # Power check: the fp32 normalization is decisively different.
        p32 = mx.exp(
            transformed_logprobs(logits16.astype(mx.float32), prof["temp"])
        ).tolist()
        self.assertGreater(p32[0], 0.72)
        # And the sampler agrees with the native-dtype transform.
        base = logits16 - mx.logsumexp(logits16, keepdims=True)
        counts = _sampler_draws(prof, base, 20_000, 9)
        self.assertLess(abs(counts[0] / 20_000 - p[0]), 0.02)

    def test_top_p_boundary_tie_support_agreement(self):
        # Ascending cumsum hits exactly 1-top_p at one token; the strict ">"
        # decides its fate. Both paths run the identical apply_top_p on the
        # identical array, so the in/out decision must agree exactly.
        logits = mx.log(mx.array([0.5, 0.25, 0.125, 0.125]))
        prof = dict(temp=0.9, top_p=0.75, top_k=0, min_p=0.0)
        lp = transformed_logprobs(logits, prof["temp"], top_p=prof["top_p"])
        support = _support(lp)
        base = logits - mx.logsumexp(logits, keepdims=True)
        counts = _sampler_draws(prof, base, 4000, 21)
        # Every surviving token has mass >= ~0.19, so 4000 draws see all of
        # them; exact equality is the agreement check.
        self.assertEqual(set(counts), support)

    def test_top_k_tie_support_agreement(self):
        # Three-way tie across the k=2 cutoff: argpartition retains an
        # implementation-selected subset — the same subset in both paths.
        logits = mx.array([1.0, 0.5, 0.5, 0.5, 0.0, -1.0])
        prof = dict(temp=0.8, top_p=1.0, top_k=2, min_p=0.0)
        lp = transformed_logprobs(logits, prof["temp"], top_k=prof["top_k"])
        support = _support(lp)
        self.assertEqual(len(support), 2)
        self.assertIn(0, support)
        base = logits - mx.logsumexp(logits, keepdims=True)
        counts = _sampler_draws(prof, base, 4000, 22)
        self.assertEqual(set(counts), support)

    def test_min_p_cutoff_tie_support_agreement(self):
        import math

        # One token sits numerically at max_p * min_p; the strict "<" in
        # apply_min_p keeps it. Same computation both paths -> same support.
        logits = mx.array([0.0, math.log(0.1), math.log(0.05), -6.0])
        prof = dict(temp=1.0, top_p=1.0, top_k=0, min_p=0.1)
        lp = transformed_logprobs(logits, prof["temp"], min_p=prof["min_p"])
        support = _support(lp)
        base = logits - mx.logsumexp(logits, keepdims=True)
        counts = _sampler_draws(prof, base, 6000, 23)
        # Tokens 0 and 1 always survive (mass >= ~0.09); the borderline
        # tokens must match the sampler exactly in whichever way the
        # floating-point tie resolved.
        self.assertEqual(set(counts), support)

    def test_masked_inputs_stay_filtered(self):
        logits = mx.array([1.0, -float("inf"), 0.5, -float("inf"), 0.0])
        lp = transformed_logprobs(logits, 0.7, top_p=0.99)
        probs = mx.exp(lp).tolist()
        self.assertEqual(probs[1], 0.0)
        self.assertEqual(probs[3], 0.0)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_transform_is_batched_row_independent(self):
        mx.random.seed(11)
        logits = mx.random.normal((3, self.VOCAB)) * 3.0
        batched = transformed_logprobs(logits, 0.7, top_p=0.8, top_k=6)
        for i in range(3):
            row = transformed_logprobs(logits[i], 0.7, top_p=0.8, top_k=6)
            self.assertTrue(
                mx.array_equal(batched[i], row, equal_nan=True).item()
            )

    def test_no_filters_returns_none_transform(self):
        self.assertIsNone(_make_sampling_transform(0.8, 1.0, 0, 0.0))
        self.assertIsNone(_make_sampling_transform(0.8, 0.0, 0, 0.0))
        # Greedy: filters keep the argmax, so the exact greedy path stands.
        self.assertIsNone(_make_sampling_transform(0.0, 0.8, 20, 0.0))
        self.assertIsNotNone(_make_sampling_transform(0.8, 0.8, 0, 0.0))
        self.assertIsNotNone(_make_sampling_transform(0.8, 1.0, 20, 0.0))
        self.assertIsNotNone(_make_sampling_transform(0.8, 1.0, 0, 0.1))


class TestBatchedResidualKernel(unittest.TestCase):
    """The batched acceptance rule on constructed transformed distributions."""

    def _dists(self):
        # Vocab 6. Target keeps {0,1,2} (top-3), draft keeps {2,3,4}:
        # overlapping but distinct supports after a top-k=3 transform.
        t_logits = mx.array([2.0, 1.5, 1.0, 0.2, 0.1, 0.0])
        d_logits = mx.array([0.0, 0.1, 1.0, 2.0, 1.5, 0.2])
        tlp = transformed_logprobs(t_logits, 0.8, top_k=3)
        dlp = transformed_logprobs(d_logits, 0.8, top_k=3)
        return tlp, dlp

    def _disjoint(self):
        # Target keeps {0,1,2}, draft keeps {3,4,5}: no overlap at all.
        t_logits = mx.array([3.0, 2.5, 2.0, -1.0, -1.5, -2.0])
        d_logits = mx.array([-1.0, -1.5, -2.0, 3.0, 2.5, 2.0])
        tlp = transformed_logprobs(t_logits, 0.8, top_k=3)
        dlp = transformed_logprobs(d_logits, 0.8, top_k=3)
        return tlp, dlp

    def test_never_accepts_outside_target_support(self):
        tlp, dlp = self._dists()
        target_support = _support(tlp)
        logprobs = mx.stack([tlp, tlp])  # k=1 verify rows + bonus row
        committed = Counter()
        for i in range(3000):
            mx.random.seed(5_000 + i)
            d = _sample_from_logprobs(dlp, 0.8)
            n_accept, bonus = _batched_residual_verify(
                logprobs, [dlp], [d], 0.8
            )
            tok = d if n_accept == 1 else bonus
            committed[tok] += 1
            if d not in target_support:
                self.assertEqual(n_accept, 0, f"accepted filtered draft {d}")
        self.assertTrue(set(committed) <= target_support)
        # And the committed marginal is the transformed TARGET distribution.
        self.assertLess(_tv(committed, mx.exp(tlp).tolist()), 0.04)

    def test_matches_expected_acceptance_rate(self):
        tlp, dlp = self._dists()
        expected = float(
            mx.sum(mx.minimum(mx.exp(tlp), mx.exp(dlp))).item()
        )
        logprobs = mx.stack([tlp, tlp])
        accepted = 0
        n = 3000
        for i in range(n):
            mx.random.seed(60_000 + i)
            d = _sample_from_logprobs(dlp, 0.8)
            n_accept, _ = _batched_residual_verify(logprobs, [dlp], [d], 0.8)
            accepted += n_accept
        self.assertLess(abs(accepted / n - expected), 0.04)

    def test_k2_preserves_first_position_marginal(self):
        tlp, dlp = self._dists()
        # Second position: identical distributions, always accepted.
        logprobs = mx.stack([tlp, tlp, tlp])
        committed = Counter()
        n = 3000
        for i in range(n):
            mx.random.seed(300_000 + i)
            d1 = _sample_from_logprobs(dlp, 0.8)
            d2 = _sample_from_logprobs(tlp, 0.8)
            n_accept, bonus = _batched_residual_verify(
                logprobs, [dlp, tlp], [d1, d2], 0.8
            )
            committed[d1 if n_accept >= 1 else bonus] += 1
        self.assertLess(_tv(committed, mx.exp(tlp).tolist()), 0.04)

    def test_disjoint_supports_force_zero_accept(self):
        # Every draft lands outside the target's transformed support: the
        # verifier must reject ALL k drafts every time, and the correction
        # (residual relu(p - q) = p on disjoint supports) is the target.
        tlp, dlp = self._disjoint()
        logprobs = mx.stack([tlp, tlp, tlp])
        committed = Counter()
        n = 2000
        for i in range(n):
            mx.random.seed(700_000 + i)
            d1 = _sample_from_logprobs(dlp, 0.8)
            d2 = _sample_from_logprobs(dlp, 0.8)
            n_accept, bonus = _batched_residual_verify(
                logprobs, [dlp, dlp], [d1, d2], 0.8
            )
            self.assertEqual(n_accept, 0)
            committed[bonus] += 1
        self.assertTrue(set(committed) <= _support(tlp))
        self.assertLess(_tv(committed, mx.exp(tlp).tolist()), 0.04)

    def test_midblock_rejection_commits_position1_residual(self):
        # Position 0 draft == target (ratio 1: always accepted); position 1
        # disjoint (always rejected). n_accept must be exactly 1 and the
        # correction distributed as the position-1 target.
        tlp, dlp = self._disjoint()
        logprobs = mx.stack([tlp, tlp, tlp])
        committed = Counter()
        n = 2000
        for i in range(n):
            mx.random.seed(800_000 + i)
            d1 = _sample_from_logprobs(tlp, 0.8)
            d2 = _sample_from_logprobs(dlp, 0.8)
            n_accept, bonus = _batched_residual_verify(
                logprobs, [tlp, dlp], [d1, d2], 0.8
            )
            self.assertEqual(n_accept, 1)
            committed[bonus] += 1
        self.assertLess(_tv(committed, mx.exp(tlp).tolist()), 0.04)

    def test_residual_underflow_falls_back_to_target(self):
        # p == q makes relu(p - q) identically zero; the fallback must
        # sample from the target (never a filtered token, never NaN).
        tlp, _ = self._dists()
        support = _support(tlp)
        for i in range(50):
            mx.random.seed(900_000 + i)
            tok = _residual_sample(tlp, tlp, 0.8)
            self.assertIn(tok, support)


class TestTransformedEndToEnd(unittest.TestCase):
    """self_mtp_generate_step with the shared transform on the tiny MTP."""

    TEMP = 0.8
    TOP_P = 0.85
    TOP_K = 8
    N = 400
    NEW = 5

    @classmethod
    def setUpClass(cls):
        mx.random.seed(0)
        cls.model = TextModel(tiny_args())
        mx.eval(cls.model.parameters())
        cls.prompt = mx.random.randint(0, 64, (16,)).astype(mx.uint32)

    def _transform(self, logits):
        return transformed_logprobs(
            logits, self.TEMP, top_p=self.TOP_P, top_k=self.TOP_K
        )

    def _plain(self, n_new, logits_processors=None):
        # Non-speculative reference sampling from the SAME transformed
        # distribution, mirroring self_mtp_generate_step's prefill split
        # and processor history convention (prompt + committed tokens).
        from mlx_lm.hybrid_speculative import _apply_logits_processors

        cache = make_prompt_cache(self.model)
        y = self.prompt.astype(mx.uint32)
        while y.size > 1:
            n = min(512, y.size - 1)
            self.model.model(y[:n][None], cache=cache)
            y = y[n:]
        history = self.prompt.astype(mx.uint32)
        hidden = self.model.model(y[None], cache=cache)
        logits = _apply_logits_processors(
            logits_processors, history, self.model.logits(hidden)[0, -1]
        )
        cur = _sample_from_logprobs(self._transform(logits), self.TEMP)
        toks = [cur]
        for _ in range(n_new - 1):
            h = self.model.model(mx.array([[cur]], mx.uint32), cache=cache)
            history = mx.concatenate(
                [history, mx.array([cur], mx.uint32)]
            )
            logits = _apply_logits_processors(
                logits_processors, history, self.model.logits(h)[0, -1]
            )
            cur = _sample_from_logprobs(self._transform(logits), self.TEMP)
            toks.append(cur)
        return toks

    def _spec(self, seed, max_tokens, persistent=True, **kwargs):
        mx.random.seed(seed)
        stats = HybridStats()
        toks = [
            int(t)
            for t, _lp, _fd in self_mtp_generate_step(
                self.prompt,
                self.model,
                num_draft=2,
                max_tokens=max_tokens,
                sampling_temp=self.TEMP,
                sampling_top_p=self.TOP_P,
                sampling_top_k=self.TOP_K,
                persistent_mtp=persistent,
                stats=stats,
                **kwargs,
            )
        ]
        return toks, stats

    @staticmethod
    def _binned_chi2(a, b, n_top=8):
        # Two-sample chi-square over the reference arm's top-n_top tokens
        # plus a pooled "other" tail. Bins come from one observed arm and
        # sparse cells are pooled, so this is a conservative bound (threshold
        # 32 >> the df<=8 critical value), not a calibrated Pearson test.
        top = [t for t, _ in a.most_common(n_top)]

        def cnt(c, tok):
            if tok == "other":
                return sum(v for k, v in c.items() if k not in top)
            return c.get(tok, 0)

        stat = 0.0
        for tok in top + ["other"]:
            x, y = cnt(a, tok), cnt(b, tok)
            if x + y > 0:
                stat += (x - y) ** 2 / (x + y)
        return stat

    def test_transformed_spec_marginals_match_plain_transformed(self):
        plain = [Counter() for _ in range(self.NEW)]
        spec = [Counter() for _ in range(self.NEW)]
        accepted = proposed = 0
        for i in range(self.N):
            mx.random.seed(1_000 + i)
            for j, t in enumerate(self._plain(self.NEW)):
                plain[j][t] += 1
            toks, stats = self._spec(500_000 + i, self.NEW)
            accepted += stats.draft_accepted
            proposed += stats.draft_proposed
            for j, t in enumerate(toks):
                spec[j][t] += 1
        for j in range(self.NEW):
            stat = self._binned_chi2(plain[j], spec[j])
            self.assertLess(stat, 32.0, f"position={j} chi2={stat:.1f}")
        # The verifier must actually speculate to prove anything.
        self.assertGreater(proposed, 0)
        self.assertGreater(accepted, 0)

    def test_transformed_with_processors_matches_plain(self):
        # logit_bias and a presence penalty compose with the transform the
        # same way they compose with the incumbent temperature-only path:
        # processors adjust the target logits per verify position, then the
        # shared transform runs on the processed logits.
        def procs():
            return make_logits_processors(
                logit_bias={3: 2.5, 7: -3.0},
                presence_penalty=1.2,
                presence_context_size=20,
            )

        n, new = 300, 4
        plain = [Counter() for _ in range(new)]
        spec = [Counter() for _ in range(new)]
        for i in range(n):
            mx.random.seed(30_000 + i)
            for j, t in enumerate(self._plain(new, logits_processors=procs())):
                plain[j][t] += 1
            toks, _ = self._spec(
                830_000 + i, new, logits_processors=procs()
            )
            for j, t in enumerate(toks):
                spec[j][t] += 1
        for j in range(new):
            stat = self._binned_chi2(plain[j], spec[j])
            self.assertLess(stat, 32.0, f"position={j} chi2={stat:.1f}")

    def test_transform_rejects_non_residual_rules(self):
        for rule in ("block", "exact"):
            gen = self_mtp_generate_step(
                self.prompt,
                self.model,
                max_tokens=4,
                sampling_temp=self.TEMP,
                sampling_top_p=self.TOP_P,
                accept_rule=rule,
            )
            with self.assertRaises(ValueError):
                next(gen)

    def test_transformed_path_composes_with_fresh_mtp_cache(self):
        toks, stats = self._spec(77, 24, persistent=False)
        self.assertEqual(len(toks), 24)
        self.assertGreater(stats.draft_proposed, 0)

    def test_transformed_path_composes_with_windowed_persistent_mtp(self):
        # Windowed MTP bounds only the draft head; verification stays
        # full-context and sampling-agnostic, so it must compose with the
        # transformed verifier. Uses the tiny Qwen4 target (the only tiny
        # fixture with a sink-window MTP cache).
        from mlx_lm.models.qwen4_exp import Model, ModelArgs
        from test_qwen4_exp import tiny_args as qwen4_tiny_args

        mx.random.seed(4)
        args = qwen4_tiny_args(mtp_num_hidden_layers=1)
        model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=args.__dict__)
        )
        mx.eval(model.parameters())
        prompt = mx.random.randint(0, 60, (16,)).astype(mx.uint32)
        mx.random.seed(88)
        stats = HybridStats()
        toks = [
            int(t)
            for t, _lp, _fd in self_mtp_generate_step(
                prompt,
                model,
                num_draft=2,
                max_tokens=24,
                sampling_temp=self.TEMP,
                sampling_top_p=self.TOP_P,
                sampling_top_k=self.TOP_K,
                persistent_mtp=True,
                mtp_window_size=8,
                mtp_sink_size=2,
                stats=stats,
            )
        ]
        self.assertEqual(len(toks), 24)
        self.assertGreater(stats.draft_proposed, 0)


class TestTransformedAdmission(unittest.TestCase):
    """_self_mtp_config gating for --self-mtp-transformed-verifier."""

    def cli(self, **overrides):
        ns = types.SimpleNamespace(
            self_mtp=True,
            self_mtp_num_draft=1,
            self_mtp_persistent=True,
            self_mtp_rate_gate=True,
            self_mtp_share_qsa_indices=False,
            self_mtp_share_qsa_indices_min_prompt_tokens=0,
            self_mtp_window_size=0,
            self_mtp_window_sink_size=4,
            self_mtp_window_min_prompt_tokens=0,
            self_mtp_transformed_verifier=False,
            kv_bits=None,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    @staticmethod
    def args(*, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, xtc=0.0):
        return types.SimpleNamespace(
            model=types.SimpleNamespace(draft="default_model"),
            prompt_lookup_ngram=0,
            sampling=types.SimpleNamespace(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                xtc_probability=xtc,
            ),
        )

    def setUp(self):
        self.model = types.SimpleNamespace(mtp=object())

    def test_flag_off_transformed_sampling_fails_closed(self):
        for kwargs in (
            {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
            {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            {"temperature": 0.7, "min_p": 0.1},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(
                    _self_mtp_config(
                        self.args(**kwargs), self.cli(), self.model
                    )
                )

    def test_flag_on_admits_live_profiles_with_transform_params(self):
        cli = self.cli(self_mtp_transformed_verifier=True)
        for kwargs in (
            {"temperature": 1.0, "top_p": 0.95, "top_k": 20},  # thinking
            {"temperature": 0.7, "top_p": 0.8, "top_k": 20},  # non-thinking
            {"temperature": 0.7, "min_p": 0.1},
        ):
            with self.subTest(kwargs=kwargs):
                config = _self_mtp_config(self.args(**kwargs), cli, self.model)
                self.assertIsNotNone(config)
                self.assertEqual(config["accept_rule"], "residual")
                self.assertEqual(config["sampling_temp"], kwargs["temperature"])
                self.assertEqual(config["top_p"], kwargs.get("top_p", 1.0))
                self.assertEqual(config["top_k"], kwargs.get("top_k", 0))
                self.assertEqual(config["min_p"], kwargs.get("min_p", 0.0))

    def test_flag_on_still_fails_closed_on_active_xtc(self):
        cli = self.cli(self_mtp_transformed_verifier=True)
        self.assertIsNone(
            _self_mtp_config(
                self.args(temperature=0.7, top_p=0.8, xtc=0.1),
                cli,
                self.model,
            )
        )

    def test_greedy_xtc_never_engages_and_stays_admitted(self):
        # At temperature 0 the sampler is argmax before any transform, so
        # XTC cannot engage; the request stays on the exact greedy path.
        for cli in (self.cli(), self.cli(self_mtp_transformed_verifier=True)):
            config = _self_mtp_config(
                self.args(temperature=0.0, xtc=0.5), cli, self.model
            )
            self.assertIsNotNone(config)
            self.assertNotIn("top_p", config)

    def test_top_p_zero_is_classified_temperature_only(self):
        # top_p == 0 is a make_sampler no-op, so the request is semantically
        # temperature-only: admitted on the exact path, no transform keys.
        for cli in (self.cli(), self.cli(self_mtp_transformed_verifier=True)):
            config = _self_mtp_config(
                self.args(temperature=0.7, top_p=0.0), cli, self.model
            )
            self.assertIsNotNone(config)
            self.assertNotIn("top_p", config)
            self.assertEqual(config["sampling_temp"], 0.7)

    def test_flag_on_keeps_incumbent_paths_untouched(self):
        cli = self.cli(self_mtp_transformed_verifier=True)
        # Temperature-only: no transform keys, so the engine keeps the
        # incumbent temperature-only path bit-exactly.
        config = _self_mtp_config(self.args(temperature=0.7), cli, self.model)
        self.assertNotIn("top_p", config)
        self.assertNotIn("top_k", config)
        self.assertNotIn("min_p", config)
        # Greedy with ignored filters likewise stays on the exact path.
        config = _self_mtp_config(
            self.args(temperature=0.0, top_p=0.8, top_k=20), cli, self.model
        )
        self.assertNotIn("top_p", config)


if __name__ == "__main__":
    unittest.main()
