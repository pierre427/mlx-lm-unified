# Copyright © 2026 Apple Inc.

"""Speculative-verify acceptance math at temperature > 0 (``accept_rule``).

Kernel tests drive the rule helpers directly with constructed distributions
(no model): the residual (Leviathan/SpecDec) rule must accept drafts the
sampled-token exact-match rule rejects while preserving the target
distribution exactly, and block verification (arXiv 2403.10444) must
preserve the target distribution while accepting at least as many tokens as
the per-token rule. End-to-end tests on the tiny qwen3_5 MTP fixture pin
greedy (temp=0) decisions to be rule-independent bit-identical and check the
temp>0 sampled marginals of every rule against plain non-speculative
sampling.

Every trial is explicitly seeded, so all statistics here are deterministic:
the bounds guard against math regressions, not sampling noise. The kernel
tests carry the exactness burden (constructed p/q make bias unmistakable);
the end-to-end chi-square gate is a consistency check — on a random tiny
model both p and q are near-uniform, so its power against a subtly biased
rule is limited by construction.
"""

import unittest
from collections import Counter

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    HybridStats,
    _accept_sampled_draft,
    _block_verify,
    _residual_sample,
    _sample_from_logprobs,
    _temperature_logprobs,
    self_mtp_generate_step,
)
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel
from mlx_lm.sample_utils import make_reasoning_budget

from test_qwen3_5_mtp import tiny_args


def _lp(probs):
    return mx.log(mx.array(probs, dtype=mx.float32))


def _tv(counts, ref):
    """Total variation between an empirical Counter and a reference pmf."""
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(i, 0) / total - ref[i]) for i in range(len(ref)))


# Constructed distributions: TV(P1, Q1) = 0.5, per-token acceptance
# sum(min(p, q)) = 0.5 at position 1 and 0.55 at position 2.
P1 = [0.5, 0.25, 0.125, 0.125]
Q1 = [0.125, 0.125, 0.25, 0.5]
P2 = [0.25, 0.25, 0.25, 0.25]
Q2 = [0.7, 0.1, 0.1, 0.1]
P3 = [0.5, 0.25, 0.125, 0.125]  # bonus row; any pmf works


class TestAcceptRuleKernel(unittest.TestCase):
    """Rule helpers on constructed logits — no model involved."""

    def test_residual_accepts_where_exact_rejects(self):
        # With p == q the residual rule accepts EVERY draft (ratio == 1),
        # while exact-match of sampled tokens only accepts when two
        # independent samples collide (sum p^2 = 0.34 here).
        lp = _lp(P1)
        n = 800
        exact_hits = rescued = 0
        for i in range(n):
            mx.random.seed(50_000 + i)
            d = _sample_from_logprobs(lp, 1.0)
            t = _sample_from_logprobs(lp, 1.0)
            self.assertTrue(_accept_sampled_draft(lp, lp, d))
            if t == d:
                exact_hits += 1
            else:
                rescued += 1  # residual accepted a draft exact rejected
        self.assertGreater(rescued, 0)
        self.assertLess(exact_hits / n, 0.5)
        self.assertGreater(exact_hits / n, 0.2)

    def test_subfloor_probabilities_accept_correctly(self):
        # min(1, p/q) must be computed in log space: q=1e-35 and p=1e-34 are
        # representable float32 logprobs, and p/q = 10 means CERTAIN
        # acceptance. A linear-space clamp max(q, 1e-30) would instead give
        # p/floor = 1e-4 and reject nearly always, biasing the committed
        # marginal.
        tlp = mx.log(mx.array([0.4, 0.6, 1e-34, 1e-35], dtype=mx.float32))
        dlp = mx.log(mx.array([0.5, 0.5, 1e-35, 1e-34], dtype=mx.float32))
        for i in range(50):
            mx.random.seed(80_000 + i)
            self.assertTrue(_accept_sampled_draft(tlp, dlp, 2))
        # The sub-floor ratio must also be honored below 1: token 3 has
        # p/q = 0.1 (clamped math would give 1e-5, i.e. ~zero accepts).
        n = 2000
        accepts = 0
        for i in range(n):
            mx.random.seed(90_000 + i)
            accepts += _accept_sampled_draft(tlp, dlp, 3)
        self.assertGreater(accepts / n, 0.05)
        self.assertLess(accepts / n, 0.2)

    def test_residual_rule_preserves_target_distribution(self):
        # Draft from q, accept w.p. min(1, p/q), resample rejects from
        # relu(p - q): the committed token must be distributed as p, and
        # acceptance must sit at sum(min(p, q)) = 0.5.
        lp1, lq1 = _lp(P1), _lp(Q1)
        n = 4000
        counts = Counter()
        accepted = 0
        for i in range(n):
            mx.random.seed(10_000 + i)
            d = _sample_from_logprobs(lq1, 1.0)
            if _accept_sampled_draft(lp1, lq1, d):
                counts[d] += 1
                accepted += 1
            else:
                counts[_residual_sample(lp1, lq1, 1.0)] += 1
        self.assertLess(_tv(counts, P1), 0.05)
        self.assertGreater(_tv(counts, Q1), 0.4)  # power: not the draft dist
        self.assertLess(abs(accepted / n - 0.5), 0.05)

    def test_block_verify_preserves_target_and_beats_per_token(self):
        # k=2 block: the first committed token must still be ~ p1, and the
        # expected accepted length must be at least the per-token rule's
        # (arXiv 2403.10444 Thm 1; the cumulative-ratio rescue is worth
        # ~+0.12 tokens/block on these distributions).
        lp1, lq1, lp2, lq2 = _lp(P1), _lp(Q1), _lp(P2), _lp(Q2)
        logprobs = mx.stack([lp1, lp2, _lp(P3)])
        n = 2500
        counts = Counter()
        block_total = token_total = 0
        for i in range(n):
            mx.random.seed(200_000 + i)
            d1 = _sample_from_logprobs(lq1, 1.0)
            d2 = _sample_from_logprobs(lq2, 1.0)
            n_accept, bonus = _block_verify(logprobs, [lq1, lq2], [d1, d2], 1.0)
            counts[d1 if n_accept >= 1 else bonus] += 1
            block_total += n_accept
            # Per-token residual rule on the SAME drafts, its own coins.
            mx.random.seed(700_000 + i)
            if _accept_sampled_draft(lp1, lq1, d1):
                token_total += 1
                if _accept_sampled_draft(lp2, lq2, d2):
                    token_total += 1
        self.assertLess(_tv(counts, P1), 0.06)
        self.assertGreater(_tv(counts, Q1), 0.35)
        self.assertGreaterEqual(block_total / n, token_total / n + 0.05)

    def test_block_verify_k1_matches_token_rule(self):
        # Block size 1 degenerates to Leviathan: acceptance min(1, p/q) and
        # residual relu(p - q), so the committed dist is p and acceptance
        # is sum(min(p, q)) = 0.5.
        lp1, lq1 = _lp(P1), _lp(Q1)
        logprobs = mx.stack([lp1, _lp(P2)])
        n = 2000
        counts = Counter()
        accepted = 0
        for i in range(n):
            mx.random.seed(400_000 + i)
            d1 = _sample_from_logprobs(lq1, 1.0)
            n_accept, bonus = _block_verify(logprobs, [lq1], [d1], 1.0)
            counts[d1 if n_accept >= 1 else bonus] += 1
            accepted += n_accept
        self.assertLess(_tv(counts, P1), 0.06)
        self.assertLess(abs(accepted / n - 0.5), 0.06)


class TestAcceptRuleEndToEnd(unittest.TestCase):
    """accept_rule through self_mtp_generate_step on the tiny qwen3_5 MTP."""

    TEMP = 0.8
    N = 500
    NEW = 5  # max_tokens: cycle 1 runs at k=2, so the block path is exercised

    @classmethod
    def setUpClass(cls):
        mx.random.seed(0)
        cls.model = TextModel(tiny_args())
        mx.eval(cls.model.parameters())
        cls.prompt = mx.random.randint(0, 64, (16,)).astype(mx.uint32)

    def _spec(self, rule, seed, temp, max_tokens=48, **kwargs):
        mx.random.seed(seed)
        stats = HybridStats()
        toks = [
            int(t)
            for t, _lp_, _fd in self_mtp_generate_step(
                self.prompt,
                self.model,
                num_draft=2,
                max_tokens=max_tokens,
                sampling_temp=temp,
                accept_rule=rule,
                stats=stats,
                **kwargs,
            )
        ]
        return toks, stats

    def _plain(self, n_new):
        # Non-speculative reference with the module's own sampling helpers,
        # mirroring self_mtp_generate_step's prefill split so the logits
        # feeding the first sample are value-identical.
        cache = make_prompt_cache(self.model)
        y = self.prompt.astype(mx.uint32)
        while y.size > 1:
            n = min(512, y.size - 1)
            self.model.model(y[:n][None], cache=cache)
            y = y[n:]
        hidden = self.model.model(y[None], cache=cache)
        lp = _temperature_logprobs(self.model.logits(hidden)[0, -1], self.TEMP)
        cur = _sample_from_logprobs(lp, self.TEMP)
        toks = [cur]
        for _ in range(n_new - 1):
            h = self.model.model(mx.array([[cur]], mx.uint32), cache=cache)
            lp = _temperature_logprobs(self.model.logits(h)[0, -1], self.TEMP)
            cur = _sample_from_logprobs(lp, self.TEMP)
            toks.append(cur)
        return toks

    @staticmethod
    def _binned_chi2(a, b, n_top=8):
        # Two-sample chi-square over the reference arm's top-n_top tokens
        # plus a pooled tail: sum (a_i - b_i)^2 / (a_i + b_i), df = n_top.
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

    def test_greedy_decisions_identical_across_rules(self):
        # temp=0 must be EXACTLY unchanged: the rule only engages when
        # sampling, so every accept_rule (and the default) yields the same
        # greedy stream, with and without the persistent MTP cache.
        for persistent in (False, True):
            mx.random.seed(3)
            base, _ = self._spec("residual", 3, 0.0, persistent_mtp=persistent)
            for rule in ("exact", "block"):
                toks, _ = self._spec(rule, 3, 0.0, persistent_mtp=persistent)
                self.assertEqual(toks, base)

    def test_invalid_rule_raises(self):
        gen = self_mtp_generate_step(
            self.prompt, self.model, max_tokens=4, accept_rule="bogus"
        )
        with self.assertRaises(ValueError):
            next(gen)

    def test_stateful_logits_processor_matches_plain_generation(self):
        # The MTP verifier walks speculative prefixes before it knows how many
        # draft tokens will commit. A rewind-aware processor must see the same
        # committed histories, and force the same token, as sequential decode.
        close = 5

        def procs():
            return [
                make_reasoning_budget(
                    think_close=close, max_think_tokens=6, check_every=10**6
                )
            ]

        prompt = mx.array(
            [token for token in self.prompt.tolist() if token != close],
            dtype=mx.uint32,
        )
        n = 18
        plain = [
            int(t)
            for t, _ in generate_step(
                prompt, self.model, max_tokens=n, logits_processors=procs()
            )
        ]
        spec = [
            int(t)
            for t, _lp, _fd in self_mtp_generate_step(
                prompt,
                self.model,
                num_draft=2,
                max_tokens=n,
                logits_processors=procs(),
            )
        ]
        self.assertEqual(spec, plain)
        self.assertIn(close, spec)

    def test_temp_sampling_marginals_match_plain(self):
        # Per-position marginals over N seeded trials for each rule vs the
        # plain non-speculative reference. Threshold 32 ~ chi2(df=8) 0.9999
        # quantile; observed values with these seeds sit at <= ~26.
        n = self.N
        arms = {}
        base_seed = {
            "plain": 1000,
            "residual": 20_000,
            "block": 40_000,
            "exact": 60_000,
        }
        acc = {}
        for name in ("plain", "residual", "block", "exact"):
            counts = [Counter() for _ in range(self.NEW)]
            accepted = proposed = 0
            for i in range(n):
                mx.random.seed(base_seed[name] + i)
                if name == "plain":
                    toks = self._plain(self.NEW)
                else:
                    toks, stats = self._spec(
                        name,
                        base_seed[name] + i,
                        self.TEMP,
                        max_tokens=self.NEW,
                    )
                    accepted += stats.draft_accepted
                    proposed += stats.draft_proposed
                for j, t in enumerate(toks):
                    counts[j][t] += 1
            arms[name] = counts
            acc[name] = accepted / max(proposed, 1)
        for name in ("residual", "block", "exact"):
            for j in range(self.NEW):
                stat = self._binned_chi2(arms["plain"][j], arms[name][j])
                self.assertLess(stat, 32.0, f"arm={name} position={j} chi2={stat:.1f}")
        # The whole point of the residual rule: at temp>0 it accepts a large
        # fraction of the drafts the sampled-token exact-match rule throws
        # away (observed ~0.52 vs ~0.008 here), and block verification
        # accepts at least as much as per-token residual.
        self.assertGreater(acc["residual"], acc["exact"] + 0.3)
        self.assertGreaterEqual(acc["block"], acc["residual"] - 0.03)


if __name__ == "__main__":
    unittest.main()
