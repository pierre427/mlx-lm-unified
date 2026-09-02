# Copyright © 2026 mlx-uag lab
"""Absorbed-MLA query-width gate.

The MLA models used to take the absorbed (latent-space) attention branch only
at ``L == 1``, so every speculative-decode verify step (``L = k + 1``) and
every chunked-prefill step fell into the expanded branch and materialized
``k``/``v`` over the whole cache.  These tests pin

  * the crossover arithmetic (``mla.absorbed_max_query``),
  * that both branches are numerically the same function for ``L > 1`` with a
    populated cache and causal masking, and
  * that every MLA twin actually consults the gate.
"""

import os
import unittest

# fp32 allclose harness: MLX defaults fp32 matmul to TF32 on M-series NAX, which
# is ~1e-3 accurate and would swamp the branch-equivalence tolerance below.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx

from mlx_lm.models import mla
from mlx_lm.models.cache import make_prompt_cache

# Files that carry an absorbed/expanded MLA attention branch.  sarvam_mla and
# longcat_flash_ngram reuse deepseek_v3 / longcat_flash attention wholesale.
# This list is *checked* against the tree (see test_every_twin_consults_the_gate)
# rather than trusted, so a new twin cannot quietly copy-paste the old gate.
MLA_TWINS = [
    "bailing_moe_v3",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v32",
    "glm4_moe_lite",
    "kimi_linear",
    "longcat_flash",
]

ROPE_SCALING = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
}


def _tiny_v2():
    from mlx_lm.models import deepseek_v2

    args = deepseek_v2.ModelArgs(
        model_type="deepseek_v2",
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        # r=64, dn=dv=16 -> crossover 64*32/(128-32) = 21, so L in {1..8} is
        # inside the absorbed band and the gate is actually exercised.
        kv_lora_rank=64,
        q_lora_rank=32,
        qk_rope_head_dim=32,
        v_head_dim=16,
        qk_nope_head_dim=16,
        rope_scaling=ROPE_SCALING,
    )
    return deepseek_v2.Model(args), args


def _tiny_v3():
    from mlx_lm.models import deepseek_v3

    args = deepseek_v3.ModelArgs(
        model_type="deepseek_v3",
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        n_shared_experts=1,
        kv_lora_rank=64,
        q_lora_rank=32,
        qk_rope_head_dim=32,
        v_head_dim=16,
        qk_nope_head_dim=16,
        rope_scaling=ROPE_SCALING,
    )
    return deepseek_v3.Model(args), args


class TestAbsorbedCrossover(unittest.TestCase):
    def tearDown(self):
        mla.set_absorbed_max_query_override(None)

    # ---------------------------------------------------------------- formula

    def test_crossover_for_deepseek_sarvam_geometry(self):
        # r=512, dn=dv=128 -> 512*256 / (1024-256) = 170.67 -> 170.
        self.assertEqual(mla.absorbed_max_query(512, 128, 128), 170)

    def test_crossover_matches_closed_form(self):
        for r, dn, dv in [(512, 128, 128), (512, 192, 256), (256, 64, 64), (64, 16, 16)]:
            denom = 2 * r - dn - dv
            self.assertGreater(denom, 0)
            self.assertEqual(
                mla.absorbed_max_query(r, dn, dv), (r * (dn + dv)) // denom
            )

    def test_degenerate_geometry_is_absorbed_at_every_width(self):
        # 2r <= dn+dv: L*(S(2r-D) + rD) <= SrD holds for every L, so the
        # absorbed form is the cheaper one at every query width.
        self.assertEqual(mla.absorbed_max_query(4, 32, 16), mla.ABSORBED_UNBOUNDED)
        self.assertEqual(mla.absorbed_max_query(64, 64, 64), mla.ABSORBED_UNBOUNDED)
        self.assertTrue(mla.use_absorbed_path(64, 4096, (4, 32, 16)))

    def test_limit_depends_on_the_attended_cache_length(self):
        geom = (512, 128, 128)  # DeepSeek-V2/V3/Sarvam
        # A cold prefill (S == L) is never worth absorbing above decode: with
        # S = L the inequality reduces to 2r <= dn+dv, which this geometry
        # (1024 > 256) does not satisfy.
        for length in (2, 8, 32, 128, 170, 512):
            self.assertFalse(
                mla.use_absorbed_path(length, length, geom), f"cold L={length}"
            )
        self.assertTrue(mla.use_absorbed_path(1, 1, geom))
        # Verify widths against a warm cache are comfortably inside the band,
        # and the limit rises toward the asymptote as the cache grows.
        for cache_len in (1024, 4096, 16384, 32768):
            self.assertTrue(mla.use_absorbed_path(8, cache_len, geom), cache_len)
        limits = [mla.absorbed_max_query(*geom, cache_len=s) for s in (1024, 32768)]
        self.assertLess(limits[0], limits[1])
        self.assertLess(limits[-1], mla.absorbed_max_query(*geom))
        self.assertEqual(mla.absorbed_max_query(*geom), 170)

    def test_forward_consults_the_gate_with_the_attended_cache_length(self):
        """Assert the mechanism ran: the forward must pass (L, S), not (L,)."""
        from mlx_lm.models import deepseek_v2

        model, args = _tiny_v2()
        model.eval()
        seen = []
        real = deepseek_v2.use_absorbed_path

        def spy(query_len, cache_len, geometry):
            seen.append((query_len, cache_len, geometry))
            return real(query_len, cache_len, geometry)

        deepseek_v2.use_absorbed_path = spy
        try:
            cache = make_prompt_cache(model)
            mx.eval(model(mx.zeros((1, 12), dtype=mx.int32), cache=cache))
            prefill = list(seen)
            seen.clear()
            mx.eval(model(mx.zeros((1, 3), dtype=mx.int32), cache=cache))
            verify = list(seen)
        finally:
            deepseek_v2.use_absorbed_path = real

        self.assertEqual(len(prefill), args.num_hidden_layers)
        self.assertEqual({(q, s) for q, s, _ in prefill}, {(12, 12)})
        self.assertEqual({(q, s) for q, s, _ in verify}, {(3, 15)})
        # Cold 12-token prefill goes expanded; the 3-row verify chunk against
        # the warm cache goes absorbed.
        self.assertFalse(real(*prefill[0]))
        self.assertTrue(real(*verify[0]))

    def test_override_replaces_the_geometric_limit(self):
        self.assertEqual(mla.absorbed_query_limit(170), 170)
        mla.set_absorbed_max_query_override(0)
        self.assertEqual(mla.absorbed_query_limit(170), 0)
        mla.set_absorbed_max_query_override(4096)
        self.assertEqual(mla.absorbed_query_limit(170), 4096)
        mla.set_absorbed_max_query_override(None)
        self.assertEqual(mla.absorbed_query_limit(170), 170)

    def test_env_parser(self):
        try:
            for raw, want in [(None, None), ("", None), ("  ", None),
                              ("0", 0), ("8", 8), ("-3", 0)]:
                if raw is None:
                    os.environ.pop(mla._ABSORBED_MAX_QUERY_ENV, None)
                else:
                    os.environ[mla._ABSORBED_MAX_QUERY_ENV] = raw
                self.assertEqual(mla._absorbed_env_override(), want, raw)
            # A typo must fail closed, not silently disable the knob.
            os.environ[mla._ABSORBED_MAX_QUERY_ENV] = "junk"
            with self.assertRaises(ValueError):
                mla._absorbed_env_override()
        finally:
            os.environ.pop(mla._ABSORBED_MAX_QUERY_ENV, None)

    # ------------------------------------------------------------------ wiring

    def test_every_twin_consults_the_gate(self):
        import pathlib

        models_dir = pathlib.Path(mla.__file__).parent
        # Discover the twins from the tree instead of trusting the list above:
        # a module that builds its own absorbed-MLA output projection carries
        # the branch, so a *new* twin that copy-pastes the old gate is caught
        # here rather than shipping ungated.
        found = {
            path.stem: path.read_text()
            for path in sorted(models_dir.glob("*.py"))
            if "self.unembed_out = MultiLinear(" in path.read_text()
        }
        self.assertEqual(
            sorted(found), sorted(MLA_TWINS), "MLA twin set changed: update the gate"
        )
        for name, src in found.items():
            self.assertIn("self.absorbed_geometry = (", src, name)
            self.assertIn(
                "use_absorbed_path(\n            L, pe_scores.shape[-1], "
                "self.absorbed_geometry\n        )",
                src.replace("length, pe_scores", "L, pe_scores"),
                name,
            )
            self.assertIn("if absorbed:\n            output = self.unembed_out", src, name)
            # The stale decode-only gate must be gone from the attention branch.
            self.assertNotIn("if L == 1:\n            q_nope = self.embed_q", src, name)
            self.assertNotIn(
                "if length == 1:\n            q_nope = self.embed_q", src, name
            )
            # And no twin may re-introduce a cache-length-blind limit.
            self.assertNotIn("absorbed_query_limit(self.absorbed_max_query)", src, name)

    def test_attention_threshold_from_resolved_geometry(self):
        _, args = _tiny_v2()
        from mlx_lm.models import deepseek_v2

        attn = deepseek_v2.DeepseekV2Attention(args)
        self.assertEqual(attn.absorbed_max_query, mla.absorbed_max_query(64, 16, 16))
        self.assertEqual(attn.absorbed_max_query, 21)

    # ------------------------------------------------------------- equivalence

    def _branch_logits(self, model, args, prompt_len, widths):
        """Return {L: (expanded_logits, absorbed_logits)} with a warm cache."""
        mx.random.seed(0)
        prompt = mx.random.randint(0, args.vocab_size, (1, prompt_len))
        out = {}
        for L in widths:
            chunk = mx.random.randint(0, args.vocab_size, (1, L))
            arms = {}
            for label, override in (("expanded", 0), ("absorbed", 10**6)):
                mla.set_absorbed_max_query_override(override)
                cache = make_prompt_cache(model)
                # Prefill under the *expanded* branch in both arms so the two
                # arms differ only in how the verify-shaped chunk is attended.
                mla.set_absorbed_max_query_override(0)
                model(prompt, cache=cache)
                mx.eval(cache[0].keys)
                mla.set_absorbed_max_query_override(override)
                logits = model(chunk, cache=cache)
                mx.eval(logits)
                arms[label] = logits
            out[L] = (arms["expanded"], arms["absorbed"])
        return out

    def _assert_branches_agree(self, model, args, name):
        for L, (exp, abs_) in self._branch_logits(
            model, args, prompt_len=24, widths=(1, 2, 3, 5, 8)
        ).items():
            self.assertEqual(exp.shape, abs_.shape, f"{name} L={L}")
            self.assertTrue(
                mx.allclose(exp, abs_, atol=1e-4, rtol=1e-4).item(),
                f"{name} L={L}: max |Δ| = "
                f"{mx.max(mx.abs(exp - abs_)).item():.3e}",
            )
            # Greedy stream must be identical, not merely close.
            self.assertTrue(
                mx.array_equal(
                    mx.argmax(exp, axis=-1), mx.argmax(abs_, axis=-1)
                ).item(),
                f"{name} L={L}: greedy tokens differ",
            )

    def test_deepseek_v2_branches_agree(self):
        mx.random.seed(7)
        model, args = _tiny_v2()
        model.eval()
        self._assert_branches_agree(model, args, "deepseek_v2")

    def test_deepseek_v3_branches_agree(self):
        mx.random.seed(11)
        model, args = _tiny_v3()
        model.eval()
        self._assert_branches_agree(model, args, "deepseek_v3")

    def test_verify_shaped_greedy_stream_is_identical(self):
        """A speculative-verify sequence: repeated L=k+1 chunks off one cache."""
        mx.random.seed(13)
        model, args = _tiny_v2()
        model.eval()
        prompt = mx.random.randint(0, args.vocab_size, (1, 32))

        def run(override, width, rounds):
            mla.set_absorbed_max_query_override(override)
            cache = make_prompt_cache(model)
            model(prompt, cache=cache)
            toks = [int(prompt[0, -1].item())]
            stream = []
            for _ in range(rounds):
                chunk = mx.array([toks[-1:] * width])
                logits = model(chunk, cache=cache)
                picks = mx.argmax(logits, axis=-1)
                mx.eval(picks)
                stream.extend(picks[0].tolist())
                toks.append(int(picks[0, -1].item()))
            return stream

        for width in (2, 3, 5, 8):
            self.assertEqual(
                run(0, width, 6), run(10**6, width, 6), f"verify width {width}"
            )

    def test_causal_mask_is_respected_on_the_absorbed_branch(self):
        """Row i of an L-row chunk must not see rows > i."""
        mx.random.seed(17)
        model, args = _tiny_v2()
        model.eval()
        prompt = mx.random.randint(0, args.vocab_size, (1, 16))
        chunk = mx.random.randint(0, args.vocab_size, (1, 6))

        mla.set_absorbed_max_query_override(10**6)
        cache = make_prompt_cache(model)
        model(prompt, cache=cache)
        wide = model(chunk, cache=cache)
        mx.eval(wide)

        # Token-at-a-time through the same absorbed branch: if the L=6 pass
        # leaked future rows, the two would not line up.
        cache = make_prompt_cache(model)
        model(prompt, cache=cache)
        rows = []
        for i in range(chunk.shape[1]):
            rows.append(model(chunk[:, i : i + 1], cache=cache))
        step = mx.concatenate(rows, axis=1)
        mx.eval(step)
        self.assertTrue(
            mx.allclose(wide, step, atol=1e-4, rtol=1e-4).item(),
            f"max |Δ| = {mx.max(mx.abs(wide - step)).item():.3e}",
        )
        self.assertTrue(
            mx.array_equal(
                mx.argmax(wide, axis=-1), mx.argmax(step, axis=-1)
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
