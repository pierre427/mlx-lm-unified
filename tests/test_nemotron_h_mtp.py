import unittest

import mlx.core as mx

from mlx_lm.models.nemotron_h import Model, ModelArgs, NemotronHMTP


def tiny_args(**overrides):
    kwargs = dict(
        model_type="nemotron_h",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        max_position_embeddings=1024,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        # >= 32 so the fused ssm step kernel's tiling stays valid
        ssm_state_size=32,
        conv_kernel=4,
        n_groups=2,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern=["M", "*", "E", "-"],
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        moe_latent_size=16,
        n_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


def ssm_states_of(cache_list):
    return [
        (c[0], c[1])
        for c in cache_list
        if c is not None and not c.is_trimmable()
    ]


class TestNemotronHMTP(unittest.TestCase):
    def test_rollback_replay_matches_sequential(self):
        """After a speculative verify forward of `block` tokens,
        rollback(keep=j) must leave Mamba caches in the same state as a
        model that only ever saw the first j block tokens."""
        mx.random.seed(0)
        model = Model(tiny_args())
        prompt = mx.random.randint(0, 64, (1, 8))
        block = mx.random.randint(0, 64, (1, 4))

        for keep in range(0, 5):
            # Ground truth: sequential path that never saw rejected tokens.
            ref_cache = model.make_cache()
            model(prompt, cache=ref_cache)
            if keep > 0:
                model(block[:, :keep], cache=ref_cache)
            ref_states = ssm_states_of(ref_cache)

            # Speculative path: full-width verify, then rollback.
            cache = model.make_cache()
            model(prompt, cache=cache)
            sink = []
            model.backbone(block, cache=cache, ssm_sink=sink)
            model.rollback_speculative_cache(cache, sink, keep, block.shape[1])
            got_states = ssm_states_of(cache)

            for (rc, rs), (gc, gs) in zip(ref_states, got_states):
                if keep == 0 and gs is None:
                    self.assertIsNone(rs) if rs is None else self.assertTrue(
                        mx.allclose(rs, mx.zeros_like(rs)).item()
                    )
                    continue
                # Batched (verify-width) vs sequential matmuls differ by
                # kernel-scheduling numerics (~1e-3 fp32); a wrong slice or
                # stale state is orders of magnitude larger.
                self.assertTrue(
                    mx.allclose(rc, gc, atol=5e-3).item(),
                    f"conv state mismatch at keep={keep}",
                )
                self.assertTrue(
                    mx.allclose(rs, gs, atol=5e-3).item(),
                    f"ssm state mismatch at keep={keep}",
                )
            # KV caches must hold prompt + keep tokens.
            kv = next(c for c in cache if c.is_trimmable())
            self.assertEqual(kv.offset, 8 + keep)

    def test_mtp_step_shapes_and_chaining(self):
        mx.random.seed(0)
        model = Model(tiny_args())
        self.assertIsInstance(model.mtp, NemotronHMTP)
        cache = model.make_cache()
        mtp_cache = model.make_mtp_cache()
        prompt = mx.random.randint(0, 64, (1, 8))
        hidden = model.backbone(prompt, cache=cache)
        logits, post = model.mtp_step(hidden[:, :-1], prompt[:, 1:], mtp_cache)
        self.assertEqual(logits.shape, (1, 7, 64))
        self.assertEqual(post.shape, (1, 7, 32))
        self.assertEqual(mtp_cache[0].offset, 7)
        # Recursive chaining on the MTP's own hidden.
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        logits2, _ = model.mtp_step(post[:, -1:], tok, mtp_cache)
        self.assertEqual(logits2.shape, (1, 1, 64))
        self.assertEqual(mtp_cache[0].offset, 8)
        mtp_cache[0].trim(1)
        self.assertEqual(mtp_cache[0].offset, 7)

    def test_mtp_module_dropped_without_weights(self):
        model = Model(tiny_args())
        # A converted checkpoint without mtp tensors: module must drop so
        # strict loading stays consistent.
        weights = {"backbone.norm_f.weight": mx.ones((32,))}
        model.sanitize(dict(weights))
        self.assertIsNone(model.mtp)

    def test_mtp_expert_stacking(self):
        """sanitize must stack per-expert weights for mtp layers exactly
        as it does for backbone layers."""
        model = Model(tiny_args())
        weights = {"mtp.layers.0.eh_proj.weight": mx.zeros((32, 64))}
        for e in range(4):
            weights[f"mtp.layers.1.mixer.experts.{e}.up_proj.weight"] = mx.zeros(
                (16, 16)
            )
            weights[f"mtp.layers.1.mixer.experts.{e}.down_proj.weight"] = mx.zeros(
                (16, 16)
            )
            weights[f"backbone.layers.2.mixer.experts.{e}.up_proj.weight"] = mx.zeros(
                (16, 16)
            )
            weights[f"backbone.layers.2.mixer.experts.{e}.down_proj.weight"] = (
                mx.zeros((16, 16))
            )
        out = model.sanitize(weights)
        self.assertEqual(
            out["mtp.layers.1.mixer.switch_mlp.fc1.weight"].shape, (4, 16, 16)
        )
        self.assertEqual(
            out["backbone.layers.2.mixer.switch_mlp.fc2.weight"].shape, (4, 16, 16)
        )
        self.assertNotIn("mtp.layers.1.mixer.experts.0.up_proj.weight", out)

    def test_no_mtp_module_without_config(self):
        model = Model(tiny_args(num_nextn_predict_layers=0))
        self.assertFalse(hasattr(model, "mtp"))


if __name__ == "__main__":
    unittest.main()
