import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from mlx_lm import utils
from mlx_lm.models import agnes


def _text_config():
    return {
        "model_type": "agnes_text",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "parallel_ffn_intermediate_size": 32,
        "vocab_size": 64,
        "layer_types": [agnes.LAYER_DELTA, agnes.LAYER_GLOBAL],
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 64,
        "mtp_num_hidden_layers": 0,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    }


def _write_checkpoint(path: Path, *, quantized: bool) -> None:
    mx.random.seed(7)
    config = {"model_type": "agnes", "text_config": _text_config()}
    model = agnes.Model(agnes.ModelArgs.from_dict(config))
    if quantized:
        projection_names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda name, module: any(
                name.endswith(suffix) for suffix in projection_names
            ),
        )
        config["quantization"] = {
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
        }
    mx.eval(model.parameters())
    utils.save_model(path, model)
    utils.save_config(config, path / "config.json")


def _without_fusion_env():
    environment = dict(os.environ)
    environment.pop("MLX_AGNES_GDN_PROJ_FUSION", None)
    return patch.dict(os.environ, environment, clear=True)


def test_quantized_agnes_load_enables_all_gdn_layers_by_default():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        _write_checkpoint(path, quantized=True)
        with _without_fusion_env():
            model, _ = utils.load_model(path)

    assert model.agnes_gdn_projection_fusion_load_hook_reached == 1
    assert model.agnes_gdn_projection_fusion_layers == 1
    assert model.agnes_gdn_projection_fusion_status == "enabled"
    assert hasattr(model.layers[0].delta_attn, "in_proj_fused")


def test_agnes_load_explicit_opt_out_is_unchanged():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        _write_checkpoint(path, quantized=True)
        with patch.dict(os.environ, {"MLX_AGNES_GDN_PROJ_FUSION": "0"}):
            model, _ = utils.load_model(path)

    assert not hasattr(model, "agnes_gdn_projection_fusion_load_hook_reached")
    assert not hasattr(model.layers[0].delta_attn, "in_proj_fused")
    assert hasattr(model.layers[0].delta_attn, "in_proj_qkv")


def test_unquantized_agnes_load_stays_stock_and_reports_skip():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        _write_checkpoint(path, quantized=False)
        with _without_fusion_env():
            model, _ = utils.load_model(path)

    assert model.agnes_gdn_projection_fusion_load_hook_reached == 1
    assert model.agnes_gdn_projection_fusion_layers == 0
    assert model.agnes_gdn_projection_fusion_status == "skipped_unquantized"
    assert not hasattr(model.layers[0].delta_attn, "in_proj_fused")
    assert hasattr(model.layers[0].delta_attn, "in_proj_qkv")


def test_non_agnes_model_is_untouched():
    class OtherModel:
        pass

    model = OtherModel()
    with _without_fusion_env():
        utils._maybe_fuse_agnes_gdn_projections(model, {"model_type": "llama"})
    assert vars(model) == {}
