"""Opt-in CPU parity gate against Agnes' pinned Transformers source.

Run from the repository root after placing the official revision's
``configuration_agnes.py`` and ``modeling_agnes.py`` in ``/tmp/agnes-source``::

    python tests/agnes_reference_parity.py
"""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_lm.models import agnes


PINNED_REVISION = "24f712ce59379b54c4a141d2708c35daf5ff613b"


def _load_reference(source_dir: Path):
    package_name = "_agnes_pinned_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    sys.modules[package_name] = package
    config = importlib.import_module(f"{package_name}.configuration_agnes")
    model = importlib.import_module(f"{package_name}.modeling_agnes")
    return config, model


def _config():
    return {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 24,
        "parallel_ffn_intermediate_size": 8,
        "vocab_size": 32,
        "layer_types": [agnes.LAYER_DELTA, agnes.LAYER_GLOBAL],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
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


def _run(config_module, model_module, dtype):
    torch.manual_seed(7)
    config = _config()
    reference = model_module.AgnesForCausalLM(
        config_module.AgnesTextConfig(**config)
    ).eval()
    if dtype == "bfloat16":
        reference = reference.to(torch.bfloat16)

    model = agnes.Model(
        agnes.ModelArgs.from_dict(
            {
                "model_type": "agnes",
                "text_config": {"model_type": "agnes_text", **config},
            }
        )
    )
    weight_dtype = mx.bfloat16 if dtype == "bfloat16" else mx.float32
    raw_weights = {
        key: mx.array(value.detach().float().cpu().numpy()).astype(weight_dtype)
        for key, value in reference.state_dict().items()
    }
    weights = model.sanitize(raw_weights)
    model.load_weights(list(weights.items()), strict=True)

    token_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with torch.no_grad():
        expected = reference(torch.from_numpy(token_ids)).logits.float().numpy()
    actual_array = model(mx.array(token_ids.astype(np.int32))).astype(mx.float32)
    cache = model.make_cache()
    cached_array = mx.concatenate(
        [
            model(mx.array(token_ids[:, i : i + 1].astype(np.int32)), cache=cache)
            for i in range(token_ids.shape[1])
        ],
        axis=1,
    ).astype(mx.float32)
    mx.eval(actual_array, cached_array)
    actual = np.array(actual_array)
    cached = np.array(cached_array)

    return {
        "dtype": dtype,
        "reference_revision": PINNED_REVISION,
        "reference_max_abs": float(np.max(np.abs(expected - actual))),
        "reference_mean_abs": float(np.mean(np.abs(expected - actual))),
        "cached_max_abs": float(np.max(np.abs(actual - cached))),
        "argmax_equal": bool(
            np.array_equal(np.argmax(expected, axis=-1), np.argmax(actual, axis=-1))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/agnes-source"))
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    config_module, model_module = _load_reference(args.source_dir)

    results = [
        _run(config_module, model_module, "float32"),
        _run(config_module, model_module, "bfloat16"),
    ]
    print(json.dumps(results, indent=2))
    assert results[0]["reference_max_abs"] < 1e-6
    assert results[1]["reference_max_abs"] <= 1e-3
    assert all(result["argmax_equal"] for result in results)


if __name__ == "__main__":
    main()
