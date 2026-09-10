import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models import qwen3_5
from mlx_lm.models.qwen3_5 import (
    GatedDeltaNet,
    TextModelArgs,
    _array_bytes,
    _can_fuse_gdn_projections,
    _fuse_gdn_projection_layer,
    fuse_gated_delta_net_projections,
)


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def _tiny_args():
    return TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        vocab_size=128,
    )


class TinyGDNModel(nn.Module):
    def __init__(self, quantized=True):
        super().__init__()
        self.layers = [GatedDeltaNet(_tiny_args()) for _ in range(2)]
        if quantized:
            nn.quantize(self, group_size=32, bits=4)
        mx.eval(self.parameters())


def _input(rows, seed):
    return (
        mx.random.normal((1, rows, 256), key=mx.random.key(seed)) * 0.3
    ).astype(mx.bfloat16)


def _assert_bits_equal(left, right):
    mx.eval(left, right)
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert _array_bytes(left) == _array_bytes(right)


def test_fusion_is_explicitly_opt_in():
    model = TinyGDNModel()
    assert fuse_gated_delta_net_projections(model) == 0
    assert all(hasattr(layer, "in_proj_qkv") for layer in model.layers)


def test_structural_gate_rejects_unquantized_and_sharded_layers():
    unquantized = TinyGDNModel(quantized=False).layers[0]
    assert not _can_fuse_gdn_projections(unquantized)

    sharded = TinyGDNModel().layers[0]
    sharded.sharding_group = object()
    assert not _can_fuse_gdn_projections(sharded)


def test_enabled_installer_commits_only_verified_layers(monkeypatch):
    model = TinyGDNModel()
    monkeypatch.setattr(
        qwen3_5,
        "_probe_gdn_projection_parity",
        lambda layer: frozenset({mx.bfloat16}),
    )
    assert fuse_gated_delta_net_projections(model, enabled=True) == 2
    assert all(hasattr(layer, "in_proj_fused") for layer in model.layers)


def test_projection_dispatch_is_byte_exact_for_narrow_and_wide_inputs():
    layer = TinyGDNModel().layers[0]
    assert _can_fuse_gdn_projections(layer)

    expected = {}
    for rows in (1, 8, 16):
        inputs = _input(rows, rows)
        expected[rows] = layer._input_projections(inputs)
        mx.eval(expected[rows])

    # This unit test checks the implementation on the CPU. Hardware-specific
    # eligibility remains the install-time probe's responsibility.
    _fuse_gdn_projection_layer(layer, frozenset({mx.bfloat16}))
    for rows in (1, 8, 16):
        actual = layer._input_projections(_input(rows, rows))
        for stock, fused in zip(expected[rows], actual):
            _assert_bits_equal(stock, fused)


def test_full_layer_cached_prefill_and_decode_are_byte_exact():
    stock = GatedDeltaNet(_tiny_args())
    fused = GatedDeltaNet(_tiny_args())
    fused.update(stock.parameters())
    for layer in (stock, fused):
        nn.quantize(layer, group_size=32, bits=4)
        layer.train()
    fused.update(stock.parameters())
    mx.eval(stock.parameters(), fused.parameters())
    _fuse_gdn_projection_layer(fused, frozenset({mx.bfloat16}))

    stock_cache = ArraysCache(size=2)
    fused_cache = ArraysCache(size=2)
    for rows, seed in ((16, 41), (1, 42)):
        inputs = _input(rows, seed)
        stock_output = stock(inputs, cache=stock_cache)
        fused_output = fused(inputs, cache=fused_cache)
        _assert_bits_equal(stock_output, fused_output)
        _assert_bits_equal(stock_cache[0], fused_cache[0])
        _assert_bits_equal(stock_cache[1], fused_cache[1])

    # Exercise the local speculative-rollback branch that is absent from the
    # Rapid-MLX call-body patch and must remain intact in this integration.
    stock_cache.start_speculation()
    fused_cache.start_speculation()
    speculative = _input(4, 43)
    _assert_bits_equal(
        stock(speculative, cache=stock_cache),
        fused(speculative, cache=fused_cache),
    )
    assert stock_cache.trim(2) == fused_cache.trim(2) == 2
    _assert_bits_equal(stock_cache[0], fused_cache[0])
    _assert_bits_equal(stock_cache[1], fused_cache[1])
