"""CPU guard tests for ``_concat_tables`` (qwen3_next fused-scale choke point).

The choke point re-fuses split quantized projection tables. Parts that differ
in quant mode, bits, group_size, or dtype cannot be concatenated without
silently corrupting the scales (arXiv 2609.04098 re-derives this bug class);
``mx.concatenate`` alone would not object. These tests pin the guard: a matching
quartet still concatenates exactly, a mismatched one raises ValueError.
"""

import mlx.core as mx
import pytest

from mlx_lm.models.qwen3_next import _concat_tables


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def _table(rows, k, bits, group_size=64, mode="affine", seed=0):
    """A ``(weight, scales, biases, group_size, bits, mode)`` quant table."""
    mx.random.seed(seed)
    w = mx.random.normal((rows, k)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return (wq, scales, biases, group_size, bits, mode)


def test_matching_quartet_concatenates_exactly():
    # Four parts, identical quant params: the valid case must be untouched.
    parts = [_table(8, 128, bits=4, seed=i) for i in range(4)]
    weight, scales, biases, group_size, bits, mode = _concat_tables(parts, axis=0)
    # Fused along axis 0: shapes add up and the bytes are the parts stacked.
    assert weight.shape[0] == sum(p[0].shape[0] for p in parts)
    assert scales.shape[0] == sum(p[1].shape[0] for p in parts)
    assert biases.shape[0] == sum(p[2].shape[0] for p in parts)
    assert (group_size, bits, mode) == (64, 4, "affine")
    expected = mx.concatenate([p[0] for p in parts], axis=0)
    assert mx.array_equal(weight, expected).item()
    exp_scales = mx.concatenate([p[1] for p in parts], axis=0)
    assert mx.array_equal(scales, exp_scales).item()


def test_bits_mismatch_raises():
    parts = [_table(8, 128, bits=4, seed=0), _table(8, 128, bits=8, seed=1)]
    with pytest.raises(ValueError, match="quant parameter mismatch"):
        _concat_tables(parts, axis=0)


def test_group_size_mismatch_raises():
    parts = [_table(8, 128, bits=4, group_size=64, seed=0),
             _table(8, 128, bits=4, group_size=128, seed=1)]
    with pytest.raises(ValueError, match="quant parameter mismatch"):
        _concat_tables(parts, axis=0)


def test_scales_dtype_mismatch_raises():
    a = _table(8, 128, bits=4, seed=0)
    b = _table(8, 128, bits=4, seed=1)
    # Same quant grid, but scales downcast: the fused scale bytes would be wrong.
    b = (b[0], b[1].astype(mx.bfloat16), b[2], b[3], b[4], b[5])
    with pytest.raises(ValueError, match="scales dtype mismatch"):
        _concat_tables([a, b], axis=0)


def test_quantized_and_dense_mix_raises():
    quant = _table(8, 128, bits=4, seed=0)
    dense = (mx.zeros((8, 128), dtype=mx.float16), None, None, None, None, None)
    with pytest.raises(ValueError, match="quantized and non-quantized"):
        _concat_tables([quant, dense], axis=0)


def test_dense_parts_still_concatenate():
    # A non-quantized fusion (scales None) stays valid and untouched.
    parts = [(mx.ones((4, 16), dtype=mx.float16), None, None, None, None, None),
             (mx.zeros((6, 16), dtype=mx.float16), None, None, None, None, None)]
    weight, scales, biases, *_ = _concat_tables(parts, axis=0)
    assert weight.shape == (10, 16)
    assert scales is None and biases is None
