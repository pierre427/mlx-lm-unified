import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.quant.gptq import gptq_quantize


class OneLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.proj.weight = weight

    def __call__(self, x):
        return self.proj(x)


def reference_gptq(weight, data, bits, group_size):
    """Column-by-column GPTQ without the lazy per-group update."""
    hessian = data.T @ data
    with mx.stream(mx.cpu):
        damp = 1e-2 * mx.mean(mx.diag(hessian))
        diag = mx.arange(hessian.shape[0])
        hessian[diag, diag] += damp
        hessian = mx.linalg.cholesky(hessian)
        hessian = mx.linalg.cholesky_inv(hessian)
        inverse = mx.linalg.cholesky(hessian, upper=True)
    mx.eval(inverse)

    n_bins = 2**bits - 1
    weight = weight.astype(mx.float32)
    all_scales = []
    all_biases = []
    for start in range(0, weight.shape[-1], group_size):
        end = start + group_size
        _, scales, biases = mx.quantize(
            weight[..., start:end], bits=bits, group_size=group_size
        )
        all_scales.append(scales)
        all_biases.append(biases)
        for column in range(start, end):
            values = weight[..., column : column + 1]
            quantized = mx.clip(
                mx.round((values - biases) / scales), 0.0, n_bins
            )
            quantized = scales * quantized + biases
            error = (values - quantized) / inverse[column, column]
            weight[..., column:] -= (
                error @ inverse[column : column + 1, column:]
            )
            mx.eval(weight)

    scales = mx.concatenate(all_scales, axis=-1)
    biases = mx.concatenate(all_biases, axis=-1)
    quantized = mx.unflatten(weight, -1, (scales.shape[-1], -1))
    quantized = mx.clip(
        mx.round((quantized - biases[..., None]) / scales[..., None]),
        0.0,
        n_bins,
    )
    quantized = scales[..., None] * quantized + biases[..., None]
    return quantized.flatten(-2, -1)


class TestGPTQ(unittest.TestCase):
    def _check(self, in_features, group_size, bits=4, out_features=8):
        mx.random.seed(0)
        weight = mx.random.normal((out_features, in_features))
        data = mx.random.normal((256, in_features))
        data = data + 0.5 * mx.sum(data, axis=-1, keepdims=True)
        mx.eval(weight, data)

        expected = reference_gptq(weight, data, bits, group_size)
        model, _ = gptq_quantize(
            OneLinear(weight),
            data,
            bits=bits,
            group_size=group_size,
            fallback_bits=bits,
            fallback_group_size=group_size,
            batch_size=32,
        )
        layer = model.proj
        actual = mx.dequantize(
            layer.weight, layer.scales, layer.biases, group_size, bits
        ).astype(mx.float32)
        mx.eval(expected, actual)
        self.assertTrue(
            mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item(),
            f"max abs diff {mx.abs(actual - expected).max().item()}",
        )

    def test_single_group(self):
        self._check(in_features=32, group_size=32)

    def test_multiple_groups(self):
        self._check(in_features=128, group_size=32)


if __name__ == "__main__":
    unittest.main()
