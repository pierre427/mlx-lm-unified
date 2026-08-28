# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


def _mlx_version_tuple():
    try:
        return tuple(int(p) for p in mx.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        return (0, 0)


# Metal mishandles the ragged tail tile in the sorted gather_qmm/gather_mm
# path when the flattened row count exceeds 32768 and is not a multiple of
# 64: expert outputs for most rows come out wrong (silent MoE corruption at
# long single-shot prefill, e.g. gpt-oss rambling at 32k context). Only the
# mxfp4 manifestation was fixed in 0.32.0; the affine 4/8-bit NAX variant
# (affine_gather_qmm_rhs_nax short row index, upstream mlx#3922) reproduces
# on 0.32.0.dev20260708 on M5 (re-verified 2026-08-26: rel err 0.68 at
# 32772 rows vs exact at 32768). The pad guard is therefore unconditional:
# pad the sorted rows up to a multiple of 64 (duplicating the last row,
# which keeps the indices sorted); _scatter_unsort gathers only the
# original rows, so the padding never reaches the output. Measured net
# faster than unpadded at the trigger sizes, so there is no cost to
# keeping it on everywhere.
_SORTED_GATHER_TAIL_BUG = True

# Minimum routed-assignment count (tokens x top_k) before the expert-major
# sorted path engages. At 64 with top-10 routing this is M >= 7 tokens, so
# plain decode (M=1) and MTP verify at k=2 (M=3) never sort -- even though a
# 2026-08-28 routing probe measured 1.36x expert reuse already present at M=3
# (results/moe-expert-reuse-20260828.json). Module-level so an A/B can switch
# it on a resident model.
# Default 20 (2026-08-28): engages the sorted path at M>=2, capturing the
# +2.08% MTP-verify win (bit-identical output, non-overlapping IQRs at n=12).
# 20 not lower: sorting at M=1 costs ~0.78% and reuse there is exactly 1.00x,
# so decode (M=1, 10 assignments) stays unsorted.
_GATHER_SORT_MIN_ASSIGNMENTS = 20


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x = x.flatten(0, -3)[order // M]
    indices = indices[order]
    n = indices.size
    if _SORTED_GATHER_TAIL_BUG and n > 32768 and n % 64 != 0:
        pad = 64 - n % 64
        x = mx.concatenate(
            [x, mx.broadcast_to(x[-1:], (pad,) + x.shape[1:])], axis=0
        )
        indices = mx.concatenate(
            [indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0
        )
    return x, indices, inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= _GATHER_SORT_MIN_ASSIGNMENTS
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= _GATHER_SORT_MIN_ASSIGNMENTS
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
