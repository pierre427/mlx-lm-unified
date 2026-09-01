import mlx.core as mx


def sum_head_axis(x: mx.array, *, keepdims: bool = False) -> mx.array:
    """Sum axis 1 without a large strided Metal reduction."""
    if x.ndim < 2 or x.shape[1] == 0:
        raise ValueError("head-axis reduction requires a non-empty axis 1")

    total = x[:, :1]
    for head in range(1, x.shape[1]):
        total = total + x[:, head : head + 1]
    return total if keepdims else total[:, 0]
