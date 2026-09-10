"""Transcendentals that keep their precision inside an ``mx.compile`` span.

MLX's ``Sigmoid`` operator struct spells an **unqualified** ``metal::exp``
(``kernels/unary_ops.h:311``) where its 37 siblings spell
``metal::precise::exp``. The prebuilt ``mlx.metallib`` is compiled offline
with ``-fno-fast-math``, which resolves that call to the precise
implementation, so the eager ``mx.sigmoid`` is accurate. Every kernel MLX
builds at *runtime* from source -- which is what ``mx.compile`` emits for a
fused elementwise chain -- resolves the same call to the fast approximation,
and ``MTLMathModeSafe`` does not move it. A compiled span that swallows an
eager sigmoid is therefore **less accurate than eager**, not merely reordered.
Root cause and bit-level proof:
``wiki/docs/research/mlx-compile-fused-sigmoid-rca-2026-09-03.md``.

``sigmoid`` here is the struct's own text with ``precise::exp`` spelled out,
run through ``mx.fast.metal_kernel``. Two properties, both tested in
``tests/test_compiled_decode.py``:

* it is **bit-identical to eager ``mx.sigmoid``** for float32 and bfloat16;
* being a custom primitive it is **opaque to fusion**, so it stays
  bit-identical when it appears inside an ``mx.compile`` span.

float16 is deliberately excluded: there the eager metallib sigmoid matches the
*fast* form, so the precise kernel would be the one that diverges. Anything
outside float32/bfloat16 falls back to ``mx.sigmoid``.

Call sites use :func:`gate_sigmoid`, which is plain ``mx.sigmoid`` unless a
compiled span is being traced (:func:`precise_span`). Default off means every
existing path keeps its bytes; only a traced decode step opts in.

``nn.silu`` and the ``_precise_swiglu`` / ``compute_g`` helpers are
deliberately **not** routed through this: they are already ``mx.compile``d
upstream, so they use the fast form in *both* arms. Making them precise would
move the stock digest instead of matching it.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import mlx.core as mx

__all__ = ["sigmoid", "gate_sigmoid", "precise_span", "PRECISE_DTYPES"]

PRECISE_DTYPES = (mx.float32, mx.bfloat16)

_SOURCE = """
    uint i = thread_position_in_grid.x;
    T x = inp[i];
    T y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    out[i] = (x < 0) ? y : 1 - y;
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="precise_sigmoid",
            input_names=["inp"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


def sigmoid(x: mx.array) -> mx.array:
    """``mx.sigmoid``'s bytes, in a primitive ``mx.compile`` cannot fuse."""
    if x.dtype not in PRECISE_DTYPES or not mx.metal.is_available():
        return mx.sigmoid(x)
    n = x.size
    return _get_kernel()(
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


# Set only while a compiled decode step is being traced. Read at trace time,
# so the choice is baked into the replayed graph and costs nothing per step.
_IN_PRECISE_SPAN = ContextVar("mlx_lm_in_precise_span", default=False)


@contextmanager
def precise_span():
    """Route :func:`gate_sigmoid` to the precise kernel inside this block."""
    token = _IN_PRECISE_SPAN.set(True)
    try:
        yield
    finally:
        _IN_PRECISE_SPAN.reset(token)


def in_precise_span() -> bool:
    return _IN_PRECISE_SPAN.get()


def gate_sigmoid(x: mx.array) -> mx.array:
    """``mx.sigmoid``, except inside a traced span where it would lose bits."""
    return sigmoid(x) if _IN_PRECISE_SPAN.get() else mx.sigmoid(x)
