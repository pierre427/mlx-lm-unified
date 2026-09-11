"""Runtime flags and mechanism counters for the self-MTP round levers.

Lab staging, 2026-09-08. Three default-off levers on the single-lane self-MTP
round (``hybrid_speculative._mtp_draft_verify_loop_impl``) plus a runtime
toggle for the eager-dispatch lever that already lives in ``qwen4_exp``:

``MLX_QWEN4_PLE_VERIFY_PREFETCH`` (lever a)
    Hash, read and dequantize the PLE rows of the verify slab's host-known
    positions on the PLE prefetch pool while the draft chain is still being
    built, so the layer-2 foreground lookup finds dequantized rows waiting.
``MLX_QWEN4_MTP_DEVICE_SAMPLING`` (lever b)
    Temperature-only drafts take the same categorical draw on device (no
    per-draft ``.item()``), extending the greedy device chain to sampled
    requests. Same op, same lane key, so the token stream is unchanged.
``MLX_QWEN4_MTP_HEDGE_DRAFT`` (lever c)
    After the verify graph is built, queue the NEXT round's draft chain
    behind it under the assumption that every draft lands. A hit skips the
    next round's draft build; a miss rewinds the head cache by the entries
    the hedge appended. Greedy, persistent-head lanes only.
``MLX_QWEN4_EAGER_DISPATCH`` (lever d, jundot/omlx #3469 phase 2)
    Owned by ``qwen4_exp``; ``set_lever("eager_dispatch", ...)`` forwards to
    ``qwen4_exp.set_qwen4_eager_dispatch`` so an in-process A/B can flip
    every arm on one model load.

Flags are read from the environment once at import and can be flipped at
runtime with :func:`set_lever`. Every lever bumps a counter when its
mechanism actually runs; a harness must refuse to record an arm whose
counter stayed at zero (wiki lessons/assert-the-mechanism-ran).
"""

from __future__ import annotations

import os
from typing import Dict


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


PLE_VERIFY_PREFETCH = _env_flag("MLX_QWEN4_PLE_VERIFY_PREFETCH")
DEVICE_SAMPLING = _env_flag("MLX_QWEN4_MTP_DEVICE_SAMPLING")
HEDGE_DRAFT = _env_flag("MLX_QWEN4_MTP_HEDGE_DRAFT")

_LEVER_ATTRS = {
    "ple_verify_prefetch": "PLE_VERIFY_PREFETCH",
    "device_sampling": "DEVICE_SAMPLING",
    "hedge_draft": "HEDGE_DRAFT",
}

COUNTER_NAMES = (
    # Cached-prefix lane preparation: known uncached tail prefetch.
    "ple_tail_prefetch_requests",
    "ple_tail_prefetch_tables",
    "ple_tail_prefetch_declined",
    "ple_tail_prefetch_failures",
    # lever a
    "ple_prefetch_submitted",
    "ple_prefetch_rows",
    "ple_dq_hits",
    "ple_dq_misses",
    # lever b
    "device_sampled_drafts",
    # lever c
    "hedge_built",
    "hedge_hit",
    "hedge_miss",
    "hedge_consumed",
    "hedge_skipped",
    "hedge_discarded",
    # lever d
    "eager_async_evals",
)

_COUNTERS: Dict[str, float] = {name: 0 for name in COUNTER_NAMES}


def bump(name: str, amount=1) -> None:
    """Increment a mechanism counter (GIL-serialized; used from the hot loop
    and the PLE prefetch pool, where a lost increment costs nothing)."""
    _COUNTERS[name] += amount


def counters() -> Dict[str, float]:
    return dict(_COUNTERS)


def reset_counters() -> None:
    for name in COUNTER_NAMES:
        _COUNTERS[name] = 0


def levers() -> Dict[str, bool]:
    """Current state of every lever, including the qwen4_exp-owned one."""
    state = {name: bool(globals()[attr]) for name, attr in _LEVER_ATTRS.items()}
    try:
        from .models import qwen4_exp

        state["eager_dispatch"] = bool(qwen4_exp._EAGER_DISPATCH)
        state["eager_dispatch_stride"] = int(qwen4_exp._EAGER_DISPATCH_STRIDE)
    except Exception:  # pragma: no cover - model module unavailable
        pass
    return state


def set_lever(name: str, enabled: bool) -> bool:
    """Flip one lever at runtime; returns the previous value."""
    if name == "eager_dispatch":
        from .models import qwen4_exp

        return qwen4_exp.set_qwen4_eager_dispatch(bool(enabled))
    attr = _LEVER_ATTRS.get(name)
    if attr is None:
        raise KeyError(f"unknown lever {name!r}; known: {sorted(_LEVER_ATTRS)} + eager_dispatch")
    previous = bool(globals()[attr])
    globals()[attr] = bool(enabled)
    return previous
