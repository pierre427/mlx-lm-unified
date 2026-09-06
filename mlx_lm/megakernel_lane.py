# Copyright © 2026 Apple Inc.

"""Plain-decode lane on the whole-token persistent megakernel (Qwen4-Exp).

After a stock prefill, every further token of a width-1 completion is one
persistent Metal dispatch: the embedding and PLE lookups stay on the host, the
48-layer chain plus ``lm_head`` run inside the kernel, and the decoder commits
its own ledgers after each token. Measured stock-oracled on the M5 Max
(2026-09-04/05, one pass per cell): 62 tok/s at 1K, 59 at 32K, 58 at 64K, 55
at 128K, 48 at 261,888 tokens against stock plain 36 / 34 / 32 / 31 / 28.

The lane is exact at the token level under the recorded class-2 contract: the
kernel's logits are closer to fp32 than stock at every boundary but not
bit-identical, and a greedy near-tie can resolve differently from the eager
path. It is therefore never combined with speculation, quantized KV, rotating
caches, caller-owned caches, or batching.

At a clean close the lane can PUBLISH its state: it copies the kernel's live
QSA KV columns, raw index keys, pooled block summaries, GDN conv / recurrent
states and PLE conv + n-gram state back into the request's stock cache objects
(``MegakernelDecoder.commit_to_caches``), so the server inserts the stock form
into the prefix cache and a later turn warm-restores the shared prefix instead
of re-prefilling it.  Publish-back is opt-in via ``MLX_QWEN4_MEGAKERNEL_PUBLISH``
(default off, so the deployed lane still stops at the prefill boundary); a
poisoned or failed close never publishes.

One decoder is packed per process (about 4 s and ~7 GiB on the 4-bit
Flash-Next) and reused across requests; a process-wide lock serializes lanes.
``MLX_QWEN4_MEGAKERNEL=1`` enables the lane; ``MLX_QWEN4_MEGAKERNEL_LANE=0``
disables the lane alone.
"""

import logging
import os
import threading
from typing import Optional

import mlx.core as mx

_LANE_LOCK = threading.Lock()


def megakernel_lane_enabled() -> bool:
    try:
        from .models import qwen4_megakernel as MK
    except Exception:
        return False
    if not MK._megakernel_enabled():
        return False
    return os.environ.get("MLX_QWEN4_MEGAKERNEL_LANE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def megakernel_publish_enabled() -> bool:
    """Whether a clean lane close publishes its cache to the prefix cache.

    Default OFF: the deployed lane stops at the prefill boundary until the
    publish-back has been measured.  On, a clean close copies the kernel's KV
    columns, index keys, pooled summaries, GDN conv / recurrent states and PLE
    state back into the request's stock cache objects so a later turn
    warm-restores the shared prefix instead of re-prefilling it.
    """
    return os.environ.get("MLX_QWEN4_MEGAKERNEL_PUBLISH", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _text_model(model):
    text_model = getattr(model, "language_model", None)
    if text_model is None:
        return None
    layers = getattr(getattr(text_model, "model", None), "layers", None)
    if not layers or not hasattr(getattr(layers[0], "self_attn", None), "indexer") and not any(
        hasattr(getattr(layer, "self_attn", None), "indexer") for layer in layers
    ):
        return None
    return text_model


class MegakernelLanePoisoned(RuntimeError):
    """The pack failed after the source rebind began: the model may be mutated
    and must not serve eagerly either. The process has to be restarted."""


def _decoder_for(model):
    """Pack once per process; the pack is keyed by the model object."""
    from .models import qwen4_megakernel_runtime as MR

    poisoned = getattr(model, "_megakernel_lane_poisoned", None)
    if poisoned:
        raise MegakernelLanePoisoned(poisoned)
    decoder = getattr(model, "_megakernel_lane_decoder", None)
    if decoder is not None:
        return decoder
    text_model = _text_model(model)
    args = text_model.args
    try:
        decoder = MR.MegakernelDecoder(
            model, args, max_context=int(args.max_position_embeddings),
            rebind=True, validate=False,
        )
    except MR.MegakernelConstructionError as exc:
        reason = f"{exc} (cause: {exc.__cause__!r})"
        model._megakernel_lane_poisoned = reason
        logging.exception("megakernel lane: construction failed; the model is poisoned")
        raise MegakernelLanePoisoned(reason) from exc
    decision = decoder.admit()
    if not decision.accepted:
        raise RuntimeError(f"megakernel admission refused: {decision.reason}")
    model._megakernel_lane_decoder = decoder
    logging.info("megakernel lane: decoder packed, %s phases", decoder.status_fields().get("phases"))
    return decoder


def preload_megakernel_lane(model):
    """Pack at model load so a construction failure is a startup failure, not a
    mid-request fallback onto a possibly mutated model."""
    if not megakernel_lane_enabled() or _text_model(model) is None:
        return None
    return _decoder_for(model)


class MegakernelLane:
    """Callable replacement for the eager decode ``model(y, cache=...)``."""

    def __init__(self, model, decoder, prompt_cache, status):
        from .models import qwen4_exp as QE

        self.decoder = decoder
        self.status = status
        self.prompt_cache = prompt_cache
        text_model = _text_model(model)
        self.embed = text_model.model.embed_tokens
        self.indexer_of = lambda i: text_model.model.layers[i].self_attn.indexer
        self.ple_fn = None
        self.ple_cache = None
        if decoder.ple_layers:
            index = decoder.ple_layers[0]
            self.ple_fn = text_model.model.layers[index].ple.ple_embedding
            self.ple_cache = QE.Qwen4ArraysCache(size=4)
            self.ple_cache[3] = prompt_cache[index][3]
        self.publish = megakernel_publish_enabled()
        self.tokens = 0
        self.closed = False
        self._publish_receipt = {
            "published": False,
            "reason": ("publish-back disabled"
                       if not self.publish else "close not reached"),
        }
        self.status["publish_back"] = dict(self._publish_receipt)

    def __call__(self, input_tokens: mx.array) -> mx.array:
        ids = input_tokens.reshape(1, -1)
        if ids.shape[1] != 1:
            raise RuntimeError("megakernel lane decodes one token per step")
        embedding = self.embed(ids)[0]
        ple = None
        if self.ple_fn is not None:
            ple = self.ple_fn(ids, self.ple_cache, None)[0]
        logits = self.decoder.step(
            embedding[0], ple_embedding=(ple[0] if ple is not None else None),
            position=self.decoder.position, record=True,
        )
        # The persistent launch is one dispatch; commit needs its outputs, so
        # the step is synchronous by construction (the ladder measures the same).
        mx.eval(logits)
        self.decoder.commit()
        self.tokens += 1
        return logits.reshape(1, 1, -1)

    # -- CompiledDecodeStep-compatible surface for generate_step's loop -------
    def materialize_and_confirm(self, completion_output, y, logprobs, *, phase=""):
        return None  # every step is already synchronous and committed

    def poison(self, error, *, phase=""):
        self.close(error=True)
        return error

    def drain_pending(self):
        self.close(error=False)

    def receipt(self):
        return {
            "kind": "megakernel_lane", "tokens": self.tokens,
            "final_position": self.decoder.position, "closed": self.closed,
            "publish_back": dict(self._publish_receipt),
        }

    def _publish_back(self):
        """Copy the kernel's live state into the request's stock caches.

        Only reached from a clean close. Any refusal is recorded in the
        receipt so the server declines to insert the cache, and never raised:
        a close must not throw.
        """
        try:
            with mx.stream(mx.default_stream(mx.default_device())):
                info = self.decoder.commit_to_caches(
                    self.prompt_cache, indexer_of=self.indexer_of,
                    ple_embedding_cache=self.ple_cache)
            self._publish_receipt = {"published": True, **info}
        except Exception as exc:  # noqa: BLE001 -- a failed publish must not throw
            self._publish_receipt = {
                "published": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            logging.exception("megakernel lane: publish-back failed")

    def close(self, error: bool = False):
        if self.closed:
            return
        self.closed = True
        try:
            if error and getattr(self.decoder, "_pending", None) is not None:
                self.decoder.rollback()
            clean = (
                not error
                and getattr(self.decoder, "_pending", None) is None
                and getattr(self.decoder, "_poisoned_reason", None) is None
            )
            if not self.publish:
                self._publish_receipt = {
                    "published": False, "reason": "publish-back disabled"}
            elif not clean:
                self._publish_receipt = {
                    "published": False, "reason": "close was not clean"}
            else:
                self._publish_back()
        finally:
            self.status["tokens"] = self.tokens
            self.status["final_position"] = self.decoder.position
            self.status["publish_back"] = dict(self._publish_receipt)
            _LANE_LOCK.release()


def attach_megakernel_lane(model, prompt_cache, *, max_tokens, status: Optional[dict]):
    """Return ``(lane, decline_reason)``; a decline leaves the eager path."""
    if not megakernel_lane_enabled():
        return None, "MLX_QWEN4_MEGAKERNEL is not enabled"
    text_model = _text_model(model)
    if text_model is None:
        return None, "model has no Qwen4-Exp indexed-attention stack"
    if not _LANE_LOCK.acquire(blocking=False):
        return None, "another megakernel lane is active"
    try:
        decoder = _decoder_for(model)
        length = max(
            (int(getattr(c, "offset", 0)) for c in prompt_cache if hasattr(c, "offset")),
            default=0,
        )
        capacity = int(text_model.args.max_position_embeddings)
        if length <= 0:
            raise RuntimeError("megakernel lane needs a non-empty prefill")
        if length + max_tokens + 1 > capacity:
            raise RuntimeError(
                f"projected endpoint {length + max_tokens + 1} exceeds the model's "
                f"{capacity} positions"
            )
        with mx.stream(mx.default_stream(mx.default_device())):
            decoder.seed_from_caches(prompt_cache, indexer_of=lambda i:
                                     text_model.model.layers[i].self_attn.indexer)
        decoder.position = length
        lane = MegakernelLane(model, decoder, prompt_cache, status if status is not None else {})
    except MegakernelLanePoisoned:
        _LANE_LOCK.release()
        raise
    except Exception as exc:  # noqa: BLE001 -- any other refusal falls back to eager
        _LANE_LOCK.release()
        return None, f"{type(exc).__name__}: {exc}"
    if status is not None:
        status.update(used=True, seeded_position=length)
    return lane, None
