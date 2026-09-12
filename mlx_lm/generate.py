# Copyright © 2023-2026 Apple Inc.

import argparse
import contextlib
import copy
import functools
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from .batch_admission import AdmissionState, LinearStateCost, StateBudget
from .megakernel_lane import attach_megakernel_lane, megakernel_lane_enabled
from .compiled_decode import (
    CompiledDecodePoisoned,
    CompiledDecodeStep,
    compiled_decode_context_policy,
    compiled_decode_enabled,
    compiled_decode_numerics_accepted,
    compiled_decode_serving_reason,
    model_is_compilable,
    to_shape_stable_cache,
)
from .models import cache
from .models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    TokenBuffer,
    load_prompt_cache,
    make_prompt_cache,
    record_state_checkpoints,
    trim_prompt_cache,
)
from .sample_utils import (
    LaneRNG,
    draw_key,
    make_sampler,
)
from .tokenizer_utils import TokenizerWrapper
from .utils import does_model_support_input_embeddings, load

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.1
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = None
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000
DEFAULT_PREFILL_STEP_SIZE = 2048

# Materialize cache fields that are not otherwise consumed by the sampled-token
# graph often enough to bound their lazy update chains.  This is a defensive
# decode-time maintenance interval, not a performance tuning knob.
CACHE_STATE_EVAL_INTERVAL = 256


def str2bool(string):
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--extra-eos-token",
        type=str,
        default=(),
        nargs="+",
        help="Add tokens in the list of eos tokens that stop generation.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--prefill-response",
        default=None,
        help="Prefill response to be used for the chat template",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=DEFAULT_XTC_THRESHOLD,
        help="Threshold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=DEFAULT_MIN_TOKENS_TO_KEEP,
        help="Minimum tokens to keep for min-p sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        help="Additional config for `apply_chat_template`. Should be a dictionary of"
        " string keys to values represented as a JSON decodable string.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of prompt tokens to process at a time. Smaller values "
        f"lower peak memory during prefill (default: {DEFAULT_PREFILL_STEP_SIZE})",
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-key-bits",
        type=int,
        help="Number of bits for key-cache quantization. Overrides --kv-bits.",
        default=None,
    )
    parser.add_argument(
        "--kv-value-bits",
        type=int,
        help="Number of bits for value-cache quantization. Overrides --kv-bits.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    parser.add_argument(
        "--kv-rotate",
        action="store_true",
        help="Hadamard-rotate the KV cache before quantization, keeping low-bit "
        "--kv-bits near full precision (scores are preserved; head_dim must be a "
        "supported Hadamard size).",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    return parser


# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    device_info = mx.device_info()
    if (
        not mx.metal.is_available()
        or "max_recommended_working_set_size" not in device_info
    ):
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = device_info["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB "
                f"which is close to the maximum recommended size of {max_rec_mb} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    """
    The output of :func:`stream_generate`.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        token (int): The next token.
        from_draft (bool): Whether the token was generated by the draft model.
        logprobs (mx.array): A vector of log probabilities.
        prompt_tokens (int): The number of tokens in the prompt.
        prompt_tps (float): The prompt processing tokens-per-second.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        peak_memory (float): The peak memory used so far in GB.
        finish_reason (str): The reason the response is being sent: "length", "stop" or `None`
        effective_quantized_kv_start (int, optional): The resolved
          ``quantized_kv_start`` actually in effect for this run when
          ``kv_bits`` is set (``None`` otherwise). Recorded as a receipt so a
          harness that reads stats off this response cannot silently measure
          a quantize-from-token-0 schedule while reporting the server's
          delayed-start schedule, or vice versa.
    """

    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None
    effective_quantized_kv_start: Optional[int] = None


def _resolve_kv_bits(kv_bits, key_bits, value_bits):
    if kv_bits is None and key_bits is None and value_bits is None:
        return None, None
    key_bits = kv_bits if key_bits is None else key_bits
    value_bits = kv_bits if value_bits is None else value_bits
    if key_bits is None or value_bits is None:
        raise ValueError(
            "Both key and value bits are required; set --kv-bits as a fallback "
            "or provide both --kv-key-bits and --kv-value-bits."
        )
    return key_bits, value_bits


def validate_kv_quantization_args(
    kv_bits, key_bits, value_bits, group_size, quantized_kv_start
):
    """Validate all KV quantization CLI fields before model loading."""
    key_bits, value_bits = _resolve_kv_bits(kv_bits, key_bits, value_bits)
    if key_bits is not None:
        QuantizedKVCache._validate_config(group_size, key_bits, value_bits)
    if (
        isinstance(quantized_kv_start, bool)
        or not isinstance(quantized_kv_start, int)
        or quantized_kv_start < 0
    ):
        raise ValueError("quantized_kv_start must be a non-negative integer")
    return key_bits, value_bits


def maybe_quantize_kv_cache(
    prompt_cache,
    quantized_kv_start,
    kv_group_size,
    kv_bits,
    kv_key_bits=None,
    kv_value_bits=None,
    kv_rotate=False,
):
    key_bits, value_bits = _resolve_kv_bits(kv_bits, kv_key_bits, kv_value_bits)
    if key_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if isinstance(c, CacheList):
            # Composite per-layer caches (hybrid attention+recurrent layers)
            # hold their KV caches one level down; recurse so nested KV
            # leaves honor kv_bits instead of silently staying fp.
            leaves = list(c.caches)
            maybe_quantize_kv_cache(
                leaves,
                quantized_kv_start,
                kv_group_size,
                kv_bits,
                kv_key_bits=kv_key_bits,
                kv_value_bits=kv_value_bits,
                kv_rotate=kv_rotate,
            )
            c.caches = tuple(leaves)
        elif hasattr(c, "to_quantized"):
            # A cache that can NEVER be quantized is refused here, before the
            # offset gate, so the failure lands at setup rather than at the
            # step where ``offset`` first crosses ``quantized_kv_start``. The
            # cache owns the reason text; a class that also declares this
            # attribute must still define ``to_quantized`` (raising
            # NotImplementedError), or the ``hasattr`` gate above would skip it
            # and the flag would be a silent no-op.
            reason = getattr(c, "kv_quantization_unsupported", None)
            if reason:
                raise ValueError(
                    "KV cache quantization is not available for "
                    f"{type(c).__name__}. {reason}"
                )
            reached_start = c.offset >= quantized_kv_start
            if isinstance(reached_start, mx.array):
                # Batch caches carry one logical offset per row. Format
                # conversion is group-wide, so wait until every row has
                # reached the requested boundary. This is a setup/membership
                # boundary where the one host read is acceptable.
                reached_start = bool(mx.all(reached_start).item())
            if not reached_start:
                continue
            symmetric = key_bits == value_bits and not kv_rotate
            if (
                isinstance(c, (RotatingKVCache, BatchRotatingKVCache))
                and not symmetric
            ):
                # Rotating/sliding-window quantized caches (mlx-lm#1584) support
                # only symmetric, non-rotated quantization — their ring-layout
                # to_quantized(group_size, bits) has no key/value-side or
                # Hadamard-rotate extension. Symmetric --kv-bits is handled below.
                raise ValueError(
                    "Asymmetric or Hadamard-rotated KV quantization is not "
                    f"supported for {type(c).__name__} (sliding-window/rotating "
                    "cache). Use a symmetric --kv-bits without --kv-rotate, or "
                    "quantize only plain KVCache instances."
                )
            try:
                if symmetric:
                    # Preserve the public duck-typed protocol used by
                    # third-party caches: to_quantized(group_size, bits).
                    # Side-specific kwargs and rotation are new extensions.
                    prompt_cache[e] = c.to_quantized(
                        group_size=kv_group_size, bits=key_bits
                    )
                else:
                    prompt_cache[e] = c.to_quantized(
                        group_size=kv_group_size,
                        bits=kv_bits if kv_bits is not None else key_bits,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        rotate=kv_rotate,
                    )
            except NotImplementedError as exc:
                # Keep the cache's own reason, when it gave one -- otherwise
                # the top-level message says only that it declined.
                detail = str(exc).strip()
                raise ValueError(
                    "KV cache quantization is not available for "
                    f"{type(c).__name__}." + (f" {detail}" if detail else "")
                ) from exc


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[
        List[Callable[[mx.array, mx.array], mx.array]]
    ] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: Optional[int] = None,
    kv_rotate: bool = False,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    kv_key_bits: Optional[int] = None,
    kv_value_bits: Optional[int] = None,
    compiled_decode: Optional[bool] = None,
    _prompt_cache_is_request_private: bool = False,
    _compiled_decode_status: Optional[dict] = None,
    _megakernel_status: Optional[dict] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_key_bits (int, optional): Number of bits for key-cache quantization.
          Overrides ``kv_bits`` for keys. Default: ``None``.
        kv_value_bits (int, optional): Number of bits for value-cache quantization.
          Overrides ``kv_bits`` for values. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int, optional): Step to begin using a quantized KV
          cache when ``kv_bits`` is non-None. ``None`` resolves to
          :data:`DEFAULT_QUANTIZED_KV_START`, the same default the CLI/server
          apply -- so a library caller that omits this argument quantizes on
          the same schedule the server serves, instead of silently starting
          from token 0. Pass ``0`` explicitly to quantize from token 0.
          Default: ``None``.
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.
        compiled_decode (bool, optional): Trace the width-1 decode step once
          with ``mx.compile`` and replay it, instead of rebuilding its graph
          every token. Request-private KV caches are converted to
          ``RingKVCache`` after prefill; caller-owned prompt caches are declined
          so the numerical/performance class cannot leak into later requests.
          ``None`` reads ``MLX_LM_COMPILED_DECODE``; the default is on for
          width-1 decode (``0`` opts out). With the default class-1 bucket
          ladder (1023 and 1024 present) no numerical acceptance is needed;
          a custom ladder without them needs
          ``MLX_LM_COMPILED_DECODE_ACCEPTANCE=class3-padded-sdpa-v1`` to
          record explicit acceptance of the padded-SDPA reorder. The
          loader also requires an operator-approved checkpoint manifest at
          ``MLX_LM_COMPILED_DECODE_QUALIFICATION``; family eligibility alone
          is only for direct ``CompiledDecodeStep`` research. The
          default context policy is limited to 4096 tokens. Explicit ``memory``
          and ``latency`` policies extend that to 16384 while keeping or
          skipping the 16384 KV bucket, respectively.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    if quantized_kv_start is None:
        quantized_kv_start = DEFAULT_QUANTIZED_KV_START
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = None

    # Compiled replay currently owns its shape-stable cache for the whole
    # request.  Do not silently replace a caller-owned/session cache: doing so
    # would leak RingKVCache's mask cost and numerical class into later eager
    # requests.
    if _prompt_cache_is_request_private and prompt_cache is None:
        raise ValueError("a request-private prompt cache must be provided")
    caller_supplied_prompt_cache = (
        prompt_cache is not None and not _prompt_cache_is_request_private
    )
    if _compiled_decode_status is not None:
        _compiled_decode_status.update(
            used=False,
            request_private=bool(_prompt_cache_is_request_private),
            decline_reason=None,
        )

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    if kv_bits is not None and any(
        isinstance(c, (cache.RotatingKVCache, cache.BatchRotatingKVCache))
        for c in prompt_cache
    ):
        # Fail at setup, not mid-generation: rotating caches have no
        # quantized implementation, and max_kv_size (now honored by hybrid
        # models like qwen3_next) produces RotatingKVCache layers.
        raise ValueError(
            "kv_bits cannot be combined with rotating/sliding-window caches "
            "(RotatingKVCache quantization is not implemented). Drop kv_bits "
            "or max_kv_size."
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_key_bits=kv_key_bits,
        kv_value_bits=kv_value_bits,
        kv_rotate=kv_rotate,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    if compiled_decode is None:
        compiled_decode = compiled_decode_enabled()
    compiled_step = None

    def _try_compiled_decode():
        """Swap the decode step for a compiled replay, or say why not.

        Declining is the normal outcome for anything this does not cover; the
        caller gets the eager path and a debug line, never an exception.
        """
        nonlocal compiled_step
        if kv_bits is not None:
            return "kv_bits (a quantized cache is not shape-stable)"
        if max_kv_size is not None:
            return "max_kv_size (rotating caches are not shape-stable)"
        if input_embeddings is not None:
            return "input embeddings"
        if caller_supplied_prompt_cache:
            return (
                "caller-owned prompt_cache "
                "(compiled replay requires a private cache)"
            )
        why = model_is_compilable(model, prompt_cache)
        if why is not None:
            return why
        context_tokens = max(
            (c.size() for c in prompt_cache if hasattr(c, "size")), default=0
        )
        why, policy = compiled_decode_context_policy(
            context_tokens, max_tokens
        )
        if why is not None:
            return why
        if not compiled_decode_numerics_accepted(policy):
            return (
                "this bucket ladder is not the class-1 ladder (1023 and 1024 "
                "present) and the padded-SDPA class-3 reorder has not been "
                "accepted; set MLX_LM_COMPILED_DECODE_ACCEPTANCE="
                "class3-padded-sdpa-v1 only after the production-model "
                "numerical gate passes"
            )
        why = compiled_decode_serving_reason(model, policy)
        if why is not None:
            return why
        try:
            converted_cache = to_shape_stable_cache(
                prompt_cache, buckets=policy.buckets, in_place=False
            )
            # Materialize the candidate before publishing it. Ring conversion
            # is lazy in MLX, so allocation/copy failures would otherwise
            # surface only after the caller-owned cache had been replaced.
            mx.eval([c.state for c in converted_cache])
            compiled_step = CompiledDecodeStep(
                model, converted_cache, context_policy=policy
            )
        except (TypeError, ValueError, RuntimeError) as e:
            return str(e)
        prompt_cache[:] = converted_cache
        # The slot plan holds the same cache objects. Point model calls at the
        # request-private list only after the whole setup transaction succeeds.
        compiled_step.cache = prompt_cache
        if _compiled_decode_status is not None:
            _compiled_decode_status.update(
                used=True,
            )
        return None

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        if compiled_step is not None:
            return compiled_step(input_tokens)
        return model(input_tokens, cache=prompt_cache)

    def _compiled_failure(error, phase):
        if compiled_step is None or isinstance(error, CompiledDecodePoisoned):
            return error
        return compiled_step.poison(error, phase=phase)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            completion_output = logits if compiled_step is not None else None
            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0), completion_output

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        checkpoint_base = max(
            (c.size() for c in prompt_cache if hasattr(c, "size")), default=0
        )
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        # NVMe-backed PLE tables expose a prefetcher that warms the next
        # chunk's rows while the current chunk evaluates on the GPU.
        prefill_prefetch = getattr(model, "prefill_prefetch_hook", None)
        prefill_prefetch = (
            prefill_prefetch() if callable(prefill_prefetch) else None
        )
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            processed_tokens = prompt[:n_to_process]
            _model_call(
                input_tokens=processed_tokens[None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            if prefill_prefetch is not None and len(prompt) > n_to_process:
                context_start = max(
                    0, n_to_process - prefill_prefetch.context_len
                )
                prefill_prefetch(
                    np.asarray(
                        prompt[n_to_process : n_to_process + prefill_step_size]
                    ),
                    np.asarray(prompt[context_start:n_to_process]),
                )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            # Prefill advances the model cache without calling _step(), but
            # logits processors still need the complete token history when
            # they first run on the final prompt token.  Preserve prompt ids
            # here even when embeddings supply the model inputs: processors
            # operate on token history, not embedding values.
            if logits_processors and len(processed_tokens) > 0:
                tokens = (
                    mx.concat([tokens, processed_tokens])
                    if tokens is not None
                    else processed_tokens
                )
            prompt_processed_tokens += n_to_process
            record_state_checkpoints(
                prompt_cache, [checkpoint_base + prompt_processed_tokens]
            )
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()

        # The end-of-prefill boundary is the position a regenerate-style
        # prefix-cache trim lands on, so always record it.
        if prompt_processed_tokens > 0:
            record_state_checkpoints(
                prompt_cache,
                [checkpoint_base + prompt_processed_tokens],
                force=True,
            )

        y, logprobs, completion_output = _step(
            input_tokens=prompt, input_embeddings=input_embeddings
        )

        if compiled_decode and max_tokens != 0:
            mx.eval([c.state for c in prompt_cache])
            declined = _try_compiled_decode()
            if declined:
                if _compiled_decode_status is not None:
                    _compiled_decode_status["decline_reason"] = declined
                logging.debug("compiled decode declined: %s", declined)
        if (
            compiled_step is None
            and max_tokens != 0
            and kv_bits is None
            and max_kv_size is None
            and input_embeddings is None
            and not caller_supplied_prompt_cache
            and megakernel_lane_enabled()
        ):
            mx.eval([c.state for c in prompt_cache])
            lane, declined = attach_megakernel_lane(
                model, prompt_cache, max_tokens=max_tokens, status=_megakernel_status
            )
            if lane is not None:
                # The lane speaks CompiledDecodeStep's step interface, so the
                # decode loop drives it unchanged.
                compiled_step = lane
            if declined:
                if _megakernel_status is not None:
                    _megakernel_status["decline_reason"] = declined
                logging.debug("megakernel lane declined: %s", declined)

    n = 0
    terminal_reason = "closed"
    try:
        mx.async_eval(y, logprobs)
        while True:
            if n != max_tokens:
                try:
                    next_y, next_logprobs, next_completion_output = _step(y)
                    mx.async_eval(next_y, next_logprobs)
                except Exception as error:
                    poisoned = _compiled_failure(error, "decode submission")
                    if poisoned is error:
                        raise
                    raise poisoned from error
            if n == 0:
                mx.eval(y)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
            if n == max_tokens:
                terminal_reason = "length"
                break
            try:
                # The first output is eager; later outputs own FIFO receipts.
                if compiled_step is not None and n > 0:
                    compiled_step.materialize_and_confirm(
                        completion_output,
                        y,
                        logprobs,
                        phase="output materialization",
                    )
                token = y.item()
            except Exception as error:
                poisoned = _compiled_failure(error, "output materialization")
                if poisoned is error:
                    raise
                raise poisoned from error
            yield token, logprobs
            if n % CACHE_STATE_EVAL_INTERVAL == 0:
                try:
                    mx.eval([c.state for c in prompt_cache])
                except Exception as error:
                    poisoned = _compiled_failure(error, "cache materialization")
                    if poisoned is error:
                        raise
                    raise poisoned from error
                mx.clear_cache()
            y, logprobs, completion_output = (
                next_y,
                next_logprobs,
                next_completion_output,
            )
            n += 1
    except Exception as error:
        terminal_reason = "error"
        poisoned = _compiled_failure(error, "decode iteration")
        if poisoned is error:
            raise
        raise poisoned from error
    finally:
        if compiled_step is not None:
            try:
                compiled_step.drain_pending()
            except Exception:
                terminal_reason = "error"
                raise
            finally:
                if _compiled_decode_status is not None:
                    _compiled_decode_status["receipt"] = compiled_step.receipt()
                    _compiled_decode_status["terminal_reason"] = (
                        "error" if terminal_reason == "error" else
                        _compiled_decode_status.get("stop_reason", terminal_reason)
                    )


# Reasoning-trace channel tags used by K2-V2 style models. Each pair must
# encode to a single special token id for the relaxed-thinking feature to
# activate on that pair; otherwise the pair is ignored.
THINK_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<think_fast>", "</think_fast>"),
    ("<think_faster>", "</think_faster>"),
)


class ThinkChannelState:
    """Tracks whether generation is currently inside a thinking channel.

    The state machine operates on *committed* tokens only:
      - committing an opening tag id flips ``in_think`` to True
      - committing a closing tag id flips ``in_think`` to False

    Chat templates typically pre-open the think channel in the prompt (the
    model only emits the closing tag), so the initial state is derived by
    scanning the tail of the prompt for an opening tag not followed by a
    closing tag.
    """

    def __init__(self, tag_pairs, prompt_tokens=None, scan_window=50):
        self.open_ids = {o for o, _ in tag_pairs}
        self.close_ids = {c for _, c in tag_pairs}
        self.tag_ids = self.open_ids | self.close_ids
        self.enabled = bool(tag_pairs)
        self.in_think = False
        self._tokenizer = None
        self._numeric_cache = {}
        if self.enabled and prompt_tokens is not None:
            self.in_think = self._scan(list(prompt_tokens)[-scan_window:])

    @classmethod
    def from_tokenizer(cls, tokenizer, prompt_tokens=None, scan_window=50):
        pairs = []
        for open_tag, close_tag in THINK_TAG_PAIRS:
            try:
                oid = tokenizer.encode(open_tag, add_special_tokens=False)
                cid = tokenizer.encode(close_tag, add_special_tokens=False)
            except Exception:
                continue
            if len(oid) == 1 and len(cid) == 1:
                pairs.append((oid[0], cid[0]))
        state = cls(pairs, prompt_tokens, scan_window)
        state._tokenizer = tokenizer
        return state

    def _scan(self, tail):
        state = False
        for tok in tail:
            if tok in self.open_ids:
                state = True
            elif tok in self.close_ids:
                state = False
        return state

    def is_tag(self, tok):
        return tok in self.tag_ids

    def is_numeric(self, tok):
        """True if the token's surface text contains a digit. Numeric tokens
        are exempted from relaxed acceptance: a plausible-but-wrong digit in a
        reasoning trace poisons downstream computation (measured: top-10
        relaxation corrupted '12'->'1' and failed arithmetic probes)."""
        if self._tokenizer is None:
            return False
        v = self._numeric_cache.get(tok)
        if v is None:
            try:
                text = self._tokenizer.decode([tok])
            except Exception:
                text = ""
            v = any(c.isdigit() for c in text)
            self._numeric_cache[tok] = v
        return v

    def on_commit(self, tok):
        if not self.enabled:
            return
        if tok in self.open_ids:
            self.in_think = True
        elif tok in self.close_ids:
            self.in_think = False


def draft_tokens_for_budget(num_draft_tokens: int, remaining_tokens: int) -> int:
    """Return the useful proposal width for a bonus-producing verify cycle.

    Speculative verification always produces one target token in addition to
    the accepted draft prefix.  Reserve that final output slot instead of
    drafting a token that cannot be delivered.  A negative remaining budget
    denotes an unbounded generation.
    """
    num_draft_tokens = max(0, int(num_draft_tokens))
    if remaining_tokens < 0:
        return num_draft_tokens
    return max(0, min(num_draft_tokens, int(remaining_tokens) - 1))


def speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: Optional[int] = None,
    kv_key_bits: Optional[int] = None,
    kv_value_bits: Optional[int] = None,
    kv_rotate: bool = False,
    tokenizer: Optional[Union[PreTrainedTokenizer, TokenizerWrapper]] = None,
    relaxed_topk: Optional[int] = None,
    relaxed_delta: Optional[float] = None,
    speculative_stats: Optional[dict] = None,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        draft_model (nn.Module): The draft model for speculative decoding.
        num_draft_tokens (int, optional): The number of draft tokens for
          speculative decoding. Default: ``2``.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[[mx.array], mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place. The cache must be trimmable.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_key_bits (int, optional): Number of bits for key-cache quantization.
          Overrides ``kv_bits`` for keys. Default: ``None``.
        kv_value_bits (int, optional): Number of bits for value-cache quantization.
          Overrides ``kv_bits`` for values. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int, optional): Step to begin using a quantized KV
          cache when ``kv_bits`` is non-None. ``None`` resolves to
          :data:`DEFAULT_QUANTIZED_KV_START`, the same default the CLI/server
          apply. Pass ``0`` explicitly to quantize from token 0. Default:
          ``None``.
        tokenizer (optional): The tokenizer, used only to resolve the think
          channel tags for relaxed verification / telemetry. Default: ``None``.
        relaxed_topk (int, optional): When set (and the tokenizer has
          single-token think tags), draft tokens proposed *inside* a thinking
          channel are accepted if they fall within the target's top-k, even if
          they are not the argmax. Tokens outside the think channel and the
          think tags themselves are always verified strictly. ``None``
          disables relaxed acceptance entirely (bit-identical to the default
          behavior). Same-tokenizer speculative decoding only. Default: ``None``.
        relaxed_delta (float, optional): Additional logit-gap constraint for
          relaxed acceptance: the draft token is only accepted if
          ``logit_argmax - logit_token <= relaxed_delta``. ``None`` = no gap
          test. Only meaningful when ``relaxed_topk`` is set. Default: ``None``.
        speculative_stats (dict, optional): If provided, mutated in place with
          live telemetry: relaxed/strict accept counts per channel, reject
          counts per channel, current ``in_think`` flag, and the generated
          token index at which the think channel closed. Default: ``None``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, a vector of log probabilities,
          and a bool indicating if the token was generated by the draft model
    """
    if quantized_kv_start is None:
        quantized_kv_start = DEFAULT_QUANTIZED_KV_START

    if max_tokens == 0:
        # A zero-token request must not prefill either model or enter
        # speculative cache bookkeeping.
        return

    # Relaxed speculative verification inside thinking channels (F4).
    # Only built when requested so the default path stays untouched.
    think_state = None
    relaxed_active = False
    if (
        relaxed_topk is not None or speculative_stats is not None
    ) and tokenizer is not None:
        think_state = ThinkChannelState.from_tokenizer(tokenizer, prompt.tolist())
        relaxed_active = relaxed_topk is not None and think_state.enabled
        if speculative_stats is not None:
            speculative_stats.update(
                tags_resolved=think_state.enabled,
                relaxed_active=relaxed_active,
                in_think=think_state.in_think,
                initial_in_think=think_state.in_think,
                relaxed_accepted=0,
                strict_accepted_think=0,
                strict_accepted_answer=0,
                rejected_think=0,
                rejected_answer=0,
                target_committed=0,
                think_committed=0,
                answer_committed=0,
                think_flip_ntoks=None,
            )

    def _commit_telemetry(tok, relaxed=False, from_draft=True, rejected=False):
        # Update channel state + counters for one committed token. Called only
        # when think_state is not None; pure bookkeeping, no numeric effects.
        was_in_think = think_state.in_think
        think_state.on_commit(tok)
        if speculative_stats is None:
            return
        s = speculative_stats
        if was_in_think:
            s["think_committed"] += 1
        else:
            s["answer_committed"] += 1
        if from_draft:
            if relaxed:
                s["relaxed_accepted"] += 1
            elif was_in_think:
                s["strict_accepted_think"] += 1
            else:
                s["strict_accepted_answer"] += 1
        else:
            s["target_committed"] += 1
            if rejected:
                if was_in_think:
                    s["rejected_think"] += 1
                else:
                    s["rejected_answer"] += 1
        if was_in_think and not think_state.in_think:
            s["think_flip_ntoks"] = s["think_committed"] + s["answer_committed"]
        s["in_think"] = think_state.in_think

    y = prompt.astype(mx.uint32)
    # _step appends its input before applying processors.  Seed it with every
    # prompt token except the final prefill token so the first draft and target
    # decisions see the same full-prompt history as generate_step().  The
    # existing rewind arithmetic then preserves this immutable prefix while
    # removing tentative draft suffixes.
    prev_tokens = y[:-1] if logits_processors else None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = make_prompt_cache(model)
        draft_cache = make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    def _can_speculate(c, owner):
        # Trimmable directly, or able to record an exact rollback while
        # speculating (see ArraysCache.record_rollback). The model must also
        # declare support: recording is done by its recurrent layers.
        return c.is_trimmable() or (
            hasattr(c, "record_rollback")
            and getattr(owner, "supports_speculative_rollback", False)
        )

    if not all(_can_speculate(c, model) for c in model_cache):
        types = {
            type(c).__name__ for c in model_cache if not _can_speculate(c, model)
        }
        raise ValueError(
            f"Speculative decoding requires a trimmable target cache "
            f"(got {types})."
        )
    if not all(_can_speculate(c, draft_model) for c in draft_cache):
        types = {
            type(c).__name__
            for c in draft_cache
            if not _can_speculate(c, draft_model)
        }
        raise ValueError(
            f"Speculative decoding requires a trimmable draft cache "
            f"(got {types})."
        )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_key_bits=kv_key_bits,
        kv_value_bits=kv_value_bits,
        kv_rotate=kv_rotate,
    )

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(model, cache, y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=cache)
            logits = logits[:, -n_predict:, :]

            quantize_cache_fn(cache)
            if logits_processors:
                nonlocal prev_tokens
                out_y, out_logprobs = [], []
                if n_predict > 1:
                    y = y[: -(n_predict - 1)]
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, y])
                        if prev_tokens is not None
                        else y
                    )
                    y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(y)
                    out_logprobs.append(logprobs)
                return mx.concatenate(out_y, axis=0), mx.concatenate(
                    out_logprobs, axis=0
                )
            else:
                return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        base = max((c.size() for c in cache if hasattr(c, "size")), default=0)
        processed = 0
        while y.size > 1:
            n_to_process = min(prefill_step_size, y.size - 1)
            model(y[:n_to_process][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            processed += n_to_process
            record_state_checkpoints(cache, [base + processed])
            y = y[n_to_process:]
            mx.clear_cache()
        if processed > 0:
            record_state_checkpoints(cache, [base + processed], force=True)
        return y

    def _rewind_cache(num_draft, num_accept):
        trim_prompt_cache(model_cache, num_draft - num_accept)
        trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y, _ = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    with mx.stream(generation_stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    # After prefill (recording a rollback for the whole prompt would hold on to
    # prompt-sized tensors), let rollback-capable caches start recording so
    # verify steps can be trimmed exactly (no-op for regular KV caches). The
    # draft cache records too: hybrid draft models (e.g. a small Qwen3.5) need
    # their recurrent state rewound just like the target's.
    for c in model_cache + draft_cache:
        c.start_speculation()

    ntoks = 0
    # Set these so the finally block doesn't raise
    num_draft = 0
    n = 0
    try:
        while True:
            remaining = -1 if max_tokens < 0 else max_tokens - ntoks
            num_draft = draft_tokens_for_budget(num_draft_tokens, remaining)
            # Until this round's verify step has advanced the caches there is
            # nothing to rewind; keep n == num_draft so an exception here (or a
            # generator close) makes the finally-block rewind a no-op instead
            # of trimming with values from a previous round.
            n = num_draft
            draft_tokens = _draft_generate(draft_y, num_draft)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
            y = mx.concatenate([y, draft_tokens])
            tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
            mx.eval(tokens, draft_tokens)
            draft_tokens = draft_tokens.tolist()
            tokens = tokens.tolist()
            n = 0
            rejected = False
            while n < num_draft:
                tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                relaxed = False
                if tn != dtn:
                    # Relaxed acceptance: only inside a think channel, and
                    # never for tag tokens (channel boundaries stay
                    # strictly on-policy).
                    if (
                        relaxed_active
                        and think_state.in_think
                        and not think_state.is_tag(dtn)
                        and not think_state.is_tag(tn)
                        and not think_state.is_numeric(dtn)
                        and not think_state.is_numeric(tn)
                    ):
                        gap = (mx.max(lpn) - lpn[dtn]).item()
                        if relaxed_delta is None or gap <= relaxed_delta:
                            n_better = (lpn > lpn[dtn]).sum().item()
                            relaxed = n_better < relaxed_topk
                    if not relaxed:
                        rejected = True
                        break
                n += 1
                ntoks += 1
                committed = dtn if relaxed else tn
                if think_state is not None:
                    _commit_telemetry(committed, relaxed=relaxed)
                yield committed, lpn, True
                if ntoks == max_tokens:
                    break
            if max_tokens < 0 or ntoks < max_tokens:
                ntoks += 1
                if think_state is not None:
                    _commit_telemetry(tokens[n], from_draft=False, rejected=rejected)
                yield tokens[n], logprobs[n], False

            if max_tokens >= 0 and ntoks == max_tokens:
                break

            y = mx.array([tokens[n]], mx.uint32)
            draft_y = y

            # If we accepted all the draft tokens, include the last
            # draft token in the next draft step since it hasn't been
            # processed yet by the draft model
            if n == num_draft:
                draft_y = mx.concatenate(
                    [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                )

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
            _rewind_cache(num_draft, n)
    finally:
        _rewind_cache(num_draft, n)
        for c in model_cache + draft_cache:
            c.stop_speculation()


def _pld_snapshot(caches):
    """Pre-forward cache snapshot so a rejected multi-token proposal can be undone.

    - KVCache: recorded by offset and undone with trim().
    - RotatingKVCache: snapshotted by COPYING its (small, <=window) buffers even
      when is_trimmable() is True at snapshot time. trim() only adjusts offset/_idx
      (it does NOT shrink the key buffer), so if the upcoming multi-token forward
      pushes the cache past the window, a later trim()-rewind desyncs the buffer
      from the mask and crashes attention (mask K-len != cached K-len). Copying is
      always correct.
    - ArraysCache: snapshotted by its epoch-bound speculative position.  The
      rollback deque is bounded, so its retained-span sum is not monotonic
      once old records are evicted and cannot serve as a position marker.
    - CacheList: recursed element-wise.
    Any other cache type is unsupported; prompt-lookup decoding raises rather
    than risk a silently-wrong rewind."""
    def snap_one(c):
        if isinstance(c, CacheList):
            return ("list", [snap_one(sub) for sub in c.caches])
        if isinstance(c, ArraysCache):
            if not c.is_trimmable():
                raise NotImplementedError(
                    "prompt-lookup decoding requires ArraysCache speculation "
                    "recording to be active."
                )
            return ("array_marker", c.rollback_marker())
        if isinstance(c, RotatingKVCache):
            k = None if c.keys is None else mx.array(c.keys)
            v = None if c.values is None else mx.array(c.values)
            return ("restore", k, v, c.offset, getattr(c, "_idx", None))
        if isinstance(c, KVCache):
            return ("trim", c.offset)
        if hasattr(c, "offset") and hasattr(c, "trim"):
            return ("trim", c.offset)
        raise NotImplementedError(
            f"prompt-lookup decoding does not support cache type "
            f"'{type(c).__name__}'. Supported: KVCache, RotatingKVCache (and "
            "rollback-capable ArraysCache / CacheList of those). Disable "
            "prompt_lookup for this model."
        )
    return [snap_one(c) for c in caches]


def _pld_validate_caches(caches):
    """Validate PLD rollback support without enabling speculation."""

    def validate_one(c):
        if isinstance(c, CacheList):
            for sub in c.caches:
                validate_one(sub)
        elif isinstance(c, (ArraysCache, RotatingKVCache, KVCache)):
            return
        elif not (hasattr(c, "offset") and hasattr(c, "trim")):
            raise NotImplementedError(
                f"prompt-lookup decoding does not support cache type "
                f"'{type(c).__name__}'. Supported: KVCache, RotatingKVCache "
                "(and rollback-capable ArraysCache / CacheList of those). "
                "Disable prompt_lookup for this model."
            )

    for c in caches:
        validate_one(c)


def _pld_stop_speculation(caches):
    """Stop speculation on every cache, even if one cleanup hook fails."""
    first_error = None
    for c in caches:
        try:
            c.stop_speculation()
        except Exception as e:
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error


def _pld_start_speculation(caches, rollback_window):
    try:
        for c in caches:
            try:
                c.start_speculation(rollback_window=rollback_window)
            except TypeError:
                c.start_speculation()
    except Exception:
        # A later cache may fail after earlier caches have started. Never leak
        # their rollback buffers on this partial-setup path.
        try:
            _pld_stop_speculation(caches)
        except Exception:
            pass
        raise


def _pld_rewind(caches, snaps):
    def rewind_one(c, s):
        if s[0] == "list":
            for sub, subsnap in zip(c.caches, s[1]):
                rewind_one(sub, subsnap)
        elif s[0] == "array_marker":
            c.rewind_to_rollback_marker(s[1])
        elif s[0] == "trim":
            c.trim(c.offset - s[1])
        else:
            _, k, v, off, idx = s
            c.keys, c.values, c.offset = k, v, off
            if idx is not None:
                c._idx = idx
    for c, s in zip(caches, snaps):
        rewind_one(c, s)


def _pld_offset(c):
    """Logical token offset of a possibly-nested cache leaf. A per-layer
    ``CacheList`` has no offset of its own; its sub-caches advance together, so
    descend to the first offset-bearing sub-cache."""
    if isinstance(c, (list, tuple)):
        for item in c:
            try:
                return _pld_offset(item)
            except AttributeError:
                continue
        raise AttributeError("no offset-bearing cache leaf found")
    while isinstance(c, CacheList):
        c = next((s for s in c.caches if hasattr(s, "offset")), c.caches[0])
    return c.offset


def prompt_lookup_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    backend: Any = "ngram",
    num_draft: int = 8,
    ngram_max: int = 3,
    ngram_min: int = 1,
    prompt_only: bool = False,
    adaptive: bool = False,
    cliff_aware_span: bool = False,
    warmup: int = 48,
    gate: float = 0.12,
    rate_gate: bool = False,
    rate_gate_probe: int = 32,
    rate_gate_margin: float = 0.0,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    stats: Optional[Any] = None,
    history_prompt: Optional[mx.array] = None,
    **_ignored,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Draft-free (prompt-lookup) speculative decoding.

    A pluggable proposer suggests a continuation of the running sequence and the
    target verifies it in one batched forward. ``backend`` selects the proposer:
    ``"ngram"`` (tail n-gram lookup) or ``"suffix_automaton"`` (longest repeated
    suffix — stronger retrieval), or pass a proposer object with
    ``observe(token)`` / ``propose(seq, max_span, prompt_len)``.

    Lossless under the given ``sampler``: at each proposed position the target
    distribution is sampled and the proposal is accepted iff it equals that
    sample (speculative-sampling acceptance for a deterministic drafter), so the
    output distribution matches the target's own (batched) greedy/sampled decode.
    On a partial reject the cache is rewound (handling trimmable and rotating/
    windowed caches). The prompt_cache is left representing exactly prompt+emitted
    at every stop boundary.

    ``adaptive`` enables a one-way never-lose latch: after ``warmup`` tokens, if
    the accepted-proposal fraction is below ``gate`` (i.e. the work isn't
    copy-heavy), it latches once to a plain ``generate_step`` tail — zero
    regression on non-copy output; copy-heavy work never latches. ``stats``
    (a HybridStats) is filled in place. Yields ``(token, logprobs, from_draft)``.

    ``history_prompt`` may be the full prompt when ``prompt`` is only an
    uncached tail backed by a prefilled ``prompt_cache``. Retrieval proposals use
    the full history, while target verification forwards only the uncached tail.
    """
    from .prompt_lookup import (
        HybridStats,
        NgramProposer,
        make_proposer,
        plan_proposal_around_verify_cliff,
        snap_proposal_around_verify_cliff,
    )

    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)
    # Validate cache types up front so unsupported models (e.g. ArraysCache/SSM)
    # fail loud immediately, and record the base offset so a non-empty (reused)
    # cache's existing prefix is preserved by the end-of-run reconciliation.
    _pld_validate_caches(prompt_cache)
    base_offset = _pld_offset(prompt_cache)
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    stats = stats if stats is not None else HybridStats()

    seq = prompt.tolist() if isinstance(prompt, mx.array) else list(prompt)
    history_seq = (
        history_prompt.tolist()
        if isinstance(history_prompt, mx.array)
        else list(history_prompt)
        if history_prompt is not None
        else list(seq)
    )
    if not seq:
        raise ValueError("prompt-lookup decoding requires a non-empty prompt tail")
    if len(history_seq) < len(seq) or history_seq[-len(seq):] != seq:
        raise ValueError("history_prompt must end with the uncached prompt tail")
    prompt_len = len(seq)
    history_prompt_len = len(history_seq)

    if backend == "ngram":
        proposer = NgramProposer(ngram_max, ngram_min, prompt_only)
    else:
        proposer = make_proposer(backend)  # empty; the observe loop below is the
    for t in history_seq:                  # single seeding authority (avoids a
        proposer.observe(t)                # double-seed that desyncs SAM coords)

    # Prefill everything but the last token.
    with mx.stream(generation_stream):
        head = seq[:-1]
        processed = 0
        prompt_progress_callback(0, prompt_len)
        while processed < len(head):
            chunk = head[processed : processed + prefill_step_size]
            model(mx.array(chunk)[None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            processed += len(chunk)
            record_state_checkpoints(prompt_cache, [base_offset + processed])
            prompt_progress_callback(processed, prompt_len)
            mx.clear_cache()
        if processed > 0:
            record_state_checkpoints(
                prompt_cache, [base_offset + processed], force=True
            )
        prompt_progress_callback(prompt_len, prompt_len)

    pending = [seq[-1]]
    generated = 0
    retrieved = 0  # tokens emitted from accepted proposals
    latched = False
    prompt_len = len(seq)
    last_snap = None  # snapshot before the most recent proposal forward
    rate_probed = False  # measured-rate gate is one-shot
    # Wall-clock window for the speculative rate. Armed only after the first
    # speculative cycle: that cycle absorbs one-time costs that are not
    # speculation (kernel warm-up, and on a prompt-cache hit the restore of the
    # reused cache on first use). Measured 2026-09-05 on the 35B: with
    # warmup=8 an un-armed window read 25 ms/token against a 10 ms plain probe
    # and de-latched every cache-hit request after two cycles.
    spec_t0 = None
    spec_gen0 = 0
    # Begin recording recurrent/rotating rollback state only after prompt
    # prefill. Starting earlier retains prompt-sized replay closures in
    # ArraysCache and can cause a large transient memory spike.
    _pld_start_speculation(prompt_cache, max(64, num_draft + 2))
    try:
        while (max_tokens < 0 or generated < max_tokens) and not latched:
            stats.cycles += 1
            # Clamp the proposal to the remaining budget so a cycle never forwards
            # (and commits) more tokens than it will yield -> the cache stays
            # consistent with the yielded prefix at the max_tokens boundary.
            span = num_draft
            available_span = max(num_draft, 15) if max_tokens < 0 else max(
                max_tokens - generated - 1, 0
            )
            if max_tokens >= 0:
                span = min(num_draft, max(max_tokens - generated - 1, 0))
            request_span = span
            if cliff_aware_span and span > 0:
                request_span = plan_proposal_around_verify_cliff(
                    span, available_span, len(pending)
                )
            prop = (
                proposer.propose(history_seq, request_span, history_prompt_len)
                if (request_span > 0 and len(pending) <= 2)
                else []
            )
            if cliff_aware_span and prop:
                raw_prop_len = len(prop)
                prop = snap_proposal_around_verify_cliff(prop, len(pending))
                if len(prop) != raw_prop_len:
                    stats.span_snap_cycles += 1
                    stats.span_snap_tokens += raw_prop_len - len(prop)
                if len(prop) > span:
                    stats.span_extend_cycles += 1
                    stats.span_extend_tokens += len(prop) - span
            x = pending + prop
            verify_rows = len(x)
            stats.verify_span_hist[verify_rows] = (
                stats.verify_span_hist.get(verify_rows, 0) + 1
            )
            snaps = _pld_snapshot(prompt_cache) if prop else None
            if snaps is not None:
                last_snap = snaps

            with mx.stream(generation_stream):
                logits = model(mx.array(x)[None], cache=prompt_cache)[0]
                if logits_processors:
                    mx.eval(logits)
                    logprobs = None
                else:
                    logprobs = logits - mx.logsumexp(
                        logits, axis=-1, keepdims=True
                    )
                    mx.eval(logprobs)

            base = len(pending) - 1
            emit = []  # (token, logprobs_row, from_draft)
            for j in range(len(prop) + 1):
                if logits_processors:
                    # Row j predicts after the committed history plus the j
                    # tentative proposal tokens preceding it. Applying each
                    # processor sequentially with that exact history preserves
                    # generate_step semantics while still verifying all target
                    # rows in one model forward.
                    row_logits = logits[base + j][None]
                    processor_tokens = mx.array(history_seq + prop[:j])
                    for processor in logits_processors:
                        row_logits = processor(processor_tokens, row_logits)
                    row = row_logits[0] - mx.logsumexp(row_logits[0])
                    mx.eval(row)
                else:
                    row = logprobs[base + j]
                s = int(sampler(row[None])[0].item())
                if j < len(prop) and prop[j] == s:
                    emit.append((prop[j], row, True))
                else:
                    emit.append((s, row, False))
                    break
            n_acc = len(emit) - 1

            if prop:
                stats.retrieval_cycles += 1
                stats.retrieval_proposed += len(prop)
            else:
                stats.plain_cycles += 1

            if prop and n_acc < len(prop):
                _pld_rewind(prompt_cache, snaps)
                pending = pending + [t for t, _, _ in emit]
            else:
                pending = [emit[-1][0]]

            for tok, row, from_draft in emit:
                # Count delivered tokens at the yield boundary. The caller may
                # stop on EOS in the middle of an accepted verify batch; eager
                # accounting would then overstate retrieval/output telemetry.
                if prop:
                    if from_draft:
                        stats.retrieval_accepted += 1
                    else:
                        stats.bonus_tokens += 1
                else:
                    stats.plain_tokens += 1
                seq.append(tok)
                history_seq.append(tok)
                proposer.observe(tok)
                generated += 1
                if from_draft:
                    retrieved += 1
                yield tok, row, from_draft
                if max_tokens >= 0 and generated >= max_tokens:
                    return

            if spec_t0 is None:
                spec_t0 = time.perf_counter()
                spec_gen0 = generated
                continue

            # Measured never-slower-than-plain gate: after warmup tokens inside
            # the armed window, time a short plain-decode probe against the
            # observed speculative rate and latch to the plain tail if
            # speculation isn't actually faster. Unlike the acceptance-fraction
            # heuristic below, this measures the real wall-clock break-even
            # (model- and context-dependent). One-shot; its probe tokens are
            # ordinary committed output, so the run stays lossless.
            if (
                rate_gate
                and not rate_probed
                and generated - spec_gen0 >= warmup
                and (max_tokens < 0 or max_tokens - generated > 1)
            ):
                rate_probed = True
                stats.rate_gate_probed = True
                spec_ms = (
                    (time.perf_counter() - spec_t0) * 1000.0
                    / max(generated - spec_gen0, 1)
                )
                budget = rate_gate_probe
                if max_tokens >= 0:
                    budget = min(budget, max_tokens - generated)
                p0 = time.perf_counter()
                n_probe = 0
                while n_probe < budget:
                    with mx.stream(generation_stream):
                        logits = model(mx.array(pending)[None], cache=prompt_cache)[0]
                        last = logits[-1][None]
                        if logits_processors:
                            processor_tokens = mx.array(history_seq)
                            for processor in logits_processors:
                                last = processor(processor_tokens, last)
                        last = last[0]
                        row = last - mx.logsumexp(last, keepdims=True)
                        mx.eval(row)
                    s = int(sampler(row[None])[0].item())
                    seq.append(s)
                    history_seq.append(s)
                    proposer.observe(s)
                    pending = [s]
                    generated += 1
                    n_probe += 1
                    stats.plain_tokens += 1
                    yield s, row, False
                    if max_tokens >= 0 and generated >= max_tokens:
                        return
                plain_ms = (time.perf_counter() - p0) * 1000.0 / max(n_probe, 1)
                stats.rate_gate_spec_ms_per_tok = spec_ms
                stats.rate_gate_plain_ms_per_tok = plain_ms
                if spec_ms > plain_ms * (1.0 - rate_gate_margin):
                    latched = True
                    stats.rate_gate_delatched = True
                else:
                    spec_t0 = time.perf_counter()  # reset window; keep speculating
                    spec_gen0 = generated

            # One-way never-lose latch: once we have enough evidence the work
            # isn't copy-heavy, switch to a plain generate_step tail (bit-exact,
            # no per-cycle proposal overhead) for all remaining tokens.
            if adaptive and generated >= warmup and retrieved / generated < gate:
                latched = True

        if latched and (max_tokens < 0 or generated < max_tokens):
            stats.latched = True
            remaining = -1 if max_tokens < 0 else (max_tokens - generated)
            tail_processors = logits_processors
            if logits_processors:
                if history_seq[-len(pending) :] != pending:
                    raise RuntimeError(
                        "prompt-lookup processor history is not aligned with "
                        "the plain-tail prompt"
                    )
                processor_prefix = mx.array(history_seq[: -len(pending)])

                def with_history(processor):
                    def wrapped(tokens, logits):
                        full_tokens = (
                            mx.concatenate([processor_prefix, tokens])
                            if len(processor_prefix)
                            else tokens
                        )
                        return processor(full_tokens, logits)

                    return wrapped

                tail_processors = [with_history(p) for p in logits_processors]
            for tok, lp in generate_step(
                mx.array(pending),
                model,
                max_tokens=remaining,
                sampler=sampler,
                logits_processors=tail_processors,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
            ):
                seq.append(int(tok))
                generated += 1
                stats.plain_tokens += 1
                yield int(tok), lp, False
    finally:
        try:
            # Leave prompt_cache representing EXACTLY prompt + emitted tokens
            # (like generate_step), even though PLD forwards in batches and may
            # stop mid-batch (e.g. caller breaks on EOS). This keeps callers that
            # persist/reuse the cache (e.g. an LRU prompt cache) correct.
            #   behind -> forward the missing emitted tail.
            #   ahead  -> undo the last proposal forward, then forward the
            #     emitted tail instead.
            target = base_offset + prompt_len + generated
            off = _pld_offset(prompt_cache)
            if off > target and last_snap is not None:
                _pld_rewind(prompt_cache, last_snap)
                off = _pld_offset(prompt_cache)
            if off < target:
                miss = seq[off - base_offset : prompt_len + generated]
                if miss:
                    with mx.stream(generation_stream):
                        model(mx.array(miss)[None], cache=prompt_cache)
                        mx.eval([c.state for c in prompt_cache])
            elif off > target:
                for c in prompt_cache:
                    if c.is_trimmable():
                        c.trim(off - target)
        finally:
            # Reconciliation itself may fail (for example, a model error while
            # forwarding the missing tail). Rollback recording must still stop.
            _pld_stop_speculation(prompt_cache)


def prefill_prompt_cache(
    model: nn.Module,
    tokens: Union[mx.array, List[int]],
    prompt_cache: List[Any],
    *,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """Run ``tokens`` through the model once and leave the state in
    ``prompt_cache``.

    Unlike :func:`generate_step`, every given token is processed: the caller
    keeps the seed token it wants generation to start from. Used by the
    parallel-sampling path, which prefills once at batch size one and then
    replicates the finished cache into one row per sample.

    Returns the same ``prompt_cache`` list for convenience.
    """
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)
    progress_callback = progress_callback or (lambda *_: None)

    total = int(tokens.size)
    with mx.stream(generation_stream):
        checkpoint_base = max(
            (c.size() for c in prompt_cache if hasattr(c, "size")), default=0
        )
        processed = 0
        progress_callback(processed, total)
        while processed < total:
            n_to_process = min(prefill_step_size, total - processed)
            chunk = tokens[processed : processed + n_to_process]
            model(chunk[None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            processed += n_to_process
            record_state_checkpoints(prompt_cache, [checkpoint_base + processed])
            progress_callback(processed, total)
            mx.clear_cache()
        if processed > 0:
            # The prefill boundary is where a prefix-cache trim lands.
            record_state_checkpoints(
                prompt_cache, [checkpoint_base + processed], force=True
            )
    return prompt_cache


def _non_speculative_tokens(token_generator):
    """Forward close() to the inner generator, including early stream exits."""
    with contextlib.closing(token_generator):
        for token, logprobs in token_generator:
            yield token, logprobs, False


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    prompt_lookup: Optional[dict] = None,
    self_mtp: Optional[dict] = None,
    _prompt_cache_is_request_private: bool = False,
    _compiled_decode_status: Optional[dict] = None,
    _megakernel_status: Optional[dict] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt string or
          integer tokens.
        max_tokens (int): The maximum number of tokens to generate.
          Default: ``256``.
        draft_model (Optional[nn.Module]): An optional draft model. If provided
          then speculative decoding is used. The draft model must use the same
          tokenizer as the main model. Default: ``None``.
        self_mtp (Optional[dict]): Enable the model's internal depth-one MTP
          head. This route is intended for a caller that has already gated the
          request to an exact sampling regime and either a cold target cache or
          an exact matching ``state`` sidecar.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens

    # Receipt for the run: the schedule actually in effect when a quantized
    # KV cache is requested. generate_step / speculative_generate_step
    # resolve their own ``quantized_kv_start=None`` to DEFAULT_QUANTIZED_KV_START
    # internally, so mirror that resolution here purely for reporting -- this
    # value is not itself threaded back into kwargs.
    if (
        kwargs.get("kv_bits") is not None
        or kwargs.get("kv_key_bits") is not None
        or kwargs.get("kv_value_bits") is not None
    ):
        _requested_start = kwargs.get("quantized_kv_start")
        effective_quantized_kv_start = (
            DEFAULT_QUANTIZED_KV_START if _requested_start is None else _requested_start
        )
    else:
        effective_quantized_kv_start = None

    # Prompt-lookup (draft-free) speculative decoding. Engages only when it is
    # safe to do losslessly: no draft model, no KV-cache quantization, and no
    # input embeddings / bounded KV (unsupported by the rewind path). Logits
    # processors are applied row-by-row with the exact committed/tentative
    # history after the batched target forward.
    mtp_safe = (
        self_mtp
        and getattr(model, "mtp", None) is not None
        and draft_model is None
        and not prompt_lookup
        and kwargs.get("kv_bits") is None
        and kwargs.get("input_embeddings") is None
        and kwargs.get("max_kv_size") is None
    )
    pld_safe = (
        prompt_lookup
        and draft_model is None
        and kwargs.get("kv_bits") is None
        and kwargs.get("input_embeddings") is None
        and kwargs.get("max_kv_size") is None
    )
    if mtp_safe:
        # Local import avoids generate <-> hybrid_speculative import recursion.
        from .hybrid_speculative import self_mtp_generate_step

        token_generator = self_mtp_generate_step(
            prompt,
            model,
            num_draft=self_mtp.get("num_draft", 1),
            max_tokens=max_tokens,
            prefill_step_size=kwargs.get(
                "prefill_step_size", DEFAULT_PREFILL_STEP_SIZE
            ),
            sampling_temp=self_mtp.get("sampling_temp", 0.0),
            sampling_top_p=self_mtp.get("top_p", 1.0),
            sampling_top_k=self_mtp.get("top_k", 0),
            sampling_min_p=self_mtp.get("min_p", 0.0),
            accept_rule=self_mtp.get("accept_rule", "residual"),
            persistent_mtp=self_mtp.get("persistent", True),
            mtp_window_size=self_mtp.get("window_size"),
            mtp_sink_size=self_mtp.get("sink_size", 4),
            mtp_share_qsa_indices=self_mtp.get("share_qsa_indices", False),
            rate_gate=self_mtp.get("rate_gate", True),
            # Optional per-request depth controller (e.g. the adaptive depth
            # ceiling); None keeps today's fixed-depth path byte-identical.
            speculation_router=self_mtp.get("speculation_router"),
            stats=self_mtp.get("stats"),
            prompt_cache=kwargs.get("prompt_cache"),
            # The server reconstructs and evaluates this key on the generation
            # thread before entering this function.
            lane_rng=self_mtp.get("lane_rng"),
            mtp_state=self_mtp.get("state"),
            mtp_state_out=self_mtp.get("state_out"),
            logits_processors=kwargs.get("logits_processors"),
        )
    elif pld_safe:
        for k in (
            "num_draft_tokens", "relaxed_topk", "relaxed_delta", "speculative_stats",
            "kv_bits", "kv_group_size", "quantized_kv_start",
            "input_embeddings", "max_kv_size", "compiled_decode",
        ):
            kwargs.pop(k, None)
        token_generator = prompt_lookup_generate_step(
            prompt,
            model,
            backend=prompt_lookup.get("backend", "ngram"),
            num_draft=prompt_lookup.get("num_draft", 8),
            ngram_max=prompt_lookup.get("ngram_max", 3),
            ngram_min=prompt_lookup.get("ngram_min", 1),
            prompt_only=prompt_lookup.get("prompt_only", False),
            adaptive=prompt_lookup.get("adaptive", False),
            cliff_aware_span=prompt_lookup.get("cliff_aware_span", False),
            warmup=prompt_lookup.get("warmup", 48),
            gate=prompt_lookup.get("gate", 0.12),
            rate_gate=prompt_lookup.get("rate_gate", False),
            rate_gate_probe=prompt_lookup.get("rate_gate_probe", 32),
            rate_gate_margin=prompt_lookup.get("rate_gate_margin", 0.0),
            stats=prompt_lookup.get("stats"),
            history_prompt=prompt_lookup.get("history_prompt"),
            **kwargs,
        )
    elif draft_model is None:
        kwargs.pop("num_draft_tokens", None)
        kwargs.pop("relaxed_topk", None)
        kwargs.pop("relaxed_delta", None)
        kwargs.pop("speculative_stats", None)
        token_generator = generate_step(
            prompt,
            model,
            _prompt_cache_is_request_private=_prompt_cache_is_request_private,
            _compiled_decode_status=_compiled_decode_status,
            _megakernel_status=_megakernel_status,
            **kwargs,
        )
        # from_draft always false for non-speculative generation
        token_generator = _non_speculative_tokens(token_generator)
    else:
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        # Compiled replay and the megakernel lane are width-1 paths.
        kwargs.pop("compiled_decode", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, tokenizer=tokenizer, **kwargs
        )
    # ``self_mtp_generate_step`` owns the generation stream and may operate
    # within only a few GB of the recommended working-set ceiling. Reapplying
    # ``mx.set_wired_limit`` here after a 100+ GB model has already loaded can
    # trigger a Metal watchdog timeout on the first PLE command. The server
    # establishes its process-wide wired limit at startup; direct callers keep
    # MLX's existing limit. Other generators retain the historical context.
    limit_context = (
        contextlib.nullcontext()
        if mtp_safe
        else wired_limit(model, [generation_stream])
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(limit_context)
        close_token_generator = getattr(token_generator, "close", None)
        if close_token_generator is not None:
            def close_with_reason(exc_type, _exc, _traceback):
                if (
                    _compiled_decode_status is not None
                    and exc_type is not None
                    and exc_type is not GeneratorExit
                ):
                    _compiled_decode_status["stop_reason"] = "error"
                close_token_generator()

            stack.push(close_with_reason)
        tic = time.perf_counter()
        # max_tokens=0 (or a generator that yields nothing) must not reach
        # the final response, which reads the loop variables.
        token = None
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                if _compiled_decode_status is not None:
                    _compiled_decode_status["stop_reason"] = "eos"
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                if _compiled_decode_status is not None:
                    _compiled_decode_status["stop_reason"] = "length"
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
                effective_quantized_kv_start=effective_quantized_kv_start,
            )

        # A final response is completion evidence: settle its lookahead first.
        if close_token_generator is not None:
            close_token_generator()
        detokenizer.finalize()
        if token is None:
            return
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
            effective_quantized_kv_start=effective_quantized_kv_start,
        )


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (Union[str, List[int]]): The input prompt string or integer tokens.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       kwargs: The remaining options get passed to :func:`stream_generate`.
          See :func:`stream_generate` for more details.
    """
    if verbose:
        print("=" * 10)

    text = ""
    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text

    if verbose:
        print()
        print("=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return
        print(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {response.peak_memory:.3f} GB")
        if response.effective_quantized_kv_start is not None:
            print(
                "Quantized KV cache from step: "
                f"{response.effective_quantized_kv_start}"
            )
    return text


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
        effective_quantized_kv_start (int, optional): The
          ``quantized_kv_start`` the generator's caches were built with, when
          ``kv_bits`` is set (``None`` otherwise). ``BatchGenerator`` only
          supports immediate quantization in the batched path, so this is
          always ``0`` when populated -- recorded here so a harness cannot
          conflate it with the delayed-start schedule of the non-batched
          entry points.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0
    effective_quantized_kv_start: Optional[int] = None


def _merge_caches(caches):
    batch_cache = []

    if not caches:
        return batch_cache

    for i in range(len(caches[0])):
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
        else:
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _build_trie(sequences):
    """Build an Aho-Corasick trie from the provided sequences

    See https://en.wikipedia.org/wiki/Aho–Corasick_algorithm .
    """
    trie = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for tok in seq:
                node = node.setdefault(tok, {})
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {})
            node["__match__"] = ((seq,), idx)

    # BFS to set failure links and propagate matches.
    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]
    return trie


def _step_trie(node, trie, x):
    """One step in the Aho-Corasick trie."""
    while x not in node and node is not trie:
        node = node["__fail__"]
    if x in node:
        node = node[x]
    return node


class StopSequenceMatcher:
    """Detect stop sequences in a stream of tokens using an Aho-Corasick trie.

    Any matched sequence signals stop. Used by the batch generator for EOS and
    stop word detection.
    """

    def __init__(self, stop_sequences=None):
        self._trie = _build_trie(stop_sequences) if stop_sequences else {}

    def __deepcopy__(self, memo):
        new = object.__new__(StopSequenceMatcher)
        new._trie = self._trie
        return new

    def make_state(self):
        return self._trie

    @staticmethod
    def match(state, trie, x):
        """Advance by one token. Returns (new_state, matched)."""
        node = _step_trie(state, trie, x)
        return node, node.get("__match__") is not None


class TextStateMachine:
    """A state machine that matches decoded text to track state transitions
    (reasoning, tool calling) and strip the matched control sequences from the
    output.

    Transitions are provided as state -> [(text, new_state)]. Matching on text
    rather than token ids is robust to tokenization differences (e.g. a
    marker's trailing ``>`` being merged with the following byte).

    The runtime state carries a buffer holding text that might be part of a
    control sequence. Text is only emitted once it is known not to be part of
    any match.

    Example:

        sm = TextStateMachine(
            transitions={
                "normal": [("<think>", "reasoning"), ("<tool_call>", "tool")],
                "reasoning": [("</think>", "normal")],
                "tool": [("</tool_call>", "normal")],
            },
        )
        state = sm.make_state(initial="normal")
    """

    def __init__(self, transitions=None):
        self._states = {}
        for src, edges in (transitions or {}).items():
            strings, dst = zip(*edges) if edges else ([], [])
            self._states[src] = (_build_trie(strings), dst)

    def make_state(self, initial="normal"):
        """Create a fresh runtime state (state_name, trie_node, states, buffer)."""
        if initial not in self._states:
            self._states[initial] = (_build_trie([]), [])
        return (initial, self._states[initial][0], self._states, "")

    @staticmethod
    def step(state, text):
        """Consume a chunk of decoded text.

        Returns (new_state, emittable_text, current_state_name) where
        emittable_text is the text safe to show (control sequences stripped,
        possible partial matches held back in the buffer).
        """
        s, n, states, buf = state
        buf += text
        trie = states[s][0]
        emittable = ""
        # buf[:consumed] has been emitted or discarded; buf[consumed:] pending.
        consumed = 0

        for i in range(len(buf)):
            ch = buf[i]
            while ch not in n and n is not trie:
                n = n["__fail__"]
            if ch in n:
                n = n[ch]

            match = n.get("__match__")
            if match is not None:
                match_start = i + 1 - len(match[0])
                emittable += buf[consumed:match_start]
                consumed = i + 1
                s = states[s][1][match[1]]
                if s is None:
                    return (s, None, states, buf[consumed:]), emittable, s
                trie = states[s][0]
                n = trie
            elif n is trie:
                # At the root: no partial match in progress, everything is safe.
                emittable += buf[consumed : i + 1]
                consumed = i + 1

        return (s, n, states, buf[consumed:]), emittable, s

    @staticmethod
    def flush(state):
        """Emit the remaining buffer (use on finish_reason="length")."""
        s, n, states, buf = state
        trie = states[s][0] if s is not None else None
        return (s, trie, states, ""), buf, s

    @staticmethod
    def discard(state):
        """Drop the remaining buffer (use on finish_reason="stop")."""
        s, n, states, buf = state
        trie = states[s][0] if s is not None else None
        return (s, trie, states, ""), s


def make_stop_matcher(tokenizer, stop_words=None):
    """Build a StopSequenceMatcher from EOS tokens and stop words."""
    stop_sequences = [(t,) for t in tokenizer.eos_token_ids]
    for w in stop_words or []:
        stop_sequences.append(tuple(tokenizer.encode(w, add_special_tokens=False)))
    return StopSequenceMatcher(stop_sequences)


def make_text_state_machine(tokenizer, stop_words=None):
    """Build a TextStateMachine with reasoning/tool transitions and stop words.

    Stop words are added as self-transitions in every state so they are
    stripped from the output without changing state.
    """
    transitions = {}

    if tokenizer.has_thinking:
        transitions.setdefault("normal", []).append(
            (tokenizer.think_start, "reasoning")
        )
        transitions["reasoning"] = [(tokenizer.think_end, "normal")]

    if tokenizer.has_tool_calling:
        transitions.setdefault("normal", []).append((tokenizer.tool_call_start, "tool"))
        if tokenizer.has_thinking:
            transitions["reasoning"].append((tokenizer.tool_call_start, "tool"))
        transitions["tool"] = (
            [(tokenizer.tool_call_end, "normal")] if tokenizer.tool_call_end else []
        )

    if stop_words:
        for state_name in set(transitions) | {"normal"}:
            for w in stop_words:
                transitions.setdefault(state_name, []).append((w, state_name))

    return TextStateMachine(transitions or None)


# Optional hook: set mlx_lm.generate.BATCH_UID_HOOK to a callable(uids:
# List[int]) to be notified of the current batch's row-to-uid order
# immediately before each `self.model(...)` forward call inside
# BatchGenerator. None (the default) is a no-op -- zero effect on any
# caller that doesn't set it. Exists so a caller can correlate per-row
# model internals (e.g. per-request steering) with which request occupies
# which batch row, since that mapping isn't otherwise observable from
# outside BatchGenerator once prefill/decode are in flight.
BATCH_UID_HOOK = None


class PromptProcessingBatch:
    """
    A batch processor for prompt tokens with support for incremental processing.

    This class handles batched prompt processing, managing KV caches and preparing
    tokens for generation. It supports extending, filtering, and splitting batches.
    """

    @dataclass
    class Response:
        uid: int
        progress: tuple
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        caches: List[List[Any]],
        tokens: Optional[List[List[int]]] = None,
        prefill_step_size: int = 2048,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        fallback_sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        max_tokens: Optional[List[int]] = None,
        prompt_trim_rollback_tokens: int = 0,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)
        self.prompt_trim_rollback_tokens = max(0, int(prompt_trim_rollback_tokens))
        self._restart_prompt_rollback()
        self.tokens = tokens if tokens is not None else [[] for _ in uids]

        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.stop_matchers = (
            stop_matchers
            if stop_matchers is not None
            else [StopSequenceMatcher()] * len(uids)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else [DEFAULT_MAX_TOKENS] * len(self.uids)
        )

    def __len__(self):
        return len(self.uids)

    def _restart_prompt_rollback(self):
        # (Re)start exact-rollback recording on the batch caches. Records only
        # cover forwards made while batch membership is stable, so this is
        # called whenever the lanes change (init / extend / filter).
        if self.prompt_trim_rollback_tokens > 0:
            for c in self.prompt_cache:
                c.start_speculation(self.prompt_trim_rollback_tokens)

    def _stop_prompt_rollback(self):
        if self.prompt_trim_rollback_tokens > 0:
            for c in self.prompt_cache:
                c.stop_speculation()

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def extend(self, batch):
        if not any(self.samplers):
            self.samplers = [None] * len(self.uids)
        if not any(self.logits_processors):
            # Invariant: empty processor lanes are always [] (an iterable),
            # never None -- the GenerationBatch._step consumer iterates each
            # lane. Build an independent list per lane (never a shared object).
            self.logits_processors = [[] for _ in range(len(self.uids))]
        samplers = batch.samplers if any(batch.samplers) else [None] * len(batch.uids)
        logits_processors = (
            batch.logits_processors
            if any(batch.logits_processors)
            else [[] for _ in range(len(batch.uids))]
        )

        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.prompt_trim_rollback_tokens = max(
            self.prompt_trim_rollback_tokens,
            getattr(batch, "prompt_trim_rollback_tokens", 0),
        )
        # Lanes changed; restart rollback recording on the extended caches.
        self._restart_prompt_rollback()
        self.tokens.extend(batch.tokens)
        self.samplers.extend(samplers)
        self.logits_processors.extend(logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_matchers.extend(batch.stop_matchers)

    def _copy(self, deep: bool = True):
        new_batch = self.__class__.__new__(self.__class__)
        new_batch.model = self.model
        new_batch.uids = list(self.uids)
        new_batch.prompt_trim_rollback_tokens = self.prompt_trim_rollback_tokens
        new_batch.prompt_cache = (
            copy.deepcopy(self.prompt_cache) if deep else self.prompt_cache
        )
        new_batch.tokens = list(self.tokens)
        new_batch.prefill_step_size = self.prefill_step_size
        new_batch.samplers = list(self.samplers)
        new_batch.fallback_sampler = self.fallback_sampler
        new_batch.logits_processors = list(self.logits_processors)
        new_batch.stop_matchers = list(self.stop_matchers)
        new_batch.max_tokens = list(self.max_tokens)
        return new_batch

    def split(self, indices: List[int]):
        indices = sorted(indices)
        indices_left = sorted(set(range(len(self.uids))) - set(indices))
        if not indices_left:
            # Every row leaves: hand the merged cache over instead of deep
            # copying it and immediately dropping the original. The two halves
            # only need independent caches when both keep rows.
            new_batch = self._copy(deep=False)
            self.prompt_cache = []
            self.filter([])
            return new_batch
        new_batch = self._copy()
        self.filter(indices_left)
        new_batch.filter(indices)

        return new_batch

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        else:
            self.samplers = [None] * len(keep)
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        else:
            # Independent [] per lane -- [[]] * n would share one list object
            # so an in-place append on one lane would mutate all of them.
            self.logits_processors = [[] for _ in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_matchers = [self.stop_matchers[idx] for idx in keep]
        # Lane indices changed; recorded rollbacks refer to the old lanes.
        self._restart_prompt_rollback()

    def prompt(self, tokens: List[List[int]]):
        """
        Process prompt tokens through the model.

        Args:
            tokens: List of token sequences to process.
        """
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")

        if not tokens:
            return

        # Add the tokens to the self.tokens so they represent the tokens
        # contained in the KV Cache.
        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        # Calculate if we need to pad
        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)

        # Absolute per-lane positions for recurrent-state checkpoints. A lane
        # that exhausts its (right-padded) prompt mid-chunk holds a frozen,
        # exact state at its own total, so clamp per lane.
        totals = [len(st) for st in self.tokens]
        bases = [t - l for t, l in zip(totals, lengths)]

        # Prepare the caches and inputs. Right pad if needed otherwise just
        # cast to array.
        if max_padding > 0:
            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        # Actual prompt processing loop
        processed = 0
        # NVMe-backed PLE tables expose a prefetcher that warms the next
        # chunk's rows while the current chunk evaluates on the GPU.
        prefill_prefetch = getattr(self.model, "prefill_prefetch_hook", None)
        prefill_prefetch = (
            prefill_prefetch() if callable(prefill_prefetch) else None
        )
        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            if BATCH_UID_HOOK is not None:
                BATCH_UID_HOOK(list(self.uids))
            self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
            if prefill_prefetch is not None and tokens.shape[1] > n_to_process:
                context_start = max(
                    0, n_to_process - prefill_prefetch.context_len
                )
                prefill_prefetch(
                    np.asarray(
                        tokens[
                            :, n_to_process : n_to_process + self.prefill_step_size
                        ]
                    ),
                    np.asarray(tokens[:, context_start:n_to_process]),
                )
            mx.eval([c.state for c in self.prompt_cache])
            processed += n_to_process
            record_state_checkpoints(
                self.prompt_cache,
                [b + min(processed, l) for b, l in zip(bases, lengths)],
            )
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]

        # Finalize the cache if there was any padding
        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

        # Segment boundaries are exactly the positions the server keys prefix
        # cache entries on; always record the final one.
        record_state_checkpoints(self.prompt_cache, totals, force=True)

    def generate(self, tokens: List[List[int]]):
        """
        Transition from prompt processing to generation.

        Args:
            tokens: Final tokens for each sequence to start generation.

        Returns:
            A GenerationBatch ready for token generation.
        """
        if any(len(t) > 1 for t in tokens):
            self.prompt([t[:-1] for t in tokens])
        last_token = mx.array([t[-1] for t in tokens])

        # Rollback recording only covers prompt processing; release it before
        # the caches are handed off to generation.
        self._stop_prompt_rollback()

        generation = GenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.stop_matchers,
            self.max_tokens,
        )

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []

        return generation

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
        prefill_step_size: int = 2048,
        prompt_trim_rollback_tokens: int = 0,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_matchers=[],
            prompt_trim_rollback_tokens=prompt_trim_rollback_tokens,
        )


class GenerationBatch:
    """
    A batched token generator that manages multiple sequences in parallel.

    This class handles the generation phase after prompt processing, managing
    KV caches, sampling, and stop sequence detection for multiple sequences.
    """

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]
        from_draft: bool = False
        mtp_state: Optional[Tuple[List[Any], mx.array]] = None
        lane_rng: Optional[LaneRNG] = None
        rng_draws: int = 0
        mtp_receipt: Optional[dict] = None

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        tokens: List[List[int]],
        samplers: Optional[List[Callable[[mx.array], mx.array]]],
        fallback_sampler: Callable[[mx.array], mx.array],
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ],
        stop_matchers: List[StopSequenceMatcher],
        max_tokens: List[int],
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens

        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.stop_matchers = stop_matchers
        self.max_tokens = max_tokens

        if self.samplers and len(self.samplers) != len(self.uids):
            raise ValueError("Insufficient number of samplers provided")
        if self.logits_processors and len(self.logits_processors) != len(self.uids):
            raise ValueError("Insufficient number of logits_processors provided")

        self._current_tokens = None
        self._current_logprobs = []
        self._decode_steps = 0
        self._next_tokens = inputs
        self._next_logprobs = []
        self._token_context = [TokenBuffer(t) for t in tokens]
        self._num_tokens = [0] * len(self.uids)
        self._matcher_states = [m.make_state() for m in stop_matchers]

        if self.uids:
            self._step()

    def __len__(self):
        return len(self.uids)

    def extend(self, batch):
        """Extend this batch with another generation batch."""
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_matchers.extend(batch.stop_matchers)
        if self._current_tokens is None:
            self._current_tokens = batch._current_tokens
            self._current_logprobs = batch._current_logprobs
        elif batch._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, batch._current_tokens]
            )
            self._current_logprobs.extend(batch._current_logprobs)
        if self._next_tokens is None:
            self._next_tokens = batch._next_tokens
            self._next_logprobs = batch._next_logprobs
        elif batch._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
            self._next_logprobs.extend(batch._next_logprobs)
        self._token_context.extend(batch._token_context)
        self._num_tokens.extend(batch._num_tokens)
        self._matcher_states.extend(batch._matcher_states)

    def _step(self) -> Tuple[List[int], List[mx.array]]:
        """
        Perform a single generation step.

        Returns:
            Tuple of token list and logprobs list.
        """
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        # Forward pass
        if BATCH_UID_HOOK is not None:
            BATCH_UID_HOOK(list(self.uids))
        logits = self.model(inputs[:, None], cache=self.prompt_cache)
        logits = logits[:, -1, :]

        # Logits processors
        token_context = []
        if any(self.logits_processors):
            # Update the token context that will be used by the logits processors
            token_context = [
                tc.update_and_fetch(inputs[i : i + 1])
                for i, tc in enumerate(self._token_context)
            ]
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                for processor in self.logits_processors[e]:
                    sample_logits = processor(token_context[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        # Normalize the logits
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Sample
        if any(self.samplers):
            # Group rows sharing the same sampler object so each unique
            # sampler runs once, vectorized over its rows. Only samplers
            # that declare themselves row-independent (batch_groupable, set
            # by make_sampler) are grouped: an arbitrary callable may be
            # stateful or draw shared randomness, so it keeps the original
            # one-call-per-row contract.
            groups = {}
            order = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                if getattr(sample_sampler, "batch_groupable", False):
                    key = id(sample_sampler)
                else:
                    key = (e,)
                if key not in groups:
                    groups[key] = (sample_sampler, [])
                    order.append(key)
                groups[key][1].append(e)
            if len(groups) == 1:
                ((sample_sampler, rows),) = groups.values()
                sampled = sample_sampler(logprobs)
            else:
                all_samples = [None] * len(self.uids)
                for key in order:
                    sample_sampler, rows = groups[key]
                    if len(rows) == 1:
                        group_sampled = sample_sampler(logprobs[rows[0] : rows[0] + 1])
                    else:
                        group_sampled = sample_sampler(logprobs[mx.array(rows)])
                    for j, e in enumerate(rows):
                        all_samples[e] = group_sampled[j : j + 1]
                sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)

        # Assign the next step to member variables and start computing it
        # asynchronously
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs)
        self._decode_steps += 1
        eval_targets = [self._next_tokens, self._next_logprobs, token_context]
        if self._decode_steps % CACHE_STATE_EVAL_INTERVAL == 0:
            eval_targets.append([c.state for c in self.prompt_cache])
        mx.async_eval(*eval_targets)

        # Eval the current tokens and current logprobs. After that also add
        # them to self.tokens so that it always represents the tokens contained
        # in the KV Cache.
        mx.eval(inputs, self._current_logprobs)
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs):
            sti.append(ti)
        return inputs, self._current_logprobs

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        # Always keep samplers/logits_processors index-aligned with uids. A
        # per-lane list (len == old uids) must be filtered even when every
        # lane is falsy (all-None samplers / all-[] processors); otherwise it
        # stays longer than uids and a later extend appends at the wrong index,
        # silently binding lanes to the wrong sampler/processor. An empty []
        # (no per-lane info) is left untouched.
        if self.samplers:
            self.samplers = [self.samplers[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_matchers = [self.stop_matchers[idx] for idx in keep]

        self._next_tokens = self._next_tokens[keep] if keep else None
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matcher_states = [self._matcher_states[idx] for idx in keep]

    def next(self) -> List[Response]:
        """
        Generate the next batch of tokens.

        Returns:
            List of Response objects for each sequence in the batch.
        """
        if not self.uids:
            return []

        tokens, logprobs = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None

            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            self._matcher_states[i], matched = StopSequenceMatcher.match(
                self._matcher_states[i],
                self.stop_matchers[i]._trie,
                tokens[i],
            )
            if matched:
                finish_reason = "stop"

            if finish_reason is not None:
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=finish_reason,
                        prompt_cache=self.extract_cache(i),
                        all_tokens=self.tokens[i],
                    )
                )
            else:
                keep.append(i)
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=None,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_matchers=[],
        )


@dataclass
class _PausedMTPGenerationLane:
    detached: Any
    initial_output: Optional[Any]
    stop_matcher: StopSequenceMatcher
    matcher_state: Any
    num_tokens: int


def _segment_aware_live_tip_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    if config is None:
        return False
    from .segmented_self_mtp import segmented_self_mtp_enabled

    explicit = config.get("segment_aware_live_tip") if (
        "segment_aware_live_tip" in config
    ) else None
    return segmented_self_mtp_enabled(explicit)


def _segmented_async_qsa_promotion_enabled(
    config: Optional[Mapping[str, Any]],
) -> bool:
    """Return the default-off first-cycle segmented-to-physical policy."""

    if config is None or not _segment_aware_live_tip_enabled(config):
        return False
    if "segment_aware_async_qsa_promotion" in config:
        return bool(config["segment_aware_async_qsa_promotion"])
    return os.environ.get(
        "MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION", "0"
    ).lower() in {"1", "true", "yes", "on"}


def _segmented_async_qsa_min_remaining_tokens(
    config: Optional[Mapping[str, Any]],
) -> int:
    """Return the known-output budget that stays segmented."""

    value = (
        config.get("segment_aware_async_qsa_min_remaining_tokens")
        if config is not None
        and "segment_aware_async_qsa_min_remaining_tokens" in config
        else os.environ.get(
            "MLX_LM_SEGMENTED_ASYNC_QSA_MIN_REMAINING_TOKENS", "16"
        )
    )
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "segment_aware_async_qsa_min_remaining_tokens must be an integer"
        ) from error
    if value < 0:
        raise ValueError(
            "segment_aware_async_qsa_min_remaining_tokens must be non-negative"
        )
    return value


def _segmented_async_qsa_promotion_for_budget(
    config: Optional[Mapping[str, Any]],
    remaining_tokens: int,
    *,
    record: bool = False,
) -> bool:
    """Admit physical promotion only when its destination can be reused."""

    if not _segmented_async_qsa_promotion_enabled(config):
        return False
    remaining_tokens = max(0, int(remaining_tokens))
    cutoff = _segmented_async_qsa_min_remaining_tokens(config)
    admitted = remaining_tokens > cutoff
    if record:
        from .segmented_self_mtp import note_segmented_self_mtp

        note_segmented_self_mtp("async_qsa_budget_checks")
        note_segmented_self_mtp(
            "async_qsa_budget_remaining_tokens_cumulative", remaining_tokens
        )
        note_segmented_self_mtp(
            "async_qsa_budget_cutoff_tokens_cumulative", cutoff
        )
        note_segmented_self_mtp(
            "async_qsa_budget_promotions"
            if admitted
            else "async_qsa_budget_retained_segmented"
        )
    return admitted


def _segmented_async_qsa_prequeue_enabled(
    config: Optional[Mapping[str, Any]],
    remaining_tokens: Optional[int] = None,
) -> bool:
    """Return the shared-prefix N=2 prequeue policy, nested under promotion."""

    enabled = (
        _segmented_async_qsa_promotion_enabled(config)
        if remaining_tokens is None
        else _segmented_async_qsa_promotion_for_budget(
            config, remaining_tokens
        )
    )
    if not enabled:
        return False
    if config is not None and "segment_aware_async_qsa_prequeue" in config:
        return bool(config["segment_aware_async_qsa_prequeue"])
    return os.environ.get(
        "MLX_LM_SEGMENTED_ASYNC_QSA_PREQUEUE", "0"
    ).lower() in {"1", "true", "yes", "on"}


def _prefetch_known_mtp_tail(model, history, prompt, config) -> int:
    """Asynchronously stage file-backed PLE rows for a known MTP tail.

    APC gives lane preparation both the committed prefix and its uncached tail.
    Qwen4 can therefore hash and stage those PLE rows before the target catch-up
    starts. This is a performance hint only: unsupported models, empty tails,
    and submission failures all fall back to the ordinary foreground lookup.
    """
    if not config.get("prefetch_known_tail_ple", False) or not prompt:
        return 0
    from . import round_levers as _lv

    _lv.bump("ple_tail_prefetch_requests")
    prefetch = getattr(model, "ple_prefetch_verify", None)
    if not callable(prefetch):
        _lv.bump("ple_tail_prefetch_declined")
        return 0
    try:
        tables = int(prefetch(list(history), list(prompt)))
    except Exception as error:
        _lv.bump("ple_tail_prefetch_failures")
        logging.warning("Known-tail PLE prefetch declined: %s", error)
        return 0
    if tables <= 0:
        _lv.bump("ple_tail_prefetch_declined")
        return 0
    _lv.bump("ple_tail_prefetch_tables", tables)
    return tables


def _close_segmented_detached(detached: Any, *, release_cache: bool) -> None:
    """Release a detached lane's ledger and, when discarded, COW owner pin."""

    first_error = None
    transaction = getattr(detached, "segment_transaction", None)
    if transaction is not None:
        try:
            transaction.close()
        except BaseException as error:
            first_error = error
        detached.segment_transaction = None
    if release_cache:
        close_target = getattr(detached.caches.target, "close", None)
        if callable(close_target):
            try:
                close_target()
            except BaseException as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error


class MTPGenerationBatch:
    """Scheduler wrapper for Agent A's batched self-MTP transaction."""

    Response = GenerationBatch.Response

    def __init__(
        self,
        model: nn.Module,
        detached_lanes: Sequence[Any],
        initial_outputs: Sequence[Any],
        stop_matchers: Sequence[StopSequenceMatcher],
        *,
        prepared_caches: Optional[Any] = None,
        segmented_live_tip: bool = False,
        async_qsa_promotion: bool = False,
        arm_async_qsa_promotion: bool = True,
        async_qsa_prequeue: Optional[Any] = None,
        mtp_admission: Optional[
            Callable[
                [Sequence[Tuple[int, int, int, bool, float]]],
                Mapping[int, Union[int, str]],
            ]
        ] = None,
    ):
        if len(detached_lanes) != len(initial_outputs):
            raise ValueError("initial_outputs must have one entry per MTP lane")
        if len(detached_lanes) != len(stop_matchers):
            raise ValueError("stop_matchers must have one entry per MTP lane")

        from .hybrid_speculative import (
            BatchedSelfMTPState,
            SegmentedSelfMTPState,
            SelfMTPCachePair,
            attach_prebatched_self_mtp_lanes,
            attach_segmented_self_mtp_lanes,
            attach_self_mtp_lanes,
        )

        self.model = model
        self.segmented_live_tip = bool(segmented_live_tip)
        # This is a cohort policy, not the current cache representation.  A
        # successfully promoted batch is physical but must still compose with
        # later physical joins created under the same configured policy.
        self.async_qsa_promotion = bool(async_qsa_promotion)
        if self.segmented_live_tip and prepared_caches is not None:
            raise ValueError("segmented B1 state cannot accept a physical B2 cache")
        if self.segmented_live_tip:
            self.state = (
                attach_segmented_self_mtp_lanes(
                    model, None, list(detached_lanes)
                )
                if detached_lanes
                else SegmentedSelfMTPState([], [], [], 0)
            )
        elif prepared_caches is not None:
            if not isinstance(prepared_caches, SelfMTPCachePair):
                raise TypeError("prepared_caches must be a SelfMTPCachePair")
            self.state = attach_prebatched_self_mtp_lanes(
                model, detached_lanes, prepared_caches
            )
        elif detached_lanes:
            self.state = attach_self_mtp_lanes(model, None, list(detached_lanes))
        else:
            self.state = BatchedSelfMTPState([], SelfMTPCachePair([], []), 0)
        self.stop_matchers = list(stop_matchers)
        self._matcher_states = [m.make_state() for m in stop_matchers]
        self._num_tokens = [0] * len(detached_lanes)
        self._initial_outputs = list(initial_outputs)
        self._paused: Dict[int, _PausedMTPGenerationLane] = {}
        self._plain_ready: List[_PausedMTPGenerationLane] = []
        self.mtp_admission = mtp_admission
        self._async_qsa_ticket = None
        self._async_qsa_receipt = None
        self._async_qsa_receipts_by_uid = {}
        self._async_qsa_pending = (
            self.async_qsa_promotion and self.segmented_live_tip
        )
        if async_qsa_prequeue is not None:
            from .segmented_physical_promotion import (
                SegmentedPhysicalPromotionDeclined,
            )
            from .segmented_self_mtp import note_segmented_self_mtp

            if not self._async_qsa_pending:
                async_qsa_prequeue.cancel_and_drain()
                raise ValueError(
                    "async QSA prequeue requires segmented async promotion"
                )
            try:
                self._async_qsa_ticket = async_qsa_prequeue.bind(
                    self.state, note=note_segmented_self_mtp
                )
            except SegmentedPhysicalPromotionDeclined as error:
                async_qsa_prequeue.cancel_and_drain()
                note_segmented_self_mtp("async_qsa_prequeue_declined")
                logging.info("Async QSA prequeue declined at bind: %s", error)
                self._arm_async_qsa_promotion()
            else:
                note_segmented_self_mtp("async_qsa_prequeue_bound")
        elif arm_async_qsa_promotion:
            self._arm_async_qsa_promotion()

    def _arm_async_qsa_promotion(self) -> None:
        """Queue immutable QSA-base formation once, before the first cycle."""

        if (
            not self._async_qsa_pending
            or self._async_qsa_ticket is not None
            or not self.segmented_live_tip
            or not self.state.lanes
        ):
            return
        from .segmented_physical_promotion import (
            SegmentedPhysicalPromotionDeclined,
            begin_segmented_physical_promotion,
        )
        from .segmented_self_mtp import note_segmented_self_mtp

        note_segmented_self_mtp("async_qsa_promotion_requests")
        reserve_tail = max(int(lane.num_draft) + 1 for lane in self.state.lanes)
        try:
            self._async_qsa_ticket = begin_segmented_physical_promotion(
                self.state,
                reserve_tail=reserve_tail,
                stream=mx.new_stream(mx.gpu),
                note=note_segmented_self_mtp,
            )
        except SegmentedPhysicalPromotionDeclined as error:
            self._async_qsa_pending = False
            note_segmented_self_mtp("async_qsa_promotion_declined")
            logging.warning("Async QSA promotion declined at queue: %s", error)
        except Exception as error:
            self._async_qsa_pending = False
            note_segmented_self_mtp("async_qsa_promotion_failures")
            logging.warning("Async QSA promotion failed at queue: %s", error)
        else:
            note_segmented_self_mtp("async_qsa_promotion_queued")

    def _decline_async_qsa_promotion(self, reason: str) -> None:
        if self._async_qsa_ticket is None and not self._async_qsa_pending:
            return
        from .segmented_self_mtp import note_segmented_self_mtp

        ticket = self._async_qsa_ticket
        if ticket is not None:
            try:
                drain = getattr(ticket, "cancel_and_drain", None)
                if callable(drain):
                    drain()
            except BaseException as error:
                note_segmented_self_mtp("async_qsa_promotion_failures")
                logging.warning("Async QSA promotion drain failed: %s", error)
            else:
                if getattr(ticket, "stream", None) is not None:
                    note_segmented_self_mtp("device_synchronizations")
        self._async_qsa_ticket = None
        self._async_qsa_pending = False
        note_segmented_self_mtp("async_qsa_promotion_declined")
        logging.info("Async QSA promotion retained segmented state: %s", reason)

    def __len__(self):
        return len(self.state.lanes)

    @property
    def uids(self):
        return [lane.uid for lane in self.state.lanes]

    @property
    def prompt_cache(self):
        if self.segmented_live_tip:
            return [
                cache
                for pair in self.state.row_caches
                for cache in pair.target
            ]
        return self.state.caches.target

    @property
    def cache_nbytes(self):
        if self.segmented_live_tip:
            total = sum(
                cache.nbytes
                for pair in self.state.row_caches
                for cache in pair.target + pair.draft
            )
        else:
            total = sum(
                cache.nbytes
                for cache in self.state.caches.target + self.state.caches.draft
            )
        total += sum(
            cache.nbytes
            for paused in self._paused.values()
            for cache in paused.detached.caches.target + paused.detached.caches.draft
        )
        if self.segmented_live_tip:
            view = getattr(self.state, "_segmented_caches", None)
            if view is not None:
                # Segmented QSA adapters report zero; recurrent adapters own
                # real joined B2 arrays in addition to the persistent B1 rows.
                total += sum(
                    cache.nbytes for cache in view.target + view.draft
                )
        if self._async_qsa_ticket is not None:
            # Count the complete lazy physical candidate immediately so
            # admission sees the peak B1+B2 overlap, not only resident pages.
            total += int(self._async_qsa_ticket.reserved_bytes)
        return total

    @property
    def tokens(self):
        return [self._prefix_tokens(lane) for lane in self.state.lanes]

    @property
    def max_tokens(self):
        return [lane.max_tokens for lane in self.state.lanes]

    @staticmethod
    def _prefix_tokens(lane):
        values = lane.token_prefix.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(token) for token in values]

    def mtp_cycle_state(self):
        if self.segmented_live_tip:
            overlap = 0
            view = getattr(self.state, "_segmented_caches", None)
            if view is not None:
                overlap += sum(
                    cache.nbytes for cache in view.target + view.draft
                )
            if self._async_qsa_ticket is not None:
                overlap += int(self._async_qsa_ticket.reserved_bytes)
            overlap_per_lane = overlap / max(len(self.state.lanes), 1)
            rows = [
                (
                    lane.uid,
                    len(self._prefix_tokens(lane)) + 1,
                    lane.num_draft,
                    True,
                    (sum(cache.nbytes for cache in pair.target + pair.draft)
                    + overlap_per_lane)
                    / float(1 << 30),
                )
                for lane, pair in zip(self.state.lanes, self.state.row_caches)
            ]
        else:
            active_bytes = sum(
                cache.nbytes
                for cache in self.state.caches.target + self.state.caches.draft
            )
            active_cache_gib = (
                active_bytes / max(len(self.state.lanes), 1) / float(1 << 30)
            )
            rows = [
                (
                    lane.uid,
                    len(self._prefix_tokens(lane)) + 1,
                    lane.num_draft,
                    True,
                    active_cache_gib,
                )
                for lane in self.state.lanes
            ]
        rows.extend(
            (
                uid,
                len(self._prefix_tokens(paused.detached.lane)) + 1,
                paused.detached.lane.num_draft,
                False,
                sum(
                    cache.nbytes
                    for cache in (
                        paused.detached.caches.target
                        + paused.detached.caches.draft
                    )
                )
                / float(1 << 30),
            )
            for uid, paused in self._paused.items()
        )
        return rows

    def set_num_draft(self, depths: Union[int, Mapping[int, int]]):
        if self.state.proposal_open:
            raise RuntimeError("cannot change MTP depth while a proposal is open")
        if isinstance(depths, int):
            depths = {uid: depths for uid in self.uids}
        unknown = set(depths) - set(self.uids) - set(self._paused)
        if unknown:
            raise KeyError(f"unknown MTP lane uids: {sorted(unknown)}")
        desired = {
            int(depths.get(lane.uid, lane.num_draft)) for lane in self.state.lanes
        }
        if len(desired) > 1:
            raise ValueError("adaptive per-lane self-MTP depth is excluded")
        for lane in self.state.lanes:
            if lane.uid in depths:
                depth = int(depths[lane.uid])
                if depth < 1:
                    raise ValueError("MTP draft depth must be positive")
                lane.num_draft = depth
        for uid, paused in self._paused.items():
            if uid in depths:
                depth = int(depths[uid])
                if depth < 1:
                    raise ValueError("MTP draft depth must be positive")
                paused.detached.lane.num_draft = depth

    def _detach_packages(self, indices: Sequence[int]):
        from .hybrid_speculative import detach_self_mtp_lanes

        indices = sorted(set(int(i) for i in indices))
        if not indices:
            return []
        if self._async_qsa_ticket is not None:
            self._decline_async_qsa_promotion("membership changed before first cycle")
        old_matchers = self.stop_matchers
        old_states = self._matcher_states
        old_counts = self._num_tokens
        old_initial = self._initial_outputs
        self.state, detached = detach_self_mtp_lanes(self.model, self.state, indices)
        packages = [
            _PausedMTPGenerationLane(
                lane,
                old_initial[idx],
                old_matchers[idx],
                old_states[idx],
                old_counts[idx],
            )
            for idx, lane in zip(indices, detached)
        ]
        keep = [i for i in range(len(old_matchers)) if i not in set(indices)]
        self.stop_matchers = [old_matchers[i] for i in keep]
        self._matcher_states = [old_states[i] for i in keep]
        self._num_tokens = [old_counts[i] for i in keep]
        self._initial_outputs = [old_initial[i] for i in keep]
        return packages

    def _attach_packages(self, packages: Sequence[_PausedMTPGenerationLane]):
        if not packages:
            return
        from .hybrid_speculative import (
            attach_segmented_self_mtp_lanes,
            attach_self_mtp_lanes,
        )

        if self.state.lanes:
            depths = {lane.num_draft for lane in self.state.lanes}
            if len(depths) != 1:
                raise RuntimeError("active self-MTP lanes have mixed draft depths")
            depth = depths.pop()
            for package in packages:
                package.detached.lane.num_draft = depth
        attach = (
            attach_segmented_self_mtp_lanes
            if self.segmented_live_tip
            else attach_self_mtp_lanes
        )
        self.state = attach(
            self.model, self.state, [package.detached for package in packages]
        )
        self.stop_matchers.extend(package.stop_matcher for package in packages)
        self._matcher_states.extend(package.matcher_state for package in packages)
        self._num_tokens.extend(package.num_tokens for package in packages)
        self._initial_outputs.extend(package.initial_output for package in packages)
        self._arm_async_qsa_promotion()

    def _apply_admission(self):
        if self.mtp_admission is None:
            return
        decisions = dict(self.mtp_admission(tuple(self.mtp_cycle_state())) or {})
        drop = [
            i
            for i, uid in enumerate(self.uids)
            if decisions.get(uid) in ("queue", "plain")
        ]
        for package in self._detach_packages(drop):
            mode = decisions.get(package.detached.lane.uid)
            if mode == "plain":
                self._plain_ready.append(package)
            else:
                self._paused[package.detached.lane.uid] = package

        self.set_num_draft(
            {uid: value for uid, value in decisions.items() if isinstance(value, int)}
        )

        joining = []
        for uid, value in decisions.items():
            if isinstance(value, int) and uid in self._paused:
                joining.append(self._paused.pop(uid))
        self._attach_packages(joining)

    def take_plain_fallbacks(self):
        ready, self._plain_ready = self._plain_ready, []
        self._normalize_empty_segmented_admission()
        return ready

    def _normalize_empty_segmented_admission(self):
        """Restore configured segmented admission at an ownership-free seam."""

        if (
            not self.async_qsa_promotion
            or self.state.lanes
            or self._paused
            or self._plain_ready
        ):
            return
        from .hybrid_speculative import SegmentedSelfMTPState

        epoch = int(getattr(self.state, "membership_epoch", 0))
        self.state = SegmentedSelfMTPState([], [], [], epoch)
        self.segmented_live_tip = True
        self._async_qsa_pending = True
        self._async_qsa_receipt = None
        self._async_qsa_receipts_by_uid.clear()

    def close(self):
        first_error = None
        if self._async_qsa_ticket is not None:
            self._decline_async_qsa_promotion("batch closed before promotion")
        else:
            # An ownership-free batch is normalized back to segmented
            # admission so a later join can arm promotion.  Closing that idle
            # shell is not a declined promotion attempt and must not inflate
            # the production counter.
            self._async_qsa_pending = False
        if self.segmented_live_tip:
            from .hybrid_speculative import close_segmented_self_mtp_state

            try:
                close_segmented_self_mtp_state(self.state)
            except BaseException as error:
                first_error = error
        for package in [*self._paused.values(), *self._plain_ready]:
            try:
                _close_segmented_detached(package.detached, release_cache=True)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._paused.clear()
        self._plain_ready.clear()
        if first_error is not None:
            raise first_error

    def extend(self, batch):
        if not isinstance(batch, MTPGenerationBatch):
            raise TypeError("MTPGenerationBatch can extend only another MTP batch")
        if self.segmented_live_tip != batch.segmented_live_tip:
            raise ValueError("cannot mix segmented and physical self-MTP batches")
        if self.async_qsa_promotion != batch.async_qsa_promotion:
            if not self.segmented_live_tip:
                raise ValueError(
                    "cannot mix async-promotion and control self-MTP batches"
                )
            if self._async_qsa_ticket is not None:
                self._decline_async_qsa_promotion(
                    "short-output cohort joined before promotion"
                )
            if batch._async_qsa_ticket is not None:
                batch._decline_async_qsa_promotion(
                    "short-output cohort already owns admission"
                )
            # The shortest known output budget owns the cohort decision. A
            # later join can safely suppress an unpublished destination, but
            # it cannot make a short row pay for a cache it will not reuse.
            self.async_qsa_promotion = (
                self.async_qsa_promotion and batch.async_qsa_promotion
            )
            self._async_qsa_pending = (
                self.async_qsa_promotion and self.segmented_live_tip
            )
        packages = batch._detach_packages(range(len(batch)))
        packages.extend(batch._paused.values())
        batch._paused.clear()
        if self.mtp_admission is None:
            self._attach_packages(packages)
            return
        for package in packages:
            uid = package.detached.lane.uid
            if uid in self._paused or uid in self.uids:
                raise ValueError(f"duplicate self-MTP lane uid {uid}")
            self._paused[uid] = package
        # Joining cache rows participate in the same cycle-boundary decision
        # before merge/extend allocates the wider verify batch.
        self._apply_admission()

    def extract_cache(self, idx: int) -> List[Any]:
        if self.state.proposal_open:
            raise RuntimeError("cannot extract an MTP cache during a proposal")
        if not (0 <= idx < len(self)):
            raise IndexError(idx)
        packages = self._detach_packages(range(len(self)))
        result = copy.deepcopy(packages[idx].detached.caches.target)
        self._attach_packages(packages)
        return result

    def filter(self, keep: List[int]):
        keep = sorted(set(keep))
        drop = [i for i in range(len(self)) if i not in set(keep)]
        dropped_uids = [self.uids[i] for i in drop]
        for package in self._detach_packages(drop):
            _close_segmented_detached(package.detached, release_cache=True)
        for uid in dropped_uids:
            self._async_qsa_receipts_by_uid.pop(uid, None)
        self._normalize_empty_segmented_admission()

    def extract_uid(self, uid: int):
        if uid in self.uids:
            idx = self.uids.index(uid)
            return self.extract_cache(idx), self.tokens[idx]
        if uid in self._paused:
            lane = self._paused[uid].detached
            return copy.deepcopy(lane.caches.target), self._prefix_tokens(lane.lane)
        raise KeyError(uid)

    def remove_uids(self, uids):
        requested = set(uids)
        drop = [i for i, uid in enumerate(self.uids) if uid in requested]
        packages = self._detach_packages(drop)
        for package in packages:
            _close_segmented_detached(package.detached, release_cache=True)
        for uid in requested:
            package = self._paused.pop(uid, None)
            if package is not None:
                _close_segmented_detached(package.detached, release_cache=True)
        for uid in requested:
            self._async_qsa_receipts_by_uid.pop(uid, None)
        self._normalize_empty_segmented_admission()

    @staticmethod
    def _finish_reason(
        token: int,
        count: int,
        maximum: int,
        matcher_state,
        matcher: StopSequenceMatcher,
    ):
        reason = "length" if count >= maximum else None
        matcher_state, matched = StopSequenceMatcher.match(
            matcher_state, matcher._trie, token
        )
        if matched:
            reason = "stop"
        return matcher_state, reason

    def _complete_responses(self, terminal_indices, last_response_by_index):
        packages = self._detach_packages(terminal_indices)
        for idx, package in zip(sorted(terminal_indices), packages):
            response = last_response_by_index[idx]
            lane = package.detached.lane
            response.prompt_cache = package.detached.caches.target
            response.all_tokens = self._prefix_tokens(lane)
            response.mtp_state = (package.detached.caches.draft, lane.seed_h)
            response.lane_rng = lane.rng
            response.rng_draws = lane.rng.draws if lane.rng is not None else 0
            stats = dict(vars(lane.stats))
            stats["total_emitted"] = int(lane.stats.total_emitted)
            stats["draft_acceptance"] = (
                float(lane.stats.draft_accepted)
                / max(int(lane.stats.draft_proposed), 1)
            )
            response.mtp_receipt = {
                "route": (
                    "segmented_b1_self_mtp"
                    if self.segmented_live_tip
                    else "continuous_batched_self_mtp"
                ),
                "num_draft": int(lane.num_draft),
                "accept_rule": str(lane.accept_rule),
                "sampling_temperature": float(lane.sampling_temp),
                "stats": stats,
                "async_qsa_promotion": self._async_qsa_receipts_by_uid.pop(
                    lane.uid, None
                ),
            }
            _close_segmented_detached(package.detached, release_cache=False)
        self._normalize_empty_segmented_admission()

    def _emit_initial(self):
        responses = []
        terminal = []
        last = {}
        for i, output in enumerate(self._initial_outputs):
            if output is None:
                continue
            self._initial_outputs[i] = None
            self._num_tokens[i] += 1
            self._matcher_states[i], reason = self._finish_reason(
                output.token,
                self._num_tokens[i],
                self.max_tokens[i],
                self._matcher_states[i],
                self.stop_matchers[i],
            )
            response = self.Response(
                uid=self.uids[i],
                token=output.token,
                logprobs=output.logprobs,
                finish_reason=reason,
                prompt_cache=None,
                all_tokens=None,
                from_draft=output.from_draft,
            )
            responses.append(response)
            if reason is not None:
                terminal.append(i)
                last[i] = response
        if terminal:
            self._complete_responses(terminal, last)
        return responses

    def next(self) -> List[Response]:
        if any(output is not None for output in self._initial_outputs):
            return self._emit_initial()

        self._apply_admission()
        if not self.state.lanes:
            return []

        from .hybrid_speculative import (
            abort_batched_self_mtp,
            commit_batched_self_mtp,
            propose_batched_self_mtp,
        )

        proposal = propose_batched_self_mtp(self.model, self.state)
        try:
            emitted_counts = []
            terminal = []
            responses = []
            last = {}
            for i, outputs in enumerate(proposal.outputs):
                emitted = 0
                is_terminal = False
                for output in outputs:
                    emitted += 1
                    self._num_tokens[i] += 1
                    self._matcher_states[i], reason = self._finish_reason(
                        output.token,
                        self._num_tokens[i],
                        self.max_tokens[i],
                        self._matcher_states[i],
                        self.stop_matchers[i],
                    )
                    response = self.Response(
                        uid=self.uids[i],
                        token=output.token,
                        logprobs=output.logprobs,
                        finish_reason=reason,
                        prompt_cache=None,
                        all_tokens=None,
                        from_draft=output.from_draft,
                    )
                    responses.append(response)
                    last[i] = response
                    if reason is not None:
                        is_terminal = True
                        break
                emitted_counts.append(emitted)
                terminal.append(is_terminal)

            commit_batched_self_mtp(
                self.state,
                proposal,
                emitted_counts=emitted_counts,
                terminal=terminal,
            )
            ticket = self._async_qsa_ticket
            if ticket is not None:
                if any(terminal):
                    self._decline_async_qsa_promotion(
                        "a lane terminated in the first segmented cycle"
                    )
                else:
                    from dataclasses import asdict

                    from .segmented_physical_promotion import (
                        SegmentedPhysicalPromotionDeclined,
                    )
                    from .segmented_self_mtp import note_segmented_self_mtp

                    try:
                        self.state, receipt = ticket.finish()
                    except SegmentedPhysicalPromotionDeclined as error:
                        self._decline_async_qsa_promotion(str(error))
                    except BaseException:
                        note_segmented_self_mtp("async_qsa_promotion_failures")
                        raise
                    else:
                        self._async_qsa_ticket = None
                        self._async_qsa_pending = False
                        self.segmented_live_tip = False
                        self._async_qsa_receipt = asdict(receipt)
                        self._async_qsa_receipts_by_uid.update(
                            (lane.uid, dict(self._async_qsa_receipt))
                            for lane in self.state.lanes
                        )
                        note_segmented_self_mtp("async_qsa_promotion_engaged")
                        note_segmented_self_mtp(
                            "async_qsa_promotion_reserved_bytes",
                            receipt.reserved_bytes,
                        )
                        note_segmented_self_mtp(
                            "async_qsa_promotion_patched_bytes",
                            receipt.patched_bytes,
                        )
                        note_segmented_self_mtp(
                            "async_qsa_promotion_wait_ns", receipt.stream_wait_ns
                        )
                        if getattr(ticket, "stream", None) is not None:
                            note_segmented_self_mtp("device_synchronizations")
        except BaseException as error:
            if self.state.proposal_open:
                abort_batched_self_mtp(self.state, proposal, cause=error)
            raise
        terminal_indices = [i for i, value in enumerate(terminal) if value]
        if terminal_indices:
            self._complete_responses(terminal_indices, last)
        return responses

    @classmethod
    def empty(
        cls,
        model,
        *,
        mtp_admission=None,
        segmented_live_tip: bool = False,
        async_qsa_promotion: bool = False,
    ):
        return cls(
            model,
            [],
            [],
            [],
            mtp_admission=mtp_admission,
            segmented_live_tip=segmented_live_tip,
            async_qsa_promotion=async_qsa_promotion,
        )


class BatchGenerator:
    """
    A batch generator implements continuous batching.

    This class provides automatic management of prompt processing and generation
    batches, handling the transition between the two.

    It also allows for segmented prompt processing which guarantees that the
    generator will stop at these boundaries when processing an input.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        prefill_batch_window: Optional[int] = None,
        max_kv_size: Optional[int] = None,
        kv_budget_bytes: Optional[int] = None,
        kv_cost: Optional[Tuple[float, float, Optional[int]]] = None,
        state_budget: Optional[StateBudget] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        # Deliberately NOT Optional[None]->DEFAULT_QUANTIZED_KV_START like
        # generate_step/speculative_generate_step: 0 is the only value this
        # class supports (see the NotImplementedError below), so there is no
        # separate "omitted" case to resolve against the CLI/server default.
        quantized_kv_start: int = 0,
        stream=None,
        prompt_trim_rollback_tokens: int = 0,
        self_mtp: Optional[dict] = None,
        mtp_admission: Optional[
            Callable[
                [Sequence[Tuple[int, int, int, bool, float]]],
                Mapping[int, Union[int, str]],
            ]
        ] = None,
    ):
        if kv_bits is not None and quantized_kv_start != 0:
            # Validated before any state is set, so a rejected config never
            # leaves a partially-constructed instance behind (__del__ calls
            # close(), which needs self._old_wired_limit to already exist).
            # Per-job caches are created once, empty (offset=0), at insertion
            # time via _make_new_cache() — maybe_quantize_kv_cache's
            # offset-threshold gate would never trigger later for a nonzero
            # quantized_kv_start, since there's no per-step re-check in the
            # batching path. Only immediate quantization is supported here.
            raise NotImplementedError(
                "BatchGenerator only supports quantized_kv_start=0 with kv_bits "
                "set — delayed/threshold quantization is not implemented for "
                "the continuous-batching path."
            )
        self.model = model
        self.self_mtp = dict(self_mtp) if self_mtp is not None else None
        self.mtp_admission = mtp_admission
        if self.self_mtp is not None:
            if not self.self_mtp.get("persistent", True):
                raise ValueError("batched self-MTP requires persistent_mtp=True")
            if self.self_mtp.get("window_size") is not None:
                raise ValueError("windowed MTP is not batchable")
            if self.self_mtp.get("rate_gate", False):
                raise ValueError("runtime rate gating is not batchable")
            if self.self_mtp.get("speculation_router") is not None:
                raise ValueError("adaptive per-lane MTP depth is not batchable")
            if max_kv_size is not None:
                raise ValueError("bounded (windowed) KV caches are not MTP batchable")
            if kv_bits is not None and not self.self_mtp.get("allow_quantized_kv"):
                # Quantized KV is opt-in for self-MTP (allow_quantized_kv). The
                # batched transaction is bit-exact on a quantized target cache;
                # this refusal is the policy gate, not a capability limit.
                raise ValueError(
                    "quantized KV caches are not MTP batchable unless "
                    "allow_quantized_kv is set in the self-MTP config"
                )
        self.max_tokens = max_tokens
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = logits_processors or []
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.prefill_batch_window = (
            1 if prefill_batch_window is None else prefill_batch_window
        )
        if self.prefill_batch_window < 1:
            raise ValueError("prefill_batch_window must be positive")
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.max_kv_size = max_kv_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.prompt_trim_rollback_tokens = max(0, int(prompt_trim_rollback_tokens))
        if state_budget is not None and (
            kv_budget_bytes is not None or kv_cost is not None
        ):
            raise ValueError(
                "state_budget cannot be combined with kv_budget_bytes/kv_cost"
            )
        if (
            state_budget is not None
            and type(state_budget.project) is not LinearStateCost
        ):
            raise ValueError(
                "BatchGenerator state_budget requires LinearStateCost; "
                "peak-only iterative policies such as StepStateCost must be "
                "driven by their model-native scheduler with actual resident "
                "request state"
            )
        if (
            state_budget is not None
            and state_budget.project.bytes_per_unit > 0
            and state_budget.project.allocation_step_units is None
        ):
            raise ValueError(
                "BatchGenerator budgeting over growing state requires "
                "allocation_step_units: shared batch caches allocate in "
                "steps at the cohort-max width, and unrounded admission "
                "can exceed the budget. Fixed-only state may omit the step."
            )
        if kv_budget_bytes is not None:
            if kv_cost is None:
                raise ValueError(
                    "kv_budget_bytes requires kv_cost=(fixed_bytes_per_row, "
                    "bytes_per_token) measured for this model"
                )
            if len(kv_cost) < 3 or (kv_cost[1] > 0 and kv_cost[2] is None):
                raise ValueError(
                    "kv_cost must be (fixed_bytes, bytes_per_token, "
                    "allocation_step_units) with a validated step whenever "
                    "bytes_per_token > 0: unrounded per-token cost "
                    "cannot budget shared stepped batch caches safely"
                )
            fixed, per_token = kv_cost[0], kv_cost[1]
            if not (
                math.isfinite(kv_budget_bytes)
                and math.isfinite(fixed)
                and math.isfinite(per_token)
            ):
                raise ValueError("kv_budget_bytes and kv_cost must be finite")
            if kv_budget_bytes <= 0 or fixed < 0 or per_token < 0:
                raise ValueError("kv_budget_bytes must be positive and kv_cost >= 0")
        self.kv_budget_bytes = kv_budget_bytes
        self.kv_cost = kv_cost
        self.state_budget = state_budget
        if kv_budget_bytes is not None:
            step = kv_cost[2] if len(kv_cost) > 2 else None
            self.state_budget = StateBudget(
                kv_budget_bytes,
                LinearStateCost(
                    fixed,
                    per_token,
                    max_units=max_kv_size,
                    allocation_step_units=step,
                ),
            )

        self._stream = stream or generation_stream

        self._default_stop_matcher = StopSequenceMatcher(
            stop_tokens if stop_tokens else None,
        )
        self._uid_count = 0
        self._prompt_batch = PromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=prefill_step_size,
            prompt_trim_rollback_tokens=self.prompt_trim_rollback_tokens,
        )
        if self.self_mtp is None:
            self._generation_batch = GenerationBatch.empty(self.model, self.sampler)
        else:
            self._generation_batch = MTPGenerationBatch.empty(
                self.model,
                mtp_admission=self.mtp_admission,
                segmented_live_tip=_segment_aware_live_tip_enabled(self.self_mtp),
                async_qsa_promotion=_segmented_async_qsa_promotion_enabled(
                    self.self_mtp
                ),
            )
        self._plain_fallback_batch = GenerationBatch.empty(self.model, self.sampler)
        self._unprocessed_sequences = deque()
        self._currently_processing = []
        self._mtp_states = {}
        self._mtp_lane_rngs = {}
        self._mtp_configs = {}
        # Exact P-1 target/draft snapshots captured before a joining lane's
        # final prompt token is consumed.  The server pops each snapshot at
        # end-of-prompt and transfers ownership to APC.
        self._mtp_prompt_boundaries = {}

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        device_info = mx.device_info()
        if (
            mx.metal.is_available()
            and "max_recommended_working_set_size" in device_info
        ):
            self._old_wired_limit = mx.set_wired_limit(
                device_info["max_recommended_working_set_size"]
            )
        else:
            self._old_wired_limit = None

    @property
    def stream(self):
        return self._stream

    def close(self):
        # getattr guards against __del__ firing on an instance whose __init__
        # raised before this attribute was ever set (e.g. an invalid kv_bits
        # config) — a bare self._old_wired_limit would AttributeError there.
        if getattr(self, "_old_wired_limit", None) is not None:
            mx.synchronize(self._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None
        generation_batch = getattr(self, "_generation_batch", None)
        if isinstance(generation_batch, MTPGenerationBatch):
            generation_batch.close()
        # These snapshots have not yet been transferred to APC. Releasing the
        # last references here prevents an interrupted prompt from pinning a
        # full target+draft prefix until GC happens to collect the generator.
        getattr(self, "_mtp_prompt_boundaries", {}).clear()

    def __del__(self):
        self.close()

    @contextlib.contextmanager
    def stats(self, stats=None):
        stats = stats or BatchStats()
        if self.kv_bits is not None:
            stats.effective_quantized_kv_start = self.quantized_kv_start
        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        tic = time.perf_counter()
        try:
            yield stats
        finally:
            toc = time.perf_counter()
            total_time = toc - tic
            gen_time = total_time - self._prompt_time_counter
            stats.prompt_tokens += self._prompt_tokens_counter
            stats.prompt_time += self._prompt_time_counter
            stats.prompt_tps = stats.prompt_tokens / stats.prompt_time
            stats.generation_tokens += self._gen_tokens_counter
            stats.generation_time += gen_time
            stats.generation_tps = stats.generation_tokens / stats.generation_time
            stats.peak_memory = max(stats.peak_memory, mx.get_peak_memory() / 1e9)

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        mtp_states: Optional[List[Optional[Tuple[List[Any], mx.array]]]] = None,
        lane_rngs: Optional[List[Optional[LaneRNG]]] = None,
        self_mtp_configs: Optional[List[dict]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
            mtp_states,
            lane_rngs,
            self_mtp_configs,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        mtp_states: Optional[List[Optional[Tuple[List[Any], mx.array]]]] = None,
        lane_rngs: Optional[List[Optional[LaneRNG]]] = None,
        self_mtp_configs: Optional[List[dict]] = None,
    ):
        uids = []

        max_tokens = max_tokens or [self.max_tokens] * len(segments)
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = samplers or [None] * len(segments)
        logits_processors = logits_processors or (
            [self.logits_processors] * len(segments)
        )
        stop_matchers = stop_matchers or ([self._default_stop_matcher] * len(segments))
        mtp_states = mtp_states or [None] * len(segments)
        lane_rngs = lane_rngs or [None] * len(segments)
        self_mtp_configs = self_mtp_configs or [{} for _ in segments]
        for name, values in (
            ("mtp_states", mtp_states),
            ("lane_rngs", lane_rngs),
            ("self_mtp_configs", self_mtp_configs),
        ):
            if len(values) != len(segments):
                raise ValueError(f"{name} must have one entry per sequence")

        caches = caches or [None] * len(segments)
        for i in range(len(segments)):
            if caches[i] is None:
                caches[i] = self._make_new_cache()
            elif self.kv_bits is not None:
                # Externally supplied (snapshot-restored / transplanted)
                # caches must honor the generator's kv-quant config: the lane
                # would otherwise silently run unquantized and the mixed
                # fp/quantized cohort breaks the class-specific merges.
                maybe_quantize_kv_cache(
                    caches[i],
                    self.quantized_kv_start,
                    self.kv_group_size,
                    self.kv_bits,
                )
            if self.kv_bits is not None:
                # Both lanes, not just supplied caches: validate every nested
                # leaf after quantization so custom cache classes fail here,
                # rather than later inside _merge_caches.
                for c in caches[i]:
                    # CacheList.merge merges leaf-wise, so every nested leaf
                    # must be mergeable too — check leaves, not the wrapper.
                    leaves = c.caches if isinstance(c, CacheList) else (c,)
                    for leaf in leaves:
                        if not hasattr(leaf, "merge"):
                            raise ValueError(
                                f"kv_bits is set but the cache for this job "
                                f"quantizes to {type(leaf).__name__}, which "
                                "does not support batching. Batched kv-quant "
                                "currently requires rotating caches (set "
                                "max_kv_size) or an unquantized generator."
                            )

        for seq, m, c, at, s, lp, sm, mtp_state, lane_rng, mtp_config in zip(
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
            mtp_states,
            lane_rngs,
            self_mtp_configs,
        ):
            seq = list(seq)
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            self._unprocessed_sequences.append(
                (self._uid_count, seq, m, c, at, s, lp, sm)
            )
            if self.self_mtp is not None:
                # The plain batch path never consumes or clears these maps;
                # populating them per request would leak entries for the
                # lifetime of a plain generator.
                self._mtp_states[self._uid_count] = mtp_state
                self._mtp_lane_rngs[self._uid_count] = lane_rng
                self._mtp_configs[self._uid_count] = dict(mtp_config)
            uids.append(self._uid_count)
            self._uid_count += 1

        return uids

    def _make_new_cache(self):
        if self.max_kv_size is None:
            new_cache = cache.make_prompt_cache(self.model)
        else:
            new_cache = [
                (
                    RotatingKVCache(max_size=self.max_kv_size)
                    if isinstance(ci, KVCache)
                    else ci
                )
                for ci in cache.make_prompt_cache(self.model)
            ]

        if self.kv_bits is not None:
            # quantized_kv_start is always 0 here (enforced in __init__), so this
            # quantizes every layer immediately — the cache is empty (offset=0),
            # so there's no precision to lose. All subsequent update_and_fetch
            # calls go through the quantized cache classes from this point on;
            # no further re-quantization is needed anywhere else.
            maybe_quantize_kv_cache(
                new_cache, self.quantized_kv_start, self.kv_group_size, self.kv_bits
            )
        return new_cache

    @staticmethod
    def _validate_mtp_config(config):
        if not config.get("persistent", True):
            raise ValueError("batched self-MTP requires persistent_mtp=True")
        if config.get("window_size") is not None:
            raise ValueError("windowed MTP is not batchable")
        if config.get("rate_gate", False):
            raise ValueError("runtime rate gating is not batchable")
        if config.get("speculation_router") is not None:
            raise ValueError("adaptive per-lane MTP depth is not batchable")
        if config.get("xtc_probability", 0.0) > 0.0:
            raise ValueError("stochastic XTC is not batchable with self-MTP")

    def _admit_mtp_joining(self, n: int) -> int:
        """Budget joining lanes' caches BEFORE ``_make_mtp_batch`` allocates.

        The admission callback sees the live rows plus the first ``n`` queued
        sequences (context, configured depth, retained cache bytes) and the
        gate admits the longest queue prefix whose lanes were approved (an MTP
        depth or plain; ``queue`` stops the prefix). A lane approved only as
        plain is still prepared here — the merge-boundary admission pass then
        migrates it — while a queued lane allocates nothing this cycle.
        """
        if n <= 0 or self.mtp_admission is None:
            return n
        rows = list(self._generation_batch.mtp_cycle_state())
        joining = []
        for sequence in list(self._unprocessed_sequences)[:n]:
            uid, segments, _maximum, prompt_cache, history = sequence[:5]
            config = dict(self.self_mtp or {})
            config.update(self._mtp_configs.get(uid, {}))
            prompt_len = sum(len(segment) for segment in segments)
            context = len(history) + prompt_len
            target_bytes = 0
            covered = 0
            for leaf in prompt_cache:
                target_bytes += int(getattr(leaf, "nbytes", 0))
                covered = max(covered, int(getattr(leaf, "offset", 0)))
            # Preparation prefills the uncached suffix into the target cache,
            # so budget the post-prepare size, not the restored size.
            if covered > 0:
                target_bytes = int(target_bytes * (max(context, covered) / covered))
            # A restored APC sidecar's draft cache is retained alongside the
            # target cache and re-allocated during preparation; budget its
            # actual bytes BEFORE that allocation, not only at the later
            # merge-boundary check. A lane without a measurable sidecar still
            # allocates a fresh single-layer draft cache during preparation,
            # so it takes at least one layer's share of the projected target.
            draft_bytes = 0
            draft_covered = 0
            mtp_state = self._mtp_states.get(uid)
            if mtp_state is not None:
                for leaf in mtp_state[0]:
                    draft_bytes += int(getattr(leaf, "nbytes", 0))
                    draft_covered = max(
                        draft_covered, int(getattr(leaf, "offset", 0))
                    )
            if draft_covered > 0:
                draft_bytes = int(
                    draft_bytes * (max(context, draft_covered) / draft_covered)
                )
            else:
                layers = max(sum(1 for _ in prompt_cache), 1)
                draft_bytes = max(draft_bytes, target_bytes // layers)
            cache_gib = (target_bytes + draft_bytes) / float(1 << 30)
            joining.append(
                (
                    int(uid),
                    context,
                    int(config.get("num_draft", 1)),
                    False,
                    cache_gib,
                )
            )
        decisions = dict(self.mtp_admission(tuple(rows + joining)) or {})
        admitted = 0
        for row in joining:
            decision = decisions.get(row[0])
            if isinstance(decision, int) or decision == "plain":
                admitted += 1
            else:
                break
        return admitted

    def _make_mtp_batch(self, n: int):
        from .hybrid_speculative import prepare_self_mtp_lane

        sequences = [self._unprocessed_sequences.popleft() for _ in range(n)]
        detached = []
        initial = []
        stop_matchers = []
        progress = []
        for (
            uid,
            segments,
            maximum,
            prompt_cache,
            history,
            _,
            processors,
            matcher,
        ) in sequences:
            prompt = [token for segment in segments for token in segment]
            config = dict(self.self_mtp or {})
            config.update(self._mtp_configs.pop(uid, {}))
            self._validate_mtp_config(config)
            processors = list(processors or [])
            prefix = mx.array(history, dtype=mx.uint32)
            prepare_processors = [
                (
                    lambda y, logits, processor=processor: processor(
                        mx.concatenate([prefix, y]), logits
                    )
                )
                for processor in processors
            ]
            _prefetch_known_mtp_tail(self.model, history, prompt, config)
            prompt_boundary = {}
            lane, first = prepare_self_mtp_lane(
                mx.array(prompt, dtype=mx.uint32),
                self.model,
                uid=uid,
                max_tokens=maximum,
                prompt_cache=prompt_cache,
                mtp_state=self._mtp_states.pop(uid, None),
                lane_rng=self._mtp_lane_rngs.pop(uid, None),
                num_draft=int(config.get("num_draft", 1)),
                sampling_temp=float(config.get("sampling_temp", 0.0)),
                sampling_top_p=float(config.get("top_p", 1.0)),
                sampling_top_k=int(config.get("top_k", 0)),
                sampling_min_p=float(config.get("min_p", 0.0)),
                accept_rule=config.get("accept_rule", "residual"),
                logits_processors=prepare_processors,
                prefill_step_size=int(
                    config.get("prefill_step_size", self.prefill_step_size)
                ),
                share_qsa_indices=bool(config.get("share_qsa_indices", False)),
                diagnostic_stages=config.get("_diagnostic_prepare_stages"),
                fused_gdn_catchup=bool(config.get("fused_gdn_catchup", False)),
                prompt_boundary_out=prompt_boundary,
            )
            lane.lane.token_prefix = mx.array(history + prompt, dtype=mx.uint32)
            lane.lane.logits_processors = processors
            detached.append(lane)
            initial.append(first)
            stop_matchers.append(matcher)
            total = len(history) + len(prompt)
            progress.append(PromptProcessingBatch.Response(uid, (total, total), True, True))
            if prompt_boundary:
                covered = int(prompt_boundary["covered_tokens"])
                full_prompt = history + prompt
                if covered <= len(full_prompt):
                    prompt_boundary["tokens"] = list(full_prompt[:covered])
                    self._mtp_prompt_boundaries[uid] = prompt_boundary
        configured_segmented = _segment_aware_live_tip_enabled(self.self_mtp)
        existing_rows = bool(self._generation_batch.mtp_cycle_state())
        segmented_join = configured_segmented and (
            not existing_rows or self._generation_batch.segmented_live_tip
        )
        return (
            MTPGenerationBatch(
                self.model,
                detached,
                initial,
                stop_matchers,
                mtp_admission=self.mtp_admission,
                segmented_live_tip=segmented_join,
                async_qsa_promotion=(
                    _segmented_async_qsa_promotion_for_budget(
                        self.self_mtp,
                        min(
                            max(0, item.lane.max_tokens - item.lane.ntoks)
                            for item in detached
                        ),
                        record=True,
                    )
                ),
                arm_async_qsa_promotion=False,
            ),
            progress,
        )

    @staticmethod
    def _plain_sampler_for_mtp_lane(lane):
        def sample(logprobs):
            if lane.sampling_temp <= 0:
                return mx.argmax(logprobs, axis=-1)
            if lane.logprob_transform is not None:
                transformed = lane.logprob_transform(logprobs)
            else:
                transformed = logprobs / float(lane.sampling_temp)
            return mx.random.categorical(transformed, key=draw_key(lane.rng))

        sample.batch_groupable = False
        return sample

    def _migrate_plain_fallbacks(self):
        """Move plain-approved MTP lanes into the plain batch.

        Returns the responses this migration itself must emit: a lane that
        joined and was routed to plain at the merge boundary still carries its
        prepared-but-unemitted first token (``initial_output``, already
        counted in ``lane.ntoks``). It is delivered here exactly once, so the
        response neither drops that token nor double-subtracts it from the
        remaining budget; with ``max_tokens=1`` it is the lane's entire,
        terminal output.
        """
        if self.self_mtp is None:
            return []
        responses = []
        for package in self._generation_batch.take_plain_fallbacks():
            lane = package.detached.lane
            matcher_state = package.matcher_state
            initial = package.initial_output
            if initial is not None:
                matcher_state, reason = MTPGenerationBatch._finish_reason(
                    initial.token,
                    package.num_tokens + 1,
                    lane.max_tokens,
                    matcher_state,
                    package.stop_matcher,
                )
                response = MTPGenerationBatch.Response(
                    uid=lane.uid,
                    token=initial.token,
                    logprobs=initial.logprobs,
                    finish_reason=reason,
                    prompt_cache=None,
                    all_tokens=None,
                    from_draft=initial.from_draft,
                )
                responses.append(response)
                if reason is not None:
                    # Terminal on its prepared token: complete exactly like
                    # the MTP path's ``_complete_responses``.
                    response.prompt_cache = package.detached.caches.target
                    response.all_tokens = MTPGenerationBatch._prefix_tokens(lane)
                    response.mtp_state = (
                        package.detached.caches.draft,
                        lane.seed_h,
                    )
                    response.lane_rng = lane.rng
                    response.rng_draws = (
                        lane.rng.draws if lane.rng is not None else 0
                    )
                    _close_segmented_detached(
                        package.detached, release_cache=False
                    )
                    continue
            remaining = lane.max_tokens - lane.ntoks
            if remaining <= 0:
                _close_segmented_detached(package.detached, release_cache=True)
                continue
            plain = GenerationBatch(
                self.model,
                [lane.uid],
                mx.array([lane.cur], dtype=mx.uint32),
                _merge_caches([package.detached.caches.target]),
                [MTPGenerationBatch._prefix_tokens(lane)],
                [self._plain_sampler_for_mtp_lane(lane)],
                self.sampler,
                [lane.logits_processors],
                [package.stop_matcher],
                [remaining],
            )
            plain._matcher_states[0] = matcher_state
            self._plain_fallback_batch.extend(plain)
            _close_segmented_detached(package.detached, release_cache=True)
        return responses

    def mtp_cycle_state(self):
        if self.self_mtp is None:
            return []
        return self._generation_batch.mtp_cycle_state()

    def set_mtp_num_draft(self, depths: Union[int, Mapping[int, int]]):
        if self.self_mtp is None:
            raise RuntimeError("BatchGenerator is not in self-MTP mode")
        self._generation_batch.set_num_draft(depths)

    def _find_uids(self, uids):
        uids = set(uids)
        results = {}
        for i, uid_i in enumerate(self._generation_batch.uids):
            if uid_i in uids:
                results[uid_i] = (2, i)
        if self.self_mtp is not None:
            active = set(self._generation_batch.uids)
            for uid_i, *_ in self._generation_batch.mtp_cycle_state():
                if uid_i in uids and uid_i not in active:
                    results[uid_i] = (2, -1)
        for i, uid_i in enumerate(self._plain_fallback_batch.uids):
            if uid_i in uids:
                results[uid_i] = (3, i)
        for i, uid_i in enumerate(self._prompt_batch.uids):
            if uid_i in uids:
                results[uid_i] = (1, i)
        for i, seq in enumerate(self._unprocessed_sequences):
            if seq[0] in uids:
                results[seq[0]] = (0, i)
        return results

    def extract_cache(self, uids):
        results = {}
        for uid, (stage, idx) in self._find_uids(uids).items():
            if stage == 0:
                results[uid] = self._unprocessed_sequences[idx][3:5]
            elif stage == 1:
                results[uid] = (
                    self._prompt_batch.extract_cache(idx),
                    self._prompt_batch.tokens[idx],
                )
            else:
                if stage == 2:
                    if self.self_mtp is not None:
                        results[uid] = self._generation_batch.extract_uid(uid)
                    else:
                        results[uid] = (
                            self._generation_batch.extract_cache(idx),
                            self._generation_batch.tokens[idx],
                        )
                else:
                    results[uid] = (
                        self._plain_fallback_batch.extract_cache(idx),
                        self._plain_fallback_batch.tokens[idx],
                    )
        return results

    def pop_mtp_prompt_boundary(self, uid: int):
        """Transfer one committed prompt-boundary checkpoint to the server."""
        return self._mtp_prompt_boundaries.pop(int(uid), None)

    def remove(self, uids, return_prompt_caches=False):
        caches = {}
        if return_prompt_caches:
            caches = self.extract_cache(uids)

        keep = (
            set(range(len(self._unprocessed_sequences))),
            set(range(len(self._prompt_batch))),
            set(range(len(self._generation_batch))),
            set(range(len(self._plain_fallback_batch))),
        )
        found = self._find_uids(uids)
        for uid in uids:
            self._mtp_prompt_boundaries.pop(uid, None)
        for stage, idx in found.values():
            if idx >= 0:
                keep[stage].remove(idx)

        if len(keep[0]) < len(self._unprocessed_sequences):
            self._unprocessed_sequences = deque(
                x for i, x in enumerate(self._unprocessed_sequences) if i in keep[0]
            )
            for uid in uids:
                self._mtp_states.pop(uid, None)
                self._mtp_lane_rngs.pop(uid, None)
                self._mtp_configs.pop(uid, None)
        if len(keep[1]) < len(self._prompt_batch):
            self._prompt_batch.filter(sorted(keep[1]))
            self._currently_processing = [
                x for i, x in enumerate(self._currently_processing) if i in keep[1]
            ]
        if self.self_mtp is not None:
            self._generation_batch.remove_uids(uids)
        elif len(keep[2]) < len(self._generation_batch):
            self._generation_batch.filter(sorted(keep[2]))
        if len(keep[3]) < len(self._plain_fallback_batch):
            self._plain_fallback_batch.filter(sorted(keep[3]))

        return caches

    @property
    def prompt_cache_nbytes(self):
        total = sum(c.nbytes for p in self._unprocessed_sequences for c in p[3])
        # Queued lanes also retain their restored draft sidecars in
        # ``_mtp_states`` until preparation consumes them.
        total += sum(
            int(getattr(leaf, "nbytes", 0))
            for state in self._mtp_states.values()
            if state is not None
            for leaf in state[0]
        )
        total += sum(c.nbytes for c in self._prompt_batch.prompt_cache)
        total += sum(
            int(getattr(leaf, "nbytes", 0))
            for snapshot in getattr(self, "_mtp_prompt_boundaries", {}).values()
            for leaf in (
                list(snapshot.get("target_cache", ()))
                + list((snapshot.get("mtp_state") or ((), None))[0])
            )
        )
        if self.self_mtp is None:
            total += sum(c.nbytes for c in self._generation_batch.prompt_cache)
        else:
            total += self._generation_batch.cache_nbytes
        total += sum(c.nbytes for c in self._plain_fallback_batch.prompt_cache)
        return total

    def _make_batch(self, n: int):
        selected = self._select_prefill_indices(n)
        if selected == list(range(n)):
            sequences = [self._unprocessed_sequences.popleft() for _ in range(n)]
        else:
            selected = set(selected)
            queued = list(self._unprocessed_sequences)
            sequences = [sequence for i, sequence in enumerate(queued) if i in selected]
            self._unprocessed_sequences = deque(
                sequence for i, sequence in enumerate(queued) if i not in selected
            )

        uids = []
        caches = []
        tokens = []
        samplers = []
        logits_processors = []
        max_tokens = []
        stop_matchers = []
        for sequence in sequences:
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            stop_matchers.append(sequence[7])
            self._currently_processing.append(
                [
                    sequence[1],
                    0,
                    sum(len(s) for s in sequence[1]),
                    sum(c.nbytes for c in sequence[3]) == 0,
                    len(sequence[4]) if sequence[4] else 0,
                ]
            )

        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=logits_processors,
            stop_matchers=stop_matchers,
            max_tokens=max_tokens,
            prompt_trim_rollback_tokens=self.prompt_trim_rollback_tokens,
        )

    def _prefill_chunk_length(self, segments):
        if len(segments) == 1 and len(segments[0]) == 1:
            return 0
        return min(len(segments[0]), self.prefill_step_size)

    def _select_prefill_indices(self, n: int):
        """Select a padding-efficient, starvation-bounded admission cohort."""
        if n <= 0:
            return []

        window = min(
            len(self._unprocessed_sequences),
            max(n, self.prefill_batch_window),
        )
        if window == n:
            return list(range(n))

        candidates = list(self._unprocessed_sequences)[:window]
        candidate_lengths = [
            self._prefill_chunk_length(sequence[1]) for sequence in candidates
        ]
        active_lengths = [
            self._prefill_chunk_length(sequence[0])
            for sequence in self._currently_processing
            if not (len(sequence[0]) == 1 and len(sequence[0][0]) == 1)
        ]

        # Always admit the oldest request. This bounds every queued request's
        # wait even if later requests keep arriving with friendlier lengths.
        selected = [0]
        selected_lengths = list(active_lengths)
        if candidate_lengths[0] > 0:
            selected_lengths.append(candidate_lengths[0])

        remaining = set(range(1, window))
        while len(selected) < n:

            def padding_after_adding(i):
                lengths = selected_lengths
                if candidate_lengths[i] > 0:
                    lengths = lengths + [candidate_lengths[i]]
                if not lengths:
                    return 0
                return max(lengths) * len(lengths) - sum(lengths)

            best = min(remaining, key=lambda i: (padding_after_adding(i), i))
            selected.append(best)
            if candidate_lengths[best] > 0:
                selected_lengths.append(candidate_lengths[best])
            remaining.remove(best)

        return sorted(selected)

    def _capped_tokens(self, tokens):
        if self.max_kv_size is not None:
            return min(tokens, self.max_kv_size)
        return tokens

    def _budget_admissible(self, n):
        """How many of the first n queued sequences fit the state budget.

        Shared batch caches (BatchKVCache) allocate every row at the
        cohort-max step-rounded width, so cost is NON-ADDITIVE: each prefix
        length is evaluated by recomputing the full cohort projection at
        final extents (via the policy's ``cohort_bytes``), floored by actual
        live bytes. No per-row resident credit is granted under stepped
        geometry — a supplied cache's bytes cannot reduce the shared width.
        """
        if self.state_budget is None or n <= 0:
            return n
        self._sync_budget_mutation()
        queued = list(self._unprocessed_sequences)[:n]
        return self._admit_states(
            [self._candidate_admission_state(seq) for seq in queued]
        )

    def _sync_budget_mutation(self):
        if (
            self.kv_budget_bytes is not None
            and self.state_budget.budget_bytes != self.kv_budget_bytes
        ):
            # Preserve the experimental F3 attribute's mutability for callers
            # while routing its implementation through the generic policy.
            if not math.isfinite(self.kv_budget_bytes) or self.kv_budget_bytes <= 0:
                raise ValueError("kv_budget_bytes must be finite and positive")
            self.state_budget.budget_bytes = self.kv_budget_bytes

    def _candidate_admission_state(self, seq):
        new_tokens = sum(len(s) for s in seq[1])
        history = len(seq[4]) if seq[4] else 0
        total = history + new_tokens + seq[2]
        existing = sum(c.nbytes for c in seq[3])
        # Reviewer constraint: credit only against verifiable geometry.
        # A supplied cache WITH history is inside the projection target;
        # one WITHOUT history is unverifiable — its bytes sit OUTSIDE the
        # projection and are charged on top, never credited.
        unverified = float(existing) if (existing > 0 and history == 0) else 0.0
        return AdmissionState(
            seq[0],
            total,
            history,
            metadata={
                "phase": "queued",
                "prompt_units": new_tokens,
                "unverified_bytes": unverified,
            },
        )

    def _final_extent_states(self):
        """AdmissionStates of every admitted row at FINAL extent, for the
        shared-width cohort projection."""
        states = []
        gb = self._generation_batch
        for i in range(len(gb)):
            current = len(gb.tokens[i])
            final = current + max(gb.max_tokens[i] - gb._num_tokens[i], 0)
            states.append(
                AdmissionState(
                    gb.uids[i], final, current, metadata={"phase": "generation"}
                )
            )
        for i, seq in enumerate(self._currently_processing):
            history = seq[4] if len(seq) > 4 else 0
            final = history + seq[2] + self._prompt_batch.max_tokens[i]
            states.append(
                AdmissionState(
                    self._prompt_batch.uids[i],
                    final,
                    history + seq[1],
                    metadata={"phase": "prefill"},
                )
            )
        return states

    def _cohort_committed(self, candidate_states):
        """Projected committed bytes with the exact ``candidate_states``
        prefix admitted.

        The global cohort projection (all admitted rows + the selected
        prefix, at final extents, at the shared cohort-max rounded width)
        dominates both the current separate prompt/generation allocations
        and their eventual merge. Resident bytes of still-UNSELECTED queued
        rows are simultaneous with that future growth and are ADDED — never
        folded into a max — as are unverifiable supplied-cache bytes of
        selected candidates. The result is floored by admitted-batch actual
        live bytes (a stale wide allocation never assumed smaller than
        reality). State admission budget only; not a total process
        peak-memory guarantee (split/extend allocator transients are out of
        scope).
        """
        cost = self.state_budget.project
        cands = list(candidate_states)
        all_states = self._final_extent_states() + cands
        if hasattr(cost, "cohort_bytes"):
            projected = cost.cohort_bytes(all_states)
        else:
            projected = sum(self.state_budget.projected_bytes(s) for s in all_states)
        selected_unverified = sum(
            s.metadata.get("unverified_bytes", 0.0) for s in cands
        )
        selected_uids = {s.uid for s in cands}
        unselected_live = sum(
            float(sum(c.nbytes for c in seq[3]))
            for seq in self._unprocessed_sequences
            if seq[0] not in selected_uids
        )
        admitted_live = float(
            sum(c.nbytes for c in self._generation_batch.prompt_cache)
        ) + float(sum(c.nbytes for c in self._prompt_batch.prompt_cache))
        # Order matters: the live floor applies to the BASE projection only;
        # unverified selected bytes and unselected resident bytes are
        # simultaneous additions a large floor must never absorb.
        return max(projected, admitted_live) + selected_unverified + unselected_live

    def _admit_states(self, states):
        """How many of ``states`` fit, in order — recomputing the full
        non-additive cohort cost for each exact prefix length."""
        states = list(states)
        admitted = 0
        for k in range(1, len(states) + 1):
            if self._cohort_committed(states[:k]) > self.state_budget.budget_bytes:
                break
            admitted = k
        if (
            admitted == 0
            and states
            and len(self._generation_batch) == 0
            and len(self._prompt_batch) == 0
        ):
            # Liveness contract: a request whose projection alone exceeds
            # the budget is admitted when nothing else is running, mirroring
            # count-cap semantics where a single request always proceeds.
            # Best effort, not a guarantee against out-of-memory.
            logging.warning(
                "Request %s projects above the state budget "
                "(%d needed, %d budget) but nothing is running; "
                "admitting it anyway (best effort, not a guarantee "
                "against out-of-memory)",
                states[0].uid,
                int(self._cohort_committed(states[:1])),
                int(self.state_budget.budget_bytes),
            )
            admitted = 1
        return admitted

    def _next_mtp(self):
        generation_responses = []
        prompt_responses = []

        if self._generation_batch.mtp_cycle_state():
            generation_responses.extend(self._generation_batch.next())
        generation_responses.extend(self._migrate_plain_fallbacks())
        if len(self._plain_fallback_batch) > 0:
            generation_responses.extend(self._plain_fallback_batch.next())

        if generation_responses:
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        occupied = (
            len(self._generation_batch.mtp_cycle_state())
            + len(self._plain_fallback_batch)
        )
        n = min(
            self.completion_batch_size - occupied,
            len(self._unprocessed_sequences),
        )
        if _segment_aware_live_tip_enabled(self.self_mtp):
            # This first production gate is deliberately N=2. Keep dynamic
            # admission from silently growing a wider independent-B1 cohort.
            n = min(n, max(0, 2 - occupied))
        n = self._budget_admissible(n)
        n = self._admit_mtp_joining(n)
        if n > 0:
            batch, progress = self._make_mtp_batch(n)
            self._generation_batch.extend(batch)
            prompt_responses.extend(progress)

        return prompt_responses, generation_responses

    def _next(self):
        if self.self_mtp is not None:
            return self._next_mtp()

        generation_responses = []
        prompt_responses = []

        # Generate tokens first
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        # Exit early because we already have our hands full with decoding
        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        # Check if we have sequences and add them to the prompt batch
        n = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        n = self._budget_admissible(n)
        if n > 0:
            self._prompt_batch.extend(self._make_batch(n))

        # Split the prompt sequences to the ones moving to generation and the rest
        keep = []
        split = []
        for i, seq in enumerate(self._currently_processing):
            segments = seq[0]
            if len(segments) == 1 and len(segments[0]) == 1:
                split.append(i)
            else:
                keep.append(i)

        # Actually split off part of the prompt batch and start generation
        if split:
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            gen_batch = self._prompt_batch.split(split).generate(last_inputs)
            for i, p in enumerate(progress):
                prompt_responses.append(
                    PromptProcessingBatch.Response(
                        gen_batch.uids[i],
                        p,
                        True,
                        True,
                    )
                )
            self._generation_batch.extend(gen_batch)

        # Extract the next prompts input
        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            n = min(len(segments[0]), self.prefill_step_size)
            prompts.append(segments[0][:n])
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)

        # Process the prompts
        self._prompt_tokens_counter += sum(len(p) for p in prompts)
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        toc = time.perf_counter()
        self._prompt_time_counter += toc - tic

        return prompt_responses, generation_responses

    def next(self):
        """
        Get the next batch of responses.

        Returns:
            Tuple of prompt processing responses and generation responses.
        """
        with mx.stream(self._stream):
            return self._next()

    def next_generated(self):
        """
        Return only generated tokens ignoring batch generation responses.

        Returns:
            List of GenerationBatch.Response objects
        """
        with mx.stream(self._stream):
            while True:
                prompt_responses, generation_responses = self._next()
                if not generation_responses and prompt_responses:
                    continue
                return generation_responses


class ParallelSampleGenerator:
    """Decode ``n`` independent samples from one already prefilled prompt cache.

    This is the OpenAI ``n>1`` shape: one prompt, one prefill, ``n`` rows. The
    caller prefills the prompt once (see :func:`prefill_prompt_cache`) and hands
    the finished cache here; the cache is replicated into ``n`` batch rows by
    the same ``merge`` the continuous-batching path uses, so the prefix compute
    is paid once and each row then keeps its own sampling draws, its own token
    history and its own continuation.

    Per-row state that must not be shared is the caller's responsibility:
    ``logits_processors`` must be a fresh list per row (processors may hold
    mutable state), while a sampler object may be shared — rows sharing one
    ``batch_groupable`` sampler are sampled in a single vectorized call which
    draws independently per row.
    """

    def __init__(
        self,
        model: nn.Module,
        prompt_cache: List[Any],
        seed_token: int,
        n: int,
        *,
        max_tokens: int,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        all_tokens: Optional[List[int]] = None,
        prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
        stream=None,
        self_mtp: Optional[dict] = None,
        mtp_state: Optional[Tuple[List[Any], mx.array]] = None,
        lane_rng: Optional[LaneRNG] = None,
        mtp_prompt: Optional[Sequence[int]] = None,
        mtp_admission: Optional[
            Callable[
                [Sequence[Tuple[int, int, int, bool, float]]],
                Mapping[int, Union[int, str]],
            ]
        ] = None,
        prepared_prompt_cache: Optional[List[Any]] = None,
        prepared_prompt_cache_owner: Any = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
    ):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if logits_processors is not None and len(logits_processors) != n:
            raise ValueError("logits_processors must have one entry per sample")
        if logits_processors is not None and len(
            {id(lp) for lp in logits_processors}
        ) != len(logits_processors):
            # A shared list would let one row's processor state follow another.
            raise ValueError("each sample needs its own logits_processors list")

        self.n = n
        self._prepared_prompt_cache_owner = prepared_prompt_cache_owner
        history = list(all_tokens or [])
        if self_mtp is None:
            self._generator = BatchGenerator(
                model,
                completion_batch_size=n,
                prefill_batch_size=n,
                prefill_step_size=prefill_step_size,
                stream=stream,
            )
            if prepared_prompt_cache is None:
                uids = self._generator.insert(
                    prompts=[[int(seed_token)] for _ in range(n)],
                    max_tokens=[max_tokens] * n,
                    # merge() reads these leaves and creates independent rows.
                    caches=[list(prompt_cache) for _ in range(n)],
                    all_tokens=[list(history) for _ in range(n)],
                    samplers=samplers,
                    logits_processors=logits_processors,
                    stop_matchers=stop_matchers,
                )
            else:
                # The cache was materialized from the one shared APC/prefill
                # result. Enter generation directly so BatchGenerator cannot
                # merge the same B1 history a second time before the first
                # authoritative model consumer.
                uids = list(range(n))
                matchers = stop_matchers or [StopSequenceMatcher()] * n
                row_samplers = samplers or [None] * n
                processors = logits_processors or [[] for _ in range(n)]
                try:
                    with mx.stream(self._generator.stream):
                        self._generator._generation_batch = GenerationBatch(
                            model,
                            uids,
                            mx.array([int(seed_token)] * n, dtype=mx.uint32),
                            prepared_prompt_cache,
                            [list(history) for _ in range(n)],
                            row_samplers,
                            self._generator.sampler,
                            processors,
                            matchers,
                            [max_tokens] * n,
                        )
                except BaseException:
                    owner = self._prepared_prompt_cache_owner
                    self._prepared_prompt_cache_owner = None
                    if owner is not None:
                        try:
                            mx.synchronize(self._generator.stream)
                        except BaseException as cleanup_error:
                            logging.warning(
                                "Failed to drain prepared-cache stream after "
                                "constructor error: %s",
                                cleanup_error,
                            )
                        try:
                            owner.close(synchronize=False)
                        except BaseException as cleanup_error:
                            logging.warning(
                                "Failed to release prepared cache after "
                                "constructor error: %s",
                                cleanup_error,
                            )
                    raise
                self._generator._uid_count = n
        else:
            if prepared_prompt_cache is not None:
                raise ValueError(
                    "prepared cache capsules are not yet a self-MTP cache ABI"
                )
            # Route processor-bearing requests to plain BEFORE any self-MTP
            # validation can raise: this branch must fail closed, never crash.
            processors = logits_processors or [[] for _ in range(n)]
            if any(processors):
                # The server normally routes this case to plain before prefill.
                # Keep the direct generator API fail-closed too.
                self_mtp = None
                self._generator = BatchGenerator(
                    model,
                    completion_batch_size=n,
                    prefill_batch_size=n,
                    prefill_step_size=prefill_step_size,
                    stream=stream,
                )
                uids = self._generator.insert(
                    prompts=[[int(seed_token)] for _ in range(n)],
                    max_tokens=[max_tokens] * n,
                    caches=[list(prompt_cache) for _ in range(n)],
                    all_tokens=[list(history) for _ in range(n)],
                    samplers=samplers,
                    logits_processors=processors,
                    stop_matchers=stop_matchers,
                )
                self._index = {uid: i for i, uid in enumerate(uids)}
                self._active = set(uids)
                return
            if lane_rng is None:
                raise ValueError("parallel self-MTP requires a generation-thread LaneRNG")
            config = dict(self_mtp)
            BatchGenerator._validate_mtp_config(config)
            if kv_bits is not None:
                if not config.get("allow_quantized_kv"):
                    raise ValueError(
                        "quantized KV caches are not MTP batchable unless "
                        "allow_quantized_kv is set in the self-MTP config"
                    )
                maybe_quantize_kv_cache(
                    prompt_cache, 0, kv_group_size, kv_bits
                )
            lane_rngs = lane_rng.fork(n)
            mx.eval([rng.key for rng in lane_rngs])
            matchers = stop_matchers or [StopSequenceMatcher() for _ in range(n)]
            prompt_tail = list(mtp_prompt) if mtp_prompt is not None else [seed_token]

            from .hybrid_speculative import (
                DetachedSelfMTPLane,
                MTPToken,
                SelfMTPCachePair,
                prepare_self_mtp_lane,
            )

            segmented_requested = _segment_aware_live_tip_enabled(config)
            if segmented_requested and n != 2:
                raise ValueError(
                    "segment-aware live-tip self-MTP is qualified only for N=2"
                )
            fanout_requested = bool(config.get("gdn_prefix_fanout", False))
            fanout_candidate = (
                fanout_requested and n == 2 and not segmented_requested
            )
            if fanout_requested:
                from .gdn_prefix_fanout import _note_serving_event

                _note_serving_event("requests")
                if not fanout_candidate:
                    _note_serving_event("declined_not_n2")

            _prefetch_known_mtp_tail(model, history, prompt_tail, config)

            canonical, first = prepare_self_mtp_lane(
                mx.array(prompt_tail, dtype=mx.uint32),
                model,
                uid=0,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                mtp_state=mtp_state,
                lane_rng=lane_rngs[0],
                num_draft=int(config.get("num_draft", 1)),
                sampling_temp=float(config.get("sampling_temp", 0.0)),
                sampling_top_p=float(config.get("top_p", 1.0)),
                sampling_top_k=int(config.get("top_k", 0)),
                sampling_min_p=float(config.get("min_p", 0.0)),
                accept_rule=config.get("accept_rule", "residual"),
                logits_processors=processors[0],
                prefill_step_size=prefill_step_size,
                share_qsa_indices=bool(config.get("share_qsa_indices", False)),
                record_prefix_fanout=fanout_candidate,
                diagnostic_stages=config.get("_diagnostic_prepare_stages"),
                fused_gdn_catchup=bool(config.get("fused_gdn_catchup", False)),
            )
            canonical.lane.token_prefix = mx.array(
                history + prompt_tail, dtype=mx.uint32
            )
            if segmented_requested:
                prefix_bytes = np.asarray(
                    history + prompt_tail, dtype="<u4"
                ).tobytes()
                canonical.shared_qsa_prefix_id = hashlib.sha256(
                    prefix_bytes
                ).hexdigest()
            async_qsa_prequeue = None
            remaining_budget = max(
                0, canonical.lane.max_tokens - canonical.lane.ntoks
            )
            if _segmented_async_qsa_prequeue_enabled(
                config, remaining_budget
            ):
                from .segmented_physical_promotion import (
                    SegmentedPhysicalPromotionDeclined,
                    begin_shared_prefix_physical_promotion,
                )
                from .segmented_self_mtp import note_segmented_self_mtp

                note_segmented_self_mtp("async_qsa_prequeue_requests")
                try:
                    async_qsa_prequeue = begin_shared_prefix_physical_promotion(
                        canonical,
                        rows=n,
                        reserve_tail=int(config.get("num_draft", 1)) + 1,
                        stream=mx.new_stream(mx.gpu),
                    )
                except SegmentedPhysicalPromotionDeclined as error:
                    note_segmented_self_mtp("async_qsa_prequeue_declined")
                    logging.info("Async QSA shared-prefix prequeue declined: %s", error)
                else:
                    note_segmented_self_mtp("async_qsa_prequeue_queued")
                    note_segmented_self_mtp("async_qsa_promotion_requests")
                    note_segmented_self_mtp("async_qsa_promotion_queued")
            prepared_caches = None
            if fanout_candidate:
                owner = None
                try:
                    from .gdn_prefix_fanout import HybridCachePrefixFanout

                    owner = HybridCachePrefixFanout.from_prompt_cache(
                        canonical.caches.target,
                        enabled=True,
                        strict=False,
                    )
                    if owner is not None:
                        if bool(config.get("gdn_prefix_fanout_consume", False)):
                            lease = owner.fork_live_tip()
                        else:
                            lease = owner.fork(owner.span)
                        target_batch = lease.take_batch()
                        draft_batch = [
                            type(cache).merge([cache, cache])
                            for cache in canonical.caches.draft
                        ]
                        mx.eval(
                            [cache.state for cache in target_batch],
                            [cache.state for cache in draft_batch],
                        )
                        prepared_caches = SelfMTPCachePair(
                            target=target_batch,
                            draft=draft_batch,
                        )
                    else:
                        _note_serving_event("declined_cache")
                except Exception as error:
                    logging.warning(
                        "GDN prefix fan-out declined; using the ordinary "
                        "self-MTP cache merge: %s",
                        error,
                    )
                    prepared_caches = None
                    _note_serving_event("declined_error")
                finally:
                    if owner is not None:
                        owner.close()
                    for cache in canonical.caches.target:
                        cache.stop_speculation()
                    _note_serving_event("cleanups")

            lanes = [canonical]
            first_outputs = [first]
            for uid in range(1, n):
                lane = (
                    DetachedSelfMTPLane(
                        lane=copy.deepcopy(canonical.lane),
                        caches=canonical.caches,
                    )
                    if prepared_caches is not None
                    else copy.deepcopy(canonical)
                )
                lane.lane.uid = uid
                lane.lane.rng = lane_rngs[uid]
                lane.lane.logits_processors = processors[uid]
                if lane.lane.sampling_temp > 0:
                    token = mx.random.categorical(
                        first.logprobs, key=draw_key(lane_rngs[uid])
                    )
                    mx.eval(token)
                    token = int(token.item())
                else:
                    token = int(mx.argmax(first.logprobs).item())
                lane.lane.cur = token
                lanes.append(lane)
                first_outputs.append(MTPToken(token, first.logprobs, False))

            self._generator = BatchGenerator(
                model,
                completion_batch_size=n,
                prefill_batch_size=n,
                prefill_step_size=prefill_step_size,
                stream=stream,
                self_mtp=config,
                mtp_admission=mtp_admission,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
            )
            # The admission callback re-budgets at every cycle boundary, so the
            # lanes can drop k, migrate to plain, or pause under pressure.
            try:
                generation_batch = MTPGenerationBatch(
                    model,
                    lanes,
                    first_outputs,
                    matchers,
                    prepared_caches=prepared_caches,
                    mtp_admission=mtp_admission,
                    segmented_live_tip=segmented_requested,
                    async_qsa_promotion=(
                        _segmented_async_qsa_promotion_for_budget(
                            config,
                            min(
                                max(0, item.lane.max_tokens - item.lane.ntoks)
                                for item in lanes
                            ),
                            record=True,
                        )
                    ),
                    async_qsa_prequeue=async_qsa_prequeue,
                )
                if prepared_caches is not None:
                    _note_serving_event("engaged")
            except Exception as error:
                if async_qsa_prequeue is not None:
                    async_qsa_prequeue.cancel_and_drain()
                if segmented_requested:
                    from .segmented_self_mtp import note_segmented_self_mtp

                    logging.warning(
                        "Segment-aware B1 self-MTP declined; rebuilding through "
                        "the ordinary physical batch: %s",
                        error,
                    )
                    note_segmented_self_mtp("declined")
                    note_segmented_self_mtp("failures")
                    note_segmented_self_mtp("fallback_physical_b2")
                    note_segmented_self_mtp("physical_b2_formations")
                    generation_batch = MTPGenerationBatch(
                        model,
                        lanes,
                        first_outputs,
                        matchers,
                        mtp_admission=mtp_admission,
                        segmented_live_tip=False,
                    )
                elif prepared_caches is None:
                    raise
                else:
                    logging.warning(
                        "GDN prefix fan-out attach declined; rebuilding through "
                        "the ordinary self-MTP merge: %s",
                        error,
                    )
                    _note_serving_event("declined_error")
                    fallback_lanes = [canonical]
                    for lane in lanes[1:]:
                        fallback_lanes.append(
                            DetachedSelfMTPLane(
                                lane=lane.lane,
                                caches=copy.deepcopy(canonical.caches),
                            )
                        )
                    generation_batch = MTPGenerationBatch(
                        model,
                        fallback_lanes,
                        first_outputs,
                        matchers,
                        mtp_admission=mtp_admission,
                    )
            self._generator._generation_batch = generation_batch
            uids = list(range(n))
        self._index = {uid: i for i, uid in enumerate(uids)}
        self._active = set(uids)

    def __len__(self):
        return len(self._active)

    @property
    def active_samples(self) -> List[int]:
        return sorted(self._index[uid] for uid in self._active)

    def next(self) -> List[Tuple[int, "GenerationBatch.Response"]]:
        """Advance one decode step over all live rows.

        Returns ``(sample_index, response)`` pairs. A row is dropped from the
        generator once its response carries a ``finish_reason``.
        """
        if not self._active:
            return []
        _, generation_responses = self._generator.next()
        results = []
        for r in generation_responses:
            index = self._index[r.uid]
            if r.finish_reason is not None:
                self._active.discard(r.uid)
            results.append((index, r))
        results.sort(key=lambda pair: pair[0])
        return results

    def close(self):
        first_error = None
        try:
            self._generator.close()
        except BaseException as error:
            first_error = error
        owner = self._prepared_prompt_cache_owner
        if owner is not None:
            try:
                mx.synchronize(self._generator.stream)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                batch = getattr(self._generator, "_generation_batch", None)
                if batch is not None:
                    batch.prompt_cache = []
                self._prepared_prompt_cache_owner = None
                try:
                    owner.close(synchronize=False)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error


@dataclass
class BatchResponse:
    """
    A data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        caches: Optional prompt caches for each sequence.
        token_ids (Optional[List[List[int]]]): The generated token IDs for each
            prompt. Only present when ``return_token_ids=True``.
        logprobs (Optional[List[List[float]]]): The per-token log-probabilities
            of the sampled tokens for each prompt. Only present when
            ``return_logprobs=True``.
    """

    texts: List[str]
    stats: BatchStats
    caches: Optional[List[List[Any]]]
    token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[List[float]]] = None


def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    return_token_ids: bool = False,
    return_logprobs: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompts (List[List[int]]): The input prompts.
       prompt_caches (List[List[Any]], optional): Pre-computed prompt-caches
          for each input prompt. Note, unlike ``generate_step``, the caches
          won't be updated in-place.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       return_prompt_caches (bool): Return the prompt caches in the batch
          responses. Default: ``False``.
       return_token_ids (bool): Return the generated token IDs in the batch
          responses. Default: ``False``.
       return_logprobs (bool): Return the per-token log-probability of the
          sampled token for each generated token. Useful for reinforcement
          learning (e.g. RLOO, PPO) where behavior log-probabilities are needed
          for importance weighting. Default: ``False``.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(prompts, max_tokens, caches=prompt_caches)
    results = {uid: [] for uid in uids}
    logprob_results = {uid: [] for uid in uids} if return_logprobs else None
    prompt_caches = {}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    if return_prompt_caches:
                        prompt_caches[r.uid] = r.prompt_cache
                    if verbose:
                        fin += 1
                        print(
                            f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                            end="\r",
                        )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
                    if return_logprobs:
                        logprob_results[r.uid].append(r.logprobs[r.token].item())
    gen.close()
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    caches = [prompt_caches[uid] for uid in uids] if return_prompt_caches else None
    token_ids = [results[uid] for uid in uids] if return_token_ids else None
    logprobs = [logprob_results[uid] for uid in uids] if return_logprobs else None
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
        if stats.effective_quantized_kv_start is not None:
            print(
                "[batch_generate] Quantized KV cache from step: "
                f"{stats.effective_quantized_kv_start}"
            )
    return BatchResponse(texts, stats, caches, token_ids, logprobs)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    try:
        validate_kv_quantization_args(
            args.kv_bits,
            args.kv_key_bits,
            args.kv_value_bits,
            args.kv_group_size,
            args.quantized_kv_start,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.seed is not None:
        mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            key_bits, value_bits = _resolve_kv_bits(
                args.kv_bits, args.kv_key_bits, args.kv_value_bits
            )
            if key_bits is not None and (
                key_bits != prompt_cache[0].key_bits
                or value_bits != prompt_cache[0].value_bits
            ):
                raise ValueError(
                    "KV quantization bits do not match the cache loaded from "
                    "--prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = args.trust_remote_code

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
        model_config={"quantize_activations": args.quantize_activations},
        trust_remote_code=args.trust_remote_code,
    )
    for eos_token in args.extra_eos_token:
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.has_chat_template:
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        has_prefill = args.prefill_response is not None
        if has_prefill:
            messages.append({"role": "assistant", "content": args.prefill_response})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **template_kwargs,
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=has_prefill,
                add_generation_prompt=not has_prefill,
            )
            prompt = prompt[test_prompt.index("<query>") :]
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt = tokenizer.encode(prompt)

    if args.draft_model is not None:
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    else:
        draft_model = None
    sampler = make_sampler(
        args.temp,
        args.top_p,
        args.min_p,
        args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prefill_step_size=args.prefill_step_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        kv_rotate=args.kv_rotate,
        kv_key_bits=args.kv_key_bits,
        kv_value_bits=args.kv_value_bits,
        draft_model=draft_model,
        num_draft_tokens=args.num_draft_tokens,
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.generate...` directly is deprecated."
        " Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead."
    )
    main()
