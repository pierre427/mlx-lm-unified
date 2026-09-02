# Copyright © 2023-2026 Apple Inc.

import argparse
import hmac
import importlib
import json
import logging
import math
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Condition, Lock, Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .apc import AutomaticPrefixCache, MTPAPCSidecar, _walk_cache_entries
from .generate import (
    DEFAULT_QUANTIZED_KV_START,
    BatchGenerator,
    ParallelSampleGenerator,
    StopSequenceMatcher,
    TextStateMachine,
    validate_kv_quantization_args,
    maybe_quantize_kv_cache,
    make_stop_matcher,
    make_text_state_machine,
    prefill_prompt_cache,
    stream_generate,
)
from .models.cache import LRUPromptCache, RotatingKVCache, make_prompt_cache
from .sample_utils import LaneRNG, make_logits_processors, make_sampler
from .spec_policy import MAX_DRAFT_TOKENS
from .speculation_router import DepthCeilingController
from .utils import _parse_size, load, sharded_load


def validate_kv_args(args):
    """Fail at server startup for incomplete per-side KV precision flags."""
    return validate_kv_quantization_args(
        args.kv_bits,
        args.kv_key_bits,
        args.kv_value_bits,
        args.kv_group_size,
        args.quantized_kv_start,
    )


class RequestCompositionError(ValueError):
    """The request is well formed but asks for an unsupported combination.

    Mapped to HTTP 400 so a client can tell it apart from an unknown model,
    which stays 404.
    """


def get_system_fingerprint():
    gpu_arch = mx.device_info()["architecture"]
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class ToolCallFormatter:
    def __init__(self, tool_parser, tools, streaming=False):
        self._idx = 0
        self._tool_parser = tool_parser
        self._tools = tools
        self._streaming = streaming

    def _format(self, tc):
        # Copy before mutating -- `tc` is owned by the tool parser and may be
        # reused/inspected by its caller; pop/assign must not touch it.
        tc = dict(tc)
        tc_id = tc.pop("id", None) or str(uuid.uuid4())
        tc["arguments"] = json.dumps(tc["arguments"], ensure_ascii=False)
        out = {
            "function": tc,
            "type": "function",
            "id": tc_id,
        }
        if self._streaming:
            out["index"] = self._idx
            self._idx += 1
        return out

    def __call__(self, tool_calls):
        if not tool_calls or self._tool_parser is None:
            return []

        result = []
        for tool_text in tool_calls:
            try:
                parsed = self._tool_parser(tool_text, self._tools)
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Failed to parse tool call ({type(e).__name__}: {e}) — "
                    f"tool text was likely truncated mid-generation."
                )
                continue
            if not isinstance(parsed, list):
                parsed = [parsed]
            for tc in parsed:
                try:
                    result.append(self._format(tc))
                except (KeyError, TypeError, ValueError) as e:
                    # One malformed call (e.g. missing "arguments") must not
                    # discard the valid siblings already parsed from this block.
                    logging.warning(
                        f"Dropping malformed tool call ({type(e).__name__}: {e})"
                    )
                    continue
        return result


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping or default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. It converts the 'content' field
    to a string instead of a list of text fragments.

    Args:
        message_list (list): A list of dictionaries, where each dictionary may
          have a 'content' key containing a list of dictionaries with 'type' and
          'text' keys.

    Raises:
        ValueError: If the 'content' type is not supported or if 'text' is missing.

    """
    for message in messages:
        # OpenAI-compatible clients commonly echo the server's ``reasoning``
        # field, while Qwen's preserved-thinking template consumes
        # ``reasoning_content``. Keep both spellings losslessly equivalent so
        # historical traces remain part of chat serialization and APC identity.
        if (
            "reasoning_content" not in message
            and isinstance(message.get("reasoning"), str)
        ):
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                if func := tool_call.get("function"):
                    if args := func.get("arguments"):
                        func["arguments"] = json.loads(args)


@dataclass
class ModelDescription:
    model: str
    draft: str
    adapter: str


@dataclass
class SamplingArguments:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    xtc_probability: float
    xtc_threshold: float


@dataclass
class LogitsProcessorArguments:
    logit_bias: Optional[Dict[int, float]]
    repetition_penalty: float
    repetition_context_size: int
    presence_penalty: float
    presence_context_size: int
    frequency_penalty: float
    frequency_context_size: int


@dataclass
class GenerationArguments:
    model: ModelDescription
    sampling: SamplingArguments
    logits: LogitsProcessorArguments

    stop_words: List[str]

    max_tokens: int
    num_draft_tokens: int
    logprobs: bool
    top_logprobs: int
    seed: Optional[int]
    chat_template_kwargs: Optional[Dict[str, Any]]
    # OpenAI ``n``: how many independent samples to draw from this one prompt.
    n: int = 1
    prompt_lookup_ngram: int = 0
    prompt_lookup_tokens: int = 8
    prompt_lookup_adaptive: bool = True
    prompt_lookup_rate_gate: bool = True
    prompt_lookup_warmup: int = 48
    prompt_lookup_gate: float = 0.12
    prompt_lookup_rate_gate_probe: int = 32
    prompt_lookup_rate_gate_margin: float = 0.0


@dataclass
class CompletionRequest:
    request_type: Literal["chat", "text"]

    prompt: str

    messages: List[Any]
    tools: Optional[List[Any]]
    role_mapping: Optional[Dict[str, Any]]


@dataclass
class GenerationContext:
    has_tool_calling: bool
    has_thinking: bool
    tool_parser: Callable[[str, Any], Dict]

    text_sm: TextStateMachine
    initial_state: str

    prompt: List[int]
    prompt_cache_count: int = -1

    _should_stop: bool = False

    def stop(self):
        self._should_stop = True


@dataclass
class Response:
    text: str
    token: int
    logprob: float
    finish_reason: Optional[str]
    top_tokens: Tuple[Dict[str, Any]]
    # Which sample (OpenAI choice index) this token belongs to. Always 0 on
    # the single-sample path.
    index: int = 0


class TimeBudget:
    def __init__(self, budget=0.5, iterations=25, sync_frequency=10):
        self._is_distributed = mx.distributed.init().size() > 1
        self._budget = budget
        self._iterations = iterations
        self._sync_frequency = sync_frequency
        self._start = None
        self._current_iterations = None
        self._loops = 0
        self._time_spent = 0

    def __iter__(self):
        self._start = time.time()
        self._current_iterations = 0
        return self

    def __next__(self):
        if not self._is_distributed:
            if time.time() - self._start > self._budget:
                raise StopIteration()
            return None

        self._current_iterations += 1
        if self._current_iterations <= self._iterations:
            return None

        self._loops += 1
        self._time_spent += time.time() - self._start
        if self._loops % self._sync_frequency == 0:
            loop_time = mx.distributed.all_sum(self._time_spent).item()
            avg_loop_time = loop_time / (
                mx.distributed.init().size() * self._sync_frequency
            )
            factor = self._budget / avg_loop_time
            self._iterations = max(round(self._iterations * factor), 1)
            self._loops = 0
            self._time_spent = 0
        raise StopIteration()


def _measure_kv_cost(model):
    """Measure (raw_fixed_bytes, bytes_per_token, common_step_units) for one
    sequence row of this model's cache.

    Returns the RAW linear fit plus the validated COMMON allocation step of
    all growing stepped leaves; step-aware (cohort-level) rounding is applied
    by the admission layer, not here — returning padded values would double
    count once the cohort projector rounds. Budgeting is refused when:
    growing leaves disagree on step, a growing leaf lacks a usable step, a
    composite cache cannot be recursed, or a rotating cache saturates inside
    the probe range. A non-boundary verification (528 tokens) checks the
    step-rounded projection covers observed bytes.
    """
    caches = make_prompt_cache(model)

    def leaves(cs):
        for c in cs:
            inner = getattr(c, "caches", None)
            if inner:
                yield from leaves(inner)
            elif hasattr(c, "nbytes"):
                yield c
            else:
                raise ValueError(
                    f"opaque cache component {type(c).__name__}: cannot "
                    f"verify allocation growth, refusing byte budgeting"
                )

    def rotating_windows(cs):
        for c in leaves(cs):
            if isinstance(c, RotatingKVCache) and c.max_size is not None:
                yield c.max_size

    windows = list(rotating_windows(caches))
    warm, probe = 256, 1024
    verify_at = 528  # deliberately NOT a step boundary
    third_at = 2048  # independent consistency point (step boundary)
    if windows and max(warm + probe, verify_at, third_at) >= min(windows):
        raise ValueError(
            f"--state-budget-gb needs a linear-growth cache probe, but a "
            f"rotating cache saturates at {min(windows)} tokens (inside the "
            f"probe range). Byte budgeting is not supported for this "
            f"model configuration."
        )

    def forward(cs, n, start):
        toks = mx.array([[(start + i) % 100 + 1 for i in range(n)]])
        model(toks, cache=cs)
        mx.eval([c.state for c in cs])

    leaf_list = list(leaves(caches))
    forward(caches, warm, 0)
    base_per_leaf = [c.nbytes for c in leaf_list]
    forward(caches, probe, warm)
    grown_per_leaf = [c.nbytes for c in leaf_list]

    per_token = sum(g - b for g, b in zip(grown_per_leaf, base_per_leaf)) / probe
    for c, b, g in zip(leaf_list, base_per_leaf, grown_per_leaf):
        leaf_slope = (g - b) / probe
        leaf_intercept = b - leaf_slope * warm
        if leaf_intercept < -(0.01 * max(b, 1.0) + leaf_slope):
            # Aggregate intercepts can cancel across leaves; a materially
            # negative PER-LEAF intercept means that leaf's growth is not
            # linear from zero
            raise ValueError(
                f"state-cost fit for leaf {type(c).__name__} has a "
                f"materially negative intercept ({leaf_intercept:.0f} "
                f"bytes); the cache does not fit fixed+linear growth, "
                f"refusing byte budgeting"
            )
    raw_fixed = sum(base_per_leaf) - per_token * warm
    if raw_fixed < -(0.01 * sum(base_per_leaf) + per_token):
        # A materially negative intercept means the growth is not linear
        # from zero (silently clamping would hide superlinear early growth)
        raise ValueError(
            f"state-cost fit has a materially negative intercept "
            f"({raw_fixed:.0f} bytes); the cache does not fit "
            f"fixed+linear growth, refusing byte budgeting"
        )
    raw_fixed = max(raw_fixed, 0.0)

    # Validate ONE common allocation step across all growing leaves;
    # fail closed on anything unverifiable (reviewer requirements).
    steps = set()
    for c, b, g in zip(leaf_list, base_per_leaf, grown_per_leaf):
        if g > b:
            step = getattr(c, "step", None)
            if (
                step is None
                or isinstance(step, bool)
                or not isinstance(step, int)
                or step <= 0
            ):
                raise ValueError(
                    f"growing cache leaf {type(c).__name__} has no usable "
                    f"allocation step ({step!r}); refusing byte budgeting"
                )
            steps.add(step)
    if len(steps) > 1:
        raise ValueError(
            f"growing cache leaves disagree on allocation step {sorted(steps)}; "
            f"mixed-step budgeting is not supported"
        )
    common_step = steps.pop() if steps else 1

    # Non-boundary verification: step-rounded projection must cover reality
    vcaches = make_prompt_cache(model)
    forward(vcaches, verify_at, 0)
    observed = sum(c.nbytes for c in leaves(vcaches))
    rounded_units = -(-verify_at // common_step) * common_step
    projected = raw_fixed + per_token * rounded_units
    if observed > projected:
        raise ValueError(
            f"state-cost safety check failed: observed {observed} bytes at "
            f"{verify_at} tokens exceeds step-rounded projection "
            f"{projected:.0f}; refusing byte budgeting"
        )

    # Independent higher point: the two-point fit is measured on
    # [256, 1280]; a slope that changes past 1280 would still pass the
    # 528 check. Verify PER LEAF at 2048 with a FRESH cache and a SINGLE
    # aligned chunk: capacity after unaligned multi-chunk growth is
    # previous_logical + round_up(chunk, step), which exceeds
    # round_up(total) (root cause of the +16-token capacity observation
    # on the earlier two-stage path) — a fresh aligned forward removes
    # that term, so a consistent leaf predicts EXACTLY.
    third_caches = make_prompt_cache(model)
    forward(third_caches, third_at, 0)
    vleaves = list(leaves(third_caches))
    for c, b, g in zip(vleaves, base_per_leaf, grown_per_leaf):
        leaf_slope = (g - b) / probe
        predicted = b + leaf_slope * (third_at - warm)
        observed_leaf = c.nbytes
        # Single aligned chunk on a fresh cache: exact prediction expected;
        # tolerance covers arithmetic error only
        tolerance = max(1.0, 1e-6 * abs(predicted))
        if abs(observed_leaf - predicted) > tolerance:
            raise ValueError(
                f"state-cost consistency check failed for leaf "
                f"{type(c).__name__}: observed {observed_leaf} bytes at "
                f"{third_at} tokens vs predicted {predicted:.0f} (fit "
                f"measured on 256..1280); the cache does not grow "
                f"linearly, refusing byte budgeting"
            )
    logging.info(
        "state-cost third-point verification passed: %d leaves at %d "
        "tokens, all within tolerance of the 256..1280 linear fit",
        len(vleaves),
        third_at,
    )
    logging.info(
        "state-cost fit: per_token %.0f B, raw fixed %.0f B, common step %d "
        "(leaves: %s); verified at %d tokens: observed %d <= projected %.0f",
        per_token,
        raw_fixed,
        common_step,
        sorted({type(c).__name__ for c in leaf_list}),
        verify_at,
        observed,
        projected,
    )
    return raw_fixed, per_token, common_step


def _release_int8_prefill_overlay(*, reapply: bool = True) -> bool:
    """Flush the int8 prefill overlay at a model-unload boundary.

    The overlay caches per-channel scales (and, with MLX_LM_INT8_CACHE=ttl,
    int8 weight copies) keyed by module identity; entries must not outlive the
    model that owned them. release() runs remove() (restore + flush) and, when
    ``reapply`` is set, reinstalls the patch with empty caches so the next
    model keeps int8 prefill; pass reapply=False at process shutdown. A no-op
    when the overlay was never applied.
    """
    from .int8_prefill import release

    return release(reapply=reapply)


def _maybe_apply_int8_prefill(args) -> bool:
    """Install the int8 NAX prefill overlay when requested by CLI or env."""
    enabled = getattr(args, "int8_prefill", False) or os.environ.get(
        "MLX_LM_INT8_PREFILL", ""
    ).lower() in ("1", "true", "yes", "on")
    if not enabled:
        return False
    from .int8_prefill import apply

    apply()
    logging.info("int8 NAX prefill patch applied.")
    return True


SOFT_RELOAD_PATH = "/v1/admin/soft_reload"
EFFECTIVE_CONFIG_PATH = "/v1/admin/config"

# Longest a soft reload waits for in-flight generation before it gives up.
DEFAULT_SOFT_RELOAD_DRAIN_TIMEOUT = 120.0
# Longest a new request waits at the closed admission gate before a 503.
DEFAULT_SOFT_RELOAD_ADMISSION_TIMEOUT = 30.0
# Bounded generation-thread backoff when a serve slice makes no progress
# (every self-MTP lane queued/paused, or the memory probe unavailable), and
# the ordinary idle poll interval. Queueing is fine; hot retry is not.
BATCH_IDLE_BACKOFF_SECONDS = 0.1


class SoftReloadError(ValueError):
    """The requested soft reload is malformed or not permitted (400)."""


class SoftReloadRestartRequired(SoftReloadError):
    """The requested key exists but only a process restart can change it."""


class SoftReloadBusy(RuntimeError):
    """The server could not quiesce, or a reload is already running (503)."""


def _reload_flag(value):
    if isinstance(value, bool):
        return value
    if value in (0, 1) and isinstance(value, int):
        return bool(value)
    raise SoftReloadError(f"expected a boolean, got {value!r}")


def _reload_int(low, high, *, allow_none=False):
    def check(value):
        if value is None:
            if allow_none:
                return None
            raise SoftReloadError("expected an integer, got null")
        if isinstance(value, bool) or not isinstance(value, int):
            raise SoftReloadError(f"expected an integer, got {value!r}")
        if not low <= value <= high:
            raise SoftReloadError(f"{value} is outside [{low}, {high}]")
        return value

    return check


def _reload_float(low, high):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SoftReloadError(f"expected a number, got {value!r}")
        value = float(value)
        if not low <= value <= high:
            raise SoftReloadError(f"{value} is outside [{low}, {high}]")
        return value

    return check


def _reload_auto_flag(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto"}:
        return None
    return _reload_flag(value)


@dataclass(frozen=True)
class MutableKey:
    """One permitted soft-reload target.

    ``kind`` is ``cli_args`` for a serving argument read per request, or
    ``module`` for a model-code lever constant read inside a forward pass.
    """

    kind: str
    attr: str
    validate: Callable[[Any], Any]
    module: Optional[str] = None


# The soft-reload whitelist. Every entry is checked to be read *per request*
# (cli_args) or *per forward* (module), so the change takes effect on the next
# request without rebuilding anything. Nothing outside this registry can be
# written: the reload route never does a bare setattr from a request body.
SOFT_RELOAD_KEYS: Dict[str, MutableKey] = {
    # Self-MTP speculative decoding, read by _self_mtp_config per request.
    "self_mtp": MutableKey("cli_args", "self_mtp", _reload_flag),
    "self_mtp_num_draft": MutableKey(
        "cli_args", "self_mtp_num_draft", _reload_int(1, MAX_DRAFT_TOKENS)
    ),
    "self_mtp_adaptive_depth_ceiling": MutableKey(
        "cli_args",
        "self_mtp_adaptive_depth_ceiling",
        _reload_int(1, MAX_DRAFT_TOKENS, allow_none=True),
    ),
    "self_mtp_persistent": MutableKey(
        "cli_args", "self_mtp_persistent", _reload_flag
    ),
    "self_mtp_rate_gate": MutableKey("cli_args", "self_mtp_rate_gate", _reload_flag),
    "self_mtp_transformed_verifier": MutableKey(
        "cli_args", "self_mtp_transformed_verifier", _reload_flag
    ),
    "self_mtp_share_qsa_indices": MutableKey(
        "cli_args", "self_mtp_share_qsa_indices", _reload_flag
    ),
    "self_mtp_share_qsa_indices_min_prompt_tokens": MutableKey(
        "cli_args",
        "self_mtp_share_qsa_indices_min_prompt_tokens",
        _reload_int(0, 1 << 22),
    ),
    "self_mtp_window_size": MutableKey(
        "cli_args", "self_mtp_window_size", _reload_int(0, 1 << 22)
    ),
    "self_mtp_window_sink_size": MutableKey(
        "cli_args", "self_mtp_window_sink_size", _reload_int(0, 1 << 16)
    ),
    "self_mtp_window_min_prompt_tokens": MutableKey(
        "cli_args", "self_mtp_window_min_prompt_tokens", _reload_int(0, 1 << 22)
    ),
    "self_mtp_apc_retain_min_prompt_tokens": MutableKey(
        "cli_args", "self_mtp_apc_retain_min_prompt_tokens", _reload_int(0, 1 << 22)
    ),
    # Draft-model and prompt-lookup speculation, read per request in do_POST.
    "num_draft_tokens": MutableKey(
        "cli_args", "num_draft_tokens", _reload_int(0, MAX_DRAFT_TOKENS)
    ),
    "prompt_lookup_ngram": MutableKey(
        "cli_args", "prompt_lookup_ngram", _reload_int(0, 16)
    ),
    "prompt_lookup_tokens": MutableKey(
        "cli_args", "prompt_lookup_tokens", _reload_int(0, MAX_DRAFT_TOKENS)
    ),
    "prompt_lookup_adaptive": MutableKey(
        "cli_args", "prompt_lookup_adaptive", _reload_flag
    ),
    "prompt_lookup_rate_gate": MutableKey(
        "cli_args", "prompt_lookup_rate_gate", _reload_flag
    ),
    "prompt_lookup_warmup": MutableKey(
        "cli_args", "prompt_lookup_warmup", _reload_int(0, 1 << 20)
    ),
    "prompt_lookup_gate": MutableKey(
        "cli_args", "prompt_lookup_gate", _reload_float(0.0, 1.0)
    ),
    "prompt_lookup_rate_gate_probe": MutableKey(
        "cli_args", "prompt_lookup_rate_gate_probe", _reload_int(1, 1 << 20)
    ),
    "prompt_lookup_rate_gate_margin": MutableKey(
        "cli_args", "prompt_lookup_rate_gate_margin", _reload_float(-1.0, 1.0)
    ),
    # Server-side sampling and length defaults, read per request in do_POST.
    "temp": MutableKey("cli_args", "temp", _reload_float(0.0, 4.0)),
    "top_p": MutableKey("cli_args", "top_p", _reload_float(0.0, 1.0)),
    "top_k": MutableKey("cli_args", "top_k", _reload_int(0, 1 << 20)),
    "min_p": MutableKey("cli_args", "min_p", _reload_float(0.0, 1.0)),
    "max_tokens": MutableKey("cli_args", "max_tokens", _reload_int(1, 1 << 22)),
    # Model-code levers whose constant is read inside a forward pass. Each was
    # checked to be read per call, not bound at module or layer construction.
    "qwen4_rmsnorm_fast": MutableKey(
        "module", "_RMSNORM_FAST", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_pooled_key_cache": MutableKey(
        "module", "_QSA_POOLED_KEY_CACHE", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_apc_summaries": MutableKey(
        "module", "_QSA_APC_SUMMARIES", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_scatter_chosen": MutableKey(
        "module", "_QSA_SCATTER_CHOSEN", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_dense_shortcircuit": MutableKey(
        "module", "_QSA_DENSE_SHORTCIRCUIT", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_fused_proj": MutableKey(
        "module", "_QSA_FUSED_PROJ", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_nax_kernel": MutableKey(
        "module", "_QSA_NAX_KERNEL", _reload_auto_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_gather_kv": MutableKey(
        "module", "_QSA_GATHER_KV", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_gather_tile_rows": MutableKey(
        "module", "_QSA_GATHER_TILE_ROWS", _reload_int(1, 1 << 10),
        "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_gather_min_context": MutableKey(
        "module", "_QSA_GATHER_MIN_CONTEXT", _reload_int(0, 1 << 22),
        "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_gather_max_context": MutableKey(
        "module", "_QSA_GATHER_MAX_CONTEXT", _reload_int(0, 1 << 22),
        "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_gather_min_query": MutableKey(
        "module", "_QSA_GATHER_MIN_QUERY", _reload_int(1, 1 << 10),
        "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_gather_max_query": MutableKey(
        "module", "_QSA_GATHER_MAX_QUERY", _reload_int(1, 1 << 10),
        "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_qsa_stage1_kernel": MutableKey(
        "module",
        "_QSA_STAGE1_KERNEL",
        _reload_auto_flag,
        "mlx_lm.models.qwen4_exp",
    ),
    "qwen4_qsa_indexed": MutableKey(
        "module",
        "_QSA_INDEXED_ENABLED",
        _reload_auto_flag,
        "mlx_lm.models.qwen4_qsa_indexed",
    ),
    "qwen4_ple_vector_shift": MutableKey(
        "module", "_PLE_VECTOR_SHIFT", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_ple_gather_concat": MutableKey(
        "module", "_PLE_GATHER_CONCAT", _reload_flag, "mlx_lm.models.qwen4_exp"
    ),
    "qwen4_moe_gate_compile": MutableKey(
        "module", "_MOE_GATE_COMPILE", _reload_flag, "mlx_lm.models.qwen3_next"
    ),
}

_RELOAD_NEEDS_MODEL = "requires a model reload; restart the server"
_RELOAD_NEEDS_BUILD = (
    "is bound when the batch generator or the prompt cache is built; "
    "restart the server"
)
_RELOAD_NEEDS_PROCESS = "is a process-level setting; restart the server"
_RELOAD_NEEDS_LAYERS = (
    "changes the built module or weight layout, not just arithmetic; "
    "restart the server"
)

# Keys a caller plausibly wants but which cannot take effect in place. These
# are refused loudly. Silently accepting one and doing nothing would be worse
# than refusing: the caller would measure the old configuration under the new
# label.
SOFT_RELOAD_RESTART_KEYS: Dict[str, str] = {
    "model": _RELOAD_NEEDS_MODEL,
    "adapter_path": _RELOAD_NEEDS_MODEL,
    "draft_model": _RELOAD_NEEDS_MODEL,
    "quantization": _RELOAD_NEEDS_MODEL,
    "quantize": _RELOAD_NEEDS_MODEL,
    "trust_remote_code": _RELOAD_NEEDS_MODEL,
    "chat_template": _RELOAD_NEEDS_MODEL,
    "chat_template_args": _RELOAD_NEEDS_MODEL,
    "use_default_chat_template": _RELOAD_NEEDS_MODEL,
    "single_model": _RELOAD_NEEDS_MODEL,
    "kv_bits": _RELOAD_NEEDS_BUILD,
    "kv_key_bits": _RELOAD_NEEDS_BUILD,
    "kv_value_bits": _RELOAD_NEEDS_BUILD,
    "kv_group_size": _RELOAD_NEEDS_BUILD,
    "quantized_kv_start": _RELOAD_NEEDS_BUILD,
    "decode_concurrency": _RELOAD_NEEDS_BUILD,
    "prompt_concurrency": _RELOAD_NEEDS_BUILD,
    "prefill_step_size": _RELOAD_NEEDS_BUILD,
    "prompt_batch_window": _RELOAD_NEEDS_BUILD,
    "prompt_cache_size": _RELOAD_NEEDS_BUILD,
    "prompt_cache_bytes": _RELOAD_NEEDS_BUILD,
    "state_budget_gb": _RELOAD_NEEDS_BUILD,
    "parallel_sampling_state_budget_gb": _RELOAD_NEEDS_BUILD,
    "host": _RELOAD_NEEDS_PROCESS,
    "port": _RELOAD_NEEDS_PROCESS,
    "pipeline": _RELOAD_NEEDS_PROCESS,
    "int8_prefill": _RELOAD_NEEDS_PROCESS,
    "allowed_origins": _RELOAD_NEEDS_PROCESS,
    "soft_reload_key": _RELOAD_NEEDS_PROCESS,
    # These read their constant in __init__ / at load, so the built model
    # already carries the old layout. Flipping the constant would change only
    # the next model that gets built.
    "qwen4_moe_shared_in_gather": _RELOAD_NEEDS_LAYERS,
    "qwen4_moe_fused_gate_up": _RELOAD_NEEDS_LAYERS,
    "qwen35_gdn_proj_fusion_subclass": _RELOAD_NEEDS_LAYERS,
}


def _soft_reload_target(spec: MutableKey):
    """Resolve a registry entry to the object that owns the attribute."""
    if spec.kind == "module":
        try:
            return importlib.import_module(spec.module)
        except ImportError as exc:
            raise SoftReloadError(
                f"model module {spec.module} is not importable: {exc}"
            ) from exc
    raise SoftReloadError(f"unknown target kind {spec.kind!r}")


def read_effective_config(cli_args) -> Dict[str, Any]:
    """Report the live value of every soft-reloadable key.

    A caller must be able to read back what the server actually runs, not what
    it believes it set. A module lever that is not imported yet reports
    ``None``: no model built from it exists, so it has no live value.
    """
    values = {}
    for name, spec in SOFT_RELOAD_KEYS.items():
        if spec.kind == "cli_args":
            values[name] = getattr(cli_args, spec.attr, None)
            continue
        module = sys.modules.get(spec.module)
        values[name] = None if module is None else getattr(module, spec.attr, None)
    return values


def plan_soft_reload(cli_args, config: Any) -> List[Tuple[str, MutableKey, Any, Any]]:
    """Validate a requested config change without applying any of it.

    Returns ``(name, spec, old, new)`` per key. Everything is validated before
    anything is written, so a rejected request leaves the server untouched and
    never half-applied.
    """
    if not isinstance(config, dict):
        raise SoftReloadError("config must be a JSON object")

    plan = []
    for name, value in config.items():
        if name in SOFT_RELOAD_RESTART_KEYS:
            raise SoftReloadRestartRequired(
                f"'{name}' {SOFT_RELOAD_RESTART_KEYS[name]}"
            )
        spec = SOFT_RELOAD_KEYS.get(name)
        if spec is None:
            raise SoftReloadError(
                f"'{name}' is not a soft-reloadable key. "
                f"GET {EFFECTIVE_CONFIG_PATH} lists the permitted keys."
            )
        try:
            new = spec.validate(value)
        except SoftReloadError as exc:
            raise SoftReloadError(f"'{name}': {exc}") from None
        if spec.kind == "cli_args":
            old = getattr(cli_args, spec.attr, None)
        else:
            old = getattr(_soft_reload_target(spec), spec.attr, None)
        plan.append((name, spec, old, new))
    return plan


def apply_soft_reload(cli_args, plan) -> Dict[str, Dict[str, Any]]:
    """Write a validated plan and report old -> new per key."""
    changes = {}
    for name, spec, old, new in plan:
        target = cli_args if spec.kind == "cli_args" else _soft_reload_target(spec)
        setattr(target, spec.attr, new)
        changes[name] = {"old": old, "new": new}
    return changes


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False

        group = mx.distributed.init()
        self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
        self.tensor_group = (
            group if group.size() > 1 and not cli_args.pipeline else None
        )
        self.is_distributed = group.size() > 1

        # Maps model and adapter paths the actual paths to be used. Used to
        # map 'default_model' to the provided model by cli argument but could
        # be used for more in the future.
        self._model_map = {}
        self._adapter_map = {}
        self._draft_model_map = {}
        self._model_map["default_model"] = self.cli_args.model
        self._adapter_map["default_model"] = self.cli_args.adapter_path
        self._draft_model_map["default_model"] = self.cli_args.draft_model

        # Build the tokenizer config for later use in load
        self._tokenizer_config = {"trust_remote_code": cli_args.trust_remote_code}
        if cli_args.chat_template:
            self._tokenizer_config["chat_template"] = cli_args.chat_template

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if self.is_distributed and (
            adapter_path is not None or draft_model_path is not None
        ):
            raise ValueError(
                "Loading with adapters or draft models not supported in distributed mode"
            )

        # Remove the old model if it exists.
        had_loaded_model = self.model is not None or self.draft_model is not None
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        if had_loaded_model:
            # Drop the int8 prefill overlay's per-module caches before the
            # freed modules' ids can be reused by the replacement model.
            _release_int8_prefill_overlay()
            # Return buffers from the previous model before allocating its replacement.
            mx.clear_cache()

        # Load the model and tokenizer
        if self.is_distributed:
            model, tokenizer = sharded_load(
                model_path,
                pipeline_group=self.pipeline_group,
                tensor_group=self.tensor_group,
                tokenizer_config=self._tokenizer_config,
                trust_remote_code=self.cli_args.trust_remote_code,
            )
        else:
            model, tokenizer = load(
                model_path,
                adapter_path=adapter_path,
                tokenizer_config=self._tokenizer_config,
                trust_remote_code=self.cli_args.trust_remote_code,
            )

        # Use the default chat template if needed
        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        # Load the draft model for speculative decoding
        draft_model = None
        if draft_model_path is not None:
            draft_model, draft_tokenizer = load(draft_model_path)
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # Compute batchability
        is_batchable = draft_model is None
        is_batchable = is_batchable and all(
            hasattr(c, "merge") for c in make_prompt_cache(model)
        )
        # Update the member variables
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.is_batchable = is_batchable

    def load_default(self):
        if self._model_map["default_model"] is not None:
            self.load("default_model", None, "default_model")

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        adapter_path = self._adapter_map.get(model_path, adapter_path)
        model_path = self._model_map.get(model_path, model_path)
        draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

        model_key = (model_path, adapter_path, draft_model_path)
        if self.model_key != model_key:
            self._load(*model_key)

        return self.model, self.tokenizer


_sampler_cache = {}


def _make_sampler(args, tokenizer):
    # Memoize on the sampling parameters so concurrent requests with identical
    # settings share one sampler object. GenerationBatch groups rows by sampler
    # identity, letting a whole batch sample in a single vectorized call.
    xtc_special_tokens = (
        tuple(tokenizer.eos_token_ids),
        tuple(tokenizer.encode("\n")),
    )
    key = (
        xtc_special_tokens,
        args.sampling.temperature,
        args.sampling.top_p,
        args.sampling.top_k,
        args.sampling.min_p,
        args.sampling.xtc_probability,
        args.sampling.xtc_threshold,
    )
    if key in _sampler_cache:
        return _sampler_cache[key]
    if len(_sampler_cache) > 64:
        _sampler_cache.clear()
    sampler = _uncached_make_sampler(args, tokenizer)
    _sampler_cache[key] = sampler
    return sampler


def _uncached_make_sampler(args, tokenizer):
    return make_sampler(
        args.sampling.temperature,
        top_p=args.sampling.top_p,
        top_k=args.sampling.top_k,
        min_p=args.sampling.min_p,
        xtc_probability=args.sampling.xtc_probability,
        xtc_threshold=args.sampling.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )


def _make_logits_processors(args):
    return make_logits_processors(
        args.logits.logit_bias,
        args.logits.repetition_penalty,
        args.logits.repetition_context_size,
        args.logits.presence_penalty,
        args.logits.presence_context_size,
        args.logits.frequency_penalty,
        args.logits.frequency_context_size,
    )


def _request_thinking_enabled(cli_args, chat_template_kwargs=None):
    """Resolve the effective chat-template thinking mode for admission."""
    template_args = dict(getattr(cli_args, "chat_template_args", {}) or {})
    template_args.update(chat_template_kwargs or {})
    return bool(template_args.get("enable_thinking", False))


def _request_sampling_profile(cli_args, chat_template_kwargs=None):
    """Return the configured mode profile; explicit request fields override it."""
    thinking = _request_thinking_enabled(cli_args, chat_template_kwargs)
    name = (
        "thinking_sampling_profile"
        if thinking
        else "nonthinking_sampling_profile"
    )
    return dict(getattr(cli_args, name, None) or {})


def _request_output_ceiling(cli_args, chat_template_kwargs=None):
    """Return a mode-specific admission ceiling, never a generation default."""
    thinking = _request_thinking_enabled(cli_args, chat_template_kwargs)
    name = "thinking_output_ceiling" if thinking else "nonthinking_output_ceiling"
    return getattr(cli_args, name, None)


def _make_lane_rng(args, root, sidecar=None):
    """Build this request's decode lane key.

    A lane draws from its own key, never from the global ``mx.random`` stream,
    so its tokens do not depend on the traffic decoded beside it.

    * A resumed request continues the stream carried by its APC sidecar, so it
      does not repeat the draws the earlier turn already made. Two requests
      that hit the SAME stored sidecar therefore continue the same stream:
      the snapshot is a position, and equal inputs give equal outputs.
    * An explicit request ``seed`` reproduces one lane exactly.
    * Otherwise the lane is forked from the server root, which advances, so
      concurrent requests never share a key.
    """
    carried = getattr(sidecar, "rng_key", None)
    if carried is not None:
        # Rebuild the carried KEY on this thread. Evaluating a descendant of a
        # lazy key created by another thread does not remove its stream ancestry.
        carried = mx.array(carried.tolist(), dtype=carried.dtype)
        lane = LaneRNG.from_key(
            carried, int(getattr(sidecar, "rng_draws", 0) or 0)
        )
    else:
        seed = getattr(args, "seed", None)
        if seed is None and root is None:
            raise RuntimeError("generation-thread lane RNG root is not initialized")
        lane = LaneRNG(int(seed)) if seed is not None else root.fork(1)[0]
    # This helper is called by the generation thread.  Materializing here is
    # intentional: a key lazily created or restored on the HTTP/main thread
    # must not first execute on a different Metal stream inside generation.
    mx.eval(lane.key)
    return lane


def _make_generation_thread_lane_rng_root():
    """Create and materialize the unseeded root on the generation thread."""
    key = mx.random.split(mx.random.state[0])[1]
    root = LaneRNG.from_key(key)
    mx.eval(root.key)
    return root


def _self_mtp_config(
    args,
    cli_args,
    model,
    *,
    cached_prompt_tokens=0,
    prompt_tokens=0,
    mtp_state=None,
    lane_rng=None,
):
    """Return an exact self-MTP route or fail closed to ordinary decoding.

    The engine is exact for greedy and temperature-only sampling. With
    ``--self-mtp-transformed-verifier`` it is also exact for top-p/top-k/min-p
    sampling: the same transform is applied to draft and target
    log-probabilities before residual acceptance. XTC (a stochastic transform)
    is not implemented, so it fails closed whenever it could engage
    (``temperature > 0``); at ``temperature == 0`` the sampler is argmax and
    XTC never engages, so such requests stay admitted as greedy. Without the
    flag every transformed request remains on the ordinary sampler. Likewise,
    an APC hit has target state but no matching MTP hidden/KV state; it keeps
    the valuable prefix hit and decodes plainly.
    """
    if not getattr(cli_args, "self_mtp", False):
        return None
    if getattr(model, "mtp", None) is None or (
        cached_prompt_tokens and mtp_state is None
    ):
        return None
    if args.model.draft != "default_model" or args.prompt_lookup_ngram:
        return None
    sampling = args.sampling
    # ``top_p <= 0`` is a sampler no-op (make_sampler skips the filter), so
    # it is classified untransformed and keeps the temperature-only path.
    transformed_sampling = (
        sampling.temperature > 0
        and (
            0.0 < sampling.top_p < 1.0
            or sampling.top_k > 0
            or sampling.min_p > 0.0
            or sampling.xtc_probability > 0.0
        )
    )
    transformed_verifier = transformed_sampling and getattr(
        cli_args, "self_mtp_transformed_verifier", False
    )
    if transformed_sampling and (
        not transformed_verifier or sampling.xtc_probability > 0.0
    ):
        return None
    quantized_kv = getattr(cli_args, "kv_bits", None) is not None
    allow_quantized_kv = getattr(cli_args, "self_mtp_allow_quantized_kv", False)
    if quantized_kv and not allow_quantized_kv:
        # Quantized KV self-MTP is opt-in: the batched transaction is bit-exact
        # on a quantized target cache (proven via the batched-B1 oracle), but it
        # stays behind --self-mtp-allow-quantized-kv until promoted by default.
        return None
    share_qsa_minimum = getattr(
        cli_args, "self_mtp_share_qsa_indices_min_prompt_tokens", 0
    )
    config = {
        "num_draft": cli_args.self_mtp_num_draft,
        "persistent": cli_args.self_mtp_persistent,
        "rate_gate": cli_args.self_mtp_rate_gate,
        "share_qsa_indices": (
            getattr(cli_args, "self_mtp_share_qsa_indices", False)
            and prompt_tokens >= share_qsa_minimum
        ),
        "sampling_temp": sampling.temperature,
        "accept_rule": "residual",
        "state_out": {},
    }
    if quantized_kv:
        # Tag so the BatchGenerator constructor admits the quantized cache; the
        # cache is already built quantized by _make_new_cache.
        config["allow_quantized_kv"] = True
    if transformed_verifier:
        config["top_p"] = sampling.top_p
        config["top_k"] = sampling.top_k
        config["min_p"] = sampling.min_p
    depth_ceiling = getattr(cli_args, "self_mtp_adaptive_depth_ceiling", None)
    if depth_ceiling is not None:
        # The configured depth becomes the floor; the ceiling is reached only
        # on sustained measured acceptance. The controller is per-request
        # state: fresh at admission, carried across cycles, never persisted
        # into APC sidecars.
        config["num_draft"] = int(depth_ceiling)
        config["speculation_router"] = DepthCeilingController(
            cli_args.self_mtp_num_draft, int(depth_ceiling)
        )
    if mtp_state is not None:
        config["state"] = mtp_state
    if lane_rng is not None:
        config["lane_rng"] = lane_rng
    window_size = getattr(cli_args, "self_mtp_window_size", 0)
    window_minimum = getattr(cli_args, "self_mtp_window_min_prompt_tokens", 0)
    if (
        window_size
        and cli_args.self_mtp_persistent
        and prompt_tokens >= window_minimum
    ):
        config["window_size"] = window_size
        config["sink_size"] = getattr(cli_args, "self_mtp_window_sink_size", 4)
    return config


@dataclass(frozen=True)
class SelfMTPLaneAdmission:
    """One cycle-boundary memory decision for batched self-MTP.

    ``modes`` and ``draft_depths`` are aligned with the caller's lane order.
    An MTP lane has depth 1 or 2, a plain lane has depth 0, and a queued lane
    has depth ``None``.  The decision is immutable so the exact budget checked
    for a cycle can be logged or asserted without later membership changes
    rewriting it.
    """

    modes: Tuple[Literal["self_mtp", "plain", "queue"], ...]
    draft_depths: Tuple[Optional[int], ...]
    stage: Literal["full", "fewer_lanes", "lower_k", "plain", "queue"]
    estimated_gib: float
    usable_gib: float

    @property
    def mtp_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, mode in enumerate(self.modes) if mode == "self_mtp")


class SelfMTPLaneAdmissionController:
    """Fail-closed memory/context policy for the M=(k+1)N verify forward.

    The production PLE-offload operating point is about 72.5 GiB resident on
    a 128 GiB host, leaving about 55.5 GiB free.  A 20 GiB hard margin (16 GiB
    service reserve plus 4 GiB for the driver) leaves 35.5 GiB for lane cache
    growth and the verify transient.  The linear envelope below is calibrated
    so that this operating point admits N=16 around 1K and N=4 around 16K:

      cache/context share: 0.44 GiB per 1K tokens per lane
      k=2 verify transient: 1.76 GiB per lane

    Reducing k from 2 to 1 halves only the verify transient.  Plain M=N keeps
    one third of the k=2 verify transient.  This is deliberately an envelope,
    not a claim about allocator internals.  Inputs that cannot be measured are
    queued rather than guessed.

    Memory is not the only ceiling.  The M=(k+1)N verify forward saturates the
    GPU near a fixed lane count; past it, aggregate throughput falls even when
    memory permits more lanes.  A dense Qwen3.8-27B (~18 GiB resident) leaves
    ~95 GiB free, so the memory envelope alone would admit N~40 at short
    context -- 2.5x past the measured throughput peak (N=16: 270 t/s agg;
    N=40: 52 t/s).  ``SATURATION_LANE_CAP`` bounds the admitted subset so the
    N~16 knee holds regardless of how much memory is free.  At Flash-Next's
    72.5 GiB PLE operating point the memory envelope already caps near 16 at
    1K, so this cap is a no-op there and only binds when free memory is large.
    The k=2 transient is calibrated on Flash-Next (MoE, 6B active); a dense
    27B measures ~3.1 GiB/lane, so ``transient_gib_per_lane`` is configurable
    for dense deployments at long context where the transient dominates.

    ``decide`` is stateless on purpose.  The server calls it at every decode
    cycle boundary with fresh free memory and current per-lane contexts, so
    lane joins/leaves and cache growth are always reflected in the next plan.
    """

    BASE_RESIDENT_GIB = 72.5
    HOST_MEMORY_GIB = 128.0
    SERVICE_RESERVE_GIB = 16.0
    DRIVER_ALLOWANCE_GIB = 4.0
    CACHE_GIB_PER_1K_TOKENS = 0.44
    K2_TRANSIENT_GIB_PER_LANE = 1.76
    SATURATION_LANE_CAP = 16

    def __init__(
        self,
        *,
        service_reserve_gib: float = SERVICE_RESERVE_GIB,
        driver_allowance_gib: float = DRIVER_ALLOWANCE_GIB,
        transient_gib_per_lane: float = K2_TRANSIENT_GIB_PER_LANE,
        saturation_lane_cap: Optional[int] = SATURATION_LANE_CAP,
    ):
        if service_reserve_gib < self.SERVICE_RESERVE_GIB:
            raise ValueError("self-MTP service reserve must be at least 16 GiB")
        if driver_allowance_gib < 0:
            raise ValueError("self-MTP driver allowance must be non-negative")
        if not math.isfinite(transient_gib_per_lane) or transient_gib_per_lane <= 0:
            raise ValueError("self-MTP transient GiB per lane must be positive")
        if saturation_lane_cap is not None and (
            isinstance(saturation_lane_cap, bool)
            or not isinstance(saturation_lane_cap, int)
            or saturation_lane_cap < 1
        ):
            raise ValueError("self-MTP saturation lane cap must be a positive int or None")
        self.service_reserve_gib = float(service_reserve_gib)
        self.driver_allowance_gib = float(driver_allowance_gib)
        self.transient_gib_per_lane = float(transient_gib_per_lane)
        self.saturation_lane_cap = saturation_lane_cap

    @property
    def hard_reserve_gib(self) -> float:
        return self.service_reserve_gib + self.driver_allowance_gib

    def lane_gib(
        self, context_tokens: int, draft_depth: int, cache_gib: float = 0.0
    ) -> float:
        """Conservative cache plus transient cost for one lane."""
        if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
            raise ValueError("context_tokens must be an integer")
        if context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        if draft_depth not in (0, 1, 2):
            raise ValueError("draft_depth must be 0, 1, or 2")
        cache_gib = float(cache_gib)
        if not math.isfinite(cache_gib) or cache_gib < 0:
            raise ValueError("cache_gib must be finite and non-negative")
        context_gib = max(
            self.CACHE_GIB_PER_1K_TOKENS * (context_tokens / 1024.0),
            cache_gib,
        )
        transient_scale = {0: 1.0 / 3.0, 1: 0.5, 2: 1.0}[draft_depth]
        return context_gib + self.transient_gib_per_lane * transient_scale

    def _fit(
        self,
        indices: Sequence[int],
        contexts: Sequence[int],
        cache_gib: Sequence[float],
        draft_depth: int,
        usable_gib: float,
        max_lanes: Optional[int] = None,
    ) -> Tuple[Tuple[int, ...], float]:
        # Admit the cheapest lanes first; retain their original relative order
        # in the returned batch.  One long request therefore cannot force a
        # wider unsafe M=3N forward or evict several short safe lanes.  The
        # compute-saturation cap stops admitting once ``max_lanes`` cheapest
        # lanes fit, so a large free-memory envelope cannot widen the M=(k+1)N
        # forward past the throughput knee.
        ranked = sorted(
            indices,
            key=lambda i: (
                self.lane_gib(contexts[i], draft_depth, cache_gib[i]),
                i,
            ),
        )
        chosen = []
        used = 0.0
        for i in ranked:
            if max_lanes is not None and len(chosen) >= max_lanes:
                break
            cost = self.lane_gib(contexts[i], draft_depth, cache_gib[i])
            if used + cost <= usable_gib:
                chosen.append(i)
                used += cost
        return tuple(sorted(chosen)), used

    def decide(
        self,
        context_tokens: Sequence[int],
        free_memory_gib: float,
        *,
        eligible: Optional[Sequence[bool]] = None,
        cache_gib: Optional[Sequence[float]] = None,
        max_draft: int = 2,
    ) -> SelfMTPLaneAdmission:
        """Return the next-cycle plan in the frozen degradation order.

        Excluded lanes route directly to plain decode.  For eligible lanes the
        controller tries, in order: the largest safe k=2 subset; if no k=2
        lane fits, the largest safe k=1 subset; if none fits, one safe plain
        lane; otherwise queue.  Thus lowering k never jumps ahead of admitting
        fewer full-depth lanes.
        """
        contexts = tuple(context_tokens)
        if eligible is None:
            eligible = (True,) * len(contexts)
        else:
            eligible = tuple(eligible)
        if len(eligible) != len(contexts):
            raise ValueError("eligible must align with context_tokens")
        if cache_gib is None:
            cache_gib = (0.0,) * len(contexts)
        else:
            cache_gib = tuple(cache_gib)
        if len(cache_gib) != len(contexts):
            raise ValueError("cache_gib must align with context_tokens")
        if max_draft not in (1, 2):
            raise ValueError("max_draft must be 1 or 2")

        modes: List[Literal["self_mtp", "plain", "queue"]] = [
            "plain" if not ok else "queue" for ok in eligible
        ]
        depths: List[Optional[int]] = [0 if not ok else None for ok in eligible]
        mtp_candidates = [i for i, ok in enumerate(eligible) if ok]
        if not mtp_candidates:
            return SelfMTPLaneAdmission(
                tuple(modes), tuple(depths), "plain", 0.0, 0.0
            )

        try:
            free = float(free_memory_gib)
            valid = math.isfinite(free) and free >= 0
            # Validate every candidate before choosing a subset.  Partially
            # trusting a malformed context vector would make the estimate
            # lane-order-dependent and is not fail closed.
            for i in mtp_candidates:
                self.lane_gib(contexts[i], 2, cache_gib[i])
        except (TypeError, ValueError, OverflowError):
            valid = False
            free = 0.0
        usable = max(free - self.hard_reserve_gib, 0.0) if valid else 0.0
        if not valid:
            return SelfMTPLaneAdmission(
                tuple(modes), tuple(depths), "queue", 0.0, usable
            )

        if max_draft == 2:
            chosen, used = self._fit(
                mtp_candidates, contexts, cache_gib, 2, usable,
                self.saturation_lane_cap,
            )
            if chosen:
                for i in chosen:
                    modes[i] = "self_mtp"
                    depths[i] = 2
                stage = (
                    "full"
                    if len(chosen) == len(mtp_candidates)
                    else "fewer_lanes"
                )
                return SelfMTPLaneAdmission(
                    tuple(modes), tuple(depths), stage, used, usable
                )

        chosen, used = self._fit(
            mtp_candidates, contexts, cache_gib, 1, usable,
            self.saturation_lane_cap,
        )
        if chosen:
            for i in chosen:
                modes[i] = "self_mtp"
                depths[i] = 1
            stage = (
                "lower_k"
                if max_draft == 2
                else "full" if len(chosen) == len(mtp_candidates) else "fewer_lanes"
            )
            return SelfMTPLaneAdmission(tuple(modes), tuple(depths), stage, used, usable)

        chosen, used = self._fit(mtp_candidates, contexts, cache_gib, 0, usable)
        if chosen:
            # The degradation contract says "drop a lane to plain" before
            # queue/reject.  Admit only the cheapest lane here; the remainder
            # stays queued for a later cycle rather than widening M=N without
            # a fresh batch-level budget check.
            i = chosen[0]
            modes[i] = "plain"
            depths[i] = 0
            used = self.lane_gib(contexts[i], 0, cache_gib[i])
            return SelfMTPLaneAdmission(
                tuple(modes), tuple(depths), "plain", used, usable
            )

        return SelfMTPLaneAdmission(
            tuple(modes), tuple(depths), "queue", 0.0, usable
        )


def _system_available_memory_bytes() -> Optional[int]:
    """Return reclaimable system memory without using allocator headroom."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            match = re.search(r"page size of (\d+) bytes", result.stdout)
            if match is None:
                return None
            page_size = int(match.group(1))
            counts = {}
            for line in result.stdout.splitlines()[1:]:
                if ":" not in line:
                    continue
                name, value = line.split(":", 1)
                counts[name] = int(value.strip().rstrip("."))
            pages = sum(
                counts.get(name, 0)
                for name in ("Pages free", "Pages inactive", "Pages speculative")
            )
            return pages * page_size if pages > 0 else None
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size if pages > 0 and page_size > 0 else None
    except (
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


def _current_self_mtp_free_memory_gib() -> Optional[float]:
    """Return actual available system memory for this admission boundary."""
    available = _system_available_memory_bytes()
    if available is None or available <= 0:
        return None
    return available / float(1 << 30)


def _make_self_mtp_admission_callback(
    controller: Optional[SelfMTPLaneAdmissionController] = None,
    free_memory: Callable[[], Optional[float]] = _current_self_mtp_free_memory_gib,
    *,
    max_draft: int = 2,
) -> Callable[
    [Sequence[Tuple[int, int, int, bool, float]]],
    Mapping[int, Union[int, str]],
]:
    """Adapt the pure controller to the generator's cycle-boundary seam.

    The generator calls this before every proposal, when no transaction is
    open.  It owns the resulting detach/pause/plain migration; the server owns
    the policy and the live memory measurement.  Rows are
    ``(uid, logical_context_len, current_k, active, cache_gib)`` and include
    paused or joining lanes, so retained cache rows participate before merge.
    """
    controller = controller or SelfMTPLaneAdmissionController()

    def admit(rows):
        rows = tuple(rows)
        if not rows:
            return {}
        current_free = free_memory()
        decision = controller.decide(
            [int(row[1]) for row in rows],
            math.nan if current_free is None else current_free,
            cache_gib=[float(row[4]) if len(row) > 4 else 0.0 for row in rows],
            max_draft=max_draft,
        )
        actions: Dict[int, Union[int, str]] = {}
        for row, mode, depth in zip(rows, decision.modes, decision.draft_depths):
            uid = int(row[0])
            actions[uid] = int(depth) if mode == "self_mtp" else mode
        return actions

    return admit


def _batched_self_mtp_config(
    args,
    cli_args,
    model,
    *,
    cached_prompt_tokens=0,
    prompt_tokens=0,
    mtp_state=None,
    lane_rng=None,
):
    """Return a fixed-depth persistent config eligible for an MTP batch.

    These exclusions are intentionally stricter than single-lane self-MTP.
    Every uncertain or unsupported combination falls back to the plain batch
    kind; in particular shared-QSA remains excluded even after its padded-query
    seam lands, until that mode is promoted separately.
    """
    config = _self_mtp_config(
        args,
        cli_args,
        model,
        cached_prompt_tokens=cached_prompt_tokens,
        prompt_tokens=prompt_tokens,
        mtp_state=mtp_state,
        lane_rng=lane_rng,
    )
    if config is None:
        return None
    if not config.get("persistent"):
        return None
    if config.get("window_size") is not None:
        return None
    if config.get("rate_gate"):
        return None
    if config.get("speculation_router") is not None:
        return None
    if config.get("share_qsa_indices"):
        return None
    if config.get("num_draft") not in (1, 2):
        return None
    return config


def _batch_kind_key(model_identity, self_mtp=None):
    """The exact cohort key: plain and self-MTP lanes never mix."""
    if self_mtp is None:
        return (model_identity, "plain")
    return (
        model_identity,
        "self_mtp",
        bool(self_mtp["persistent"]),
        int(self_mtp["num_draft"]),
        self_mtp.get("window_size", "native"),
        bool(self_mtp.get("share_qsa_indices", False)),
    )


def _batched_kv_quantization(cli_args, self_mtp):
    """Return immediate KV quantization settings for an opted-in MTP batch."""
    if self_mtp is None or not self_mtp.get("allow_quantized_kv"):
        return {}
    kv_bits = getattr(cli_args, "kv_bits", None)
    if kv_bits is None:
        return {}
    return {
        "kv_bits": kv_bits,
        "kv_group_size": getattr(cli_args, "kv_group_size", 64),
    }


def _batched_prompt_cache_model_key(model_identity, cli_args, self_mtp):
    """Separate opted-in quantized APC entries from full-precision entries."""
    quantization = _batched_kv_quantization(cli_args, self_mtp)
    if not quantization:
        return model_identity
    return (
        model_identity,
        "batched_quantized_kv",
        quantization["kv_bits"],
        quantization["kv_group_size"],
    )


def _quantize_batched_self_mtp_cache(prompt_cache, cli_args, self_mtp):
    quantization = _batched_kv_quantization(cli_args, self_mtp)
    if quantization:
        maybe_quantize_kv_cache(
            prompt_cache,
            0,
            quantization["kv_group_size"],
            quantization["kv_bits"],
        )
    return prompt_cache


PARALLEL_SAMPLING_MTP_MODES = ("mtp", "plain", "refuse")


def _parallel_sampling_route(
    args,
    cli_args,
    model,
    *,
    prompt_tokens=0,
    cached_prompt_tokens=0,
    mtp_state=None,
    has_logits_processors=False,
):
    """Choose the n-way plain or persistent batched-self-MTP engine.

    Eligibility is evaluated after the shared APC lookup.  A sidecar-less
    target-cache hit and every frozen exclusion stay plain.  ``refuse`` and
    ``plain`` remain explicit compatibility policies; the default is now the
    implemented MTP path rather than the old global refusal.
    """
    mode = getattr(cli_args, "parallel_sampling_mtp", "mtp")
    config = _batched_self_mtp_config(
        args,
        cli_args,
        model,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        mtp_state=mtp_state,
    )
    if config is None:
        return "plain", None
    if has_logits_processors:
        return "plain", "logits processors require fail-closed plain decode"
    if mode == "plain":
        return (
            "plain",
            "self-MTP disabled for this request by parallel-sampling policy",
        )
    if mode == "refuse":
        raise RequestCompositionError(
            "n>1 self-MTP was refused by --parallel-sampling-mtp refuse. "
            "Use 'mtp' for batched self-MTP or 'plain' for ordinary decode."
        )
    return "self_mtp", None


def _cache_state_bytes(prompt_cache):
    """Bytes held by one row of this prompt cache, plus its token offset."""
    nbytes = 0
    offset = 0
    for leaf in _walk_cache_entries(prompt_cache):
        nbytes += int(getattr(leaf, "nbytes", 0))
        offset = max(offset, int(getattr(leaf, "offset", 0)))
    return nbytes, offset


def _parallel_sampling_state_bytes(prompt_cache, n, prompt_tokens, max_tokens):
    """Project the state an ``n>1`` request needs, or ``None`` if unmeasurable.

    ``n>1`` replicates the whole prefix cache into one row per sample and each
    row then grows for its own completion, so the count cap alone is not a
    memory bound. The per-token cost is measured on the cache in hand; the
    source cache stays alive as the shared prefix entry, hence ``n + 1``.
    """
    nbytes, offset = _cache_state_bytes(prompt_cache)
    if offset <= 0 or nbytes <= 0:
        return None
    per_token = nbytes / offset
    prefix_bytes = per_token * max(prompt_tokens, offset)
    return int((n + 1) * prefix_bytes + n * per_token * max(max_tokens, 0))


def _parallel_self_mtp_required_gib(
    controller, prompt_cache, n, prompt_tokens, draft_depth, mtp_state=None
):
    """Budget source retention, n lane caches, and M=(k+1)N transient.

    ``mtp_state`` is the request's restored draft sidecar, when present. Every
    MTP lane (``draft_depth > 0``) also carries its own draft cache, so an
    additive per-lane draft floor is included: the sidecar's measured bytes
    scaled to the full prompt when available, else the single-MTP-layer share
    of the projected per-row target cost (measured or envelope, whichever is
    larger — a cache miss measures 0 but still allocates a draft cache).
    Plain projections (``draft_depth == 0``) allocate no draft rows and take
    no draft floor.
    """
    nbytes, offset = _cache_state_bytes(prompt_cache)
    measured = (nbytes / offset * prompt_tokens / float(1 << 30)) if offset else 0.0
    envelope = controller.CACHE_GIB_PER_1K_TOKENS * (prompt_tokens / 1024.0)
    cache_per_row = max(measured, envelope)
    draft_per_row = 0.0
    if draft_depth > 0:
        draft_bytes = 0
        draft_offset = 0
        if mtp_state is not None:
            for leaf in mtp_state[0]:
                draft_bytes += int(getattr(leaf, "nbytes", 0))
                draft_offset = max(draft_offset, int(getattr(leaf, "offset", 0)))
        if draft_offset > 0:
            draft_per_row = (
                draft_bytes / draft_offset * prompt_tokens / float(1 << 30)
            )
        else:
            layers = max(len(list(prompt_cache or ())), 1)
            draft_per_row = cache_per_row / layers
    transient = controller.lane_gib(0, draft_depth)
    return (n + 1) * cache_per_row + n * (draft_per_row + transient)


def _parallel_prompt_cache_key(prompt, prompt_cache, self_mtp):
    """Return the exact token span covered by the retained source cache."""
    if self_mtp is None:
        return list(prompt[:-1])
    _, covered = _cache_state_bytes(prompt_cache)
    if not 0 < covered <= len(prompt):
        raise RuntimeError("parallel self-MTP source cache has an invalid prompt cursor")
    return list(prompt[:covered])


def _state_budget_bytes(cli_args):
    """The ceiling an ``n>1`` request must project under, or ``None``.

    ``--parallel-sampling-state-budget-gb`` sets it explicitly. Otherwise it is
    the device's remaining recommended working set, which is what an OOM kill
    would hit.
    """
    explicit = getattr(cli_args, "parallel_sampling_state_budget_gb", None)
    if explicit:
        return int(float(explicit) * (1 << 30))
    if not mx.metal.is_available():
        return None
    limit = mx.device_info().get("max_recommended_working_set_size")
    if not limit:
        return None
    headroom = int(limit) - int(mx.get_active_memory())
    return max(int(0.9 * headroom), 0)


def _discard_small_sidecarless_apc_hit_for_mtp(
    cli_args, model, cached_prompt_tokens, mtp_sidecar
):
    """Prefer MTP over a trivial APC hit that cannot restore draft state."""
    retain_minimum = getattr(
        cli_args, "self_mtp_apc_retain_min_prompt_tokens", 64
    )
    return bool(
        getattr(cli_args, "self_mtp", False)
        and getattr(model, "mtp", None) is not None
        and mtp_sidecar is None
        and 0 < cached_prompt_tokens < retain_minimum
    )


def _segment_by_state(sm_state, text):
    """Advance a ``TextStateMachine`` one character at a time so emitted text is
    attributed to the state it was actually produced in, rather than to the
    chunk's single final state.

    A decoded token can merge body bytes with a control marker (e.g. a token
    that decodes to ``"}</tool_call>"``). ``TextStateMachine.step`` returns the
    whole chunk's emittable text plus one final-state label, so the ``"}"``
    would be labelled ``normal`` and leak to content while going missing from
    the tool text. Feeding the machine char by char preserves the same total
    emittable text and buffer semantics, but splits it into per-state segments
    and also surfaces state transitions that emit no text.

    Returns ``(new_sm_state, segments)`` where ``segments`` is a list of
    ``(emitted_text, state_name)``; a segment may carry empty text when it only
    marks a transition (so callers can flush per-state buffers on the boundary).
    """
    segments = []
    prev = sm_state[0]
    for ch in text:
        sm_state, emitted, cur = TextStateMachine.step(sm_state, ch)
        if emitted or cur != prev:
            segments.append((emitted, cur))
        prev = cur
        if sm_state[0] is None:
            break
    return sm_state, segments


class _ChoiceAssembler:
    """Accumulate one choice's text, tool calls and logprobs from raw tokens.

    One instance per OpenAI choice. With ``n>1`` the samples interleave on one
    response stream, so every piece of assembly state -- the text state machine,
    the tool buffer, the token list -- has to be per choice.
    """

    def __init__(self, index, ctx, tool_formatter, logprobs=False, top_logprobs=0):
        self.index = index
        self.text_sm = ctx.text_sm
        self.tool_formatter = tool_formatter
        self.sm_state = ctx.text_sm.make_state(ctx.initial_state)
        self.prev_state = ctx.initial_state
        self.finish_reason = "stop"
        self.reasoning_text = ""
        self.made_tool_call = False
        self.tool_text = ""
        self.tool_calls = []
        self.text = ""
        self.tokens = []
        self.token_logprobs = []
        self.top_tokens = []
        self.terminal_sent = False
        self._finalized = False
        self._want_logprobs = logprobs
        self._want_top_logprobs = top_logprobs

    def feed(self, gen):
        """Consume one token. Returns the current state name."""
        # Attribute emitted text per state segment (a decoded token can merge
        # body bytes with a marker, e.g. "}</tool_call>"), rather than routing
        # the whole chunk by its single final state.
        if gen.finish_reason == "stop":
            self.sm_state, _ = TextStateMachine.discard(self.sm_state)
            segments = []
        elif gen.finish_reason == "length":
            self.sm_state, segments = _segment_by_state(self.sm_state, gen.text)
            self.sm_state, flushed, flush_state = TextStateMachine.flush(self.sm_state)
            if flushed:
                segments.append((flushed, flush_state))
        else:
            self.sm_state, segments = _segment_by_state(self.sm_state, gen.text)
        current_state = self.sm_state[0]

        # Collect the clean text by state: reasoning, tool, or normal.
        for seg_text, seg_state in segments:
            if seg_state == "reasoning":
                self.reasoning_text += seg_text
            elif seg_state == "tool":
                self.tool_text += seg_text
            elif seg_state == "normal":
                if self.prev_state == "tool":
                    self.tool_calls.append(self.tool_text)
                    self.tool_text = ""
                    self.made_tool_call = True
                self.text += seg_text
            self.prev_state = seg_state

        self.tokens.append(gen.token)
        if self._want_logprobs:
            self.token_logprobs.append(gen.logprob)
        if self._want_top_logprobs > 0:
            self.top_tokens.append(gen.top_tokens)

        if gen.finish_reason is not None:
            self.finish_reason = gen.finish_reason

        self.prev_state = current_state
        return current_state

    def has_pending_stream_text(self):
        return bool(self.text or self.tool_calls or self.reasoning_text)

    def take_stream_payload(self):
        """Return the text/tool/reasoning accumulated since the last chunk."""
        payload = (self.text, self.tool_formatter(self.tool_calls), self.reasoning_text)
        self.reasoning_text = ""
        self.text = ""
        self.tool_calls = []
        return payload

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        if self.prev_state == "tool" and self.tool_text:
            self.tool_calls.append(self.tool_text)
            self.made_tool_call = True
        if self.finish_reason == "stop" and self.made_tool_call:
            self.finish_reason = "tool_calls"


def _format_top_logprobs(logprobs, top_n, tokenizer) -> Tuple[Dict[str, Any]]:
    """Returns info dicts for the top `top_n` tokens from `logprobs`"""
    if top_n <= 0:
        return ()
    sorted_indices = mx.argpartition(-logprobs, kth=top_n - 1)
    top_indices = sorted_indices[:top_n].tolist()
    top_probs = logprobs[top_indices].tolist()
    txts = tokenizer.convert_ids_to_tokens(top_indices)
    return tuple(
        {"id": i, "token": s, "logprob": g}
        for i, s, g in zip(top_indices, txts, top_probs)
    )


def _fetch_single_request_prompt_cache(
    prompt_cache, model_key, prompt, *, external_draft: bool
):
    """Fetch reusable state for one request, unless it is a model-draft pair.

    ``speculative_generate_step`` intentionally ends with the target cache one
    token behind the emitted stream and may leave the draft cache one token
    further behind. The current prompt-cache format stores only one token key
    and has no way to persist that pair-specific coverage/debt. Reusing such an
    entry can therefore omit committed context from both models. Fail closed
    until paired cache entries carry an explicit shared-coverage contract.
    """
    if external_draft:
        logging.info(
            "External-draft speculative decoding bypasses prompt-cache lookup "
            "because paired target/draft coverage is not persisted"
        )
        return None, prompt, None
    if hasattr(prompt_cache, "lookup"):
        lookup = prompt_cache.lookup(model_key, prompt)
        return lookup.cache, lookup.remaining_tokens, lookup.sidecar
    cache, rest = prompt_cache.fetch_nearest_cache(model_key, prompt)
    return cache, rest, None


def _store_single_request_prompt_cache(
    prompt_cache,
    model_key,
    tokens,
    cache,
    *,
    sidecar=None,
    external_draft: bool,
):
    """Store one request's cache when its token key fully describes the state."""
    if external_draft:
        logging.info(
            "External-draft speculative decoding bypasses prompt-cache storage "
            "because paired target/draft coverage is not persisted"
        )
        return False
    if isinstance(prompt_cache, AutomaticPrefixCache):
        prompt_cache.insert_cache(model_key, tokens, cache, sidecar=sidecar)
    else:
        prompt_cache.insert_cache(model_key, tokens, cache)
    return True


class ResponseGenerator:
    def __init__(self, model_provider: ModelProvider, prompt_cache: LRUPromptCache):
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache
        self.requests = Queue()
        self._state_machine_cache = {}
        # The generation thread creates and evaluates this root before use.
        self._lane_rng_root = None
        _cli = self.model_provider.cli_args
        _max_lanes = int(getattr(_cli, "self_mtp_max_lanes", 0) or 0)
        _transient = getattr(_cli, "self_mtp_lane_transient_gib", None)
        self._self_mtp_admission_controller = SelfMTPLaneAdmissionController(
            saturation_lane_cap=(
                _max_lanes if _max_lanes >= 1
                else SelfMTPLaneAdmissionController.SATURATION_LANE_CAP
            ),
            transient_gib_per_lane=(
                float(_transient) if _transient
                else SelfMTPLaneAdmissionController.K2_TRANSIENT_GIB_PER_LANE
            ),
        )
        self._self_mtp_admission = _make_self_mtp_admission_callback(
            self._self_mtp_admission_controller,
            max_draft=min(
                int(getattr(self.model_provider.cli_args, "self_mtp_num_draft", 2)),
                2,
            ),
        )

        # Soft-reload admission gate. ``_paused`` closes the door while a
        # reload swaps configuration; ``_inflight`` counts requests that have
        # been handed to the generation thread and not yet finished.
        self._admission = Condition()
        self._paused = False
        self._inflight = 0
        self._reload_lock = Lock()
        self.admission_timeout = DEFAULT_SOFT_RELOAD_ADMISSION_TIMEOUT

        self._time_budget = TimeBudget()
        self._is_distributed = mx.distributed.init().size() > 1
        self._rank = mx.distributed.init().rank()
        self._stop = False
        self._generation_error = None
        self._generation_thread = Thread(target=self._run_generation)
        self._generation_thread.start()

    def _run_generation(self):
        """Thread body. Keeps the reason generation stopped, so that
        health_report can name it instead of only reporting a dead thread."""
        try:
            self._lane_rng_root = _make_generation_thread_lane_rng_root()
            self._generate()
        except BaseException as e:
            self._generation_error = e
            logging.error("Generation thread stopped: %r", e, exc_info=True)

    def stop_and_join(self):
        self._stop = True
        self._generation_thread.join()

    def join(self):
        self._generation_thread.join()

    def health_report(self) -> Dict[str, Any]:
        """Report whether generation is still running.

        Only an unrequested exit is a fault. A soft reload closes admission
        and a shutdown stops the thread on purpose, so both stay healthy: a
        supervisor must not kill a server that does what it was told.
        """
        thread = getattr(self, "_generation_thread", None)
        stopping = getattr(self, "_stop", False)
        # A plain bool read. The health path never waits on the gate.
        paused = getattr(self, "_paused", False)

        # No thread means none was ever started (a generator built for a
        # test). Nothing died, so there is nothing to report as dead.
        if thread is None or thread.is_alive():
            report = {"healthy": True, "status": "ok"}
            if stopping:
                report["state"] = "stopping"
            elif paused:
                report["state"] = "reloading"
            return report

        if stopping:
            return {"healthy": True, "status": "ok", "state": "stopped"}

        error = getattr(self, "_generation_error", None)
        return {
            "healthy": False,
            "status": "error",
            "state": "dead",
            "reason": (
                f"generation thread stopped: {error!r}"
                if error is not None
                else "generation thread is not running"
            ),
        }

    @property
    def is_healthy(self) -> bool:
        return self.health_report()["healthy"]

    def _log_cache_stats(self):
        n_sequences = len(self.prompt_cache)
        n_bytes = self.prompt_cache.nbytes
        logging.info(f"Prompt Cache: {n_sequences} sequences, {n_bytes / 1e9:.2f} GB")
        for cache_type, stats in self.prompt_cache.stats_by_type().items():
            n_sequences = stats["n_sequences"]
            n_bytes = stats["n_bytes"]
            logging.info(
                f"- {cache_type}: {n_sequences} sequences, {n_bytes / 1e9:.2f} GB"
            )

    def _next_request(self, timeout=None):
        request = None
        if not self._is_distributed or self._rank == 0:
            try:
                if timeout is not None:
                    request = self.requests.get(timeout=timeout)
                else:
                    request = self.requests.get_nowait()
            except QueueEmpty:
                pass
        return self._share_request(request)

    def _share_object(self, obj):
        if not self._is_distributed:
            return obj

        if self._rank == 0:
            if obj is None:
                mx.eval(mx.distributed.all_sum(0))
                return None
            data = mx.array(pickle.dumps(obj))
            mx.eval(mx.distributed.all_sum(data.size))
            mx.eval(mx.distributed.all_sum(data))
            return obj
        else:
            size = mx.distributed.all_sum(0).item()
            if size == 0:
                return None
            data = mx.zeros(size, dtype=mx.uint8)
            data = mx.distributed.all_sum(data)
            return pickle.loads(data)

    def _share_request(self, request):
        if not self._is_distributed:
            return request

        shareable = request[1:] if request is not None else None
        shareable = self._share_object(shareable)
        if shareable is None:
            return None

        rq = request[0] if request is not None else Queue()
        return rq, *shareable

    def _tokenize(self, tokenizer, request, args):
        """Tokenize a request and split the prompt into segments.

        Returns a tuple

          * prompt - Full list of tokens
          * segments - A list of lists of tokens. Up to 3 segments that
            correspond to system prompt, context, thinking tail.
          * segment_types - A string per segment indicating if the segment is a
            system prompt or a user prompt or nothing special.
          * initial state - A string that contains the initial state of the
            state machine (normal or thinking depending on whether we have tail
            or not)
        """
        if request.request_type == "chat":
            messages = request.messages
            tools = request.tools
            role_mapping = request.role_mapping

            if tokenizer.has_chat_template:
                process_message_content(messages)
                if tools and not tokenizer.has_tool_calling:
                    logging.warning(
                        "Received tools but model does not support tool calling. "
                        "If you think this is an error, file an issue here: "
                        "https://github.com/ml-explore/mlx-lm/issues"
                    )

                chat_template_args = self.model_provider.cli_args.chat_template_args
                if args.chat_template_kwargs:
                    chat_template_args = chat_template_args.copy()
                    chat_template_args.update(args.chat_template_kwargs)
                template_kwargs = dict(
                    tools=tools,
                    tokenize=True,
                    **chat_template_args,
                )
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            else:
                prompt = tokenizer.encode(convert_chat(messages, role_mapping))
                return prompt, [prompt], ["assistant"], "normal"
        else:
            prompt = tokenizer.encode(request.prompt)
            return prompt, [prompt], ["assistant"], "normal"

        # If we are here it means we have a chat request so we need to search
        # for segments for better cache management.

        # Choose the initial state among only reasoning or normal
        initial_state = "normal"
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt)
            think_end = tokenizer.rfind_think_end(prompt)
            if think_start > think_end:
                initial_state = "reasoning"

        # It is not a user message so no segmentation needed.
        if messages[-1]["role"] != "user":
            return prompt, [prompt], ["assistant"], initial_state

        segments = []
        segment_types = []

        # Find where the system prompt ends and add it as a segment.
        num_system = 0
        sys_end = 0
        for m in messages:
            if m["role"] == "system":
                num_system += 1
            else:
                break
        if num_system > 0:
            sys_tokens = tokenizer.apply_chat_template(
                messages[:num_system] + [{"role": "user", "content": ""}],
                add_generation_prompt=False,
                **template_kwargs,
            )
            for i, (a, b) in enumerate(zip(sys_tokens, prompt)):
                if a != b:
                    sys_end = i
                    break
            if sys_end > 0 and sys_end < len(prompt):
                segments.append(prompt[:sys_end])
                segment_types.append("system")

        # Find a tail segment that contains thinking tokens (small up to 11
        # tokens)
        tail_start = len(prompt)
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt, start=tail_start - 11)
            if think_start >= 0:
                tail_start = think_start

        # Finalize the segments and return
        if sys_end < tail_start:
            segments.append(prompt[sys_end:tail_start])
            segment_types.append("user")
        if tail_start < len(prompt):
            segments.append(prompt[tail_start:])
            segment_types.append("assistant")
        if not segments:
            segments = [prompt]
            segment_types = ["assistant"]

        return prompt, segments, segment_types, initial_state

    def _make_state_machine(self, model_key, tokenizer, stop_words):
        """Make (and cache) a StopSequenceMatcher and TextStateMachine."""
        cache_key = (model_key, tuple(stop_words))
        rs = self._state_machine_cache.get(cache_key)
        if rs is not None:
            return rs

        stop_matcher = make_stop_matcher(tokenizer, stop_words)
        text_sm = make_text_state_machine(tokenizer, stop_words)

        if len(self._state_machine_cache) > 100:
            self._state_machine_cache.clear()
        self._state_machine_cache[cache_key] = (stop_matcher, text_sm)

        return stop_matcher, text_sm

    def _is_batchable(self, args):
        if not self.model_provider.is_batchable:
            return False
        # n>1 owns its own batch: one shared prefill replicated into n rows.
        # It runs on the dedicated parallel-sampling path, never mixed into
        # the continuous batch.
        if getattr(args, "n", 1) > 1:
            return False
        # Seeded ordinary batches still share the global sampler stream.  A
        # potentially eligible self-MTP request is allowed through this static
        # precheck because its exact route is decided after APC lookup and its
        # LaneRNG is built/evaluated on this generation thread.
        if args.seed is not None and not getattr(self.cli_args, "self_mtp", False):
            return False
        # Prompt-lookup speculative decoding runs on the single-stream path.
        if getattr(args, "prompt_lookup_ngram", 0):
            return False

        return True

    def _generate(self):
        # Local thread stream that we 'll pass to the BatchGenerator to make
        # sure that all generation runs in the same stream as the
        # synchronization messages.
        generation_stream = mx.default_stream(mx.default_device())

        # Load the default model if it is given
        self.model_provider.load_default()

        current_model = None
        current_sampling = None
        current_tokenizer = None
        current_model_key = None
        current_batch_kind = None
        current_self_mtp = None
        batch_generator = None
        drain_batch = False
        batch_results = {}

        unprocessed_requests = []

        def get_next_request(timeout=None):
            if unprocessed_requests:
                return unprocessed_requests.pop()
            else:
                return self._next_request(timeout)

        if self._is_distributed:
            seed = mx.distributed.all_sum(mx.random.state[0]).view(mx.uint64).item()
            mx.random.seed(seed)

        # True when the last serve slice made no progress (every self-MTP lane
        # queued/paused by admission, or the memory probe unavailable). The
        # next queue poll then blocks for a bounded interval instead of
        # busy-spinning admission (and its vm_stat probe) at 100% CPU; a new
        # request still wakes the loop immediately.
        batch_idle = False

        while not self._stop:
            request = None
            if not drain_batch:
                timeout = (
                    None
                    if (
                        batch_generator is not None
                        and len(batch_results) > 0
                        and not batch_idle
                    )
                    else BATCH_IDLE_BACKOFF_SECONDS
                )
                request = get_next_request(timeout=timeout)

            # We got a request
            if request is not None:
                rqueue, request, args = request

                # Can it be added to the current batch?
                if (
                    batch_generator is not None
                    and current_model == args.model
                    and self._is_batchable(args)
                ):
                    try:
                        prompt, segments, segment_types, initial_state = self._tokenize(
                            current_tokenizer, request, args
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    stop_matcher, text_sm = self._make_state_machine(
                        self.model_provider.model_key,
                        tokenizer,
                        args.stop_words,
                    )

                    self._log_cache_stats()
                    lookup_self_mtp = _batched_self_mtp_config(
                        args,
                        self.cli_args,
                        self.model_provider.model,
                        prompt_tokens=len(prompt),
                    )
                    lookup_model_key = _batched_prompt_cache_model_key(
                        self.model_provider.model_key,
                        self.cli_args,
                        lookup_self_mtp,
                    )
                    if hasattr(self.prompt_cache, "lookup"):
                        lookup = self.prompt_cache.lookup(lookup_model_key, prompt)
                        cache, rest = lookup.cache, lookup.remaining_tokens
                        mtp_sidecar = lookup.sidecar
                    else:
                        cache, rest = self.prompt_cache.fetch_nearest_cache(
                            lookup_model_key, prompt
                        )
                        mtp_sidecar = None
                    prompt_cache_count = len(prompt) - len(rest)
                    mtp_state = (
                        mtp_sidecar.state if mtp_sidecar is not None else None
                    )
                    self_mtp = _batched_self_mtp_config(
                        args,
                        self.cli_args,
                        self.model_provider.model,
                        prompt_tokens=len(prompt),
                        cached_prompt_tokens=prompt_cache_count,
                        mtp_state=mtp_state,
                    )
                    if self_mtp is not None and current_self_mtp is not None:
                        # A dynamically lowered fixed-depth cohort remains one
                        # batch kind.  New lanes enter at its current depth;
                        # the cycle callback may raise/lower all named lanes at
                        # the next transaction boundary.
                        self_mtp = dict(self_mtp)
                        self_mtp["num_draft"] = current_self_mtp["num_draft"]
                    candidate_kind = _batch_kind_key(lookup_model_key, self_mtp)
                    # A seeded request which became plain after APC/exclusion
                    # cannot enter the global-RNG plain batch.
                    if self_mtp is None and args.seed is not None:
                        drain_batch = True
                        unprocessed_requests.append((rqueue, request, args))
                        del cache
                        continue
                    if candidate_kind != current_batch_kind:
                        drain_batch = True
                        unprocessed_requests.append((rqueue, request, args))
                        del cache
                        continue
                    N = prompt_cache_count
                    while N > 0:
                        if N >= len(segments[0]):
                            N -= len(segments.pop(0))
                            segment_types.pop(0)
                        else:
                            segments[0] = segments[0][N:]
                            break

                    ctx = GenerationContext(
                        has_tool_calling=tokenizer.has_tool_calling,
                        has_thinking=tokenizer.has_thinking,
                        tool_parser=tokenizer.tool_parser,
                        text_sm=text_sm,
                        initial_state=initial_state,
                        prompt=prompt,
                        prompt_cache_count=prompt_cache_count,
                    )
                    rqueue.put(ctx)

                    lane_rng = (
                        _make_lane_rng(args, self._lane_rng_root, mtp_sidecar)
                        if self_mtp is not None
                        else None
                    )
                    (uid,) = batch_generator.insert_segments(
                        segments=[segments],
                        max_tokens=[args.max_tokens],
                        caches=[cache],
                        all_tokens=[prompt[:prompt_cache_count]],
                        samplers=[_make_sampler(args, tokenizer)],
                        logits_processors=[_make_logits_processors(args)],
                        stop_matchers=[stop_matcher],
                        self_mtp_configs=[self_mtp] if self_mtp is not None else None,
                        mtp_states=[mtp_state] if self_mtp is not None else None,
                        lane_rngs=[lane_rng] if self_mtp is not None else None,
                    )
                    batch_results[uid] = {
                        "ctx": ctx,
                        "rqueue": rqueue,
                        "detokenizer": tokenizer.detokenizer,
                        "segment_types": segment_types[::-1],
                        "top_logprobs": args.top_logprobs,
                        "mtp": self_mtp is not None,
                    }
                    # just making sure we don't leave a reference around
                    del cache

                    if self.model_provider.cli_args.prompt_cache_bytes is not None:
                        total = self.model_provider.cli_args.prompt_cache_bytes
                        active = batch_generator.prompt_cache_nbytes
                        self.prompt_cache.trim_to(n_bytes=total - active)
                    continue

                # No batch generator. Load the model and if it's not
                # batchable serve sequential, o/w make a batch generaotr and
                # serve batched
                elif batch_generator is None:
                    try:
                        model, tokenizer = self.model_provider.load(
                            args.model.model, args.model.adapter, args.model.draft
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    if not self._is_batchable(args):
                        self._serve_request(
                            (rqueue, request, args), generation_stream
                        )
                        continue

                    # The batch kind is unknowable until tokenization and APC
                    # lookup.  Probe that exact route before constructing a
                    # generator; the request itself is re-queued below and
                    # performs a fresh owned lookup when inserted.
                    try:
                        prompt, _, _, _ = self._tokenize(tokenizer, request, args)
                        lookup_self_mtp = _batched_self_mtp_config(
                            args,
                            self.cli_args,
                            model,
                            prompt_tokens=len(prompt),
                        )
                        lookup_model_key = _batched_prompt_cache_model_key(
                            self.model_provider.model_key,
                            self.cli_args,
                            lookup_self_mtp,
                        )
                        if hasattr(self.prompt_cache, "lookup"):
                            lookup = self.prompt_cache.lookup(
                                lookup_model_key, prompt
                            )
                            probe_cache = lookup.cache
                            probe_rest = lookup.remaining_tokens
                            probe_sidecar = lookup.sidecar
                        else:
                            probe_cache, probe_rest = (
                                self.prompt_cache.fetch_nearest_cache(
                                    lookup_model_key, prompt
                                )
                            )
                            probe_sidecar = None
                        cached_prompt_tokens = len(prompt) - len(probe_rest)
                        mtp_state = (
                            probe_sidecar.state
                            if probe_sidecar is not None
                            else None
                        )
                        current_self_mtp = _batched_self_mtp_config(
                            args,
                            self.cli_args,
                            model,
                            prompt_tokens=len(prompt),
                            cached_prompt_tokens=cached_prompt_tokens,
                            mtp_state=mtp_state,
                        )
                        del probe_cache
                        if current_self_mtp is not None:
                            free_gib = _current_self_mtp_free_memory_gib()
                            initial = self._self_mtp_admission_controller.decide(
                                [len(prompt)],
                                math.nan if free_gib is None else free_gib,
                                max_draft=int(current_self_mtp["num_draft"]),
                            )
                            if initial.modes[0] == "self_mtp":
                                current_self_mtp = dict(current_self_mtp)
                                current_self_mtp["num_draft"] = int(
                                    initial.draft_depths[0]
                                )
                            elif initial.modes[0] == "plain":
                                current_self_mtp = None
                            else:
                                raise RequestCompositionError(
                                    "self-MTP admission cannot preserve the "
                                    "20 GiB hard memory reserve; request queued/"
                                    "rejected before decode"
                                )
                        if current_self_mtp is None and args.seed is not None:
                            self._serve_request(
                                (rqueue, request, args), generation_stream
                            )
                            continue
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    current_model = args.model
                    current_tokenizer = tokenizer
                    current_model_key = lookup_model_key
                    current_batch_kind = _batch_kind_key(
                        current_model_key, current_self_mtp
                    )
                    batch_results = {}
                    try:
                        kv_budget_bytes = None
                        kv_cost = None
                        if self.cli_args.state_budget_gb is not None:
                            # INTERIM SAFETY DISABLE: shared batch caches
                            # allocate every row at the cohort-max width, so
                            # per-row linear projection can admit above
                            # budget for heterogeneous cohorts (reviewer
                            # P1). The flag refuses until cohort-aware
                            # accounting lands; _measure_kv_cost stays
                            # exercised by tests for the coming rework.
                            raise ValueError(
                                "--state-budget-gb is disabled: cohort-"
                                "aware state accounting is not yet safe "
                                "for shared batch caches (see "
                                "batch_admission cohort_bytes work)"
                            )
                        batch_generator = BatchGenerator(
                            model,
                            completion_batch_size=self.cli_args.decode_concurrency,
                            prefill_batch_size=self.cli_args.prompt_concurrency,
                            prefill_step_size=self.cli_args.prefill_step_size,
                            prefill_batch_window=self.cli_args.prompt_batch_window,
                            kv_budget_bytes=kv_budget_bytes,
                            kv_cost=kv_cost,
                            stream=generation_stream,
                            self_mtp=current_self_mtp,
                            mtp_admission=(
                                self._self_mtp_admission
                                if current_self_mtp is not None
                                else None
                            ),
                            **_batched_kv_quantization(
                                self.cli_args, current_self_mtp
                            ),
                        )
                    except Exception as e:
                        # Probe or constructor failure (rotating-cache
                        # refusal, invalid budget, ...) must reach the
                        # requester, not kill the generation thread.
                        batch_generator = None
                        rqueue.put(e)
                        continue
                    unprocessed_requests.append((rqueue, request, args))
                    continue

                # We have a batch but this request cannot be added to the
                # batch so drain it to process the request.
                else:
                    drain_batch = True
                    unprocessed_requests.append((rqueue, request, args))
                    continue

            # No request so serve from the current batch
            elif batch_generator is not None:
                if len(batch_results) == 0:
                    if drain_batch:
                        current_model = None
                        current_sampling = None
                        current_tokenizer = None
                        current_model_key = None
                        current_batch_kind = None
                        current_self_mtp = None
                        batch_generator.close()
                        batch_generator = None
                        drain_batch = False
                    continue

                uids_to_remove = []
                batch_idle = True
                for _ in self._time_budget:
                    prompt_responses, gen_responses = batch_generator.next()
                    if not prompt_responses and not gen_responses:
                        break
                    batch_idle = False

                    # Progress report for prompt processing
                    for r in prompt_responses:
                        result = batch_results[r.uid]
                        result["rqueue"].put(r.progress)
                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                    # Save the caches at end of segments
                    eos_ids = [
                        r.uid
                        for r in prompt_responses
                        if r.end_of_segment
                        and not r.end_of_prompt
                        and batch_results[r.uid]["segment_types"]
                    ]
                    caches = batch_generator.extract_cache(eos_ids)
                    for uid, (cache, cache_key) in caches.items():
                        self.prompt_cache.insert_cache(
                            self.model_provider.model_key,
                            cache_key[:],
                            cache,
                            cache_type=batch_results[uid]["segment_types"].pop(),
                        )
                    del caches

                    for r in gen_responses:
                        result = batch_results[r.uid]

                        # Don't decode the final stop token
                        if r.finish_reason == "stop":
                            result["detokenizer"].finalize()
                            text = result["detokenizer"].last_segment
                        elif r.finish_reason == "length":
                            result["detokenizer"].add_token(r.token)
                            result["detokenizer"].finalize()
                            text = result["detokenizer"].last_segment
                        else:
                            result["detokenizer"].add_token(r.token)
                            text = result["detokenizer"].last_segment

                        result["rqueue"].put(
                            Response(
                                text,
                                r.token,
                                r.logprobs[r.token].item(),
                                r.finish_reason,
                                _format_top_logprobs(
                                    r.logprobs,
                                    result["top_logprobs"],
                                    current_tokenizer,
                                ),
                            )
                        )

                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            sidecar = None
                            if result.get("mtp") and getattr(r, "mtp_state", None):
                                carried_rng = getattr(r, "lane_rng", None)
                                rng_key = getattr(carried_rng, "key", carried_rng)
                                rng_draws = int(
                                    getattr(
                                        r,
                                        "rng_draws",
                                        getattr(carried_rng, "draws", 0),
                                    )
                                    or 0
                                )
                                if rng_key is not None:
                                    mx.eval(rng_key)
                                sidecar = MTPAPCSidecar(
                                    r.mtp_state,
                                    len(r.all_tokens),
                                    rng_key=rng_key,
                                    rng_draws=rng_draws,
                                )
                            if isinstance(
                                self.prompt_cache, AutomaticPrefixCache
                            ):
                                self.prompt_cache.insert_cache(
                                    current_model_key,
                                    r.all_tokens[:],
                                    r.prompt_cache,
                                    cache_type="assistant",
                                    sidecar=sidecar,
                                )
                            else:
                                self.prompt_cache.insert_cache(
                                    current_model_key,
                                    r.all_tokens[:],
                                    r.prompt_cache,
                                    cache_type="assistant",
                                )
                            del batch_results[r.uid]

                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                uids_to_remove = self._share_object(uids_to_remove)
                if uids_to_remove:
                    batch_generator.remove(uids_to_remove)
                    for uid in uids_to_remove:
                        # It may have already been removed during
                        # generation
                        batch_results.pop(uid, None)

                if batch_idle and drain_batch and len(batch_results) > 0:
                    # A draining batch skips the queue poll (and its bounded
                    # timeout), so a fully queued/paused membership would
                    # otherwise busy-spin admission here. Sleep the same
                    # bounded interval instead.
                    time.sleep(BATCH_IDLE_BACKOFF_SECONDS)

    def _check_parallel_sampling_state_budget(
        self, cache, n, prompt_tokens, max_tokens
    ):
        """Refuse an ``n>1`` request whose replicated cache would not fit."""
        budget = _state_budget_bytes(self.cli_args)
        if budget is None:
            return
        projected = _parallel_sampling_state_bytes(
            cache, n, prompt_tokens, max_tokens
        )
        if projected is None or projected <= budget:
            return
        raise RequestCompositionError(
            f"n={n} would need about {projected / (1 << 30):.1f} GB of cache "
            f"state for a {prompt_tokens}-token prompt and {max_tokens} new "
            f"tokens, over the {budget / (1 << 30):.1f} GB budget. Send a "
            f"smaller n, a shorter prompt, or fewer max_tokens."
        )

    def _serve_request(self, request, generation_stream=None):
        """Route one non-batchable request to the single or n-way path."""
        if getattr(request[2], "n", 1) > 1:
            self._serve_parallel_samples(request, generation_stream)
        else:
            self._serve_single(request)

    def _serve_parallel_samples(self, request, generation_stream=None):
        """Serve an OpenAI ``n>1`` request: one prefill, n independent samples.

        The prompt is prefilled once at batch size one (reusing any prefix-cache
        hit), then the finished cache is replicated into one row per sample. Each
        sample keeps its own sampling draws, its own logits processors and its
        own continuation; the prefix compute and the prefix cache entry are
        shared.
        """
        rqueue, request, args = request
        parallel = None
        try:
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            n = int(args.n)

            # n>1 decodes as a batch, so the model has to be batchable. A
            # draft model makes it unbatchable and its speculation has no
            # n-way path: refuse instead of dropping it silently.
            if self.model_provider.draft_model is not None:
                raise RequestCompositionError(
                    "n>1 is not supported with a draft model: speculative "
                    "decoding has no batched path here. Send n=1, or start "
                    "the server without --draft-model."
                )
            if not self.model_provider.is_batchable:
                raise RequestCompositionError(
                    "n>1 needs a batchable model: this model's cache cannot "
                    "be replicated into per-sample rows. Send n=1."
                )

            prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
            if len(prompt) < 1:
                raise ValueError("n>1 requires a non-empty prompt")

            stop_matcher, text_sm = self._make_state_machine(
                self.model_provider.model_key,
                tokenizer,
                args.stop_words,
            )
            ctx = GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                text_sm=text_sm,
                initial_state=initial_state,
                prompt=prompt,
            )

            if args.seed is not None:
                mx.random.seed(args.seed)

            # One sampler object for every row: make_sampler marks it
            # batch_groupable, so the batch samples all rows in one vectorized
            # call that draws independently per row. Logits processors are
            # per-row instances -- they may carry state.
            sampler = _make_sampler(args, tokenizer)
            logits_processors = [_make_logits_processors(args) for _ in range(n)]

            # The prefix is shared, so the prefix-cache lookup happens once.
            self._log_cache_stats()
            lookup_self_mtp = _batched_self_mtp_config(
                args,
                self.cli_args,
                model,
                prompt_tokens=len(prompt),
            )
            if any(logits_processors) or getattr(
                self.cli_args, "parallel_sampling_mtp", "mtp"
            ) != "mtp":
                lookup_self_mtp = None
            prompt_cache_model_key = _batched_prompt_cache_model_key(
                self.model_provider.model_key,
                self.cli_args,
                lookup_self_mtp,
            )
            if hasattr(self.prompt_cache, "lookup"):
                lookup = self.prompt_cache.lookup(
                    prompt_cache_model_key, prompt
                )
                cache, rest = lookup.cache, lookup.remaining_tokens
                mtp_sidecar = lookup.sidecar
            else:
                cache, rest = self.prompt_cache.fetch_nearest_cache(
                    prompt_cache_model_key, prompt
                )
                mtp_sidecar = None
            ctx.prompt_cache_count = len(prompt) - len(rest)
            if not rest:
                raise ValueError(
                    "prefix cache returned an empty remainder; n>1 needs at "
                    "least one uncached token to seed generation"
                )
            if cache is None:
                cache = make_prompt_cache(model)

            # Route with the same inputs the n=1 path uses: the cache result
            # decides whether MTP would have been admitted at all.
            parallel_kind, note = _parallel_sampling_route(
                args,
                self.cli_args,
                model,
                prompt_tokens=len(prompt),
                cached_prompt_tokens=ctx.prompt_cache_count,
                mtp_state=(
                    mtp_sidecar.state if mtp_sidecar is not None else None
                ),
                has_logits_processors=any(logits_processors),
            )
            if note:
                logging.info("Parallel sampling (n=%d): %s", n, note)

            self_mtp = None
            lane_rng = None
            mtp_prompt = None
            parallel_admission = None
            mtp_state = mtp_sidecar.state if mtp_sidecar is not None else None
            if parallel_kind == "self_mtp":
                self_mtp = _batched_self_mtp_config(
                    args,
                    self.cli_args,
                    model,
                    prompt_tokens=len(prompt),
                    cached_prompt_tokens=ctx.prompt_cache_count,
                    mtp_state=mtp_state,
                )
                _quantize_batched_self_mtp_cache(
                    cache, self.cli_args, self_mtp
                )
                free_gib = _current_self_mtp_free_memory_gib()
                usable = (
                    max(
                        free_gib
                        - self._self_mtp_admission_controller.hard_reserve_gib,
                        0.0,
                    )
                    if free_gib is not None
                    else 0.0
                )
                # OpenAI n is indivisible: partial-lane admission would return
                # fewer choices.  Try all lanes at k=2, then k=1, then plain;
                # otherwise reject before a verify forward is submitted.
                chosen_depth = None
                configured_depth = min(int(self_mtp["num_draft"]), 2)
                for depth in range(configured_depth, 0, -1):
                    projected = _parallel_self_mtp_required_gib(
                        self._self_mtp_admission_controller,
                        cache,
                        n,
                        len(prompt),
                        depth,
                        mtp_state=mtp_state,
                    )
                    if projected <= usable:
                        chosen_depth = depth
                        break
                if chosen_depth is None:
                    plain_projected = _parallel_self_mtp_required_gib(
                        self._self_mtp_admission_controller,
                        cache,
                        n,
                        len(prompt),
                        0,
                    )
                    if plain_projected <= usable:
                        logging.info(
                            "Parallel self-MTP degraded to plain: n=%d "
                            "prompt=%d usable=%.2f GiB",
                            n,
                            len(prompt),
                            usable,
                        )
                        self_mtp = None
                        parallel_kind = "plain"
                    else:
                        raise RequestCompositionError(
                            "parallel self-MTP cannot preserve the 20 GiB hard "
                            "memory reserve even after k=1 and plain fallback; "
                            "send a smaller n or shorter prompt"
                        )
                else:
                    self_mtp = dict(self_mtp)
                    self_mtp["num_draft"] = chosen_depth
                    lane_rng = _make_lane_rng(
                        args, self._lane_rng_root, mtp_sidecar
                    )
                    mtp_prompt = rest
                    # Admission is not one-shot: the generator re-budgets at
                    # every cycle boundary against fresh system free memory,
                    # capped at this request's configured depth, so the lanes
                    # can drop k, migrate to plain, or pause under pressure.
                    parallel_admission = _make_self_mtp_admission_callback(
                        self._self_mtp_admission_controller,
                        max_draft=configured_depth,
                    )

            if parallel_kind == "plain":
                # Prefill everything but the seed token, once. The first chunk
                # runs before acceptance so replicated state can be refused
                # cleanly instead of OOM-killing the process mid-stream.
                head_size = self.cli_args.prefill_step_size
                body = rest[:-1]
                prefill_prompt_cache(
                    model,
                    body[:head_size],
                    cache,
                    prefill_step_size=head_size,
                )
                self._check_parallel_sampling_state_budget(
                    cache, n, len(prompt), args.max_tokens
                )
                prefilled = min(len(body), head_size)

                def progress(processed, total):
                    rqueue.put(
                        (
                            processed + prefilled + ctx.prompt_cache_count,
                            len(prompt),
                        )
                    )

                prefill_prompt_cache(
                    model,
                    body[head_size:],
                    cache,
                    prefill_step_size=head_size,
                    progress_callback=progress,
                )
                progress(len(rest) - prefilled, len(rest))

            rqueue.put(ctx)
            parallel_history = (
                prompt[: ctx.prompt_cache_count]
                if self_mtp is not None
                else list(prompt[:-1])
            )

            parallel = ParallelSampleGenerator(
                model,
                cache,
                prompt[-1],
                n,
                max_tokens=args.max_tokens,
                samplers=[sampler] * n,
                logits_processors=logits_processors,
                stop_matchers=[stop_matcher] * n,
                all_tokens=parallel_history,
                prefill_step_size=self.cli_args.prefill_step_size,
                stream=generation_stream,
                self_mtp=self_mtp,
                mtp_state=mtp_state if self_mtp is not None else None,
                lane_rng=lane_rng,
                mtp_prompt=mtp_prompt,
                mtp_admission=(
                    parallel_admission if self_mtp is not None else None
                ),
                **_batched_kv_quantization(self.cli_args, self_mtp),
            )
            logging.info(
                "Parallel sampling: kind=%s n=%d prompt=%d cached=%d",
                parallel_kind,
                n,
                len(prompt),
                ctx.prompt_cache_count,
            )

            # One detokenizer per sample. Stop sequences are matched inside the
            # batch (the stop matcher is passed per row), so finish_reason is
            # already authoritative here.
            detokenizers = [tokenizer.detokenizer for _ in range(n)]
            while len(parallel) > 0:
                step_responses = parallel.next()
                if not step_responses:
                    # Every lane is paused by cycle-boundary admission (or the
                    # memory probe is unavailable): back off boundedly instead
                    # of busy-spinning admission and its vm_stat probe.
                    if ctx._should_stop:
                        break
                    time.sleep(BATCH_IDLE_BACKOFF_SECONDS)
                    continue
                for index, r in step_responses:
                    detokenizer = detokenizers[index]
                    if r.finish_reason == "stop":
                        # Don't decode the final stop token.
                        detokenizer.finalize()
                    elif r.finish_reason == "length":
                        detokenizer.add_token(r.token)
                        detokenizer.finalize()
                    else:
                        detokenizer.add_token(r.token)
                    rqueue.put(
                        Response(
                            detokenizer.last_segment,
                            r.token,
                            r.logprobs[r.token].item(),
                            r.finish_reason,
                            _format_top_logprobs(
                                r.logprobs, args.top_logprobs, tokenizer
                            ),
                            index,
                        )
                    )
                if ctx._should_stop:
                    break

            # The rows hold their own copies of the prefix state (merge reads
            # the source and writes new per-row buffers), so the prefilled cache
            # becomes the shared prefix entry. Per-sample continuations are
            # deliberately NOT stored: the samples diverge, and keeping one of
            # them would bias later prefix hits toward an arbitrary sample.
            # The key is derived HERE, after lane preparation advanced the
            # retained cache, so the stored span and its key always agree
            # (self-MTP retains the fully prefilled prompt; plain retains
            # exactly prompt[:-1]).
            self.prompt_cache.insert_cache(
                prompt_cache_model_key,
                _parallel_prompt_cache_key(prompt, cache, self_mtp),
                cache,
            )

            rqueue.put(None)
        except Exception as e:
            rqueue.put(e)
        finally:
            if parallel is not None:
                parallel.close()

    def _serve_single(self, request):
        rqueue, request, args = request

        # Define the progress callback
        def progress(tokens_processed, tokens_total):
            rqueue.put((tokens_processed, tokens_total))

        try:
            # Load the model and tokenizer
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model

            # Prepare the prompt and state machine
            prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
            stop_matcher, text_sm = self._make_state_machine(
                self.model_provider.model_key,
                tokenizer,
                args.stop_words,
            )

            # Start the generation context
            ctx = GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                text_sm=text_sm,
                initial_state=initial_state,
                prompt=prompt,
            )
            rqueue.put(ctx)

            # Seed if requested
            if args.seed is not None:
                mx.random.seed(args.seed)

            # Make the sampler and logit processor
            sampler = _make_sampler(args, tokenizer)
            logits_processors = _make_logits_processors(args)

            # Load the KV cache
            self._log_cache_stats()
            external_draft = draft_model is not None
            cache, rest, mtp_sidecar = _fetch_single_request_prompt_cache(
                self.prompt_cache,
                self.model_provider.model_key,
                prompt,
                external_draft=external_draft,
            )
            ctx.prompt_cache_count = len(prompt) - len(rest)
            if _discard_small_sidecarless_apc_hit_for_mtp(
                self.cli_args,
                model,
                ctx.prompt_cache_count,
                mtp_sidecar,
            ):
                logging.info(
                    "Discarding small sidecar-less APC hit (%d tokens) to "
                    "preserve self-MTP admission",
                    ctx.prompt_cache_count,
                )
                cache = None
                rest = prompt
                mtp_sidecar = None
                ctx.prompt_cache_count = 0
            cache_key = prompt[:]
            if cache is None:
                cache = make_prompt_cache(self.model_provider.model)
                if self.model_provider.draft_model is not None:
                    cache += make_prompt_cache(self.model_provider.draft_model)

            # Process the prompt and generate tokens
            stop_state = stop_matcher.make_state()
            lane_rng = (
                _make_lane_rng(args, self._lane_rng_root, mtp_sidecar)
                if getattr(self.cli_args, "self_mtp", False)
                else None
            )
            self_mtp = _self_mtp_config(
                args,
                self.cli_args,
                model,
                cached_prompt_tokens=ctx.prompt_cache_count,
                prompt_tokens=len(prompt),
                mtp_state=(mtp_sidecar.state if mtp_sidecar is not None else None),
                lane_rng=lane_rng,
            )
            if self_mtp is not None:
                depth_router = self_mtp.get("speculation_router")
                # k stays a plain integer on both paths (the native/floor
                # depth); the adaptive ceiling gets its own field so log
                # parsers keyed on numeric k keep working.
                logging.info(
                    "Self-MTP admitted: prompt=%d cached=%d sidecar=%s "
                    "k=%d adaptive_ceiling=%s window=%s sink=%s",
                    len(prompt),
                    ctx.prompt_cache_count,
                    mtp_sidecar is not None,
                    (
                        depth_router.floor
                        if depth_router is not None
                        else self_mtp["num_draft"]
                    ),
                    (
                        depth_router.ceiling
                        if depth_router is not None
                        else "none"
                    ),
                    self_mtp.get("window_size", "native"),
                    self_mtp.get("sink_size", "native"),
                )
            elif getattr(self.cli_args, "self_mtp", False):
                reason = (
                    "APC target-cache hit without matching MTP state"
                    if ctx.prompt_cache_count
                    else "request sampling/speculation regime"
                )
                logging.info("Self-MTP bypassed: %s", reason)
            prompt_lookup_config = None
            prompt_lookup_stats = None
            if getattr(args, "prompt_lookup_ngram", 0):
                from .prompt_lookup import HybridStats

                prompt_lookup_stats = HybridStats()
                prompt_lookup_config = {
                    "ngram_max": args.prompt_lookup_ngram,
                    "num_draft": args.prompt_lookup_tokens,
                    "adaptive": args.prompt_lookup_adaptive,
                    "rate_gate": args.prompt_lookup_rate_gate,
                    "warmup": args.prompt_lookup_warmup,
                    "gate": args.prompt_lookup_gate,
                    "rate_gate_probe": args.prompt_lookup_rate_gate_probe,
                    "rate_gate_margin": args.prompt_lookup_rate_gate_margin,
                    "stats": prompt_lookup_stats,
                    # The target APC already owns model state for this prefix;
                    # PLD separately needs token IDs so suffix matches can
                    # cross the cached-prefix boundary.
                    "history_prompt": prompt,
                }
            token_stream = stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=rest,
                max_tokens=args.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=cache,
                draft_model=draft_model,
                num_draft_tokens=args.num_draft_tokens,
                prompt_lookup=prompt_lookup_config,
                self_mtp=self_mtp,
                prompt_progress_callback=progress,
                prefill_step_size=self.cli_args.prefill_step_size,
                kv_bits=getattr(self.cli_args, "kv_bits", None),
                kv_key_bits=getattr(self.cli_args, "kv_key_bits", None),
                kv_value_bits=getattr(self.cli_args, "kv_value_bits", None),
                kv_group_size=getattr(self.cli_args, "kv_group_size", 64),
                quantized_kv_start=getattr(
                    self.cli_args,
                    "quantized_kv_start",
                    DEFAULT_QUANTIZED_KV_START,
                ),
            )
            completed = False
            try:
                for gen in token_stream:
                    finish_reason = gen.finish_reason

                    # Token-level stop word detection
                    stop_state, matched = StopSequenceMatcher.match(
                        stop_state, stop_matcher._trie, gen.token
                    )
                    if matched:
                        finish_reason = "stop"

                    rqueue.put(
                        Response(
                            gen.text,
                            gen.token,
                            gen.logprobs[gen.token].item(),
                            finish_reason,
                            _format_top_logprobs(
                                gen.logprobs, args.top_logprobs, tokenizer
                            ),
                        )
                    )
                    cache_key.append(gen.token)

                    if ctx._should_stop:
                        if self._is_distributed:
                            raise NotImplementedError()
                        break

                    if finish_reason is not None:
                        completed = True
                        break
            finally:
                token_stream.close()
                if prompt_lookup_stats is not None:
                    logging.info(
                        "Prompt lookup: %s | rate_probe=%s delatched=%s "
                        "spec_ms_tok=%.3f plain_ms_tok=%.3f",
                        prompt_lookup_stats.summary(),
                        prompt_lookup_stats.rate_gate_probed,
                        prompt_lookup_stats.rate_gate_delatched,
                        prompt_lookup_stats.rate_gate_spec_ms_per_tok,
                        prompt_lookup_stats.rate_gate_plain_ms_per_tok,
                    )
                if self_mtp is not None:
                    adaptive_router = self_mtp.get("speculation_router")
                    if adaptive_router is not None:
                        # Depth-engagement telemetry: one JSON snapshot per
                        # adaptive request so an A/B can verify the ceiling
                        # actually engaged (expansions/backoffs/final depth).
                        logging.info(
                            "Self-MTP adaptive depth: %s",
                            json.dumps(
                                adaptive_router.snapshot(), sort_keys=True
                            ),
                        )

            rqueue.put(None)

            # Save the KV cache again
            sidecar = None
            if self_mtp is not None and completed:
                captured = self_mtp["state_out"]
                covered = int(captured.get("covered_tokens", 0))
                cache_offset = max(
                    (getattr(c, "offset", 0) for c in cache), default=0
                )
                if (
                    captured.get("reusable")
                    and covered == cache_offset
                    and 0 < covered <= len(cache_key)
                ):
                    # Carry the lane key too: a resume that dropped it would
                    # fall back to the global stream and repeat draws.
                    sidecar = MTPAPCSidecar(
                        captured["state"],
                        covered,
                        rng_key=captured.get("rng_key"),
                        rng_draws=int(captured.get("rng_draws") or 0),
                    )
                else:
                    logging.warning(
                        "Self-MTP APC sidecar not reusable: captured=%s "
                        "covered=%d target_offset=%d cache_key=%d",
                        captured.get("reusable"),
                        covered,
                        cache_offset,
                        len(cache_key),
                    )
            _store_single_request_prompt_cache(
                self.prompt_cache,
                self.model_provider.model_key,
                cache_key,
                cache,
                sidecar=sidecar,
                external_draft=external_draft,
            )

        except Exception as e:
            rqueue.put(e)

    def generate(
        self,
        request: CompletionRequest,
        generation_args: GenerationArguments,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self._admit_request()
        released = False

        def _release():
            nonlocal released
            if not released:
                released = True
                self._retire_request()

        def _inner():
            try:
                while True:
                    response = response_queue.get()
                    if response is None:
                        break
                    if isinstance(response, Exception):
                        raise response
                    if isinstance(response, tuple):
                        if progress_callback is not None:
                            progress_callback(*response)
                        continue
                    yield response
            finally:
                _release()

        try:
            response_queue = Queue()
            self.requests.put((response_queue, request, generation_args))
            ctx = response_queue.get()
            if isinstance(ctx, Exception):
                raise ctx
        except BaseException:
            _release()
            raise

        stream = _inner()
        # The finally above releases the slot on a normal or raised exit. This
        # covers the remaining case: a caller that drops the generator without
        # ever starting it, e.g. after the client disconnects.
        weakref.finalize(stream, _release)
        return ctx, stream

    def _admit_request(self, timeout: Optional[float] = None):
        """Take an in-flight slot, waiting while a soft reload holds the gate."""
        timeout = self.admission_timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, timeout)
        with self._admission:
            while self._paused:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SoftReloadBusy(
                        "server is applying a configuration change; retry shortly"
                    )
                self._admission.wait(remaining)
            self._inflight += 1

    def _retire_request(self):
        with self._admission:
            self._inflight = max(0, self._inflight - 1)
            if self._inflight == 0:
                self._admission.notify_all()

    @property
    def inflight(self) -> int:
        with self._admission:
            return self._inflight

    def _wait_for_drain(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._admission:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._admission.wait(remaining)
            return True

    def clear_prompt_cache(self) -> Dict[str, int]:
        """Drop every cached prefix. Returns what was dropped."""
        clear = getattr(self.prompt_cache, "clear", None)
        if callable(clear):
            return clear()
        # A plain LRUPromptCache has no clear(); trimming to zero is equivalent.
        entries = len(self.prompt_cache)
        n_bytes = int(getattr(self.prompt_cache, "nbytes", 0))
        self.prompt_cache.trim_to(n_sequences=0, n_bytes=0)
        return {"entries": entries, "sidecars": 0, "bytes": n_bytes}

    def soft_reload(
        self,
        config: Any,
        *,
        drain_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Change serving configuration in place, keeping the weights resident.

        The order is fixed and it matters:

        1. validate the whole request against the whitelist, before touching
           anything, so a bad request cannot half-apply;
        2. close the admission gate and wait for in-flight generation to
           finish -- a lever that lands mid-sequence would split one sequence
           across two numerical regimes;
        3. apply the change;
        4. drop the prefix cache. This is not housekeeping. The same tokens
           under two settings give different state, and some levers change the
           cache layout, so entries are dropped rather than kept;
        5. reopen the gate.

        NOT for promotion-grade A/B measurement. The A/B harness runs one
        process per arm on purpose: that isolates the allocator, the
        compilation cache, module constants and every KV byte between arms.
        Soft reload keeps all of it and trades that isolation for speed. Use it
        for interactive serving and exploratory sweeps. Do not rewrite the
        benchmark harness onto it.
        """
        drain_timeout = (
            DEFAULT_SOFT_RELOAD_DRAIN_TIMEOUT
            if drain_timeout is None
            else float(drain_timeout)
        )
        plan = plan_soft_reload(self.cli_args, config)

        if not self._reload_lock.acquire(blocking=False):
            raise SoftReloadBusy("a soft reload is already in progress")
        started = time.monotonic()
        try:
            with self._admission:
                self._paused = True
                inflight_at_start = self._inflight
            try:
                if not self._wait_for_drain(drain_timeout):
                    raise SoftReloadBusy(
                        f"generation did not drain within {drain_timeout:g}s; "
                        "nothing was changed"
                    )
                changes = apply_soft_reload(self.cli_args, plan)
                cache_report = self.clear_prompt_cache()
            finally:
                with self._admission:
                    self._paused = False
                    self._admission.notify_all()
        finally:
            self._reload_lock.release()

        logging.info(
            "Soft reload applied: %s; dropped %d prompt-cache entries.",
            {name: change["new"] for name, change in changes.items()},
            cache_report["entries"],
        )
        return {
            "object": "soft_reload",
            "changed": {k: v for k, v in changes.items() if v["old"] != v["new"]},
            "unchanged": [k for k, v in changes.items() if v["old"] == v["new"]],
            "prompt_cache": {
                "entries_dropped": cache_report["entries"],
                "sidecars_dropped": cache_report["sidecars"],
                "bytes_freed": cache_report["bytes"],
            },
            "drain": {
                "inflight_at_start": inflight_at_start,
                "waited_s": round(time.monotonic() - started, 4),
            },
            "effective_config": read_effective_config(self.cli_args),
        }

    @property
    def cli_args(self):
        return self.model_provider.cli_args


class APIHandler(BaseHTTPRequestHandler):
    # OpenAI ``n``. Set from the request body in do_POST; the class default
    # keeps every other entry point (and older callers) on one sample.
    n = 1

    def __init__(
        self,
        response_generator: ResponseGenerator,
        *args,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.response_generator = response_generator
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        allowed_origins = self.response_generator.cli_args.allowed_origins
        origin = self.headers.get("Origin")
        if "*" in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        if self.path == SOFT_RELOAD_PATH:
            self.handle_soft_reload()
            return

        request_factories = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        if self.path not in request_factories:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._set_completion_headers(411)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Content-Length header is required"}).encode()
            )
            return
        try:
            content_length = int(content_length)
        except ValueError:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Invalid Content-Length header"}).encode()
            )
            return
        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e} - Raw body: {raw_body.decode()}")
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"Invalid JSON in request body: {e}"}).encode()
            )
            return

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            debug_body = json.dumps(self.body, indent="\t")
            logging.debug(f"Incoming Request Body: {debug_body}")
        if not isinstance(self.body, dict):
            debug_body = json.dumps(self.body, indent="\t")
            logging.error(f"Invalid Request Body: {debug_body}")
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Request should be a JSON dictionary"}).encode()
            )
            return

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.requested_draft_model = self.body.get("draft_model", "default_model")
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.response_generator.cli_args.num_draft_tokens
        )
        self.prompt_lookup_ngram = self.body.get(
            "prompt_lookup_ngram", self.response_generator.cli_args.prompt_lookup_ngram
        )
        self.prompt_lookup_tokens = self.body.get(
            "prompt_lookup_tokens",
            self.response_generator.cli_args.prompt_lookup_tokens,
        )
        self.prompt_lookup_adaptive = self.body.get(
            "prompt_lookup_adaptive",
            getattr(self.response_generator.cli_args, "prompt_lookup_adaptive", True),
        )
        self.prompt_lookup_rate_gate = self.body.get(
            "prompt_lookup_rate_gate",
            getattr(self.response_generator.cli_args, "prompt_lookup_rate_gate", True),
        )
        self.prompt_lookup_warmup = self.body.get(
            "prompt_lookup_warmup",
            getattr(self.response_generator.cli_args, "prompt_lookup_warmup", 48),
        )
        self.prompt_lookup_gate = self.body.get(
            "prompt_lookup_gate",
            getattr(self.response_generator.cli_args, "prompt_lookup_gate", 0.12),
        )
        self.prompt_lookup_rate_gate_probe = self.body.get(
            "prompt_lookup_rate_gate_probe",
            getattr(
                self.response_generator.cli_args,
                "prompt_lookup_rate_gate_probe",
                32,
            ),
        )
        self.prompt_lookup_rate_gate_margin = self.body.get(
            "prompt_lookup_rate_gate_margin",
            getattr(
                self.response_generator.cli_args,
                "prompt_lookup_rate_gate_margin",
                0.0,
            ),
        )
        self.adapter = self.body.get("adapters", None)
        self.chat_template_kwargs = self.body.get("chat_template_kwargs")
        # Read before validate_model_parameters runs, so its type is checked
        # here rather than raising TypeError inside the profile lookup.
        if self.chat_template_kwargs is not None and not isinstance(
            self.chat_template_kwargs, dict
        ):
            self._bad_request("chat_template_kwargs must be of type dict")
            return
        sampling_profile = _request_sampling_profile(
            self.response_generator.cli_args, self.chat_template_kwargs
        )
        self.output_token_ceiling = _request_output_ceiling(
            self.response_generator.cli_args, self.chat_template_kwargs
        )
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get(
                "max_tokens", self.response_generator.cli_args.max_tokens
            )
        self.temperature = self.body.get(
            "temperature",
            sampling_profile.get("temperature", self.response_generator.cli_args.temp),
        )
        self.top_p = self.body.get(
            "top_p",
            sampling_profile.get("top_p", self.response_generator.cli_args.top_p),
        )
        self.top_k = self.body.get(
            "top_k",
            sampling_profile.get("top_k", self.response_generator.cli_args.top_k),
        )
        self.min_p = self.body.get(
            "min_p",
            sampling_profile.get("min_p", self.response_generator.cli_args.min_p),
        )
        self.repetition_penalty = self.body.get(
            "repetition_penalty", sampling_profile.get("repetition_penalty", 0.0)
        )
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.presence_penalty = self.body.get(
            "presence_penalty", sampling_profile.get("presence_penalty", 0.0)
        )
        self.presence_context_size = self.body.get("presence_context_size", 20)
        self.frequency_penalty = self.body.get("frequency_penalty", 0.0)
        self.frequency_context_size = self.body.get("frequency_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.1)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", False)
        self.top_logprobs = self.body.get("top_logprobs", -1)
        self.seed = self.body.get("seed", None)
        self.n = self.body.get("n", 1)
        try:
            self.validate_model_parameters()
        except ValueError as e:
            # Parameter validation happens before completion/stream headers are
            # emitted, so malformed requests receive a normal JSON 400 instead
            # of a dropped connection after a streaming response has begun.
            self._bad_request(str(e))
            return

        # Get stop sequences
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words

        # Create the completion request
        request = request_factories[self.path]()
        self.handle_completion(request, stop_words)

    def _bad_request(self, message):
        """Write a JSON 400 for a malformed request body."""
        self._set_completion_headers(400)
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    def _json_error(self, status, message):
        self._set_completion_headers(status)
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    def _json_ok(self, payload):
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
        self.wfile.flush()

    def _admin_authorized(self):
        """Gate the admin routes on the configured bearer key.

        The completion API carries no authentication, so there is no house
        convention to copy; these routes mutate serving state, so they take the
        bearer header the OpenAI-compatible clients of this server already
        speak. With no key configured the routes do not exist at all: an
        unauthenticated write endpoint must never be the default.
        """
        key = getattr(self.response_generator.cli_args, "soft_reload_key", None)
        key = key or os.environ.get("MLX_LM_SOFT_RELOAD_KEY") or None
        if not key:
            self._json_error(
                404,
                "Soft reload is disabled. Start the server with "
                "--soft-reload-key (or MLX_LM_SOFT_RELOAD_KEY) to enable it.",
            )
            return False
        header = self.headers.get("Authorization") or ""
        presented = header[7:] if header.startswith("Bearer ") else ""
        if not hmac.compare_digest(presented, str(key)):
            self._json_error(401, "Invalid or missing bearer key.")
            return False
        return True

    def _read_json_object(self):
        """Read a JSON object body. Writes its own error and returns None."""
        content_length = self.headers.get("Content-Length")
        try:
            content_length = int(content_length)
        except (TypeError, ValueError):
            self._json_error(411, "Content-Length header is required")
            return None
        try:
            body = json.loads(self.rfile.read(content_length).decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._json_error(400, f"Invalid JSON in request body: {e}")
            return None
        if not isinstance(body, dict):
            self._json_error(400, "Request should be a JSON dictionary")
            return None
        return body

    def handle_soft_reload(self):
        """Apply a whitelisted configuration change without a model reload."""
        if not self._admin_authorized():
            return
        body = self._read_json_object()
        if body is None:
            return
        config = body.get("config", {})
        drain_timeout = body.get("drain_timeout")
        if drain_timeout is not None and (
            isinstance(drain_timeout, bool)
            or not isinstance(drain_timeout, (int, float))
            or drain_timeout < 0
        ):
            self._json_error(400, "drain_timeout must be a non-negative number")
            return
        try:
            report = self.response_generator.soft_reload(
                config, drain_timeout=drain_timeout
            )
        except SoftReloadRestartRequired as e:
            self._json_error(409, str(e))
        except SoftReloadError as e:
            self._json_error(400, str(e))
        except SoftReloadBusy as e:
            self._json_error(503, str(e))
        else:
            self._json_ok(report)

    def handle_effective_config(self):
        """Report the live value of every soft-reloadable key."""
        if not self._admin_authorized():
            return
        cli_args = self.response_generator.cli_args
        self._json_ok(
            {
                "object": "effective_config",
                "effective_config": read_effective_config(cli_args),
                "restart_required": dict(SOFT_RELOAD_RESTART_KEYS),
                "prompt_cache": {
                    "entries": len(self.response_generator.prompt_cache),
                    "bytes": int(
                        getattr(self.response_generator.prompt_cache, "nbytes", 0)
                    ),
                },
                "inflight": self.response_generator.inflight,
            }
        )

    def _validate(
        self,
        name,
        expected_type,
        min_val=None,
        max_val=None,
        optional=False,
        whitelist=None,
    ):
        value = getattr(self, name)
        if optional and value is None:
            return
        if not isinstance(value, expected_type):
            try:
                allowed = tuple(et.__name__ for et in expected_type)
            except TypeError:
                allowed = expected_type.__name__
            raise ValueError(f"{name} must be of type {allowed}")
        if whitelist is not None and value in whitelist:
            return
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be at most {max_val}")

    def validate_parallel_sampling(self, cli_args):
        """Validate ``n`` against the server's parallel-sampling limit.

        Greedy ``n>1`` is refused: at temperature 0 the sampler is argmax, so
        every sample is the same continuation of the same prompt. Returning n
        identical choices would bill n times for one answer, so the request is
        an error rather than a silent duplicate.
        """
        if isinstance(self.n, bool):
            raise ValueError("n must be of type int")
        self._validate("n", int, min_val=1)
        if self.n == 1:
            return
        max_n = getattr(cli_args, "parallel_sampling_max_n", 1) or 1
        if max_n < 2:
            raise ValueError(
                "n>1 is not enabled on this server; start it with "
                "--parallel-sampling-max-n N"
            )
        if self.n > max_n:
            raise ValueError(f"n must be at most {max_n}")
        if self.temperature <= 0:
            raise ValueError(
                "n>1 requires temperature > 0: greedy sampling makes every "
                "sample identical"
            )

    def validate_model_parameters(self):
        """Validate that the passed model parameters have correct types and values."""
        self._validate("stream", bool)
        if getattr(self, "stream_options", None) is not None:
            self._validate("stream_options", dict)
        self._validate(
            "max_tokens",
            int,
            min_val=0,
            max_val=getattr(self, "output_token_ceiling", None),
        )
        self._validate("temperature", (float, int), min_val=0)
        self._validate("top_p", (float, int), min_val=0, max_val=1)
        self._validate("top_k", int, min_val=0)
        self._validate("min_p", (float, int), min_val=0, max_val=1)
        self._validate("num_draft_tokens", int, min_val=0)
        for name in ("prompt_lookup_ngram", "prompt_lookup_tokens"):
            if isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be of type int")
        self._validate("prompt_lookup_ngram", int, min_val=0)
        self._validate("prompt_lookup_tokens", int, min_val=1)
        self._validate("prompt_lookup_adaptive", bool)
        self._validate("prompt_lookup_rate_gate", bool)
        self._validate("prompt_lookup_warmup", int, min_val=1)
        self._validate("prompt_lookup_gate", (float, int), min_val=0, max_val=1)
        self._validate("prompt_lookup_rate_gate_probe", int, min_val=1)
        self._validate(
            "prompt_lookup_rate_gate_margin", (float, int), min_val=0, max_val=1
        )
        self._validate("repetition_penalty", (float, int), min_val=0)
        self._validate("repetition_context_size", int, min_val=0)
        self._validate("presence_penalty", (float, int))
        self._validate("presence_context_size", int, min_val=0)
        self._validate("frequency_penalty", (float, int))
        self._validate("frequency_context_size", int, min_val=0)
        self._validate("logprobs", bool)
        self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])
        self._validate("xtc_probability", float, min_val=0, max_val=1)
        self._validate("xtc_threshold", float, min_val=0, max_val=1)
        self._validate("requested_model", str)
        response_generator = getattr(self, "response_generator", None)
        cli_args = getattr(response_generator, "cli_args", None)
        self.validate_parallel_sampling(cli_args)
        if getattr(cli_args, "single_model", False):
            configured = cli_args.model
            allowed = {"default_model", configured}
            configured_path = Path(configured)
            if configured_path.exists():
                allowed.add(str(configured_path.resolve()))
            if self.requested_model not in allowed:
                raise ValueError(
                    "This server only admits its configured model: "
                    f"{configured}"
                )
        self._validate("adapter", str, optional=True)
        self._validate("seed", int, optional=True)
        self._validate("logit_bias", dict, optional=True)

        if self.logit_bias is not None:
            # Normalize into {int: float}. Malformed entries (null / non-numeric
            # values, bool values, non-integer keys) and non-finite values must
            # all surface as ValueError so the request gets the JSON 400 path,
            # never an uncaught TypeError.
            normalized = {}
            for k, v in self.logit_bias.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError("logit_bias must be a dict of int to float")
                try:
                    key = int(str(k))
                    value = float(v)
                except (ValueError, TypeError):
                    raise ValueError("logit_bias must be a dict of int to float")
                if not math.isfinite(value):
                    raise ValueError("logit_bias values must be finite")
                normalized[key] = value
            self.logit_bias = normalized

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        prompt_cache_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Tuple[Dict[str, Any]]]] = None,
        tokens: Optional[List[int]] = None,
        tool_calls: Optional[List[str]] = None,
        reasoning_text: Optional[str] = None,
        index: int = 0,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            prompt_cache_count (Optional[int]): The portion of prompt_token_count
              that was found in the cache when servicing the request.
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Tuple[Dict[str, Any]]]]): List of outputs from
              _format_top_logprobs, giving info on the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure
            tool_calls (Optional[List[str]]): List of tool calls.
            reasoning_text (Optional[str]): The reasoning text generated by the model.

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": index,
                    "finish_reason": finish_reason,
                },
            ],
        }

        if top_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    dict(i[0], top_logprobs=i) if i else {} for i in top_logprobs
                ]
            }
        elif token_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    dict(id=i, logprob=g) for i, g in zip(tokens, token_logprobs)
                ]
            }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }
            if prompt_cache_count is not None and prompt_cache_count >= 0:
                response["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": prompt_cache_count,
                }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant"}
            if not self.stream:
                # The schema requires "content" field to be present
                if text or not tool_calls:
                    choice[key_name]["content"] = text if text else None
            elif text:
                choice[key_name]["content"] = text
            if reasoning_text:
                choice[key_name]["reasoning"] = reasoning_text
                choice[key_name]["reasoning_content"] = reasoning_text
            if tool_calls:
                choice[key_name]["tool_calls"] = tool_calls
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def _write_terminal_chunk(self, assembler) -> None:
        """Emit one choice's closing stream chunk, once."""
        if assembler.terminal_sent:
            return
        assembler.finalize()
        assembler.terminal_sent = True
        resp = self.generate_response(
            assembler.text,
            assembler.finish_reason,
            tool_calls=assembler.tool_formatter(assembler.tool_calls),
            reasoning_text=assembler.reasoning_text,
            index=assembler.index,
        )
        self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
        self.wfile.flush()

    @staticmethod
    def _merge_choice_responses(responses: List[dict]) -> dict:
        """Fold per-choice responses into one multi-choice response.

        ``usage`` counts the prompt once and the completions of every choice,
        which is how the OpenAI API bills ``n>1``.
        """
        merged = dict(responses[0])
        merged["choices"] = [r["choices"][0] for r in responses]
        usage = merged.get("usage")
        if usage is not None:
            completion = sum(r["usage"]["completion_tokens"] for r in responses)
            usage = dict(usage)
            usage["completion_tokens"] = completion
            usage["total_tokens"] = usage["prompt_tokens"] + completion
            merged["usage"] = usage
        return merged

    def handle_completion(self, request: CompletionRequest, stop_words: List[str]):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (List[int]): The tokenized prompt.
            stop_words (List[str]): A list of stop words
        """
        args = GenerationArguments(
            model=ModelDescription(
                model=self.requested_model,
                draft=self.requested_draft_model,
                adapter=self.adapter,
            ),
            sampling=SamplingArguments(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=self.logit_bias,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                presence_penalty=self.presence_penalty,
                presence_context_size=self.presence_context_size,
                frequency_penalty=self.frequency_penalty,
                frequency_context_size=self.frequency_context_size,
            ),
            stop_words=stop_words,
            max_tokens=self.max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            prompt_lookup_ngram=self.prompt_lookup_ngram,
            prompt_lookup_tokens=self.prompt_lookup_tokens,
            prompt_lookup_adaptive=self.prompt_lookup_adaptive,
            prompt_lookup_rate_gate=self.prompt_lookup_rate_gate,
            prompt_lookup_warmup=self.prompt_lookup_warmup,
            prompt_lookup_gate=self.prompt_lookup_gate,
            prompt_lookup_rate_gate_probe=self.prompt_lookup_rate_gate_probe,
            prompt_lookup_rate_gate_margin=self.prompt_lookup_rate_gate_margin,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            seed=self.seed,
            chat_template_kwargs=self.chat_template_kwargs,
            n=max(1, int(getattr(self, "n", 1))),
        )

        # Keep connection allive during long prompt processing (and also log
        # the progress)
        def keepalive_callback(processed, total):
            logging.info(f"Prompt processing progress: {processed}/{total}")
            if self.stream:
                msg = f": keepalive {processed}/{total}\n\n".encode()
                self.wfile.write(msg)
                self.wfile.flush()

        # Create the token generator
        try:
            ctx, response = self.response_generator.generate(
                request,
                args,
                progress_callback=keepalive_callback,
            )
        except Exception as e:
            # An unsupported request composition is a 400; a closed admission
            # gate is a 503; 404 stays for an unknown model.
            if isinstance(e, SoftReloadBusy):
                status = 503
            elif isinstance(e, RequestCompositionError):
                status = 400
            else:
                status = 404
            self._set_completion_headers(status)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Prepare the headers
        if self.stream:
            self._set_stream_headers(200)
            self.end_headers()
            logging.debug("Starting stream:")
        else:
            self._set_completion_headers(200)
            logging.debug("Starting completion:")

        # One assembler per choice; with n>1 the samples interleave on one
        # stream and each carries its own choice index.
        n = max(1, int(args.n))
        assemblers = [
            _ChoiceAssembler(
                i,
                ctx,
                ToolCallFormatter(ctx.tool_parser, request.tools, self.stream),
                logprobs=args.logprobs,
                top_logprobs=args.top_logprobs,
            )
            for i in range(n)
        ]

        try:
            for gen in response:
                logging.debug(gen.text)

                assembler = assemblers[gen.index]
                current_state = assembler.feed(gen)

                if (
                    self.stream
                    and current_state != "tool"
                    and assembler.has_pending_stream_text()
                ):
                    text, tool_calls, reasoning_text = (
                        assembler.take_stream_payload()
                    )
                    resp = self.generate_response(
                        text,
                        None,
                        tool_calls=tool_calls,
                        reasoning_text=reasoning_text,
                        index=assembler.index,
                    )
                    self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                    self.wfile.flush()

                # Close a finished sample right away instead of holding its
                # terminal chunk until every other sample is done.
                if self.stream and gen.finish_reason is not None:
                    self._write_terminal_chunk(assembler)

            for assembler in assemblers:
                assembler.finalize()
            completion_tokens = sum(len(a.tokens) for a in assemblers)

            if self.stream:
                for assembler in assemblers:
                    self._write_terminal_chunk(assembler)
                if (
                    self.stream_options is not None
                    and self.stream_options.get("include_usage")
                ):
                    resp = self.completion_usage_response(
                        len(ctx.prompt),
                        completion_tokens,
                        ctx.prompt_cache_count,
                    )
                    self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write("data: [DONE]\n\n".encode())
                self.wfile.flush()
            else:
                resp = self._merge_choice_responses(
                    [
                        self.generate_response(
                            a.text,
                            a.finish_reason,
                            len(ctx.prompt),
                            len(a.tokens),
                            ctx.prompt_cache_count,
                            token_logprobs=a.token_logprobs,
                            top_tokens=a.top_tokens,
                            tokens=a.tokens,
                            reasoning_text=a.reasoning_text,
                            tool_calls=a.tool_formatter(a.tool_calls),
                            index=a.index,
                        )
                        for a in assemblers
                    ]
                )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    response_debug = json.dumps(resp, indent="\t")
                    logging.debug(f"Outgoing Response: {response_debug}")

                response_json = json.dumps(resp).encode()
                self.send_header("Content-Length", str(len(response_json)))
                self.end_headers()
                self.wfile.write(response_json)
                self.wfile.flush()
        finally:
            ctx.stop()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        prompt_cache_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        if prompt_cache_count is not None and prompt_cache_count >= 0:
            response["usage"]["prompt_tokens_details"] = {
                "cached_tokens": prompt_cache_count,
            }
        return response

    def handle_chat_completions(self) -> CompletionRequest:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        return CompletionRequest(
            "chat",
            "",
            body["messages"],
            body.get("tools") or None,
            body.get("role_mapping"),
        )

    def handle_text_completions(self) -> CompletionRequest:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return CompletionRequest(
            "text",
            self.body["prompt"],
            [],
            None,
            None,
        )

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        if self.path.startswith("/v1/models"):
            self.handle_models_request()
        elif self.path == "/health":
            self.handle_health_check()
        elif self.path == EFFECTIVE_CONFIG_PATH:
            self.handle_effective_config()
        elif self.path == "/v1/status/qwen4-qsa-nax":
            self.handle_qwen4_qsa_nax_status()
        elif self.path == "/v1/status/qwen4-qsa-stage1":
            self.handle_qwen4_qsa_stage1_status()
        elif self.path == "/v1/status/qwen4-qsa-indexed":
            self.handle_qwen4_qsa_indexed_status()
        elif self.path == "/v1/status/qwen4-ple-compile":
            self.handle_qwen4_ple_compile_status()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_qwen4_qsa_nax_status(self):
        """Expose bounded NAX admission/engagement evidence."""

        from mlx_lm.models.qwen4_exp import qsa_nax_admission_status

        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(qsa_nax_admission_status()).encode())
        self.wfile.flush()

    def handle_qwen4_qsa_stage1_status(self):
        """Expose bounded stage-one engagement evidence."""

        from mlx_lm.models.qwen4_exp import qsa_stage1_status

        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(qsa_stage1_status()).encode())
        self.wfile.flush()

    def handle_qwen4_qsa_indexed_status(self):
        """Expose bounded indexed-QSA engagement evidence."""

        from mlx_lm.models.qwen4_qsa_indexed import qsa_indexed_status

        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(qsa_indexed_status()).encode())
        self.wfile.flush()

    def handle_qwen4_ple_compile_status(self):
        """Expose bounded compiled-PLE-chain receipts.

        The lever is default-on for every Flash-Next request, so ``fallbacks``
        being 0 is a production invariant; this is the only way to read it
        from outside the process.  Read-only: it never resets the counters.
        """

        from mlx_lm.models.qwen4_exp import qwen4_ple_compile_status

        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(qwen4_ple_compile_status()).encode())
        self.wfile.flush()

    def handle_health_check(self):
        """
        Handle a GET request for the /health endpoint.

        200 says the process listens AND the generation thread still runs;
        503 says that thread stopped on its own. The endpoint answered 200
        unconditionally before, so a dead generator read as healthy. This is
        liveness, not the serving path: only a real completion proves that
        generation still answers.
        """
        report = self.response_generator.health_report()
        self._set_completion_headers(200 if report["healthy"] else 503)
        self.end_headers()

        self.wfile.write(
            json.dumps({k: v for k, v in report.items() if k != "healthy"}).encode()
        )
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        configured_model = self.response_generator.cli_args.model
        if getattr(self.response_generator.cli_args, "single_model", False):
            model_path = Path(configured_model)
            model_id = (
                str(model_path.resolve()) if model_path.exists() else configured_model
            )
            response = {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": self.created,
                    }
                ],
            }
            self.wfile.write(json.dumps(response).encode())
            self.wfile.flush()
            return

        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            if filter_repo_id is not None and repo.repo_id != filter_repo_id:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        if configured_model:
            model_path = Path(configured_model)
            if model_path.exists():
                model_id = str(model_path.resolve())
                models.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": self.created,
                    }
                )

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


def _run_http_server(
    host: str,
    port: int,
    response_generator,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            response_generator,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        response_generator.stop_and_join()
        # Process shutdown: restore QuantizedLinear, flush overlay caches and
        # stop the reaper thread without reinstalling the patch.
        _release_int8_prefill_overlay(reapply=False)


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    group = mx.distributed.init()
    prompt_cache = AutomaticPrefixCache(
        model_provider.cli_args.prompt_cache_size,
        max_bytes=(
            model_provider.cli_args.prompt_cache_bytes
            if model_provider.cli_args.prompt_cache_bytes is not None
            else 1 << 63
        ),
    )
    response_generator = ResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _run_http_server(host, port, response_generator)
    else:
        response_generator.join()


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help=(
            "Advertise and admit only --model. This prevents a request from "
            "dynamically loading another locally cached model."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: x.split(","),
        default="*",
        help="Allowed origins (default: *)",
    )
    parser.add_argument(
        "--soft-reload-key",
        type=str,
        default=None,
        help=(
            "Bearer key that enables the soft-reload admin routes "
            f"(POST {SOFT_RELOAD_PATH}, GET {EFFECTIVE_CONFIG_PATH}). "
            "Without it, or MLX_LM_SOFT_RELOAD_KEY, the routes stay disabled."
        ),
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
    parser.add_argument(
        "--self-mtp",
        action="store_true",
        help=(
            "Enable the model's internal MTP head for exact greedy or "
            "temperature-only requests. Transformed sampling and APC hits "
            "fail closed to ordinary decoding."
        ),
    )
    parser.add_argument(
        "--self-mtp-transformed-verifier",
        action="store_true",
        help=(
            "Admit top-p/top-k/min-p sampling into self-MTP with an exact "
            "transformed-distribution verifier (the same transform is applied "
            "to draft and target before residual acceptance). Active XTC "
            "(temperature > 0) still fails closed. Default: off."
        ),
    )
    parser.add_argument(
        "--self-mtp-num-draft",
        type=int,
        default=1,
        choices=range(1, 8),
        metavar="{1..7}",
        help="MTP draft depth. Qwen4 is trained at depth 1 (default: 1).",
    )
    parser.add_argument(
        "--self-mtp-max-lanes",
        type=int,
        default=16,
        metavar="N",
        help=(
            "Compute-saturation ceiling on concurrent batched self-MTP lanes. "
            "The M=(k+1)N verify forward saturates the GPU near this many "
            "lanes; past it aggregate throughput falls even when memory "
            "permits more. Measured knee for the dense Qwen3.8-27B is 16 "
            "(N=16: 270 t/s agg vs N=40: 52). Default: 16. Raise for lighter "
            "(MoE) models that saturate later; the memory envelope still caps."
        ),
    )
    parser.add_argument(
        "--self-mtp-lane-transient-gib",
        type=float,
        default=None,
        metavar="GIB",
        help=(
            "Per-lane k=2 verify-transient the admission controller budgets. "
            "Unset uses the MoE-calibrated 1.76 GiB; dense models measure "
            "higher (~3.1 GiB for Qwen3.8-27B). Raise so fewer lanes are "
            "admitted at long context where the transient dominates."
        ),
    )
    parser.add_argument(
        "--self-mtp-adaptive-depth-ceiling",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Adapt the MTP draft depth per request between "
            "--self-mtp-num-draft (the floor/native depth) and this ceiling, "
            "expanding only on sustained full-native-prefix acceptance and "
            "backing off when it falls. Unset keeps today's fixed depth. "
            "The 'Self-MTP admitted' log keeps k=<floor> numeric and adds "
            "adaptive_ceiling=<N> when this flag is set."
        ),
    )
    parser.add_argument(
        "--self-mtp-persistent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep and teacher-force the MTP cache across committed tokens.",
    )
    parser.add_argument(
        "--self-mtp-rate-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure once and fall back when self-MTP is slower (default: on).",
    )
    parser.add_argument(
        "--self-mtp-allow-quantized-kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow self-MTP (single-lane and batched) to run with a quantized "
            "KV target cache when --kv-bits is set (default: off). The batched "
            "transaction is bit-exact on quantized caches; windowed MTP stays "
            "unbatchable regardless."
        ),
    )
    parser.add_argument(
        "--self-mtp-window-size",
        type=int,
        default=0,
        help=(
            "Bound only the persistent MTP draft-head cache to this recent "
            "window (0 disables; target verification remains full-context)."
        ),
    )
    parser.add_argument(
        "--self-mtp-share-qsa-indices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse QSA top-k blocks after the first step of each chained MTP "
            "draft cycle (default: off; target verification is unchanged)."
        ),
    )
    parser.add_argument(
        "--self-mtp-share-qsa-indices-min-prompt-tokens",
        type=int,
        default=0,
        help=(
            "Enable MTP QSA top-k sharing only at or above this full prompt "
            "length (default: 0)."
        ),
    )
    parser.add_argument(
        "--self-mtp-window-sink-size",
        type=int,
        default=4,
        help="Attention-sink tokens retained with the MTP draft window (default: 4).",
    )
    parser.add_argument(
        "--self-mtp-window-min-prompt-tokens",
        type=int,
        default=0,
        help=(
            "Enable --self-mtp-window-size only at or above this full prompt "
            "length (default: 0)."
        ),
    )
    parser.add_argument(
        "--self-mtp-apc-retain-min-prompt-tokens",
        type=int,
        default=64,
        help=(
            "Retain a sidecar-less APC hit only when it saves at least this "
            "many prompt tokens; smaller hits are discarded so self-MTP can "
            "run (default: 64)."
        ),
    )
    parser.add_argument(
        "--prompt-lookup-ngram",
        type=int,
        default=0,
        help="Enable draft-free prompt-lookup (n-gram) speculative decoding with "
        "this max n-gram size (0 disables). Great for code editing / long copies.",
    )
    parser.add_argument(
        "--prompt-lookup-tokens",
        type=int,
        default=8,
        help="Max tokens to propose per prompt-lookup step. Default: 8.",
    )
    parser.add_argument(
        "--prompt-lookup-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Latch prompt lookup to ordinary decode when accepted proposals "
            "stay below the configured gate (default: enabled)."
        ),
    )
    parser.add_argument(
        "--prompt-lookup-rate-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure prompt-lookup versus ordinary decode once and delatch if "
            "it is not faster (default: enabled)."
        ),
    )
    parser.add_argument("--prompt-lookup-warmup", type=int, default=48)
    parser.add_argument("--prompt-lookup-gate", type=float, default=0.12)
    parser.add_argument("--prompt-lookup-rate-gate-probe", type=int, default=32)
    parser.add_argument(
        "--prompt-lookup-rate-gate-margin", type=float, default=0.0
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    parser.add_argument(
        "--thinking-sampling-profile",
        type=json.loads,
        default=None,
        help=(
            "JSON sampling defaults used when effective enable_thinking=true. "
            "Explicit request parameters override each field."
        ),
    )
    parser.add_argument(
        "--nonthinking-sampling-profile",
        type=json.loads,
        default=None,
        help=(
            "JSON sampling defaults used when effective enable_thinking=false. "
            "Explicit request parameters override each field."
        ),
    )
    parser.add_argument(
        "--thinking-output-ceiling",
        type=int,
        default=None,
        help=(
            "Reject thinking-mode requests whose max token budget exceeds this "
            "ceiling. This does not change --max-tokens."
        ),
    )
    parser.add_argument(
        "--nonthinking-output-ceiling",
        type=int,
        default=None,
        help=(
            "Reject non-thinking requests whose max token budget exceeds this "
            "ceiling. This does not change --max-tokens."
        ),
    )
    parser.add_argument(
        "--decode-concurrency",
        type=int,
        default=32,
        help="When a request is batchable then decode that many requests in parallel",
    )
    parser.add_argument(
        "--prompt-concurrency",
        type=int,
        default=8,
        help="When a request is batchable then process that many prompts in parallel",
    )
    parser.add_argument(
        "--parallel-sampling-state-budget-gb",
        type=float,
        default=None,
        help=(
            "Cache-state ceiling for one n>1 request. A request whose "
            "replicated per-sample caches would exceed it is refused. "
            "Default: the device's remaining recommended working set."
        ),
    )
    parser.add_argument(
        "--parallel-sampling-max-n",
        type=int,
        default=1,
        help=(
            "Largest OpenAI 'n' this server accepts. 1 (the default) refuses "
            "n>1. Samples share one prefill and decode as one row each."
        ),
    )
    parser.add_argument(
        "--parallel-sampling-mtp",
        type=str,
        default="mtp",
        choices=list(PARALLEL_SAMPLING_MTP_MODES),
        help=(
            "What to do when an n>1 request would otherwise use self-MTP. "
            "'mtp' (default) uses persistent batched self-MTP; 'plain' serves "
            "with MTP disabled; 'refuse' rejects the composition."
        ),
    )
    parser.add_argument(
        "--state-budget-gb",
        type=float,
        default=None,
        help=(
            "Cap projected model-state bytes (KV cache and recurrent/hybrid "
            "state) across concurrent requests to this budget (GiB). The "
            "per-model cost is measured with a short probe at load. The "
            "budget should already account for weights and activation "
            "headroom; no extra factor is applied. Default: off (admission "
            "is count-based only)."
        ),
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prefill processing (default: 2048)",
    )
    parser.add_argument(
        "--prompt-batch-window",
        type=int,
        default=None,
        help=(
            "Maximum queued prompts considered for length-aware admission "
            "(default: 1, preserves FIFO; try 4 times --prompt-concurrency)"
        ),
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of distinct KV caches to hold in the prompt cache",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        help="Maximum size in bytes of the KV caches",
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
        type=int,
        help="When KV bits are set, start quantizing the cache from this step.",
        default=DEFAULT_QUANTIZED_KV_START,
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--int8-prefill",
        action="store_true",
        help=(
            "Route prefill-sized MLP matmuls through W8A8 int8 GEMMs on the "
            "M5 GPU neural accelerators (decode keeps the quantized kernels). "
            "Requires an M5-class GPU. Maps to the MLX_LM_INT8_PREFILL env var."
        ),
    )
    return parser


def _validate_adaptive_depth_ceiling(args):
    """Enforce 1 <= --self-mtp-num-draft <= ceiling <= MAX_DRAFT_TOKENS."""
    ceiling = getattr(args, "self_mtp_adaptive_depth_ceiling", None)
    if ceiling is None:
        return
    num_draft = args.self_mtp_num_draft
    if not 1 <= num_draft <= ceiling <= MAX_DRAFT_TOKENS:
        raise ValueError(
            f"--self-mtp-adaptive-depth-ceiling {ceiling} requires "
            f"1 <= --self-mtp-num-draft ({num_draft}) <= ceiling <= "
            f"{MAX_DRAFT_TOKENS} (the M5 verify-width cap)"
        )


def _configure_process_wired_limit(args):
    """Apply the legacy server clamp unless internal MTP owns the stream.

    Qwen4 self-MTP plus its separately cached head runs close to the M5 working
    set ceiling. Setting the recommended limit before loading this 100+ GB
    model reproducibly causes a Metal watchdog timeout on its first PLE
    command; MLX's existing default limit completes the same stream.
    """
    if not mx.metal.is_available() or getattr(args, "self_mtp", False):
        return None
    wired_limit = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(wired_limit)
    return wired_limit


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    for name in (
        "self_mtp_window_size",
        "self_mtp_window_sink_size",
        "self_mtp_window_min_prompt_tokens",
        "self_mtp_share_qsa_indices_min_prompt_tokens",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.self_mtp_window_size and not args.self_mtp_persistent:
        parser.error("--self-mtp-window-size requires --self-mtp-persistent")
    try:
        _validate_adaptive_depth_ceiling(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_kv_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    _configure_process_wired_limit(args)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    _maybe_apply_int8_prefill(args)
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
