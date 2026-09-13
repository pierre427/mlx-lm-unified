# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Qwen4-aligned per-plane segmentation and replayable traffic calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import time
from typing import Iterable, Mapping, Optional
import zlib

from .cache_planes import CachePlaneKind
from .cache_scheduler import PlaneRetentionManager, PlaneRetentionRecord


_RECOMPUTE_BANDWIDTH_BYTES_PER_US = 50_000.0
_RECOMPUTE_SERIAL_US_PER_TOKEN = {
    CachePlaneKind.PROMPT_HOST: 0.01,
    CachePlaneKind.ATTENTION_KV: 0.15,
    CachePlaneKind.ATTENTION_RING: 0.12,
    CachePlaneKind.QSA_SUMMARY: 0.04,
    CachePlaneKind.GDN_RECURRENT: 0.75,
    CachePlaneKind.MTP_DRAFT: 0.08,
}


def recompute_cost_proxy_us(
    kind: CachePlaneKind,
    token_count: int,
    logical_bytes: int,
) -> float:
    """Return a documented model-free eviction proxy, never a GPU estimate."""
    if token_count < 0 or logical_bytes < 0:
        raise ValueError("recomputation proxy inputs must be non-negative")
    serial_cost = token_count * _RECOMPUTE_SERIAL_US_PER_TOKEN.get(kind, 0.05)
    memory_cost = logical_bytes / _RECOMPUTE_BANDWIDTH_BYTES_PER_US
    return 25.0 + serial_cost + memory_cost


@dataclass(frozen=True)
class SegmentValueCandidate:
    """Measured evidence for protecting one immutable cache segment."""

    segment_id: str
    expected_future_hits: float
    recompute_cost_us: float
    branch_fanout: int
    resident_bytes: int
    stable: bool = True
    contains_mutable_request_data: bool = False

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment value candidate needs an id")
        if not math.isfinite(self.expected_future_hits) or self.expected_future_hits < 0:
            raise ValueError("expected future hits must be finite and non-negative")
        if not math.isfinite(self.recompute_cost_us) or self.recompute_cost_us < 0:
            raise ValueError("recompute cost must be finite and non-negative")
        if self.branch_fanout < 1 or self.resident_bytes < 1:
            raise ValueError("fanout and resident bytes must be positive")

    @property
    def expected_value_us(self) -> float:
        return self.expected_future_hits * self.recompute_cost_us * self.branch_fanout

    @property
    def value_per_resident_byte(self) -> float:
        return self.expected_value_us / self.resident_bytes

    @property
    def retention_eligible(self) -> bool:
        return self.stable and not self.contains_mutable_request_data


@dataclass(frozen=True)
class ProtectedSegmentHotset:
    """Deterministic, byte-bounded protection recommendation."""

    segment_ids: tuple[str, ...]
    resident_bytes: int
    expected_value_us: float


def select_protected_segment_hotset(
    candidates: Iterable[SegmentValueCandidate], *, byte_budget: int
) -> ProtectedSegmentHotset:
    """Greedily admit stable segments by expected saved work per byte.

    This emits metadata only. Callers retain authority over materialization,
    eviction, precision, and correctness-bearing cache ownership.
    """
    if byte_budget < 0:
        raise ValueError("byte budget must be non-negative")
    eligible = sorted(
        (candidate for candidate in candidates if candidate.retention_eligible),
        key=lambda candidate: (
            -candidate.value_per_resident_byte,
            -candidate.expected_value_us,
            candidate.segment_id,
        ),
    )
    selected = []
    resident_bytes = 0
    expected_value_us = 0.0
    for candidate in eligible:
        if resident_bytes + candidate.resident_bytes > byte_budget:
            continue
        selected.append(candidate.segment_id)
        resident_bytes += candidate.resident_bytes
        expected_value_us += candidate.expected_value_us
    return ProtectedSegmentHotset(
        tuple(selected), resident_bytes, expected_value_us
    )


@dataclass(frozen=True)
class Qwen4CacheGeometry:
    layers: int = 48
    full_attention_stride: int = 4
    num_kv_heads: int = 2
    head_dim: int = 256
    linear_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    mtp_layers: int = 1
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    bytes_per_element: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.layers,
            self.full_attention_stride,
            self.num_kv_heads,
            self.head_dim,
            self.linear_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.mtp_layers,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
            self.bytes_per_element,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in positive
        ):
            raise ValueError("Qwen4 cache geometry values must be positive integers")
        if self.layers % self.full_attention_stride:
            raise ValueError("full-attention stride must divide the layer count")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer ratio must divide its budget")

    @property
    def full_attention_layers(self) -> int:
        return self.layers // self.full_attention_stride

    @property
    def linear_attention_layers(self) -> int:
        return self.layers - self.full_attention_layers

    @property
    def qsa_block_tokens(self) -> int:
        return self.indexer_compress_ratio

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def dense_shortcircuit_max_tokens(self) -> int:
        return self.indexer_budget + self.indexer_compress_ratio - 1

    @property
    def full_kv_bytes_per_token(self) -> int:
        return (
            self.full_attention_layers
            * self.num_kv_heads
            * self.head_dim
            * 2
            * self.bytes_per_element
        )

    @property
    def mtp_kv_bytes_per_token(self) -> int:
        return (
            self.mtp_layers
            * self.num_kv_heads
            * self.head_dim
            * 2
            * self.bytes_per_element
        )

    @property
    def gdn_recurrent_bytes(self) -> int:
        return (
            self.linear_attention_layers
            * self.linear_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * self.bytes_per_element
        )

    @property
    def qsa_bytes_per_block(self) -> int:
        return (
            self.full_attention_layers
            * self.indexer_kv_heads
            * self.indexer_head_dim
            * self.bytes_per_element
        )

    def plane_bytes(self, kind: CachePlaneKind, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token count must be non-negative")
        if kind == CachePlaneKind.PROMPT_HOST:
            return token_count * 4
        if kind in (CachePlaneKind.ATTENTION_KV, CachePlaneKind.ATTENTION_RING):
            return token_count * self.full_kv_bytes_per_token
        if kind == CachePlaneKind.QSA_SUMMARY:
            return (token_count // self.qsa_block_tokens) * self.qsa_bytes_per_block
        if kind == CachePlaneKind.GDN_RECURRENT:
            return self.gdn_recurrent_bytes
        if kind == CachePlaneKind.MTP_DRAFT:
            return token_count * self.mtp_kv_bytes_per_token
        if kind == CachePlaneKind.PLE_HINTS:
            return max(64, token_count // 8)
        return max(64, token_count * 8)


@dataclass(frozen=True)
class QSAWindow:
    token_count: int
    complete_blocks: int
    incomplete_tail_tokens: int
    block_topk: int
    dense_shortcircuit: bool


def qsa_window(token_count: int, geometry: Qwen4CacheGeometry) -> QSAWindow:
    if token_count < 0:
        raise ValueError("token count must be non-negative")
    blocks, tail = divmod(token_count, geometry.qsa_block_tokens)
    return QSAWindow(
        token_count,
        blocks,
        tail,
        geometry.block_topk,
        blocks <= geometry.block_topk,
    )


_DEFAULT_SEGMENT_BLOCKS = (
    (CachePlaneKind.PROMPT_HOST, 128),
    (CachePlaneKind.ATTENTION_KV, 64),
    (CachePlaneKind.ATTENTION_RING, 64),
    (CachePlaneKind.QSA_SUMMARY, 128),
    (CachePlaneKind.GDN_RECURRENT, 128),
    (CachePlaneKind.MTP_DRAFT, 16),
    (CachePlaneKind.PLE_HINTS, 128),
    (CachePlaneKind.COMPILED_SCHEDULE, 128),
)


@dataclass(frozen=True)
class SegmentationPolicyConfig:
    enabled: bool = False
    inline_threshold_bytes: int = 2048
    high_entropy_threshold: float = 7.5
    max_delta_depth: int = 8
    max_fragmentation_ratio: float = 0.15
    segment_blocks: tuple[tuple[CachePlaneKind, int], ...] = _DEFAULT_SEGMENT_BLOCKS

    def __post_init__(self) -> None:
        if self.inline_threshold_bytes < 0 or self.max_delta_depth < 1:
            raise ValueError("segmentation thresholds are invalid")
        if not 0 <= self.high_entropy_threshold <= 8:
            raise ValueError("entropy threshold must be between zero and eight")
        if not 0 <= self.max_fragmentation_ratio <= 1:
            raise ValueError("fragmentation threshold must be between zero and one")
        kinds = tuple(kind for kind, _ in self.segment_blocks)
        if len(set(kinds)) != len(kinds):
            raise ValueError("segment block policy contains duplicate planes")
        if any(blocks < 1 for _, blocks in self.segment_blocks):
            raise ValueError("segment block counts must be positive")

    @classmethod
    def uniform(
        cls,
        segment_blocks: int,
        **kwargs,
    ) -> "SegmentationPolicyConfig":
        return cls(
            segment_blocks=tuple(
                (kind, segment_blocks) for kind, _ in _DEFAULT_SEGMENT_BLOCKS
            ),
            **kwargs,
        )


@dataclass(frozen=True)
class PlaneSegmentCandidate:
    kind: CachePlaneKind
    token_start: int
    token_stop: int
    logical_bytes: int
    entropy_bits_per_byte: float
    delta_depth: int
    branch_fanout: int
    recompute_cost_us: float
    reuse_distance: int
    stable: bool
    contains_mutable_request_data: bool

    def __post_init__(self) -> None:
        if not 0 <= self.token_start <= self.token_stop:
            raise ValueError("invalid candidate token interval")
        if self.logical_bytes < 0 or self.delta_depth < 0:
            raise ValueError("candidate bytes and delta depth must be non-negative")
        if self.branch_fanout < 1 or self.reuse_distance < 0:
            raise ValueError("fanout and reuse distance are invalid")
        if not 0 <= self.entropy_bits_per_byte <= 8:
            raise ValueError("entropy must be between zero and eight")
        if not math.isfinite(self.recompute_cost_us) or self.recompute_cost_us < 0:
            raise ValueError("recompute cost must be finite and non-negative")


@dataclass(frozen=True)
class SegmentSlice:
    token_start: int
    token_stop: int
    logical_bytes: int


@dataclass(frozen=True)
class PlaneSegmentationPlan:
    kind: CachePlaneKind
    segments: tuple[SegmentSlice, ...]
    incomplete_tail_tokens: int
    inline: bool
    codec_id: str | None
    compact_delta: bool
    retention_eligible: bool
    fragmentation_bytes: int
    shared_branch_bytes_avoided: int
    reason: str


class PerPlaneSegmentationPolicy:
    """Build 4-token-aligned segment/checkpoint candidates only."""

    _SEGMENT_HEADER_BYTES = 64

    def __init__(
        self,
        config: SegmentationPolicyConfig | None = None,
        geometry: Qwen4CacheGeometry | None = None,
    ) -> None:
        self.config = config or SegmentationPolicyConfig()
        self.geometry = geometry or Qwen4CacheGeometry()
        self._blocks = dict(self.config.segment_blocks)

    def decide(self, candidate: PlaneSegmentCandidate) -> PlaneSegmentationPlan:
        if not self.config.enabled:
            return PlaneSegmentationPlan(
                candidate.kind, (), 0, True, None, False, False, 0, 0, "disabled"
            )
        block_tokens = self.geometry.qsa_block_tokens
        if candidate.token_start % block_tokens:
            raise ValueError("segment start must align to a QSA block")
        complete_stop = candidate.token_stop - candidate.token_stop % block_tokens
        tail = candidate.token_stop - complete_stop
        inline = candidate.logical_bytes <= self.config.inline_threshold_bytes
        codec_id = None
        if not inline and candidate.entropy_bits_per_byte < self.config.high_entropy_threshold:
            codec_id = "zlib-v1"

        segments = self._segments(candidate, complete_stop, inline=inline)
        allocated = sum(item.logical_bytes for item in segments)
        segmentable_bytes = self._segmentable_bytes(candidate, complete_stop)
        fragmentation = max(0, allocated - segmentable_bytes)
        if not inline:
            fragmentation += len(segments) * self._SEGMENT_HEADER_BYTES
        fragmentation_ratio = fragmentation / max(1, candidate.logical_bytes)
        compact = (
            candidate.delta_depth >= self.config.max_delta_depth
            or fragmentation_ratio >= self.config.max_fragmentation_ratio
        )
        eligible = (
            candidate.stable
            and not candidate.contains_mutable_request_data
            and bool(segments)
        )
        shared = candidate.logical_bytes * max(0, candidate.branch_fanout - 1)
        reason = "inline" if inline else "segmented"
        if codec_id is None and not inline:
            reason = "segmented_high_entropy_no_codec"
        return PlaneSegmentationPlan(
            candidate.kind,
            segments,
            tail,
            inline,
            codec_id,
            compact,
            eligible,
            fragmentation,
            shared,
            reason,
        )

    def _segmentable_bytes(
        self, candidate: PlaneSegmentCandidate, complete_stop: int
    ) -> int:
        if complete_stop <= candidate.token_start:
            return 0
        if candidate.kind in (
            CachePlaneKind.GDN_RECURRENT,
            CachePlaneKind.QSA_SUMMARY,
        ):
            return candidate.logical_bytes
        total_tokens = max(1, candidate.token_stop - candidate.token_start)
        fraction = (complete_stop - candidate.token_start) / total_tokens
        return math.ceil(candidate.logical_bytes * fraction)

    def _segments(
        self,
        candidate: PlaneSegmentCandidate,
        complete_stop: int,
        *,
        inline: bool,
    ) -> tuple[SegmentSlice, ...]:
        if complete_stop <= candidate.token_start:
            return ()
        segmentable_bytes = self._segmentable_bytes(candidate, complete_stop)
        if inline:
            return (
                SegmentSlice(
                    candidate.token_start,
                    complete_stop,
                    segmentable_bytes,
                ),
            )
        if candidate.kind == CachePlaneKind.GDN_RECURRENT:
            return (
                SegmentSlice(
                    candidate.token_start,
                    complete_stop,
                    segmentable_bytes,
                ),
            )
        block_count = self._blocks.get(candidate.kind)
        if block_count is None:
            raise ValueError(f"no segment policy for {candidate.kind.value}")
        segment_tokens = block_count * self.geometry.qsa_block_tokens
        segmentable_tokens = complete_stop - candidate.token_start
        result = []
        start = candidate.token_start
        while start < complete_stop:
            stop = min(complete_stop, start + segment_tokens)
            fraction = (stop - start) / segmentable_tokens
            nbytes = math.ceil(segmentable_bytes * fraction)
            result.append(SegmentSlice(start, stop, nbytes))
            start = stop
        return tuple(result)


@dataclass(frozen=True)
class SyntheticTraceEvent:
    event_id: str
    session_id: str
    turn: int
    kind: CachePlaneKind
    token_count: int
    structural_prefix_id: str
    logical_bytes: int
    entropy_class: str
    payload_seed: int
    branch_fanout: int
    delta_depth: int
    reuse_distance: int
    recompute_cost_us: float
    stable: bool
    contains_mutable_request_data: bool

    def sample_payload(self, maximum_bytes: int = 4096) -> bytes:
        size = min(maximum_bytes, max(1, self.logical_bytes))
        if self.entropy_class == "high":
            return random.Random(self.payload_seed).randbytes(size)
        motif = hashlib.sha256(self.structural_prefix_id.encode()).digest()[:16]
        return (motif * math.ceil(size / len(motif)))[:size]

    def candidate(self) -> PlaneSegmentCandidate:
        entropy = 7.9 if self.entropy_class == "high" else 2.0
        return PlaneSegmentCandidate(
            self.kind,
            0,
            self.token_count,
            self.logical_bytes,
            entropy,
            self.delta_depth,
            self.branch_fanout,
            self.recompute_cost_us,
            self.reuse_distance,
            self.stable,
            self.contains_mutable_request_data,
        )


@dataclass(frozen=True)
class SyntheticServingTrace:
    seed: int
    sessions: int
    turns: int
    events: tuple[SyntheticTraceEvent, ...]

    @classmethod
    def generate(
        cls,
        *,
        seed: int = 427,
        sessions: int = 16,
        turns: int = 12,
        geometry: Qwen4CacheGeometry | None = None,
    ) -> "SyntheticServingTrace":
        if sessions < 1 or turns < 1:
            raise ValueError("trace sessions and turns must be positive")
        shape = geometry or Qwen4CacheGeometry()
        rng = random.Random(seed)
        lengths = (128, 512, 2048, 2051, 2052, 4096, 8192)
        kinds = (
            CachePlaneKind.PROMPT_HOST,
            CachePlaneKind.ATTENTION_KV,
            CachePlaneKind.ATTENTION_RING,
            CachePlaneKind.QSA_SUMMARY,
            CachePlaneKind.GDN_RECURRENT,
            CachePlaneKind.MTP_DRAFT,
        )
        events = []
        sequence = 0
        for turn in range(turns):
            for session in range(sessions):
                token_count = lengths[(turn + session) % len(lengths)]
                template_id = f"template-{session % 4}"
                fanout = 1 + (session + turn) % 4
                for kind in kinds:
                    stable = kind not in (
                        CachePlaneKind.ATTENTION_RING,
                        CachePlaneKind.MTP_DRAFT,
                    )
                    mutable = kind in (
                        CachePlaneKind.ATTENTION_RING,
                        CachePlaneKind.MTP_DRAFT,
                    )
                    prefix = (
                        template_id
                        if kind == CachePlaneKind.PROMPT_HOST
                        else f"session-{session}"
                    )
                    entropy_class = (
                        "high"
                        if (sequence + session) % 11 == 0
                        else "low"
                    )
                    events.append(
                        SyntheticTraceEvent(
                            event_id=f"e{sequence:06d}",
                            session_id=f"s{session:03d}",
                            turn=turn,
                            kind=kind,
                            token_count=token_count,
                            structural_prefix_id=prefix,
                            logical_bytes=shape.plane_bytes(kind, token_count),
                            entropy_class=entropy_class,
                            payload_seed=rng.randrange(1 << 63),
                            branch_fanout=fanout,
                            delta_depth=(turn + session) % 12,
                            reuse_distance=(turn * sessions + session) % 97,
                            recompute_cost_us=recompute_cost_proxy_us(
                                kind,
                                token_count,
                                shape.plane_bytes(kind, token_count),
                            ),
                            stable=stable,
                            contains_mutable_request_data=mutable,
                        )
                    )
                    sequence += 1
        return cls(seed, sessions, turns, tuple(events))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        payload = {
            "seed": self.seed,
            "sessions": self.sessions,
            "turns": self.turns,
            "events": [
                {**asdict(event), "kind": event.kind.value}
                for event in self.events
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "SyntheticServingTrace":
        decoded = json.loads(payload)
        events = tuple(
            SyntheticTraceEvent(
                **{**item, "kind": CachePlaneKind(item["kind"])}
            )
            for item in decoded["events"]
        )
        return cls(
            int(decoded["seed"]),
            int(decoded["sessions"]),
            int(decoded["turns"]),
            events,
        )


@dataclass(frozen=True)
class CalibrationResult:
    label: str
    trace_digest: str
    events: int
    host_latency_ms: float
    compression_cost_ms: float
    bytes_before: int
    bytes_after: int
    compression_ratio: float
    segment_hits: int
    segment_misses: int
    hit_rate: float
    partial_hit_events: int
    partial_hit_rate: float
    recomputation_avoided_us: float
    fragmentation_bytes: int
    compactions: int
    shared_branch_bytes_avoided: int
    retained_bytes: int
    mean_reuse_distance: float
    p95_reuse_distance: int


class TraceCalibrator:
    def __init__(
        self,
        policy: PerPlaneSegmentationPolicy,
        plane_budgets: Mapping[CachePlaneKind, int],
    ) -> None:
        self.policy = policy
        self.plane_budgets = dict(plane_budgets)

    def run(
        self, trace: SyntheticServingTrace, *, label: str
    ) -> CalibrationResult:
        retention = PlaneRetentionManager(self.plane_budgets, enabled=True)
        start_ns = time.perf_counter_ns()
        compression_ns = 0
        bytes_before = 0
        bytes_after = 0
        hits = 0
        misses = 0
        partial_events = 0
        recompute_avoided = 0.0
        fragmentation = 0
        compactions = 0
        shared_avoided = 0

        for sequence, event in enumerate(trace.events):
            plan = self.policy.decide(event.candidate())
            if not plan.segments:
                continue
            payload = event.sample_payload()
            bytes_before += len(payload)
            encoded = payload
            if plan.codec_id == "zlib-v1":
                codec_start = time.perf_counter_ns()
                encoded = zlib.compress(payload)
                restored = zlib.decompress(encoded)
                compression_ns += time.perf_counter_ns() - codec_start
                if restored != payload:
                    raise RuntimeError("lossless trace codec changed the payload")
            bytes_after += len(encoded)
            ratio = len(encoded) / max(1, len(payload))
            event_hits = 0
            event_misses = 0
            scope = (
                event.structural_prefix_id
                if event.kind == CachePlaneKind.PROMPT_HOST
                else event.session_id
            )
            segment_cost = event.recompute_cost_us / len(plan.segments)
            for segment in plan.segments:
                key_source = (
                    scope,
                    event.kind.value,
                    segment.token_start,
                    segment.token_stop,
                    plan.codec_id,
                )
                key = hashlib.sha256(repr(key_source).encode()).hexdigest()
                if retention.contains(key):
                    retention.touch(key, sequence)
                    hits += 1
                    event_hits += 1
                    recompute_avoided += segment_cost
                    continue
                misses += 1
                event_misses += 1
                stored_bytes = max(1, math.ceil(segment.logical_bytes * ratio))
                retention.consider(
                    PlaneRetentionRecord(
                        key,
                        scope,
                        event.kind,
                        stored_bytes,
                        segment_cost,
                        sequence,
                        contains_mutable_request_data=(
                            event.contains_mutable_request_data
                        ),
                    )
                )
            if event_hits and event_misses:
                partial_events += 1
            fragmentation += plan.fragmentation_bytes
            compactions += int(plan.compact_delta)
            shared_avoided += plan.shared_branch_bytes_avoided

        host_latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        total_segments = hits + misses
        retained_bytes = sum(item.size_bytes for item in retention.records())
        reuse_distances = sorted(event.reuse_distance for event in trace.events)
        p95_index = max(0, math.ceil(len(reuse_distances) * 0.95) - 1)
        return CalibrationResult(
            label,
            trace.digest,
            len(trace.events),
            host_latency_ms,
            compression_ns / 1e6,
            bytes_before,
            bytes_after,
            bytes_before / max(1, bytes_after),
            hits,
            misses,
            hits / max(1, total_segments),
            partial_events,
            partial_events / max(1, len(trace.events)),
            recompute_avoided,
            fragmentation,
            compactions,
            shared_avoided,
            retained_bytes,
            sum(reuse_distances) / max(1, len(reuse_distances)),
            reuse_distances[p95_index] if reuse_distances else 0,
        )


def pareto_front(results: Iterable[CalibrationResult]) -> tuple[str, ...]:
    rows = tuple(results)

    def dominates(left: CalibrationResult, right: CalibrationResult) -> bool:
        no_worse = (
            left.host_latency_ms <= right.host_latency_ms
            and left.retained_bytes <= right.retained_bytes
            and left.fragmentation_bytes <= right.fragmentation_bytes
            and left.hit_rate >= right.hit_rate
            and left.recomputation_avoided_us >= right.recomputation_avoided_us
        )
        strict = (
            left.host_latency_ms < right.host_latency_ms
            or left.retained_bytes < right.retained_bytes
            or left.fragmentation_bytes < right.fragmentation_bytes
            or left.hit_rate > right.hit_rate
            or left.recomputation_avoided_us > right.recomputation_avoided_us
        )
        return no_worse and strict

    return tuple(
        sorted(
            row.label
            for row in rows
            if not any(other is not row and dominates(other, row) for other in rows)
        )
    )


__all__ = [
    "CalibrationResult",
    "PerPlaneSegmentationPolicy",
    "PlaneSegmentCandidate",
    "PlaneSegmentationPlan",
    "QSAWindow",
    "Qwen4CacheGeometry",
    "ProtectedSegmentHotset",
    "SegmentValueCandidate",
    "SegmentSlice",
    "SegmentationPolicyConfig",
    "SyntheticServingTrace",
    "SyntheticTraceEvent",
    "TraceCalibrator",
    "pareto_front",
    "qsa_window",
    "recompute_cost_proxy_us",
    "select_protected_segment_hotset",
]
