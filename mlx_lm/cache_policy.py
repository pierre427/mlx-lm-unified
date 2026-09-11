# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Default-off placement, codec, and persistence policy for cache planes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Optional, Protocol
import tempfile
import zlib

from .cache_planes import CachePlaneFingerprint, CachePlaneKind, PLEResidencyHints
from .heterogeneous_execution import (
    Engine,
    HeterogeneousExecutionProfile,
    OperationGeometry,
    OperationNode,
    PlanningRefused,
)


def layered_cache_policy_enabled(value: Optional[bool] = None) -> bool:
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_LAYERED_CACHE_POLICY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def cache_plane_operation(kind: CachePlaneKind, operation: str) -> str:
    if not isinstance(kind, CachePlaneKind) or not operation:
        raise ValueError("cache plane operation identity is incomplete")
    return f"cache-plane/{kind.value}/{operation}"


class CachePolicyError(RuntimeError):
    pass


class PersistenceRefused(CachePolicyError):
    pass


class IntegrityError(CachePolicyError):
    pass


def _finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


class PlaneCodec(Protocol):
    codec_id: str
    lossless: bool
    experimental_accuracy_gated: bool

    def encode(self, payload: bytes) -> bytes: ...

    def decode(self, payload: bytes) -> bytes: ...


class IdentityCodec:
    codec_id = "identity-v1"
    lossless = True
    experimental_accuracy_gated = False

    def encode(self, payload: bytes) -> bytes:
        return payload

    def decode(self, payload: bytes) -> bytes:
        return payload


class ZlibCodec:
    codec_id = "zlib-v1"
    lossless = True
    experimental_accuracy_gated = False

    def encode(self, payload: bytes) -> bytes:
        return zlib.compress(payload)

    def decode(self, payload: bytes) -> bytes:
        return zlib.decompress(payload)


class CachePolicyMetrics:
    """Bounded host counters with no device synchronization or timers."""

    _KEYS = (
        "placement_decisions",
        "placement_refusals",
        "persistence_writes",
        "persistence_deduplications",
        "persistence_reads",
        "persistence_rejections",
        "compression_admissions",
        "compression_rejections",
    )

    def __init__(self) -> None:
        self._values = {key: 0 for key in self._KEYS}
        self._lock = Lock()

    def add(self, key: str, amount: int = 1) -> None:
        if key not in self._values:
            raise KeyError(f"unknown cache policy metric: {key}")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("metric amount must be a non-negative integer")
        with self._lock:
            self._values[key] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            result = dict(self._values)
        result["device_synchronizations"] = 0
        result["timed_hot_path_sections"] = 0
        return result


@dataclass(frozen=True)
class CachePlanePlacementRequest:
    request_id: str
    lineage_id: str
    kind: CachePlaneKind
    operation: str
    geometry: OperationGeometry
    ple_hints: PLEResidencyHints | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.lineage_id or not self.operation:
            raise ValueError("placement request identity is incomplete")
        if not isinstance(self.kind, CachePlaneKind):
            raise ValueError("placement cache plane kind is invalid")
        if self.ple_hints is not None and not isinstance(
            self.ple_hints, PLEResidencyHints
        ):
            raise ValueError("PLE residency hints are invalid")


@dataclass(frozen=True)
class CachePlanePlacementDecision:
    engine: Engine | None
    reason: str
    critical_path_us: float | None
    advisory_ple_hints_seen: bool


class CachePlanePlacementPolicy:
    """Use exact host-profile evidence; PLE hints never select an engine."""

    def __init__(
        self,
        profile: HeterogeneousExecutionProfile,
        *,
        enabled: bool = False,
        metrics: CachePolicyMetrics | None = None,
    ) -> None:
        self.profile = profile
        self.enabled = bool(enabled)
        self.metrics = metrics or CachePolicyMetrics()

    def decide(
        self, request: CachePlanePlacementRequest
    ) -> CachePlanePlacementDecision:
        hint_seen = request.ple_hints is not None
        if not self.enabled:
            self.metrics.add("placement_refusals")
            return CachePlanePlacementDecision(
                None, "disabled", None, hint_seen
            )
        try:
            plan = self.profile.plan(
                (
                    OperationNode(
                        request.request_id,
                        cache_plane_operation(request.kind, request.operation),
                        request.geometry,
                        f"cache-plane:{request.kind.value}",
                    ),
                )
            )
        except PlanningRefused as error:
            self.metrics.add("placement_refusals")
            return CachePlanePlacementDecision(
                None, f"unqualified:{error}", None, hint_seen
            )
        self.metrics.add("placement_decisions")
        return CachePlanePlacementDecision(
            plan.placements[0].engine,
            "measured_host_capability",
            plan.critical_path_us,
            hint_seen,
        )


@dataclass(frozen=True)
class StablePlaneRecord:
    kind: CachePlaneKind
    fingerprint: CachePlaneFingerprint
    model_fingerprint: str
    generation: int
    payload: bytes
    stable: bool = True
    contains_mutable_request_data: bool = False

    def __post_init__(self) -> None:
        if self.fingerprint.kind != self.kind:
            raise ValueError("plane kind and fingerprint disagree")
        if not self.model_fingerprint:
            raise ValueError("model fingerprint is required")
        if isinstance(self.generation, bool) or not isinstance(
            self.generation, int
        ):
            raise ValueError("plane generation must be a non-negative integer")
        if self.generation < 0:
            raise ValueError("plane generation must be a non-negative integer")
        if not isinstance(self.payload, bytes):
            raise ValueError("persistent plane payload must be bytes")


@dataclass(frozen=True)
class AccuracyGate:
    model_fingerprint: str
    min_samples: int
    max_abs_logit_error: float
    require_exact_tokens: bool = True

    def __post_init__(self) -> None:
        if (
            not self.model_fingerprint
            or isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < 1
        ):
            raise ValueError("accuracy gate identity and sample count are required")
        _finite_nonnegative(
            "accuracy gate logit error", self.max_abs_logit_error
        )

    @property
    def digest(self) -> str:
        payload = (
            self.model_fingerprint,
            self.min_samples,
            self.max_abs_logit_error,
            self.require_exact_tokens,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()


@dataclass(frozen=True)
class AccuracyEvidence:
    model_fingerprint: str
    plane_fingerprint_digest: str
    codec_id: str
    samples: int
    max_abs_logit_error: float
    exact_tokens: bool

    def __post_init__(self) -> None:
        if not self.model_fingerprint or not self.plane_fingerprint_digest:
            raise ValueError("accuracy evidence fingerprints are required")
        if (
            not self.codec_id
            or isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples < 0
        ):
            raise ValueError("accuracy evidence codec and samples are required")
        _finite_nonnegative(
            "accuracy evidence logit error", self.max_abs_logit_error
        )

    @property
    def digest(self) -> str:
        payload = (
            self.model_fingerprint,
            self.plane_fingerprint_digest,
            self.codec_id,
            self.samples,
            self.max_abs_logit_error,
            self.exact_tokens,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()


@dataclass(frozen=True)
class CompressionAdmission:
    accepted: bool
    reason: str
    codec_id: str
    model_fingerprint: str
    plane_fingerprint_digest: str
    gate_digest: str
    evidence_digest: str


class ColdKVCompressionPolicy:
    """Admission only; no lossy codec implementation is supplied here."""

    _COLD_KV_PLANES = frozenset(
        {CachePlaneKind.ATTENTION_KV, CachePlaneKind.ATTENTION_RING}
    )

    def __init__(
        self,
        gate: AccuracyGate,
        *,
        enabled: bool = False,
        metrics: CachePolicyMetrics | None = None,
    ) -> None:
        self.gate = gate
        self.enabled = bool(enabled)
        self.metrics = metrics or CachePolicyMetrics()

    def admit(
        self,
        *,
        kind: CachePlaneKind,
        codec: PlaneCodec,
        evidence: AccuracyEvidence,
    ) -> CompressionAdmission:
        reason = self._refusal_reason(kind, codec, evidence)
        accepted = reason is None
        self.metrics.add(
            "compression_admissions" if accepted else "compression_rejections"
        )
        return CompressionAdmission(
            accepted,
            "accepted" if accepted else reason or "refused",
            codec.codec_id,
            evidence.model_fingerprint,
            evidence.plane_fingerprint_digest,
            self.gate.digest,
            evidence.digest,
        )

    def _refusal_reason(
        self,
        kind: CachePlaneKind,
        codec: PlaneCodec,
        evidence: AccuracyEvidence,
    ) -> str | None:
        if not self.enabled:
            return "disabled"
        if kind not in self._COLD_KV_PLANES:
            return "not_cold_kv"
        if codec.lossless or not codec.experimental_accuracy_gated:
            return "not_experimental_lossy_codec"
        if evidence.codec_id != codec.codec_id:
            return "codec_mismatch"
        if evidence.model_fingerprint != self.gate.model_fingerprint:
            return "model_mismatch"
        if evidence.samples < self.gate.min_samples:
            return "insufficient_samples"
        if evidence.max_abs_logit_error > self.gate.max_abs_logit_error:
            return "logit_error"
        if self.gate.require_exact_tokens and not evidence.exact_tokens:
            return "token_mismatch"
        return None


@dataclass(frozen=True)
class PersistedPlanePayload:
    kind: CachePlaneKind
    fingerprint_digest: str
    model_fingerprint: str
    generation: int
    codec_id: str
    payload: bytes
    lossless: bool


class ContentAddressedPlaneStore:
    """Persist only stable immutable plane bytes under a content digest."""

    _DIGEST = re.compile(r"^[0-9a-f]{64}$")

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool = False,
        codecs: Mapping[str, PlaneCodec] | None = None,
        metrics: CachePolicyMetrics | None = None,
    ) -> None:
        supplied = codecs or {
            IdentityCodec.codec_id: IdentityCodec(),
            ZlibCodec.codec_id: ZlibCodec(),
        }
        self.codecs = MappingProxyType(dict(supplied))
        for codec_id, codec in self.codecs.items():
            if codec_id != codec.codec_id or not codec_id:
                raise ValueError("codec registry key does not match codec id")
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.metrics = metrics or CachePolicyMetrics()

    def put(
        self,
        record: StablePlaneRecord,
        *,
        codec_id: str = IdentityCodec.codec_id,
        compression_admission: CompressionAdmission | None = None,
    ) -> str:
        if not self.enabled:
            raise self._refused("disabled")
        if not record.stable:
            raise self._refused("unstable_plane")
        if record.contains_mutable_request_data:
            raise self._refused("mutable_request_data")
        codec = self.codecs.get(codec_id)
        if codec is None:
            raise self._refused("unknown_codec")
        if not codec.lossless:
            self._validate_lossy(record, codec, compression_admission)
        encoded = codec.encode(record.payload)
        if not isinstance(encoded, bytes):
            raise PersistenceRefused("codec output must be bytes")
        if codec.lossless:
            try:
                round_trip = codec.decode(encoded)
            except Exception as error:
                raise self._refused("lossless_codec_decode_failed") from error
            if round_trip != record.payload:
                raise self._refused("lossless_codec_changed_payload")
        source_digest = hashlib.sha256(record.payload).hexdigest()
        encoded_digest = hashlib.sha256(encoded).hexdigest()
        header = {
            "codec": codec.codec_id,
            "encoded_digest": encoded_digest,
            "fingerprint": record.fingerprint.digest,
            "generation": record.generation,
            "kind": record.kind.value,
            "lossless": bool(codec.lossless),
            "model_fingerprint": record.model_fingerprint,
            "source_digest": source_digest,
            "version": 1,
        }
        if not codec.lossless:
            assert compression_admission is not None
            header["compression_admission"] = {
                "codec": compression_admission.codec_id,
                "evidence": compression_admission.evidence_digest,
                "gate": compression_admission.gate_digest,
                "model": compression_admission.model_fingerprint,
                "plane": compression_admission.plane_fingerprint_digest,
            }
        header_bytes = json.dumps(
            header, sort_keys=True, separators=(",", ":")
        ).encode()
        blob = header_bytes + b"\n" + encoded
        content_id = hashlib.sha256(blob).hexdigest()
        destination = self._path(content_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != blob:
                raise IntegrityError("content-address collision or corruption")
            self.metrics.add("persistence_deduplications")
            return content_id
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".plane-", delete=False
            ) as handle:
                temporary_name = handle.name
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        self.metrics.add("persistence_writes")
        return content_id

    def load(
        self,
        content_id: str,
        *,
        expected_fingerprint: CachePlaneFingerprint,
        compression_admission: CompressionAdmission | None = None,
    ) -> PersistedPlanePayload:
        if not self.enabled:
            raise self._refused("disabled")
        if not self._DIGEST.fullmatch(content_id):
            raise IntegrityError("invalid content id")
        blob = self._path(content_id).read_bytes()
        if hashlib.sha256(blob).hexdigest() != content_id:
            raise IntegrityError("content digest mismatch")
        try:
            header_bytes, encoded = blob.split(b"\n", 1)
            header = json.loads(header_bytes)
        except (ValueError, json.JSONDecodeError) as error:
            raise IntegrityError("invalid plane envelope") from error
        if header.get("fingerprint") != expected_fingerprint.digest:
            raise IntegrityError("plane fingerprint mismatch")
        if header.get("kind") != expected_fingerprint.kind.value:
            raise IntegrityError("plane kind mismatch")
        if header.get("version") != 1:
            raise IntegrityError("plane envelope version mismatch")
        generation = header.get("generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise IntegrityError("invalid persisted generation")
        if hashlib.sha256(encoded).hexdigest() != header.get("encoded_digest"):
            raise IntegrityError("encoded payload digest mismatch")
        codec = self.codecs.get(header.get("codec"))
        if codec is None:
            raise IntegrityError("persisted codec is unavailable")
        if bool(header.get("lossless")) != bool(codec.lossless):
            raise IntegrityError("persisted codec policy changed")
        try:
            payload = codec.decode(encoded)
        except Exception as error:
            raise IntegrityError("persisted codec decode failed") from error
        if not isinstance(payload, bytes):
            raise IntegrityError("decoded payload is not bytes")
        lossless = bool(header.get("lossless"))
        if not lossless:
            self._validate_loaded_lossy(
                header, codec, expected_fingerprint, compression_admission
            )
        if lossless and hashlib.sha256(payload).hexdigest() != header.get(
            "source_digest"
        ):
            raise IntegrityError("decoded payload digest mismatch")
        self.metrics.add("persistence_reads")
        return PersistedPlanePayload(
            expected_fingerprint.kind,
            expected_fingerprint.digest,
            str(header.get("model_fingerprint")),
            generation,
            codec.codec_id,
            payload,
            lossless,
        )

    def _validate_lossy(
        self,
        record: StablePlaneRecord,
        codec: PlaneCodec,
        admission: CompressionAdmission | None,
    ) -> None:
        if record.kind not in ColdKVCompressionPolicy._COLD_KV_PLANES:
            raise self._refused("lossy_codec_not_cold_kv")
        if not codec.experimental_accuracy_gated:
            raise self._refused("lossy_codec_not_accuracy_gated")
        if admission is None or not admission.accepted:
            raise self._refused("compression_not_admitted")
        if admission.codec_id != codec.codec_id:
            raise self._refused("compression_codec_mismatch")
        if admission.model_fingerprint != record.model_fingerprint:
            raise self._refused("compression_model_mismatch")
        if admission.plane_fingerprint_digest != record.fingerprint.digest:
            raise self._refused("compression_plane_mismatch")

    def _validate_loaded_lossy(
        self,
        header: dict,
        codec: PlaneCodec,
        expected_fingerprint: CachePlaneFingerprint,
        admission: CompressionAdmission | None,
    ) -> None:
        if admission is None or not admission.accepted:
            raise self._refused("compression_revalidation_required")
        persisted = header.get("compression_admission")
        expected = {
            "codec": admission.codec_id,
            "evidence": admission.evidence_digest,
            "gate": admission.gate_digest,
            "model": admission.model_fingerprint,
            "plane": admission.plane_fingerprint_digest,
        }
        if persisted != expected:
            raise self._refused("compression_admission_changed")
        if admission.codec_id != codec.codec_id:
            raise self._refused("compression_codec_mismatch")
        if admission.plane_fingerprint_digest != expected_fingerprint.digest:
            raise self._refused("compression_plane_mismatch")

    def _path(self, content_id: str) -> Path:
        return self.root / content_id[:2] / f"{content_id}.plane"

    def _refused(self, reason: str) -> PersistenceRefused:
        self.metrics.add("persistence_rejections")
        return PersistenceRefused(reason)


__all__ = [
    "AccuracyEvidence",
    "AccuracyGate",
    "CachePlanePlacementDecision",
    "CachePlanePlacementPolicy",
    "CachePlanePlacementRequest",
    "CachePolicyMetrics",
    "ColdKVCompressionPolicy",
    "CompressionAdmission",
    "ContentAddressedPlaneStore",
    "IdentityCodec",
    "IntegrityError",
    "PersistedPlanePayload",
    "PersistenceRefused",
    "PlaneCodec",
    "StablePlaneRecord",
    "ZlibCodec",
    "cache_plane_operation",
    "layered_cache_policy_enabled",
]
