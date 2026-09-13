"""Guarded ANE target-head candidate for greedy Qwen4 scheduling.

The stateful model trunk stays on MLX/Metal.  This module owns a stateless,
partitioned full-vocabulary projection in a separate Core ML process.  A
generation stamp prevents a late result from authorizing a different batch
state.  The scheduler must still use the GPU fallback when eligibility,
deadline, source, or confidence gates fail.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

Mode = Literal["off", "shadow", "active"]


@dataclass(frozen=True)
class ANEVerifierConfig:
    mode: Mode = "off"
    deadline_ms: float = 15.0
    max_inflight: int = 2
    min_margin: float = 0.125
    min_gpu_marginal_delay_ms: float = 0.25
    qualified_service_p95_ms: float = 11.0
    host_overhead_ms: float = 0.10
    min_net_savings_ms: float = 0.20
    require_ane: bool = True
    require_source_match: bool = True
    allow_approximate_commit: bool = False

    def validated(self) -> "ANEVerifierConfig":
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError(f"invalid ANE verifier mode: {self.mode!r}")
        if self.mode == "active" and not self.allow_approximate_commit:
            raise ValueError(
                "active ANE verification requires allow_approximate_commit=True; "
                "the FP16 ANE head is confidence-gated, not bit-exact to q4 Metal"
            )
        for name, value in (
            ("deadline_ms", self.deadline_ms),
            ("min_margin", self.min_margin),
            ("min_gpu_marginal_delay_ms", self.min_gpu_marginal_delay_ms),
            ("qualified_service_p95_ms", self.qualified_service_p95_ms),
            ("host_overhead_ms", self.host_overhead_ms),
            ("min_net_savings_ms", self.min_net_savings_ms),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        return self


@dataclass(frozen=True)
class ANEVerifierStamp:
    membership_epoch: int
    lane_uids: tuple[int, ...]
    generations: tuple[int, ...]
    verify_positions: tuple[int, ...]


@dataclass(frozen=True)
class ANEVerifierEligibility:
    eligible: bool
    reason: str
    predicted_net_savings_ms: float = 0.0


def ane_verifier_eligibility(
    config: ANEVerifierConfig,
    *,
    greedy: bool,
    has_logits_processors: bool,
    needs_full_logprobs: bool,
    package_resident: bool,
    memory_headroom_gib: float,
    package_gib: float,
    predicted_gpu_marginal_delay_ms: float,
    available_overlap_ms: float,
    predicted_ane_interference_ms: float = 0.0,
    expected_remaining_verifications: int = 1,
    predicted_final_drain_ms: Optional[float] = None,
) -> ANEVerifierEligibility:
    """Gate on measured scheduler state rather than batch width.

    ``predicted_gpu_marginal_delay_ms`` is the incremental critical-path cost
    of keeping the head on Metal.  It may be learned from online service-time
    samples and is intentionally not inferred from row count here.
    """
    config.validated()
    if config.mode == "off":
        return ANEVerifierEligibility(False, "disabled")
    if not greedy:
        return ANEVerifierEligibility(False, "non_greedy")
    if has_logits_processors:
        return ANEVerifierEligibility(False, "logits_processors")
    if needs_full_logprobs:
        return ANEVerifierEligibility(False, "full_logprobs_requested")
    if not package_resident:
        return ANEVerifierEligibility(False, "package_not_resident")
    if memory_headroom_gib < package_gib:
        return ANEVerifierEligibility(False, "insufficient_memory_headroom")
    if predicted_gpu_marginal_delay_ms < config.min_gpu_marginal_delay_ms:
        return ANEVerifierEligibility(False, "gpu_marginal_cost_is_low")
    if expected_remaining_verifications < 1:
        return ANEVerifierEligibility(False, "no_remaining_verification")
    residual_wait = max(0.0, config.qualified_service_p95_ms - available_overlap_ms)
    final_drain = (
        config.qualified_service_p95_ms
        if predicted_final_drain_ms is None
        else max(0.0, predicted_final_drain_ms)
    )
    amortized_drain = final_drain / expected_remaining_verifications
    predicted_net = (
        predicted_gpu_marginal_delay_ms
        - max(0.0, predicted_ane_interference_ms)
        - residual_wait
        - amortized_drain
        - config.host_overhead_ms
    )
    if predicted_net < config.min_net_savings_ms:
        return ANEVerifierEligibility(
            False, "predicted_net_savings_too_low", predicted_net
        )
    return ANEVerifierEligibility(True, "eligible", predicted_net)


@dataclass
class ANEVerifierTicket:
    ticket_id: str
    stamp: ANEVerifierStamp
    submitted_ns: int


@dataclass(frozen=True)
class ANEVerifierResult:
    token_ids: tuple[int, ...]
    scores: tuple[float, ...]
    margin: float
    receipt: dict[str, Any]


@dataclass
class ANEVerifierStats:
    counts: Counter = field(default_factory=Counter)
    last_receipt: Optional[dict[str, Any]] = None

    def snapshot(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "last_receipt": self.last_receipt}


def _output_arrays(outputs):
    import numpy as np

    integers = [
        np.asarray(value, dtype=np.int64)
        for value in outputs.values()
        if np.issubdtype(np.asarray(value).dtype, np.integer)
    ]
    floats = [
        np.asarray(value, dtype=np.float32)
        for value in outputs.values()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    ]
    if len(integers) != 1 or len(floats) != 1:
        raise RuntimeError(f"unexpected ANE verifier outputs: {list(outputs)}")
    return integers[0], floats[0]


def _part_candidates(outputs, manifest, rows):
    import numpy as np

    indices, scores = _output_arrays(outputs)
    chunks = int(manifest["row_count"]) // int(manifest["chunk_rows"])
    top_k = int(manifest["top_k"])
    local = indices.reshape(chunks, top_k)
    values = scores.reshape(chunks, top_k)
    offsets = np.arange(chunks, dtype=np.int64)[:, None] * int(manifest["chunk_rows"])
    local = (local + offsets).reshape(-1)
    values = values.reshape(-1)
    valid = local < int(manifest.get("valid_row_count", len(rows)))
    return [
        (float(score), int(rows[index]))
        for index, score in zip(local[valid], values[valid])
    ]


def _worker_main(commands, results, package_paths):
    import coremltools as ct
    import mlx.core as mx
    import numpy as np

    parts = []
    try:
        for package_text in package_paths:
            package = Path(package_text)
            manifest = json.loads(
                package.with_suffix(package.suffix + ".json").read_text()
            )
            model = ct.models.MLModel(
                str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            rows_path = Path(manifest["source"]["rows_sidecar"])
            rows = np.asarray(mx.load(str(rows_path))["rows"], dtype=np.uint32)
            parts.append((manifest, model, rows))
        results.send(
            {
                "ready": True,
                "hidden_size": int(parts[0][0]["hidden_size"]),
                "package_bytes": sum(
                    int(part[0]["package_disk_bytes"]) for part in parts
                ),
            }
        )
        while True:
            command = commands.recv()
            if command[0] == "stop":
                return
            _, ticket_id, hidden = command
            started_ns = time.perf_counter_ns()
            try:
                candidates = []
                part_ms = []
                for manifest, model, rows in parts:
                    input_name = model.get_spec().description.input[0].name
                    part_started = time.perf_counter_ns()
                    outputs = model.predict({input_name: hidden})
                    part_ms.append((time.perf_counter_ns() - part_started) / 1e6)
                    candidates.extend(_part_candidates(outputs, manifest, rows))
                candidates.sort(reverse=True)
                selected = candidates[: int(parts[0][0]["top_k"])]
                payload = {
                    "ticket_id": ticket_id,
                    "ok": True,
                    "token_ids": [token for _score, token in selected],
                    "scores": [score for score, _token in selected],
                    "part_ms": part_ms,
                    "started_ns": started_ns,
                    "finished_ns": time.perf_counter_ns(),
                }
            except Exception as error:  # return failure to the owner process
                payload = {
                    "ticket_id": ticket_id,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "started_ns": started_ns,
                    "finished_ns": time.perf_counter_ns(),
                }
            results.send(payload)
    except Exception as error:
        results.send({"ready": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        commands.close()
        results.close()


class ANEVerifierProcess:
    """Persistent process owner so Core ML orchestration can overlap MLX."""

    def __init__(self, package_paths: Sequence[Path]):
        context = multiprocessing.get_context("spawn")
        child_commands, self._commands = context.Pipe(duplex=False)
        self._results, child_results = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_worker_main,
            args=(child_commands, child_results, tuple(map(str, package_paths))),
            name="mlx-lm-ane-verifier",
        )
        self._process.start()
        child_commands.close()
        child_results.close()
        ready = self._results.recv()
        if not ready.get("ready"):
            raise RuntimeError(f"ANE verifier failed to initialize: {ready}")
        self.hidden_size = int(ready["hidden_size"])
        self.package_bytes = int(ready["package_bytes"])
        self._buffer: dict[str, dict[str, Any]] = {}
        self._abandoned: set[str] = set()

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def submit(self, ticket_id: str, hidden) -> None:
        if not self.is_alive:
            raise RuntimeError("ANE verifier worker is not alive")
        self._commands.send(("verify", ticket_id, hidden))

    def receive(self, ticket_id: str, timeout_s: Optional[float]):
        if ticket_id in self._buffer:
            return self._buffer.pop(ticket_id)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if not self._results.poll(remaining):
                return None
            result = self._results.recv()
            result_id = result["ticket_id"]
            if result_id in self._abandoned:
                self._abandoned.remove(result_id)
            elif result_id == ticket_id:
                return result
            else:
                self._buffer[result_id] = result

    def abandon(self, ticket_id: str) -> None:
        self._buffer.pop(ticket_id, None)
        self._abandoned.add(ticket_id)

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._commands.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._commands.close()
        self._results.close()


class ANEVerifierController:
    def __init__(self, config: ANEVerifierConfig, runner):
        self.config = config.validated()
        self.runner = runner
        self.stats = ANEVerifierStats()
        self._tickets: dict[str, ANEVerifierTicket] = {}

    @property
    def package_gib(self) -> float:
        return self.runner.package_bytes / (1024**3)

    def submit(self, stamp: ANEVerifierStamp, hidden) -> Optional[ANEVerifierTicket]:
        import mlx.core as mx
        import numpy as np

        if self.config.mode == "off":
            self.stats.counts["declined_disabled"] += 1
            return None
        if len(self._tickets) >= self.config.max_inflight:
            self.stats.counts["declined_queue_full"] += 1
            return None
        if tuple(hidden.shape) != (1, 1, self.runner.hidden_size):
            raise ValueError(
                "ANE verifier hidden must have shape "
                f"(1, 1, {self.runner.hidden_size}), got {tuple(hidden.shape)}"
            )
        device_hidden = mx.array(hidden).astype(mx.float16)
        mx.eval(device_hidden)
        host_hidden = np.ascontiguousarray(np.asarray(device_hidden))
        ticket = ANEVerifierTicket(uuid.uuid4().hex, stamp, time.perf_counter_ns())
        self.runner.submit(ticket.ticket_id, host_hidden)
        self._tickets[ticket.ticket_id] = ticket
        self.stats.counts["submitted"] += 1
        return ticket

    def resolve(
        self, ticket: Optional[ANEVerifierTicket], current_stamp: ANEVerifierStamp
    ) -> Optional[ANEVerifierResult]:
        if ticket is None:
            self.stats.counts["fallback_no_ticket"] += 1
            return None
        live = self._tickets.pop(ticket.ticket_id, None)
        if live is not ticket or ticket.stamp != current_stamp:
            self.runner.abandon(ticket.ticket_id)
            self.stats.counts["fallback_stale"] += 1
            return None
        timeout_s = (
            None if self.config.deadline_ms == 0 else self.config.deadline_ms / 1000
        )
        payload = self.runner.receive(ticket.ticket_id, timeout_s)
        if payload is None:
            self.runner.abandon(ticket.ticket_id)
            self.stats.counts["fallback_timeout"] += 1
            return None
        if not payload.get("ok"):
            self.stats.counts["fallback_worker_error"] += 1
            return None
        scores = tuple(float(value) for value in payload["scores"])
        margin = scores[0] - scores[1]
        receipt = {
            "ticket_id": ticket.ticket_id,
            "queue_ms": (payload["started_ns"] - ticket.submitted_ns) / 1e6,
            "runtime_ms": (payload["finished_ns"] - payload["started_ns"]) / 1e6,
            "total_ms": (payload["finished_ns"] - ticket.submitted_ns) / 1e6,
            "part_ms": payload["part_ms"],
            "margin": margin,
        }
        self.stats.last_receipt = receipt
        if margin < self.config.min_margin:
            self.stats.counts["fallback_low_margin"] += 1
            return None
        if self.config.mode == "shadow":
            self.stats.counts["shadow_resolved"] += 1
            return None
        self.stats.counts["resolved"] += 1
        return ANEVerifierResult(
            tuple(int(value) for value in payload["token_ids"]), scores, margin, receipt
        )

    def abandon(
        self, ticket: Optional[ANEVerifierTicket], reason: str = "cancelled"
    ) -> None:
        if ticket is None:
            return
        if self._tickets.pop(ticket.ticket_id, None) is not None:
            self.runner.abandon(ticket.ticket_id)
            self.stats.counts[f"abandoned_{reason}"] += 1

    def close(self) -> None:
        for ticket in tuple(self._tickets.values()):
            self.abandon(ticket, "controller_close")
        self.runner.close()


def load_ane_verifier(
    package_dir: Path,
    config: Optional[ANEVerifierConfig] = None,
    package_glob: str = "part*.mlpackage",
    runner=None,
    model_dir: Optional[Path] = None,
) -> ANEVerifierController:
    package_dir = Path(package_dir)
    package_paths = sorted(package_dir.glob(package_glob))
    if not package_paths:
        raise ValueError(f"no ANE verifier packages matched {package_glob!r}")
    manifests = [
        json.loads(path.with_suffix(path.suffix + ".json").read_text())
        for path in package_paths
    ]
    config = (config or ANEVerifierConfig()).validated()
    if any(row.get("stage") != "full_vocab_greedy_verify" for row in manifests):
        raise ValueError("ANE verifier package has the wrong stage")
    hidden_sizes = {int(row["hidden_size"]) for row in manifests}
    if len(hidden_sizes) != 1:
        raise ValueError("ANE verifier partitions disagree on hidden size")
    valid_rows = sum(
        int(row.get("valid_row_count", row["row_count"])) for row in manifests
    )
    row_sets = []
    for manifest in manifests:
        preferred = manifest.get("compute_plan", {}).get("preferred_counts", {})
        if config.require_ane and not preferred.get("MLNeuralEngineComputeDevice"):
            raise ValueError("ANE verifier partition is not ANE-resident")
        row_sets.append(manifest["source"]["rows_sidecar"])
    import mlx.core as mx
    import numpy as np

    rows = [np.asarray(mx.load(path)["rows"], dtype=np.int64) for path in row_sets]
    merged = np.concatenate(rows)
    if valid_rows != len(merged) or set(map(int, merged)) != set(range(valid_rows)):
        raise ValueError(
            "ANE verifier partitions do not cover the vocabulary exactly once"
        )
    if config.require_source_match:
        if model_dir is None:
            raise ValueError(
                "model_dir is required to verify ANE verifier source weights"
            )
        model_dir = Path(model_dir).resolve()
        model_config = json.loads((model_dir / "config.json").read_text())
        quantization = model_config["quantization"]
        weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        prefix = "language_model.lm_head."
        tensors = mx.load(str(model_dir / weight_map[prefix + "weight"]))
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            for manifest, selected_rows in zip(manifests, rows):
                ids = mx.array(selected_rows, mx.uint32)
                selected = [
                    mx.take(tensors[prefix + suffix], ids, axis=0)
                    for suffix in ("weight", "scales", "biases")
                ]
                weight = mx.dequantize(
                    *selected,
                    group_size=int(quantization["group_size"]),
                    bits=int(quantization["bits"]),
                    mode=quantization.get("mode", "affine"),
                ).astype(mx.float16)
                mx.eval(weight)
                digest = hashlib.sha256()
                digest.update(
                    np.ascontiguousarray(selected_rows, dtype=np.uint32).tobytes()
                )
                digest.update(np.ascontiguousarray(np.asarray(weight)).tobytes())
                actual = digest.hexdigest()
                expected = manifest.get("source_fingerprint")
                if actual != expected:
                    raise ValueError(
                        "ANE verifier partition does not match the live model: "
                        f"expected {expected}, got {actual}"
                    )
        finally:
            mx.set_default_device(previous_device)
    if runner is None:
        runner = ANEVerifierProcess(package_paths)
    controller = ANEVerifierController(config, runner)
    controller.verified_source_model = None if model_dir is None else str(model_dir)
    return controller
