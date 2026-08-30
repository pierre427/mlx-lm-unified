#!/usr/bin/env python3
"""Opt-in, append-only real-checkpoint Qwen4 transactional-state attestation.

This is deliberately not a benchmark.  It loads exactly one pinned Flash-Next
checkpoint, constructs immutable B=1 and (when safe) B=2 cache bases, and runs
two independent eager arms for every k=2 accepted-prefix geometry.  The arms
must be raw-bit identical across the full target/draft/batch oracle and the
one-token continuation.  Evidence is appended and fsynced one JSON object per
event so an interruption cannot rewrite earlier results.

The script imports MLX only after the operator opt-in, environment identity,
and checkpoint identity gates pass.  ``--help`` is therefore model/GPU-free.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch


SCHEMA = "qwen4-transactional-state-attestation-v1"
REPO = Path(__file__).resolve().parents[1]
MODEL = Path(
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
PLE_SIDECAR = MODEL / "ple_rows.bin"
EXPECTED_CONFIG_SHA256 = (
    "2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b"
)
EXPECTED_INDEX_SHA256 = (
    "f643b2bd4f768a6a68fcdec89870f080f71420e8adbd084801291835df0cca5c"
)
EXPECTED_PLE_BYTES = 32_000_153_600
MIN_B2_AVAILABLE_BYTES = 20 * 1024**3
DEFAULT_RAPID_ROOT = Path(
    "/Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX-worktrees/"
    "qwen4-runtime-selector-integration-20260830"
)
K = 2
PROMPTS = (
    "Briefly explain why a checksum detects an accidental file change.",
    "Give one concise example of a transaction that must be atomic.",
)
REQUIRED_ENV = {
    "MLX_ENABLE_TF32": "0",
    "MLX_QWEN4_PLE_NVME": str(PLE_SIDECAR),
    "MLX_QWEN4_PLE_NVME_LRU_MB": "256",
    "MLX_LM_UBC_EVICT": "1",
    "MLX_QWEN4_QSA_POOLED_KEY_CACHE": "1",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attest the pinned Flash-Next checkpoint's eager transactional state "
            "with a raw-bit B=1/B=2 k=2 oracle. This loads a ~72 GiB model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSONL file opened append-only; parent directory must already exist.",
    )
    parser.add_argument(
        "--rapid-root",
        type=Path,
        default=DEFAULT_RAPID_ROOT,
        help="Rapid-MLX checkout providing the receipt contract.",
    )
    parser.add_argument(
        "--i-understand-this-loads-one-72gib-model",
        action="store_true",
        help="Required explicit opt-in; without it the script exits before MLX import.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()


def _available_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        match = re.search(r"page size of (\d+) bytes", result.stdout)
        if match is None:
            return None
        page_size = int(match.group(1))
        counts: dict[str, int] = {}
        for line in result.stdout.splitlines()[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                counts[name] = int(value.strip().rstrip("."))
        pages = sum(
            counts.get(name, 0)
            for name in ("Pages free", "Pages inactive", "Pages speculative")
        )
        return pages * page_size if pages > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _thermal_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"qualified": False, "limits": {}, "raw": ""}
    try:
        result = subprocess.run(
            ["/usr/bin/pmset", "-g", "therm"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
        return snapshot
    raw = result.stdout.strip()
    snapshot["raw"] = raw
    limits = {
        name: int(value)
        for name, value in re.findall(r"(\w+(?:_\w+)*)\s*=\s*(\d+)", raw)
    }
    snapshot["limits"] = limits
    speed_limits = [
        value for name, value in limits.items() if name.endswith("Speed_Limit")
    ]
    thermal_levels = [
        value for name, value in limits.items() if "Thermal" in name
    ]
    healthy_notes = all(
        note in raw
        for note in (
            "No thermal warning level has been recorded",
            "No performance warning level has been recorded",
            "No CPU power status has been recorded",
        )
    )
    numeric_healthy = bool(speed_limits) and all(
        value >= 100 for value in speed_limits
    ) and all(value == 0 for value in thermal_levels)
    snapshot["qualified"] = healthy_notes or numeric_healthy
    return snapshot


class EvidenceWriter:
    def __init__(self, path: Path, common: Mapping[str, Any]):
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output parent does not exist: {path.parent}")
        self.path = path
        self.common = dict(common)

    def append(self, event: str, **values: Any) -> None:
        record = {
            **self.common,
            "event": event,
            "recorded_at_unix": time.time(),
            **values,
        }
        encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(self.path, flags, 0o644)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _validate_static_identity(args: argparse.Namespace) -> dict[str, Any]:
    if not args.i_understand_this_loads_one_72gib_model:
        raise RuntimeError(
            "refusing model load without --i-understand-this-loads-one-72gib-model"
        )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("this attestation requires Apple-silicon Darwin")
    if Path.cwd().resolve() != REPO:
        raise RuntimeError(f"run from the pinned checkout root: {REPO}")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing a dirty source checkout")
    for path in (MODEL / "config.json", MODEL / "model.safetensors.index.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not PLE_SIDECAR.is_file() or PLE_SIDECAR.stat().st_size != EXPECTED_PLE_BYTES:
        raise RuntimeError("production PLE-NVMe sidecar is missing or has wrong size")
    observed_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    mismatched = {
        name: {"expected": expected, "actual": observed_env[name]}
        for name, expected in REQUIRED_ENV.items()
        if observed_env[name] != expected
    }
    if mismatched:
        raise RuntimeError(f"production environment mismatch: {mismatched}")
    config_hash = _sha256(MODEL / "config.json")
    index_hash = _sha256(MODEL / "model.safetensors.index.json")
    if config_hash != EXPECTED_CONFIG_SHA256 or index_hash != EXPECTED_INDEX_SHA256:
        raise RuntimeError("checkpoint config/index identity changed")
    rapid_root = args.rapid_root.resolve()
    receipt_module = (
        rapid_root
        / "vllm_mlx"
        / "spec_decode"
        / "mtp"
        / "prompt_lookup_attestation.py"
    )
    if not receipt_module.is_file():
        raise FileNotFoundError(f"Rapid receipt contract is missing: {receipt_module}")
    rapid_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=rapid_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    return {
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "model_path": str(MODEL),
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "ple_sidecar": str(PLE_SIDECAR),
        "ple_sidecar_bytes": PLE_SIDECAR.stat().st_size,
        "environment": observed_env,
        "rapid_root": str(rapid_root),
        "rapid_commit": rapid_commit,
        "rapid_receipt_module_sha256": _sha256(receipt_module),
    }


def _load_oracle_module():
    path = REPO / "tests" / "test_verify_state_oracle.py"
    spec = importlib.util.spec_from_file_location("qwen4_state_oracle_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load oracle support from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_receipt_contract(rapid_root: Path):
    root = rapid_root.resolve()
    sys.path.insert(0, str(root))
    try:
        from vllm_mlx.spec_decode.mtp import prompt_lookup_attestation as contract
    finally:
        sys.path.remove(str(root))
    loaded = Path(contract.__file__).resolve()
    if not loaded.is_relative_to(root):
        raise RuntimeError(
            f"receipt contract resolved outside --rapid-root: {loaded}"
        )
    return contract


def _capture_digest(capture: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path, atom in sorted(capture.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(atom.kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(atom.dtype.encode("utf-8"))
        digest.update(json.dumps(atom.shape).encode("ascii"))
        payload = atom.payload
        if isinstance(payload, bytes):
            digest.update(payload)
        else:
            digest.update(
                json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
            )
        digest.update(b"\n")
    return digest.hexdigest()


def _atom_payload(atom: Any) -> dict[str, Any]:
    payload = atom.payload
    if isinstance(payload, bytes):
        payload = {"base64": base64.b64encode(payload).decode("ascii")}
    return {
        "kind": atom.kind,
        "dtype": atom.dtype,
        "shape": list(atom.shape),
        "payload": payload,
    }


def _bundle_raw_bits(contract, capture: Mapping[str, Any], paths: Sequence[str]):
    if not paths:
        raise AssertionError("receipt surface selected no oracle atoms")
    encoded = json.dumps(
        [[path, _atom_payload(capture[path])] for path in sorted(paths)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return contract.RawBitValue.from_buffer(
        dtype="oracle-atom-bundle-v1", shape=(len(paths),), raw_bits=encoded
    )


def _phase_prefix(contract, phase: Any) -> str:
    if phase is contract.OraclePhase.POST_TRIM:
        return "post_commit.batch."
    return "post_continuation.batch."


def _surface_paths(contract, capture: Mapping[str, Any], surface: str, phase: Any):
    prefix = _phase_prefix(contract, phase)
    paths = tuple(capture)

    def selected(predicate):
        return [path for path in paths if predicate(path)]

    if surface == "verify_logits":
        exact = "verify.logits" if phase is contract.OraclePhase.POST_TRIM else "continuation.logits"
        return selected(lambda path: path == exact)
    if surface == "target_cache_kv":
        return selected(
            lambda path: path.startswith(prefix + "target.")
            and any(token in path for token in (".keys", ".values", ".cache["))
        )
    if surface == "mtp_cache_kv":
        return selected(
            lambda path: path.startswith(prefix + "draft.")
            and any(token in path for token in (".keys", ".values", ".cache["))
        )
    if surface == "seed_hidden":
        return selected(lambda path: path.startswith(prefix + "lane[") and path.endswith(".seed_h"))
    if surface == "gdn_conv_tail":
        return selected(lambda path: path.startswith(prefix) and ".cache[0]" in path)
    if surface == "gdn_matrix":
        return selected(lambda path: path.startswith(prefix) and ".cache[1]" in path)
    if surface == "gdn_host_metadata":
        return selected(
            lambda path: path.startswith(prefix)
            and ".layer[" in path
            and any(
                token in path
                for token in (
                    ".type", ".max_size", ".offset", "._right_padding",
                    "._valid_lengths", "._rollback_window", "._rollback_invalid_reason",
                )
            )
        )
    if surface == "gdn_rollback_stack":
        return selected(lambda path: path.startswith(prefix) and "._rollbacks" in path)
    if surface == "ple_conv_state":
        return selected(lambda path: path.startswith(prefix) and ".cache[2]" in path)
    if surface == "ple_token_history":
        return selected(lambda path: path.startswith(prefix) and ".cache[3]" in path)
    if surface == "ple_atomic_rollback":
        return selected(lambda path: path.startswith(prefix) and "._ple_rollback" in path)
    if surface == "qsa_keys":
        return selected(lambda path: path.startswith(prefix) and path.endswith(".keys"))
    if surface == "qsa_values":
        return selected(lambda path: path.startswith(prefix) and path.endswith(".values"))
    if surface == "qsa_offset":
        return selected(lambda path: path.startswith(prefix) and path.endswith(".offset"))
    if surface == "qsa_index_keys":
        return selected(lambda path: path.startswith(prefix) and path.endswith(".index_keys"))
    if surface == "qsa_pooled_keys":
        return selected(lambda path: path.startswith(prefix) and path.endswith("._qsa_pooled_keys"))
    if surface == "qsa_pooled_ratio":
        return selected(lambda path: path.startswith(prefix) and path.endswith("._qsa_pooled_ratio"))
    if surface == "qsa_share_topk_flag":
        return selected(lambda path: path.startswith(prefix + "lane[") and path.endswith(".share_qsa_indices"))
    if surface == "qsa_shared_topk":
        return selected(lambda path: path.startswith(prefix) and path.endswith("._mtp_shared_topk"))
    if surface == "batch_membership_epoch":
        return selected(lambda path: path == prefix + "membership_epoch")
    if surface == "batch_proposal_state":
        return selected(
            lambda path: path in {
                "post_proposal.batch.proposal_open",
                "post_proposal.batch._open_proposal.present",
                prefix + "proposal_open",
                prefix + "_open_proposal.present",
            }
        )
    if surface == "proposal_transaction_metadata":
        return selected(
            lambda path: path.startswith("post_proposal.batch._open_proposal.")
            and not any(token in path for token in (".outputs", "._old_seed_hs", "._vhidden", "._logprobs"))
        )
    if surface == "proposal_outputs":
        return selected(lambda path: path.startswith("post_proposal.batch._open_proposal.outputs"))
    if surface == "lane_identity_metadata":
        return selected(
            lambda path: path == prefix + "lane_count"
            or path.startswith(prefix + "lane_uids")
            or (path.startswith(prefix + "lane[") and path.endswith((".uid", ".max_tokens", ".num_draft")))
        )
    if surface == "lane_decode_state":
        return selected(
            lambda path: path.startswith(prefix + "lane[")
            and path.endswith((".cur", ".seed_h", ".token_prefix", ".ntoks"))
        )
    if surface == "lane_pending_state":
        return selected(
            lambda path: path.startswith(prefix + "lane[")
            and any(token in path for token in (".pending_hs", ".pending_ts"))
        )
    if surface == "lane_statistics":
        return selected(lambda path: path.startswith(prefix + "lane[") and ".stats." in path)
    if surface == "lane_rng_aliasing":
        return selected(
            lambda path: path.startswith(prefix + "lane[")
            and path.endswith((".rng.present", ".rng.type", ".rng.alias"))
        )
    if surface == "lane_rng_state":
        return selected(
            lambda path: path.startswith(prefix + "lane[")
            and path.endswith((".rng.key", ".rng.draws"))
        )
    raise AssertionError(f"unmapped receipt surface: {surface}")


def _receipt_cases(contract, accepted: tuple[int, ...], reference, candidate):
    cases = []
    for phase in contract.OraclePhase:
        surfaces = []
        for name in sorted(contract.REQUIRED_STATE_SURFACES):
            reference_paths = _surface_paths(contract, reference, name, phase)
            candidate_paths = _surface_paths(contract, candidate, name, phase)
            if reference_paths != candidate_paths:
                raise AssertionError(f"receipt surface path mismatch: {name}")
            surfaces.append(
                contract.SurfaceEvidence(
                    surface=name,
                    reference=_bundle_raw_bits(contract, reference, reference_paths),
                    candidate=_bundle_raw_bits(contract, candidate, candidate_paths),
                )
            )
        cases.append(
            contract.OracleCaseEvidence(
                key=contract.OracleCaseKey(accepted, phase),
                surfaces=tuple(surfaces),
            )
        )
    return tuple(cases)


def _prepare_base(oracle, model, token_rows: Sequence[Sequence[int]], uid_base: int):
    detached = []
    for row, tokens in enumerate(token_rows):
        lane, _ = oracle.prepare_self_mtp_lane(
            oracle.mx.array(tokens, oracle.mx.uint32),
            model,
            uid=uid_base + row,
            max_tokens=32,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=oracle.LaneRNG(50_000 + uid_base + row),
            num_draft=K,
            sampling_temp=0.8,
            sampling_top_p=1.0,
            sampling_top_k=8,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=8,
            share_qsa_indices=False,
        )
        detached.append(lane)
    base = oracle.attach_self_mtp_lanes(model, None, detached)
    proof = oracle.capture_batched_state(base, "immutable_base")
    oracle.assert_oracle_equal(
        proof, oracle.capture_batched_state(oracle._clone_batch(base), "immutable_base")
    )
    return base, _capture_digest(proof), len(proof)


def _run_geometry(
    oracle, contract, model, base, accepted: tuple[int, ...]
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    left = oracle._run_scenario(model, base, accepted)
    right = oracle._run_scenario(model, base, accepted)
    oracle.assert_oracle_equal(left, right)
    mtp_calls = left["route.mtp_step_calls"].payload
    backbone_calls = left["route.backbone_calls"].payload
    if not isinstance(mtp_calls, int) or mtp_calls <= 0:
        raise AssertionError("MTP route did not execute")
    if not isinstance(backbone_calls, int) or backbone_calls <= 0:
        raise AssertionError("target backbone route did not execute")
    if "continuation.logits" not in left:
        raise AssertionError("continuation logits are absent from oracle capture")
    reference = _run_eager_target_reference(oracle, model, base, accepted)
    record = {
        "accepted": list(accepted),
        "candidate_capture_atoms": len(left),
        "candidate_capture_sha256": _capture_digest(left),
        **reference,
        "mtp_step_calls": mtp_calls,
        "backbone_calls": backbone_calls,
        "raw_bit_equal": True,
        "continuation_compared": True,
    }
    return record, _receipt_cases(contract, accepted, left, right)


def _live_boundary(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize away rollback-journal representation, not live state.

    A speculative M+1 forward followed by ragged rewind and an ordinary eager
    accepted-prefix forward legitimately retain different rollback *record
    partitions*.  The candidate-only full oracle compares every record and
    replay result above.  The independent reference compares live arrays, PLE
    staged state, QSA ledgers, padding/cursors, and the next continuation while
    excluding only ``_rollbacks`` and checkpoint history representation.
    """

    return {
        path: atom
        for path, atom in capture.items()
        if "._rollbacks" not in path and "._checkpoints" not in path
    }


def _assert_array_equal(oracle, expected, actual, label: str) -> None:
    left = {label: oracle._array_atom(expected)}
    right = {label: oracle._array_atom(actual)}
    oracle.assert_oracle_equal(left, right)


def _run_eager_target_reference(
    oracle, model, base, accepted: tuple[int, ...]
) -> dict[str, Any]:
    """Compare speculative trim/commit with ordinary eager accepted-prefix work."""

    candidate = oracle._clone_batch(base)
    candidate_logits = []
    original_logits = model.logits

    def recording_logits(hidden):
        output = original_logits(hidden)
        oracle.mx.eval(output)
        candidate_logits.append(oracle.mx.array(output))
        return output

    pending = iter(accepted)

    def force_accept(logprobs, *_args, **_kwargs):
        count = next(pending)
        return count, int(oracle.mx.argmax(logprobs[count]).item())

    model.logits = recording_logits
    try:
        with patch(
            "mlx_lm.hybrid_speculative._batched_residual_verify",
            side_effect=force_accept,
        ):
            proposal = oracle.propose_batched_self_mtp(model, candidate)
    finally:
        model.logits = original_logits
    verify_logits = candidate_logits[-1]
    oracle.commit_batched_self_mtp(
        candidate,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False] * len(candidate.lanes),
    )

    # Logits are shape-sensitive on Metal even for a causally independent
    # prefix. Compare the candidate verifier against an independently cloned
    # target forward with the SAME B x (K+1) slab and padding contract.
    verify_reference = oracle._clone_batch(base)
    verify_rows = [
        [proposal._old_curs[row], *proposal._drafts[row]]
        for row in range(len(accepted))
    ]
    verify_lengths = [len(row) for row in verify_rows]
    verify_width = max(verify_lengths)
    verify_right_padding = [verify_width - length for length in verify_lengths]
    verify_padded = [
        row + [0] * (verify_width - len(row)) for row in verify_rows
    ]
    oracle._prepare_self_mtp_cache_group(
        verify_reference.caches.target,
        lengths=verify_lengths,
        right_padding=verify_right_padding,
    )
    try:
        reference_hidden, _ = model.mtp_backbone(
            oracle.mx.array(verify_padded, dtype=oracle.mx.uint32),
            cache=verify_reference.caches.target,
        )
        shape_matched_logits = original_logits(reference_hidden)
        oracle.mx.eval(shape_matched_logits)
    finally:
        oracle._finalize_self_mtp_cache_group(verify_reference.caches.target)
    for row, length in enumerate(verify_lengths):
        _assert_array_equal(
            oracle,
            verify_logits[row : row + 1, :length],
            shape_matched_logits[row : row + 1, :length],
            f"shape_matched.verify_logits[{row}]",
        )

    # The stronger transactional boundary reference advances only the prefix
    # that survives acceptance. Its different T shape is intentionally NOT a
    # logit oracle; it must instead match all post-trim live state and the next
    # same-shape continuation exactly.
    reference = oracle._clone_batch(base)
    token_rows = [
        [proposal._old_curs[row], *proposal._drafts[row][:count]]
        for row, count in enumerate(accepted)
    ]
    lengths = [len(row) for row in token_rows]
    width = max(lengths)
    right_padding = [width - length for length in lengths]
    padded = [row + [0] * (width - len(row)) for row in token_rows]
    oracle._prepare_self_mtp_cache_group(
        reference.caches.target, lengths=lengths, right_padding=right_padding
    )
    try:
        reference_hidden, reference_aux = model.mtp_backbone(
            oracle.mx.array(padded, dtype=oracle.mx.uint32),
            cache=reference.caches.target,
        )
        oracle._materialize_tree((reference_hidden, reference_aux))
    finally:
        oracle._finalize_self_mtp_cache_group(reference.caches.target)

    candidate_boundary = _live_boundary(
        oracle.capture_cache_list(candidate.caches.target, "target")
    )
    reference_boundary = _live_boundary(
        oracle.capture_cache_list(reference.caches.target, "target")
    )
    oracle.assert_oracle_equal(candidate_boundary, reference_boundary)

    continuation_ids = oracle.mx.array(
        [[lane.cur] for lane in candidate.lanes], dtype=oracle.mx.uint32
    )

    def continuation(batch):
        oracle._prepare_self_mtp_cache_group(
            batch.caches.target,
            lengths=[1] * len(batch.lanes),
            right_padding=[0] * len(batch.lanes),
        )
        try:
            hidden, _ = model.mtp_backbone(
                continuation_ids, cache=batch.caches.target
            )
            logits = original_logits(hidden)
            oracle.mx.eval(logits)
        finally:
            oracle._finalize_self_mtp_cache_group(batch.caches.target)
        return logits, _live_boundary(
            oracle.capture_cache_list(batch.caches.target, "target")
        )

    candidate_continuation, candidate_after = continuation(candidate)
    reference_continuation, reference_after = continuation(reference)
    _assert_array_equal(
        oracle,
        candidate_continuation,
        reference_continuation,
        "reference.continuation_logits",
    )
    oracle.assert_oracle_equal(candidate_after, reference_after)
    return {
        "reference_kind": "ordinary_eager_target_accepted_prefix",
        "reference_capture_atoms": len(reference_after),
        "reference_capture_sha256": _capture_digest(reference_after),
        "candidate_reference_raw_bit_equal": True,
        "semantic_alignment_assumptions": [
            "candidate proposed token ids and selected bonus define the transaction input",
            "verify logits compare against an independent eager target with the identical B x (K+1) slab",
            "ordinary accepted-prefix target is a boundary/continuation oracle, not a cross-shape logit oracle",
            "ordinary eager target advances only old_cur plus the accepted draft prefix",
            "rollback record partition and checkpoint history are representation-specific",
            "live recurrent/KV/PLE/QSA/host state and one continuation are compared raw-bit",
            "full candidate rollback, draft, lane, RNG, and proposal state is separately self-consistency attested",
        ],
    }


def _token_rows(tokenizer) -> list[list[int]]:
    rows = [list(map(int, tokenizer.encode(prompt))) for prompt in PROMPTS]
    if any(len(row) < 2 for row in rows):
        raise RuntimeError("attestation prompts encoded to fewer than two tokens")
    return rows


def _attestation_subject(
    contract,
    oracle,
    identity: Mapping[str, Any],
    model,
    base,
    geometry,
):
    install = importlib.import_module(
        "vllm_mlx.spec_decode.mtp.prompt_lookup_attestation_install"
    )
    manifest = importlib.import_module(
        "vllm_mlx.spec_decode.mtp.prompt_lookup_checkpoint_manifest"
    )
    if getattr(model, "model_type", None) != "qwen4_exp":
        raise RuntimeError("loaded model is not the qwen4_exp runtime")
    layers = tuple(model.language_model.model.layers)
    mtp_layers = tuple(model.mtp.layers)
    layer_types = tuple(
        "linear_attention" if layer.is_linear else "full_attention"
        for layer in layers
    )
    mtp_layer_types = tuple(
        "linear_attention" if layer.is_linear else "full_attention"
        for layer in mtp_layers
    )
    ple_layer_ids = tuple(
        index + 1 for index, layer in enumerate(layers) if layer.ple is not None
    )
    cache_classes = tuple(
        [f"target:{type(cache).__name__}" for cache in base.caches.target]
        + [f"mtp:{type(cache).__name__}" for cache in base.caches.draft]
    )
    cache_capture = {
        **oracle.capture_cache_list(base.caches.target, "target"),
        **oracle.capture_cache_list(base.caches.draft, "mtp"),
    }
    state_dtypes = tuple(
        dict.fromkeys(
            atom.dtype
            for _, atom in sorted(cache_capture.items())
            if atom.kind == "array"
        )
    )
    weight_manifest = manifest.compute_checkpoint_weight_manifest_sha256(MODEL)
    runtime = install.build_runtime_attestation_identity(
        model_id="qwen4_exp",
        checkpoint_config_sha256=str(identity["config_sha256"]),
        checkpoint_index_sha256=str(identity["index_sha256"]),
        checkpoint_weight_manifest_sha256=weight_manifest,
        rapid_runtime_commit=str(identity["rapid_commit"]),
        mlx_lm_runtime_commit=str(identity["source_commit"]),
        layer_types=layer_types,
        ple_layer_ids=ple_layer_ids,
        mtp_layer_types=mtp_layer_types,
        cache_classes=cache_classes,
        state_dtypes=state_dtypes,
        verify_geometry=geometry.fingerprint,
        oracle_version=SCHEMA,
    )
    return runtime.to_subject()


def _issue_receipt(
    writer: EvidenceWriter,
    contract,
    oracle,
    identity: Mapping[str, Any],
    model,
    base,
    *,
    batch_size: int,
    cases: Sequence[Any],
) -> None:
    geometry = contract.OracleGeometry(batch_size=batch_size, verify_width=K)
    subject = _attestation_subject(
        contract, oracle, identity, model, base, geometry
    )
    evidence = contract.PromptLookupOracleEvidence(
        subject=subject,
        geometry=geometry,
        production_metal=True,
        route=contract.RouteEngagementEvidence(
            route_name=contract.HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
            execution_kind=contract.OracleExecutionKind.EAGER_TRANSACTION,
            compiled_candidate=False,
            candidate_invocations=len(cases),
            fallback_count=0,
            compile_count=0,
            warmup_compile_count=0,
            post_warmup_recompile_count=0,
        ),
        cases=tuple(cases),
    )
    authority = contract.issue_prompt_lookup_attestation(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=contract.HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )
    writer.append(
        "trusted_receipt",
        batch_size=batch_size,
        receipt_issuer=authority.issuer,
        receipt=contract.prompt_lookup_attestation_receipt_to_payload(
            authority.receipt
        ),
        note=(
            "canonical payload is durable evidence; runtime authority remains "
            "the process-local sealed TrustedPromptLookupReceipt"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = str(uuid.uuid4())
    identity = _validate_static_identity(args)
    writer = EvidenceWriter(
        args.output.resolve(),
        {"schema": SCHEMA, "run_id": run_id, "identity": identity},
    )
    writer.append(
        "start",
        available_memory_bytes=_available_memory_bytes(),
        thermal=_thermal_snapshot(),
    )
    try:
        # MLX imports begin only after every static refusal above has passed.
        import mlx.core as mx
        from mlx_lm import load

        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable")
        oracle = _load_oracle_module()
        contract = _load_receipt_contract(Path(identity["rapid_root"]))
        model, tokenizer = load(str(MODEL))
        model.eval()
        mx.eval(model.parameters())
        rows = _token_rows(tokenizer)
        writer.append(
            "model_loaded",
            model_type=type(model).__name__,
            available_memory_bytes=_available_memory_bytes(),
            thermal=_thermal_snapshot(),
        )

        b1, base_digest, base_atoms = _prepare_base(oracle, model, rows[:1], 100)
        writer.append(
            "immutable_base",
            batch_size=1,
            capture_sha256=base_digest,
            capture_atoms=base_atoms,
        )
        b1_cases = []
        for accepted in product(range(K + 1), repeat=1):
            scenario, receipt_cases = _run_geometry(
                oracle, contract, model, b1, accepted
            )
            writer.append(
                "scenario",
                batch_size=1,
                k=K,
                **scenario,
            )
            b1_cases.extend(receipt_cases)
        _issue_receipt(
            writer,
            contract,
            oracle,
            identity,
            model,
            b1,
            batch_size=1,
            cases=b1_cases,
        )
        # Do not retain B=1 cache slabs or receipt evidence while deciding
        # whether the larger geometry still satisfies the 20 GiB reserve.
        del b1, b1_cases
        gc.collect()
        mx.clear_cache()

        b2_completed = False
        b2_completed_cases = 0
        available = _available_memory_bytes()
        thermal = _thermal_snapshot()
        if (
            available is None
            or available < MIN_B2_AVAILABLE_BYTES
            or not thermal["qualified"]
        ):
            writer.append(
                "batch_skipped",
                batch_size=2,
                reason="memory_reserve_or_thermal_gate",
                available_memory_bytes=available,
                required_available_memory_bytes=MIN_B2_AVAILABLE_BYTES,
                thermal=thermal,
            )
        else:
            b2, base_digest, base_atoms = _prepare_base(oracle, model, rows, 200)
            writer.append(
                "immutable_base",
                batch_size=2,
                capture_sha256=base_digest,
                capture_atoms=base_atoms,
                available_memory_bytes=_available_memory_bytes(),
                thermal=_thermal_snapshot(),
            )
            b2_cases = []
            for accepted in product(range(K + 1), repeat=2):
                # Re-check before every cell; a later pressure/throttle event
                # stops expansion without invalidating already-fsynced B=1 proof.
                cell_memory = _available_memory_bytes()
                cell_thermal = _thermal_snapshot()
                if (
                    cell_memory is None
                    or cell_memory < MIN_B2_AVAILABLE_BYTES
                    or not cell_thermal["qualified"]
                ):
                    writer.append(
                        "batch_aborted",
                        batch_size=2,
                        next_accepted=list(accepted),
                        reason="memory_reserve_or_thermal_gate",
                        available_memory_bytes=cell_memory,
                        thermal=cell_thermal,
                    )
                    break
                scenario, receipt_cases = _run_geometry(
                    oracle, contract, model, b2, accepted
                )
                writer.append(
                    "scenario",
                    batch_size=2,
                    k=K,
                    **scenario,
                )
                b2_cases.extend(receipt_cases)
                b2_completed_cases += 1
            b2_completed = b2_completed_cases == (K + 1) ** 2
            if b2_completed:
                _issue_receipt(
                    writer,
                    contract,
                    oracle,
                    identity,
                    model,
                    b2,
                    batch_size=2,
                    cases=b2_cases,
                )
        writer.append(
            "complete",
            campaign_complete=True,
            qualification_complete_by_geometry={
                "batch_1": True,
                "batch_2": b2_completed,
            },
            highest_qualified_batch_size=2 if b2_completed else 1,
            b1_completed_cases=K + 1,
            b2_completed_cases=b2_completed_cases,
            required_b2_cases=(K + 1) ** 2,
            available_memory_bytes=_available_memory_bytes(),
        )
        return 0
    except BaseException as exc:
        writer.append(
            "failure",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            available_memory_bytes=_available_memory_bytes(),
            thermal=_thermal_snapshot(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
