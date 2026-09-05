"""Reviewed serving qualifications, distinct from research shape eligibility.

This module has no MLX dependency. No checkpoint is approved by default.
Set ``MLX_LM_COMPILED_DECODE_QUALIFICATION`` before loading to select an
operator-approved JSON manifest. It must contain:

* ``schema: 1``, ``qualification_id``, ``approval: "approved"``, ``approved_by``;
* every field from :func:`qualification_identity` (config and weight hashes,
  parameter shape/dtype layout, MLX artifacts, Python source and execution flags);
* ``class3_maxdelta_bound`` and ``profiles`` mapping each enabled policy name to
  ``max_context``, ``buckets`` and ``numerical_acceptance``;
* ``evidence`` and ``serving_evidence``, each ``{path: absolute_path, sha256: hash}``.

The numerical file is the benchmark JSON with ``qualification_identity`` added.
It must cover the policy endpoint and every bucket-growth boundary, with exact
greedy sequences, exact compiled-vs-ring logits, bounded stock differences,
and completion-backed replay counts. A benchmark candidate is not approval.

The separate serving file uses ``schema: "compiled-serving-e2e-v1"``, the same
``qualification_identity``, ``route: "ordinary-unseeded-n1"``, and ``cases`` for
``cold``, ``apc_hit``, ``eos``, ``length``, ``cancel`` and ``concurrent``. Each
case records matching ``stock_tokens``/``candidate_tokens``, ``compiled_used``
(false only for the caller-owned APC hit), zero ``pending``/``failed``, false
``poisoned``, and positive finite ``stock_ttft_ms``, ``candidate_ttft_ms``,
``stock_total_ms``, ``candidate_total_ms``. The concurrency case also records
``requests: 2`` and ``serialized: true``. These tests run in an isolated test
harness with explicit test-only qualification injection (as in the integration
tests), never a relaxed production gate. They do not authorize production
serving. There is no speed floor.

Approval is an operator review step after both files exist. Enablement still
requires ``MLX_LM_COMPILED_DECODE=1`` and the explicit numerical acceptance
token. A missing, stale or mismatched manifest leaves normal eager service
available. Restart/reload after changing the manifest or execution flags.
"""

import hashlib
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path

# Context limit per compiled profile; mirrors ``compiled_decode._PROFILE_LIMITS``
# (that module imports this one, so the table lives here and is asserted equal
# by the source-contract tests).
_PROFILE_LIMITS = {"short": 4096, "memory": 16384, "latency": 16384, "long": 262144}

SERVING_QUALIFICATIONS = {}
_MANIFEST_ENV = "MLX_LM_COMPILED_DECODE_QUALIFICATION"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_environment():
    """Bind execution flags, not harness provenance or activation controls."""
    controls = {
        _MANIFEST_ENV,
        "MLX_LM_COMPILED_DECODE",
        "MLX_LM_COMPILED_DECODE_ACCEPTANCE",
        "MLX_LM_COMPILED_DECODE_CONTEXT_POLICY",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("MLX", "QWEN"))
        and not key.startswith("MLXUAG_")
        and key not in controls
    }


def runtime_identity(mlx_core):
    root = Path(__file__).parent
    mlx_root = Path(mlx_core.__file__).parent
    return {
        "mlx_version": mlx_core.__version__,
        "mlx_artifacts": {
            str(path.relative_to(mlx_root)): _file_digest(path)
            for path in sorted(mlx_root.rglob("*"))
            if path.is_file()
            and path.suffix in (".so", ".dylib", ".metallib", ".metal", ".h", ".py")
        },
        "sources": {
            str(path.relative_to(root)): _file_digest(path)
            for path in sorted(root.rglob("*.py"))
        },
    }


def qualification_identity(config, weight_files, *, runtime, parameters):
    """Produce candidate identity, never an approval; hashing is CPU-only."""
    return {
        "config_sha256": _digest(config),
        "weights_sha256": {Path(p).name: _file_digest(p) for p in sorted(weight_files)},
        "parameter_layout": [
            [name, list(value.shape), str(value.dtype)] for name, value in parameters
        ],
        "runtime": runtime,
        "environment": execution_environment(),
    }


def _read_json(path, limit=64 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("compiled qualification JSON exceeds size limit")
    with path.open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("compiled qualification JSON must be an object")
    return result


_IDENTITY_KEYS = (
    "config_sha256",
    "weights_sha256",
    "parameter_layout",
    "runtime",
    "environment",
)


def _revalidation(record):
    """Operator-approved re-validation of prior evidence after a source change.

    ``record["revalidation"]`` names the ``mlx_lm`` sources that changed since
    the prior evidence was produced (``changed_sources``: name -> the digest
    the evidence was produced with), the operator approval, and a spot-check
    receipt produced on the live tree. Prior evidence is then accepted only if
    its identity differs from the record's in exactly those sources at exactly
    those digests; the spot check must carry the record's identity and be
    bit-identical at every point it holds. Without this block, evidence must
    match the record exactly (the default contract).
    """
    reval = record.get("revalidation")
    if reval is None:
        return None
    changed = reval.get("changed_sources") if isinstance(reval, dict) else None
    if (
        not isinstance(changed, dict)
        or not changed
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not digest
            for name, digest in changed.items()
        )
        or not isinstance(reval.get("approved_by"), str)
        or not reval["approved_by"].strip()
        or not isinstance(reval.get("spot_check"), dict)
    ):
        raise ValueError("compiled serving manifest revalidation is malformed")
    return reval


def _evidence_identity_matches(record, evidence_identity, reval):
    identity = {key: record.get(key) for key in _IDENTITY_KEYS}
    if evidence_identity == identity:
        return True
    if reval is None or not isinstance(evidence_identity, dict):
        return False
    if any(
        evidence_identity.get(key) != identity.get(key)
        for key in _IDENTITY_KEYS
        if key != "runtime"
    ):
        return False
    live_runtime = identity.get("runtime")
    prior_runtime = evidence_identity.get("runtime")
    if not isinstance(live_runtime, dict) or not isinstance(prior_runtime, dict):
        return False
    if any(
        live_runtime.get(key) != prior_runtime.get(key)
        for key in set(live_runtime) | set(prior_runtime)
        if key != "sources"
    ):
        return False
    live_sources = live_runtime.get("sources")
    prior_sources = prior_runtime.get("sources")
    if (
        not isinstance(live_sources, dict)
        or not isinstance(prior_sources, dict)
        or set(live_sources) != set(prior_sources)
    ):
        return False
    differing = {
        name for name in live_sources if live_sources[name] != prior_sources[name]
    }
    changed = reval["changed_sources"]
    return differing == set(changed) and all(
        changed[name] == prior_sources[name] for name in differing
    )


def _validate_point(point, *, name, buckets, endpoint, seed, bound):
    if not isinstance(point, dict):
        raise ValueError("qualification lacks endpoint or growth evidence")
    steps = point.get("steps")
    if (
        type(steps) is not int
        or not (128 if seed is None else 2) <= steps < endpoint
        or point.get("context_policy") != name
        or point.get("policy_buckets") != buckets
        or point.get("measurement_end_context") != endpoint
        or point.get("seed_context") != endpoint - steps
        or (seed is not None and point["seed_context"] != seed)
    ):
        raise ValueError("qualification numerical geometry is inconsistent")
    tokens = point.get("digest_kv")
    if (
        not isinstance(tokens, list)
        or len(tokens) != steps
        or any(type(token) is not int or token < 0 for token in tokens)
        or tokens != point.get("digest_ring")
        or tokens != point.get("digest_compiled")
    ):
        raise ValueError("qualification greedy token evidence differs")
    if (
        point.get("submitted") != steps
        or point.get("completed") != steps
        or point.get("pending") != 0
        or point.get("failed") != 0
        or point.get("poisoned") is not False
        or point.get("single_trace") is not True
        or any(
            type(point.get(key)) is not int
            for key in ("submitted", "completed", "pending", "failed")
        )
        or not isinstance(point.get("traces"), list)
        or not point["traces"]
        or any(type(n) is not int or n != 1 for n in point["traces"])
    ):
        raise ValueError("qualification replay completion evidence is incomplete")
    expected_caps = sorted(
        {
            next(b for b in buckets if b >= cursor)
            for cursor in range(endpoint - steps + 1, endpoint + 1)
        }
    )
    if (
        point.get("capacities_observed") != expected_caps
        or point.get("ring_capacities_observed") != expected_caps
    ):
        raise ValueError("qualification capacity evidence differs")
    for field in (
        "logits_compiled_vs_ring",
        "logits_ring_vs_kv",
        "logits_compiled_vs_kv",
    ):
        comparison = point.get(field, {})
        if not isinstance(comparison, dict):
            raise ValueError("qualification logit comparison must be an object")
        delta = comparison.get("maxdelta")
        allowed = 0 if field == "logits_compiled_vs_ring" else bound
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
            or not 0 <= delta <= allowed
            or (
                field == "logits_compiled_vs_ring"
                and comparison.get("bitidentical") is not True
            )
        ):
            raise ValueError("qualification logits exceed accepted numerical class")


def _spot_check_evidence(record, evidence):
    """A receipt produced on the live tree after a source change: it must carry
    the record's exact identity and bound, and every point it holds must pass
    the same checks as qualification evidence for its profile."""
    if evidence.get("qualification_identity") != {
        key: record.get(key) for key in _IDENTITY_KEYS
    }:
        raise ValueError("spot-check evidence was not produced on the live runtime")
    if evidence.get("class3_maxdelta_bound") != record.get("class3_maxdelta_bound"):
        raise ValueError("spot-check numerical bound differs")
    points = evidence.get("numerical_operating_points", {})
    growth = evidence.get("numerical_growth_boundaries", {})
    if not isinstance(points, dict) or not points or not isinstance(growth, dict):
        raise ValueError("spot-check evidence has no operating points")
    bound = record["class3_maxdelta_bound"]
    checked = 0
    for key, point in list(points.items()) + list(growth.items()):
        name = key.rsplit("_", 1)[-1] if key in points else key.split("_boundary", 1)[0]
        profile = record["profiles"].get(name)
        if not isinstance(profile, dict):
            raise ValueError("spot-check evidence names an unqualified profile")
        buckets = profile["buckets"]
        if key in points:
            endpoint, seed = int(key[len("ctx"):].split("_", 1)[0]), None
        else:
            boundary = int(key.split("_boundary", 1)[1])
            endpoint, seed = boundary + 1, boundary - 1
        _validate_point(
            point, name=name, buckets=buckets, endpoint=endpoint, seed=seed, bound=bound
        )
        checked += 1
    return checked


def _numerical_evidence(record, evidence, reval=None):
    """Check the actual per-profile token/completion/capacity evidence."""
    if not _evidence_identity_matches(
        record, evidence.get("qualification_identity"), reval
    ):
        raise ValueError("qualification evidence belongs to another model/runtime")
    bound = evidence.get("class3_maxdelta_bound")
    if (
        isinstance(bound, bool)
        or not isinstance(bound, (int, float))
        or not math.isfinite(bound)
        or bound < 0
        or bound != record.get("class3_maxdelta_bound")
    ):
        raise ValueError("qualification numerical bound is missing or differs")
    points = evidence.get("numerical_operating_points", {})
    growth = evidence.get("numerical_growth_boundaries", {})
    if not isinstance(points, dict) or not isinstance(growth, dict):
        raise ValueError("qualification evidence has no operating points")
    for name, policy in record["profiles"].items():
        if not isinstance(policy, dict):
            raise ValueError("qualification profile must be an object")
        end = policy.get("max_context")
        buckets = policy.get("buckets")
        if (
            name not in _PROFILE_LIMITS
            or end != _PROFILE_LIMITS[name]
            or not isinstance(buckets, list)
            or not buckets
            or any(type(x) is not int or x <= 0 for x in buckets)
            or buckets != sorted(set(buckets))
            or buckets[-1] < end
            or policy.get("numerical_acceptance")
            not in ("class3-padded-sdpa-v1", "class1-bucketed-v1")
        ):
            raise ValueError("qualification profile is malformed")
        required = [(points.get(f"ctx{end}_M1_{name}"), end, None)]
        required += [
            (growth.get(f"{name}_boundary{b}"), b + 1, b - 1)
            for b in buckets
            if 1 < b < end
        ]
        for point, endpoint, seed in required:
            _validate_point(
                point, name=name, buckets=buckets, endpoint=endpoint, seed=seed, bound=bound
            )


def qualification_records():
    """Load an explicitly approved operator record; no environment auto-approval."""
    path = os.environ.get(_MANIFEST_ENV)
    if not path:
        return SERVING_QUALIFICATIONS
    record = _read_json(path, 1024 * 1024)
    if (
        record.get("schema") != 1
        or not isinstance(record.get("qualification_id"), str)
        or not record["qualification_id"]
        or not isinstance(record.get("approved_by"), str)
        or not record["approved_by"].strip()
        or record.get("approval") != "approved"
        or not isinstance(record.get("profiles"), dict)
        or not record["profiles"]
    ):
        raise ValueError("compiled serving manifest lacks explicit operator approval")
    reference = record.get("evidence", {})
    if not isinstance(reference, dict):
        raise ValueError("qualification evidence reference must be an object")
    evidence_path = Path(reference.get("path", ""))
    if not evidence_path.is_absolute() or _file_digest(evidence_path) != reference.get(
        "sha256"
    ):
        raise ValueError("compiled serving evidence digest mismatch")
    reval = _revalidation(record)
    _numerical_evidence(record, _read_json(evidence_path), reval)
    reference = record.get("serving_evidence", {})
    if not isinstance(reference, dict):
        raise ValueError("serving evidence reference must be an object")
    serving_path = Path(reference.get("path", ""))
    if not serving_path.is_absolute() or _file_digest(serving_path) != reference.get(
        "sha256"
    ):
        raise ValueError("compiled serving end-to-end evidence digest mismatch")
    _serving_evidence(record, _read_json(serving_path), reval)
    if reval is not None:
        spot = reval["spot_check"]
        spot_path = Path(spot.get("path", ""))
        if not spot_path.is_absolute() or _file_digest(spot_path) != spot.get("sha256"):
            raise ValueError("compiled serving spot-check evidence digest mismatch")
        _spot_check_evidence(record, _read_json(spot_path))
    return {record["qualification_id"]: record}


def _serving_evidence(record, evidence, reval=None):
    if evidence.get("schema") != "compiled-serving-e2e-v1" or not (
        _evidence_identity_matches(record, evidence.get("qualification_identity"), reval)
    ):
        raise ValueError("serving evidence belongs to another model/runtime")
    cases = evidence.get("cases")
    required = {"cold", "apc_hit", "eos", "length", "cancel", "concurrent"}
    if (
        not isinstance(cases, dict)
        or not required.issubset(cases)
        or evidence.get("route") != "ordinary-unseeded-n1"
    ):
        raise ValueError("serving evidence is missing request lifecycle cases")
    for name in required:
        case = cases[name]
        if not isinstance(case, dict):
            raise ValueError("serving evidence case is malformed")
        stock, candidate = case.get("stock_tokens"), case.get("candidate_tokens")
        if (
            not isinstance(stock, list)
            or not stock
            or stock != candidate
            or any(type(token) is not int or token < 0 for token in stock)
            or case.get("compiled_used") is not (name != "apc_hit")
            or case.get("pending") != 0
            or case.get("failed") != 0
            or case.get("poisoned") is not False
        ):
            raise ValueError("serving evidence lacks exact tokens/completed execution")
        for metric in (
            "stock_ttft_ms",
            "candidate_ttft_ms",
            "stock_total_ms",
            "candidate_total_ms",
        ):
            value = case.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("serving evidence lacks end-to-end timing")
    if (
        cases["concurrent"].get("requests") != 2
        or cases["concurrent"].get("serialized") is not True
    ):
        raise ValueError("serving evidence does not cover serialized concurrency")


@dataclass(frozen=True)
class ServingBinding:
    qualification_id: str
    record_digest: str
    model_id: int
    parameter_ids: tuple
    environment_digest: str
    manifest_path: str = ""
    manifest_digest: str = ""
    record_json: str = ""


def bind_serving_qualification(model, config, weight_files, *, runtime, parameters):
    """Bind an exact reviewed record after strict loading, never approve one.

    Full weight hashing runs only when the catalogue has a matching config.
    Parameters are ``(name, array)`` pairs; reading shape/dtype does not eval.
    """
    model._compiled_decode_serving_binding = None
    records = qualification_records()
    config_digest = _digest(config)
    candidates = [
        (name, record)
        for name, record in records.items()
        if record.get("config_sha256") == config_digest
    ]
    if not candidates:
        return
    parameters = tuple(parameters)
    layout = [[name, list(value.shape), str(value.dtype)] for name, value in parameters]
    weights = {Path(path).name: _file_digest(path) for path in sorted(weight_files)}
    environment = execution_environment()
    for name, record in candidates:
        if (
            record.get("schema") == 1
            and record.get("weights_sha256") == weights
            and _digest(record.get("parameter_layout")) == _digest(layout)
            and record.get("runtime") == runtime
            and record.get("environment") == environment
            and record.get("evidence")
            and record.get("profiles")
        ):
            model._compiled_decode_serving_binding = ServingBinding(
                name,
                _digest(record),
                id(model),
                tuple((key, id(value)) for key, value in parameters),
                _digest(environment),
                os.environ.get(_MANIFEST_ENV, ""),
                (
                    _file_digest(os.environ[_MANIFEST_ENV])
                    if os.environ.get(_MANIFEST_ENV)
                    else ""
                ),
                json.dumps(record, sort_keys=True),
            )
            return


def serving_qualification_reason(model, *, parameters, policy=None):
    binding = getattr(model, "_compiled_decode_serving_binding", None)
    if not isinstance(binding, ServingBinding) or binding.model_id != id(model):
        error = getattr(model, "_compiled_decode_qualification_error", None)
        return "no reviewed checkpoint qualification bound by the model loader" + (
            f": {error}" if error else ""
        )
    if binding.manifest_path:
        try:
            if (
                os.environ.get(_MANIFEST_ENV) != binding.manifest_path
                or _file_digest(binding.manifest_path) != binding.manifest_digest
            ):
                return "compiled serving manifest changed; reload the model"
        except OSError:
            return "compiled serving manifest is unavailable; reload the model"
        record = json.loads(binding.record_json)
    else:
        record = SERVING_QUALIFICATIONS.get(binding.qualification_id)
    if record is None or _digest(record) != binding.record_digest:
        return "the bound compiled serving qualification changed or was retired"
    if _digest(execution_environment()) != binding.environment_digest:
        return "compiled serving environment changed since model loading"
    if tuple((name, id(value)) for name, value in parameters) != binding.parameter_ids:
        return "model parameters changed since compiled serving qualification"
    if policy is not None:
        profile = record["profiles"].get(policy.name)
        if profile != {
            "max_context": policy.max_context,
            "buckets": list(policy.buckets),
            "numerical_acceptance": policy.numerical_acceptance,
        }:
            return "this compiled context profile has not been qualified for serving"
    return None
