"""Composable, model-agnostic anti-rigging checks — the ``proof_battery``.

Background
----------
The lab's most valuable results are its *negative controls*: the counterfactual
that eliminated a rival explanation for an apparent win. The same handful of
counterfactuals recur across audits (see
``wiki/docs/research/conversation-mined-insights-2026-07-13.md`` §6–§7):

* varied filler + a paraphrased query (PFlash hard-needle rigging),
* correct / wrong-number / alien-sentence controls (grounding vs. parroting),
* same-norm random steering directions (steering effect vs. any perturbation),
* exact token-count and finish-reason checks (delivery / protocol contracts),
* a batched-vs-stepwise greedy reference for quantized near-ties (the Vegas
  FP-near-tie artifact),
* mutation observed through every public state accessor (detached-copy bugs),
* same-identity eviction/reinsertion and claim/error interleavings (lifecycle
  identity),
* logging-off vs. logging-on parity (instrumentation must be side-effect free).

This module turns those counterfactuals into reusable check functions so the
lesson that *proof harnesses are trusted code*
(``wiki/docs/lessons/proof-harnesses-are-trusted-code.md``) becomes executable
infrastructure rather than a review rule.

Design contract
---------------
Every check is **model-agnostic**: it takes plain values and callables, never a
live model, so the whole battery runs on CPU against tiny fixtures with no
weight load, download, or GPU. Every check **fails closed** — on any ambiguity
(missing field, non-finite value, exception inside a supplied callable, empty
sequence, a no-op that should have mutated) it raises :class:`ProofBatteryError`
or returns a struct whose verdict boolean defaults to the *unsafe-to-promote*
value. A detector that cannot prove the property is treated as failing.

Two naming conventions:

* ``assert_*`` / verb-style checks **raise** :class:`ProofBatteryError` on
  failure and return ``None`` on success.
* ``*_controls`` checks **return a result struct** the caller asserts on; the
  struct's verdict boolean is ``False`` whenever the check could not be
  established, including on internal error.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = [
    "ProofBatteryError",
    "l2_norm",
    "is_finite_number",
    "same_norm_random_direction",
    "assert_same_norm",
    "assert_token_path_equal",
    "batched_vs_stepwise_greedy",
    "assert_finish_reason_and_token_count",
    "ControlResult",
    "wrong_number_and_alien_controls",
    "RobustnessResult",
    "varied_filler_paraphrase_controls",
    "state_accessor_mutation_probe",
    "logging_parity",
    "eviction_reinsertion_identity",
    "interleaving_invariant",
]


class ProofBatteryError(AssertionError):
    """A proof-battery check failed or could not be established.

    Subclasses :class:`AssertionError` so existing ``pytest``/``unittest``
    assertion handling and ``self.assertRaises`` continue to work, while still
    being distinguishable from an incidental ``assert`` elsewhere in a harness.
    """


# ---------------------------------------------------------------------------
# Numeric primitives (fail closed on non-finite / malformed input)
# ---------------------------------------------------------------------------


def is_finite_number(x: Any) -> bool:
    """Return ``True`` only for a real, finite ``int``/``float``.

    ``bool`` is rejected (it is an ``int`` subclass but never a valid magnitude
    here) and ``NaN``/``±inf`` are rejected — a gate that accepts NaN was one of
    the July-9 sweep's dominant defect classes.
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def l2_norm(vec: Sequence[float]) -> float:
    """L2 norm of a 1-D numeric sequence.

    Fails closed if ``vec`` is empty or holds any non-finite entry.
    """
    seq = list(vec)
    if not seq:
        raise ProofBatteryError("l2_norm: empty vector")
    for i, v in enumerate(seq):
        if not is_finite_number(v):
            raise ProofBatteryError(f"l2_norm: non-finite entry at index {i}: {v!r}")
    return math.sqrt(sum(float(v) * float(v) for v in seq))


def same_norm_random_direction(vec: Sequence[float], seed: int) -> list[float]:
    """A control vector with the **same L2 norm** as ``vec`` but a random
    direction, deterministic in ``seed``.

    This is the steering control: to claim a steering direction *matters*, show
    the effect vanishes (or differs) when you substitute a random direction of
    identical magnitude. Equal norm removes the trivial "any perturbation of
    that size moves the output" explanation.

    Fails closed if ``vec`` is empty, non-finite, or has zero norm (a zero-norm
    input has no magnitude to preserve — the control would be ill-defined).
    """
    norm = l2_norm(vec)  # validates finiteness / non-emptiness
    if norm == 0.0:
        raise ProofBatteryError(
            "same_norm_random_direction: zero-norm input has no magnitude to preserve"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ProofBatteryError("same_norm_random_direction: seed must be an int")

    n = len(vec)
    rng = random.Random(seed)
    # Draw a Gaussian vector (isotropic direction) and renormalize to ``norm``.
    # Retry on the measure-zero chance of an all-zero draw so we never emit a
    # degenerate control.
    for _ in range(64):
        raw = [rng.gauss(0.0, 1.0) for _ in range(n)]
        raw_norm = math.sqrt(sum(r * r for r in raw))
        if raw_norm > 0.0 and math.isfinite(raw_norm):
            scale = norm / raw_norm
            return [r * scale for r in raw]
    raise ProofBatteryError(
        "same_norm_random_direction: failed to draw a non-degenerate direction"
    )


def assert_same_norm(
    a: Sequence[float], b: Sequence[float], rel_tol: float = 1e-9
) -> None:
    """Raise unless ``a`` and ``b`` share the same L2 norm within ``rel_tol``.

    Used to validate that a control (e.g. from
    :func:`same_norm_random_direction`) is genuinely magnitude-matched.
    """
    if not is_finite_number(rel_tol) or rel_tol < 0:
        raise ProofBatteryError(f"assert_same_norm: bad rel_tol {rel_tol!r}")
    na, nb = l2_norm(a), l2_norm(b)
    if not math.isclose(na, nb, rel_tol=rel_tol, abs_tol=rel_tol):
        raise ProofBatteryError(f"assert_same_norm: norms differ ({na!r} vs {nb!r})")


# ---------------------------------------------------------------------------
# Token-path equality (the Vegas FP-near-tie lesson)
# ---------------------------------------------------------------------------


def _as_token_list(seq: Any, label: str) -> list[int]:
    """Coerce ``seq`` to a list of ints, failing closed on anything ambiguous.

    Token ids — not floating logits and not decoded strings — are the correct
    comparison unit. Comparing logits or strings is exactly how the Vegas 32k
    "divergence" turned out to be a floating-point near-tie artifact.
    """
    try:
        items = list(seq)
    except TypeError as exc:
        raise ProofBatteryError(f"{label}: not iterable ({exc})") from exc
    if not items:
        raise ProofBatteryError(f"{label}: empty token path")
    out: list[int] = []
    for i, t in enumerate(items):
        if isinstance(t, bool) or not isinstance(t, int):
            raise ProofBatteryError(
                f"{label}: token at index {i} is not an int: {t!r}"
            )
        out.append(t)
    return out


def assert_token_path_equal(a: Any, b: Any) -> None:
    """Raise unless two token-id sequences are exactly equal (length + order).

    Fails closed on empty, non-int, or unequal-length paths.
    """
    ta = _as_token_list(a, "assert_token_path_equal[a]")
    tb = _as_token_list(b, "assert_token_path_equal[b]")
    if len(ta) != len(tb):
        raise ProofBatteryError(
            f"assert_token_path_equal: length mismatch ({len(ta)} vs {len(tb)})"
        )
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            raise ProofBatteryError(
                f"assert_token_path_equal: divergence at index {i} ({x} vs {y})"
            )


def batched_vs_stepwise_greedy(
    run_batched: Callable[[], Any], run_stepwise: Callable[[], Any]
) -> None:
    """Assert a batched greedy decode and a stepwise greedy decode produce the
    **identical token path**.

    This is the reference control for "is my optimization lossless?" claims on
    quantized / near-tie paths. A batched-vs-stepwise equality on token *ids*
    removes false divergences caused by float noise in logits or by string
    comparison. Both callables take no arguments and return a token-id sequence.

    Fails closed if either callable raises, or if the two paths differ.
    """
    try:
        a = run_batched()
    except Exception as exc:  # noqa: BLE001 — fail closed on any harness error
        raise ProofBatteryError(f"batched_vs_stepwise_greedy: run_batched raised: {exc}") from exc
    try:
        b = run_stepwise()
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(f"batched_vs_stepwise_greedy: run_stepwise raised: {exc}") from exc
    assert_token_path_equal(a, b)


# ---------------------------------------------------------------------------
# Delivery / protocol contracts (finish reason + exact token count)
# ---------------------------------------------------------------------------


def _get_field(result: Any, keys: Sequence[str]) -> Any:
    """Fetch a field from a mapping or an attribute-carrying object.

    Tries each candidate name as a mapping key first, then as an attribute.
    Raises if no candidate resolves — a missing delivery field is itself a
    failure (the answer may exist but be hidden by a protocol violation).
    """
    for k in keys:
        try:
            if hasattr(result, "__getitem__") and not isinstance(result, (str, bytes)):
                try:
                    return result[k]
                except (KeyError, IndexError, TypeError):
                    pass
        except Exception:  # noqa: BLE001
            pass
        if hasattr(result, k):
            return getattr(result, k)
    raise ProofBatteryError(
        f"delivery field not found; tried {list(keys)!r} on {type(result).__name__}"
    )


def assert_finish_reason_and_token_count(
    result: Any,
    expected_reason: str,
    expected_n: int,
    *,
    reason_keys: Sequence[str] = ("finish_reason", "stop_reason"),
    tokens_keys: Sequence[str] = ("tokens", "token_ids", "generated_tokens"),
    count_keys: Sequence[str] = ("n_tokens", "num_tokens", "token_count"),
) -> None:
    """Assert a generation result stopped for ``expected_reason`` and emitted
    exactly ``expected_n`` tokens.

    ``result`` may be a mapping or an object with attributes. Token count is
    read from an explicit count field if present, else from the length of a
    token sequence. Fails closed on: missing fields, a non-string finish reason,
    a reason mismatch, a non-finite/negative expected count, or a token-count
    mismatch. Exact-count + finish-reason is the delivery contract that
    distinguishes "formed but unreleased" from a genuine natural stop.
    """
    if isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n < 0:
        raise ProofBatteryError(
            f"assert_finish_reason_and_token_count: bad expected_n {expected_n!r}"
        )
    if not isinstance(expected_reason, str) or not expected_reason:
        raise ProofBatteryError(
            f"assert_finish_reason_and_token_count: bad expected_reason {expected_reason!r}"
        )

    reason = _get_field(result, reason_keys)
    if not isinstance(reason, str):
        raise ProofBatteryError(
            f"assert_finish_reason_and_token_count: finish reason is not a str: {reason!r}"
        )
    if reason != expected_reason:
        raise ProofBatteryError(
            f"assert_finish_reason_and_token_count: reason {reason!r} != {expected_reason!r}"
        )

    # Prefer an explicit count field; fall back to len(tokens).
    n: Optional[int] = None
    try:
        raw = _get_field(result, count_keys)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ProofBatteryError(
                f"assert_finish_reason_and_token_count: token count is not an int: {raw!r}"
            )
        n = raw
    except ProofBatteryError:
        toks = _get_field(result, tokens_keys)
        try:
            n = len(toks)
        except TypeError as exc:
            raise ProofBatteryError(
                "assert_finish_reason_and_token_count: token field has no length"
            ) from exc

    if n != expected_n:
        raise ProofBatteryError(
            f"assert_finish_reason_and_token_count: token count {n} != {expected_n}"
        )


# ---------------------------------------------------------------------------
# Grounding controls: correct / wrong-number / alien-sentence
# ---------------------------------------------------------------------------


@dataclass
class ControlResult:
    """Outcome of a correct / wrong-number / alien-sentence control sweep.

    ``discriminates`` is the caller's verdict: it is ``True`` only if the
    ``correct`` control produced a strictly higher signal than *both* the
    wrong-number and alien controls by at least ``margin``, and every measure
    was finite. Any internal error sets ``discriminates=False`` and records a
    ``reason`` (fail closed — an un-establishable result is not a pass).
    """

    correct: Optional[float]
    wrong: Optional[float]
    alien: Optional[float]
    margin: float
    discriminates: bool
    reason: str = ""


def wrong_number_and_alien_controls(
    measure: Callable[[Any], float],
    correct: Any,
    wrong: Any,
    alien: Any,
    *,
    margin: float = 0.0,
) -> ControlResult:
    """Run a grounding control triple and return a :class:`ControlResult`.

    ``measure`` maps a control payload to a numeric signal for the *target*
    answer (e.g. the log-prob the model assigns to the grounded fact, or a
    graded-correctness score). A genuinely grounded system scores the
    ``correct`` context above both a ``wrong``-number variant (same shape,
    different value — catches numeric parroting) and an ``alien`` unrelated
    sentence (catches "answers regardless of context"). The caller asserts on
    ``result.discriminates``.

    Fails closed: any exception in ``measure`` or any non-finite value yields
    ``discriminates=False`` with a recorded reason.
    """
    if not is_finite_number(margin) or margin < 0:
        return ControlResult(None, None, None, margin, False, f"bad margin {margin!r}")

    vals: dict[str, float] = {}
    for label, payload in (("correct", correct), ("wrong", wrong), ("alien", alien)):
        try:
            v = measure(payload)
        except Exception as exc:  # noqa: BLE001
            return ControlResult(
                vals.get("correct"), vals.get("wrong"), vals.get("alien"),
                margin, False, f"measure({label}) raised: {exc}",
            )
        if not is_finite_number(v):
            return ControlResult(
                vals.get("correct"), vals.get("wrong"), vals.get("alien"),
                margin, False, f"measure({label}) non-finite: {v!r}",
            )
        vals[label] = float(v)

    c, w, a = vals["correct"], vals["wrong"], vals["alien"]
    discriminates = (c - w >= margin) and (c - a >= margin) and (c > w) and (c > a)
    reason = "" if discriminates else (
        f"correct={c} did not exceed wrong={w} and alien={a} by margin {margin}"
    )
    return ControlResult(c, w, a, margin, discriminates, reason)


# ---------------------------------------------------------------------------
# Varied filler + paraphrased query (PFlash hard-needle rigging)
# ---------------------------------------------------------------------------


@dataclass
class RobustnessResult:
    """Outcome of a varied-filler / paraphrased-query robustness sweep.

    ``robust`` is ``True`` only if *every* (paraphrase × filler) cell passed.
    A win that only survives the original phrasing on a single fixed filler is
    a rigged-needle / lexical-overlap artifact, not a mechanism. Any exception
    or non-finite score fails the whole sweep closed.
    """

    n_cells: int
    n_passed: int
    robust: bool
    failures: list[tuple[Any, Any]] = field(default_factory=list)
    reason: str = ""


def varied_filler_paraphrase_controls(
    measure: Callable[[Any, Any], Any],
    paraphrases: Sequence[Any],
    fillers: Sequence[Any],
    *,
    passes: Optional[Callable[[Any], bool]] = None,
) -> RobustnessResult:
    """Sweep a claimed retrieval/recall win across paraphrased queries and
    varied filler and require it to hold in **every** cell.

    ``measure(query, filler)`` returns either a bool (pass/fail directly) or a
    numeric score graded by ``passes`` (default: ``score >= 1.0`` after a finite
    check). ``paraphrases`` and ``fillers`` must each be non-empty — a
    single-phrasing, single-filler "sweep" cannot rule out the rigged-needle
    explanation, so an empty axis fails closed.

    The caller asserts on ``result.robust``.
    """
    para = list(paraphrases)
    fill = list(fillers)
    if not para or not fill:
        return RobustnessResult(
            0, 0, False, [], "paraphrases and fillers must both be non-empty"
        )

    def default_passes(score: Any) -> bool:
        return is_finite_number(score) and float(score) >= 1.0

    grade = passes or default_passes
    n_cells = 0
    n_passed = 0
    failures: list[tuple[Any, Any]] = []
    for q in para:
        for f in fill:
            n_cells += 1
            try:
                raw = measure(q, f)
            except Exception as exc:  # noqa: BLE001
                return RobustnessResult(
                    n_cells, n_passed, False, failures,
                    f"measure raised on (query={q!r}, filler={f!r}): {exc}",
                )
            if isinstance(raw, bool):
                ok = raw
            else:
                try:
                    ok = bool(grade(raw))
                except Exception as exc:  # noqa: BLE001
                    return RobustnessResult(
                        n_cells, n_passed, False, failures,
                        f"grader raised on {raw!r}: {exc}",
                    )
            if ok:
                n_passed += 1
            else:
                failures.append((q, f))

    robust = not failures
    reason = "" if robust else f"{len(failures)}/{n_cells} cells failed"
    return RobustnessResult(n_cells, n_passed, robust, failures, reason)


# ---------------------------------------------------------------------------
# State-accessor mutation probe (detached-copy / stale-view detection)
# ---------------------------------------------------------------------------


def state_accessor_mutation_probe(
    accessors: Sequence[Callable[[], Any]],
    mutate: Callable[[], Any],
) -> None:
    """Prove that a mutation is visible through **every** public accessor.

    ``accessors`` is a list of zero-arg callables that each read the same
    logical piece of state through a different public route (getter, property,
    exported view, serialized snapshot, …). ``mutate`` performs an in-place
    change to that state. After mutation, every accessor's value must differ
    from its pre-mutation value; an accessor whose value is unchanged is a
    detached copy / stale view and is a reuse-correctness hazard.

    Fails closed on: an empty accessor list, an accessor or ``mutate`` that
    raises, or any accessor that does not reflect the mutation (including the
    case where ``mutate`` was a silent no-op).
    """
    accs = list(accessors)
    if not accs:
        raise ProofBatteryError("state_accessor_mutation_probe: no accessors supplied")

    before = []
    for i, acc in enumerate(accs):
        try:
            before.append(acc())
        except Exception as exc:  # noqa: BLE001
            raise ProofBatteryError(
                f"state_accessor_mutation_probe: accessor[{i}] raised before mutate: {exc}"
            ) from exc

    try:
        mutate()
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(
            f"state_accessor_mutation_probe: mutate raised: {exc}"
        ) from exc

    for i, acc in enumerate(accs):
        try:
            after = acc()
        except Exception as exc:  # noqa: BLE001
            raise ProofBatteryError(
                f"state_accessor_mutation_probe: accessor[{i}] raised after mutate: {exc}"
            ) from exc
        if after == before[i]:
            raise ProofBatteryError(
                f"state_accessor_mutation_probe: accessor[{i}] did not reflect the "
                f"mutation (stale/detached view or no-op mutate); value stayed {after!r}"
            )


# ---------------------------------------------------------------------------
# Logging-off vs logging-on parity (side-effect-free instrumentation)
# ---------------------------------------------------------------------------


def logging_parity(
    run_with: Callable[[], Any],
    run_without: Callable[[], Any],
    *,
    equal: Optional[Callable[[Any, Any], bool]] = None,
) -> None:
    """Assert instrumentation is side-effect-free: the primary output is
    identical with logging/telemetry on vs. off.

    ``run_with`` runs the computation with logging enabled; ``run_without`` with
    it disabled. Both take no arguments and return the primary result (token
    path, answer, digest, …). Equality is ``==`` by default or a supplied
    ``equal`` predicate. Fails closed if either callable raises, if the
    predicate raises, or if the outputs differ — telemetry that perturbs the
    result invalidates every measurement taken with it on.
    """
    try:
        a = run_with()
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(f"logging_parity: run_with raised: {exc}") from exc
    try:
        b = run_without()
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(f"logging_parity: run_without raised: {exc}") from exc

    if equal is None:
        same = a == b
    else:
        try:
            same = bool(equal(a, b))
        except Exception as exc:  # noqa: BLE001
            raise ProofBatteryError(f"logging_parity: equal() raised: {exc}") from exc
    if not same:
        raise ProofBatteryError(
            f"logging_parity: outputs differ with logging on vs off ({a!r} vs {b!r})"
        )


# ---------------------------------------------------------------------------
# Lifecycle identity: same-identity eviction / reinsertion
# ---------------------------------------------------------------------------


def eviction_reinsertion_identity(
    read: Callable[[Any], Any],
    evict: Callable[[Any], Any],
    reinsert: Callable[[Any, Any], Any],
    key: Any,
    *,
    sentinel: Any = None,
    equal: Optional[Callable[[Any, Any], bool]] = None,
) -> None:
    """Prove that evicting an entry and reinserting it under the **same
    identity** restores its exact state.

    Steps, each fail-closed:

    1. ``read(key)`` must return the live state (not ``sentinel``), captured as
       the snapshot to restore.
    2. ``evict(key)`` must actually remove it — a subsequent ``read(key)`` must
       return ``sentinel`` (a fake/no-op eviction is a failure, because it hides
       whether reinsertion truly reconstructs state).
    3. ``reinsert(key, snapshot)`` followed by ``read(key)`` must equal the
       original snapshot. A mismatch means identity was not preserved across the
       lifecycle (the hostile-boundary class from the stateful-optimization
       lessons).

    Equality is ``==`` by default or a supplied ``equal`` predicate.
    """

    def eq(x: Any, y: Any) -> bool:
        if equal is None:
            return x == y
        try:
            return bool(equal(x, y))
        except Exception as exc:  # noqa: BLE001
            raise ProofBatteryError(f"eviction_reinsertion_identity: equal() raised: {exc}") from exc

    try:
        snapshot = read(key)
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(f"eviction_reinsertion_identity: read raised: {exc}") from exc
    if eq(snapshot, sentinel):
        raise ProofBatteryError(
            "eviction_reinsertion_identity: key absent before eviction (nothing to test)"
        )

    try:
        evict(key)
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(f"eviction_reinsertion_identity: evict raised: {exc}") from exc
    try:
        after_evict = read(key)
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(
            f"eviction_reinsertion_identity: read after evict raised: {exc}"
        ) from exc
    if not eq(after_evict, sentinel):
        raise ProofBatteryError(
            "eviction_reinsertion_identity: eviction was a no-op (entry still present)"
        )

    try:
        reinsert(key, snapshot)
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(f"eviction_reinsertion_identity: reinsert raised: {exc}") from exc
    try:
        restored = read(key)
    except Exception as exc:  # noqa: BLE001
        raise ProofBatteryError(
            f"eviction_reinsertion_identity: read after reinsert raised: {exc}"
        ) from exc
    if not eq(restored, snapshot):
        raise ProofBatteryError(
            "eviction_reinsertion_identity: reinserted state does not match original "
            f"({restored!r} != {snapshot!r})"
        )


# ---------------------------------------------------------------------------
# Interleaving invariant (claim/error interleavings and friends)
# ---------------------------------------------------------------------------


def interleaving_invariant(
    steps: Sequence[Callable[[], Any]],
    invariant: Callable[[], bool],
) -> None:
    """Run a sequence of interleaved operations and assert an ``invariant``
    holds **after every step**.

    ``steps`` is an ordered list of zero-arg thunks — e.g. an interleaving of
    claim / release / error operations against a shared ledger or cache.
    ``invariant`` is a zero-arg predicate read after each step; it must return
    ``True`` at every point. This catches lifecycle corruption that only appears
    under a specific interleaving (an error after a claim that fails to roll
    back, a double-release, a reinsertion racing an eviction).

    Fails closed if ``steps`` is empty, if any step or the invariant raises, or
    if the invariant is violated after any step. The invariant is also checked
    once before the first step so a bad starting state is caught.
    """
    seq = list(steps)
    if not seq:
        raise ProofBatteryError("interleaving_invariant: no steps supplied")

    def check(where: str) -> None:
        try:
            ok = invariant()
        except Exception as exc:  # noqa: BLE001
            raise ProofBatteryError(
                f"interleaving_invariant: invariant raised {where}: {exc}"
            ) from exc
        if not isinstance(ok, bool):
            raise ProofBatteryError(
                f"interleaving_invariant: invariant returned non-bool {where}: {ok!r}"
            )
        if not ok:
            raise ProofBatteryError(f"interleaving_invariant: invariant violated {where}")

    check("at start")
    for i, step in enumerate(seq):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            raise ProofBatteryError(
                f"interleaving_invariant: step[{i}] raised: {exc}"
            ) from exc
        check(f"after step[{i}]")
