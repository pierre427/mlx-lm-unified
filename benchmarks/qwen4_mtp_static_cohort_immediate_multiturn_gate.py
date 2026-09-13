#!/usr/bin/env python3
"""Five-conversation immediate-follow-up gate for static Qwen4 B4 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from qwen4_mtp_dynamic_join_gate import (
    DEFAULT_MODEL,
    classify_first_envelope_flip,
    exact_prompt,
)


def digest(tokens):
    return hashlib.sha256(b"".join(int(x).to_bytes(4, "little") for x in tokens)).hexdigest()


def envelope(logprobs):
    """Capture the bounded top-two receipt used by the established B>1 gate."""
    import mlx.core as mx

    values = mx.reshape(logprobs, (-1,))
    top = mx.argsort(values)[-2:]
    scale = mx.max(mx.abs(values))
    mx.eval(top, values[top], scale)
    runner_up, winner = (int(value) for value in top.tolist())
    runner_up_value, winner_value = (float(value) for value in values[top].tolist())
    return {
        "top1_token": winner,
        "top1_logprob": winner_value,
        "top2_token": runner_up,
        "top2_logprob": runner_up_value,
        "top2_margin": winner_value - runner_up_value,
        "scale": float(scale.item()),
    }


def compare_baseline(tokens, envelopes, baseline):
    """Classify each request-turn only through its first divergent token."""
    diagnostics = {}
    for key, candidate_tokens in tokens.items():
        label = f"{key[0]}:{key[1]}"
        diagnostics[label] = classify_first_envelope_flip(
            baseline["tokens"][label],
            candidate_tokens,
            baseline["envelopes"][label],
            envelopes[key],
            shape_noise_band=0.15,
        )
    return diagnostics


def first_turns_within_batch_shape_band(diagnostics):
    """Accept exact or documented near-tie first-turn B1/B>1 differences.

    Follow-up turns deliberately do not enter this diagnostic: their carried
    recurrent and attention states were produced by different compute shapes,
    so a later logit comparison is no longer an aligned first-divergence test.
    """
    return all(
        diagnostic is None or diagnostic["near_tie_candidate"]
        for label, diagnostic in diagnostics.items()
        if label.endswith(":1")
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def continuation(tokenizer, request_id):
    return tokenizer.encode(
        f"\nUser: follow-up {request_id}; verify continuation.\nAssistant:",
        add_special_tokens=False,
    )


def insert(generator, prompt, maximum, carry=None):
    args = {"max_tokens": [maximum]}
    if carry is not None:
        args.update(
            caches=[carry.prompt_cache], all_tokens=[carry.all_tokens],
            mtp_states=[carry.mtp_state], lane_rngs=[carry.lane_rng],
        )
    return generator.insert([prompt], **args)[0]


def run(args):
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.segmented_self_mtp import segmented_self_mtp_stats
    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    generator = BatchGenerator(
        model, completion_batch_size=5, prefill_batch_size=5,
        prefill_step_size=args.prefill_step_size,
        self_mtp={"persistent": True, "num_draft": args.num_draft,
                  "sampling_temp": 0.0, "share_qsa_indices": True,
                  "segment_aware_live_tip": True,
                  "segment_aware_cohort_size": 4},
    )
    before = segmented_self_mtp_stats(reset=False)
    start = time.monotonic()
    initial = [
        insert(generator, exact_prompt(tokenizer, args.context, f"immediate-{i}"),
               args.max_tokens - i * args.token_stagger)
        for i in range(4)
    ]
    lanes = {uid: (i, 1) for i, uid in enumerate(initial)}
    tokens = {(i, turn): [] for i in range(5) for turn in (1, 2)}
    envelopes = {(i, turn): [] for i in range(5) for turn in (1, 2)}
    queued_at = {}
    followups = {}
    events = []
    initial_fifth = None
    initial_fifth_queued = False
    initial_b4_locked = False
    empty_since = False
    completed = set()
    try:
        for step in range(args.max_steps):
            _prompt, responses = generator.next()
            batch = generator._generation_batch
            active = list(batch.uids)
            queued = [int(row[0]) for row in generator._unprocessed_sequences]
            if (
                args.serial_schedule_control
                and step >= 2
                and len(active) == 4
                and set(active) == set(initial)
            ):
                # Preserve the candidate's empty-seam schedule while the
                # segmented consumer executes each row as B1. This is a
                # diagnostic oracle, not a true-batching qualification arm.
                batch._segmented_compute_width_locked = True
            locked_initial = bool(
                getattr(batch, "_segmented_compute_width_locked", False)
                and len(active) == 4 and set(active) == set(initial)
            )
            initial_b4_locked |= locked_initial
            events.append({"step": step, "active": active, "queued": queued,
                           "locked_initial_b4": locked_initial})
            if locked_initial and initial_fifth is None:
                initial_fifth = insert(
                    generator, exact_prompt(tokenizer, args.context, "immediate-4"),
                    args.max_tokens - 4 * args.token_stagger,
                )
                lanes[initial_fifth] = (4, 1)
                queued_at[initial_fifth] = step
            if initial_fifth is not None and initial_fifth in queued:
                initial_fifth_queued = True
            if not active:
                empty_since = True
            ready_followups = []
            for response in responses:
                request_id, turn = lanes[response.uid]
                tokens[(request_id, turn)].append(int(response.token))
                envelopes[(request_id, turn)].append(envelope(response.logprobs))
                if response.finish_reason is None:
                    continue
                completed.add((request_id, turn))
                if turn != 1:
                    continue
                valid = (
                    response.prompt_cache is not None
                    and response.all_tokens is not None
                    and response.mtp_state is not None
                )
                if not valid:
                    continue
                ready_followups.append((request_id, response))
            # Response callbacks are delivered after BatchGenerator has
            # committed every terminal lane from this scheduler tick.  Model
            # the server's boundary exactly: enqueue all resulting user turns
            # only after that completion set is detached.
            post_terminal_active = list(generator._generation_batch.uids)
            for request_id, response in ready_followups:
                uid = insert(generator, continuation(tokenizer, request_id),
                             args.followup_tokens, response)
                lanes[uid] = (request_id, 2)
                followups[uid] = {"request": request_id, "insert_step": step,
                                  "active_at_insert": post_terminal_active,
                                  "empty_seen": empty_since}
                queued_at[uid] = step
            events[-1]["post_response_active"] = list(generator._generation_batch.uids)
            if len(completed) == 10:
                break
        after = segmented_self_mtp_stats(reset=False)
    finally:
        generator.close()
    delta = {k: int(v) - int(before.get(k, 0)) for k, v in after.items()
             if isinstance(v, int) and isinstance(before.get(k, 0), int)}
    def waited_for_empty(uid, meta):
        later = [event for event in events if event["step"] > meta["insert_step"]]
        if not meta["active_at_insert"]:
            return True
        first_active = next(
            (index for index, event in enumerate(later) if uid in event["active"]),
            None,
        )
        if first_active is None:
            return False
        if any(uid in event["queued"] for event in later[:first_active]):
            return True
        # An empty seam can occur inside BatchGenerator.next(): terminal lanes
        # detach, then the next cohort forms before the next external snapshot.
        # In that case no member of the preceding active cohort survives into
        # the follow-up's first cohort.
        previous = later[first_active - 1]["active"] if first_active else meta["active_at_insert"]
        return not (set(previous) & set(later[first_active]["active"]))

    followup_waited_for_empty = all(
        waited_for_empty(uid, meta) for uid, meta in followups.items()
    )
    result = {
        "config": vars(args), "elapsed_s": time.monotonic() - start,
        "events": events, "followups": followups,
        "tokens": {f"{i}:{t}": tokens[(i, t)] for i in range(5) for t in (1, 2)},
        "envelopes": {f"{i}:{t}": envelopes[(i, t)] for i in range(5) for t in (1, 2)},
        "digests": {f"{i}:{t}": digest(tokens[(i, t)]) for i in range(5) for t in (1, 2)},
        "segmented_delta": delta,
        "checks": {
            "initial_b4_locked": initial_b4_locked,
            "initial_fifth_queued": initial_fifth_queued,
            "all_five_two_turns_complete": len(completed) == 10,
            "five_immediate_followups": len(followups) == 5,
            "followups_wait_for_empty_seam": followup_waited_for_empty,
        },
    }
    if args.serial_schedule_control:
        result["checks"].update(
            segmented_b1_control=delta.get("b1_target_forwards", 0) > 0,
            no_batched_target=delta.get("batched_target_forwards", 0) == 0,
        )
    else:
        result["checks"].update(
            true_batched=delta.get("true_batched_engaged", 0) > 0,
            no_b1_target=delta.get("b1_target_forwards", 0) == 0,
        )
    if args.baseline_digests:
        baseline = json.loads(Path(args.baseline_digests).read_text())
        result["baseline"] = str(args.baseline_digests)
        if "tokens" in baseline and "envelopes" in baseline:
            result["first_divergences"] = compare_baseline(
                tokens, envelopes, baseline
            )
        if args.baseline_mode == "same-shape-exact":
            result["checks"]["same_shape_replay_exact"] = (
                result["digests"] == baseline.get("digests")
            )
        else:
            if not baseline.get("config", {}).get("serial_schedule_control"):
                raise ValueError(
                    "b1-first-turn-diagnostic requires a same-schedule B1 control"
                )
            if "first_divergences" not in result:
                raise ValueError(
                    "b1-first-turn-diagnostic requires token and envelope receipts"
                )
            result["checks"]["b1_first_turns_within_batch_shape_band"] = (
                first_turns_within_batch_shape_band(result["first_divergences"])
            )
    result["qualified"] = all(result["checks"].values())
    write_json(args.out, result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    return 0 if result["qualified"] else 2


def run_serial(args):
    """Produce a genuine one-conversation-at-a-time B1 digest oracle."""
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    digests = {}
    token_receipts = {}
    envelope_receipts = {}
    complete = True
    for request_id in range(5):
        generator = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1,
            prefill_step_size=args.prefill_step_size,
            self_mtp={"persistent": True, "num_draft": args.num_draft,
                      "sampling_temp": 0.0, "share_qsa_indices": True,
                      # Keep the cache/sidecar implementation identical to
                      # the B4 run; the external B1 env control selects the
                      # serial consumer, while width one prevents batching.
                      "segment_aware_live_tip": True,
                      "segment_aware_cohort_size": 1},
        )
        try:
            prompt = exact_prompt(tokenizer, args.context, f"immediate-{request_id}")
            uid = insert(generator, prompt, args.max_tokens - request_id * args.token_stagger)
            terminal = None
            first_tokens = []
            first_envelopes = []
            for _ in range(args.max_steps):
                _prompt, responses = generator.next()
                for response in responses:
                    if response.uid == uid:
                        first_tokens.append(int(response.token))
                        first_envelopes.append(envelope(response.logprobs))
                        if response.finish_reason is not None:
                            terminal = response
                if terminal is not None:
                    break
            digests[f"{request_id}:1"] = digest(first_tokens)
            token_receipts[f"{request_id}:1"] = first_tokens
            envelope_receipts[f"{request_id}:1"] = first_envelopes
            if terminal is None or terminal.prompt_cache is None or terminal.mtp_state is None:
                complete = False
                continue
            uid = insert(generator, continuation(tokenizer, request_id), args.followup_tokens, terminal)
            second_tokens = []
            second_envelopes = []
            terminal = None
            for _ in range(args.max_steps):
                _prompt, responses = generator.next()
                for response in responses:
                    if response.uid == uid:
                        second_tokens.append(int(response.token))
                        second_envelopes.append(envelope(response.logprobs))
                        if response.finish_reason is not None:
                            terminal = response
                if terminal is not None:
                    break
            digests[f"{request_id}:2"] = digest(second_tokens)
            token_receipts[f"{request_id}:2"] = second_tokens
            envelope_receipts[f"{request_id}:2"] = second_envelopes
            complete &= terminal is not None
        finally:
            generator.close()
    result = {"config": vars(args), "tokens": token_receipts,
              "envelopes": envelope_receipts, "digests": digests,
              "checks": {"all_five_two_turns_complete": complete},
              "qualified": complete}
    write_json(args.out, result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    return 0 if complete else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--serial-baseline", action="store_true")
    parser.add_argument(
        "--serial-schedule-control",
        action="store_true",
        help="Keep the B4 empty-seam schedule while the environment selects B1 compute.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--followup-tokens", type=int, default=16)
    parser.add_argument("--token-stagger", type=int, default=4)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--out", default="results/qwen4-mtp-immediate-multiturn.json")
    parser.add_argument(
        "--baseline-digests",
        help="Receipt used by the selected baseline comparison mode.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("same-shape-exact", "b1-first-turn-diagnostic"),
        default="same-shape-exact",
        help=(
            "Require exact replay at the same B4 shape, or classify only the "
            "aligned first-turn B1/B4 divergences through the established "
            "batch-shape near-tie band (default: same-shape-exact)."
        ),
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return 0
    return run_serial(args) if args.serial_baseline else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
