#!/usr/bin/env python3
"""Receipted two-turn serving smoke for five static Qwen4 B4 MTP requests.

Each turn starts four requests, waits until their segmented B4 forward is
locked, and then admits the fifth.  The scheduler must leave that arrival
queued until the B4 cohort drains.  Turn two reuses each terminal response's
target cache, complete token history, MTP sidecar, and lane RNG from turn one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qwen4_mtp_dynamic_join_gate import DEFAULT_MODEL, exact_prompt


def digest(tokens):
    return hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _continuation(tokenizer, request_id):
    return tokenizer.encode(
        f"\nUser: Follow-up {request_id}: confirm cache continuation.\nAssistant:",
        add_special_tokens=False,
    )


def _carry(response):
    return {
        "cache": response.prompt_cache,
        "tokens": response.all_tokens,
        "mtp_state": response.mtp_state,
        "lane_rng": response.lane_rng,
    }


def _complete_carry(carry):
    # Greedy lanes intentionally have no RNG object.  The handoff contract is
    # that the field is preserved (including an intentional None), while the
    # target cache, token history, and MTP sidecar must be concrete state.
    return (
        carry["cache"] is not None
        and carry["tokens"] is not None
        and carry["mtp_state"] is not None
        and "lane_rng" in carry
    )


def _insert(generator, prompts, carries, max_tokens):
    if carries is None:
        return generator.insert(prompts, max_tokens=[max_tokens] * len(prompts))
    return generator.insert(
        prompts,
        max_tokens=[max_tokens] * len(prompts),
        caches=[carry["cache"] for carry in carries],
        all_tokens=[carry["tokens"] for carry in carries],
        mtp_states=[carry["mtp_state"] for carry in carries],
        lane_rngs=[carry["lane_rng"] for carry in carries],
    )


def run_turn(generator, turn, prompts, carries, max_tokens, max_steps):
    """Run a four-then-one admission pattern and return terminal state by ID."""
    uids = {}
    first_ids = _insert(generator, prompts[:4], None if carries is None else carries[:4], max_tokens)
    uids.update(dict(zip(range(4), first_ids)))
    fifth_inserted = False
    terminal = {}
    traces = {request_id: [] for request_id in range(5)}
    events = []
    saw_locked_b4 = False
    saw_fifth_queued = False
    saw_empty_after_b4 = False
    fifth_joined_after_empty = False

    for step in range(max_steps):
        _prompt, responses = generator.next()
        reverse_uids = {uid: request_id for request_id, uid in uids.items()}
        for response in responses:
            request_id = reverse_uids[response.uid]
            traces[request_id].append(int(response.token))
            if response.finish_reason is not None:
                terminal[request_id] = response

        batch = generator._generation_batch
        active = list(batch.uids)
        queued = [int(row[0]) for row in generator._unprocessed_sequences]
        locked_b4 = bool(
            getattr(batch, "_segmented_compute_width_locked", False)
            and len(active) == 4
            and set(active) == set(first_ids)
        )
        events.append(
            {
                "step": step,
                "active": active,
                "queued": queued,
                "locked_b4": locked_b4,
            }
        )
        if locked_b4:
            saw_locked_b4 = True
            if not fifth_inserted:
                fifth_carry = None if carries is None else [carries[4]]
                uids[4] = _insert(
                    generator, [prompts[4]], fifth_carry, max_tokens
                )[0]
                fifth_inserted = True
                continue
        if fifth_inserted and locked_b4 and uids[4] in queued:
            saw_fifth_queued = True
        if fifth_inserted and not active and all(i in terminal for i in range(4)):
            saw_empty_after_b4 = True
        if fifth_inserted and uids[4] in active and saw_empty_after_b4:
            fifth_joined_after_empty = True
        if len(terminal) == 5:
            break

    result = {
        "turn": turn,
        "uids": uids,
        "events": events,
        "traces_sha256": {str(i): digest(trace) for i, trace in traces.items()},
        "checks": {
            "initial_b4_locked": saw_locked_b4,
            "fifth_queued_while_b4_live": saw_fifth_queued,
            "b4_drained_before_fifth": saw_empty_after_b4,
            "fifth_joined_after_empty": fifth_joined_after_empty,
            "all_five_finished": len(terminal) == 5,
        },
    }
    return result, terminal


def execute(args):
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.segmented_self_mtp import segmented_self_mtp_stats
    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    config = {
        "persistent": True,
        "num_draft": args.num_draft,
        "sampling_temp": 0.0,
        "share_qsa_indices": True,
        "segment_aware_live_tip": True,
        "segment_aware_cohort_size": 4,
    }
    generator = BatchGenerator(
        model,
        max_tokens=args.max_tokens,
        completion_batch_size=5,
        prefill_batch_size=5,
        prefill_step_size=args.prefill_step_size,
        self_mtp=config,
    )
    before = segmented_self_mtp_stats(reset=False)
    try:
        first_prompts = [
            exact_prompt(tokenizer, args.context, f"multiturn-{request_id}")
            for request_id in range(5)
        ]
        first, first_terminal = run_turn(
            generator, 1, first_prompts, None, args.max_tokens, args.max_steps
        )
        carries = [_carry(first_terminal[i]) for i in range(5) if i in first_terminal]
        carry_complete = len(carries) == 5 and all(_complete_carry(c) for c in carries)
        second_prompts = [_continuation(tokenizer, request_id) for request_id in range(5)]
        second, second_terminal = run_turn(
            generator,
            2,
            second_prompts,
            carries if carry_complete else None,
            args.max_tokens,
            args.max_steps,
        )
        after = segmented_self_mtp_stats(reset=False)
    finally:
        generator.close()
    delta = {
        key: int(value) - int(before.get(key, 0))
        for key, value in after.items()
        if isinstance(value, int) and isinstance(before.get(key, 0), int)
    }
    checks = {
        "turn1_static_b4_admission": all(first["checks"].values()),
        "turn1_cache_mtp_state_rng_captured": carry_complete,
        "turn2_static_b4_admission": all(second["checks"].values()),
        "turn2_all_five_finished": len(second_terminal) == 5,
        "true_batched": delta.get("true_batched_engaged", 0) > 0,
        "no_b1_target": delta.get("b1_target_forwards", 0) == 0,
    }
    result = {
        "config": {
            "context": args.context,
            "max_tokens": args.max_tokens,
            "num_draft": args.num_draft,
            "segment_aware_cohort_size": 4,
        },
        "turns": [first, second],
        "turn1_handoff": {
            "requests": len(carries),
            "greedy_none_lane_rngs": sum(
                carry["lane_rng"] is None for carry in carries
            ),
        },
        "segmented_delta": delta,
        "checks": checks,
    }
    result["qualified"] = all(checks.values())
    atomic_write(args.out, result)
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", default="results/qwen4-mtp-static-multiturn-smoke.json")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
