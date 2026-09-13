#!/usr/bin/env python3
"""Receipted serving-scheduler smoke for static Qwen4 B4 self-MTP cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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


def execute(args):
    import mlx.core as mx
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
    traces = {}
    finished = set()
    events = []
    four = generator.insert(
        [
            exact_prompt(tokenizer, args.context, f"serving-static-{uid}")
            for uid in range(4)
        ],
        max_tokens=[args.max_tokens] * 4,
    )
    traces.update({uid: [] for uid in four})
    fifth = None
    saw_locked_b4 = False
    saw_fifth_queued_while_b4_live = False
    saw_empty_after_b4 = False
    fifth_joined_after_empty = False
    before = segmented_self_mtp_stats(reset=False)
    try:
        for step in range(args.max_steps):
            _prompt, responses = generator.next()
            for response in responses:
                traces[response.uid].append(int(response.token))
                if response.finish_reason is not None:
                    finished.add(response.uid)
            batch = generator._generation_batch
            active = list(batch.uids)
            queued = [int(row[0]) for row in generator._unprocessed_sequences]
            locked_b4 = bool(
                getattr(batch, "_segmented_compute_width_locked", False)
                and len(active) == 4
                and set(active) == set(four)
            )
            events.append({"step": step, "active": active, "queued": queued, "locked_b4": locked_b4})
            if locked_b4:
                saw_locked_b4 = True
                if fifth is None:
                    fifth = generator.insert(
                        [exact_prompt(tokenizer, args.context, "serving-static-fifth")],
                        max_tokens=[args.max_tokens],
                    )[0]
                    traces[fifth] = []
                    continue
            if fifth is not None and locked_b4 and fifth in queued:
                saw_fifth_queued_while_b4_live = True
            if fifth is not None and not active and all(uid in finished for uid in four):
                saw_empty_after_b4 = True
            if fifth is not None and fifth in active and saw_empty_after_b4:
                fifth_joined_after_empty = True
            if fifth is not None and len(finished) == 5:
                break
        after = segmented_self_mtp_stats(reset=False)
    finally:
        generator.close()
    delta = {key: int(value) - int(before.get(key, 0)) for key, value in after.items()
             if isinstance(value, int) and isinstance(before.get(key, 0), int)}
    result = {
        "config": {"context": args.context, "max_tokens": args.max_tokens, "num_draft": args.num_draft},
        "uids": {"initial_b4": four, "fifth": fifth},
        "traces_sha256": {str(uid): digest(tokens) for uid, tokens in traces.items()},
        "events": events,
        "segmented_delta": delta,
        "checks": {
            "initial_b4_locked": saw_locked_b4,
            "fifth_queued_while_b4_live": saw_fifth_queued_while_b4_live,
            "b4_drained_before_fifth": saw_empty_after_b4,
            "fifth_joined_after_empty": fifth_joined_after_empty,
            "all_five_finished": len(finished) == 5,
            "true_batched": delta.get("true_batched_engaged", 0) > 0,
            "no_b1_target": delta.get("b1_target_forwards", 0) == 0,
        },
    }
    result["qualified"] = all(result["checks"].values())
    atomic_write(args.out, result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
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
    parser.add_argument("--out", default="results/qwen4-mtp-static-serving-smoke.json")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
