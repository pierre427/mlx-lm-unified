#!/usr/bin/env python3
"""Real OpenAI-server gate for five immediate two-turn Qwen4 conversations.

The default invocation is plan-only. ``--execute`` starts one isolated local
server with the explicitly opt-in static segmented cohort flags, drives two
identical five-conversation waves, samples ``/v1/status/self-mtp``, writes an
atomic receipt, and always reaps the child server. The outer operator still
owns GPU serialization; run this script through ``results/cpg_job.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
SERVER_EXPERIMENT_ENV = {
    "MLX_LM_SEGMENTED_SELF_MTP": "1",
    "MLX_LM_SEGMENTED_SELF_MTP_TIMING": "1",
}
REQUIRED_CHECKS = (
    "isolated_config",
    "five_primary_two_turn_conversations",
    "immediate_followups",
    "all_http_200",
    "all_self_mtp_receipts",
    "all_server_timings",
    "true_batched",
    "no_b1_target",
    "queue_admission_observed",
    "committed_transactions",
    "service_replay_exact_or_near_tie",
    "replay_uses_apc",
    "isolated_server_reaped",
)


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def token_digest(tokens):
    return hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def response_tokens(response):
    content = response["choices"][0].get("logprobs", {}).get("content", [])
    return [int(item["id"]) for item in content]


def response_token_trace(response):
    content = response["choices"][0].get("logprobs", {}).get("content", [])
    return [
        {
            "id": int(item["id"]),
            "logprob": float(item["logprob"]),
            "top_logprobs": [
                {"id": int(candidate["id"]), "logprob": float(candidate["logprob"])}
                for candidate in item.get("top_logprobs", [])
            ],
        }
        for item in content
    ]


def response_text(response):
    message = response["choices"][0]["message"]
    return str(message.get("content") or "") + str(message.get("reasoning") or "")


def followup_text(request_id):
    return f"Immediate follow-up {request_id}: verify cache ownership briefly."


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def server_command(args):
    return [
        args.python,
        "-m",
        "mlx_lm.server",
        "--model",
        args.model,
        "--single-model",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--decode-concurrency",
        "5",
        "--prompt-concurrency",
        "5",
        "--prompt-batch-window",
        "20",
        "--prefill-step-size",
        str(args.prefill_step_size),
        "--chat-template-args",
        '{"enable_thinking":false,"preserve_thinking":true}',
        "--self-mtp",
        "--self-mtp-persistent",
        "--self-mtp-num-draft",
        str(args.num_draft),
        "--self-mtp-max-lanes",
        "5",
        "--no-self-mtp-rate-gate",
        # Batched server admission deliberately excludes request-level shared
        # QSA indices. APCv2 still retains the model's QSA summary plane, and
        # the pooled-key-cache environment remains independently eligible.
        "--self-mtp-segment-aware-live-tip",
        "--self-mtp-segment-aware-cohort-size",
        "4",
        "--log-level",
        "INFO",
    ]


def server_environment():
    environment = os.environ.copy()
    environment.update(SERVER_EXPERIMENT_ENV)
    return environment


def port_is_free(host, port):
    with socket.socket() as probe:
        return probe.connect_ex((host, port)) != 0


def wait_ready(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with rc={process.returncode}"
            )
        try:
            status, payload = get_json(f"{base_url}/v1/models", 2.0)
            if status == 200 and payload.get("data"):
                return payload
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready: {last_error}")


def template_length(tokenizer, text):
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
    )


def build_prompt(tokenizer, target, marker):
    # Put the lineage marker before the repeated body.  Each immediate
    # follow-up then finds its own longest APC entry instead of a sibling's
    # nearly-identical prefix with an unrelated MTP sidecar.
    prefix = (
        "OpenAI service batching qualification context follows.\n"
        f"Conversation marker {marker}.\n"
    )
    suffix = "\nGive a numbered cache review."
    unit = "state cache scheduler invariant rollback token x "

    def render(count):
        return prefix + unit * count + suffix

    low, high = 0, 1
    while template_length(tokenizer, render(high)) <= target:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if template_length(tokenizer, render(middle)) <= target:
            low = middle
        else:
            high = middle
    prompt = render(low)
    return prompt, template_length(tokenizer, prompt)


def compact_status(payload, elapsed):
    segmented = payload.get("segmented", {})
    return {
        "elapsed_s": elapsed,
        "engaged_requests": payload.get("engaged_requests", 0),
        "prompt_cache_entries": payload.get("prompt_cache", {}).get("entries", 0),
        "segmented": {
            key: segmented.get(key, 0)
            for key in (
                "true_batched_engaged",
                "batched_target_forwards",
                "b1_target_forwards",
                "live_width_change_deferrals",
                "committed_cycles",
                "transaction_promotions",
            )
        },
    }


def counter_delta(before, after):
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in set(before) | set(after)
        if isinstance(before.get(key, 0), int) and isinstance(after.get(key, 0), int)
    }


def one_conversation(
    base_url,
    model_id,
    wave,
    request_id,
    prompt,
    first_max,
    second_max,
    barrier,
    timeout,
):
    messages = [{"role": "user", "content": prompt}]
    barrier.wait(timeout=timeout)
    # Keep all five requests within the prompt batching window while making
    # their insertion order repeatable across the primary and replay waves.
    time.sleep(request_id * 0.001)
    turns = []
    first_submit = time.perf_counter()
    status, first = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": first_max,
            "logprobs": True,
            "top_logprobs": 2,
        },
        timeout,
    )
    first_done = time.perf_counter()
    turns.append(
        {
            "turn": 1,
            "http_status": status,
            "submitted_s": first_submit,
            "completed_s": first_done,
            "latency_ms": (first_done - first_submit) * 1000.0,
            "usage": first.get("usage", {}),
            "finish_reason": first["choices"][0].get("finish_reason"),
            "tokens": response_tokens(first),
            "token_trace": response_token_trace(first),
            "self_mtp_receipt": first.get("self_mtp_receipt"),
        }
    )
    messages.extend(
        [
            {"role": "assistant", "content": response_text(first)},
            {
                "role": "user",
                "content": followup_text(request_id),
            },
        ]
    )
    second_submit = time.perf_counter()
    status, second = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": second_max,
            "logprobs": True,
            "top_logprobs": 2,
        },
        timeout,
    )
    second_done = time.perf_counter()
    turns.append(
        {
            "turn": 2,
            "http_status": status,
            "submitted_s": second_submit,
            "completed_s": second_done,
            "latency_ms": (second_done - second_submit) * 1000.0,
            "immediate_followup_gap_ms": (second_submit - first_done) * 1000.0,
            "usage": second.get("usage", {}),
            "finish_reason": second["choices"][0].get("finish_reason"),
            "tokens": response_tokens(second),
            "token_trace": response_token_trace(second),
            "self_mtp_receipt": second.get("self_mtp_receipt"),
        }
    )
    return {"wave": wave, "request": request_id, "turns": turns}


def run_wave(args, base_url, model_id, prompts, wave):
    barrier = threading.Barrier(5)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(
                one_conversation,
                base_url,
                model_id,
                wave,
                request_id,
                prompts[request_id],
                args.first_max_tokens - request_id * args.token_stagger,
                args.second_max_tokens,
                barrier,
                args.request_timeout,
            )
            for request_id in range(5)
        ]
        conversations = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    conversations.sort(key=lambda item: item["request"])
    completion_tokens = sum(
        int(turn["usage"].get("completion_tokens", 0))
        for item in conversations
        for turn in item["turns"]
    )
    return {
        "wave": wave,
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "aggregate_completion_tps": completion_tokens / elapsed,
        "conversations": conversations,
    }


def wave_digests(wave):
    return {
        f"{item['request']}:{turn['turn']}": token_digest(turn["tokens"])
        for item in wave["conversations"]
        for turn in item["turns"]
    }


def all_turns(waves):
    return [
        turn
        for wave in waves
        for conversation in wave["conversations"]
        for turn in conversation["turns"]
    ]


def turns_by_key(wave):
    return {
        f"{conversation['request']}:{turn['turn']}": turn
        for conversation in wave["conversations"]
        for turn in conversation["turns"]
    }


def first_divergence(left, right, near_tie_logprob_gap):
    left_trace = left["token_trace"]
    right_trace = right["token_trace"]
    limit = min(len(left_trace), len(right_trace))
    index = next(
        (
            position
            for position in range(limit)
            if left_trace[position]["id"] != right_trace[position]["id"]
        ),
        None,
    )
    if index is None:
        if len(left_trace) == len(right_trace):
            return None
        return {
            "index": limit,
            "kind": "length",
            "certified_near_tie": False,
        }
    left_item = left_trace[index]
    right_item = right_trace[index]
    left_top = {
        int(candidate["id"]): float(candidate["logprob"])
        for candidate in left_item["top_logprobs"]
    }
    right_top = {
        int(candidate["id"]): float(candidate["logprob"])
        for candidate in right_item["top_logprobs"]
    }
    left_id = int(left_item["id"])
    right_id = int(right_item["id"])
    left_gap = (
        abs(left_top[left_id] - left_top[right_id])
        if left_id in left_top and right_id in left_top
        else None
    )
    right_gap = (
        abs(right_top[left_id] - right_top[right_id])
        if left_id in right_top and right_id in right_top
        else None
    )
    certified = (
        left_gap is not None
        and right_gap is not None
        and max(left_gap, right_gap) <= near_tie_logprob_gap
    )
    return {
        "index": index,
        "kind": "token",
        "left_id": left_id,
        "right_id": right_id,
        "left_gap": left_gap,
        "right_gap": right_gap,
        "threshold": near_tie_logprob_gap,
        "certified_near_tie": certified,
    }


def replay_exactness(primary, replay, near_tie_logprob_gap):
    left = turns_by_key(primary)
    right = turns_by_key(replay)
    rows = {}
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            rows[key] = {
                "exact": False,
                "missing": True,
                "certified_near_tie": False,
            }
            continue
        exact = left[key]["tokens"] == right[key]["tokens"]
        divergence = None if exact else first_divergence(
            left[key], right[key], near_tie_logprob_gap
        )
        rows[key] = {
            "exact": exact,
            "first_divergence": divergence,
            "certified_near_tie": bool(
                divergence and divergence.get("certified_near_tie")
            ),
        }
    return rows


def qualification_passed(result):
    return (
        "error" not in result
        and all(result.get("checks", {}).get(key) is True for key in REQUIRED_CHECKS)
    )


def run(args):
    from mlx_lm.utils import load_tokenizer

    if not port_is_free(args.host, args.port):
        raise RuntimeError(f"refusing occupied isolated port {args.host}:{args.port}")
    tokenizer = load_tokenizer(args.model)
    prompts_and_lengths = [
        build_prompt(tokenizer, args.context_tokens, f"request-{index}")
        for index in range(5)
    ]
    prompts = [item[0] for item in prompts_and_lengths]
    prompt_lengths = [item[1] for item in prompts_and_lengths]
    base_url = f"http://{args.host}:{args.port}"
    command = server_command(args)
    result = {
        "config": vars(args),
        "server_command": command,
        "server_experiment_env": SERVER_EXPERIMENT_ENV,
        "prompt_tokens": prompt_lengths,
        "waves": [],
        "status_samples": [],
        "checks": {},
    }
    server_log = Path(args.server_log)
    server_log.parent.mkdir(parents=True, exist_ok=True)
    process = None
    monitor_stop = threading.Event()
    monitor = None
    started = time.perf_counter()
    try:
        with server_log.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).parents[1],
                env=server_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            models = wait_ready(base_url, process, args.startup_timeout)
            model_id = models["data"][0]["id"]
            _, before = get_json(f"{base_url}/v1/status/self-mtp", 10.0)
            result["models"] = models
            result["status_before"] = before

            def sample_status():
                while not monitor_stop.wait(0.1):
                    try:
                        _, payload = get_json(
                            f"{base_url}/v1/status/self-mtp", 2.0
                        )
                        result["status_samples"].append(
                            compact_status(payload, time.perf_counter() - started)
                        )
                    except Exception:
                        pass

            monitor = threading.Thread(target=sample_status, daemon=True)
            monitor.start()
            result["waves"].append(
                run_wave(args, base_url, model_id, prompts, "primary")
            )
            result["waves"].append(
                run_wave(args, base_url, model_id, prompts, "replay")
            )
            monitor_stop.set()
            monitor.join(timeout=5.0)
            _, after = get_json(f"{base_url}/v1/status/self-mtp", 10.0)
            result["status_after"] = after
            delta = counter_delta(
                before.get("segmented", {}), after.get("segmented", {})
            )
            result["segmented_delta"] = delta
            result["digests"] = {
                wave["wave"]: wave_digests(wave) for wave in result["waves"]
            }
            result["replay_exactness"] = replay_exactness(
                result["waves"][0],
                result["waves"][1],
                args.near_tie_logprob_gap,
            )
            turns = all_turns(result["waves"])
            primary_turns = all_turns(result["waves"][:1])
            replay_turns = all_turns(result["waves"][1:])
            configured = after.get("configured", {})
            result["checks"].update(
                isolated_config=(
                    configured.get("segment_aware_live_tip") is True
                    and configured.get("segment_aware_cohort_size") == 4
                    and configured.get("enabled") is True
                ),
                five_primary_two_turn_conversations=len(primary_turns) == 10,
                immediate_followups=all(
                    turn.get("immediate_followup_gap_ms", 0.0) < 100.0
                    for turn in primary_turns
                    if turn["turn"] == 2
                ),
                all_http_200=all(turn["http_status"] == 200 for turn in turns),
                all_self_mtp_receipts=all(
                    isinstance(turn.get("self_mtp_receipt"), dict) for turn in turns
                ),
                all_server_timings=all(
                    isinstance(
                        (turn.get("self_mtp_receipt") or {}).get("server_timing_ms"),
                        dict,
                    )
                    for turn in turns
                ),
                true_batched=delta.get("true_batched_engaged", 0) > 0,
                no_b1_target=delta.get("b1_target_forwards", 0) == 0,
                queue_admission_observed=any(
                    float(
                        (
                            (turn.get("self_mtp_receipt") or {}).get(
                                "server_timing_ms"
                            )
                            or {}
                        ).get("generation_admission", 0.0)
                    )
                    > 1.0
                    for turn in turns
                ),
                committed_transactions=delta.get("committed_cycles", 0) > 0,
                same_shape_service_replay_exact=(
                    result["digests"]["primary"] == result["digests"]["replay"]
                ),
                service_replay_exact_or_near_tie=all(
                    row["exact"] or row["certified_near_tie"]
                    for row in result["replay_exactness"].values()
                ),
                replay_uses_apc=any(
                    int((turn.get("self_mtp_receipt") or {}).get("cached_prompt_tokens", 0))
                    > 0
                    for turn in replay_turns
                ),
            )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=5.0)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        result["server_returncode"] = None if process is None else process.returncode
        result["checks"]["isolated_server_reaped"] = (
            process is not None and process.poll() is not None
        )
        result["elapsed_s"] = time.perf_counter() - started
        result["qualified"] = qualification_passed(result)
        atomic_write(args.out, result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    if result.get("error"):
        print(result["error"])
    return 0 if result["qualified"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8297)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--first-max-tokens", type=int, default=32)
    parser.add_argument("--second-max-tokens", type=int, default=16)
    parser.add_argument("--token-stagger", type=int, default=4)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--near-tie-logprob-gap", type=float, default=0.5)
    parser.add_argument(
        "--out", default="results/qwen4-openai-static-multiturn.json"
    )
    parser.add_argument(
        "--server-log",
        default="results/qwen4-openai-static-multiturn.server.log",
    )
    args = parser.parse_args(argv)
    if args.context_tokens < 256:
        parser.error("--context-tokens must be at least 256")
    if args.first_max_tokens - 4 * args.token_stagger < 1:
        parser.error("first-turn stagger leaves request 4 with no output budget")
    if args.second_max_tokens < 1:
        parser.error("--second-max-tokens must be positive")
    if args.near_tie_logprob_gap < 0:
        parser.error("--near-tie-logprob-gap must be non-negative")
    if not args.execute:
        print(
            json.dumps(
                {"config": vars(args), "server_command": server_command(args)},
                indent=2,
            )
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
