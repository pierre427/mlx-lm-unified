#!/usr/bin/env python3
"""Matched APC-cached 20K, batch-four, realistic multi-turn plain/MTP gate.

Plan-only by default. With ``--execute`` this starts two isolated servers in
sequence, primes the same tokenizer-measured 20K system prefix into APC, and
runs four synchronized three-turn conversations first without any speculative
lane and then with static-cohort self-MTP depth two. Raw responses, cache and
batch engagement receipts, process RSS samples, and host stability snapshots
are written atomically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
MTP_ENV = {
    "MLX_LM_SEGMENTED_SELF_MTP": "1",
    "MLX_LM_SEGMENTED_SELF_MTP_TIMING": "1",
}
CONVERSATIONS = (
    (
        "An API latency alert is firing after a routine rollout. Give the on-call "
        "engineer a concise triage order, separating checks from actions.",
        "The cache hit rate is normal but p95 decode latency doubled. Revise the "
        "diagnosis and name the two most useful measurements.",
        "Write the final six-line incident handoff for the next shift.",
    ),
    (
        "Plan a low-risk deployment of a batching scheduler change. Include entry "
        "criteria, rollback triggers, and the minimum evidence to retain.",
        "During canary, throughput rises but tail latency regresses 12 percent. "
        "Decide whether to proceed and explain the decision briefly.",
        "Turn that into a release-manager checklist with explicit stop conditions.",
    ),
    (
        "A local inference service must support four concurrent long-context chats. "
        "Recommend cache and capacity controls without assuming speculation wins.",
        "Now assume the shared prefix is 20K tokens and is already in APC. Update "
        "the recommendation and identify what still consumes incremental memory.",
        "Summarize the capacity recommendation as three measurable SLOs.",
    ),
    (
        "Draft an internal support response for intermittent 503s under burst load. "
        "Be factual, avoid promising a root cause, and request useful diagnostics.",
        "Engineering confirms the process stayed alive and memory was stable. "
        "Rewrite the response to distinguish overload from a crash.",
        "Produce a final customer-safe update under 90 words.",
    ),
)


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def get_json(url, timeout=10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read())


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


def port_is_free(host, port):
    with socket.socket() as probe:
        return probe.connect_ex((host, port)) != 0


def wait_ready(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={process.returncode}")
        try:
            status, payload = get_json(f"{base_url}/v1/models", 2.0)
            if status == 200 and payload.get("data"):
                return payload
        except Exception as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready: {last_error}")


def template_tokens(tokenizer, messages):
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
    )


def common_prefix_length(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def system_lcp_tokens(tokenizer, system_prompt):
    first = template_tokens(
        tokenizer,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": "APC prime marker."}],
    )
    second = template_tokens(
        tokenizer,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": CONVERSATIONS[0][0]}],
    )
    return common_prefix_length(first, second)


def build_system_prompt(tokenizer, target):
    header = (
        "You are the operations copilot for a local inference engineering team. "
        "Use the following static runbook as background. Prefer reversible actions, "
        "measured evidence, precise uncertainty, and concise handoffs.\n\n"
    )
    templates = (
        "Observe service health, request latency, queue delay, throughput, cache "
        "reuse, resident memory, swap, and thermal state before changing controls.",
        "Separate configured behavior from engaged behavior; require counters or "
        "request receipts before attributing an outcome to an optimization.",
        "For incidents, preserve raw timestamps and logs, state the user impact, "
        "test the smallest hypothesis, and define a rollback condition in advance.",
        "For performance comparisons, match workload, batch shape, output budget, "
        "cache state, and sampling policy; report aggregate and per-request latency.",
        "A healthy process can still overload. Distinguish admission delay, prefill "
        "delay, decode slowdown, client timeout, and an actual generation crash.",
        "Automatic prefix caching saves repeated prefix prefill but does not erase "
        "private conversation history, generated-token state, or transient buffers.",
        "Keep deployment decisions evidence based. A throughput gain does not excuse "
        "an unbounded tail-latency or memory regression without an explicit tradeoff.",
        "Handoffs name current state, verified facts, open hypotheses, owners, next "
        "measurement, and the exact condition that requires escalation or rollback.",
    )

    def render(count):
        body = "\n".join(
            f"Runbook section {index + 1}: {templates[index % len(templates)]}"
            for index in range(count)
        )
        return header + body

    low, high = 0, 1
    while system_lcp_tokens(tokenizer, render(high)) < target:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if system_lcp_tokens(tokenizer, render(middle)) <= target:
            low = middle
        else:
            high = middle
    prompt = render(low)
    filler = "\nOperational control: measure, verify, preserve, rollback."
    while system_lcp_tokens(tokenizer, prompt + filler) <= target:
        prompt += filler
    current = system_lcp_tokens(tokenizer, prompt)
    while current < target:
        candidate = prompt + " measure"
        candidate_tokens = system_lcp_tokens(tokenizer, candidate)
        if candidate_tokens <= current or candidate_tokens > target:
            break
        prompt, current = candidate, candidate_tokens
    return prompt, current


def server_command(args, arm):
    command = [
        args.python, "-m", "mlx_lm.server", "--model", args.model,
        "--single-model", "--host", args.host, "--port", str(args.port),
        "--decode-concurrency", "4", "--prompt-concurrency", "4",
        "--prompt-batch-window", "20", "--prefill-step-size",
        str(args.prefill_step_size), "--prompt-cache-size", "64",
        "--chat-template-args",
        '{"enable_thinking":false,"preserve_thinking":true}',
    ]
    if arm == "mtp":
        command += [
            "--self-mtp", "--self-mtp-persistent", "--self-mtp-num-draft", "2",
            "--self-mtp-max-lanes", "4", "--no-self-mtp-rate-gate",
            "--self-mtp-segment-aware-live-tip",
            "--self-mtp-segment-aware-cohort-size", "4",
        ]
    return command + ["--log-level", "INFO"]


def server_environment(arm):
    environment = os.environ.copy()
    for key in MTP_ENV:
        environment.pop(key, None)
    if arm == "mtp":
        environment.update(MTP_ENV)
    return environment


def host_snapshot():
    snapshot = {}
    for name, command in {
        "vm_stat": ["vm_stat"],
        "swap": ["sysctl", "vm.swapusage"],
        "memory_pressure": ["memory_pressure", "-Q"],
        "thermal": ["pmset", "-g", "therm"],
    }.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            snapshot[name] = {
                "returncode": result.returncode,
                "output": (result.stdout + result.stderr).strip(),
            }
        except Exception as error:
            snapshot[name] = {"error": f"{type(error).__name__}: {error}"}
    return snapshot


def process_rss_bytes(pid):
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True, text=True, timeout=3,
    )
    text = result.stdout.strip()
    return int(text) * 1024 if result.returncode == 0 and text else None


def response_row(response, status, submitted, completed):
    content = response["choices"][0].get("logprobs", {}).get("content", [])
    tokens = [int(item["id"]) for item in content]
    message = response["choices"][0]["message"]
    text = str(message.get("content") or "") + str(message.get("reasoning") or "")
    return {
        "http_status": status,
        "latency_ms": (completed - submitted) * 1000.0,
        "usage": response.get("usage", {}),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "tokens": tokens,
        "token_digest": hashlib.sha256(
            b"".join(token.to_bytes(4, "little") for token in tokens)
        ).hexdigest(),
        "text": text,
        "self_mtp_receipt": response.get("self_mtp_receipt"),
    }


def run_turn_wave(args, base_url, model_id, system_prompt, histories, turn_index):
    barrier = threading.Barrier(4)

    def request_one(index):
        messages = [
            {"role": "system", "content": system_prompt},
            *histories[index],
            {"role": "user", "content": CONVERSATIONS[index][turn_index]},
        ]
        barrier.wait(timeout=args.request_timeout)
        time.sleep(index * 0.001)
        submitted = time.perf_counter()
        status, response = post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model_id,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": args.turn_tokens[turn_index],
                "logprobs": True,
                "top_logprobs": 2,
            },
            args.request_timeout,
        )
        completed = time.perf_counter()
        row = response_row(response, status, submitted, completed)
        return index, row

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(request_one, range(4)))
    elapsed = time.perf_counter() - started
    rows.sort()
    completion_tokens = sum(
        int(row["usage"].get("completion_tokens", 0)) for _, row in rows
    )
    for index, row in rows:
        histories[index].extend([
            {"role": "user", "content": CONVERSATIONS[index][turn_index]},
            {"role": "assistant", "content": row["text"]},
        ])
    return {
        "turn": turn_index + 1,
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "aggregate_completion_tps": completion_tokens / elapsed,
        "responses": [row for _, row in rows],
    }


def counter_delta(before, after):
    keys = set(before) | set(after)
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
        if type(before.get(key, 0)) is int and type(after.get(key, 0)) is int
    }


def run_arm(args, arm, system_prompt, prefix_tokens):
    if not port_is_free(args.host, args.port):
        raise RuntimeError(f"refusing occupied port {args.host}:{args.port}")
    base_url = f"http://{args.host}:{args.port}"
    command = server_command(args, arm)
    server_log = Path(args.server_log_dir) / f"{arm}.server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "arm": arm, "server_command": command,
        "server_environment": MTP_ENV if arm == "mtp" else {},
        "apc_target_prefix_tokens": args.system_tokens,
        "apc_actual_prefix_tokens": prefix_tokens,
        "host_before": host_snapshot(), "status_samples": [], "turns": [],
    }
    process = None
    stop = threading.Event()
    monitor = None
    started = time.perf_counter()
    try:
        with server_log.open("w") as log:
            process = subprocess.Popen(
                command, cwd=Path(__file__).parents[1],
                env=server_environment(arm), stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            models = wait_ready(base_url, process, args.startup_timeout)
            model_id = models["data"][0]["id"]
            result["model_id"] = model_id

            def sample():
                while not stop.wait(0.25):
                    row = {"elapsed_s": time.perf_counter() - started,
                           "rss_bytes": process_rss_bytes(process.pid)}
                    try:
                        _, status = get_json(f"{base_url}/v1/status/self-mtp", 2.0)
                        row["prompt_cache"] = status.get("prompt_cache", {})
                        row["batch_decode"] = status.get("batch_decode", {})
                        row["segmented"] = status.get("segmented", {})
                    except Exception as error:
                        row["status_error"] = f"{type(error).__name__}: {error}"
                    result["status_samples"].append(row)

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            _, before = get_json(f"{base_url}/v1/status/self-mtp")
            result["status_before_prime"] = before
            prime_status, prime = post_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "APC prime marker."},
                    ],
                    "temperature": 0.0, "max_tokens": 1,
                },
                args.request_timeout,
            )
            result["prime"] = {
                "http_status": prime_status, "usage": prime.get("usage", {}),
                "finish_reason": prime["choices"][0].get("finish_reason"),
                "self_mtp_receipt": prime.get("self_mtp_receipt"),
            }
            _, after_prime = get_json(f"{base_url}/v1/status/self-mtp")
            result["status_after_prime"] = after_prime
            histories = [[] for _ in range(4)]
            measured_started = time.perf_counter()
            for turn_index in range(3):
                result["turns"].append(
                    run_turn_wave(
                        args, base_url, model_id, system_prompt, histories, turn_index
                    )
                )
            result["measured_elapsed_s"] = time.perf_counter() - measured_started
            stop.set()
            monitor.join(timeout=5)
            _, after = get_json(f"{base_url}/v1/status/self-mtp")
            health_status, health = get_json(f"{base_url}/health")
            result["status_after"] = after
            result["health_after"] = {"http_status": health_status, "body": health}
            result["segmented_delta"] = counter_delta(
                before.get("segmented", {}), after.get("segmented", {})
            )
            responses = [r for turn in result["turns"] for r in turn["responses"]]
            result["completion_tokens"] = sum(
                int(r["usage"].get("completion_tokens", 0)) for r in responses
            )
            result["aggregate_completion_tps"] = (
                result["completion_tokens"] / result["measured_elapsed_s"]
            )
            result["latency_ms"] = {
                "median": statistics.median(r["latency_ms"] for r in responses),
                "p95_nearest_rank": sorted(r["latency_ms"] for r in responses)[-1],
                "max": max(r["latency_ms"] for r in responses),
            }
            cached = [
                int(r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))
                for r in responses
            ]
            rss = [s["rss_bytes"] for s in result["status_samples"] if s.get("rss_bytes")]
            result["memory"] = {
                "rss_sample_count": len(rss),
                "rss_min_bytes": min(rss) if rss else None,
                "rss_peak_bytes": max(rss) if rss else None,
                "rss_range_bytes": max(rss) - min(rss) if rss else None,
            }
            configured = after.get("configured", {})
            batch = after.get("batch_decode", {})
            delta = result["segmented_delta"]
            result["checks"] = {
                "all_http_200": all(r["http_status"] == 200 for r in responses),
                "four_by_three": len(responses) == 12,
                "apc_explicitly_enabled": "--prompt-cache-size" in command,
                "apc_20k_primed": after_prime.get("prompt_cache", {}).get("entries", 0) > 0,
                "measured_first_turn_uses_20k_apc": all(
                    value >= args.apc_cached_floor for value in cached[:4]
                ),
                "batch_width_four_observed": batch.get("max_generation_width") == 4,
                "healthy_after": health_status == 200 and health.get("status") == "ok",
                "process_alive_after": process.poll() is None,
                "route_matches_arm": (
                    configured.get("enabled") is True
                    and after.get("engaged_requests", 0) >= 12
                    and delta.get("true_batched_engaged", 0) > 0
                    and delta.get("b1_target_forwards", 0) == 0
                    if arm == "mtp"
                    else configured.get("enabled") is False
                    and after.get("engaged_requests", 0) == 0
                    and delta.get("batched_target_forwards", 0) == 0
                ),
            }
            result["cached_tokens"] = cached
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=5)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        result["server_returncode"] = None if process is None else process.returncode
        result["server_reaped"] = process is not None and process.poll() is not None
        result["host_after"] = host_snapshot()
        result["elapsed_s"] = time.perf_counter() - started
    return result


def compare_arms(plain, mtp):
    plain_rows = [r for turn in plain.get("turns", []) for r in turn["responses"]]
    mtp_rows = [r for turn in mtp.get("turns", []) for r in turn["responses"]]
    exact = [
        left["tokens"] == right["tokens"]
        for left, right in zip(plain_rows, mtp_rows)
    ]
    return {
        "comparable_responses": len(exact),
        "exact_responses": sum(exact),
        "all_token_exact": len(exact) == 12 and all(exact),
        "per_response_exact": exact,
        "mtp_over_plain_aggregate_tps": (
            mtp["aggregate_completion_tps"] / plain["aggregate_completion_tps"]
            if plain.get("aggregate_completion_tps") else None
        ),
    }


def run(args):
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(args.model)
    system_prompt, prefix_tokens = build_system_prompt(tokenizer, args.system_tokens)
    result = {
        "config": {**vars(args), "turn_tokens": list(args.turn_tokens)},
        "system_prompt": {
            "sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "characters": len(system_prompt),
            "apc_common_prefix_tokens": prefix_tokens,
        },
        "arms": [],
    }
    try:
        for arm in ("plain", "mtp"):
            arm_result = run_arm(args, arm, system_prompt, prefix_tokens)
            result["arms"].append(arm_result)
            atomic_write(args.out, result)
            if arm_result.get("error") or not all(arm_result.get("checks", {}).values()):
                raise RuntimeError(f"{arm} arm failed closed")
        result["comparison"] = compare_arms(*result["arms"])
        result["qualified"] = (
            all(all(arm["checks"].values()) and arm["server_reaped"] for arm in result["arms"])
            and result["comparison"]["all_token_exact"]
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["qualified"] = False
    atomic_write(args.out, result)
    print(json.dumps({
        "qualified": result["qualified"],
        "arms": [{"arm": arm["arm"], "checks": arm.get("checks"),
                  "tps": arm.get("aggregate_completion_tps")}
                 for arm in result["arms"]],
        "comparison": result.get("comparison"), "error": result.get("error"),
    }, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8298)
    parser.add_argument("--system-tokens", type=int, default=20_000)
    parser.add_argument("--apc-cached-floor", type=int, default=19_900)
    parser.add_argument("--turn-tokens", type=int, nargs=3, default=(96, 64, 64))
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument(
        "--out", default="results/qwen4-openai-cached20k-b4-realistic-ab.json"
    )
    parser.add_argument(
        "--server-log-dir", default="results/qwen4-openai-cached20k-b4-realistic-ab"
    )
    args = parser.parse_args(argv)
    if args.system_tokens < 1024:
        parser.error("--system-tokens must be at least 1024")
    if args.apc_cached_floor > args.system_tokens:
        parser.error("--apc-cached-floor cannot exceed --system-tokens")
    if any(value < 1 for value in args.turn_tokens):
        parser.error("all --turn-tokens values must be positive")
    if not args.execute:
        print(json.dumps({
            "config": {**vars(args), "turn_tokens": list(args.turn_tokens)},
            "plain_command": server_command(args, "plain"),
            "mtp_command": server_command(args, "mtp"),
        }, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
