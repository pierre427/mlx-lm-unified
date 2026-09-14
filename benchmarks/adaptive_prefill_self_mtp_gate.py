#!/usr/bin/env python3
"""Live APC-residual admission gate for adaptive prefill plus batched self-MTP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def complete(url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=1800) as response:
        result = json.load(response)
    return {"wall_ms": (time.perf_counter() - started) * 1000, "response": result}


def counter_delta(after: dict, before: dict, name: str) -> int:
    return int(after.get(name, 0)) - int(before.get(name, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8282")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    common = "\n".join(
        f"Shared clause {i:04d}: alpha beta gamma delta epsilon zeta eta theta."
        for i in range(260)
    )
    residual = "\n".join(
        f"Residual item {i:04d}: assess latency fairness and exactness."
        for i in range(6)
    )
    chat_url = f"{args.base_url}/v1/chat/completions"
    batch_url = f"{args.base_url}/v1/status/batching"
    mtp_url = f"{args.base_url}/v1/status/self-mtp"

    prime = complete(
        chat_url,
        args.model,
        [
            {"role": "system", "content": common},
            {"role": "user", "content": "Acknowledge the shared reference."},
        ],
        8,
    )
    before_batch = get_json(batch_url)
    before_mtp = get_json(mtp_url)

    prime_text = prime["response"]["choices"][0]["message"]["content"]
    incumbent_messages = [
        {
            "role": "user",
            "content": "Explain admission control in detail and continue for many paragraphs.",
        }
    ]
    residual_messages = [
        {"role": "system", "content": common},
        {"role": "user", "content": "Acknowledge the shared reference."},
        {"role": "assistant", "content": prime_text},
        {
            "role": "user",
            "content": residual + "\nConclude with the exact marker RESIDUAL_OK.",
        },
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        incumbent_future = pool.submit(
            complete, chat_url, args.model, incumbent_messages, 512
        )
        decode_seen = False
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = get_json(batch_url)
            if status.get("gauges", {}).get("active_lanes", 0) > 0:
                decode_seen = True
                break
            if incumbent_future.done():
                break
            time.sleep(0.05)
        residual_result = complete(
            chat_url, args.model, residual_messages, 32
        )
        incumbent_result = incumbent_future.result()

    after_batch = get_json(batch_url)
    after_mtp = get_json(mtp_url)
    before_scheduler = before_batch.get("scheduler", {})
    after_scheduler = after_batch.get("scheduler", {})
    before_apc = before_mtp.get("prompt_cache", {}).get("apc_stats", {}).get(
        "lifetime", {}
    )
    after_apc = after_mtp.get("prompt_cache", {}).get("apc_stats", {}).get(
        "lifetime", {}
    )
    residual_response = residual_result["response"]
    residual_receipt = residual_response.get("self_mtp_receipt") or {}
    cached_tokens = int(
        residual_response.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cached_tokens", 0)
    )
    release_rounds = counter_delta(
        after_scheduler, before_scheduler, "adaptive_prefill_release_rounds"
    )
    before_widths = before_scheduler.get("generation_width_histogram", {})
    after_widths = after_scheduler.get("generation_width_histogram", {})
    width_two_cycles = int(after_widths.get("2", 0)) - int(
        before_widths.get("2", 0)
    )
    apc_hits = counter_delta(after_apc, before_apc, "hits")
    checks = {
        "incumbent_decode_observed": decode_seen,
        "apc_hit_observed": apc_hits > 0 and cached_tokens > 0,
        "adaptive_slices_released": release_rounds > 0,
        "residual_used_self_mtp": residual_receipt.get("route")
        == "continuous_batched_self_mtp",
        "batched_decode_observed": width_two_cycles > 0,
    }
    artifact = {
        "schema": "mlx-uag.adaptive-prefill-self-mtp-gate.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "apc_hits_delta": apc_hits,
            "residual_cached_prompt_tokens": cached_tokens,
            "adaptive_prefill_release_rounds_delta": release_rounds,
            "adaptive_prefill_deadline_forced_rounds_delta": counter_delta(
                after_scheduler,
                before_scheduler,
                "adaptive_prefill_deadline_forced_rounds",
            ),
            "adaptive_prefill_chunk_histogram_before": before_scheduler.get(
                "adaptive_prefill_chunk_histogram", {}
            ),
            "adaptive_prefill_chunk_histogram_after": after_scheduler.get(
                "adaptive_prefill_chunk_histogram", {}
            ),
            "max_generation_width": after_scheduler.get("max_generation_width", 0),
            "generation_width_two_cycles_delta": width_two_cycles,
            "residual_self_mtp_receipt": residual_receipt,
        },
        "requests": {
            "prime": prime,
            "incumbent": incumbent_result,
            "residual": residual_result,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": artifact["passed"], **artifact["observed"]}, indent=2))
    raise SystemExit(0 if artifact["passed"] else 1)


if __name__ == "__main__":
    main()
