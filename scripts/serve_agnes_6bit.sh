#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MLX_LM_PYTHON:-/Users/pierrelamy/Desktop/mlx-uag/.venv-mlxmain/bin/python}"
model_path="${AGNES_MODEL:-/Users/pierrelamy/mlx-models/Agnes-3.0-Flash-Preview-MLX-6bit}"
host="${AGNES_HOST:-127.0.0.1}"
port="${AGNES_PORT:-8324}"
reasoning_effort="${AGNES_REASONING_EFFORT:-xhigh}"

# The Agnes template spells its highest tier xhigh. Accept the common OpenAI
# spelling as a launcher convenience while passing the native value through.
if [[ "${reasoning_effort}" == "high" ]]; then
  reasoning_effort="xhigh"
fi
case "${reasoning_effort}" in
  xhigh | medium | low) ;;
  *)
    echo "AGNES_REASONING_EFFORT must be high, xhigh, medium, or low" >&2
    exit 2
    ;;
esac

args=(
  "${python_bin}"
  -m
  mlx_lm.server
  --model
  "${model_path}"
  --single-model
  --host
  "${host}"
  --port
  "${port}"
  --temp
  1.0
  --top-p
  0.95
  --top-k
  20
  --max-tokens
  4096
  --thinking-output-ceiling
  4096
  --chat-template-args
  "{\"enable_thinking\":true,\"reasoning_effort\":\"${reasoning_effort}\"}"
  --thinking-sampling-profile
  '{"temperature":1.0,"top_p":0.95,"top_k":20}'
  --prefill-step-size
  "${AGNES_PREFILL_STEP_SIZE:-2048}"
  --prompt-cache-size
  "${AGNES_PROMPT_CACHE_SIZE:-10}"
)

# Prompt-lookup decoding is a separate, default-off A/B until its Agnes
# acceptance and rate-gate receipts are measured on real weights.
if [[ "${AGNES_ENABLE_PLD:-0}" == "1" ]]; then
  args+=(
    --prompt-lookup-ngram
    "${AGNES_PLD_NGRAM:-3}"
    --prompt-lookup-tokens
    "${AGNES_PLD_TOKENS:-4}"
    --prompt-lookup-adaptive
    --prompt-lookup-warmup
    "${AGNES_PLD_WARMUP:-48}"
    --prompt-lookup-gate
    "${AGNES_PLD_GATE:-0.12}"
    --prompt-lookup-rate-gate
  )
fi

# Enabling this local-only admin route exposes APCv2 hit/miss, cached-token,
# COW materialization, and per-plane segment counters at /v1/admin/config.
if [[ -n "${AGNES_ADMIN_KEY:-}" ]]; then
  args+=(--soft-reload-key "${AGNES_ADMIN_KEY}")
fi

if [[ "${1:-}" == "--check" ]]; then
  printf 'validated:'
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi

# Match the lab's managed Agnes jobs: an executable launch must be the direct
# child of the CPG job process recorded in the filesystem GPU lock. This gate
# runs with the system Python and imports no MLX/model code.
lock_path="/Users/Shared/mlxuag/gpu.lock/owner.json"
/usr/bin/python3 - "${lock_path}" "${PPID}" <<'PY'
import json
import pathlib
import sys

lock = pathlib.Path(sys.argv[1])
try:
    owner = json.loads(lock.read_text())
except (FileNotFoundError, json.JSONDecodeError) as error:
    raise SystemExit(f"Agnes server requires the CPG GPU lock: {error}")
if owner.get("pid") != int(sys.argv[2]) or not owner.get("cpg_generation"):
    raise SystemExit(
        "Run the Agnes server under cpg_job.py with --require-radio --lease --lock"
    )
PY

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${args[@]}"
