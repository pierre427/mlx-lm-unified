#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
binary="$script_dir/build/compute_icb_bench"
gpu_lock=/Users/Shared/mlxuag/gpu.lock

if [[ ${MLX_UAG_GPU_LEASE_ACK:-0} != 1 ]]; then
  echo "Refusing GPU work: set MLX_UAG_GPU_LEASE_ACK=1 after claiming the CPG GPU lease." >&2
  exit 2
fi

if [[ ! -d "$gpu_lock" || ! -f "$gpu_lock/owner.json" ]]; then
  echo "Refusing GPU work: $gpu_lock/owner.json is absent." >&2
  exit 2
fi

if [[ -z ${MLX_UAG_GPU_LEASE_AGENT:-} ]]; then
  echo "Refusing GPU work: set MLX_UAG_GPU_LEASE_AGENT to the current CPG lease owner." >&2
  exit 2
fi

owner_agent=$(/usr/bin/plutil -extract agent_id raw -o - "$gpu_lock/owner.json")
lease_expiry=$(/usr/bin/plutil -extract lease_expires_at raw -o - "$gpu_lock/owner.json")
lease_claimed=$(/usr/bin/plutil -extract claimed raw -o - "$gpu_lock/owner.json")
if [[ "$owner_agent" != "$MLX_UAG_GPU_LEASE_AGENT" || "$lease_claimed" != true ]]; then
  echo "Refusing GPU work: the acknowledged agent does not own the active filesystem lease." >&2
  exit 2
fi
current_epoch=$(/bin/date +%s)
if (( lease_expiry <= current_epoch )); then
  echo "Refusing GPU work: the filesystem lease has expired." >&2
  exit 2
fi

if [[ ! -x "$binary" ]]; then
  "$script_dir/build.sh" >/dev/null
fi

exec "$binary" --acknowledge-gpu-use "$@"
