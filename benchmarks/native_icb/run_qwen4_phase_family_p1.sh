#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
binary="$script_dir/build/qwen4_phase_family_icb"
verifier="$script_dir/verify_qwen4_phase_family_p1.py"
python="$repo_root/../.venv/bin/python"
gpu_lock=/Users/Shared/mlxuag/gpu.lock

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 ABSOLUTE_ARTIFACT_DIR ABSOLUTE_NEW_OUTPUT_DIR [REPETITIONS]" >&2
  exit 2
fi

artifact=${1:A}
out_dir=${2:A}
repetitions=${3:-100}

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
current_epoch=$(/bin/date +%s)
if [[ "$owner_agent" != "$MLX_UAG_GPU_LEASE_AGENT" || "$lease_claimed" != true ]]; then
  echo "Refusing GPU work: the acknowledged agent does not own the active filesystem lease." >&2
  exit 2
fi
if (( lease_expiry <= current_epoch )); then
  echo "Refusing GPU work: the filesystem lease has expired." >&2
  exit 2
fi
if [[ ! -d "$artifact" || ! -f "$artifact/MANIFEST.sha256" ]]; then
  echo "Refusing GPU work: artifact or MANIFEST.sha256 is absent." >&2
  exit 2
fi
if [[ -e "$out_dir" ]]; then
  echo "Refusing to overwrite output path: $out_dir" >&2
  exit 2
fi
if [[ ! -x "$binary" ]]; then
  "$script_dir/build.sh" >/dev/null
fi
if [[ ! -x "$python" ]]; then
  echo "Required MLX-UAG interpreter is absent: $python" >&2
  exit 2
fi

manifest_sha256=$(/usr/bin/awk 'NR == 1 { print $1 }' "$artifact/MANIFEST.sha256")
/bin/mkdir "$out_dir"

"$binary" \
  --acknowledge-gpu-use \
  --artifact "$artifact" \
  --manifest-sha256 "$manifest_sha256" \
  --receipt-out "$out_dir/native-receipt.json" \
  --direct-scratch-out "$out_dir/direct-scratch.f32" \
  --icb-scratch-out "$out_dir/icb-scratch.f32" \
  --repetitions "$repetitions"

"$python" "$verifier" \
  --artifact "$artifact" \
  --receipt "$out_dir/native-receipt.json" \
  --direct-scratch "$out_dir/direct-scratch.f32" \
  --icb-scratch "$out_dir/icb-scratch.f32" \
  --out "$out_dir/verification.json"

echo "$out_dir/verification.json"
