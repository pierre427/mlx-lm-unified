#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
build_dir="$script_dir/build"
mkdir -p "$build_dir"

xcrun --sdk macosx clang++ \
  -std=c++17 \
  -fobjc-arc \
  -mmacosx-version-min=11.0 \
  -Wall -Wextra -Werror \
  -framework Foundation \
  -framework Metal \
  "$script_dir/compute_icb_bench.mm" \
  -o "$build_dir/compute_icb_bench"

xcrun --sdk macosx clang++ \
  -std=c++17 \
  -fobjc-arc \
  -mmacosx-version-min=15.0 \
  -Wall -Wextra -Werror \
  -Wno-deprecated-declarations \
  -framework Foundation \
  -framework Metal \
  "$script_dir/qwen4_phase_family_icb.mm" \
  -o "$build_dir/qwen4_phase_family_icb"

echo "$build_dir/compute_icb_bench"
echo "$build_dir/qwen4_phase_family_icb"
