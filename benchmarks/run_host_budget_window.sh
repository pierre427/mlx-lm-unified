#!/bin/zsh
# Claim the shared GPU lock the instant it frees, run one bounded window, release.
# mkdir is the atomic claim: polling with -d and then creating loses the race to
# whoever is also polling, which is exactly what happened at 16:59 today.
set -uo pipefail
LOCK=/Users/Shared/mlxuag/gpu.lock
WORKTREE=/Users/pierrelamy/Desktop/mlx-uag/mlx-lm-unified-worktrees/gdn-inproj-20260902
PY=/Users/pierrelamy/Desktop/mlx-uag/.venv/bin/python
LOG=/Users/pierrelamy/Desktop/mlx-uag/results/qwen4-gdn-onekernel-20260902-hostbudget.log

deadline=$(( $(date +%s) + 7200 ))   # 120 minutes to acquire
while true; do
  if mkdir "$LOCK" 2>/dev/null; then
    print '{"agent":"opus-gdn-inproj","started":"2026-09-02","task":"per-layer megakernel prize + fused GDN in-proj A/B"}' > "$LOCK/owner.json"
    print "ACQUIRED $(date)" | tee -a "$LOG"
    break
  fi
  if [ $(date +%s) -ge $deadline ]; then
    print "GAVE UP waiting for the GPU lock at $(date)" | tee -a "$LOG"
    exit 2
  fi
  sleep 15
done
trap 'rm -f "$LOCK/owner.json"; rmdir "$LOCK" 2>/dev/null; print "RELEASED $(date)" | tee -a "$LOG"' EXIT INT TERM

# Free-memory gate: refuse to load 67 GiB onto a machine that is already full.
free=$(/usr/bin/memory_pressure 2>/dev/null | /usr/bin/awk -F': *' '/free percentage/ {gsub("%","",$2); print $2}')
print "free_percent=${free:-unknown}" | tee -a "$LOG"
if [ -n "${free:-}" ] && [ "$free" -lt 45 ]; then
  print "REFUSING: free memory ${free}% is below the 45% pre-load floor" | tee -a "$LOG"
  exit 3
fi

cd "$WORKTREE"
export PYTHONPATH="$WORKTREE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 20 GPU minutes is the bound; kill rather than overrun a shared resource.
/usr/bin/time -p "$PY" benchmarks/qwen4_host_budget.py 2>&1 | tee -a "$LOG" &
child=$!
( sleep 1500; kill -TERM $child 2>/dev/null ) & watchdog=$!
wait $child; rc=$?
kill $watchdog 2>/dev/null
print "EXIT $rc $(date)" | tee -a "$LOG"
exit $rc
