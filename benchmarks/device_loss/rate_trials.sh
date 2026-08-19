#!/usr/bin/env bash
# rate_trials.sh <label> <trials> [renders-per-trial]
set -u
export DISPLAY=:1 XAUTHORITY=/run/user/1000/xauth_OlqIBB TMPDIR=/sim/tmp/jl
LABEL=${1:?label}; T=${2:-6}; N=${3:-15}
fail=0
for i in $(seq 1 "$T"); do
  if HUNT_N=$N HUNT_NOPOOL=${HUNT_NOPOOL:-false} HUNT_NOEARLYEXIT=${HUNT_NOEARLYEXIT:-false} timeout 900 julia --project=/sim/Programmieren/VulkanDev --startup-file=no \
       "$(dirname "$0")"/rate_fast.jl 2>&1 | grep -q "SURVIVED"; then
    echo "[$LABEL] trial $i: survived $N renders"
  else
    fail=$((fail + 1)); echo "[$LABEL] trial $i: DEVICE LOST"
  fi
done
echo "[$LABEL] RESULT: $fail / $T trials lost the device ($N renders each)"
