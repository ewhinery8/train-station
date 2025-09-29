#!/bin/bash
set -euo pipefail

# Run all examples in the train-station crate and report a summary
# Usage:
#   scripts/test-examples.sh [--release] [--nocapture]
#
# Examples are discovered from train-station/Cargo.toml [[example]] entries.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CRATE_DIR="$ROOT_DIR/train-station"
LOG_DIR=""
PROFILE_FLAG="--release"
NOCAPTURE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      PROFILE_FLAG="--release"; shift ;;
    --nocapture)
      NOCAPTURE_FLAG="--"; shift ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

:

echo "===> Discovering examples from Cargo.toml"
mapfile -t EXAMPLE_NAMES < <(awk '
  /^\[\[example\]\]$/ { inex=1; next }
  inex && /^[[:space:]]*name[[:space:]]*=/ {
    gsub(/"/, "", $0);
    sub(/^.*=[[:space:]]*/, "", $0);
    print;
    inex=0
  }
' "$CRATE_DIR/Cargo.toml")

if [[ ${#EXAMPLE_NAMES[@]} -eq 0 ]]; then
  echo "No examples found in $CRATE_DIR/Cargo.toml" >&2
  exit 1
fi

echo "Found ${#EXAMPLE_NAMES[@]} examples: ${EXAMPLE_NAMES[*]}"

echo "===> Building all examples (${PROFILE_FLAG:-debug})"
pushd "$CRATE_DIR" >/dev/null
cargo build $PROFILE_FLAG --examples | cat
popd >/dev/null

PASSED=()
FAILED=()

echo "===> Running examples"
for ex in "${EXAMPLE_NAMES[@]}"; do
  echo "---> Running example: $ex"
  set +e
  # Allow short runs for long RL examples via env overrides
  case "$ex" in
    dqn)
      DQN_STEPS=${DQN_STEPS:-1200} cargo run $PROFILE_FLAG -p train-station --example "$ex" $NOCAPTURE_FLAG | cat ;;
    ppo_continuous)
      PPO_STEPS=${PPO_STEPS:-1500} cargo run $PROFILE_FLAG -p train-station --example "$ex" $NOCAPTURE_FLAG | cat ;;
    ppo_discrete)
      PPOD_STEPS=${PPOD_STEPS:-1500} cargo run $PROFILE_FLAG -p train-station --example "$ex" $NOCAPTURE_FLAG | cat ;;
    *)
      cargo run $PROFILE_FLAG -p train-station --example "$ex" $NOCAPTURE_FLAG | cat ;;
  esac
  status=${PIPESTATUS[0]}
  set -e

  if [[ $status -eq 0 ]]; then
    PASSED+=("$ex")
  else
    FAILED+=("$ex (exit=$status)")
  fi
done

echo "\n===> Summary"
echo "Passed (${#PASSED[@]}): ${PASSED[*]:-}" | sed 's/: $/: none/'
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
else
  echo "All examples passed"
fi


