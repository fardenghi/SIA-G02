#!/usr/bin/env bash
# scripts/run_many.sh - OFAT experiment runner for the genetic algorithm.
#
# Usage:
#   ./scripts/run_many.sh [image1.png image2.png ...]
#   JOBS=4 ./scripts/run_many.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JOBS=${JOBS:-1}

if [ $# -eq 0 ]; then
  IMAGES=("apple.png" "chile.png" "starry.png" "cook.png")
else
  IMAGES=("$@")
fi

FITNESS_METHODS=("linear" "rmse" "inverse_normalized" "exponential" "inverse_mse" "detail_weighted" "composite")
SELECTION_METHODS=("elite" "tournament" "probabilistic_tournament" "roulette" "universal" "boltzmann" "rank")
CROSSOVER_METHODS=("single_point" "two_point" "uniform" "annular")
MUTATION_METHODS=("single_gene" "limited_multigen" "uniform_multigen" "complete" "error_map_guided")
SURVIVAL_METHODS=("additive" "exclusive")

BASE_OUT="${REPO_ROOT}/output/run_many"

N_IMGS=${#IMAGES[@]}
VARIANTS=$((${#FITNESS_METHODS[@]} + ${#SELECTION_METHODS[@]} + ${#CROSSOVER_METHODS[@]} + ${#MUTATION_METHODS[@]} + ${#SURVIVAL_METHODS[@]}))
TOTAL=$((N_IMGS * VARIANTS))

echo "============================================================"
echo "  scripts/run_many.sh (OFAT)"
echo "  Images    : ${N_IMGS} (${IMAGES[*]})"
echo "  Variants  : ${VARIANTS} per image"
echo "  Total runs: ${TOTAL}"
echo "  Jobs      : ${JOBS}"
echo "============================================================"

GLOBAL=0
FAILED=0
SKIPPED=0
PIDS=()

drain_all() {
  for pid in "${PIDS[@]:-}"; do
    if ! wait "$pid" 2>/dev/null; then
      ((FAILED++)) || true
    fi
  done
  PIDS=()
}

wait_for_slot() {
  while [ ${#PIDS[@]} -ge "$JOBS" ]; do
    local pid="${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
    if ! wait "$pid" 2>/dev/null; then
      ((FAILED++)) || true
    fi
  done
}

run_once() {
  local label="$1"
  local out_dir="$2"
  shift 2

  ((GLOBAL++)) || true

  if [ -f "${out_dir}/metrics.csv" ]; then
    echo "[${GLOBAL}/${TOTAL}] SKIP ${IMG_NAME} | ${label}"
    ((SKIPPED++)) || true
    return
  fi

  echo "[${GLOBAL}/${TOTAL}] RUN  ${IMG_NAME} | ${label} (active: ${#PIDS[@]}/${JOBS})"

  if [ "$JOBS" -le 1 ]; then
    if ! uv run "${REPO_ROOT}/main.py" \
      --image "${REPO_ROOT}/input/${IMAGE_PATH}" \
      --output "$out_dir" \
      --quiet \
      "$@"; then
      echo "  [ERROR] Run failed, continuing."
      ((FAILED++)) || true
    fi
  else
    wait_for_slot
    mkdir -p "$out_dir"
    uv run "${REPO_ROOT}/main.py" \
      --image "${REPO_ROOT}/input/${IMAGE_PATH}" \
      --output "$out_dir" \
      --quiet \
      "$@" \
      >"${out_dir}/.run.log" 2>&1 &
    PIDS+=($!)
  fi
}

for IMAGE_PATH in "${IMAGES[@]}"; do
  BASENAME=$(basename "$IMAGE_PATH")
  IMG_NAME="${BASENAME%.*}"
  BASE="${BASE_OUT}/${IMG_NAME}"

  if [ ! -f "${REPO_ROOT}/input/${IMAGE_PATH}" ]; then
    echo "[WARN] Missing image: ${REPO_ROOT}/input/${IMAGE_PATH} - skipping."
    ((SKIPPED += VARIANTS)) || true
    continue
  fi

  echo
  echo "--- Image: ${IMAGE_PATH} ---"

  echo "--- fitness ---"
  for VAL in "${FITNESS_METHODS[@]}"; do
    run_once "fitness=${VAL}" "${BASE}/fitness/${VAL}" --fitness "$VAL"
  done

  echo "--- selection ---"
  for VAL in "${SELECTION_METHODS[@]}"; do
    run_once "selection=${VAL}" "${BASE}/selection/${VAL}" --selection "$VAL"
  done

  echo "--- crossover ---"
  for VAL in "${CROSSOVER_METHODS[@]}"; do
    run_once "crossover=${VAL}" "${BASE}/crossover/${VAL}" --crossover "$VAL"
  done

  echo "--- mutation ---"
  for VAL in "${MUTATION_METHODS[@]}"; do
    run_once "mutation=${VAL}" "${BASE}/mutation/${VAL}" --mutation "$VAL"
  done

  echo "--- survival ---"
  for VAL in "${SURVIVAL_METHODS[@]}"; do
    run_once "survival=${VAL}" "${BASE}/survival/${VAL}" --survival "$VAL"
  done
done

drain_all

echo
echo "============================================================"
echo "  Runs completed."
echo "  OK      : $((GLOBAL - FAILED - SKIPPED))"
echo "  Skipped : ${SKIPPED} (already existed)"
echo "  Failed  : ${FAILED}"
echo "  Total   : ${GLOBAL}"
echo "============================================================"

echo
echo "Running pandas analysis..."
uv run "${REPO_ROOT}/scripts/analyze_runs.py" "${BASE_OUT}"
