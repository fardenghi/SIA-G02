#!/usr/bin/env bash
# scripts/collect_results.sh - Gather all result.png from run_many outputs.
# Usage: ./scripts/collect_results.sh [base_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE="${1:-${REPO_ROOT}/output/run_many}"
DEST="${BASE}/_results"

mkdir -p "$DEST"

COUNT=0
TOTAL_DIRS=0

# Count expected run folders: {BASE}/{image}/{axis}/{value}
for run_dir in "${BASE}"/*/*/*; do
  [ -d "$run_dir" ] || continue
  value=$(basename "$run_dir")
  axis=$(basename "$(dirname "$run_dir")")
  image=$(basename "$(dirname "$(dirname "$run_dir")")")
  [ "$image" = "_results" ] && continue
  [ "$image" = "_analysis" ] && continue
  TOTAL_DIRS=$((TOTAL_DIRS + 1))

  result="${run_dir}/result.png"
  [ -f "$result" ] || continue

  dest_name="${image}-${axis}-${value}.png"
  cp "$result" "${DEST}/${dest_name}"
  COUNT=$((COUNT + 1))
done

MISSING=$((TOTAL_DIRS - COUNT))

echo "Copied: ${COUNT} results -> ${DEST}/"
echo "Missing (runs without result.png): ${MISSING}"
