#!/bin/bash
# run_many.sh — Experimento OFAT (one-factor-at-a-time) sobre el AG
#
# Uso:
#   ./run_many.sh [imagen1.png imagen2.png ...]
#   Sin argumentos usa las 4 imágenes por defecto.
#
# Paralelismo (opcional):
#   JOBS=4 ./run_many.sh           # 4 corridas en paralelo
#   JOBS=1 ./run_many.sh           # secuencial (default)
#
#   Cada corrida es un proceso independiente con su propio contexto GPU,
#   así que JOBS>1 funciona bien tanto con renderer=cpu como gpu.
#   Límite práctico: núcleos disponibles (cpu) o VRAM total (gpu).

set -euo pipefail

JOBS=${JOBS:-1}

# ── Imágenes ──────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  IMAGES=("apple.png" "chile.png" "starry.png" "cook.png")
else
  IMAGES=("$@")
fi

# ── Defaults (deben coincidir con config.yaml) ────────────────────────────────
DEFAULT_FITNESS="linear"
DEFAULT_SELECTION="tournament"
DEFAULT_CROSSOVER="uniform"
DEFAULT_MUTATION="uniform_multigen"
DEFAULT_SURVIVAL="additive"

# ── Opciones por eje ──────────────────────────────────────────────────────────
FITNESS_METHODS=("linear" "rmse" "inverse_normalized" "exponential" "inverse_mse" "detail_weighted" "composite")
SELECTION_METHODS=("elite" "tournament" "probabilistic_tournament" "roulette" "universal" "boltzmann" "rank")
CROSSOVER_METHODS=("single_point" "two_point" "uniform" "annular")
MUTATION_METHODS=("single_gene" "limited_multigen" "uniform_multigen" "complete" "error_map_guided")
SURVIVAL_METHODS=("additive" "exclusive")

# ── Totales ───────────────────────────────────────────────────────────────────
N_IMGS=${#IMAGES[@]}
VARIANTS=$((${#FITNESS_METHODS[@]} + ${#SELECTION_METHODS[@]} + ${#CROSSOVER_METHODS[@]} + ${#MUTATION_METHODS[@]} + ${#SURVIVAL_METHODS[@]}))
TOTAL=$((N_IMGS * VARIANTS))

echo "============================================================"
echo "  run_many.sh  (OFAT — one-factor-at-a-time)"
echo "  Imágenes  : ${N_IMGS}  (${IMAGES[*]})"
echo "  Variantes : ${VARIANTS} por imagen"
echo "  Total runs: ${TOTAL}"
echo "  Jobs      : ${JOBS}"
echo "============================================================"

GLOBAL=0
FAILED=0
SKIPPED=0

# Pool de PIDs para modo paralelo
PIDS=()

# Drena todos los jobs pendientes y acumula fallos
drain_all() {
  for pid in "${PIDS[@]:-}"; do
    if ! wait "$pid" 2>/dev/null; then
      ((FAILED++)) || true
    fi
  done
  PIDS=()
}

# Espera hasta que haya un slot libre en el pool
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
  # Resto de args: flags extra para python main.py

  ((GLOBAL++)) || true

  if [ -f "${out_dir}/metrics.csv" ]; then
    echo "[${GLOBAL}/${TOTAL}] SKIP  ${IMG_NAME} | ${label}"
    ((SKIPPED++)) || true
    return
  fi

  echo "[${GLOBAL}/${TOTAL}] RUN   ${IMG_NAME} | ${label}  (active: ${#PIDS[@]}/${JOBS})"

  if [ "$JOBS" -le 1 ]; then
    # ── Secuencial ────────────────────────────────────────────────────────
    if ! uv run main.py \
      --image "input/$IMAGE_PATH" \
      --output "$out_dir" \
      --quiet \
      "$@"; then
      echo "  [ERROR] Falló — continuando."
      ((FAILED++)) || true
    fi
  else
    # ── Paralelo ──────────────────────────────────────────────────────────
    wait_for_slot
    mkdir -p "$out_dir"
    uv run main.py \
      --image "input/$IMAGE_PATH" \
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
  BASE="output/run_many/${IMG_NAME}"

  if [ ! -f "input/$IMAGE_PATH" ]; then
    echo "[WARN] Imagen no encontrada: input/$IMAGE_PATH — saltando."
    ((SKIPPED += VARIANTS)) || true
    continue
  fi

  echo ""
  echo "━━━  Imagen: $IMAGE_PATH  ━━━"

  # ── Fitness ───────────────────────────────────────────────────────────────
  echo "--- fitness ---"
  for VAL in "${FITNESS_METHODS[@]}"; do
    run_once "fitness=${VAL}" "${BASE}/fitness/${VAL}" \
      --fitness "$VAL"
  done

  # ── Selección ─────────────────────────────────────────────────────────────
  echo "--- selection ---"
  for VAL in "${SELECTION_METHODS[@]}"; do
    run_once "selection=${VAL}" "${BASE}/selection/${VAL}" \
      --selection "$VAL"
  done

  # ── Cruza ─────────────────────────────────────────────────────────────────
  echo "--- crossover ---"
  for VAL in "${CROSSOVER_METHODS[@]}"; do
    run_once "crossover=${VAL}" "${BASE}/crossover/${VAL}" \
      --crossover "$VAL"
  done

  # ── Mutación ──────────────────────────────────────────────────────────────
  echo "--- mutation ---"
  for VAL in "${MUTATION_METHODS[@]}"; do
    run_once "mutation=${VAL}" "${BASE}/mutation/${VAL}" \
      --mutation "$VAL"
  done

  # ── Supervivencia ─────────────────────────────────────────────────────────
  echo "--- survival ---"
  for VAL in "${SURVIVAL_METHODS[@]}"; do
    run_once "survival=${VAL}" "${BASE}/survival/${VAL}" \
      --survival "$VAL"
  done

done

# Esperar jobs restantes
drain_all

echo ""
echo "============================================================"
echo "  Corridas completadas."
echo "  OK      : $((GLOBAL - FAILED - SKIPPED))"
echo "  Saltados: ${SKIPPED}  (ya existían)"
echo "  Fallidos: ${FAILED}"
echo "  Total   : ${GLOBAL}"
echo "============================================================"

# ── Análisis con pandas ───────────────────────────────────────────────────────
echo ""
echo "Ejecutando análisis pandas..."
uv run analyze_runs.py "output/run_many"
