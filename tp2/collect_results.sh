#!/bin/bash
# collect_results.sh — Reúne todos los result.png de run_many en una sola carpeta.
# Uso: ./collect_results.sh [directorio_base]
#      Por defecto usa output/run_many

BASE="${1:-output/run_many}"
DEST="${BASE}/_results"

mkdir -p "$DEST"

COUNT=0
MISSING=0

# Estructura: {BASE}/{image}/{axis}/{value}/result.png
for result in "${BASE}"/*/*/*/result.png; do
  # Extraer partes del path
  value=$(basename "$(dirname "$result")")
  axis=$(basename "$(dirname "$(dirname "$result")")")
  image=$(basename "$(dirname "$(dirname "$(dirname "$result")")")")

  # Saltar el propio directorio _results si hubiera algo ahí
  [ "$image" = "_results" ] && continue
  [ "$image" = "_analysis" ] && continue

  dest_name="${image}-${axis}-${value}.png"
  cp "$result" "${DEST}/${dest_name}"
  ((COUNT++)) || true
done

echo "Copiados: ${COUNT} resultados → ${DEST}/"
echo "Faltantes (runs sin result.png): $(($(find "${BASE}" -mindepth 3 -maxdepth 3 -type d | grep -v '_results\|_analysis' | wc -l) - COUNT))"
