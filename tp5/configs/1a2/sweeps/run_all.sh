#!/usr/bin/env bash
# Corre todos los configs de ablación OAT de 1a2 y luego genera los gráficos.
# Uso:
#   bash configs/1a2/sweeps/run_all.sh                # todas las dimensiones
#   bash configs/1a2/sweeps/run_all.sh lr activation  # solo algunas dimensiones
#
# Tip: cada config corre 12 restarts sin corte temprano (stop_at=null). Si querés
# que sea más rápido para probar, bajá "restarts" en generate_sweeps.py y regenerá.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> raíz del repo (tp5)

dims=("$@")
if [ ${#dims[@]} -eq 0 ]; then
  dims=(epochs lr optimizer loss activation output_activation init architecture)
fi

for dim in "${dims[@]}"; do
  echo "===== sweep: $dim ====="
  for cfg in configs/1a2/sweeps/"$dim"/*.json; do
    uv run autoencoder --config "$cfg"
  done
done

echo "===== generando gráficos ====="
uv run python configs/1a2/sweeps/plot_sweeps.py
