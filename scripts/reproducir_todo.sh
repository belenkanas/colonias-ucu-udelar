#!/usr/bin/env bash
# Reentrena y evalúa los experimentos principales del informe, en orden.
# Tiempo total aproximado en una Tesla T4: ~11 h (AGAR ocupa ~9 h).
#   bash scripts/reproducir_todo.sh                 # todos
#   bash scripts/reproducir_todo.sh v8 mlca_v5_base # solo algunos
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTOS=("${@:-v8 mlca_v5_base mlca_v5_mlca agar_base agar_mlca}")
for e in ${EXPERIMENTOS[@]}; do
  echo "=== $e ==="
  python -m colonias.train "configs/experiments/$e.yaml"
  python -m colonias.evaluate "configs/experiments/$e.yaml"
done
python scripts/tabla_resultados.py
