#!/usr/bin/env bash
# =============================================================================
# Levanta la UI de MLflow contra el backend configurado.
#
# Existe porque con backend Postgres el tracking URI contiene la password, asi
# que no se puede imprimir en un log ni pegar en la terminal. El script lo
# resuelve desde src/config.py (misma fuente que usa el pipeline) y lo pasa a
# mlflow sin que aparezca en pantalla ni en el historial del shell.
#
#   ./scripts/mlflow_ui.sh [puerto]        # por defecto 5000
# =============================================================================

set -euo pipefail

PORT="${1:-5000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Usa el python del venv si existe; si no, el del entorno activo.
PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || command -v python)"

URI="$("$PYTHON" -c "import sys; sys.path.insert(0, '$ROOT/src'); import config; print(config.MLFLOW_TRACKING_URI)")"
SAFE="$("$PYTHON" -c "import sys; sys.path.insert(0, '$ROOT/src'); import config; print(config.MLFLOW_TRACKING_URI_SAFE)")"

echo "Backend: $SAFE"
echo "UI:      http://127.0.0.1:$PORT"
echo

exec "$PYTHON" -m mlflow ui --backend-store-uri "$URI" --host 127.0.0.1 --port "$PORT"
