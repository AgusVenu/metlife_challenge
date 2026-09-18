#!/bin/bash
# =============================================================================
# MetLife ML Ops Challenge - orquestacion del pipeline
#
#   tests -> db_setup -> training -> promote_model -> scoring
#
# El paso de tests corre primero a proposito: valida el parseo de los archivos
# de produccion y la logica de monitoreo antes de gastar minutos entrenando.
# =============================================================================

set -e           # cortar ante el primer error
set -u           # cortar si se usa una variable no definida
set -o pipefail  # que un pipe falle si falla cualquiera de sus comandos

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR" /app/models /app/results /app/results/predictions /app/mlruns /app/mlflow
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"    | tee -a "$LOG_FILE"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"  | tee -a "$LOG_FILE"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"    | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"     | tee -a "$LOG_FILE"; }

echo -e "${BLUE}"
cat << "BANNER"
+==============================================================+
|                                                              |
|      MetLife Insurance Prediction - ML Pipeline v2.0         |
|      MLflow tracking + scoring batch + monitoreo             |
|                                                              |
+==============================================================+
BANNER
echo -e "${NC}"

wait_for_postgres() {
    log_info "Esperando a que PostgreSQL este listo..."
    local max_retries=30 retry_count=0
    until PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            log_error "PostgreSQL no disponible despues de $max_retries intentos"
            exit 1
        fi
        log_warn "Intento $retry_count/$max_retries - PostgreSQL aun no responde"
        sleep 2
    done
    log_success "PostgreSQL esta listo"
}

run_step() {
    local title=$1; shift
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${YELLOW}${title}${NC}"
    echo -e "${BLUE}==========================================${NC}"
    log_info "Iniciando: $title"

    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "$title completado"
        return 0
    fi
    log_error "$title fallo"
    return 1
}

log_info "Configuracion:"
log_info "  DB:                  $DB_USER@$DB_HOST/$DB_NAME"
log_info "  MLflow backend:      postgres://${DB_HOST}/${MLFLOW_DB_NAME:-mlflow_db}"
log_info "  Modelo registrado:   ${MLFLOW_MODEL_NAME:-insurance-charges-xgb}"
log_info "  Iteraciones HP:      ${HYPERPARAM_ITERATIONS:-50}"
log_info "  CV folds:            ${CV_FOLDS:-5}"
log_info "  Modo de scoring:     ${SCORING_MODE:-prod}"

# 1. Tests unitarios (no necesitan base ni MLflow)
if [ "${RUN_TESTS:-true}" = "true" ]; then
    run_step "Tests unitarios" python -m pytest tests/ -q || {
        log_error "Pipeline abortado: fallaron los tests"
        exit 1
    }
else
    log_warn "RUN_TESTS=false: se omiten los tests unitarios"
fi

# 2. Base de datos
wait_for_postgres
run_step "Setup de base de datos" python src/db_setup.py || {
    log_error "Pipeline abortado: fallo db_setup"; exit 1; }

# 3. Entrenamiento con tracking
run_step "Entrenamiento (MLflow)" python src/training.py || {
    log_error "Pipeline abortado: fallo training"; exit 1; }

# 4. Promocion. Que rechace la promocion no es un fallo del pipeline: es una
#    decision valida, y scoring seguira usando el modelo que ya esta en
#    produccion. Por eso este paso no aborta.
run_step "Promocion de modelo" python src/promote_model.py || \
    log_warn "La promocion no se aplico; scoring usara el modelo vigente"

# 5. Scoring sobre produccion + monitoreo
run_step "Scoring y monitoreo" python src/scoring.py || {
    log_error "Pipeline abortado: fallo scoring"; exit 1; }

echo ""
echo -e "${GREEN}"
cat << "BANNER"
+==============================================================+
|                PIPELINE COMPLETADO                           |
+==============================================================+
BANNER
echo -e "${NC}"

log_success "Pipeline completado"
echo ""
echo "Artefactos generados:"
echo "--------------------------------------------"
echo "MODELOS:";      ls -lh /app/models/*.pkl               2>/dev/null || echo "  (ninguno)"
echo "REPORTE DE TRAINING:"; ls -lh /app/results/training_report_*.txt 2>/dev/null || echo "  (ninguno)"
echo "MONITOREO:";    ls -lh /app/results/monitoring_*        2>/dev/null || echo "  (ninguno)"
echo "PREDICCIONES:"; ls -lh /app/results/predictions/*.csv   2>/dev/null || echo "  (ninguna)"
echo "LOGS:";         ls -lh "$LOG_FILE"
echo "--------------------------------------------"
echo ""
echo "Estado del monitoreo por lote:"
grep -E "^(prod|LOTE)" /app/results/monitoring_report_*.txt 2>/dev/null | tail -20 || true
echo ""
echo "Para explorar los experimentos:"
echo "  docker compose --profile ui up -d mlflow_ui   ->   http://localhost:5000"
echo ""

exit 0
