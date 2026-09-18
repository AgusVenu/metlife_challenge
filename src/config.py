"""Configuracion centralizada del proyecto.

El challenge pide explicitamente "una configuracion simple de experimento
mediante variables de entorno o constantes centralizadas" y "evitar hardcodeos
sensibles". Todo lo configurable del pipeline vive aca y se lee de os.environ
con defaults razonables, de modo que el proyecto corra sin .env pero se pueda
parametrizar por completo desde el entorno (Docker, CI, etc).
"""

import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv es dependencia declarada
    load_dotenv = None


# ============================================================================
# Rutas del proyecto
# ============================================================================

def get_project_root() -> Path:
    """Raiz del repo, independientemente de desde donde se invoque el script.

    Los scripts se ejecutan como `python src/training.py` desde la raiz, pero
    tambien deben funcionar si alguien los corre parado dentro de src/.
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()

# Cargar .env de la raiz si existe (no pisa variables ya presentes en el entorno,
# que es lo que queremos: en Docker manda docker-compose, en local manda .env).
if load_dotenv is not None:
    _dotenv = PROJECT_ROOT / ".env"
    if _dotenv.exists():
        load_dotenv(_dotenv, override=False)


def _resolve(path_str: str) -> Path:
    """Convierte una ruta relativa en absoluta respecto de la raiz del proyecto."""
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


# ============================================================================
# Helpers de lectura de entorno
# ============================================================================

def env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def env_int(key: str, default: int, *, fallback_key: str = None) -> int:
    """Lee un entero del entorno.

    `fallback_key` existe para tolerar el typo historico HIPERPARAM_ITERATIONS
    (ver DECISIONS.md): docker-compose exporta HYPERPARAM_ITERATIONS pero el
    codigo original leia HIPERPARAM_ITERATIONS, asi que la variable nunca tenia
    efecto. Se acepta la correcta primero y la vieja como respaldo.
    """
    raw = os.getenv(key)
    if raw in (None, "") and fallback_key:
        raw = os.getenv(fallback_key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# ============================================================================
# Base de datos
# ============================================================================

DB_HOST = env_str("DB_HOST", "localhost")
DB_PORT = env_str("DB_PORT", "5432")
DB_NAME = env_str("DB_NAME", "metlife_db")
DB_USER = env_str("DB_USER", "metlife_user")
DB_PASSWORD = env_str("DB_PASSWORD", "metlife_pass")


def get_db_url() -> str:
    return (
        f"postgresql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def get_db_url_safe() -> str:
    """Misma URL pero con la password enmascarada, para loguear sin filtrar secretos."""
    return f"postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ============================================================================
# MLflow
# ============================================================================

# ----------------------------------------------------------------------------
# Backend de tracking
# ----------------------------------------------------------------------------
# PostgreSQL, no file://, porque el Model Registry exige un backend con base de
# datos. Se usa una base SEPARADA de la de negocio (`mlflow_db` vs `metlife_db`)
# porque MLflow crea ~15 tablas propias (experiments, runs, metrics, params,
# registered_models, ...) y mezclarlas con `training_dataset` o
# `batch_monitoring` vuelve ilegible el esquema de la aplicacion.
MLFLOW_DB_NAME = env_str("MLFLOW_DB_NAME", "mlflow_db")

# El URI por defecto se DERIVA de las credenciales de la base para no duplicar
# la password en dos variables de entorno distintas. Igual se puede pisar por
# completo con MLFLOW_TRACKING_URI, por ejemplo para apuntar a un tracking
# server remoto (http://...) o volver a sqlite en un entorno sin Postgres.
_DEFAULT_TRACKING_URI = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{MLFLOW_DB_NAME}"
)

_tracking_uri_raw = env_str("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)


def _absolutize_sqlite_uri(uri: str) -> str:
    """Hace absoluta la ruta de un URI sqlite relativo.

    Ya no es el backend por defecto, pero se conserva para que un override
    explicito a `sqlite:///...` siga funcionando: sin esto el archivo se crearia
    en el cwd del proceso y training y scoring podrian terminar apuntando a
    bases distintas segun desde donde se los invoque.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    path_part = uri[len(prefix):]
    if path_part.startswith("/"):
        return uri
    abs_path = _resolve(path_part)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{abs_path}"


def mask_uri(uri: str) -> str:
    """Enmascara la password de un URI de conexion, para poder loguearlo.

    Con backend Postgres el tracking URI contiene credenciales, asi que no
    puede imprimirse tal cual en logs, reportes ni mensajes de consola.
    """
    match = re.match(r"^(?P<scheme>[^:]+://)(?P<user>[^:/@]+):(?P<pwd>[^@]*)@(?P<rest>.*)$", uri)
    if not match:
        return uri
    return f"{match.group('scheme')}{match.group('user')}:***@{match.group('rest')}"


MLFLOW_TRACKING_URI = _absolutize_sqlite_uri(_tracking_uri_raw)
MLFLOW_TRACKING_URI_SAFE = mask_uri(MLFLOW_TRACKING_URI)
MLFLOW_ARTIFACT_ROOT = _resolve(env_str("MLFLOW_ARTIFACT_ROOT", "./mlruns"))

MLFLOW_EXPERIMENT_TRAINING = env_str("MLFLOW_EXPERIMENT_TRAINING", "insurance-charges-training")
MLFLOW_EXPERIMENT_SCORING = env_str("MLFLOW_EXPERIMENT_SCORING", "insurance-charges-scoring")

MLFLOW_MODEL_NAME = env_str("MLFLOW_MODEL_NAME", "insurance-charges-xgb")

# Nombre del artefacto del modelo dentro de cada run.
MLFLOW_MODEL_ARTIFACT = "model"
# Nombre del artefacto con las estadisticas de referencia para monitoreo.
BASELINE_ARTIFACT = "baseline_stats.json"

# MLflow >= 2.9 deprecó los stages (Staging/Production) en favor de aliases.
# Se usan aliases y se mantiene un tag `stage` con el nombre clasico, para que
# el mapeo con el enunciado del challenge sea explicito. Ver DECISIONS.md.
ALIAS_PRODUCTION = "production"
ALIAS_STAGING = "staging"

# ============================================================================
# Criterio de "mejor modelo"
# ============================================================================

MODEL_SELECTION_METRIC = env_str("MODEL_SELECTION_METRIC", "val_rmse")
MODEL_SELECTION_MODE = env_str("MODEL_SELECTION_MODE", "min").lower()

# Gates de promocion (src/promote_model.py)
PROMOTION_MIN_R2 = env_float("PROMOTION_MIN_R2", 0.75)
PROMOTION_MAX_OVERFITTING = env_float("PROMOTION_MAX_OVERFITTING", 0.15)
PROMOTION_MIN_IMPROVEMENT = env_float("PROMOTION_MIN_IMPROVEMENT", 0.0)


# ============================================================================
# Training
# ============================================================================

LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()
HYPERPARAM_ITERATIONS = env_int("HYPERPARAM_ITERATIONS", 50, fallback_key="HIPERPARAM_ITERATIONS")
CV_FOLDS = env_int("CV_FOLDS", 5)
RANDOM_SEED = env_int("RANDOM_SEED", 42)
SPLIT_SEED = env_int("SPLIT_SEED", 43)
TEST_SIZE = env_float("TEST_SIZE", 0.2)

MODELS_DIR = _resolve(env_str("MODELS_DIR", "models"))
RESULTS_DIR = _resolve(env_str("RESULTS_DIR", "results"))
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Ruta legacy, mantenida por compatibilidad con la ejecucion original del proyecto.
LEGACY_MODEL_PATH = MODELS_DIR / "best_model.pkl"


# ============================================================================
# Scoring
# ============================================================================

SCORING_MODE = env_str("SCORING_MODE", "prod").lower()
PROD_DATA_DIR = _resolve(env_str("PROD_DATA_DIR", "data/prod"))
TRAINING_CSV = _resolve(env_str("TRAINING_CSV", "data/dataset.csv"))
SCORING_SAMPLE_SIZE = env_int("SCORING_SAMPLE_SIZE", 10)
FAIL_ON_ALERT = env_bool("FAIL_ON_ALERT", False)


# ============================================================================
# Features
# ============================================================================

RAW_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET_COLUMN = "charges"

CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
NUMERICAL_FEATURES = [
    "age", "bmi", "children",
    "bmi_smoker", "age_smoker",
    "bmi_squared", "age_squared",
    "bmi_obese", "age_senior",
]
DERIVED_FEATURES = [
    "bmi_smoker", "age_smoker",
    "bmi_squared", "age_squared",
    "bmi_obese", "age_senior",
]

# Features sobre las que se calcula drift. Se usan las crudas y no las derivadas
# porque las derivadas son funcion deterministica de estas: si driftea `bmi`,
# driftean `bmi_squared` y `bmi_smoker` por construccion, y reportar las tres
# como hallazgos independientes infla el ruido del reporte.
DRIFT_NUMERICAL_FEATURES = ["age", "bmi", "children"]
DRIFT_CATEGORICAL_FEATURES = ["sex", "smoker", "region"]


# ============================================================================
# Contrato de datos
# ----------------------------------------------------------------------------
# Rangos plausibles de negocio, deliberadamente mas anchos que el rango del
# dataset de training: no queremos alertar por un dato nuevo pero valido, sino
# por datos imposibles. Un BMI de 27929 (data/prod/dataset_prod3_feats) es
# imposible; un BMI de 55 es simplemente nuevo.
# ============================================================================

FEATURE_SPEC = {
    "age": {"kind": "numeric", "min": 18, "max": 100},
    "bmi": {"kind": "numeric", "min": 10.0, "max": 60.0},
    "children": {"kind": "numeric", "min": 0, "max": 10},
    "sex": {"kind": "categorical", "allowed": ["male", "female"]},
    "smoker": {"kind": "categorical", "allowed": ["yes", "no"]},
    "region": {
        "kind": "categorical",
        "allowed": ["northeast", "northwest", "southeast", "southwest"],
    },
}

# Rango plausible del target. El dataset de training va de 1.121 a 63.770 USD;
# se deja un margen amplio hacia arriba para no alertar por un caso caro real.
TARGET_SPEC = {"kind": "numeric", "min": 100.0, "max": 250_000.0}


# ============================================================================
# Umbrales de monitoreo
# ============================================================================

# PSI (Population Stability Index): los cortes 0.10 / 0.25 son el estandar
# de facto en scoring crediticio y riesgo.
PSI_WARN = env_float("PSI_WARN", 0.10)
PSI_ALERT = env_float("PSI_ALERT", 0.25)

# Degradacion de performance respecto de validacion.
PERF_WARN_RATIO = env_float("PERF_WARN_RATIO", 1.25)
PERF_ALERT_RATIO = env_float("PERF_ALERT_RATIO", 1.50)

# Caida absoluta de R2 respecto de validacion.
R2_WARN_DROP = env_float("R2_WARN_DROP", 0.05)
R2_ALERT_DROP = env_float("R2_ALERT_DROP", 0.15)

# Desvio relativo de la media del target: |mean_batch/mean_base - 1|.
TARGET_SHIFT_WARN = env_float("TARGET_SHIFT_WARN", 0.25)
TARGET_SHIFT_ALERT = env_float("TARGET_SHIFT_ALERT", 1.00)

# Porcentaje de filas que violan el contrato de datos.
SCHEMA_WARN_PCT = env_float("SCHEMA_WARN_PCT", 0.0)
SCHEMA_ALERT_PCT = env_float("SCHEMA_ALERT_PCT", 1.0)

# Cantidad de bins usados para construir las distribuciones de referencia.
PSI_BINS = env_int("PSI_BINS", 10)

# Estados posibles del semaforo, de menor a mayor severidad.
STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_ALERT = "ALERT"
STATUS_ORDER = [STATUS_OK, STATUS_WARNING, STATUS_ALERT]


def ensure_dirs() -> None:
    """Crea los directorios de salida. Idempotente."""
    for directory in (MODELS_DIR, RESULTS_DIR, PREDICTIONS_DIR, MLFLOW_ARTIFACT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Resumen de la configuracion efectiva, para loguear al arrancar cada script."""
    return "\n".join([
        f"  Project root:        {PROJECT_ROOT}",
        f"  DB:                  {get_db_url_safe()}",
        f"  MLflow tracking URI: {MLFLOW_TRACKING_URI_SAFE}",
        f"  MLflow artifacts:    {MLFLOW_ARTIFACT_ROOT}",
        f"  Experimento train:   {MLFLOW_EXPERIMENT_TRAINING}",
        f"  Experimento scoring: {MLFLOW_EXPERIMENT_SCORING}",
        f"  Modelo registrado:   {MLFLOW_MODEL_NAME}",
        f"  Criterio de seleccion: {MODEL_SELECTION_METRIC} ({MODEL_SELECTION_MODE})",
    ])


def mlflow_ui_command() -> str:
    """Comando para levantar la UI de MLflow.

    Con backend Postgres el tracking URI lleva la password, asi que no se puede
    imprimir: se apunta al script, que lo resuelve desde este mismo modulo.
    """
    if MLFLOW_TRACKING_URI.startswith("postgresql"):
        return "./scripts/mlflow_ui.sh"
    return f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5000"
