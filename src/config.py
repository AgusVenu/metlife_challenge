"""Configuracion centralizada del proyecto.

El challenge pide explicitamente "una configuracion simple de experimento
mediante variables de entorno o constantes centralizadas" y "evitar hardcodeos
sensibles". Todo lo configurable del pipeline vive aca y se lee de os.environ
con defaults razonables, de modo que el proyecto corra sin .env pero se pueda
parametrizar por completo desde el entorno (CI, orquestador, etc).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv es dependencia declarada
    load_dotenv = None

logger = logging.getLogger(__name__)


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

# Cargar .env de la raiz si existe. No pisa variables ya presentes en el entorno:
# lo que exporte quien invoca el script (CI, orquestador) tiene prioridad sobre .env.
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
    del codigo original: el entorno del proyecto exportaba
    HYPERPARAM_ITERATIONS pero el codigo leia HIPERPARAM_ITERATIONS, asi que la
    variable nunca tenia efecto. Se acepta la correcta primero y la vieja como
    respaldo.
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


def env_list(key: str, default: str) -> list:
    """Lee una lista separada por comas, ignorando espacios y elementos vacios."""
    raw = os.getenv(key)
    if raw in (None, ""):
        raw = default
    return [item.strip() for item in raw.split(",") if item.strip()]


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
# porque MLflow crea 59 tablas propias (experiments, runs, metrics, params,
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

# Un unico experimento para entrenamiento y scoring. Training y scoring loguean
# el mismo conjunto canonico de metricas (ver monitoring.canonical_metrics), asi
# que sus runs SON comparables entre si: tenerlos juntos permite graficar una
# sola serie `rmse` o `psi_bmi` y ver validacion -> prod1 -> prod2 en un mismo
# grafico, que es el objetivo del monitoreo.
#
# Los runs se distinguen por el tag `pipeline_stage`:
#     training            run principal de entrenamiento: el del modelo GANADOR de la
#                         comparacion entre familias. Es el unico que loguea el modelo
#                         y el baseline, y el unico que `resolve_model()` considera.
#     training_candidate  una familia evaluada en la comparacion (ver src/model_zoo.py).
#                         Loguea solo metricas prefijadas train_*/val_*, nunca las
#                         claves sin prefijo, que son la serie que cruza etapas.
#     training_trial      cada combinacion de la busqueda de hiperparametros
#     scoring             run de scoring (padre y por lote, ver el tag `scope`)
#
# Setear MLFLOW_EXPERIMENT_TRAINING y MLFLOW_EXPERIMENT_SCORING por separado
# vuelve a dividirlos, sin tocar codigo.
MLFLOW_EXPERIMENT = env_str("MLFLOW_EXPERIMENT", "insurance-charges")
MLFLOW_EXPERIMENT_TRAINING = env_str("MLFLOW_EXPERIMENT_TRAINING", MLFLOW_EXPERIMENT)
MLFLOW_EXPERIMENT_SCORING = env_str("MLFLOW_EXPERIMENT_SCORING", MLFLOW_EXPERIMENT)

# Valores del tag que identifica el tipo de run.
STAGE_TRAINING = "training"
STAGE_TRAINING_CANDIDATE = "training_candidate"
STAGE_TRAINING_TRIAL = "training_trial"
STAGE_SCORING = "scoring"

# El nombre designa la TAREA, no el algoritmo: desde que el entrenamiento compara
# varias familias (src/model_zoo.py), un registry llamado "-xgb" sirviendo un
# RandomForest seria enganoso. La familia de cada version queda en el tag
# `model_family`. Para continuidad con las versiones ya registradas bajo el nombre
# viejo, setear MLFLOW_MODEL_NAME=insurance-charges-xgb.
MLFLOW_MODEL_NAME = env_str("MLFLOW_MODEL_NAME", "insurance-charges-regressor")

# No registrar una version nueva si sus metricas son identicas a las de la ultima.
# Sin esto, reentrenar tres veces con la misma semilla y los mismos datos -- que es lo
# que uno hace verificando -- deja tres versiones indistinguibles en el Registry, y
# cada version deja de significar algo. Ver mlflow_utils.register_model_version.
REGISTER_SKIP_DUPLICATES = env_bool("REGISTER_SKIP_DUPLICATES", True)

# Nombre del artefacto del modelo dentro de cada run.
MLFLOW_MODEL_ARTIFACT = "model"
# Nombre del artefacto con las estadisticas de referencia para monitoreo.
BASELINE_ARTIFACT = "baseline_stats.json"

# MLflow >= 2.9 deprecó los stages (Staging/Production) en favor de aliases.
# Se usan aliases y se mantiene un tag `stage` con el nombre clasico, para que
# el mapeo con el enunciado del challenge (que pide Staging/Production) sea explicito.
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

# Presupuesto de busqueda POR FAMILIA, no un total a repartir: cada familia corre
# `round(HYPERPARAM_ITERATIONS * su n_iter_ratio)` iteraciones, topeadas por el
# tamano de su grid (ver src/model_zoo.py). Con los ratios que trae el catalogo
# (1.0 + 0.4 x 3) el default de 50 son 50+20+20+20 = 110 iteraciones en total.
# Es el unico dial: bajarlo recorta las cuatro familias a la vez.
HYPERPARAM_ITERATIONS = env_int("HYPERPARAM_ITERATIONS", 50, fallback_key="HIPERPARAM_ITERATIONS")
CV_FOLDS = env_int("CV_FOLDS", 5)

# Que familias se comparan. El orden de esta lista NO importa: `model_zoo.get_specs`
# devuelve las familias en el orden del catalogo, y ese mismo orden es el que resuelve
# los empates en la seleccion del mejor modelo (gana el incumbente, que va primero en
# MODEL_SPECS). Un nombre desconocido aca hace fallar el entrenamiento a proposito.
# TRAIN_MODEL_FAMILIES=xgboost reproduce exactamente el pipeline previo a la comparacion.
TRAIN_MODEL_FAMILIES = env_list(
    "TRAIN_MODEL_FAMILIES",
    "xgboost,random_forest,hist_gradient_boosting,elasticnet",
)

# Cuantas combinaciones de la busqueda se loguean como runs anidados, por familia.
# El cv_results_*.csv de cada familia ya conserva la busqueda entera; estos runs son
# para hojear en la UI, y con cuatro familias 10 por cabeza no se hojean.
TRIALS_TOP_N = env_int("TRIALS_TOP_N", 5)
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


# ============================================================================
# Reglas de monitoreo por feature y por lote
# ----------------------------------------------------------------------------
# Los umbrales de arriba son el DEFAULT global. Este bloque permite afinarlos para
# una feature puntual o para un lote puntual sin tocar codigo ni multiplicar
# variables de entorno, que es lo que pasaria si cada feature necesitara su propia
# PSI_WARN_BMI / PSI_ALERT_BMI / ...
#
# Precedencia, del mas especifico al mas general:
#
#     regla de (lote, feature)  >  regla de lote  >  regla de feature  >  default
#
# Que "lote" gane sobre "feature" es una decision: una regla de lote es una
# afirmacion deliberada sobre un dataset concreto, y una de feature es un refinamiento
# que vale para todos. Cuando las dos aplican y hay que ser explicito, la forma
# inequivoca de resolverlo es escribir la regla de (lote, feature).
#
# Sin archivo de reglas, `resolve_thresholds()` devuelve siempre el default y el
# comportamiento del monitoreo es identico al de antes de existir este bloque.
# ============================================================================

MONITORING_RULES_FILE = _resolve(env_str("MONITORING_RULES_FILE", "config/monitoring_rules.json"))


@dataclass(frozen=True)
class Thresholds:
    """Umbrales ya resueltos para una senal concreta.

    `source` dice de donde salio cada conjunto ("default", "feature:bmi",
    "batch:prod3", "batch:prod3+feature:bmi"). Sin ese dato, un WARNING emitido con un
    umbral custom es indistinguible de uno emitido con el global, y el reporte deja de
    ser auditable.
    """
    psi_warn: float
    psi_alert: float
    perf_warn_ratio: float
    perf_alert_ratio: float
    r2_warn_drop: float
    r2_alert_drop: float
    target_shift_warn: float
    target_shift_alert: float
    schema_warn_pct: float
    schema_alert_pct: float
    # Resumen de TODAS las reglas que aportaron algo ("feature:bmi+batch:prod3").
    source: str = "default"
    # Procedencia por grupo de umbral. Hace falta porque dos reglas pueden pisar
    # claves de grupos distintos -- una de feature los PSI y una de lote el umbral de
    # esquema -- y entonces `source` nombra a las dos, pero la senal de drift la
    # decidio solo una. Sin esto, el rastro de auditoria de cada senal seria
    # aproximado justo donde promete ser exacto.
    sources: Dict[str, str] = field(default_factory=dict)

    def source_for(self, group: str) -> str:
        """Que regla fijo los umbrales del grupo que decide ESTA senal."""
        return self.sources.get(group, "default")

    def to_dict(self) -> dict:
        return {
            "psi_warn": self.psi_warn, "psi_alert": self.psi_alert,
            "perf_warn_ratio": self.perf_warn_ratio, "perf_alert_ratio": self.perf_alert_ratio,
            "r2_warn_drop": self.r2_warn_drop, "r2_alert_drop": self.r2_alert_drop,
            "target_shift_warn": self.target_shift_warn,
            "target_shift_alert": self.target_shift_alert,
            "schema_warn_pct": self.schema_warn_pct,
            "schema_alert_pct": self.schema_alert_pct,
            "source": self.source,
            "sources": dict(self.sources),
        }

    @property
    def is_custom(self) -> bool:
        return self.source != "default"


DEFAULT_THRESHOLDS = Thresholds(
    psi_warn=PSI_WARN, psi_alert=PSI_ALERT,
    perf_warn_ratio=PERF_WARN_RATIO, perf_alert_ratio=PERF_ALERT_RATIO,
    r2_warn_drop=R2_WARN_DROP, r2_alert_drop=R2_ALERT_DROP,
    target_shift_warn=TARGET_SHIFT_WARN, target_shift_alert=TARGET_SHIFT_ALERT,
    schema_warn_pct=SCHEMA_WARN_PCT, schema_alert_pct=SCHEMA_ALERT_PCT,
)

# Cada senal de monitoreo depende de UN grupo de umbrales. La procedencia se
# registra por grupo para que el reporte pueda decir, de cada senal, exactamente que
# regla fijo el umbral que la decidio.
THRESHOLD_GROUPS = {
    "psi": ("psi_warn", "psi_alert"),
    "perf": ("perf_warn_ratio", "perf_alert_ratio"),
    "r2": ("r2_warn_drop", "r2_alert_drop"),
    "target_shift": ("target_shift_warn", "target_shift_alert"),
    "schema": ("schema_warn_pct", "schema_alert_pct"),
}
_GROUP_OF_FIELD = {f: g for g, fields in THRESHOLD_GROUPS.items() for f in fields}

_THRESHOLD_FIELDS = frozenset(_GROUP_OF_FIELD)

_rules_cache: Optional[Dict[str, Any]] = None


def load_rules(path=None, force: bool = False) -> Dict[str, Any]:
    """Lee el archivo de reglas. Cachea el resultado en memoria.

    Un archivo ausente es el caso NORMAL y devuelve {}. Un archivo presente pero mal
    formado devuelve {} con un WARNING: un JSON roto no debe tumbar un pipeline de
    scoring que por lo demas puede correr perfectamente con los umbrales globales.
    """
    global _rules_cache
    if path is None and _rules_cache is not None and not force:
        return _rules_cache

    target = Path(path or MONITORING_RULES_FILE)
    rules: Dict[str, Any] = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                rules = loaded
            else:
                logger.warning("El archivo de reglas %s no contiene un objeto JSON; se ignora.",
                               target)
        except Exception as exc:
            logger.warning("No se pudo leer el archivo de reglas %s (%s). "
                           "Se usan los umbrales globales.", target, exc)

    if path is None:
        _rules_cache = rules
    return rules


def _overrides(block: Any) -> Dict[str, float]:
    """Extrae de un bloque de reglas solo las claves que son umbrales conocidos."""
    if not isinstance(block, dict):
        return {}
    out = {}
    for key, value in block.items():
        if key in _THRESHOLD_FIELDS:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                logger.warning("Umbral '%s' con valor no numerico (%r); se ignora.", key, value)
    return out


def _block(container: Any, key: str) -> Dict[str, Any]:
    """Saca un sub-bloque de reglas, tolerando que no sea un dict.

    Todo lo que venga de un archivo escrito a mano puede estar mal formado en
    cualquier nivel, no solo en la raiz. Sin esta guarda, un
    `{"batches": {"prod3": 0.05}}` revienta con AttributeError dentro de
    `monitor_batch`; como `run_prod_scoring` atrapa la excepcion por lote, el
    resultado seria que TODOS los lotes quedan en ALERT con "no pudo procesarse",
    sin predicciones monitoreadas ni reconciliacion de alertas. Un typo en un
    archivo de configuracion no debe poner en rojo una corrida entera: debe caer a
    los umbrales globales.
    """
    if not isinstance(container, dict) or not key:
        return {}
    value = container.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning("Bloque de reglas '%s' mal formado (se esperaba un objeto, "
                       "vino %s); se ignora.", key, type(value).__name__)
        return {}
    return value


def resolve_thresholds(batch_id: str = None, feature: str = None,
                       rules: Dict[str, Any] = None) -> Thresholds:
    """Resuelve los umbrales efectivos para un lote y/o una feature."""
    rules = load_rules() if rules is None else rules
    if not isinstance(rules, dict) or not rules:
        return DEFAULT_THRESHOLDS

    thresholds, origin, sources = DEFAULT_THRESHOLDS, [], {}

    def aplicar(block, etiqueta):
        """Aplica un bloque de reglas y anota de que grupo(s) se hizo cargo."""
        nonlocal thresholds
        overrides = _overrides(block)
        if not overrides:
            return
        thresholds = replace(thresholds, **overrides)
        origin.append(etiqueta)
        for key in overrides:
            sources[_GROUP_OF_FIELD[key]] = etiqueta

    # De menos a mas especifico: el ultimo que toca un grupo es el que manda sobre el.
    aplicar(_block(_block(rules, "features"), feature), f"feature:{feature}")

    batch_block = _block(_block(rules, "batches"), batch_id)
    if batch_block:
        aplicar(batch_block, f"batch:{batch_id}")
        aplicar(_block(_block(batch_block, "features"), feature),
                f"batch:{batch_id}/feature:{feature}")

    if not origin:
        return DEFAULT_THRESHOLDS

    # `source` resume TODAS las reglas que aportaron algo; `sources` dice, grupo por
    # grupo, cual fijo el umbral que efectivamente decide cada senal. Reportar solo
    # la ultima regla -- o solo el resumen -- dejaria el rastro de auditoria
    # apuntando a una regla que no aporto el umbral que disparo.
    return replace(thresholds, source="+".join(origin), sources=sources)


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
        f"  Experimento:         {MLFLOW_EXPERIMENT_TRAINING}"
        + ("" if MLFLOW_EXPERIMENT_TRAINING == MLFLOW_EXPERIMENT_SCORING
           else f" (scoring: {MLFLOW_EXPERIMENT_SCORING})"),
        f"  Modelo registrado:   {MLFLOW_MODEL_NAME}",
        f"  Familias a comparar: {', '.join(TRAIN_MODEL_FAMILIES)}",
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
