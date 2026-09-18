"""Integracion con MLflow: tracking, Model Registry y resolucion del modelo.

Sobre stages vs aliases
-----------------------
El enunciado del challenge pide registrar el modelo "con etapa (Staging/
Production)". MLflow deprecó los stages en 2.9 y los elimino de la API en 3.x;
el reemplazo son los ALIASES, que son punteros con nombre a una version
concreta. Se usan aliases `staging` y `production` y ademas se escribe un tag
`stage` con el nombre clasico, para que la equivalencia con el enunciado quede
explicita tanto en la UI como en la base del registry. Ver DECISIONS.md.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import config

logger = logging.getLogger(__name__)


# ============================================================================
# Setup
# ============================================================================

def setup_tracking(experiment_name: str) -> str:
    """Configura tracking URI y experimento. Devuelve el experiment_id."""
    Path(config.MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = (Path(config.MLFLOW_ARTIFACT_ROOT) / experiment_name).as_uri()
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
        logger.info("Experimento MLflow creado: %s (id=%s)", experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Experimento MLflow: %s (id=%s)", experiment_name, experiment_id)

    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", config.MLFLOW_TRACKING_URI)
    return experiment_id


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient()


# ============================================================================
# Logging de artefactos
# ============================================================================

def log_json_artifact(obj: Dict[str, Any], filename: str) -> None:
    """Sube un dict como artefacto JSON del run activo."""
    mlflow.log_dict(obj, filename)


def log_text_artifact(text: str, filename: str) -> None:
    mlflow.log_text(text, filename)


# ============================================================================
# Model Registry
# ============================================================================

def register_model_version(
    run_id: str,
    metrics: Dict[str, float],
    model_artifact: str = None,
    model_name: str = None,
) -> Optional[Any]:
    """Registra el modelo del run como una nueva version y la deja en `staging`.

    La promocion a `production` NO se hace aca: la decide `promote_model.py`
    comparando metricas contra la version que hoy esta en produccion. Separar
    "registrar" de "promover" es lo que evita que un entrenamiento cualquiera
    pise el modelo que esta sirviendo.
    """
    model_name = model_name or config.MLFLOW_MODEL_NAME
    model_artifact = model_artifact or config.MLFLOW_MODEL_ARTIFACT
    client = get_client()

    try:
        client.create_registered_model(
            model_name,
            description="Predictor de costos de seguro medico (XGBoost + target log1p).",
        )
        logger.info("Modelo registrado creado: %s", model_name)
    except MlflowException:
        pass  # ya existia

    try:
        version = mlflow.register_model(f"runs:/{run_id}/{model_artifact}", model_name)
    except MlflowException as exc:
        logger.error("No se pudo registrar el modelo: %s", exc)
        return None

    client.set_model_version_tag(model_name, version.version, "stage", "Staging")
    client.set_model_version_tag(model_name, version.version, "selection_metric", config.MODEL_SELECTION_METRIC)
    for key, value in metrics.items():
        client.set_model_version_tag(model_name, version.version, f"metric.{key}", f"{value:.6f}")

    client.set_registered_model_alias(model_name, config.ALIAS_STAGING, version.version)

    logger.info("Modelo registrado: %s v%s -> alias '%s'",
                model_name, version.version, config.ALIAS_STAGING)
    return version


def get_version_by_alias(alias: str, model_name: str = None):
    """Devuelve la ModelVersion apuntada por un alias, o None si no existe."""
    model_name = model_name or config.MLFLOW_MODEL_NAME
    try:
        return get_client().get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def get_run_metrics(run_id: str) -> Dict[str, float]:
    try:
        return dict(get_client().get_run(run_id).data.metrics)
    except MlflowException:
        return {}


# ============================================================================
# Resolucion del modelo para scoring
# ============================================================================

@dataclass
class ResolvedModel:
    """El modelo que scoring va a usar, junto con su procedencia."""
    model: Any
    source: str                       # registry:production | registry:staging | run:best | file:legacy
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "val_rmse": self.metrics.get("val_rmse"),
            "val_r2": self.metrics.get("val_r2"),
        }

    def describe(self) -> str:
        return (
            f"origen={self.source} | modelo={self.model_name} v{self.model_version} | "
            f"run={self.run_id} | val_rmse={self.metrics.get('val_rmse')}"
        )


def _load_sklearn_model(uri: str):
    return mlflow.sklearn.load_model(uri)


def resolve_model() -> ResolvedModel:
    """Resuelve que modelo usar, en orden de preferencia explicito.

        1. alias `production` del Model Registry   <- lo normal en produccion
        2. alias `staging`                          <- si todavia no se promovio nada
        3. mejor run del experimento de training segun MODEL_SELECTION_METRIC
        4. models/best_model.pkl                    <- compatibilidad, con WARNING

    Que la cadena sea explicita y quede logueada es el punto: el reporte de
    scoring dice exactamente de donde salio el modelo, en vez de cargar un .pkl
    suelto del disco y esperar que sea el correcto.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # --- 1 y 2: Model Registry por alias ---
    for alias in (config.ALIAS_PRODUCTION, config.ALIAS_STAGING):
        version = get_version_by_alias(alias)
        if version is None:
            continue
        uri = f"models:/{config.MLFLOW_MODEL_NAME}@{alias}"
        try:
            model = _load_sklearn_model(uri)
        except Exception as exc:
            logger.warning("No se pudo cargar %s: %s", uri, exc)
            continue
        logger.info("Modelo resuelto desde el Model Registry: %s (v%s)", uri, version.version)
        return ResolvedModel(
            model=model, source=f"registry:{alias}",
            model_name=config.MLFLOW_MODEL_NAME, model_version=str(version.version),
            run_id=version.run_id, metrics=get_run_metrics(version.run_id),
        )

    # --- 3: mejor run del experimento de training ---
    logger.warning("No hay versiones en el Model Registry; se busca el mejor run de training.")
    order = "ASC" if config.MODEL_SELECTION_MODE == "min" else "DESC"
    try:
        runs = mlflow.search_runs(
            experiment_names=[config.MLFLOW_EXPERIMENT_TRAINING],
            order_by=[f"metrics.{config.MODEL_SELECTION_METRIC} {order}"],
            max_results=1,
            filter_string="attributes.status = 'FINISHED'",
        )
    except Exception as exc:
        logger.warning("No se pudo consultar el experimento de training: %s", exc)
        runs = None

    if runs is not None and len(runs):
        run_id = runs.iloc[0]["run_id"]
        try:
            model = _load_sklearn_model(f"runs:/{run_id}/{config.MLFLOW_MODEL_ARTIFACT}")
            logger.info("Modelo resuelto desde el run %s (mejor %s)",
                        run_id, config.MODEL_SELECTION_METRIC)
            return ResolvedModel(
                model=model, source="run:best", model_name=config.MLFLOW_MODEL_NAME,
                model_version=None, run_id=run_id, metrics=get_run_metrics(run_id),
            )
        except Exception as exc:
            logger.warning("No se pudo cargar el modelo del run %s: %s", run_id, exc)

    # --- 4: fallback legacy ---
    logger.warning(
        "FALLBACK: no hay artefactos en MLflow, se carga %s. "
        "Este modelo no tiene trazabilidad de experimento asociada.",
        config.LEGACY_MODEL_PATH,
    )
    if not Path(config.LEGACY_MODEL_PATH).exists():
        raise FileNotFoundError(
            "No hay modelo disponible: ni en el Model Registry, ni en un run de "
            f"MLflow, ni en {config.LEGACY_MODEL_PATH}. Ejecutar src/training.py primero."
        )

    import joblib
    metrics = {}
    metadata_path = Path(config.MODELS_DIR) / "best_model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metrics = {f"val_{k}": v for k, v in (metadata.get("validation_metrics") or {}).items()}

    return ResolvedModel(
        model=joblib.load(config.LEGACY_MODEL_PATH), source="file:legacy",
        model_name=None, model_version=None, run_id=None, metrics=metrics,
    )


# ============================================================================
# Baseline
# ============================================================================

def download_baseline(run_id: str) -> Optional[Dict[str, Any]]:
    """Baja el baseline_stats.json del run indicado.

    Se baja del run que produjo el modelo en uso, y no de un archivo local, para
    que el baseline contra el que se mide drift corresponda SIEMPRE a ese modelo
    y no al ultimo entrenamiento que alguien haya corrido en la maquina.
    """
    if not run_id:
        return None
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=config.BASELINE_ARTIFACT
        )
        logger.info("Baseline descargado del run %s", run_id)
        return json.loads(Path(local_path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("No se pudo descargar el baseline del run %s: %s", run_id, exc)
        return None
