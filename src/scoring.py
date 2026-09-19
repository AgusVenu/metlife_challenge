"""Pipeline de scoring batch sobre datos de produccion, con monitoreo.

Flujo
-----
    1. Resolver el modelo: Registry (@production -> @staging) -> mejor run ->
       models/best_model.pkl. El origen elegido queda logueado y se propaga a
       todas las salidas, asi que siempre se sabe QUE modelo produjo cada
       prediccion.
    2. Bajar el baseline_stats.json DEL RUN que produjo ese modelo.
    3. Descubrir los lotes de data/prod/ y, por cada uno:
         leer (parseo robusto) -> validar contrato -> feature engineering ->
         predecir -> invertir log1p -> metricas si hay target -> monitoreo
    4. Persistir predicciones (CSV + Postgres) y el reporte de monitoreo
       (JSON + CSV + TXT + dashboard HTML), y registrar todo en MLflow.

Modos (SCORING_MODE)
--------------------
    prod   : lotes de data/prod/  (default; es lo que pide el challenge)
    sample : comportamiento legacy, N filas aleatorias de training_dataset
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import text

import config
import dashboard
import data_loader
import mlflow_utils
import monitoring
from utils import (count_encoded_features, feature_engineering, get_db_engine,
                   transform_target)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Modelo y baseline
# ============================================================================

def load_model_and_baseline():
    """Resuelve el modelo a usar y el baseline que le corresponde."""
    logger.info("-" * 70)
    logger.info("RESOLUCION DEL MODELO")
    logger.info("-" * 70)

    resolved = mlflow_utils.resolve_model()
    logger.info("Modelo en uso: %s", resolved.describe())

    baseline = mlflow_utils.download_baseline(resolved.run_id)
    if baseline is None:
        # El modelo no vino de MLflow (fallback legacy): se busca el baseline
        # local mas reciente. Se avisa, porque en ese caso no hay garantia de
        # que el baseline corresponda exactamente a ese modelo.
        candidates = sorted(Path(config.RESULTS_DIR).glob("baseline_stats_*.json"))
        if candidates:
            logger.warning(
                "Sin baseline en MLflow; se usa el archivo local %s. "
                "Puede no corresponder al modelo cargado.", candidates[-1].name,
            )
            baseline = monitoring.load_baseline(candidates[-1])
        else:
            raise FileNotFoundError(
                "No hay baseline de monitoreo disponible (ni en MLflow ni en "
                f"{config.RESULTS_DIR}). Ejecutar src/training.py primero."
            )

    logger.info("Metricas de referencia: %s",
                {k: round(v, 4) for k, v in baseline.get("metrics", {}).items()})
    return resolved, baseline


# ============================================================================
# Prediccion
# ============================================================================

def predict(model, raw_features: pd.DataFrame) -> np.ndarray:
    """Aplica el MISMO feature engineering que training y devuelve dolares.

    Se reutiliza `utils.feature_engineering` a proposito: si training y scoring
    derivaran las features por separado, cualquier cambio en una sola de las dos
    introduciria training/serving skew silencioso.
    """
    engineered = feature_engineering(raw_features, is_training=False)
    predictions_log = model.predict(engineered)
    return transform_target(predictions_log, inverse=True)


def build_results_frame(batch_id: str, loaded, predictions: np.ndarray,
                        model_info: Dict[str, Any]) -> pd.DataFrame:
    """Arma el DataFrame de resultados, con o sin ground truth."""
    features = loaded.features
    results = pd.DataFrame({
        "batch_id": batch_id,
        "row_index": np.arange(len(features)),
        "age": features["age"].values,
        "sex": features["sex"].values,
        "bmi": features["bmi"].values,
        "children": features["children"].values,
        "smoker": features["smoker"].values,
        "region": features["region"].values,
        "predicted_charges": predictions,
    })

    # Bug corregido del scoring original: asumia que la columna `charges`
    # siempre existia, asi que reventaba con un lote sin etiquetas como prod3.
    if loaded.has_target:
        actual = loaded.target.values
        results["actual_charges"] = actual
        results["absolute_error"] = np.abs(actual - predictions)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(actual != 0,
                           np.abs((actual - predictions) / actual) * 100, np.nan)
        results["percentage_error"] = pct
    else:
        results["actual_charges"] = np.nan
        results["absolute_error"] = np.nan
        results["percentage_error"] = np.nan

    results["model_name"] = model_info.get("model_name")
    results["model_version"] = model_info.get("model_version")
    results["model_source"] = model_info.get("source")
    results["mlflow_run_id"] = model_info.get("run_id")
    results["scored_at"] = datetime.now()
    return results


# ============================================================================
# Persistencia
# ============================================================================

def save_predictions_csv(results: pd.DataFrame, batch_id: str, timestamp: str) -> Path:
    output_dir = Path(config.PREDICTIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictions_{batch_id}_{timestamp}.csv"
    results.to_csv(path, index=False)
    logger.info("  Predicciones -> %s", path)
    return path


def save_predictions_db(engine, results: pd.DataFrame) -> int:
    columns = [
        "batch_id", "row_index", "age", "sex", "bmi", "children", "smoker", "region",
        "predicted_charges", "actual_charges", "absolute_error", "percentage_error",
        "model_name", "model_version", "model_source", "mlflow_run_id", "scored_at",
    ]
    payload = results[columns].copy()
    payload["actual_charges"] = payload["actual_charges"].astype(object).where(
        payload["actual_charges"].notna(), None)
    payload.to_sql("batch_predictions", engine, if_exists="append", index=False, method="multi",
                   chunksize=500)
    logger.info("  %d filas -> tabla batch_predictions", len(payload))
    return len(payload)


def save_monitoring_db(engine, report: monitoring.BatchReport) -> None:
    row = {
        "batch_id": report.batch_id,
        "scored_at": report.scored_at,
        "n_rows": report.n_rows,
        "has_target": report.has_target,
        "status": report.status,
        "rmse": report.metrics.get("rmse"),
        "mae": report.metrics.get("mae"),
        "r2": report.metrics.get("r2"),
        "mape": report.metrics.get("mape"),
        "baseline_rmse": report.baseline_metrics.get("val_rmse"),
        "baseline_r2": report.baseline_metrics.get("val_r2"),
        "psi_max": report.psi_max,
        "psi_max_feature": report.psi_max_feature,
        "n_violations": len(report.violations),
        "pred_mean": report.prediction_summary.get("mean"),
        "diagnosis": report.diagnosis,
        "model_name": report.model_info.get("model_name"),
        "model_version": report.model_info.get("model_version"),
        "model_source": report.model_info.get("source"),
        "mlflow_run_id": report.model_info.get("run_id"),
        "details": json.dumps(report.to_dict()),
    }
    statement = text("""
        INSERT INTO batch_monitoring (
            batch_id, scored_at, n_rows, has_target, status, rmse, mae, r2, mape,
            baseline_rmse, baseline_r2, psi_max, psi_max_feature, n_violations,
            pred_mean, diagnosis, model_name, model_version, model_source,
            mlflow_run_id, details
        ) VALUES (
            :batch_id, :scored_at, :n_rows, :has_target, :status, :rmse, :mae, :r2, :mape,
            :baseline_rmse, :baseline_r2, :psi_max, :psi_max_feature, :n_violations,
            :pred_mean, :diagnosis, :model_name, :model_version, :model_source,
            :mlflow_run_id, CAST(:details AS JSONB)
        )
    """)
    with engine.connect() as conn:
        conn.execute(statement, row)
        conn.commit()
    logger.info("  Monitoreo -> tabla batch_monitoring (estado %s)", report.status)


# ============================================================================
# Scoring de un lote
# ============================================================================

def score_batch(batch, model, baseline, model_info, engine, timestamp):
    """Puntua y monitorea un lote, dejando un run de MLflow por lote."""
    logger.info("=" * 70)
    logger.info("LOTE: %s", batch.batch_id)
    logger.info("=" * 70)

    loaded = data_loader.load_batch(batch)

    with mlflow.start_run(run_name=f"score_{batch.batch_id}_{timestamp}", nested=True) as run:
        predictions = predict(model, loaded.features)
        logger.info("  Predicciones: %d | media=$%s | rango $%s a $%s",
                    len(predictions), f"{predictions.mean():,.2f}",
                    f"{predictions.min():,.2f}", f"{predictions.max():,.2f}")

        report = monitoring.monitor_batch(
            batch_id=batch.batch_id,
            features=loaded.features,
            predictions=predictions,
            baseline=baseline,
            target=loaded.target if loaded.has_target else None,
            violations=loaded.all_violations,
            model_info={**model_info, "scoring_run_id": run.info.run_id},
            n_features=count_encoded_features(model),
        )

        results = build_results_frame(batch.batch_id, loaded, predictions, model_info)
        csv_path = save_predictions_csv(results, batch.batch_id, timestamp)
        save_predictions_db(engine, results)
        save_monitoring_db(engine, report)

        # --- MLflow ---
        mlflow.log_params({
            "batch_id": batch.batch_id,
            "n_rows": loaded.n_rows,
            "has_target": loaded.has_target,
            "eval_dataset": batch.batch_id,
            "features_file": Path(loaded.features_path).name,
            "target_file": Path(loaded.target_path).name if loaded.target_path else "n/a",
            "model_source": model_info.get("source"),
            "model_version": model_info.get("model_version"),
            "training_run_id": model_info.get("run_id"),
        })

        metrics_to_log = {
            "psi_max": report.psi_max,
            "n_violations": len(report.violations),
            "pred_mean": report.prediction_summary.get("mean", float("nan")),
            "pred_std": report.prediction_summary.get("std", float("nan")),
        }
        metrics_to_log.update({f"psi_{k}": v for k, v in report.psi.items()})
        if report.metrics:
            metrics_to_log.update({
                k: v for k, v in report.metrics.items() if k != "n_samples"
            })
        mlflow.log_metrics({k: float(v) for k, v in metrics_to_log.items()
                            if v is not None and np.isfinite(v)})

        mlflow.set_tags({
            "pipeline_stage": "scoring",
            "batch_id": batch.batch_id,
            "monitoring_status": report.status,
            "has_target": str(loaded.has_target),
        })
        mlflow.log_dict(report.to_dict(), f"monitoring_{batch.batch_id}.json")
        mlflow.log_artifact(str(csv_path), artifact_path="predictions")

        logger.info("  ESTADO: %s", report.status)
        logger.info("  %s", report.diagnosis)

    return report


# ============================================================================
# Modo legacy
# ============================================================================

def run_sample_scoring(model, engine, model_info) -> bool:
    """Comportamiento original: N filas aleatorias de training_dataset."""
    n_samples = config.SCORING_SAMPLE_SIZE
    logger.info("Modo 'sample': %d filas aleatorias de training_dataset", n_samples)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scoring_dataset"))
        conn.execute(text(
            "CREATE TABLE scoring_dataset AS "
            "SELECT * FROM training_dataset ORDER BY RANDOM() LIMIT :n"
        ), {"n": n_samples})
        conn.commit()

    df = pd.read_sql("SELECT * FROM scoring_dataset", engine)
    actual = df[config.TARGET_COLUMN].values
    predictions = predict(model, df[config.RAW_FEATURES])

    results = pd.DataFrame({
        "scoring_id": df["id"].values if "id" in df.columns else np.arange(len(df)),
        **{column: df[column].values for column in config.RAW_FEATURES},
        "actual_charges": actual,
        "predicted_charges": predictions,
        "absolute_error": np.abs(actual - predictions),
        "percentage_error": np.abs((actual - predictions) / actual) * 100,
        "prediction_time": datetime.now(),
    })
    results.to_sql("predictions", engine, if_exists="append", index=False, method="multi")

    metrics = monitoring.regression_metrics(actual, predictions)
    logger.info("Muestras=%d | RMSE=$%s | MAE=$%s | R2=%.4f | MAPE=%.2f%%",
                metrics["n_samples"], f"{metrics['rmse']:,.2f}",
                f"{metrics['mae']:,.2f}", metrics["r2"], metrics["mape"])
    logger.info("Predicciones guardadas en la tabla 'predictions'.")
    return True


# ============================================================================
# Modo produccion
# ============================================================================

def run_prod_scoring(model, baseline, model_info, engine, timestamp) -> List[monitoring.BatchReport]:
    batches = data_loader.discover_batches()
    logger.info("Lotes descubiertos en %s: %s",
                config.PROD_DATA_DIR, ", ".join(str(b) for b in batches))

    reports = []
    for batch in batches:
        try:
            reports.append(score_batch(batch, model, baseline, model_info, engine, timestamp))
        except Exception as exc:
            # Que un lote falle no debe abortar el resto: se registra como ALERT
            # y el pipeline sigue con los demas.
            logger.error("Fallo el scoring del lote %s: %s", batch.batch_id, exc, exc_info=True)
            failed = monitoring.BatchReport(
                batch_id=batch.batch_id, n_rows=0, has_target=batch.has_target,
                status=config.STATUS_ALERT,
                diagnosis=f"El lote no pudo procesarse: {exc}",
                scored_at=datetime.now().isoformat(timespec="seconds"),
                model_info=model_info,
            )
            reports.append(failed)
    return reports


def write_reports(reports, model_info, timestamp) -> Dict[str, Path]:
    """Genera el reporte consolidado en JSON, CSV, TXT y HTML."""
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    consolidated = monitoring.consolidate(reports, model_info)

    paths = {
        "json": results_dir / f"monitoring_report_{timestamp}.json",
        "csv": results_dir / f"monitoring_report_{timestamp}.csv",
        "txt": results_dir / f"monitoring_report_{timestamp}.txt",
        "html": results_dir / f"monitoring_dashboard_{timestamp}.html",
    }

    paths["json"].write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    monitoring.to_dataframe(reports).to_csv(paths["csv"], index=False)
    text_report = monitoring.render_text_report(consolidated)
    paths["txt"].write_text(text_report, encoding="utf-8")
    dashboard.write_dashboard(consolidated, paths["html"])

    return paths, consolidated, text_report


# ============================================================================
# Main
# ============================================================================

def main() -> bool:
    try:
        logger.info("=" * 70)
        logger.info("PIPELINE DE SCORING (modo: %s)", config.SCORING_MODE)
        logger.info("=" * 70)
        logger.info("Configuracion efectiva:\n%s", config.describe())

        config.ensure_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        engine = get_db_engine()

        resolved, baseline = load_model_and_baseline()
        model_info = resolved.to_dict()

        if config.SCORING_MODE == "sample":
            return run_sample_scoring(resolved.model, engine, model_info)

        mlflow_utils.setup_tracking(config.MLFLOW_EXPERIMENT_SCORING)
        with mlflow.start_run(run_name=f"scoring_{timestamp}") as parent:
            mlflow.set_tags({"pipeline_stage": "scoring", "scope": "all_batches"})
            mlflow.log_params({
                "model_source": model_info.get("source"),
                "model_version": model_info.get("model_version"),
                "training_run_id": model_info.get("run_id"),
                "prod_data_dir": str(config.PROD_DATA_DIR),
            })

            reports = run_prod_scoring(resolved.model, baseline, model_info, engine, timestamp)
            paths, consolidated, text_report = write_reports(reports, model_info, timestamp)

            mlflow.log_metrics({
                "n_batches": len(reports),
                "n_ok": consolidated["status_counts"]["OK"],
                "n_warning": consolidated["status_counts"]["WARNING"],
                "n_alert": consolidated["status_counts"]["ALERT"],
            })
            mlflow.set_tag("overall_status", consolidated["overall_status"])
            mlflow.log_dict(consolidated, "monitoring_report.json")
            for path in paths.values():
                mlflow.log_artifact(str(path), artifact_path="monitoring")

        print("\n" + text_report + "\n")

        logger.info("=" * 70)
        logger.info("SCORING COMPLETADO - estado global: %s", consolidated["overall_status"])
        logger.info("=" * 70)
        for name, path in paths.items():
            logger.info("  %-5s -> %s", name, path)
        logger.info("  Predicciones por lote -> %s", config.PREDICTIONS_DIR)
        logger.info("  Tablas: batch_predictions, batch_monitoring")
        logger.info("")
        logger.info("  Ver los runs:  %s", config.mlflow_ui_command())

        if consolidated["overall_status"] == config.STATUS_ALERT and config.FAIL_ON_ALERT:
            logger.error("FAIL_ON_ALERT=true y hay lotes en ALERT: el pipeline termina con error.")
            return False
        return True

    except Exception as exc:
        logger.error("Error en el pipeline de scoring: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
