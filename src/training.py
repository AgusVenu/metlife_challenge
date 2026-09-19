"""Pipeline de entrenamiento con tracking de experimentos en MLflow.

Que se registra por cada ejecucion
----------------------------------
    Parametros : hiperparametros ganadores, semillas, folds, iteraciones,
                 features usadas, transformacion del target
    Metricas   : RMSE / MAE / R2 / R2 ajustado / MAPE en train y validacion,
                 en escala de dolares y en escala log, mas el score de CV y
                 el gap de overfitting
    Artefactos : modelo serializado con signature, reporte de metricas,
                 metadata, baseline_stats.json, cv_results.csv e importancia
                 de features (JSON + grafico)

Criterio de mejor modelo
------------------------
Se optimiza `MODEL_SELECTION_METRIC` (por defecto val_rmse, minimizando). Cada
corrida registra una version nueva en el Model Registry con alias `staging`; la
promocion a `production` es una decision aparte que toma src/promote_model.py
comparando contra el modelo que hoy esta sirviendo.
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # backend sin display: el pipeline corre headless
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
import mlflow_utils
import monitoring
from utils import (count_encoded_features, feature_engineering, get_db_engine,
                   get_encoded_feature_names, transform_target)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Datos
# ============================================================================

def load_training_data(engine):
    """Carga los datos de entrenamiento desde la base."""
    logger.info("Cargando datos de entrenamiento desde la base de datos...")
    df = pd.read_sql("SELECT * FROM training_dataset", engine)
    logger.info("Datos cargados: %d filas, %d columnas.", df.shape[0], df.shape[1])
    logger.info("Columnas: %s", df.columns.tolist())
    return df


def prepare_features_target(df):
    """Separa features y target, aplicando feature engineering y log1p."""
    df = df.drop([c for c in ("id", "created_at") if c in df.columns], axis=1)

    X = df.drop(config.TARGET_COLUMN, axis=1)
    y = df[config.TARGET_COLUMN]

    logger.info("Features (X): %s", X.columns.tolist())
    logger.info("Target original (y): %s", config.TARGET_COLUMN)
    logger.info("  - Min: $%s", f"{y.min():,.2f}")
    logger.info("  - Max: $%s", f"{y.max():,.2f}")
    logger.info("  - Mean: $%s", f"{y.mean():,.2f}")
    logger.info("  - Median: $%s", f"{y.median():,.2f}")
    logger.info("  - Skewness: %.3f", y.skew())

    X = feature_engineering(X, is_training=True)
    y_original = y.copy()
    y_transformed = transform_target(y, inverse=False)
    logger.info("Skewness tras log1p: %.3f", pd.Series(y_transformed).skew())

    return X, y_transformed, y_original


def split_data(X, y_transformed, y_original, test_size=None, random_state=None):
    """Split train/validation manteniendo alineadas las dos escalas del target."""
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.SPLIT_SEED if random_state is None else random_state

    X_train, X_val, y_train_log, y_val_log, y_train_orig, y_val_orig = train_test_split(
        X, y_transformed, y_original,
        test_size=test_size, random_state=random_state, shuffle=True,
    )

    logger.info("Train set: %d muestras (%.0f%%)", X_train.shape[0], (1 - test_size) * 100)
    logger.info("Validation set: %d muestras (%.0f%%)", X_val.shape[0], test_size * 100)
    logger.info("Target log  - train mean=%.3f std=%.3f | val mean=%.3f std=%.3f",
                y_train_log.mean(), y_train_log.std(), y_val_log.mean(), y_val_log.std())
    logger.info("Target $    - train mean=$%s | val mean=$%s",
                f"{y_train_orig.mean():,.2f}", f"{y_val_orig.mean():,.2f}")

    return X_train, X_val, y_train_log, y_val_log, y_train_orig, y_val_orig


# ============================================================================
# Modelo
# ============================================================================

def create_preprocessor():
    """ColumnTransformer: numericas passthrough + one-hot para las categoricas."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", config.NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
             config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    logger.info("Preprocesador configurado")
    logger.info("  - Features numericas: %s", config.NUMERICAL_FEATURES)
    logger.info("  - Features categoricas: %s", config.CATEGORICAL_FEATURES)
    return preprocessor


def define_hyperparameter_grid():
    """Grid de busqueda para XGBoost."""
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__reg_alpha": [0, 0.1, 1],        # L1
        "model__reg_lambda": [1, 10, 100],      # L2
    }
    total = int(np.prod([len(v) for v in param_distributions.values()]))
    logger.info("Hyperparameter grid: %d combinaciones posibles", total)
    return param_distributions


def train_model(X_train, y_train):
    """Entrena con RandomizedSearchCV sobre el pipeline completo."""
    logger.info("Iniciando entrenamiento con RandomizedSearchCV...")

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )),
    ])

    # Bug corregido: el codigo original leia HIPERPARAM_ITERATIONS (con typo) y
    # docker-compose exporta HYPERPARAM_ITERATIONS, asi que la variable nunca
    # tenia efecto y se entrenaba siempre con el default de 350 iteraciones.
    n_iter, cv_folds = config.HYPERPARAM_ITERATIONS, config.CV_FOLDS

    logger.info("Configuracion de busqueda:")
    logger.info("  - Metodo: RandomizedSearchCV")
    logger.info("  - Iteraciones: %d", n_iter)
    logger.info("  - Cross-validation folds: %d", cv_folds)
    logger.info("  - Scoring: neg_root_mean_squared_error (sobre el target en log)")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=define_hyperparameter_grid(),
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
        verbose=1,
        return_train_score=True,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info("RandomizedSearchCV completado. Mejores hiperparametros:")
    for param, value in search.best_params_.items():
        logger.info("  - %s: %s", param, value)
    logger.info("Mejor RMSE (CV, escala log): %.4f", -search.best_score_)

    return search.best_estimator_, search


# ============================================================================
# Evaluacion
# ============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, y_train_original, y_val_original):
    """Evalua en train y validacion, en escala log y en dolares."""
    logger.info("EVALUACION DEL MODELO")

    y_train_pred_log = model.predict(X_train)
    y_val_pred_log = model.predict(X_val)
    y_train_pred = transform_target(y_train_pred_log, inverse=True)
    y_val_pred = transform_target(y_val_pred_log, inverse=True)

    # Bug corregido: el R2 ajustado usaba la cantidad de columnas ANTES del
    # preprocesador, ignorando las dummies del one-hot. Se usa el ancho real
    # de la matriz que ve el modelo.
    n_features = count_encoded_features(model, fallback=X_train.shape[1])

    def calculate_metrics(y_true_original, y_pred_original, name):
        # Mismo conjunto canonico que usa scoring, para que las claves logueadas
        # en MLflow sean comparables entre validacion y los lotes de produccion.
        metrics = monitoring.canonical_metrics(y_true_original, y_pred_original, n_features)
        logger.info("  %s SET:", name.upper())
        logger.info("    escala LOG  -> RMSE=%.4f  MAE=%.4f  R2=%.4f",
                    metrics["rmse_log"], metrics["mae_log"], metrics["r2_log"])
        logger.info("    escala $    -> RMSE=$%s  MAE=$%s  R2=%.4f  adjR2=%.4f  MAPE=%.2f%%",
                    f"{metrics['rmse']:,.2f}", f"{metrics['mae']:,.2f}",
                    metrics["r2"], metrics["adj_r2"], metrics["mape"])
        return metrics

    train_metrics = calculate_metrics(y_train_original, y_train_pred, "train")
    val_metrics = calculate_metrics(y_val_original, y_val_pred, "validation")

    r2_diff = train_metrics["r2"] - val_metrics["r2"]
    logger.info("ANALISIS DE OVERFITTING: R2(train) - R2(val) = %.4f", r2_diff)
    if r2_diff > 0.15:
        logger.warning("  Overfitting SEVERO")
    elif r2_diff > 0.10:
        logger.warning("  Overfitting moderado")
    elif r2_diff > 0.05:
        logger.info("  Overfitting menor (aceptable)")
    else:
        logger.info("  Sin overfitting significativo")

    metrics = {"train": train_metrics, "validation": val_metrics, "overfitting_score": float(r2_diff)}
    return metrics, y_val_pred


def compute_feature_importance(model):
    """Importancia de features con los nombres reales post one-hot."""
    names = get_encoded_feature_names(model)
    try:
        importances = model.named_steps["model"].feature_importances_
    except Exception:
        return {}
    if not names or len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]
    pairs = sorted(zip(names, (float(v) for v in importances)), key=lambda kv: -kv[1])
    return dict(pairs)


def plot_feature_importance(importance: dict, output_path: Path, top_n: int = 15) -> Path:
    """Grafico de barras horizontales de la importancia de features.

    Una sola serie de magnitudes: un unico tono secuencial, sin leyenda (el
    titulo ya nombra la serie), ejes recesivos y valores directos en las barras.
    """
    items = list(importance.items())[:top_n][::-1]
    if not items:
        return None
    labels, values = zip(*items)

    ink, muted, series = "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.barh(range(len(items)), values, height=0.62, color=series)
    for bar in bars:
        bar.set_linewidth(0)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=9, color=ink)
    ax.set_xlabel("Importancia (gain relativo)", fontsize=9, color=muted)
    ax.set_title(f"Importancia de features - top {len(items)}",
                 fontsize=11, color=ink, loc="left", pad=12)

    span = max(values) if values else 1.0
    for index, value in enumerate(values):
        ax.text(value + span * 0.012, index, f"{value:.3f}",
                va="center", fontsize=8, color=muted)

    ax.set_xlim(0, span * 1.15)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(axis="x", colors=muted, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ============================================================================
# Persistencia local (compatibilidad con la ejecucion original del proyecto)
# ============================================================================

def save_model(model, metrics, best_params, timestamp, output_dir=None):
    """Guarda el modelo y su metadata en models/.

    MLflow ya es la fuente de verdad, pero se mantiene esta salida porque el
    challenge pide conservar compatibilidad con la ejecucion actual y porque es
    el ultimo recurso de la cadena de resolucion de scoring.
    """
    output_dir = Path(output_dir or config.MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"model_{timestamp}.pkl"
    joblib.dump(model, model_path, compress=3)
    logger.info("Modelo guardado en: %s", model_path)

    metadata = {
        "timestamp": timestamp,
        "model_type": "XGBRegressor",
        "target_transform": "log1p",
        "best_params": best_params,
        "train_metrics": metrics["train"],
        "validation_metrics": metrics["validation"],
        "overfitting_score": metrics["overfitting_score"],
    }
    metadata_path = output_dir / f"model_metadata_{timestamp}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata guardada en: %s", metadata_path)

    # Copia "latest" para que scoring pueda resolver un modelo aun sin MLflow.
    shutil.copy2(model_path, output_dir / "best_model.pkl")
    shutil.copy2(metadata_path, output_dir / "best_model_metadata.json")
    logger.info("Copia actualizada: %s", output_dir / "best_model.pkl")

    return model_path, metadata_path


def generate_report(metrics, best_params, search_results, timestamp, output_dir=None):
    """Reporte de evaluacion en texto plano."""
    output_dir = Path(output_dir or config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"training_report_{timestamp}.txt"

    width = 70
    lines = [
        "=" * width,
        "METLIFE INSURANCE COST PREDICTION - TRAINING REPORT",
        "=" * width,
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timestamp: {timestamp}",
        "",
        "Modelo seleccionado: XGBRegressor (target transformado con log1p)",
        "",
        "Justificacion",
        "-" * width,
        "XGBoost fue seleccionado por su performance en regresion tabular, su",
        "capacidad de capturar interacciones no lineales (en particular",
        "smoker x bmi) y su robustez frente a outliers. Su eficiencia permite",
        "una busqueda de hiperparametros amplia en minutos.",
        "",
        "Mejores hiperparametros",
        "-" * width,
    ]
    for param, value in best_params.items():
        lines.append(f"{param.replace('model__', ''):<25}: {value}")

    lines += ["", "Metricas de evaluacion", "-" * width,
              f"{'':<14}{'TRAIN':>16}{'VALIDATION':>16}"]
    for key, label, fmt in [("rmse", "RMSE", "${:,.2f}"), ("mae", "MAE", "${:,.2f}"),
                            ("r2", "R2", "{:.4f}"), ("adj_r2", "Adjusted R2", "{:.4f}"),
                            ("mape", "MAPE", "{:.2f}%")]:
        lines.append(f"{label:<14}{fmt.format(metrics['train'][key]):>16}"
                     f"{fmt.format(metrics['validation'][key]):>16}")

    r2_pct = metrics["validation"]["r2"] * 100
    overfitting = metrics["overfitting_score"]
    lines += [
        "", "Interpretacion", "-" * width,
        f"El modelo explica aproximadamente {r2_pct:.2f}% de la varianza de los",
        "costos en el set de validacion.",
        f"Error absoluto medio (MAE): ${metrics['validation']['mae']:,.2f} por prediccion.",
        f"Error porcentual medio (MAPE): {metrics['validation']['mape']:.2f}%.",
        "",
    ]
    if overfitting > 0.10:
        lines += [f"Se detecta posible overfitting (R2 train - R2 val = {overfitting:.4f}).",
                  "Considerar mas regularizacion o mas datos."]
    else:
        lines += [f"No se detecta overfitting significativo (R2 train - R2 val = {overfitting:.4f}).",
                  "El modelo generaliza bien al set de validacion."]

    lines += ["", "=" * width, "Busqueda de hiperparametros", "-" * width,
              "Metodo: RandomizedSearchCV",
              f"Iteraciones: {search_results.n_iter}",
              f"Cross-validation folds: {search_results.cv}",
              f"Scoring: {search_results.scoring}",
              f"Mejor score (CV RMSE en escala log): {-search_results.best_score_:.4f}",
              "", "Top 5 combinaciones:", "-" * width]

    results_df = pd.DataFrame(search_results.cv_results_).sort_values("rank_test_score")
    for _, row in results_df.head(5).iterrows():
        lines += [f"  Rank {int(row['rank_test_score'])}: RMSE(log)={-row['mean_test_score']:.4f}",
                  f"    {row['params']}"]

    lines += ["", "=" * width]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Reporte generado en: %s", report_path)
    return report_path


# ============================================================================
# MLflow
# ============================================================================

def log_search_trials(search_results, top_n: int = 10):
    """Registra las mejores combinaciones de la busqueda como runs anidados.

    Deja trazada la busqueda entera y no solo al ganador, que es lo que permite
    responder despues "por que este modelo y no otro".
    """
    results_df = pd.DataFrame(search_results.cv_results_).sort_values("rank_test_score")
    for _, row in results_df.head(top_n).iterrows():
        with mlflow.start_run(nested=True, run_name=f"trial_rank_{int(row['rank_test_score'])}"):
            mlflow.log_params({k.replace("model__", ""): v for k, v in row["params"].items()})
            mlflow.log_metrics({
                "cv_rmse_log": float(-row["mean_test_score"]),
                "cv_rmse_log_std": float(row["std_test_score"]),
                "cv_train_rmse_log": float(-row["mean_train_score"]),
                "rank": int(row["rank_test_score"]),
            })
            mlflow.set_tags({
                "pipeline_stage": config.STAGE_TRAINING_TRIAL,
                "trial": "hyperparameter_search",
            })


def reference_monitoring_metrics(baseline, X_val, y_val_pred, X_train_raw):
    """Metricas de monitoreo calculadas sobre validacion, en el run de training.

    Son las MISMAS claves que loguea scoring (`psi_*`, `n_violations`,
    `pred_mean`, `pred_std`), medidas contra el propio baseline. Cumplen dos
    funciones:

    1. Dan el punto de referencia del PSI. Si `psi_age` ya vale 0.08 entre train
       y validacion, el umbral de WARNING (0.10) esta calibrado demasiado fino
       para esa feature, y conviene saberlo antes de alertar en produccion.
    2. Validan los datos de ENTRENAMIENTO contra el mismo contrato que se le
       exige a produccion. Un modelo entrenado sobre datos que violan el
       contrato es un problema que hoy no se detectaba en ningun lado.
    """
    import data_loader

    _, psi_values = monitoring.evaluate_feature_drift(X_val, baseline)
    violations = data_loader.validate_features(X_train_raw)

    reference = {f"psi_{feature}": value for feature, value in psi_values.items()}
    reference["psi_max"] = max(psi_values.values()) if psi_values else 0.0
    reference["n_violations"] = len(violations)
    reference["pred_mean"] = float(np.nanmean(y_val_pred))
    reference["pred_std"] = float(np.nanstd(y_val_pred))

    if violations:
        logger.warning("Los datos de ENTRENAMIENTO violan el contrato de datos:")
        for violation in violations:
            logger.warning("  %s", violation)
    if reference["psi_max"] > config.PSI_WARN:
        logger.warning(
            "PSI entre train y validacion = %.4f (> %.2f): el split no es "
            "representativo y los umbrales de drift quedan mal calibrados.",
            reference["psi_max"], config.PSI_WARN,
        )

    return reference


def log_to_mlflow(model, metrics, search_results, X_train, artifacts: dict,
                  reference_metrics: dict = None):
    """Registra parametros, metricas y artefactos del run principal."""
    best_params = {k.replace("model__", ""): v for k, v in search_results.best_params_.items()}

    mlflow.log_params({
        **best_params,
        "model_type": "XGBRegressor",
        "target_transform": "log1p",
        "search_method": "RandomizedSearchCV",
        "search_n_iter": search_results.n_iter,
        "cv_folds": search_results.cv,
        "cv_scoring": search_results.scoring,
        "random_seed": config.RANDOM_SEED,
        "split_seed": config.SPLIT_SEED,
        "test_size": config.TEST_SIZE,
        "n_train_rows": len(X_train),
        "n_features_input": X_train.shape[1],
        "n_features_encoded": len(get_encoded_feature_names(model)),
        "eval_dataset": "validation",
        "raw_features": ",".join(config.RAW_FEATURES),
        "derived_features": ",".join(config.DERIVED_FEATURES),
    })

    flat_metrics = {}
    for split in ("train", "validation"):
        prefix = "train" if split == "train" else "val"
        for key, value in metrics[split].items():
            if value is not None and np.isfinite(value):
                flat_metrics[f"{prefix}_{key}"] = float(value)

    # Ademas de las prefijadas, se loguean las metricas SIN prefijo referidas al
    # dataset de evaluacion del run (validacion). Scoring usa esas mismas claves
    # para cada lote, asi que en la UI de MLflow se puede graficar una unica
    # serie `rmse` y ver validacion -> prod1 -> prod2 en el mismo grafico.
    for key, value in metrics["validation"].items():
        if key != "n_samples" and value is not None and np.isfinite(value):
            flat_metrics[key] = float(value)

    if reference_metrics:
        flat_metrics.update({k: float(v) for k, v in reference_metrics.items()
                             if v is not None and np.isfinite(v)})

    flat_metrics["cv_best_rmse_log"] = float(-search_results.best_score_)
    flat_metrics["overfitting_r2_diff"] = float(metrics["overfitting_score"])
    mlflow.log_metrics(flat_metrics)

    mlflow.set_tags({
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "selection_mode": config.MODEL_SELECTION_MODE,
        "pipeline_stage": config.STAGE_TRAINING,
        "dataset": "training_dataset",
    })

    # Modelo con signature: deja documentado el esquema de entrada esperado.
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train.head(5)))
    mlflow.sklearn.log_model(
        model,
        name=config.MLFLOW_MODEL_ARTIFACT,
        signature=signature,
        input_example=X_train.head(5),
        serialization_format="cloudpickle",
    )

    for path in artifacts.values():
        if path and Path(path).exists():
            mlflow.log_artifact(str(path))

    return flat_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    try:
        logger.info("=" * 70)
        logger.info("PIPELINE DE ENTRENAMIENTO")
        logger.info("=" * 70)
        logger.info("Configuracion efectiva:\n%s", config.describe())

        config.ensure_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        mlflow_utils.setup_tracking(config.MLFLOW_EXPERIMENT_TRAINING)

        with mlflow.start_run(run_name=f"train_{timestamp}") as run:
            run_id = run.info.run_id
            logger.info("MLflow run iniciado: %s", run_id)

            # 1. Datos
            engine = get_db_engine()
            df = load_training_data(engine)

            # 2. Features y target
            X, y_transformed, y_original = prepare_features_target(df)

            # 3. Split
            X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = split_data(
                X, y_transformed, y_original
            )

            # 4. Entrenamiento
            best_model, search = train_model(X_train, y_train)
            log_search_trials(search)

            # 5. Evaluacion
            metrics, y_val_pred = evaluate_model(
                best_model, X_train, y_train, X_val, y_val, y_train_orig, y_val_orig
            )

            # 6. Artefactos locales
            model_path, metadata_path = save_model(
                best_model, metrics, search.best_params_, timestamp
            )
            report_path = generate_report(metrics, search.best_params_, search, timestamp)

            cv_path = Path(config.RESULTS_DIR) / f"cv_results_{timestamp}.csv"
            pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)

            importance = compute_feature_importance(best_model)
            importance_path = Path(config.RESULTS_DIR) / f"feature_importance_{timestamp}.json"
            importance_path.write_text(json.dumps(importance, indent=2))
            plot_path = plot_feature_importance(
                importance, Path(config.RESULTS_DIR) / f"feature_importance_{timestamp}.png"
            )

            # 7. Baseline de monitoreo: viaja como artefacto DE ESTE run, de modo
            #    que scoring siempre compare contra el baseline del modelo que usa.
            baseline = monitoring.build_baseline(
                X_train=X_train,
                y_train_original=y_train_orig,
                val_metrics={
                    "rmse": metrics["validation"]["rmse"],
                    "mae": metrics["validation"]["mae"],
                    "r2": metrics["validation"]["r2"],
                    "mape": metrics["validation"]["mape"],
                },
                val_predictions=y_val_pred,
                extra={
                    "training_run_id": run_id,
                    "training_timestamp": timestamp,
                    "target_transform": "log1p",
                },
            )
            baseline_path = Path(config.RESULTS_DIR) / f"baseline_stats_{timestamp}.json"
            monitoring.save_baseline(baseline, baseline_path)
            mlflow.log_dict(baseline, config.BASELINE_ARTIFACT)
            logger.info("Baseline de monitoreo registrado como artefacto '%s'",
                        config.BASELINE_ARTIFACT)

            # 8. MLflow
            reference_metrics = reference_monitoring_metrics(
                baseline, X_val, y_val_pred, X_train
            )
            flat_metrics = log_to_mlflow(
                best_model, metrics, search, X_train,
                reference_metrics=reference_metrics,
                artifacts={
                    "report": report_path,
                    "metadata": metadata_path,
                    "cv_results": cv_path,
                    "importance": importance_path,
                    "importance_plot": plot_path,
                },
            )

            # 9. Model Registry
            version = mlflow_utils.register_model_version(
                run_id,
                metrics={
                    "val_rmse": flat_metrics["val_rmse"],
                    "val_r2": flat_metrics["val_r2"],
                    "overfitting_r2_diff": flat_metrics["overfitting_r2_diff"],
                },
            )

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETADO")
        logger.info("=" * 70)
        logger.info("  MLflow run:        %s", run_id)
        if version is not None:
            logger.info("  Version registrada: %s v%s (alias '%s')",
                        config.MLFLOW_MODEL_NAME, version.version, config.ALIAS_STAGING)
        logger.info("  Modelo local:      %s", model_path)
        logger.info("  Reporte:           %s", report_path)
        logger.info("  Validation R2:     %.4f", metrics["validation"]["r2"])
        logger.info("  Validation RMSE:   $%s", f"{metrics['validation']['rmse']:,.2f}")
        logger.info("")
        logger.info("  Ver los runs:  %s", config.mlflow_ui_command())
        return True

    except Exception as exc:
        logger.error("Error en el pipeline de entrenamiento: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
