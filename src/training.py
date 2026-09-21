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

Comparacion entre familias de modelos
-------------------------------------
Cada ejecucion entrena TODAS las familias declaradas en src/model_zoo.py sobre el
MISMO split, y compara sus metricas. Eso es lo que hace que "el mejor modelo" sea el
resultado de una medicion y no de una preferencia. La jerarquia de runs de MLflow:

    train_<ts>              pipeline_stage=training            <- el GANADOR
    +- family_<nombre>      pipeline_stage=training_candidate   (una por familia)
       +- <fam>_trial_<n>   pipeline_stage=training_trial

Solo el run padre loguea el modelo, el baseline y las metricas SIN prefijo (`rmse`,
`r2`, ...), que son la serie que cruza etapas hasta scoring. Los candidatos loguean
`val_*` y `train_*`: si los cuatro escribieran `rmse`, el grafico de
validacion -> prod1 -> prod2 dejaria de significar lo que significa.

Que haya exactamente un run `training` por ejecucion es tambien lo que mantiene
correcta a `mlflow_utils.resolve_model()`, que busca el mejor run con ese tag.

Criterio de mejor modelo
------------------------
Se optimiza `MODEL_SELECTION_METRIC` (por defecto val_rmse, minimizando), primero
entre familias dentro de la corrida y despues entre corridas. Cada ejecucion registra
una version nueva en el Model Registry con alias `staging`; la promocion a
`production` es una decision aparte que toma src/promote_model.py comparando contra el
modelo que hoy esta sirviendo.
"""

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")           # backend sin display: el pipeline corre headless
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import joblib
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import mlflow_utils
import model_zoo
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

def create_preprocessor(scale_numeric: bool = False):
    """ColumnTransformer: numericas + one-hot para las categoricas.

    `scale_numeric` existe para las familias lineales. Con `passthrough`, las numericas
    conviven en escalas muy distintas (`age_squared` llega a ~10.000 y `children` a 5),
    asi que una penalizacion L1/L2 las castiga desparejo y la familia lineal quedaria
    evaluada de forma deshonesta. Los modelos de arboles son invariantes a la escala, y
    para ellos el default `False` mantiene el pipeline identico al de siempre.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
             config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    logger.info("Preprocesador configurado%s", " (numericas escaladas)" if scale_numeric else "")
    logger.info("  - Features numericas: %s", config.NUMERICAL_FEATURES)
    logger.info("  - Features categoricas: %s", config.CATEGORICAL_FEATURES)
    return preprocessor


def define_hyperparameter_grid(spec: model_zoo.ModelSpec = None):
    """Grid de busqueda de una familia. Sin `spec`, el de XGBoost."""
    spec = spec or model_zoo.MODEL_SPECS["xgboost"]
    logger.info("Hyperparameter grid (%s): %d combinaciones posibles",
                spec.name, spec.grid_size())
    return dict(spec.param_grid)


def build_pipeline(spec: model_zoo.ModelSpec) -> Pipeline:
    """Arma el pipeline de una familia.

    Los nombres de los steps ("preprocessor" y "model") son parte del contrato del
    proyecto: `utils.get_encoded_feature_names`, `utils.count_encoded_features` y
    `compute_feature_importance` los usan por nombre.
    """
    return Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=spec.needs_scaling)),
        ("model", spec.build()),
    ])


def train_model(X_train, y_train, spec: model_zoo.ModelSpec = None):
    """Entrena una familia con RandomizedSearchCV. Devuelve (modelo, search, segundos)."""
    spec = spec or model_zoo.MODEL_SPECS["xgboost"]
    logger.info("Entrenando familia '%s' (%s) con RandomizedSearchCV...", spec.name, spec.label)

    pipeline = build_pipeline(spec)

    # Bug corregido: el codigo original leia HIPERPARAM_ITERATIONS (con typo)
    # mientras el entorno exportaba HYPERPARAM_ITERATIONS, asi que la variable
    # nunca tenia efecto y se entrenaba siempre con el default de 350 iteraciones.
    # Hoy HYPERPARAM_ITERATIONS es el presupuesto GLOBAL y cada familia consume la
    # fraccion que declara en su spec, topeada por el tamano de su grid.
    n_iter, cv_folds = spec.n_iter(), config.CV_FOLDS

    logger.info("Configuracion de busqueda:")
    logger.info("  - Metodo: RandomizedSearchCV")
    logger.info("  - Iteraciones: %d de %d combinaciones posibles", n_iter, spec.grid_size())
    logger.info("  - Cross-validation folds: %d", cv_folds)
    logger.info("  - Scoring: neg_root_mean_squared_error (sobre el target en log)")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=define_hyperparameter_grid(spec),
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        # El paralelismo vive aca y solo aca: los estimadores del zoo se construyen con
        # n_jobs=1 para que los procesos de la CV no compitan entre si por los cores.
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
        verbose=1,
        return_train_score=True,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    logger.info("Busqueda de '%s' completada en %.1fs. Mejores hiperparametros:",
                spec.name, elapsed)
    for param, value in search.best_params_.items():
        logger.info("  - %s: %s", param, value)
    logger.info("Mejor RMSE (CV, escala log): %.4f", -search.best_score_)

    return search.best_estimator_, search, elapsed


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


# Que mide cada tipo de importancia. No son intercambiables: un `gain` y un
# `|coeficiente|` no se comparan entre si, y el grafico no debe sugerir que si.
IMPORTANCE_LABEL = {
    "gain": "Importancia (gain relativo)",
    "abs_coef": "Magnitud del coeficiente |B|",
    "none": "Importancia",
}


def compute_feature_importance(model):
    """Importancia de features con los nombres reales post one-hot.

    Devuelve `(importancias, kind)`. No todas las familias exponen lo mismo:

        feature_importances_  -> "gain"      (XGBoost, RandomForest)
        coef_                 -> "abs_coef"  (ElasticNet)
        ninguno               -> "none"      (HistGradientBoosting)

    Un modelo sin importancia nativa devuelve `({}, "none")` y el resto del pipeline ya
    tolera ese caso: `plot_feature_importance` devuelve None con un dict vacio y
    `log_artifact_safe(None)` devuelve False sin romper.
    """
    estimator = model.named_steps["model"]

    raw, kind = None, "none"
    if hasattr(estimator, "feature_importances_"):
        raw, kind = estimator.feature_importances_, "gain"
    elif hasattr(estimator, "coef_"):
        raw, kind = np.abs(np.ravel(estimator.coef_)), "abs_coef"

    if raw is None:
        logger.info("El estimador %s no expone importancia de features nativa.",
                    type(estimator).__name__)
        return {}, "none"

    names = get_encoded_feature_names(model)
    if not names or len(names) != len(raw):
        names = [f"f{i}" for i in range(len(raw))]
    pairs = sorted(zip(names, (float(v) for v in raw)), key=lambda kv: -kv[1])
    return dict(pairs), kind


def plot_feature_importance(importance: dict, output_path: Path, top_n: int = 15,
                            kind: str = "gain") -> Path:
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
    ax.set_xlabel(IMPORTANCE_LABEL.get(kind, IMPORTANCE_LABEL["none"]),
                  fontsize=9, color=muted)
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
# Comparacion entre familias
# ============================================================================

@dataclass
class Candidate:
    """Una familia ya entrenada y evaluada, lista para competir por el primer puesto."""
    spec: model_zoo.ModelSpec
    model: Pipeline
    search: RandomizedSearchCV
    metrics: dict                      # {"train": ..., "validation": ..., "overfitting_score": ...}
    y_val_pred: np.ndarray
    fit_seconds: float
    run_id: Optional[str] = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def validation(self) -> dict:
        return self.metrics["validation"]

    @property
    def cv_rmse_log(self) -> float:
        return float(-self.search.best_score_)


def selection_value(metrics: dict, metric_name: str = None) -> float:
    """Extrae del dict de metricas el valor por el que se compara a los candidatos.

    Traduce los nombres que usa MLflow (`val_rmse`, `train_r2`, `overfitting_r2_diff`)
    a la estructura anidada que devuelve `evaluate_model`. Falla explicito ante un
    nombre que no mapea: una metrica de seleccion mal escrita en el entorno no debe
    resolverse en silencio a un default, porque elegiria el modelo equivocado sin que
    nadie se entere.
    """
    metric_name = metric_name or config.MODEL_SELECTION_METRIC

    if metric_name == "overfitting_r2_diff":
        return float(metrics["overfitting_score"])

    for prefix, split in (("val_", "validation"), ("train_", "train")):
        if metric_name.startswith(prefix):
            key = metric_name[len(prefix):]
            if key not in metrics[split]:
                raise KeyError(
                    f"MODEL_SELECTION_METRIC='{metric_name}' apunta a la metrica '{key}', "
                    f"que no esta en el conjunto canonico: {sorted(metrics[split])}"
                )
            return float(metrics[split][key])

    raise KeyError(
        f"MODEL_SELECTION_METRIC='{metric_name}' no se puede resolver. Se esperaba un "
        f"nombre con prefijo 'val_' o 'train_', u 'overfitting_r2_diff'."
    )


def select_best_candidate(candidates: List[Candidate]) -> Candidate:
    """Elige el ganador segun MODEL_SELECTION_METRIC / MODEL_SELECTION_MODE.

    Empate: gana el que venga primero en el catalogo, o sea el incumbente. Que el
    desempate sea deterministico y documentado importa para la reproducibilidad.

    Ademas compara el ranking por la metrica de seleccion contra el ranking por CV y
    avisa si discrepan. Elegir entre varias familias por una metrica medida sobre el
    MISMO conjunto de validacion es una comparacion multiple: el ganador tiene algo de
    ventaja por azar. La CV es la estimacion menos sesgada, asi que una discrepancia
    entre ambos rankings es la senal de que el margen no es solido. No se cambia el
    criterio -- `resolve_model()` y `promote_model.py` ordenan por val_rmse y hay que
    ser coherente con ellos -- pero el problema queda a la vista y no escondido.
    """
    if not candidates:
        raise ValueError("No hay candidatos para seleccionar.")

    maximize = config.MODEL_SELECTION_MODE == "max"

    # El desempate usa el orden del CATALOGO y no el de esta lista: si dependiera del
    # orden en que llegaron los candidatos, dos corridas identicas podrian registrar
    # modelos distintos y la reproducibilidad se romperia de la forma mas silenciosa.
    catalog = list(model_zoo.MODEL_SPECS)
    order = {c.name: (catalog.index(c.name) if c.name in catalog else len(catalog))
             for c in candidates}

    def key(candidate: Candidate):
        value = selection_value(candidate.metrics)
        return (-value if maximize else value, order[candidate.name])

    ranked = sorted(candidates, key=key)
    best = ranked[0]

    logger.info("=" * 70)
    logger.info("SELECCION DEL MEJOR MODELO (%s, %s)",
                config.MODEL_SELECTION_METRIC, config.MODEL_SELECTION_MODE)
    logger.info("=" * 70)
    for position, candidate in enumerate(ranked, start=1):
        logger.info("  %d. %-34s %s=%.4f", position, candidate.spec.label,
                    config.MODEL_SELECTION_METRIC, selection_value(candidate.metrics))
    logger.info("  Ganador: %s", best.spec.label)

    by_cv = sorted(candidates, key=lambda c: (c.cv_rmse_log, order[c.name]))
    if len(candidates) > 1 and by_cv[0].name != best.name:
        logger.warning(
            "El ranking por validacion y el ranking por CV no coinciden: gana '%s' por "
            "%s pero '%s' tiene mejor RMSE de CV (%.4f vs %.4f). El margen no es solido; "
            "tomar la diferencia con cautela.",
            best.name, config.MODEL_SELECTION_METRIC, by_cv[0].name,
            by_cv[0].cv_rmse_log, best.cv_rmse_log,
        )

    return best


def build_comparison_frame(candidates: List[Candidate]) -> pd.DataFrame:
    """Vista tabular de la comparacion, para CSV y para el reporte."""
    rows = []
    for candidate in candidates:
        validation = candidate.validation
        rows.append({
            "family": candidate.name,
            "label": candidate.spec.label,
            "n_iter": candidate.spec.n_iter(),
            "grid_size": candidate.spec.grid_size(),
            "fit_seconds": round(candidate.fit_seconds, 2),
            "cv_rmse_log": candidate.cv_rmse_log,
            "val_rmse": validation["rmse"],
            "val_mae": validation["mae"],
            "val_r2": validation["r2"],
            "val_mape": validation["mape"],
            "train_r2": candidate.metrics["train"]["r2"],
            "overfitting_r2_diff": candidate.metrics["overfitting_score"],
            "selection_value": selection_value(candidate.metrics),
            "best_params": json.dumps(
                {k.replace("model__", ""): v for k, v in candidate.search.best_params_.items()},
                default=str,
            ),
        })
    frame = pd.DataFrame(rows)
    ascending = config.MODEL_SELECTION_MODE != "max"
    return frame.sort_values("selection_value", ascending=ascending).reset_index(drop=True)


def format_comparison_table(candidates: List[Candidate], best: Candidate) -> List[str]:
    """Bloque de texto con la comparacion, para el reporte de entrenamiento."""
    frame = build_comparison_frame(candidates)
    lines = [
        f"{'Familia':<24} {'val_RMSE':>12} {'val_R2':>9} {'overfit':>9} "
        f"{'CV_RMSElog':>11} {'seg':>7}",
        "-" * 78,
    ]
    for _, row in frame.iterrows():
        marker = " <-" if row["family"] == best.name else ""
        lines.append(
            f"{row['label'][:24]:<24} ${row['val_rmse']:>11,.2f} {row['val_r2']:>9.4f} "
            f"{row['overfitting_r2_diff']:>9.4f} {row['cv_rmse_log']:>11.4f} "
            f"{row['fit_seconds']:>7.1f}{marker}"
        )
    return lines


def justification_text(candidates: List[Candidate], best: Candidate) -> List[str]:
    """Justificacion de la eleccion, derivada de los numeros de ESTA corrida.

    Reemplaza al parrafo fijo que el proyecto traia argumentando a mano por que se
    habia elegido XGBoost. Con una comparacion real corriendo, ese texto pasaria a ser
    falso en cuanto ganara otra familia: la justificacion tiene que ser un resultado,
    no una opinion escrita de antemano.
    """
    metric = config.MODEL_SELECTION_METRIC
    n = len(candidates)
    if n == 1:
        lines = [
            f"Se entreno 1 sola familia (TRAIN_MODEL_FAMILIES={','.join(config.TRAIN_MODEL_FAMILIES)}): "
            f"no hay comparacion, solo el resultado de {best.spec.label}.",
            "",
            f"{metric}={selection_value(best.metrics):,.4f}.",
        ]
    else:
        lines = [
            f"Se entrenaron {n} familias sobre el mismo split (semilla "
            f"{config.SPLIT_SEED}) y se comparo {metric}.",
            "",
            f"Gana {best.spec.label}: {metric}={selection_value(best.metrics):,.4f}.",
        ]

    others = [c for c in candidates if c.name != best.name]
    if others:
        maximize = config.MODEL_SELECTION_MODE == "max"
        runner_up = sorted(
            others,
            key=lambda c: -selection_value(c.metrics) if maximize else selection_value(c.metrics),
        )[0]
        lines.append(
            f"Segundo: {runner_up.spec.label} con {metric}="
            f"{selection_value(runner_up.metrics):,.4f}."
        )

    linear = next((c for c in candidates if c.name == "elasticnet"), None)
    if linear is not None and linear.name != best.name:
        delta = linear.validation["rmse"] - best.validation["rmse"]
        lines += [
            "",
            f"El piso lineal ({linear.spec.label}) queda en val_rmse="
            f"${linear.validation['rmse']:,.2f}, o sea ${delta:,.2f} peor que el ganador: "
            f"la complejidad no lineal se justifica con una medicion, no por defecto.",
        ]
    elif linear is not None:
        lines += [
            "",
            "Gana el modelo lineal: la complejidad adicional de los ensambles no se "
            "justifica en este dataset.",
        ]

    lines.append("")
    lines.append(f"Motivo por el que {best.spec.label} esta en la comparacion:")
    lines.append(f"  {best.spec.rationale}")
    return lines


# ============================================================================
# Persistencia local (compatibilidad con la ejecucion original del proyecto)
# ============================================================================

def save_model(model, metrics, best_params, timestamp, output_dir=None, spec=None):
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
        # El tipo se lee del modelo: hardcodearlo mentiria en cuanto ganara otra familia.
        "model_type": type(model.named_steps["model"]).__name__,
        "model_family": spec.name if spec is not None else None,
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


def generate_report(metrics, best_params, search_results, timestamp, output_dir=None,
                    candidates=None, best=None):
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
        f"Modelo seleccionado: {(best.spec.label if best is not None else 'n/d')} "
        f"(target transformado con log1p)",
        "",
        "Justificacion",
        "-" * width,
    ]

    # La justificacion sale de los numeros de la corrida, no de un parrafo fijo.
    if candidates and best is not None:
        lines += justification_text(candidates, best)
        lines += ["", "Comparacion entre familias", "-" * width]
        lines += format_comparison_table(candidates, best)
    else:
        lines.append("Corrida de una sola familia: no hay comparacion que reportar.")

    lines += [
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

def log_search_trials(search_results, top_n: int = None, family: str = None):
    """Registra las mejores combinaciones de la busqueda como runs anidados.

    Deja trazada la busqueda entera y no solo al ganador, que es lo que permite
    responder despues "por que este modelo y no otro". El `cv_results_*.csv` de la
    familia guarda la busqueda completa; estos runs son para hojear en la UI, y por eso
    `TRIALS_TOP_N` es mas chico ahora que se corren varias familias por ejecucion.
    """
    top_n = config.TRIALS_TOP_N if top_n is None else top_n
    prefix = f"{family}_" if family else ""
    results_df = pd.DataFrame(search_results.cv_results_).sort_values("rank_test_score")
    for _, row in results_df.head(top_n).iterrows():
        run_name = f"{prefix}trial_rank_{int(row['rank_test_score'])}"
        with mlflow.start_run(nested=True, run_name=run_name):
            mlflow.log_params({k.replace("model__", ""): v for k, v in row["params"].items()})
            mlflow.log_metrics({
                "cv_rmse_log": float(-row["mean_test_score"]),
                "cv_rmse_log_std": float(row["std_test_score"]),
                "cv_train_rmse_log": float(-row["mean_train_score"]),
                "rank": int(row["rank_test_score"]),
            })
            tags = {
                "pipeline_stage": config.STAGE_TRAINING_TRIAL,
                "trial": "hyperparameter_search",
            }
            if family:
                tags["model_family"] = family
            mlflow.set_tags(tags)


def log_candidate_run(candidate: Candidate, timestamp: str, cv_path: Path = None) -> str:
    """Loguea una familia como run anidado `training_candidate`. Devuelve su run_id.

    Deliberadamente NO loguea el modelo ni las metricas sin prefijo:

      - Sin modelo: `resolve_model()` busca runs con `pipeline_stage='training'`, y de
        esos hay exactamente uno por ejecucion (el ganador). Que los candidatos no
        tengan artefacto de modelo hace que sea IMPOSIBLE resolver a uno de ellos por
        accidente, incluso si alguien afloja ese filtro mas adelante.
      - Sin metricas sin prefijo: `rmse`, `mae`, `r2` y `mape` son la serie que cruza
        etapas: scoring loguea esas mismas claves para cada lote, de modo que en la
        UI se grafica una sola serie validacion -> prod1 -> prod2. Si cuatro
        candidatos las escriben, ese grafico deja de significar lo que documenta.

    Lo que si queda es todo lo necesario para reproducir la familia desde la semilla:
    sus mejores hiperparametros y su cv_results completo.
    """
    spec = candidate.spec
    with mlflow.start_run(nested=True, run_name=f"family_{spec.name}") as run:
        mlflow.log_params({
            **{k.replace("model__", ""): v for k, v in candidate.search.best_params_.items()},
            "model_family": spec.name,
            "model_type": type(candidate.model.named_steps["model"]).__name__,
            "scale_numeric": spec.needs_scaling,
            "search_n_iter": spec.n_iter(),
            "search_grid_size": spec.grid_size(),
            "cv_folds": config.CV_FOLDS,
            "random_seed": config.RANDOM_SEED,
            "split_seed": config.SPLIT_SEED,
            "target_transform": "log1p",
        })

        flat = {}
        for split, prefix in (("train", "train_"), ("validation", "val_")):
            for key, value in candidate.metrics[split].items():
                flat[f"{prefix}{key}"] = float(value)
        flat["overfitting_r2_diff"] = float(candidate.metrics["overfitting_score"])
        flat["cv_best_rmse_log"] = candidate.cv_rmse_log
        flat["fit_seconds"] = float(candidate.fit_seconds)
        mlflow.log_metrics({k: v for k, v in flat.items() if np.isfinite(v)})

        mlflow.set_tags({
            "pipeline_stage": config.STAGE_TRAINING_CANDIDATE,
            "model_family": spec.name,
            "selection_metric": config.MODEL_SELECTION_METRIC,
            "rationale": spec.rationale,
        })

        mlflow_utils.log_artifact_safe(cv_path)
        log_search_trials(candidate.search, family=spec.name)
        return run.info.run_id


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
                  reference_metrics: dict = None, spec: model_zoo.ModelSpec = None,
                  candidates: List["Candidate"] = None):
    """Registra parametros, metricas y artefactos del run principal (el del ganador)."""
    best_params = {k.replace("model__", ""): v for k, v in search_results.best_params_.items()}

    mlflow.log_params({
        **best_params,
        # Se lee del modelo en vez de hardcodearse: con la comparacion entre familias,
        # un "XGBRegressor" fijo seria mentira en cuanto ganara otra.
        "model_type": type(model.named_steps["model"]).__name__,
        "model_family": spec.name if spec is not None else "n/a",
        "families_compared": ",".join(c.name for c in candidates) if candidates else "n/a",
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

    # Una metrica por familia en el run GANADOR: con esto la fila del run padre en la
    # tabla de la UI ya muestra la comparacion completa, sin abrir los runs candidatos.
    for candidate in (candidates or []):
        value = candidate.validation.get("rmse")
        if value is not None and np.isfinite(value):
            flat_metrics[f"family_{candidate.name}_val_rmse"] = float(value)

    mlflow.log_metrics(flat_metrics)

    tags = {
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "selection_mode": config.MODEL_SELECTION_MODE,
        "pipeline_stage": config.STAGE_TRAINING,
        "dataset": "training_dataset",
    }
    if spec is not None:
        tags["model_family"] = spec.name
    if candidates:
        tags["families_compared"] = ",".join(c.name for c in candidates)
    mlflow.set_tags(tags)

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
        mlflow_utils.log_artifact_safe(path)

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

            # 4. Entrenamiento: una familia por vez, todas sobre el MISMO split.
            #    Que el split se calcule una sola vez y fuera del bucle es lo que hace
            #    comparables a los candidatos: no dependen de la semilla de ninguno.
            specs = model_zoo.get_specs()
            logger.info("Familias a comparar (%d):\n%s",
                        len(specs), model_zoo.describe_catalog(specs))

            results_dir = Path(config.RESULTS_DIR)
            candidates = []
            cv_paths = {}
            for spec in specs:
                candidate_model, candidate_search, seconds = train_model(X_train, y_train, spec)
                candidate_metrics, candidate_pred = evaluate_model(
                    candidate_model, X_train, y_train, X_val, y_val, y_train_orig, y_val_orig
                )
                candidate = Candidate(
                    spec=spec, model=candidate_model, search=candidate_search,
                    metrics=candidate_metrics, y_val_pred=candidate_pred,
                    fit_seconds=seconds,
                )

                cv_path = results_dir / f"cv_results_{timestamp}_{spec.name}.csv"
                pd.DataFrame(candidate_search.cv_results_).to_csv(cv_path, index=False)
                cv_paths[spec.name] = cv_path

                # Se loguea apenas termina cada familia: si la siguiente explota, las
                # anteriores ya quedaron trazadas en MLflow.
                candidate.run_id = log_candidate_run(candidate, timestamp, cv_path)
                candidates.append(candidate)

            # 5. Seleccion del ganador
            best = select_best_candidate(candidates)
            client = MlflowClient()
            for candidate in candidates:
                if candidate.run_id:
                    client.set_tag(candidate.run_id, "selected",
                                   "true" if candidate is best else "false")

            best_model = best.model
            search = best.search
            metrics = best.metrics
            y_val_pred = best.y_val_pred

            # 6. Artefactos locales: del GANADOR, con los nombres de siempre, para que
            #    la cadena de fallback de scoring siga encontrando lo que espera.
            model_path, metadata_path = save_model(
                best_model, metrics, search.best_params_, timestamp, spec=best.spec
            )
            report_path = generate_report(
                metrics, search.best_params_, search, timestamp,
                candidates=candidates, best=best,
            )

            comparison_path = results_dir / f"model_comparison_{timestamp}.csv"
            build_comparison_frame(candidates).to_csv(comparison_path, index=False)
            logger.info("Comparacion entre familias guardada en: %s", comparison_path)

            cv_path = cv_paths[best.name]

            importance, importance_kind = compute_feature_importance(best_model)
            importance_path, plot_path = None, None
            if importance:
                importance_path = results_dir / f"feature_importance_{timestamp}.json"
                importance_path.write_text(json.dumps(importance, indent=2))
                plot_path = plot_feature_importance(
                    importance, results_dir / f"feature_importance_{timestamp}.png",
                    kind=importance_kind,
                )
            else:
                logger.warning(
                    "La familia ganadora (%s) no expone importancia de features nativa: "
                    "la corrida no genera ese artefacto.", best.name
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
            mlflow.log_param("importance_kind", importance_kind)
            flat_metrics = log_to_mlflow(
                best_model, metrics, search, X_train,
                reference_metrics=reference_metrics,
                spec=best.spec,
                candidates=candidates,
                artifacts={
                    "report": report_path,
                    "metadata": metadata_path,
                    "cv_results": cv_path,
                    "comparison": comparison_path,
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
                family=best.name,
            )

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETADO")
        logger.info("=" * 70)
        logger.info("  MLflow run:        %s", run_id)
        logger.info("  Familia ganadora:  %s (de %d comparadas)",
                    best.spec.label, len(candidates))
        if version is not None and version.run_id == run_id:
            logger.info("  Version registrada: %s v%s (alias '%s')",
                        config.MLFLOW_MODEL_NAME, version.version, config.ALIAS_STAGING)
        elif version is not None:
            # El gate de deduplicacion devolvio una version anterior: este
            # entrenamiento dio exactamente lo mismo que ella.
            logger.info("  Sin version nueva: %s v%s ya tiene estas metricas",
                        config.MLFLOW_MODEL_NAME, version.version)
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
