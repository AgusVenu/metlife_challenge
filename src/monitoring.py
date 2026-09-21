"""Monitoreo de modelos: baseline, drift, performance por lote y semaforo.

Como se monitorea aca
---------------------
El entrenamiento serializa un `baseline_stats.json` con las distribuciones de
referencia y las metricas de validacion, y lo sube como artefacto DEL MISMO RUN
que produjo el modelo. El scoring lo baja desde ese run, de modo que el baseline
contra el que se mide el drift siempre corresponde al modelo que se esta usando,
y no a "el ultimo entrenamiento que alguien haya corrido".

Senales que se evaluan por lote:

    1. Contrato de datos   - rangos y categorias (siempre)
    2. Drift de features   - PSI contra la distribucion de training (siempre)
    3. Drift de prediccion - PSI de y_pred contra y_pred en validacion (siempre)
    4. Performance         - RMSE/MAE/R2/MAPE contra validacion (solo con target)
    5. Desvio del target   - media del batch contra media de training (solo con target)

Las senales 1-3 no necesitan ground truth, y son las unicas disponibles para un
lote como `prod3`, que llega sin etiquetas. El estado final del lote es el PEOR
de todas las senales: un unico ALERT alcanza para marcar el lote.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import config

logger = logging.getLogger(__name__)


# ============================================================================
# Utilidades
# ============================================================================

def _jsonable(obj: Any) -> Any:
    """Convierte tipos de numpy/pandas a tipos nativos serializables a JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def worst_status(statuses) -> str:
    """Devuelve el estado mas severo de una coleccion (OK < WARNING < ALERT)."""
    worst = config.STATUS_OK
    for status in statuses:
        if status is None:
            continue
        if config.STATUS_ORDER.index(status) > config.STATUS_ORDER.index(worst):
            worst = status
    return worst


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Metricas de regresion en escala de dolares.

    MAPE se calcula solo sobre las filas con y_true != 0 para no dividir por cero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"),
                "r2": float("nan"), "mape": float("nan"), "n_samples": 0}

    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any() else float("nan")
    )

    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mape": mape,
        "n_samples": int(len(y_true)),
    }


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """R2 ajustado por la cantidad de features codificadas."""
    if not np.isfinite(r2) or not n_features or n_samples - n_features - 1 <= 0:
        return float("nan")
    return float(1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))


def canonical_metrics(y_true, y_pred, n_features: int = None) -> Dict[str, float]:
    """Conjunto canonico de metricas de regresion.

    Es la MISMA funcion en training y en scoring, para que las claves que se
    loguean en MLflow sean comparables entre un run de entrenamiento y un lote
    de produccion. Si cada pipeline definiera su propio set, comparar la
    performance de validacion contra la de un lote seria imposible en la UI.

    Devuelve las metricas en escala de dolares y en escala logaritmica. Las de
    escala log se derivan de los valores en dolares con log1p, que da
    exactamente los mismos numeros que evaluarlas en el espacio en el que
    entrena el modelo, porque log1p(expm1(x)) == x.
    """
    base = regression_metrics(y_true, y_pred)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    true_log = np.log1p(np.clip(y_true[mask], 0, None))
    pred_log = np.log1p(np.clip(y_pred[mask], 0, None))

    metrics = dict(base)
    if len(true_log):
        log_base = regression_metrics(true_log, pred_log)
        metrics.update({
            "rmse_log": log_base["rmse"],
            "mae_log": log_base["mae"],
            "r2_log": log_base["r2"],
            "mape_log": log_base["mape"],
        })
    metrics["adj_r2"] = adjusted_r2(base["r2"], base["n_samples"], n_features)
    return metrics


# ============================================================================
# PSI - Population Stability Index
# ============================================================================

def bin_counts(values, interior_edges) -> np.ndarray:
    """Cuenta valores por bin usando SOLO los bordes interiores.

    Con k bordes interiores quedan k+1 bins, y los dos extremos son abiertos:

        (-inf, e0)  [e0, e1)  ...  [e_{k-1}, +inf)

    Que los extremos sean abiertos es deliberado: un valor fuera del rango visto
    en training (un bmi de 27929, por ejemplo) tiene que caer en el bin extremo
    y empujar el PSI, no quedar afuera del histograma y pasar desapercibido.
    """
    interior = np.asarray(interior_edges, dtype=float)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_bins = len(interior) + 1
    if len(values) == 0:
        return np.zeros(n_bins, dtype=int)
    return np.bincount(np.digitize(values, interior), minlength=n_bins)[:n_bins]


def make_bins(values, n_bins: int = None) -> Dict[str, list]:
    """Construye los bordes por cuantiles y las frecuencias de referencia.

    Se guardan unicamente los bordes INTERIORES y no +/-inf, porque el baseline
    se serializa a JSON y los infinitos no sobreviven ese viaje (quedarian como
    null y, al releerlos, como NaN, rompiendo el calculo en silencio).
    """
    n_bins = n_bins or config.PSI_BINS
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"edges": [], "freqs": []}

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles)).astype(float)
    interior = edges[1:-1] if len(edges) >= 2 else np.array([])

    counts = bin_counts(values, interior)
    total = max(counts.sum(), 1)
    return {"edges": interior.tolist(), "freqs": (counts / total).tolist()}


def psi_from_freqs(ref_freqs, act_freqs, eps: float = 1e-6) -> float:
    """PSI = sum((act - ref) * ln(act / ref)), con suavizado para evitar log(0)."""
    ref = np.clip(np.asarray(ref_freqs, dtype=float), eps, None)
    act = np.clip(np.asarray(act_freqs, dtype=float), eps, None)
    ref, act = ref / ref.sum(), act / act.sum()
    return float(np.sum((act - ref) * np.log(act / ref)))


def psi_numeric(reference_bins: Dict[str, list], values) -> float:
    """PSI de una variable numerica usando los bins guardados en el baseline."""
    edges = reference_bins.get("edges")
    ref_freqs = reference_bins.get("freqs") or []
    if edges is None or not ref_freqs or len(ref_freqs) < 2:
        return 0.0

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0

    counts = bin_counts(values, edges)
    return psi_from_freqs(ref_freqs, counts / max(counts.sum(), 1))


def psi_categorical(reference_freqs: Dict[str, float], values) -> float:
    """PSI de una variable categorica sobre la union de categorias de ambos lados."""
    series = pd.Series(values).dropna().astype(str).str.lower()
    if len(series) == 0 or not reference_freqs:
        return 0.0

    actual_freqs = series.value_counts(normalize=True).to_dict()
    categories = sorted(set(reference_freqs) | set(actual_freqs))
    ref = [reference_freqs.get(c, 0.0) for c in categories]
    act = [actual_freqs.get(c, 0.0) for c in categories]
    return psi_from_freqs(ref, act)


def psi_status(psi_value: float, thresholds=None) -> str:
    """Traduce un PSI a estado. Sin `thresholds`, usa los umbrales globales."""
    thresholds = thresholds or config.DEFAULT_THRESHOLDS
    if psi_value > thresholds.psi_alert:
        return config.STATUS_ALERT
    if psi_value > thresholds.psi_warn:
        return config.STATUS_WARNING
    return config.STATUS_OK


# ============================================================================
# Baseline
# ============================================================================

def build_baseline(
    X_train: pd.DataFrame,
    y_train_original,
    val_metrics: Dict[str, float],
    val_predictions,
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Construye el snapshot de referencia que scoring usara para medir drift."""
    baseline: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(X_train)),
        "psi_bins": config.PSI_BINS,
        "numeric_features": {},
        "categorical_features": {},
    }

    for column in config.DRIFT_NUMERICAL_FEATURES:
        if column not in X_train.columns:
            continue
        values = pd.to_numeric(X_train[column], errors="coerce").dropna()
        baseline["numeric_features"][column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(values.quantile(0.01)),
            "p25": float(values.quantile(0.25)),
            "p50": float(values.quantile(0.50)),
            "p75": float(values.quantile(0.75)),
            "p99": float(values.quantile(0.99)),
            "bins": make_bins(values),
        }

    for column in config.DRIFT_CATEGORICAL_FEATURES:
        if column not in X_train.columns:
            continue
        freqs = X_train[column].astype(str).str.lower().value_counts(normalize=True)
        baseline["categorical_features"][column] = {"freqs": freqs.to_dict()}

    target = pd.Series(np.asarray(y_train_original, dtype=float)).dropna()
    baseline["target"] = {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "p25": float(target.quantile(0.25)),
        "p50": float(target.quantile(0.50)),
        "p75": float(target.quantile(0.75)),
        "bins": make_bins(target),
    }

    # Distribucion de predicciones sobre validacion: es la referencia que permite
    # detectar comportamiento anomalo en lotes SIN ground truth, como prod3.
    predictions = pd.Series(np.asarray(val_predictions, dtype=float)).dropna()
    baseline["predictions"] = {
        "mean": float(predictions.mean()),
        "std": float(predictions.std()),
        "min": float(predictions.min()),
        "max": float(predictions.max()),
        "bins": make_bins(predictions),
    }

    baseline["metrics"] = {f"val_{k}": float(v) for k, v in val_metrics.items()}
    if extra:
        baseline.update(extra)

    return _jsonable(baseline)


def save_baseline(baseline: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return path


def load_baseline(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ============================================================================
# Senales y reporte por lote
# ============================================================================

@dataclass
class Signal:
    """Una senal de monitoreo evaluada, con su estado ya resuelto."""
    name: str
    category: str          # schema | feature_drift | prediction_drift | performance | target_drift
    value: Optional[float]
    status: str
    detail: str
    # De donde salieron los umbrales con los que se decidio este estado. Sin este dato
    # un WARNING emitido con una regla custom es indistinguible de uno emitido con el
    # umbral global, y el reporte deja de ser auditable.
    thresholds_source: str = "default"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category,
            "value": self.value, "status": self.status, "detail": self.detail,
            "thresholds_source": self.thresholds_source,
        }


@dataclass
class BatchReport:
    """Resultado completo del monitoreo de un lote."""
    batch_id: str
    n_rows: int
    has_target: bool
    status: str = config.STATUS_OK
    diagnosis: str = ""
    signals: List[Signal] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    psi: Dict[str, float] = field(default_factory=dict)
    violations: List[dict] = field(default_factory=list)
    prediction_summary: Dict[str, float] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    scored_at: str = ""

    def signals_by_category(self, category: str) -> List[Signal]:
        return [s for s in self.signals if s.category == category]

    @property
    def psi_max(self) -> float:
        return max(self.psi.values()) if self.psi else 0.0

    @property
    def psi_max_feature(self) -> Optional[str]:
        return max(self.psi, key=self.psi.get) if self.psi else None

    def to_dict(self) -> dict:
        return _jsonable({
            "batch_id": self.batch_id,
            "n_rows": self.n_rows,
            "has_target": self.has_target,
            "status": self.status,
            "diagnosis": self.diagnosis,
            "scored_at": self.scored_at,
            "model_info": self.model_info,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "psi": self.psi,
            "psi_max": self.psi_max,
            "psi_max_feature": self.psi_max_feature,
            "prediction_summary": self.prediction_summary,
            "thresholds": self.thresholds,
            "signals": [s.to_dict() for s in self.signals],
            "violations": self.violations,
        })

    @property
    def custom_threshold_sources(self) -> List[str]:
        """Origenes de umbral distintos del global que se aplicaron en este lote."""
        seen = []
        for signal in self.signals:
            if signal.thresholds_source != "default" and signal.thresholds_source not in seen:
                seen.append(signal.thresholds_source)
        return seen


# ---------------------------------------------------------------------------
# Evaluacion de cada senal
# ---------------------------------------------------------------------------

# Violaciones que no son una cuestion de porcentaje: rompen el contrato aunque el
# umbral de filas invalidas se afloje, asi que conservan su severidad original.
#
#   missing_column      la columna no esta
#   not_numeric         la columna esta pero no se puede interpretar
#   row_count_mismatch  features y target tienen distinta cantidad de filas, o sea
#                       que las predicciones se estan comparando contra el actual
#                       equivocado. Es el fallo de integridad mas grave de un lote y
#                       no tiene sentido negociarlo con un porcentaje.
_STRUCTURAL_VIOLATIONS = ("missing_column", "not_numeric", "row_count_mismatch")


def evaluate_schema(violations: List[Any], thresholds=None) -> List[Signal]:
    """Convierte las violaciones del contrato en senales.

    La severidad se RECALCULA aca con los umbrales del lote. `data_loader` resuelve la
    suya con los globales porque no sabe a que lote pertenece el archivo que esta
    leyendo; el monitoreo si lo sabe, y es el que decide el estado del lote.
    """
    thresholds = thresholds or config.DEFAULT_THRESHOLDS
    signals = []
    for violation in violations:
        data = violation.to_dict() if hasattr(violation, "to_dict") else dict(violation)
        pct = float(data.get("pct_rows", 0.0))

        if data.get("kind") in _STRUCTURAL_VIOLATIONS:
            status = data["severity"]
        elif pct > thresholds.schema_alert_pct:
            status = config.STATUS_ALERT
        elif pct > thresholds.schema_warn_pct:
            status = config.STATUS_WARNING
        else:
            status = config.STATUS_OK

        signals.append(Signal(
            name=f"schema:{data['column']}:{data['kind']}",
            category="schema",
            value=pct,
            status=status,
            detail=f"{data['column']}: {data['detail']}",
            thresholds_source=thresholds.source_for("schema"),
        ))
    if not signals:
        signals.append(Signal(
            name="schema", category="schema", value=0.0,
            status=config.STATUS_OK, detail="contrato de datos respetado",
            thresholds_source=thresholds.source_for("schema"),
        ))
    return signals


def evaluate_feature_drift(features: pd.DataFrame, baseline: Dict[str, Any],
                           batch_id: str = None):
    """PSI por feature contra la distribucion de training.

    Es la unica senal que resuelve umbrales POR FEATURE: `bmi` puede merecer una vara
    mas estricta que `region` sin que eso afecte al resto de las senales del lote.
    """
    signals, psi_values = [], {}

    for column, stats in baseline.get("numeric_features", {}).items():
        if column not in features.columns:
            continue
        thresholds = config.resolve_thresholds(batch_id=batch_id, feature=column)
        values = pd.to_numeric(features[column], errors="coerce")
        value = psi_numeric(stats.get("bins", {}), values)
        psi_values[column] = value
        observed_mean = float(values.dropna().mean()) if values.notna().any() else float("nan")
        signals.append(Signal(
            name=f"drift:{column}", category="feature_drift",
            value=value, status=psi_status(value, thresholds),
            detail=(f"PSI={value:.4f} | media baseline={stats['mean']:.3f} "
                    f"-> batch={observed_mean:.3f}"),
            thresholds_source=thresholds.source_for("psi"),
        ))

    for column, stats in baseline.get("categorical_features", {}).items():
        if column not in features.columns:
            continue
        thresholds = config.resolve_thresholds(batch_id=batch_id, feature=column)
        value = psi_categorical(stats.get("freqs", {}), features[column])
        psi_values[column] = value
        signals.append(Signal(
            name=f"drift:{column}", category="feature_drift",
            value=value, status=psi_status(value, thresholds),
            detail=f"PSI={value:.4f} (categorica)",
            thresholds_source=thresholds.source_for("psi"),
        ))

    return signals, psi_values


def evaluate_prediction_drift(predictions, baseline: Dict[str, Any], thresholds=None) -> Signal:
    """Drift de la distribucion de predicciones.

    Es la unica senal de comportamiento del modelo disponible cuando el lote no
    trae ground truth.
    """
    thresholds = thresholds or config.DEFAULT_THRESHOLDS
    reference = baseline.get("predictions", {})
    value = psi_numeric(reference.get("bins", {}), predictions)
    observed_mean = float(np.nanmean(np.asarray(predictions, dtype=float)))
    return Signal(
        name="drift:predictions", category="prediction_drift",
        value=value, status=psi_status(value, thresholds),
        detail=(f"PSI={value:.4f} | media baseline=${reference.get('mean', float('nan')):,.2f} "
                f"-> batch=${observed_mean:,.2f}"),
        thresholds_source=thresholds.source_for("psi"),
    )


def evaluate_performance(metrics: Dict[str, float], baseline_metrics: Dict[str, float],
                         thresholds=None) -> List[Signal]:
    """Compara RMSE y R2 del lote contra los de validacion."""
    thresholds = thresholds or config.DEFAULT_THRESHOLDS
    signals = []

    base_rmse = baseline_metrics.get("val_rmse")
    if base_rmse and np.isfinite(base_rmse) and base_rmse > 0:
        ratio = metrics["rmse"] / base_rmse
        if ratio > thresholds.perf_alert_ratio:
            status = config.STATUS_ALERT
        elif ratio > thresholds.perf_warn_ratio:
            status = config.STATUS_WARNING
        else:
            status = config.STATUS_OK
        signals.append(Signal(
            name="performance:rmse_ratio", category="performance",
            value=float(ratio), status=status,
            detail=(f"RMSE batch=${metrics['rmse']:,.2f} vs validacion=${base_rmse:,.2f} "
                    f"(x{ratio:.2f}; umbrales "
                    f"{thresholds.perf_warn_ratio}/{thresholds.perf_alert_ratio})"),
            thresholds_source=thresholds.source_for("perf"),
        ))

    base_r2 = baseline_metrics.get("val_r2")
    if base_r2 is not None and np.isfinite(base_r2):
        drop = base_r2 - metrics["r2"]
        if drop > thresholds.r2_alert_drop:
            status = config.STATUS_ALERT
        elif drop > thresholds.r2_warn_drop:
            status = config.STATUS_WARNING
        else:
            status = config.STATUS_OK
        signals.append(Signal(
            name="performance:r2_drop", category="performance",
            value=float(drop), status=status,
            detail=(f"R2 batch={metrics['r2']:.4f} vs validacion={base_r2:.4f} "
                    f"(caida={drop:.4f})"),
            thresholds_source=thresholds.source_for("r2"),
        ))

    return signals


def evaluate_target_drift(target, baseline: Dict[str, Any], thresholds=None) -> List[Signal]:
    """Desvio de la media del target y PSI de su distribucion."""
    thresholds = thresholds or config.DEFAULT_THRESHOLDS
    reference = baseline.get("target", {})
    base_mean = reference.get("mean")
    values = pd.Series(np.asarray(target, dtype=float)).dropna()
    signals = []

    if base_mean and np.isfinite(base_mean) and base_mean != 0 and len(values):
        observed_mean = float(values.mean())
        shift = abs(observed_mean / base_mean - 1.0)
        if shift > thresholds.target_shift_alert:
            status = config.STATUS_ALERT
        elif shift > thresholds.target_shift_warn:
            status = config.STATUS_WARNING
        else:
            status = config.STATUS_OK
        signals.append(Signal(
            name="target:mean_shift", category="target_drift",
            value=float(shift), status=status,
            detail=(f"media baseline=${base_mean:,.2f} -> batch=${observed_mean:,.2f} "
                    f"(ratio x{observed_mean / base_mean:,.2f})"),
            thresholds_source=thresholds.source_for("target_shift"),
        ))

    psi_value = psi_numeric(reference.get("bins", {}), values)
    signals.append(Signal(
        name="target:psi", category="target_drift",
        value=psi_value, status=psi_status(psi_value, thresholds),
        detail=f"PSI de la distribucion del target = {psi_value:.4f}",
        thresholds_source=thresholds.source_for("psi"),
    ))
    return signals


# ---------------------------------------------------------------------------
# Diagnostico
# ---------------------------------------------------------------------------

def diagnose(report: BatchReport) -> str:
    """Traduce la combinacion de senales a una explicacion accionable.

    La distincion util no es "hay drift si/no", sino DONDE esta el problema:
    features estables con target desviado apunta a un defecto de datos en la
    etiqueta, no a un modelo que dejo de servir.
    """
    def has(category: str, minimum: str = config.STATUS_WARNING) -> bool:
        floor = config.STATUS_ORDER.index(minimum)
        return any(
            config.STATUS_ORDER.index(s.status) >= floor
            for s in report.signals_by_category(category)
        )

    feature_schema_bad = any(
        s.status != config.STATUS_OK and not s.name.startswith(f"schema:{config.TARGET_COLUMN}")
        for s in report.signals_by_category("schema")
    )
    target_schema_bad = any(
        s.status != config.STATUS_OK and s.name.startswith(f"schema:{config.TARGET_COLUMN}")
        for s in report.signals_by_category("schema")
    )

    features_drifted = has("feature_drift") or feature_schema_bad
    target_drifted = has("target_drift") or target_schema_bad
    perf_degraded = has("performance")
    predictions_drifted = has("prediction_drift")

    if not report.has_target:
        if features_drifted:
            return (
                "Las features violan el contrato de datos o driftearon, y el lote NO trae "
                "ground truth: la performance real no es verificable. Revisar el origen del "
                "archivo antes de usar estas predicciones para decidir."
            )
        if predictions_drifted:
            return (
                "Las features se mantienen estables pero la distribucion de predicciones "
                "se corrio respecto de validacion. Sin ground truth no se puede confirmar "
                "degradacion; conviene conseguir etiquetas para este lote."
            )
        return (
            "Lote sin ground truth. Features y distribucion de predicciones estables "
            "respecto de training: no hay senales de anomalia."
        )

    if target_drifted and not features_drifted:
        return (
            "Las features son estables y el desvio esta SOLO en el target: apunta a un "
            "problema de CALIDAD DE DATOS en la etiqueta (unidad o escala), no a "
            "degradacion del modelo. Corregir el proceso que genera el archivo de target "
            "antes de considerar un reentrenamiento."
        )
    if features_drifted and perf_degraded:
        return (
            "Drift de covariables acompanado de caida de performance: el modelo esta "
            "operando fuera de la poblacion con la que fue entrenado. Candidato a "
            "reentrenamiento."
        )
    if features_drifted:
        return (
            "Drift de covariables sin caida medible de performance: el modelo todavia "
            "generaliza sobre la poblacion nueva. Monitorear en los proximos lotes."
        )
    if perf_degraded:
        return (
            "Performance degradada sin drift de entrada detectable: posible concept drift "
            "(cambio en la relacion entre features y target). Investigar factores externos."
        )
    return "Lote sano: metricas y distribuciones en linea con el entrenamiento."


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def monitor_batch(
    batch_id: str,
    features: pd.DataFrame,
    predictions,
    baseline: Dict[str, Any],
    target=None,
    violations: List[Any] = None,
    model_info: Dict[str, Any] = None,
    n_features: int = None,
) -> BatchReport:
    """Evalua todas las senales de un lote y devuelve su reporte consolidado."""
    baseline_metrics = baseline.get("metrics", {})
    predictions = np.asarray(predictions, dtype=float)

    # Umbrales del lote, resueltos una sola vez. El drift de features resuelve ademas
    # por feature, porque es la unica senal donde tiene sentido afinar columna por
    # columna (ver config.resolve_thresholds).
    thresholds = config.resolve_thresholds(batch_id=batch_id)

    report = BatchReport(
        batch_id=batch_id,
        n_rows=len(features),
        has_target=target is not None,
        baseline_metrics=baseline_metrics,
        model_info=model_info or {},
        thresholds=thresholds.to_dict(),
        scored_at=datetime.now().isoformat(timespec="seconds"),
        violations=[
            v.to_dict() if hasattr(v, "to_dict") else dict(v)
            for v in (violations or [])
        ],
    )

    report.signals += evaluate_schema(violations or [], thresholds)

    drift_signals, psi_values = evaluate_feature_drift(features, baseline, batch_id=batch_id)
    report.signals += drift_signals
    report.psi = psi_values

    report.signals.append(evaluate_prediction_drift(predictions, baseline, thresholds))
    report.prediction_summary = {
        "mean": float(np.nanmean(predictions)),
        "std": float(np.nanstd(predictions)),
        "min": float(np.nanmin(predictions)),
        "max": float(np.nanmax(predictions)),
    }

    if target is not None:
        report.metrics = canonical_metrics(target, predictions, n_features)
        report.signals += evaluate_performance(report.metrics, baseline_metrics, thresholds)
        report.signals += evaluate_target_drift(target, baseline, thresholds)

    report.status = worst_status(s.status for s in report.signals)
    report.diagnosis = diagnose(report)
    return report


def _as_transition_dicts(transitions):
    """Normaliza transiciones de alerta a dicts.

    Se acepta tanto el objeto `alerts.AlertTransition` como su dict para que este
    modulo no tenga que importar `alerts`: el monitoreo no deberia depender del
    historial, es al reves.

    `None` se PRESERVA y no se colapsa a `[]`: son cosas distintas. `[]` significa
    "se reconcilio y no cambio nada"; `None` significa "no se pudo reconciliar", que
    es el caso de un lote que fallo antes de llegar a esa altura.
    """
    if transitions is None:
        return None
    return [
        transition.to_dict() if hasattr(transition, "to_dict") else dict(transition)
        for transition in transitions
    ]


def _count_transitions(transitions) -> Dict[str, int]:
    counts = {"new": 0, "ongoing": 0, "resolved": 0}
    for transition in transitions or []:
        key = str(transition.get("transition", "")).lower()
        if key in counts:
            counts[key] += 1
    return counts


def consolidate(reports: List[BatchReport], model_info: Dict[str, Any] = None,
                transitions_by_batch: Dict[str, Any] = None) -> Dict[str, Any]:
    """Arma el reporte global a partir de los reportes por lote."""
    transitions_by_batch = transitions_by_batch or {}
    por_lote = {
        batch_id: _as_transition_dicts(transitions)
        for batch_id, transitions in transitions_by_batch.items()
    }

    total = {"new": 0, "ongoing": 0, "resolved": 0}
    for transitions in por_lote.values():
        for key, value in _count_transitions(transitions).items():
            total[key] += value

    return _jsonable({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_info": model_info or {},
        "overall_status": worst_status(r.status for r in reports),
        "n_batches": len(reports),
        "status_counts": {
            status: sum(1 for r in reports if r.status == status)
            for status in config.STATUS_ORDER
        },
        # El semaforo dice COMO esta cada lote; esto dice QUE CAMBIO desde la corrida
        # anterior, que es lo unico que amerita que alguien mire el reporte.
        "alerts": {"summary": total, "by_batch": por_lote},
        "batches": [r.to_dict() for r in reports],
    })


def to_dataframe(reports: List[BatchReport],
                 transitions_by_batch: Dict[str, Any] = None) -> pd.DataFrame:
    """Vista tabular del reporte, para CSV y para persistir en la base."""
    transitions_by_batch = transitions_by_batch or {}
    rows = []
    for report in reports:
        counts = _count_transitions(
            _as_transition_dicts(transitions_by_batch.get(report.batch_id))
        )
        rows.append({
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
            "alerts_new": counts["new"],
            "alerts_ongoing": counts["ongoing"],
            "alerts_resolved": counts["resolved"],
            "diagnosis": report.diagnosis,
            "model_name": report.model_info.get("model_name"),
            "model_version": report.model_info.get("model_version"),
            "model_source": report.model_info.get("source"),
            "mlflow_run_id": report.model_info.get("scoring_run_id"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Renderizado en texto
# ---------------------------------------------------------------------------

_ICON = {config.STATUS_OK: "[ OK ]", config.STATUS_WARNING: "[WARN]", config.STATUS_ALERT: "[ALRT]"}
_TRANSITION_ICON = {"NEW": "[NUEVA]   ", "ONGOING": "[PERSISTE]", "RESOLVED": "[RESUELTA]"}
_TRANSITION_ORDER = {"NEW": 0, "ONGOING": 1, "RESOLVED": 2}
_WIDTH = 78


def _render_batch_alerts(transitions, indent: str = "  ") -> List[str]:
    """Bloque de alertas de un lote: lo nuevo separado de lo que ya se sabia."""
    if transitions is None:
        return [f"{indent}el lote no se pudo procesar: no hubo reconciliacion de alertas",
                f"{indent}(las alertas previas de este lote siguen abiertas)"]
    if not transitions:
        return [f"{indent}sin cambios en el historial de alertas"]

    lines = []
    for transition in sorted(transitions,
                             key=lambda x: (_TRANSITION_ORDER.get(x.get("transition"), 9),
                                            x.get("signal_name", ""))):
        kind = transition.get("transition", "")
        icon = _TRANSITION_ICON.get(kind, "[?]       ")
        occurrences = transition.get("occurrences", 1)
        if kind == "NEW":
            contexto = "1a vez"
        elif kind == "ONGOING":
            contexto = f"{occurrences} corridas, desde {transition.get('first_seen', 'n/d')}"
            previa, actual = transition.get("escalated_from"), transition.get("severity")
            if previa and actual and previa != actual:
                subio = (config.STATUS_ORDER.index(actual) > config.STATUS_ORDER.index(previa)
                         if actual in config.STATUS_ORDER and previa in config.STATUS_ORDER
                         else True)
                contexto += f"; {'ESCALADA' if subio else 'BAJO'} {previa} -> {actual}"
        else:
            contexto = f"estuvo {occurrences} corrida(s) abierta"
        lines.append(f"{indent}{icon} {transition.get('signal_name', ''):<28} ({contexto})")
    return lines


def render_text_report(consolidated: Dict[str, Any]) -> str:
    """Reporte legible en texto, el que queda en results/monitoring_report_*.txt."""
    lines = [
        "=" * _WIDTH,
        "METLIFE INSURANCE COST PREDICTION - REPORTE DE MONITOREO DE SCORING".center(_WIDTH),
        "=" * _WIDTH,
        f"Generado:        {consolidated['generated_at']}",
    ]

    model = consolidated.get("model_info", {})
    if model:
        lines += [
            f"Modelo:          {model.get('model_name', 'n/d')} v{model.get('model_version', 'n/d')}",
            f"Origen:          {model.get('source', 'n/d')}",
            f"Run de training: {model.get('run_id', 'n/d')}",
        ]

    counts = consolidated.get("status_counts", {})
    lines += [
        f"Lotes:           {consolidated['n_batches']}",
        f"Estado global:   {_ICON[consolidated['overall_status']]} {consolidated['overall_status']}",
        f"Desglose:        OK={counts.get('OK', 0)}  WARNING={counts.get('WARNING', 0)}  ALERT={counts.get('ALERT', 0)}",
    ]

    alerts_block = consolidated.get("alerts") or {}
    resumen = alerts_block.get("summary") or {}
    if resumen:
        lines.append(
            f"Alertas:         NUEVAS={resumen.get('new', 0)}  "
            f"PERSISTEN={resumen.get('ongoing', 0)}  RESUELTAS={resumen.get('resolved', 0)}"
        )
        if resumen.get("new"):
            lines.append("                 ^ lo unico que no se sabia en la corrida anterior")

    lines += [
        "",
        "-" * _WIDTH,
        "RESUMEN POR LOTE",
        "-" * _WIDTH,
        f"{'LOTE':<10}{'ESTADO':<10}{'FILAS':>7}{'TARGET':>9}{'RMSE':>14}{'R2':>9}{'PSI max':>10}",
    ]

    for batch in consolidated["batches"]:
        metrics = batch.get("metrics") or {}
        rmse = f"${metrics['rmse']:,.0f}" if metrics.get("rmse") is not None else "n/d"
        r2 = f"{metrics['r2']:.4f}" if metrics.get("r2") is not None else "n/d"
        lines.append(
            f"{batch['batch_id']:<10}{batch['status']:<10}{batch['n_rows']:>7}"
            f"{('si' if batch['has_target'] else 'NO'):>9}{rmse:>14}{r2:>9}"
            f"{batch['psi_max']:>10.4f}"
        )

    for batch in consolidated["batches"]:
        lines += ["", "=" * _WIDTH,
                  f"LOTE: {batch['batch_id']}   ->   {_ICON[batch['status']]} {batch['status']}",
                  "=" * _WIDTH,
                  f"Filas: {batch['n_rows']}   Ground truth: {'si' if batch['has_target'] else 'NO'}"]

        metrics, baseline = batch.get("metrics") or {}, batch.get("baseline_metrics") or {}
        if metrics:
            lines += ["", "Performance (lote vs validacion del modelo):",
                      f"  {'':12}{'LOTE':>16}{'VALIDACION':>16}"]
            for key, label, fmt in [("rmse", "RMSE", "${:,.2f}"), ("mae", "MAE", "${:,.2f}"),
                                    ("r2", "R2", "{:.4f}"), ("mape", "MAPE", "{:.2f}%")]:
                base = baseline.get(f"val_{key}")
                lines.append(
                    f"  {label:<12}{fmt.format(metrics[key]):>16}"
                    f"{(fmt.format(base) if base is not None else 'n/d'):>16}"
                )
        else:
            lines += ["", "Performance: no calculable (el lote no trae ground truth)."]

        predictions = batch.get("prediction_summary") or {}
        if predictions:
            lines += ["", "Distribucion de predicciones:",
                      f"  media=${predictions['mean']:,.2f}  std=${predictions['std']:,.2f}  "
                      f"min=${predictions['min']:,.2f}  max=${predictions['max']:,.2f}"]

        # El estado de cada senal se LEE del reporte y no se recalcula: recalcularlo
        # con los umbrales globales contradiria al listado de senales de mas abajo
        # cada vez que una regla por feature o por lote este en juego.
        signals = batch.get("signals", [])
        status_by_name = {s["name"]: s for s in signals}

        if batch.get("psi"):
            lines += ["", "Drift de features (PSI):"]
            for feature, value in sorted(batch["psi"].items(), key=lambda kv: -kv[1]):
                signal = status_by_name.get(f"drift:{feature}", {})
                status = signal.get("status", psi_status(value))
                origin = signal.get("thresholds_source", "default")
                marca = "" if origin == "default" else f"  (regla {origin})"
                lines.append(f"  {_ICON[status]} {feature:<12} PSI={value:.4f}{marca}")

        violations = batch.get("violations") or []
        if violations:
            lines += ["", "Violaciones del contrato de datos:"]
            for violation in violations:
                signal = status_by_name.get(f"schema:{violation['column']}:{violation['kind']}", {})
                status = signal.get("status", violation["severity"])
                lines.append(
                    f"  {_ICON[status]} {violation['column']}: {violation['detail']} "
                    f"({violation['n_rows']} filas, {violation['pct_rows']:.1f}%)"
                )

        non_ok = [s for s in signals if s["status"] != config.STATUS_OK]
        if non_ok:
            lines += ["", "Senales que no estan en OK:"]
            for signal in non_ok:
                origin = signal.get("thresholds_source", "default")
                marca = "" if origin == "default" else f"  [regla {origin}]"
                lines.append(f"  {_ICON[signal['status']]} {signal['name']}: {signal['detail']}{marca}")

        custom = sorted({s.get("thresholds_source", "default") for s in signals} - {"default"})
        if custom:
            lines += ["", f"Reglas de umbral aplicadas: {', '.join(custom)}"]

        # Se distingue la clave AUSENTE (no hay informacion de alertas en este
        # reporte) de la clave presente con valor None (el lote existe pero no se
        # pudo reconciliar). Son estados distintos y decir lo mismo de los dos
        # ocultaria justo el caso que hay que mirar.
        by_batch = alerts_block.get("by_batch") or {}
        if batch["batch_id"] in by_batch:
            lines += ["", "Alertas:"] + _render_batch_alerts(by_batch[batch["batch_id"]])

        lines += ["", "Diagnostico:"]
        lines += ["  " + chunk for chunk in _wrap(batch["diagnosis"], _WIDTH - 2)]

    lines += ["", "=" * _WIDTH] + _render_thresholds_footer(consolidated) + ["=" * _WIDTH]
    return "\n".join(lines)


def _render_thresholds_footer(consolidated: Dict[str, Any]) -> List[str]:
    """Pie con los umbrales EFECTIVOS, no con los globales.

    Si algun lote se midio con una regla propia, imprimir solo los globales seria
    enganoso: el lector deduciria del reporte umbrales que a ese lote no se le
    aplicaron.
    """
    default = config.DEFAULT_THRESHOLDS

    def bloque(thresholds, titulo):
        return [
            titulo,
            f"  PSI:             WARNING > {thresholds.psi_warn}   ALERT > {thresholds.psi_alert}",
            f"  RMSE ratio:      WARNING > {thresholds.perf_warn_ratio}x   "
            f"ALERT > {thresholds.perf_alert_ratio}x",
            f"  Caida de R2:     WARNING > {thresholds.r2_warn_drop}   "
            f"ALERT > {thresholds.r2_alert_drop}",
            f"  Desvio target:   WARNING > {thresholds.target_shift_warn:.0%}   "
            f"ALERT > {thresholds.target_shift_alert:.0%}",
            f"  Filas invalidas: WARNING > {thresholds.schema_warn_pct}%   "
            f"ALERT > {thresholds.schema_alert_pct}%",
        ]

    lines = bloque(default, "Umbrales por defecto:")

    excepciones = []
    for batch in consolidated.get("batches", []):
        for signal in batch.get("signals", []):
            origin = signal.get("thresholds_source", "default")
            if origin != "default":
                excepciones.append((batch["batch_id"], origin, signal["name"]))

    if excepciones:
        lines += ["", "Excepciones aplicadas (ver config/monitoring_rules.json):"]
        vistos = set()
        for batch_id, origin, signal_name in excepciones:
            clave = (batch_id, origin)
            if clave in vistos:
                continue
            vistos.add(clave)
            afectadas = sorted({s for b, o, s in excepciones if (b, o) == clave})
            lines.append(f"  {batch_id:<8} regla '{origin}' -> {', '.join(afectadas)}")

    return lines


def _wrap(text: str, width: int) -> List[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]
