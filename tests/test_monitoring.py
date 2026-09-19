"""Tests de PSI, semaforo y diagnostico."""

import numpy as np
import pandas as pd
import pytest

import config
import monitoring as mon
from data_loader import Violation


# ============================================================================
# PSI
# ============================================================================

def test_psi_es_cero_si_la_distribucion_no_cambia(reference_sample):
    bins = mon.make_bins(reference_sample)
    assert mon.psi_numeric(bins, reference_sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_crece_con_la_magnitud_del_desplazamiento(reference_sample):
    bins = mon.make_bins(reference_sample)
    leve = mon.psi_numeric(bins, reference_sample + 0.5)
    medio = mon.psi_numeric(bins, reference_sample + 3.0)
    fuerte = mon.psi_numeric(bins, reference_sample + 15.0)
    assert 0 < leve < medio < fuerte


def test_valores_totalmente_fuera_de_rango_dan_psi_alto(reference_sample):
    """El caso prod3: al bmi le sacaron el punto decimal y se va de escala.

    Los bordes externos del baseline son -inf/+inf, asi que estos valores caen
    todos en el ultimo bin en vez de quedar fuera del histograma.
    """
    bins = mon.make_bins(reference_sample)
    psi = mon.psi_numeric(bins, reference_sample * 1000)
    assert psi > config.PSI_ALERT
    assert mon.psi_status(psi) == config.STATUS_ALERT


def test_psi_categorico(reference_sample):
    ref = {"male": 0.5, "female": 0.5}
    assert mon.psi_categorical(ref, ["male", "female"] * 50) == pytest.approx(0.0, abs=1e-6)
    assert mon.psi_categorical(ref, ["male"] * 100) > config.PSI_ALERT


def test_psi_categorico_detecta_categoria_nueva():
    ref = {"northeast": 0.25, "northwest": 0.25, "southeast": 0.25, "southwest": 0.25}
    assert mon.psi_categorical(ref, ["centralwest"] * 100) > config.PSI_ALERT


def test_make_bins_tolera_variable_constante():
    bins = mon.make_bins(pd.Series([5.0] * 100))
    assert mon.psi_numeric(bins, [5.0] * 100) == pytest.approx(0.0)


@pytest.mark.parametrize("psi,expected", [
    (0.0, config.STATUS_OK),
    (config.PSI_WARN, config.STATUS_OK),            # el umbral exacto todavia es OK
    (config.PSI_WARN + 1e-6, config.STATUS_WARNING),
    (config.PSI_ALERT, config.STATUS_WARNING),
    (config.PSI_ALERT + 1e-6, config.STATUS_ALERT),
])
def test_umbrales_de_psi_en_los_bordes(psi, expected):
    assert mon.psi_status(psi) == expected


# ============================================================================
# Metricas
# ============================================================================

def test_prediccion_perfecta():
    m = mon.regression_metrics([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(0.0)


def test_mape_ignora_los_ceros_en_vez_de_dividir_por_cero():
    m = mon.regression_metrics([0.0, 100.0], [10.0, 110.0])
    assert np.isfinite(m["mape"])


# ============================================================================
# Estado agregado
# ============================================================================

def test_worst_status_se_queda_con_el_mas_severo():
    assert mon.worst_status([]) == config.STATUS_OK
    assert mon.worst_status([config.STATUS_OK, config.STATUS_OK]) == config.STATUS_OK
    assert mon.worst_status([config.STATUS_OK, config.STATUS_WARNING]) == config.STATUS_WARNING
    assert mon.worst_status([config.STATUS_WARNING, config.STATUS_ALERT]) == config.STATUS_ALERT


# ============================================================================
# Fixtures de baseline / lote
# ============================================================================

@pytest.fixture
def baseline():
    rng = np.random.default_rng(7)
    n = 1000
    X = pd.DataFrame({
        "age": rng.integers(18, 65, n).astype(float),
        "bmi": rng.normal(30, 6, n),
        "children": rng.integers(0, 5, n).astype(float),
        "sex": rng.choice(["male", "female"], n),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    y = rng.normal(13270, 12000, n).clip(1200, 64000)
    preds = y + rng.normal(0, 4800, n)
    # Las metricas de referencia se derivan de las predicciones del propio
    # baseline en vez de hardcodearse: si no, el baseline afirmaria una
    # performance que sus datos no respaldan y cualquier lote sano, generado
    # con el mismo proceso, apareceria como degradado.
    return mon.build_baseline(X, y, mon.regression_metrics(y, preds), preds)


@pytest.fixture
def healthy_batch(baseline):
    rng = np.random.default_rng(11)
    n = 500
    X = pd.DataFrame({
        "age": rng.integers(18, 65, n).astype(float),
        "bmi": rng.normal(30, 6, n),
        "children": rng.integers(0, 5, n).astype(float),
        "sex": rng.choice(["male", "female"], n),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    y = rng.normal(13270, 12000, n).clip(1200, 64000)
    preds = y + rng.normal(0, 4800, n)
    return X, y, preds


def test_baseline_guarda_todo_lo_necesario(baseline):
    assert set(baseline["numeric_features"]) == set(config.DRIFT_NUMERICAL_FEATURES)
    assert set(baseline["categorical_features"]) == set(config.DRIFT_CATEGORICAL_FEATURES)
    # La distribucion de predicciones es lo que permite monitorear lotes sin target.
    assert "bins" in baseline["predictions"]
    assert baseline["metrics"]["val_rmse"] == pytest.approx(4800, rel=0.1)


def test_baseline_es_serializable_a_json(baseline, tmp_path):
    path = mon.save_baseline(baseline, tmp_path / "baseline_stats.json")
    reloaded = mon.load_baseline(path)
    assert reloaded["metrics"]["val_r2"] == pytest.approx(baseline["metrics"]["val_r2"])
    # Los bordes de los bins tienen que sobrevivir el viaje por JSON.
    assert reloaded["numeric_features"]["bmi"]["bins"]["edges"] == baseline["numeric_features"]["bmi"]["bins"]["edges"]
    assert all(e is not None for e in reloaded["numeric_features"]["bmi"]["bins"]["edges"])


# ============================================================================
# Los tres escenarios del challenge
# ============================================================================

def test_lote_sano_queda_en_ok(baseline, healthy_batch):
    X, y, preds = healthy_batch
    report = mon.monitor_batch("prod_sano", X, preds, baseline, target=y, violations=[])
    assert report.status == config.STATUS_OK
    assert "sano" in report.diagnosis.lower()


def test_target_desescalado_culpa_a_los_datos_y_no_al_modelo(baseline, healthy_batch):
    """Escenario prod2: features intactas, target multiplicado por 100."""
    X, y, preds = healthy_batch
    violations = [Violation(
        column="charges", kind="out_of_range", severity=config.STATUS_ALERT,
        n_rows=len(y), pct_rows=100.0, detail="fuera de rango",
    )]
    report = mon.monitor_batch("prod_target_x100", X, preds, baseline,
                               target=y * 100, violations=violations)

    assert report.status == config.STATUS_ALERT
    # Las features NO driftearon: el problema esta solo del lado de la etiqueta.
    assert max(report.psi.values()) < config.PSI_WARN
    assert "CALIDAD DE DATOS" in report.diagnosis
    assert "reentrenamiento" in report.diagnosis.lower()


def test_lote_sin_target_con_features_rotas(baseline, healthy_batch):
    """Escenario prod3: sin ground truth y con el bmi fuera de escala."""
    X, _, preds = healthy_batch
    X = X.copy()
    X["bmi"] = X["bmi"] * 1000
    violations = [Violation(
        column="bmi", kind="out_of_range", severity=config.STATUS_ALERT,
        n_rows=len(X), pct_rows=100.0, detail="fuera de rango",
    )]
    report = mon.monitor_batch("prod_sin_target", X, preds, baseline,
                               target=None, violations=violations)

    assert report.status == config.STATUS_ALERT
    assert report.has_target is False
    assert report.metrics == {}                      # sin target no hay performance
    assert report.psi["bmi"] > config.PSI_ALERT
    assert report.psi_max_feature == "bmi"
    assert "no es verificable" in report.diagnosis


def test_drift_de_features_con_performance_caida_sugiere_reentrenar(baseline, healthy_batch):
    X, y, preds = healthy_batch
    X = X.copy()
    X["age"] = X["age"] + 25          # poblacion claramente mas vieja
    report = mon.monitor_batch("prod_drift", X, preds * 0.4, baseline, target=y, violations=[])
    assert report.status == config.STATUS_ALERT
    assert "reentrenamiento" in report.diagnosis.lower()


# ============================================================================
# Consolidacion y render
# ============================================================================

def test_el_estado_global_es_el_peor_de_los_lotes(baseline, healthy_batch):
    X, y, preds = healthy_batch
    ok = mon.monitor_batch("a", X, preds, baseline, target=y, violations=[])
    alert = mon.monitor_batch("b", X, preds, baseline, target=y * 100, violations=[])

    consolidated = mon.consolidate([ok, alert])
    assert consolidated["overall_status"] == config.STATUS_ALERT
    assert consolidated["status_counts"]["OK"] == 1
    assert consolidated["status_counts"]["ALERT"] == 1
    assert consolidated["n_batches"] == 2


def test_el_reporte_de_texto_incluye_a_todos_los_lotes(baseline, healthy_batch):
    X, y, preds = healthy_batch
    reports = [mon.monitor_batch("prod1", X, preds, baseline, target=y, violations=[]),
               mon.monitor_batch("prod3", X, preds, baseline, target=None, violations=[])]
    text = mon.render_text_report(mon.consolidate(reports))

    assert "prod1" in text and "prod3" in text
    assert "REPORTE DE MONITOREO" in text
    assert "no trae ground truth" in text          # prod3 no tiene performance
    assert "Umbrales aplicados" in text


def test_to_dataframe_da_una_fila_por_lote(baseline, healthy_batch):
    X, y, preds = healthy_batch
    reports = [mon.monitor_batch("prod1", X, preds, baseline, target=y, violations=[]),
               mon.monitor_batch("prod3", X, preds, baseline, target=None, violations=[])]
    df = mon.to_dataframe(reports)

    assert list(df["batch_id"]) == ["prod1", "prod3"]
    assert df.loc[1, "rmse"] is None or pd.isna(df.loc[1, "rmse"])


# ============================================================================
# Metricas canonicas: el contrato que hace comparables training y scoring
# ============================================================================

def test_canonical_metrics_incluye_escala_dolar_y_log():
    rng = np.random.default_rng(5)
    y = rng.lognormal(9.2, 0.9, 400)
    pred = y * rng.normal(1.0, 0.15, 400)
    m = mon.canonical_metrics(y, pred, n_features=14)
    assert set(m) >= {"rmse", "mae", "r2", "mape", "adj_r2",
                      "rmse_log", "mae_log", "r2_log", "mape_log"}
    assert all(np.isfinite(v) for k, v in m.items() if k != "n_samples")


def test_las_metricas_log_coinciden_con_evaluar_en_el_espacio_del_modelo():
    """log1p(expm1(x)) == x, asi que derivar las metricas log desde los dolares
    da lo mismo que calcularlas en el espacio en el que entrena el modelo.

    Es lo que permite que training y scoring compartan una sola funcion: training
    tiene las predicciones en escala log y scoring solo en dolares.
    """
    rng = np.random.default_rng(6)
    y_log = rng.normal(9.1, 0.9, 300)          # lo que ve el modelo
    pred_log = y_log + rng.normal(0, 0.35, 300)

    y_dolares = np.expm1(y_log)                 # lo que ve scoring
    pred_dolares = np.expm1(pred_log)

    directo = mon.regression_metrics(y_log, pred_log)
    derivado = mon.canonical_metrics(y_dolares, pred_dolares)

    assert derivado["rmse_log"] == pytest.approx(directo["rmse"], rel=1e-9)
    assert derivado["r2_log"] == pytest.approx(directo["r2"], rel=1e-9)


def test_training_y_scoring_producen_el_mismo_conjunto_de_claves(baseline, healthy_batch):
    """Sin esto, comparar validacion contra un lote en la UI de MLflow es imposible."""
    X, y, preds = healthy_batch

    metricas_training = mon.canonical_metrics(y, preds, n_features=14)
    reporte_scoring = mon.monitor_batch("lote", X, preds, baseline,
                                        target=y, violations=[], n_features=14)

    assert set(metricas_training) == set(reporte_scoring.metrics)
    assert reporte_scoring.metrics["rmse"] == pytest.approx(metricas_training["rmse"])
    assert np.isfinite(reporte_scoring.metrics["adj_r2"])


def test_adjusted_r2_penaliza_mas_features():
    assert mon.adjusted_r2(0.90, 1000, 5) > mon.adjusted_r2(0.90, 1000, 50)
    assert np.isnan(mon.adjusted_r2(0.90, 10, 50))   # mas features que muestras
