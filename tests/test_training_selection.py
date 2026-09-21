"""Tests de la seleccion del mejor modelo entre familias.

No se entrena nada real: se arman Candidates con metricas fabricadas, porque lo que
se esta verificando es la REGLA de decision, no el modelo.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import config
import model_zoo
import training


def _metrics(val_rmse, val_r2=0.8, train_r2=0.85, mae=2000.0, mape=15.0):
    canonicas = {
        "rmse": val_rmse, "mae": mae, "r2": val_r2, "mape": mape,
        "n_samples": 268, "rmse_log": 0.4, "mae_log": 0.2, "r2_log": 0.8,
        "mape_log": 2.4, "adj_r2": val_r2 - 0.01,
    }
    train = dict(canonicas, r2=train_r2, rmse=val_rmse * 0.8)
    return {"train": train, "validation": canonicas,
            "overfitting_score": train_r2 - val_r2}


def _candidate(name, val_rmse, cv_rmse_log=0.40, **kwargs):
    spec = model_zoo.MODEL_SPECS[name]
    search = SimpleNamespace(
        best_score_=-cv_rmse_log,
        best_params_={"model__n_estimators": 100},
        cv_results_={},
    )
    return training.Candidate(
        spec=spec,
        model=SimpleNamespace(named_steps={"model": SimpleNamespace()}),
        search=search,
        metrics=_metrics(val_rmse, **kwargs),
        y_val_pred=np.zeros(268),
        fit_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# selection_value
# ---------------------------------------------------------------------------

def test_selection_value_resuelve_los_tres_tipos_de_nombre():
    metrics = _metrics(4897.22, val_r2=0.83, train_r2=0.88)
    assert training.selection_value(metrics, "val_rmse") == pytest.approx(4897.22)
    assert training.selection_value(metrics, "val_r2") == pytest.approx(0.83)
    assert training.selection_value(metrics, "train_r2") == pytest.approx(0.88)
    assert training.selection_value(metrics, "overfitting_r2_diff") == pytest.approx(0.05)


@pytest.mark.parametrize("nombre", ["val_inexistente", "rmse", "foo_rmse"])
def test_una_metrica_de_seleccion_invalida_falla_explicito(nombre):
    """Resolver en silencio a un default elegiria el modelo equivocado sin que nadie
    se entere: es preferible que el pipeline pare."""
    with pytest.raises(KeyError):
        training.selection_value(_metrics(1000.0), nombre)


def test_sin_nombre_de_metrica_se_usa_la_configurada(monkeypatch):
    """`None` y `""` significan "la de config", igual que en los helpers de entorno."""
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_r2")
    metrics = _metrics(4897.22, val_r2=0.83)
    assert training.selection_value(metrics) == pytest.approx(0.83)
    assert training.selection_value(metrics, "") == pytest.approx(0.83)


# ---------------------------------------------------------------------------
# select_best_candidate
# ---------------------------------------------------------------------------

def test_con_modo_min_gana_el_de_menor_metrica(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [_candidate("xgboost", 5000.0), _candidate("random_forest", 4800.0)]
    assert training.select_best_candidate(candidatos).name == "random_forest"


def test_con_modo_max_gana_el_de_mayor_metrica(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_r2")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "max")
    candidatos = [
        _candidate("xgboost", 5000.0, val_r2=0.79),
        _candidate("random_forest", 4800.0, val_r2=0.84),
    ]
    assert training.select_best_candidate(candidatos).name == "random_forest"


def test_un_empate_lo_gana_el_incumbente(monkeypatch):
    """El desempate por orden de catalogo tiene que ser deterministico: si no, dos
    corridas identicas podrian registrar modelos distintos."""
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [_candidate("xgboost", 4900.0), _candidate("random_forest", 4900.0)]
    assert training.select_best_candidate(candidatos).name == "xgboost"
    # y el orden de la lista de entrada no cambia el resultado
    assert training.select_best_candidate(candidatos[::-1]).name == "xgboost"


def test_avisa_cuando_el_ranking_por_cv_discrepa_del_ranking_por_validacion(
        monkeypatch, caplog):
    """Elegir entre familias por una metrica del mismo set de validacion es una
    comparacion multiple: el ganador tiene algo de ventaja por azar. Si la CV -- la
    estimacion menos sesgada -- prefiere a otro, el margen no es solido y eso tiene
    que quedar escrito."""
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [
        _candidate("xgboost", 4900.0, cv_rmse_log=0.30),        # mejor por CV
        _candidate("random_forest", 4800.0, cv_rmse_log=0.45),  # mejor por validacion
    ]
    with caplog.at_level("WARNING"):
        best = training.select_best_candidate(candidatos)
    assert best.name == "random_forest"
    assert any("no coinciden" in r.message for r in caplog.records)


def test_no_avisa_cuando_los_dos_rankings_coinciden(monkeypatch, caplog):
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [
        _candidate("xgboost", 4900.0, cv_rmse_log=0.45),
        _candidate("random_forest", 4800.0, cv_rmse_log=0.30),
    ]
    with caplog.at_level("WARNING"):
        training.select_best_candidate(candidatos)
    assert not any("no coinciden" in r.message for r in caplog.records)


def test_seleccionar_sin_candidatos_falla():
    with pytest.raises(ValueError):
        training.select_best_candidate([])


# ---------------------------------------------------------------------------
# Comparacion y justificacion
# ---------------------------------------------------------------------------

def test_la_tabla_comparativa_queda_ordenada_por_el_criterio(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [
        _candidate("xgboost", 5200.0),
        _candidate("random_forest", 4800.0),
        _candidate("elasticnet", 5900.0),
    ]
    frame = training.build_comparison_frame(candidatos)
    assert list(frame["family"]) == ["random_forest", "xgboost", "elasticnet"]
    assert frame["val_rmse"].is_monotonic_increasing
    # best_params serializado, para que el CSV sea legible sin evaluar Python
    assert json.loads(frame.iloc[0]["best_params"]) == {"n_estimators": 100}


def test_la_justificacion_sale_de_los_numeros_y_nombra_al_piso_lineal(monkeypatch):
    """El reporte original argumentaba a mano por que XGBoost. Con la comparacion
    corriendo, eso seria falso en cuanto ganara otra familia."""
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [_candidate("xgboost", 4900.0), _candidate("elasticnet", 5900.0)]
    best = training.select_best_candidate(candidatos)
    texto = "\n".join(training.justification_text(candidatos, best))
    assert "XGBoost" in texto
    assert "4,900" in texto
    assert "piso lineal" in texto
    assert "5,900" in texto


def test_si_gana_el_lineal_la_justificacion_lo_dice(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SELECTION_METRIC", "val_rmse")
    monkeypatch.setattr(config, "MODEL_SELECTION_MODE", "min")
    candidatos = [_candidate("xgboost", 5900.0), _candidate("elasticnet", 4900.0)]
    best = training.select_best_candidate(candidatos)
    texto = "\n".join(training.justification_text(candidatos, best))
    assert "Gana el modelo lineal" in texto


# ---------------------------------------------------------------------------
# Importancia de features tolerante a la familia
# ---------------------------------------------------------------------------

def test_compute_feature_importance_distingue_el_tipo_de_importancia():
    """`gain` y `|coeficiente|` no son comparables entre si, asi que el tipo tiene que
    viajar con los valores y llegar al label del grafico."""
    class _Pipe:
        def __init__(self, estimator):
            self.named_steps = {"model": estimator}

    con_gain = _Pipe(SimpleNamespace(feature_importances_=np.array([0.5, 0.3, 0.2])))
    importancias, kind = training.compute_feature_importance(con_gain)
    assert kind == "gain"
    assert list(importancias.values()) == [0.5, 0.3, 0.2]

    lineal = _Pipe(SimpleNamespace(coef_=np.array([-2.0, 1.0])))
    importancias, kind = training.compute_feature_importance(lineal)
    assert kind == "abs_coef"
    assert list(importancias.values()) == [2.0, 1.0]   # valor absoluto, ya ordenado

    sin_nada = _Pipe(SimpleNamespace())
    assert training.compute_feature_importance(sin_nada) == ({}, "none")


def test_el_grafico_etiqueta_el_eje_segun_el_tipo():
    assert training.IMPORTANCE_LABEL["gain"] != training.IMPORTANCE_LABEL["abs_coef"]
    assert set(training.IMPORTANCE_LABEL) == {"gain", "abs_coef", "none"}


def test_no_se_grafica_nada_cuando_no_hay_importancia(tmp_path):
    assert training.plot_feature_importance({}, tmp_path / "x.png") is None
