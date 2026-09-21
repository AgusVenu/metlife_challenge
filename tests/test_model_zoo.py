"""Tests del catalogo de familias de modelos.

Todo se verifica sin MLflow ni PostgreSQL: el catalogo es una tabla de datos y una
factory, y eso es exactamente lo que conviene poder testear rapido.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import model_zoo
import training


# ---------------------------------------------------------------------------
# Presupuesto de iteraciones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("budget", [1, 5, 50, 500])
def test_n_iter_nunca_supera_el_grid_ni_baja_de_uno(budget):
    """El tope por cardinalidad es lo que evita el UserWarning de ParameterSampler.

    Sin el, una familia con 20 combinaciones posibles recibiria 50 iteraciones:
    sklearn las recorta igual, pero deja un numero que no es el real en search.n_iter,
    y ese numero termina en el reporte y en MLflow.
    """
    for spec in model_zoo.MODEL_SPECS.values():
        n_iter = spec.n_iter(budget)
        assert 1 <= n_iter <= spec.grid_size(), f"{spec.name} con budget={budget}"


def test_el_presupuesto_global_recorta_todas_las_familias():
    chico = {s.name: s.n_iter(2) for s in model_zoo.MODEL_SPECS.values()}
    grande = {s.name: s.n_iter(500) for s in model_zoo.MODEL_SPECS.values()}
    assert all(chico[k] <= grande[k] for k in chico)
    assert max(chico.values()) <= 2


def test_grid_size_es_el_producto_de_las_opciones():
    spec = model_zoo.MODEL_SPECS["elasticnet"]
    esperado = 1
    for values in spec.param_grid.values():
        esperado *= len(values)
    assert spec.grid_size() == esperado


def test_todas_las_claves_del_grid_llevan_el_prefijo_model():
    """Sin el prefijo, RandomizedSearchCV no sabe a que step del Pipeline aplicarlas."""
    for spec in model_zoo.MODEL_SPECS.values():
        for key in spec.param_grid:
            assert key.startswith("model__"), f"{spec.name}: {key}"


# ---------------------------------------------------------------------------
# Seleccion del catalogo
# ---------------------------------------------------------------------------

def test_get_specs_filtra_y_preserva_el_orden_del_catalogo():
    specs = model_zoo.get_specs(["elasticnet", "xgboost"])
    assert [s.name for s in specs] == ["xgboost", "elasticnet"]


def test_get_specs_falla_fuerte_con_una_familia_desconocida():
    """Un typo en TRAIN_MODEL_FAMILIES no debe entrenar menos familias en silencio."""
    with pytest.raises(ValueError, match="lightgbm"):
        model_zoo.get_specs(["xgboost", "lightgbm"])


def test_get_specs_falla_con_una_lista_vacia():
    with pytest.raises(ValueError):
        model_zoo.get_specs([])


def test_el_default_incluye_al_incumbente_primero():
    """XGBoost va primero: el orden del catalogo es el criterio de desempate."""
    assert list(model_zoo.MODEL_SPECS)[0] == "xgboost"
    assert "xgboost" in config.TRAIN_MODEL_FAMILIES


# ---------------------------------------------------------------------------
# Construccion del pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def datos_sinteticos():
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame({
        "age": rng.integers(18, 64, n),
        "sex": rng.choice(["male", "female"], n),
        "bmi": rng.uniform(18, 45, n),
        "children": rng.integers(0, 5, n),
        "smoker": rng.choice(["yes", "no"], n),
        "region": rng.choice(["northeast", "southwest"], n),
    })
    from utils import feature_engineering
    X = feature_engineering(frame)
    y = np.log1p(rng.uniform(1000, 50000, n))
    return X, y


@pytest.mark.parametrize("name", list(model_zoo.MODEL_SPECS))
def test_cada_familia_arma_un_pipeline_con_los_steps_del_contrato(name, datos_sinteticos):
    """Los nombres "preprocessor" y "model" son contrato del proyecto.

    De ellos dependen utils.get_encoded_feature_names, utils.count_encoded_features
    (y por lo tanto el R2 ajustado que calcula scoring) y compute_feature_importance.
    """
    X, y = datos_sinteticos
    pipeline = training.build_pipeline(model_zoo.MODEL_SPECS[name])
    assert isinstance(pipeline, Pipeline)
    assert [step for step, _ in pipeline.steps] == ["preprocessor", "model"]
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(y)


def test_solo_la_familia_lineal_escala_las_numericas():
    """Sin escalado, la penalizacion L1/L2 castiga desparejo a features de escalas
    distintas (age_squared llega a ~10.000) y la familia lineal quedaria mal evaluada.
    Los arboles son invariantes a la escala, asi que conservan el passthrough original.
    """
    for spec in model_zoo.MODEL_SPECS.values():
        bloque_numerico = training.build_pipeline(spec).named_steps["preprocessor"].transformers[0][1]
        if spec.needs_scaling:
            assert isinstance(bloque_numerico, StandardScaler), spec.name
        else:
            assert bloque_numerico == "passthrough", spec.name

    assert model_zoo.MODEL_SPECS["elasticnet"].needs_scaling is True
    assert model_zoo.MODEL_SPECS["xgboost"].needs_scaling is False


def test_los_estimadores_no_piden_todos_los_cores():
    """El paralelismo vive en RandomizedSearchCV(n_jobs=-1). Si ademas el estimador
    pide todos los cores, los procesos de la CV compiten entre si y el wall-clock
    empeora."""
    for spec in model_zoo.MODEL_SPECS.values():
        params = spec.build().get_params()
        if "n_jobs" in params:
            assert params["n_jobs"] == 1, spec.name


def test_cada_familia_declara_por_que_esta_en_la_comparacion():
    for spec in model_zoo.MODEL_SPECS.values():
        assert spec.rationale.strip(), spec.name
