"""Tests de las transformaciones compartidas entre training y scoring.

Estas funciones son el punto donde se origina el training/serving skew: si
`feature_engineering` o `transform_target` se comportaran distinto en cada
pipeline, el modelo recibiria en produccion features distintas a las que vio al
entrenar, y nada fallaria a la vista.
"""

import numpy as np
import pandas as pd
import pytest

import config
from utils import feature_engineering, transform_target


# ============================================================================
# feature_engineering
# ============================================================================

def test_agrega_exactamente_las_features_derivadas_esperadas(valid_features):
    result = feature_engineering(valid_features, is_training=False)
    assert set(result.columns) == set(config.RAW_FEATURES) | set(config.DERIVED_FEATURES)


def test_no_modifica_el_dataframe_original(valid_features):
    before = valid_features.copy()
    feature_engineering(valid_features, is_training=False)
    pd.testing.assert_frame_equal(valid_features, before)


def test_interacciones_con_smoker_se_anulan_para_no_fumadores(valid_features):
    result = feature_engineering(valid_features, is_training=False)
    no_fuma = result["smoker"] == "no"
    assert (result.loc[no_fuma, "bmi_smoker"] == 0).all()
    assert (result.loc[no_fuma, "age_smoker"] == 0).all()


def test_interacciones_con_smoker_replican_el_valor_para_fumadores(valid_features):
    result = feature_engineering(valid_features, is_training=False)
    fuma = result["smoker"] == "yes"
    assert (result.loc[fuma, "bmi_smoker"] == result.loc[fuma, "bmi"]).all()
    assert (result.loc[fuma, "age_smoker"] == result.loc[fuma, "age"]).all()


def test_terminos_cuadraticos(valid_features):
    result = feature_engineering(valid_features, is_training=False)
    assert np.allclose(result["bmi_squared"], valid_features["bmi"] ** 2)
    assert np.allclose(result["age_squared"], valid_features["age"] ** 2)


@pytest.mark.parametrize("bmi,expected", [(29.99, 0), (30.0, 0), (30.01, 1)])
def test_umbral_de_obesidad_es_estrictamente_mayor_a_30(valid_features, bmi, expected):
    df = valid_features.head(1).copy()
    df["bmi"] = bmi
    assert feature_engineering(df, is_training=False)["bmi_obese"].iloc[0] == expected


@pytest.mark.parametrize("age,expected", [(50, 0), (51, 1)])
def test_umbral_de_senior_es_estrictamente_mayor_a_50(valid_features, age, expected):
    df = valid_features.head(1).copy()
    df["age"] = age
    assert feature_engineering(df, is_training=False)["age_senior"].iloc[0] == expected


def test_es_deterministica(valid_features):
    """Misma entrada, misma salida: requisito para que training y scoring coincidan."""
    pd.testing.assert_frame_equal(
        feature_engineering(valid_features, is_training=True),
        feature_engineering(valid_features, is_training=False),
    )


# ============================================================================
# transform_target
# ============================================================================

def test_log1p_ida_y_vuelta_es_la_identidad():
    charges = np.array([1121.87, 9382.03, 13270.42, 63770.43])
    recovered = transform_target(transform_target(charges, inverse=False), inverse=True)
    assert np.allclose(recovered, charges)


def test_la_transformacion_reduce_la_asimetria():
    """Es la justificacion de usar log1p: charges tiene skew ~1.5."""
    rng = np.random.default_rng(3)
    charges = pd.Series(rng.lognormal(9.2, 0.9, 5000))
    assert abs(pd.Series(transform_target(charges)).skew()) < abs(charges.skew())


def test_transform_target_soporta_series_y_arrays():
    values = [100.0, 1000.0]
    assert np.allclose(transform_target(pd.Series(values)), transform_target(np.array(values)))
