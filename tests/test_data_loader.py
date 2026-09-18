"""Tests de lectura robusta y validacion del contrato de datos."""

import pandas as pd
import pytest

import config
import data_loader as dl
from conftest import PROD_DIR


# ============================================================================
# Normalizacion del separador decimal
# ============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("14700.80931", 14700.80931),   # punto decimal, ya correcto
    ("14700,80931", 14700.80931),   # coma decimal: el caso de data/prod
    ("1.234,56", 1234.56),          # punto de miles + coma decimal
    ("1,234.56", 1234.56),          # coma de miles + punto decimal
    ("1,234,567", 1234567.0),       # comas de miles
    ("42", 42.0),                   # entero sin separador
    ("  18,5  ", 18.5),             # con espacios alrededor
    ('"7,25"', 7.25),               # entrecomillado
])
def test_normaliza_todas_las_convenciones(raw, expected):
    assert float(dl.normalize_decimal_string(raw)) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", None, "nan", "NULL", "NA"])
def test_valores_vacios_devuelven_none(raw):
    assert dl.normalize_decimal_string(raw) is None


# ============================================================================
# El fallo silencioso que motiva todo el modulo
# ============================================================================

def test_read_csv_directo_corrompe_el_target_en_silencio():
    """Deja documentado POR QUE no se puede usar pd.read_csv() a secas.

    pandas parte '14700,80931' en dos campos, usa el primero como indice y el
    segundo como valor: no lanza ninguna excepcion y devuelve datos erroneos.
    """
    path = PROD_DIR / "dataset_prod1_target.csv.csv"
    ingenuo = pd.read_csv(path)

    assert ingenuo["charges"].iloc[0] == 80931       # basura: son los decimales
    assert ingenuo.index[0] == 14700                 # la parte entera se fue al indice

    robusto = dl.read_single_column_numeric(path, "charges")
    assert robusto.iloc[0] == pytest.approx(14700.80931)


def test_target_de_prod1_es_consistente_con_training():
    """Bien parseado, prod1 tiene la misma escala que el dataset de entrenamiento."""
    serie = dl.read_single_column_numeric(PROD_DIR / "dataset_prod1_target.csv.csv", "charges")
    assert len(serie) == 1338
    assert serie.isna().sum() == 0
    assert serie.mean() == pytest.approx(13276.24, abs=1.0)   # training: 13270.42


def test_target_de_prod2_esta_multiplicado_por_cien():
    """El defecto de prod2: mismo digito a digito, pero con el decimal corrido."""
    prod1 = dl.read_single_column_numeric(PROD_DIR / "dataset_prod1_target.csv.csv", "charges")
    prod2 = dl.read_single_column_numeric(PROD_DIR / "dataset_prod2_target.csv.csv", "charges")
    ratio = prod2 / prod1
    assert ratio.min() == pytest.approx(100.0)
    assert ratio.max() == pytest.approx(100.0)


# ============================================================================
# Descubrimiento de lotes
# ============================================================================

def test_descubre_lotes_con_y_sin_target(tmp_prod_dir):
    batches = {b.batch_id: b for b in dl.discover_batches(tmp_prod_dir)}
    assert set(batches) == {"prodA", "prodB"}
    assert batches["prodA"].has_target
    assert not batches["prodB"].has_target       # sin target -> batch sin etiquetas


def test_tolera_la_doble_extension_csv_csv():
    batches = dl.discover_batches(PROD_DIR)
    assert [b.batch_id for b in batches] == ["prod1", "prod2", "prod3"]
    assert [b.has_target for b in batches] == [True, True, False]


def test_falla_si_el_directorio_no_tiene_lotes(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.discover_batches(tmp_path)


# ============================================================================
# Validacion del contrato de datos
# ============================================================================

def test_features_validas_no_producen_violaciones(valid_features):
    assert dl.validate_features(valid_features) == []


def test_detecta_el_bmi_sin_punto_decimal(bmi_without_decimal):
    """El defecto de prod3: sin target, esta es la UNICA senal disponible."""
    violations = dl.validate_features(bmi_without_decimal)
    assert len(violations) == 1
    v = violations[0]
    assert (v.column, v.kind, v.severity) == ("bmi", "out_of_range", config.STATUS_ALERT)
    assert v.pct_rows == 100.0


def test_detecta_columna_faltante(valid_features):
    violations = dl.validate_features(valid_features.drop(columns=["smoker"]))
    assert [(v.column, v.kind) for v in violations] == [("smoker", "missing_column")]


def test_detecta_categoria_desconocida(valid_features):
    df = valid_features.copy()
    df.loc[0, "region"] = "centralwest"
    violations = dl.validate_features(df)
    assert [(v.column, v.kind) for v in violations] == [("region", "unknown_category")]


def test_detecta_nulos(valid_features):
    df = valid_features.copy()
    df.loc[0, "bmi"] = None
    kinds = [(v.column, v.kind) for v in dl.validate_features(df)]
    assert ("bmi", "nulls") in kinds


def test_target_en_rango_no_produce_violaciones():
    serie = pd.Series([1121.87, 13270.42, 63770.43])
    assert dl.validate_target(serie) == []


def test_target_fuera_de_escala_dispara_alert():
    """El defecto de prod2 visto desde la validacion del target."""
    serie = pd.Series([1470080.93, 154026.16, 397923.25])
    violations = dl.validate_target(serie)
    assert len(violations) == 1
    assert violations[0].kind == "out_of_range"
    assert violations[0].severity == config.STATUS_ALERT


# ============================================================================
# Severidad agregada
# ============================================================================

def test_worst_severity_se_queda_con_la_mas_grave():
    mk = lambda sev: dl.Violation("c", "k", sev, 1, 1.0, "d")
    assert dl.worst_severity([]) == config.STATUS_OK
    assert dl.worst_severity([mk(config.STATUS_OK), mk(config.STATUS_WARNING)]) == config.STATUS_WARNING
    assert dl.worst_severity([mk(config.STATUS_ALERT), mk(config.STATUS_OK)]) == config.STATUS_ALERT


# ============================================================================
# Carga completa, contra los archivos reales del repo
# ============================================================================

@pytest.mark.parametrize("batch_id,has_target,expected_status", [
    ("prod1", True, config.STATUS_OK),
    ("prod2", True, config.STATUS_ALERT),
    ("prod3", False, config.STATUS_ALERT),
])
def test_estado_de_esquema_de_los_lotes_reales(batch_id, has_target, expected_status):
    batch = next(b for b in dl.discover_batches(PROD_DIR) if b.batch_id == batch_id)
    loaded = dl.load_batch(batch)
    assert loaded.n_rows == 1338
    assert loaded.has_target is has_target
    assert loaded.schema_status == expected_status
