"""Fixtures compartidas.

Los tests corren sin PostgreSQL y sin MLflow: todo lo que se prueba aca son
funciones puras (parseo, validacion, PSI, semaforo), que es precisamente donde
un bug se paga caro y en silencio.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Los scripts se ejecutan como `python src/<script>.py`, asi que los modulos se
# importan por nombre plano (`import config`). Se replica eso para los tests.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROD_DIR = Path(__file__).resolve().parent.parent / "data" / "prod"


@pytest.fixture
def valid_features():
    """DataFrame de features que cumple el contrato de datos."""
    return pd.DataFrame({
        "age": [19, 34, 47, 62, 28],
        "sex": ["female", "male", "female", "male", "female"],
        "bmi": [27.9, 33.77, 23.164, 29.041, 25.519],
        "children": [0, 1, 3, 0, 2],
        "smoker": ["yes", "no", "no", "no", "yes"],
        "region": ["southwest", "southeast", "northwest", "northeast", "southeast"],
    })


@pytest.fixture
def bmi_without_decimal(valid_features):
    """Reproduce el defecto de dataset_prod3_feats: al bmi le sacaron el punto."""
    df = valid_features.copy()
    df["bmi"] = [27900.0, 33770.0, 23164.0, 29041.0, 25519.0]
    return df


@pytest.fixture
def reference_sample():
    """Muestra de referencia reproducible, para los tests de drift."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=30.0, scale=6.0, size=2000))


@pytest.fixture
def tmp_prod_dir(tmp_path):
    """Directorio de produccion sintetico con dos lotes.

    `prodA` tiene target y `prodB` no, para ejercitar el descubrimiento de lotes
    sin depender de los archivos reales del repo.
    """
    prod = tmp_path / "prod"
    prod.mkdir()
    feats = "age,sex,bmi,children,smoker,region\n30,male,25.5,1,no,southeast\n41,female,31.2,2,yes,northwest\n"
    (prod / "dataset_prodA_feats.csv.csv").write_text(feats)
    (prod / "dataset_prodA_target.csv.csv").write_text("charges\n4500,75\n21000,5\n")
    (prod / "dataset_prodB_feats.csv.csv").write_text(feats)
    (prod / "README.md").write_text("no es un lote")
    return prod
