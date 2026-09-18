"""Lectura robusta y validacion de los lotes de produccion.

Por que existe este modulo
--------------------------
Los archivos de `data/prod/` no se pueden leer con un `pd.read_csv()` directo.
`dataset_prod1_target.csv.csv` usa coma decimal:

    charges
    14700,80931
    1540,261607

Con el separador por defecto (`,`), pandas ve DOS campos por fila pero un solo
nombre de columna, asi que usa el primero como indice y el segundo como valor.
El resultado es un DataFrame perfectamente valido y completamente equivocado:
`charges` pasa a valer 80931 en vez de 14700.80931, sin una sola excepcion ni
warning. Ese es exactamente el tipo de fallo silencioso que un pipeline de
produccion tiene que hacer imposible.

La solucion es no dejar que pandas parta la linea: se lee el archivo como texto
y se normaliza el separador decimal de forma explicita y testeable.
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ============================================================================
# Normalizacion de numeros escritos con distintas convenciones
# ============================================================================

def normalize_decimal_string(raw: str) -> Optional[str]:
    """Normaliza un numero escrito en cualquier convencion a formato Python.

    Casos soportados:
        "14700.80931"  -> "14700.80931"   (ya esta bien)
        "14700,80931"  -> "14700.80931"   (coma decimal, el caso de data/prod)
        "1.234,56"     -> "1234.56"       (punto de miles + coma decimal)
        "1,234.56"     -> "1234.56"       (coma de miles + punto decimal)
        "1,234,567"    -> "1234567"       (comas de miles)
        ""             -> None            (vacio)

    Devuelve None si el valor esta vacio, para que el caller lo trate como nulo.
    """
    if raw is None:
        return None
    text = str(raw).strip().strip('"').strip("'")
    if text == "" or text.lower() in ("nan", "null", "none", "na"):
        return None

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        # Convive con ambos: el ULTIMO separador que aparece es el decimal.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")   # 1.234,56
        else:
            text = text.replace(",", "")                     # 1,234.56
    elif has_comma:
        # Solo comas. Una sola coma es decimal; varias son separadores de miles.
        if text.count(",") == 1:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    # Solo punto (o ningun separador): ya es formato Python.

    return text


def _to_float_series(values: List[Optional[str]], source: str) -> pd.Series:
    """Convierte una lista de strings normalizados a float, avisando de los fallos."""
    parsed, failures = [], 0
    for value in values:
        if value is None:
            parsed.append(np.nan)
            continue
        try:
            parsed.append(float(value))
        except ValueError:
            parsed.append(np.nan)
            failures += 1
    if failures:
        logger.warning("%s: %d valores no se pudieron convertir a numero", source, failures)
    return pd.Series(parsed, dtype="float64")


def detect_decimal_convention(sample_values: List[str]) -> str:
    """Identifica la convencion decimal de una muestra, solo para reportarla."""
    joined = " ".join(str(v) for v in sample_values[:50])
    if "," in joined and "." in joined:
        return "mixta (punto y coma)"
    if "," in joined:
        return "coma decimal (es-ES / pt-BR)"
    if "." in joined:
        return "punto decimal (en-US)"
    return "enteros sin separador"


# ============================================================================
# Lectura de archivos
# ============================================================================

def read_single_column_numeric(path: Path, column_name: str = None) -> pd.Series:
    """Lee un CSV de una sola columna numerica sin dejar que pandas parta por comas.

    Se lee el archivo como texto plano justamente para que la coma decimal no
    se interprete como separador de campos.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        raise ValueError(f"Archivo vacio: {path}")

    header = lines[0].strip()
    body = lines[1:]

    # Si la primera linea no es numerica, es el encabezado.
    if normalize_decimal_string(header) is not None:
        try:
            float(normalize_decimal_string(header))
            body = lines            # no habia encabezado
            header = column_name or "value"
        except ValueError:
            pass

    name = column_name or header or "value"
    convention = detect_decimal_convention(body)
    logger.info("  %s: %d filas, convencion detectada: %s", path.name, len(body), convention)

    normalized = [normalize_decimal_string(line) for line in body]
    series = _to_float_series(normalized, path.name)
    series.name = name
    return series


def read_features_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV de features y garantiza que las columnas numericas sean numericas.

    Si alguna columna numerica llega como texto (por ejemplo por coma decimal),
    se normaliza con las mismas reglas que el target en vez de fallar.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    for column, spec in config.FEATURE_SPEC.items():
        if column not in df.columns or spec["kind"] != "numeric":
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        logger.warning(
            "  %s: columna '%s' llego como texto; se normaliza el separador decimal",
            path.name, column,
        )
        normalized = [normalize_decimal_string(v) for v in df[column]]
        df[column] = _to_float_series(normalized, f"{path.name}:{column}")

    for column in config.CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.lower()

    logger.info("  %s: %d filas x %d columnas", path.name, df.shape[0], df.shape[1])
    return df


# ============================================================================
# Descubrimiento de lotes
# ============================================================================

@dataclass
class Batch:
    """Un lote de produccion: features y, opcionalmente, su ground truth."""
    batch_id: str
    features_path: Path
    target_path: Optional[Path] = None

    @property
    def has_target(self) -> bool:
        return self.target_path is not None

    def __str__(self) -> str:
        kind = "con target" if self.has_target else "SIN target"
        return f"{self.batch_id} ({kind})"


# Tolera la doble extension `.csv.csv` que traen los archivos del challenge.
_FEATS_PATTERN = re.compile(r"^dataset_(?P<batch>.+?)_feats\.csv(\.csv)?$", re.IGNORECASE)


def discover_batches(prod_dir: Path = None) -> List[Batch]:
    """Descubre los lotes de `data/prod/` por convencion de nombre.

    Se descubre en vez de hardcodear la lista para que agregar un
    `dataset_prod4_feats.csv` no requiera tocar codigo. Para cada archivo de
    features se busca su `_target` hermano; si no existe, el lote se procesa
    como batch sin etiquetas (el caso de prod3).
    """
    prod_dir = Path(prod_dir) if prod_dir else config.PROD_DATA_DIR
    if not prod_dir.exists():
        raise FileNotFoundError(f"Directorio de produccion no encontrado: {prod_dir}")

    batches = []
    for entry in sorted(prod_dir.iterdir()):
        match = _FEATS_PATTERN.match(entry.name)
        if not match:
            continue
        batch_id = match.group("batch")

        target_path = None
        for candidate in (
            entry.with_name(entry.name.replace("_feats", "_target")),
            prod_dir / f"dataset_{batch_id}_target.csv.csv",
            prod_dir / f"dataset_{batch_id}_target.csv",
        ):
            if candidate.exists():
                target_path = candidate
                break

        batches.append(Batch(batch_id=batch_id, features_path=entry, target_path=target_path))

    if not batches:
        raise FileNotFoundError(
            f"No se encontraron archivos 'dataset_*_feats.csv*' en {prod_dir}"
        )
    return batches


# ============================================================================
# Validacion contra el contrato de datos
# ============================================================================

@dataclass
class Violation:
    """Una violacion del contrato de datos, con su severidad ya resuelta."""
    column: str
    kind: str          # missing_column | nulls | out_of_range | unknown_category | not_numeric
    severity: str      # WARNING | ALERT
    n_rows: int
    pct_rows: float
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.column}: {self.detail} ({self.n_rows} filas, {self.pct_rows:.1f}%)"


def _severity_for_pct(pct: float) -> str:
    if pct > config.SCHEMA_ALERT_PCT:
        return config.STATUS_ALERT
    if pct > config.SCHEMA_WARN_PCT:
        return config.STATUS_WARNING
    return config.STATUS_OK


def validate_features(df: pd.DataFrame, spec: dict = None) -> List[Violation]:
    """Valida un DataFrame de features contra el contrato de `config.FEATURE_SPEC`.

    Es la deteccion que hace saltar `dataset_prod3_feats.csv.csv`, donde el bmi
    perdio el punto decimal (27.929 -> 27929) y queda fuera de todo rango
    fisiologicamente posible.
    """
    spec = spec if spec is not None else config.FEATURE_SPEC
    violations: List[Violation] = []
    n_total = max(len(df), 1)

    for column, rules in spec.items():
        if column not in df.columns:
            violations.append(Violation(
                column=column, kind="missing_column", severity=config.STATUS_ALERT,
                n_rows=len(df), pct_rows=100.0,
                detail="columna ausente en el archivo",
            ))
            continue

        series = df[column]

        n_null = int(series.isna().sum())
        if n_null:
            pct = 100.0 * n_null / n_total
            violations.append(Violation(
                column=column, kind="nulls", severity=_severity_for_pct(pct),
                n_rows=n_null, pct_rows=pct, detail="valores nulos",
            ))

        if rules["kind"] == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                violations.append(Violation(
                    column=column, kind="not_numeric", severity=config.STATUS_ALERT,
                    n_rows=len(df), pct_rows=100.0,
                    detail=f"se esperaba numerica y llego como {series.dtype}",
                ))
                continue

            valid = series.dropna()
            out_of_range = valid[(valid < rules["min"]) | (valid > rules["max"])]
            if len(out_of_range):
                pct = 100.0 * len(out_of_range) / n_total
                violations.append(Violation(
                    column=column, kind="out_of_range", severity=_severity_for_pct(pct),
                    n_rows=len(out_of_range), pct_rows=pct,
                    detail=(
                        f"fuera del rango esperado [{rules['min']}, {rules['max']}] "
                        f"- observado [{valid.min():.4g}, {valid.max():.4g}]"
                    ),
                ))
        else:
            allowed = {str(v).lower() for v in rules["allowed"]}
            observed = series.dropna().astype(str).str.lower()
            unknown = observed[~observed.isin(allowed)]
            if len(unknown):
                pct = 100.0 * len(unknown) / n_total
                sample = sorted(unknown.unique())[:5]
                violations.append(Violation(
                    column=column, kind="unknown_category", severity=_severity_for_pct(pct),
                    n_rows=len(unknown), pct_rows=pct,
                    detail=f"categorias no vistas en training: {sample}",
                ))

    return violations


def validate_target(series: pd.Series, spec: dict = None) -> List[Violation]:
    """Valida el target contra el rango plausible de negocio.

    Es la deteccion que hace saltar `dataset_prod2_target.csv.csv`, cuyos
    valores estan multiplicados por 100 (media 1.327.624 contra 13.270 en
    training) y caen enteros fuera del rango de una poliza real.
    """
    spec = spec if spec is not None else config.TARGET_SPEC
    violations: List[Violation] = []
    n_total = max(len(series), 1)

    n_null = int(series.isna().sum())
    if n_null:
        pct = 100.0 * n_null / n_total
        violations.append(Violation(
            column=config.TARGET_COLUMN, kind="nulls", severity=_severity_for_pct(pct),
            n_rows=n_null, pct_rows=pct, detail="valores nulos en el target",
        ))

    valid = series.dropna()
    if len(valid):
        out_of_range = valid[(valid < spec["min"]) | (valid > spec["max"])]
        if len(out_of_range):
            pct = 100.0 * len(out_of_range) / n_total
            violations.append(Violation(
                column=config.TARGET_COLUMN, kind="out_of_range",
                severity=_severity_for_pct(pct),
                n_rows=len(out_of_range), pct_rows=pct,
                detail=(
                    f"fuera del rango esperado [{spec['min']:,.0f}, {spec['max']:,.0f}] "
                    f"- observado [{valid.min():,.2f}, {valid.max():,.2f}]"
                ),
            ))

    return violations


def worst_severity(violations: List[Violation]) -> str:
    """Devuelve la severidad mas alta de una lista de violaciones."""
    worst = config.STATUS_OK
    for violation in violations:
        if config.STATUS_ORDER.index(violation.severity) > config.STATUS_ORDER.index(worst):
            worst = violation.severity
    return worst


# ============================================================================
# Carga completa de un lote
# ============================================================================

@dataclass
class LoadedBatch:
    """Resultado de cargar y validar un lote, listo para scoring."""
    batch_id: str
    features: pd.DataFrame
    target: Optional[pd.Series]
    feature_violations: List[Violation] = field(default_factory=list)
    target_violations: List[Violation] = field(default_factory=list)
    features_path: str = ""
    target_path: Optional[str] = None

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def n_rows(self) -> int:
        return len(self.features)

    @property
    def all_violations(self) -> List[Violation]:
        return self.feature_violations + self.target_violations

    @property
    def schema_status(self) -> str:
        return worst_severity(self.all_violations)


def load_batch(batch: Batch) -> LoadedBatch:
    """Carga un lote completo: lectura robusta + validacion de contrato."""
    logger.info("Cargando lote %s", batch)

    features = read_features_csv(batch.features_path)
    feature_violations = validate_features(features)

    target = None
    target_violations: List[Violation] = []
    if batch.has_target:
        target = read_single_column_numeric(batch.target_path, config.TARGET_COLUMN)
        if len(target) != len(features):
            target_violations.append(Violation(
                column=config.TARGET_COLUMN, kind="row_count_mismatch",
                severity=config.STATUS_ALERT,
                n_rows=abs(len(target) - len(features)), pct_rows=100.0,
                detail=f"features tiene {len(features)} filas y target {len(target)}",
            ))
            n = min(len(target), len(features))
            features, target = features.iloc[:n].copy(), target.iloc[:n].copy()
        target = target.reset_index(drop=True)
        target_violations += validate_target(target)
    else:
        logger.info("  %s no tiene archivo de target: se procesa como batch sin etiquetas", batch.batch_id)

    loaded = LoadedBatch(
        batch_id=batch.batch_id,
        features=features.reset_index(drop=True),
        target=target,
        feature_violations=feature_violations,
        target_violations=target_violations,
        features_path=str(batch.features_path),
        target_path=str(batch.target_path) if batch.has_target else None,
    )

    if loaded.all_violations:
        logger.warning("  %s: %d violacion(es) del contrato de datos:",
                       batch.batch_id, len(loaded.all_violations))
        for violation in loaded.all_violations:
            logger.warning("    %s", violation)
    else:
        logger.info("  %s: contrato de datos OK", batch.batch_id)

    return loaded
