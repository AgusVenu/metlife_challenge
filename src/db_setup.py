"""Setup de la base de datos: esquema y carga del dataset de entrenamiento.

Tablas
------
    training_dataset  - dataset historico que consume el entrenamiento
    predictions       - scoring legacy sobre muestra aleatoria (SCORING_MODE=sample)
    batch_predictions - predicciones fila a fila de los lotes de data/prod/
    batch_monitoring  - una fila por lote scoreado, con metricas y estado
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

import config
from utils import get_db_engine

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def drop_tables(engine):
    """Elimina las tablas para arrancar de cero."""
    logger.info("Eliminando tablas existentes (si existen)...")
    with engine.connect() as conn:
        for table in ("batch_monitoring", "batch_predictions", "predictions",
                      "scoring_dataset", "training_dataset"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
        conn.commit()
    logger.info("Tablas eliminadas.")


def create_tables(engine):
    """Crea el esquema completo."""
    logger.info("Creando tablas...")

    statements = {
        "training_dataset": """
            CREATE TABLE IF NOT EXISTS training_dataset (
                id SERIAL PRIMARY KEY,
                age INTEGER NOT NULL,
                sex VARCHAR(10) NOT NULL,
                bmi FLOAT NOT NULL,
                children INTEGER NOT NULL,
                smoker VARCHAR(5) NOT NULL,
                region VARCHAR(20) NOT NULL,
                charges FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        # Scoring legacy: muestra aleatoria del propio training_dataset.
        "predictions": """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                scoring_id INTEGER,
                age INTEGER NOT NULL,
                sex VARCHAR(10) NOT NULL,
                bmi FLOAT NOT NULL,
                children INTEGER NOT NULL,
                smoker VARCHAR(5) NOT NULL,
                region VARCHAR(20) NOT NULL,
                actual_charges FLOAT,
                predicted_charges FLOAT,
                absolute_error FLOAT,
                percentage_error FLOAT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        # Scoring batch sobre data/prod/. actual_charges es NULL para los lotes
        # que llegan sin ground truth (el caso de prod3).
        "batch_predictions": """
            CREATE TABLE IF NOT EXISTS batch_predictions (
                id SERIAL PRIMARY KEY,
                batch_id VARCHAR(50) NOT NULL,
                row_index INTEGER NOT NULL,
                age INTEGER,
                sex VARCHAR(10),
                bmi FLOAT,
                children INTEGER,
                smoker VARCHAR(5),
                region VARCHAR(20),
                predicted_charges FLOAT NOT NULL,
                actual_charges FLOAT,
                absolute_error FLOAT,
                percentage_error FLOAT,
                model_name VARCHAR(100),
                model_version VARCHAR(20),
                model_source VARCHAR(50),
                mlflow_run_id VARCHAR(64),
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        # Una fila por lote scoreado: el historial de monitoreo consultable.
        "batch_monitoring": """
            CREATE TABLE IF NOT EXISTS batch_monitoring (
                id SERIAL PRIMARY KEY,
                batch_id VARCHAR(50) NOT NULL,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                n_rows INTEGER NOT NULL,
                has_target BOOLEAN NOT NULL,
                status VARCHAR(10) NOT NULL,
                rmse FLOAT,
                mae FLOAT,
                r2 FLOAT,
                mape FLOAT,
                baseline_rmse FLOAT,
                baseline_r2 FLOAT,
                psi_max FLOAT,
                psi_max_feature VARCHAR(50),
                n_violations INTEGER,
                pred_mean FLOAT,
                diagnosis TEXT,
                model_name VARCHAR(100),
                model_version VARCHAR(20),
                model_source VARCHAR(50),
                mlflow_run_id VARCHAR(64),
                details JSONB
            )
        """,
    }

    with engine.connect() as conn:
        for table, statement in statements.items():
            conn.execute(text(statement))
            logger.info("  Tabla '%s' lista.", table)
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_batch_predictions_batch "
            "ON batch_predictions (batch_id, scored_at)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_batch_monitoring_batch "
            "ON batch_monitoring (batch_id, scored_at)"
        ))
        conn.commit()


def load_dataset(engine, csv_path=None):
    """Carga el CSV de entrenamiento en training_dataset, validando antes."""
    csv_path = Path(csv_path or config.TRAINING_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")

    logger.info("Cargando dataset desde %s ...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Dataset cargado: %d filas, %d columnas.", df.shape[0], df.shape[1])

    expected = config.RAW_FEATURES + [config.TARGET_COLUMN]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Valores nulos encontrados:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("No se encontraron valores nulos.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        logger.warning("Se encontraron %d filas duplicadas; se eliminan.", duplicate_count)
        df = df.drop_duplicates()
        logger.info("Nuevo tamano del dataset: %d filas.", df.shape[0])
    else:
        logger.info("No se encontraron filas duplicadas.")

    df[expected].to_sql("training_dataset", con=engine, if_exists="append", index=False)
    logger.info("%d filas insertadas en 'training_dataset'.", len(df))

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM training_dataset;")).scalar()
        logger.info("Total de filas en 'training_dataset': %d", count)


def main():
    try:
        logger.info("Iniciando setup de la base de datos...")
        logger.info("Destino: %s", config.get_db_url_safe())
        engine = get_db_engine()
        drop_tables(engine)
        create_tables(engine)
        load_dataset(engine)
        logger.info("Setup de la base de datos completado exitosamente.")
        return True
    except Exception as exc:
        logger.error("Error durante el setup de la base de datos: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
