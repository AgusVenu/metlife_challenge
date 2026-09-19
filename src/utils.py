import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

"""Script de utilidades para el proyecto."""

def get_db_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL.

    La configuracion se lee de src/config.py, que centraliza las variables de
    entorno, en vez de releer os.environ en cada modulo.
    """
    import config
    return create_engine(config.get_db_url())

def feature_engineering(X, is_training=True):
    """Feature engineering aplicable a training y a scoring
    
    Args:
        X (pd.DataFrame): DataFrame con las features originales
        is_training (bool): Indica si se incluyen prints de logging
    Returns:
        pd.DataFrame: DataFrame con las features transformadas
    """
    
    X=X.copy()
    
    #Variables nuevas
    X['bmi_smoker'] = X['bmi'] * (X['smoker'] == 'yes').astype(int)
    X['age_smoker'] = X['age'] * (X['smoker'] == 'yes').astype(int)
    
    #Variables cuadraticas
    X['bmi_squared'] = X['bmi'] ** 2
    X['age_squared'] = X['age'] ** 2
    
    #Variables binarias
    X['bmi_obese'] = (X['bmi'] > 30).astype(int)
    X['age_senior'] = (X['age'] > 50).astype(int)
    
    if is_training:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Feature engineering aplicado:")
        logger.info(f"  - Nuevas features: bmi_smoker, age_smoker, bmi_squared, age_squared, bmi_obese, age_senior")
        logger.info(f"  - Shape resultante: {X.shape}")
    
    return X

def transform_target(y, inverse=False):
    """Transforma target (charges) usando log
    
    Args:
        y: array-like con valores de charges
        inverse: si True, aplica la transformación inversa (exp)
    Returns:
        array-like con valores transformados
    """
    
    if inverse:
        # Volver a escala original: exp(log_charges) - 1
        return np.expm1(y)  # expm1(x) = exp(x) - 1
    else:
        # Aplicar log: log(charges + 1)
        return np.log1p(y)  # log1p(x) = log(1 + x)


def get_encoded_feature_names(model):
    """Nombres de las features DESPUES del ColumnTransformer (post one-hot).

    La usan training y scoring para calcular el R2 ajustado con el ancho real
    de la matriz que ve el modelo, y no con el numero de columnas de entrada.
    """
    try:
        return list(model.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        return []


def count_encoded_features(model, fallback=None):
    """Cantidad de features codificadas, con fallback si el modelo no lo expone."""
    names = get_encoded_feature_names(model)
    return len(names) if names else fallback
