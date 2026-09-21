"""Catalogo de familias de modelos a comparar en el entrenamiento.

Por que un modulo aparte
------------------------
El catalogo es una tabla de datos, no logica: conviene poder leerlo de un vistazo y
testearlo sin levantar MLflow ni PostgreSQL. `training.py` ya tiene el pipeline; aca
vive el "que" y alla el "como".

Invariante que no se puede romper
---------------------------------
Toda familia se entrena como `Pipeline([("preprocessor", ...), ("model", ...)])` con
EXACTAMENTE esos dos nombres de step. De eso dependen `utils.get_encoded_feature_names`,
`utils.count_encoded_features` (y por lo tanto el R2 ajustado que calcula scoring) y
`training.compute_feature_importance`. Las claves de los grids llevan el prefijo
`model__` por la misma razon.

Sobre el paralelismo
--------------------
Las factories fijan `n_jobs=1` en el estimador a proposito. El paralelismo vive en
`RandomizedSearchCV(n_jobs=-1)`: si ademas el estimador pide todos los cores, los
procesos de la validacion cruzada compiten entre si por los mismos cores y el wall-clock
empeora. `HistGradientBoostingRegressor` no acepta `n_jobs` -- su paralelismo es OpenMP y
se controla con la variable de entorno OMP_NUM_THREADS.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

import config

logger = logging.getLogger(__name__)


# ============================================================================
# Especificacion de una familia
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    """Todo lo que el entrenamiento necesita saber de una familia de modelos."""

    name: str                          # clave estable; va al tag `model_family`
    label: str                         # nombre legible, para reportes
    build: Callable[[], Any]           # factory del estimador, ya configurado
    param_grid: Dict[str, list]        # claves con prefijo model__
    rationale: str                     # por que esta familia esta en la comparacion
    needs_scaling: bool = False        # True -> el bloque numerico usa StandardScaler
    n_iter_ratio: float = 1.0          # fraccion del presupuesto global que consume

    def grid_size(self) -> int:
        """Cantidad de combinaciones distintas que tiene el grid."""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

    def n_iter(self, budget: int = None) -> int:
        """Iteraciones efectivas de la busqueda para esta familia.

        Se topea por la cardinalidad del grid: sin el tope, una familia con 20
        combinaciones posibles recibiria 50 iteraciones. `ParameterSampler` las recorta
        igual, pero emite un UserWarning y deja un numero que no es el real en
        `search.n_iter`, que despues termina en el reporte y en MLflow.
        """
        budget = config.HYPERPARAM_ITERATIONS if budget is None else budget
        wanted = max(1, round(budget * self.n_iter_ratio))
        return min(wanted, self.grid_size())


# ============================================================================
# El catalogo
# ============================================================================
# El ORDEN importa dos veces: es el orden en que se entrenan las familias y es el
# criterio de desempate cuando dos empatan en la metrica de seleccion. XGBoost va
# primero porque es el incumbente, asi que un empate lo conserva.

def _xgboost():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        verbosity=0,
    )


def _random_forest():
    return RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=1)


def _hist_gradient_boosting():
    # Sin n_jobs: paraleliza con OpenMP (OMP_NUM_THREADS), no con joblib.
    return HistGradientBoostingRegressor(random_state=config.RANDOM_SEED)


def _elasticnet():
    # max_iter alto para que el descenso por coordenadas converja con los alphas
    # chicos del grid y no ensucie la salida con ConvergenceWarning.
    return ElasticNet(random_state=config.RANDOM_SEED, max_iter=10_000)


MODEL_SPECS: "OrderedDict[str, ModelSpec]" = OrderedDict()

for _spec in (
    ModelSpec(
        name="xgboost",
        label="XGBoost",
        build=_xgboost,
        rationale="Incumbente: el modelo que el equipo de datos ya venia usando. Es la "
                  "referencia a batir, no una opcion mas.",
        param_grid={
            # Grid original del proyecto, sin tocar, para que el run de esta familia
            # siga siendo comparable con los resultados ya documentados.
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__reg_alpha": [0, 0.1, 1],        # L1
            "model__reg_lambda": [1, 10, 100],      # L2
        },
        n_iter_ratio=1.0,
    ),
    ModelSpec(
        name="random_forest",
        label="Random Forest",
        build=_random_forest,
        rationale="Bagging en vez de boosting: otro perfil de sesgo/varianza. Si gana, "
                  "el problema tiene menos senal secuencial de la que se suponia.",
        param_grid={
            "model__n_estimators": [100, 200, 400, 800],
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt", 0.5],
        },
        n_iter_ratio=0.4,
    ),
    ModelSpec(
        name="hist_gradient_boosting",
        label="HistGradientBoosting (sklearn)",
        build=_hist_gradient_boosting,
        rationale="Otra implementacion de boosting, con binning de histogramas. Separa "
                  "'el boosting funciona' de 'la implementacion de XGBoost funciona'. "
                  "Cumple el rol de LightGBM sin agregar una dependencia.",
        param_grid={
            "model__max_iter": [100, 200, 400],
            "model__max_depth": [None, 3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__min_samples_leaf": [5, 10, 20],
        },
        n_iter_ratio=0.4,
    ),
    ModelSpec(
        name="elasticnet",
        label="ElasticNet (lineal regularizado)",
        build=_elasticnet,
        rationale="El piso lineal. Sin un baseline lineal no se puede afirmar que la "
                  "complejidad no lineal aporta algo: solo se puede suponerlo.",
        param_grid={
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        # Necesita escalado: sin el, la penalizacion L1/L2 castiga desparejo a features
        # que conviven en escalas muy distintas (age_squared llega a ~10.000 y children
        # a 5), y la evaluacion de la familia lineal dejaria de ser honesta.
        needs_scaling=True,
        n_iter_ratio=0.4,
    ),
):
    MODEL_SPECS[_spec.name] = _spec

del _spec


# ============================================================================
# Acceso
# ============================================================================

def get_specs(names: Sequence[str] = None) -> List[ModelSpec]:
    """Devuelve las specs pedidas, en el orden del catalogo.

    Falla fuerte con un nombre desconocido en vez de ignorarlo: un typo en
    TRAIN_MODEL_FAMILIES que se traga en silencio significa entrenar menos familias de
    las pedidas y no enterarse hasta leer el reporte.
    """
    names = list(config.TRAIN_MODEL_FAMILIES if names is None else names)
    unknown = [n for n in names if n not in MODEL_SPECS]
    if unknown:
        raise ValueError(
            f"Familia(s) de modelo desconocida(s): {', '.join(unknown)}. "
            f"Disponibles: {', '.join(MODEL_SPECS)}"
        )
    if not names:
        raise ValueError("No se especifico ninguna familia de modelos para entrenar.")
    wanted = set(names)
    return [spec for spec in MODEL_SPECS.values() if spec.name in wanted]


def describe_catalog(specs: Sequence[ModelSpec] = None) -> str:
    """Resumen del catalogo efectivo, para loguear al arrancar el entrenamiento."""
    specs = get_specs() if specs is None else specs
    lines = []
    for spec in specs:
        lines.append(
            f"  {spec.label:<34} grid={spec.grid_size():>4} combinaciones  "
            f"n_iter={spec.n_iter():>3}"
            + ("  (con escalado)" if spec.needs_scaling else "")
        )
    return "\n".join(lines)
