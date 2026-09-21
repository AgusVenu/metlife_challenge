#!/usr/bin/env python
"""Registra en MLflow el modelo documentado en el README original, por unica vez.

Por que existe
--------------
El `README.md` del equipo de ciencia de datos publica un modelo con metricas concretas
(R2 = 0,8353 / RMSE = $4.835) y sus hiperparametros ganadores. Ese modelo es el punto
de partida del challenge, pero **no estaba en ningun lado**: el .pkl original no se
versiono y el repo no traia artefactos. Quedaba como una afirmacion en un documento.

Este script lo pone en el Model Registry como linea de base, de modo que la comparacion
"lo que habia" vs "lo que eligio el pipeline" se pueda hacer *dentro de MLflow* y no
leyendo dos documentos en paralelo.

Que es y que NO es
------------------
Es una REPRODUCCION, no el artefacto original. Lo que se conserva del original es:

    - los hiperparametros exactos que publica el README
    - el mismo split (test_size=0.2, random_state=43) -- verificado contra el commit
      inicial del repo, `git show decb3a1:src/training.py`
    - el mismo feature engineering y la misma transformacion log1p del target
    - las mismas semillas del estimador

Como el split y las semillas coinciden, la reproduccion es deterministica y sus
metricas se pueden comparar contra las que el README declara. El script imprime esa
comparacion: si no coinciden, lo dice en vez de disimularlo.

No se entrena una busqueda de hiperparametros: se usan directamente los ganadores que
el README reporta. Correr la busqueda entera solo agregaria varianza a una comparacion
que se quiere exacta.

Donde queda
-----------
En un modelo registrado APARTE (`insurance-charges-baseline-ds`), no como una version
del modelo que sirve en produccion. Es una referencia historica, no un candidato: si
viviera en la misma cadena de versiones, daria a entender que compite por el alias
`production`, y no compite.

Uso
---
    python scripts/register_readme_baseline.py
    python scripts/register_readme_baseline.py --dry-run     # no registra, solo compara
    python scripts/register_readme_baseline.py --force       # re-registra si ya existe
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mlflow                                                    # noqa: E402
import xgboost as xgb                                            # noqa: E402
from mlflow.exceptions import MlflowException                    # noqa: E402
from sklearn.pipeline import Pipeline                            # noqa: E402

import config                                                    # noqa: E402
import mlflow_utils                                              # noqa: E402
import monitoring                                                # noqa: E402
import training                                                  # noqa: E402
from utils import get_db_engine                                  # noqa: E402

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lo que declara el README original
# ============================================================================
# Hiperparametros: README.md, seccion "Hiperparametros Finales".
README_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.01,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
}

# Metricas: README.md, seccion "Resultados del Modelo". Se guardan para poder
# CONTRASTAR la reproduccion contra lo que el documento afirma.
README_METRICS = {
    "train": {"r2": 0.8806, "adj_r2": 0.8793, "rmse": 4199.0, "mae": 1814.0, "mape": 14.51},
    "validation": {"r2": 0.8353, "adj_r2": 0.8276, "rmse": 4835.0, "mae": 2102.0, "mape": 17.70},
}

BASELINE_MODEL_NAME = "insurance-charges-baseline-ds"
BASELINE_ALIAS = "reference"
STAGE_BASELINE = "baseline"


def build_readme_model() -> Pipeline:
    """El pipeline tal como lo describe el README: mismo preprocesador, sus params."""
    return Pipeline([
        ("preprocessor", training.create_preprocessor()),
        ("model", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            verbosity=0,
            **README_PARAMS,
        )),
    ])


def compare_against_readme(metrics: dict) -> bool:
    """Imprime reproducido vs. declarado. Devuelve True si todo cierra.

    La tolerancia es 1% relativo: el README redondea sus cifras (RMSE a dolares
    enteros, R2 a cuatro decimales), asi que exigir igualdad exacta seria exigirle al
    documento una precision que no tiene.
    """
    logger.info("=" * 78)
    logger.info("REPRODUCCION vs. LO QUE DECLARA EL README")
    logger.info("=" * 78)
    logger.info("  %-12s %16s %16s %12s", "", "README", "reproducido", "delta")

    todo_ok = True
    for split in ("train", "validation"):
        logger.info("  --- %s ---", split)
        for key in ("r2", "adj_r2", "rmse", "mae", "mape"):
            declarado = float(README_METRICS[split][key])
            obtenido = float(metrics[split][key])
            delta = obtenido - declarado
            relativo = abs(delta) / abs(declarado) if declarado else 0.0
            marca = "" if relativo <= 0.01 else "   <-- NO COINCIDE"
            if marca:
                todo_ok = False
            if key in ("rmse", "mae"):
                fila = f"  {key:<12} {declarado:>16,.2f} {obtenido:>16,.2f} {delta:>+12.2f}{marca}"
            else:
                fila = f"  {key:<12} {declarado:>16.4f} {obtenido:>16.4f} {delta:>+12.4f}{marca}"
            logger.info("%s", fila)

    logger.info("=" * 78)
    if todo_ok:
        logger.info("%s", "La reproduccion coincide con el README dentro del 1% relativo.")
    else:
        logger.warning(
            "La reproduccion NO coincide con alguna metrica del README. El modelo que se "
            "registra es el que produce ESTE codigo con los hiperparametros documentados; "
            "la discrepancia queda anotada en los tags de la version."
        )
    return todo_ok


def main(argv=None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="entrena y compara, pero no registra nada")
    parser.add_argument("--force", action="store_true",
                        help="registra otra version aunque la linea de base ya exista")
    args = parser.parse_args(argv)

    try:
        logger.info("=" * 78)
        logger.info("LINEA DE BASE DEL README — registro por unica vez")
        logger.info("=" * 78)
        logger.info("Hiperparametros declarados: %s", README_PARAMS)

        client = mlflow_utils.get_client()
        existente = mlflow_utils.get_version_by_alias(BASELINE_ALIAS, BASELINE_MODEL_NAME)
        if existente is not None and not args.force and not args.dry_run:
            logger.info(
                "La linea de base ya esta registrada: %s v%s (alias '%s'). "
                "Es por unica vez; usar --force para re-registrarla.",
                BASELINE_MODEL_NAME, existente.version, BASELINE_ALIAS,
            )
            return True

        # --- datos: mismo camino y mismo split que el pipeline actual -------------
        engine = get_db_engine()
        df = training.load_training_data(engine)
        X, y_log, y_orig = training.prepare_features_target(df)
        X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = training.split_data(
            X, y_log, y_orig
        )

        # --- entrenamiento sin busqueda: los params ya vienen dados ---------------
        model = build_readme_model()
        model.fit(X_train, y_train)

        metrics, y_val_pred = training.evaluate_model(
            model, X_train, y_train, X_val, y_val, y_train_orig, y_val_orig
        )
        coincide = compare_against_readme(metrics)

        if args.dry_run:
            logger.info("(--dry-run: no se registra nada)")
            return True

        # --- run de MLflow --------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlflow_utils.setup_tracking(config.MLFLOW_EXPERIMENT_TRAINING)

        with mlflow.start_run(run_name=f"baseline_readme_{timestamp}") as run:
            run_id = run.info.run_id

            mlflow.log_params({
                **README_PARAMS,
                "model_type": "XGBRegressor",
                "model_family": "xgboost",
                "target_transform": "log1p",
                "search_method": "ninguno (hiperparametros documentados en README.md)",
                "split_seed": config.SPLIT_SEED,
                "random_seed": config.RANDOM_SEED,
                "test_size": config.TEST_SIZE,
                "n_train_rows": len(X_train),
                "eval_dataset": "validation",
                "source": "README.md del equipo de ciencia de datos",
            })

            planas = {}
            for split, prefijo in (("train", "train_"), ("validation", "val_")):
                for key, value in metrics[split].items():
                    planas[f"{prefijo}{key}"] = float(value)
            planas["overfitting_r2_diff"] = float(metrics["overfitting_score"])
            # Delta contra lo declarado: deja medible la fidelidad de la reproduccion.
            for key, declarado in README_METRICS["validation"].items():
                planas[f"readme_delta_val_{key}"] = float(metrics["validation"][key] - declarado)
            mlflow.log_metrics(planas)

            # `baseline`, no `training`: es una referencia historica, no un candidato.
            # resolve_model() filtra por pipeline_stage='training', asi que este run
            # nunca puede terminar sirviendo en produccion por accidente.
            mlflow.set_tags({
                "pipeline_stage": STAGE_BASELINE,
                "model_family": "xgboost",
                "dataset": "training_dataset",
                "reproduccion_fiel": str(coincide).lower(),
                "nota": ("Reproduccion del modelo documentado en README.md a partir de sus "
                         "hiperparametros publicados. No es el artefacto original, que nunca "
                         "se versiono."),
            })

            signature = mlflow.models.infer_signature(X_train, model.predict(X_train.head(5)))
            mlflow.sklearn.log_model(
                model, name=config.MLFLOW_MODEL_ARTIFACT,
                signature=signature, input_example=X_train.head(5),
                serialization_format="cloudpickle",
            )

            # Baseline de monitoreo, para que la version sea autosuficiente si alguna
            # vez alguien la resuelve a mano.
            monitoring_baseline = monitoring.build_baseline(
                X_train=X_train, y_train_original=y_train_orig,
                val_metrics={k: metrics["validation"][k] for k in ("rmse", "mae", "r2", "mape")},
                val_predictions=y_val_pred,
                extra={"training_run_id": run_id, "source": "README.md"},
            )
            mlflow.log_dict(monitoring_baseline, config.BASELINE_ARTIFACT)

        # --- registry: modelo aparte, con alias de referencia ----------------------
        try:
            client.create_registered_model(
                BASELINE_MODEL_NAME,
                description=("Linea de base historica: el modelo documentado en el README.md "
                             "del equipo de ciencia de datos, reproducido desde sus "
                             "hiperparametros publicados. NO sirve en produccion."),
            )
        except MlflowException:
            pass  # ya existia

        version = mlflow.register_model(
            f"runs:/{run_id}/{config.MLFLOW_MODEL_ARTIFACT}", BASELINE_MODEL_NAME
        )
        for key, value in {
            "stage": "Baseline",
            "model_family": "xgboost",
            "source": "README.md",
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "reproduccion_fiel": str(coincide).lower(),
            "metric.val_rmse": f"{metrics['validation']['rmse']:.6f}",
            "metric.val_r2": f"{metrics['validation']['r2']:.6f}",
            "readme_val_rmse": f"{README_METRICS['validation']['rmse']:.2f}",
            "readme_val_r2": f"{README_METRICS['validation']['r2']:.4f}",
        }.items():
            client.set_model_version_tag(BASELINE_MODEL_NAME, version.version, key, value)

        client.set_registered_model_alias(BASELINE_MODEL_NAME, BASELINE_ALIAS, version.version)

        logger.info("=" * 78)
        logger.info("LINEA DE BASE REGISTRADA")
        logger.info("=" * 78)
        logger.info("  Modelo:   %s v%s (alias '%s')",
                    BASELINE_MODEL_NAME, version.version, BASELINE_ALIAS)
        logger.info("  Run:      %s  (pipeline_stage=%s)", run_id, STAGE_BASELINE)
        logger.info("  val_rmse: $%s   val_r2: %.4f",
                    f"{metrics['validation']['rmse']:,.2f}", metrics["validation"]["r2"])
        logger.info("")
        logger.info("  Comparar en la UI:  tags.model_family = 'xgboost'")
        return True

    except Exception as exc:
        logger.error("Error registrando la linea de base: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
