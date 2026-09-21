"""Promocion de modelos basada en metricas (bonus del challenge).

Por que la promocion es un paso aparte
--------------------------------------
`training.py` registra SIEMPRE una version nueva y la deja en `staging`. Si el
entrenamiento promoviera solo, cualquier corrida experimental pasaria a servir
en produccion sin control. Aca se decide, comparando contra el modelo que hoy
esta sirviendo, y la decision queda explicada en el log y en los tags de la
version.

Gates que se aplican
--------------------
    Absolutos (siempre):
        val_r2               >= PROMOTION_MIN_R2
        overfitting_r2_diff  <  PROMOTION_MAX_OVERFITTING
    Relativo (solo si ya hay un modelo en produccion):
        val_rmse <= rmse_produccion * (1 - PROMOTION_MIN_IMPROVEMENT)

Uso
---
    python src/promote_model.py                 # evalua la version en staging
    python src/promote_model.py --version 3     # evalua una version puntual
    python src/promote_model.py --dry-run       # explica la decision sin aplicarla
"""

import argparse
import logging
import sys
from datetime import datetime

import mlflow
from mlflow.exceptions import MlflowException

import config
import mlflow_utils

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_candidate(client, version_number=None):
    """Elige la version a evaluar: la indicada, la de `staging`, o la ultima."""
    if version_number:
        return client.get_model_version(config.MLFLOW_MODEL_NAME, str(version_number))

    staged = mlflow_utils.get_version_by_alias(config.ALIAS_STAGING)
    if staged is not None:
        return staged

    versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def evaluate_gates(candidate_metrics, production_metrics):
    """Aplica los gates y devuelve (promover, lista_de_razones)."""
    reasons, passed = [], True

    r2 = candidate_metrics.get("val_r2")
    if r2 is None:
        reasons.append("RECHAZO: la version no tiene la metrica val_r2 registrada")
        passed = False
    elif r2 < config.PROMOTION_MIN_R2:
        reasons.append(f"RECHAZO: val_r2={r2:.4f} < minimo requerido {config.PROMOTION_MIN_R2}")
        passed = False
    else:
        reasons.append(f"OK: val_r2={r2:.4f} >= {config.PROMOTION_MIN_R2}")

    overfitting = candidate_metrics.get("overfitting_r2_diff")
    if overfitting is None:
        reasons.append("AVISO: sin metrica overfitting_r2_diff; no se puede validar el gate")
    elif overfitting >= config.PROMOTION_MAX_OVERFITTING:
        reasons.append(
            f"RECHAZO: overfitting={overfitting:.4f} >= maximo tolerado "
            f"{config.PROMOTION_MAX_OVERFITTING}"
        )
        passed = False
    else:
        reasons.append(f"OK: overfitting={overfitting:.4f} < {config.PROMOTION_MAX_OVERFITTING}")

    candidate_rmse = candidate_metrics.get("val_rmse")
    production_rmse = (production_metrics or {}).get("val_rmse")

    if production_rmse is None:
        reasons.append("OK: no hay modelo en produccion; alcanza con pasar los gates absolutos")
    elif candidate_rmse is None:
        reasons.append("RECHAZO: la version no tiene la metrica val_rmse registrada")
        passed = False
    else:
        threshold = production_rmse * (1 - config.PROMOTION_MIN_IMPROVEMENT)
        if candidate_rmse <= threshold:
            reasons.append(
                f"OK: val_rmse={candidate_rmse:,.2f} <= umbral {threshold:,.2f} "
                f"(produccion={production_rmse:,.2f})"
            )
        else:
            reasons.append(
                f"RECHAZO: val_rmse={candidate_rmse:,.2f} > umbral {threshold:,.2f} "
                f"(produccion={production_rmse:,.2f}); el candidato no mejora al modelo actual"
            )
            passed = False

    return passed, reasons


def promote(version, reason_text, client, previous=None):
    """Aplica el alias `production` y deja registrada la traza de la decision.

    Ademas archiva la version que estaba sirviendo y saca a la nueva de
    `staging`. El alias es la fuente de verdad, pero si los tags `stage` no se
    mantienen al dia el registry termina mostrando dos versiones marcadas como
    Production, que es justo la confusion que el tag venia a evitar.
    """
    name = config.MLFLOW_MODEL_NAME

    if previous is not None and str(previous.version) != str(version.version):
        client.set_model_version_tag(name, previous.version, "stage", "Archived")
        client.set_model_version_tag(name, previous.version, "archived_at",
                                     datetime.now().isoformat(timespec="seconds"))
        client.set_model_version_tag(name, previous.version, "superseded_by", str(version.version))
        logger.info("Version %s archivada (reemplazada por la v%s).",
                    previous.version, version.version)

    client.set_registered_model_alias(name, config.ALIAS_PRODUCTION, version.version)
    client.set_model_version_tag(name, version.version, "stage", "Production")
    client.set_model_version_tag(name, version.version, "promoted_at",
                                 datetime.now().isoformat(timespec="seconds"))
    client.set_model_version_tag(name, version.version, "promoted_by", "promote_model.py")
    client.set_model_version_tag(name, version.version, "promotion_reason", reason_text[:480])

    # La version graduo: deja de ser candidata en staging.
    staged = mlflow_utils.get_version_by_alias(config.ALIAS_STAGING)
    if staged is not None and str(staged.version) == str(version.version):
        try:
            client.delete_registered_model_alias(name, config.ALIAS_STAGING)
        except MlflowException as exc:
            logger.warning("No se pudo limpiar el alias '%s': %s", config.ALIAS_STAGING, exc)

    logger.info("Version %s promovida: alias '%s' aplicado.", version.version, config.ALIAS_PRODUCTION)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Promueve un modelo a produccion segun sus metricas.")
    parser.add_argument("--version", type=int, default=None,
                        help="Version a evaluar (por defecto, la que tiene el alias staging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Explica la decision sin aplicar cambios")
    args = parser.parse_args(argv)

    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = mlflow_utils.get_client()

        candidate = _resolve_candidate(client, args.version)
        if candidate is None:
            logger.error("No hay versiones registradas de '%s'. Ejecutar src/training.py primero.",
                         config.MLFLOW_MODEL_NAME)
            return False

        candidate_metrics = mlflow_utils.get_run_metrics(candidate.run_id)
        production = mlflow_utils.get_version_by_alias(config.ALIAS_PRODUCTION)
        production_metrics = mlflow_utils.get_run_metrics(production.run_id) if production else {}

        logger.info("=" * 70)
        logger.info("EVALUACION DE PROMOCION - %s", config.MLFLOW_MODEL_NAME)
        logger.info("=" * 70)
        # La familia es informativa, no un gate: se promueve por metrica, no por
        # algoritmo. Pero sin ella el log no dice QUE se esta reemplazando.
        candidate_family = (candidate.tags or {}).get("model_family", "n/d")
        production_family = (production.tags or {}).get("model_family", "n/d") if production else None

        logger.info("Candidato:   v%s (run %s) - familia: %s",
                    candidate.version, candidate.run_id, candidate_family)
        logger.info("  val_rmse=%s  val_r2=%s  overfitting=%s",
                    _fmt(candidate_metrics.get("val_rmse")),
                    _fmt(candidate_metrics.get("val_r2"), 4),
                    _fmt(candidate_metrics.get("overfitting_r2_diff"), 4))
        if production is not None:
            logger.info("Produccion:  v%s (run %s) - familia: %s",
                        production.version, production.run_id, production_family)
            logger.info("  val_rmse=%s  val_r2=%s",
                        _fmt(production_metrics.get("val_rmse")),
                        _fmt(production_metrics.get("val_r2"), 4))
            if candidate_family != production_family:
                logger.info("  Cambio de familia: %s -> %s", production_family, candidate_family)
        else:
            logger.info("Produccion:  (ninguna version promovida todavia)")

        if production is not None and str(production.version) == str(candidate.version):
            logger.info("La version candidata YA esta en produccion. No hay nada que hacer.")
            return True

        should_promote, reasons = evaluate_gates(candidate_metrics, production_metrics)

        logger.info("-" * 70)
        logger.info("Gates aplicados:")
        for reason in reasons:
            logger.info("  %s", reason)
        logger.info("-" * 70)

        reason_text = " | ".join(reasons)
        if not should_promote:
            logger.warning("DECISION: NO promover la v%s. Se mantiene en '%s'.",
                           candidate.version, config.ALIAS_STAGING)
            if not args.dry_run:
                client.set_model_version_tag(config.MLFLOW_MODEL_NAME, candidate.version,
                                             "promotion_rejected_at",
                                             datetime.now().isoformat(timespec="seconds"))
                client.set_model_version_tag(config.MLFLOW_MODEL_NAME, candidate.version,
                                             "promotion_reason", reason_text[:480])
            return True

        logger.info("DECISION: PROMOVER la v%s a produccion.", candidate.version)
        if args.dry_run:
            logger.info("(--dry-run: no se aplica ningun cambio)")
            return True

        promote(candidate, reason_text, client, previous=production)
        logger.info("=" * 70)
        logger.info("El scoring ahora resolvera: models:/%s@%s -> v%s",
                    config.MLFLOW_MODEL_NAME, config.ALIAS_PRODUCTION, candidate.version)
        return True

    except MlflowException as exc:
        logger.error("Error de MLflow: %s", exc, exc_info=True)
        return False
    except Exception as exc:
        logger.error("Error en la promocion: %s", exc, exc_info=True)
        return False


def _fmt(value, decimals=2):
    return f"{value:,.{decimals}f}" if value is not None else "n/d"


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
