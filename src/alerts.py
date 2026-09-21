"""Historial de alertas con estado: que es nuevo, que persiste y que se resolvio.

Por que hace falta
------------------
`batch_monitoring` guarda una fila por lote por corrida. Eso responde "como esta el
lote hoy", pero no responde las dos preguntas que hacen util a una alerta:

    - Esto es NUEVO, o ya lo sabia?
    - Lo que estaba mal, se arreglo?

Un reporte que dice exactamente lo mismo en cada corrida no es un sistema de alertas:
es un informe. La diferencia esta en tener memoria.

El modelo de estado
-------------------
La identidad de una alerta es la tupla `(batch_id, signal_name)` -- por ejemplo
`("prod3", "drift:bmi")`. Sobre esa clave hay tres transiciones:

    senal en WARNING/ALERT + no hay alerta abierta   -> NEW       (se abre)
    senal en WARNING/ALERT + ya hay alerta abierta   -> ONGOING   (se acumula)
    senal en OK o ausente  + hay alerta abierta      -> RESOLVED  (se cierra)
    senal en OK            + no hay alerta abierta   -> nada (no se registra ruido)

La deduplicacion es el punto: NEW es el UNICO evento notificable. Una alerta que lleva
cinco corridas abierta aparece una vez como ONGOING con `occurrences=5`, no como cinco
alertas. Un escalamiento (WARNING -> ALERT) sobre una alerta abierta se anota en
`severity_history` y NO abre una alerta nueva.

Ese invariante esta impuesto por la base: `idx_alert_open` es un indice unico parcial
sobre `state='open'`, asi que dos alertas abiertas para la misma senal no se pueden
insertar aunque el codigo se equivoque.

Uso como CLI
------------
    python src/alerts.py                      # alertas abiertas
    python src/alerts.py --batch prod3        # solo las de un lote
    python src/alerts.py --history            # incluye las ya resueltas
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

import config

logger = logging.getLogger(__name__)


# Las tres transiciones posibles.
NEW = "NEW"
ONGOING = "ONGOING"
RESOLVED = "RESOLVED"

STATE_OPEN = "open"
STATE_RESOLVED = "resolved"


# ============================================================================
# Transiciones
# ============================================================================

@dataclass
class AlertTransition:
    """Lo que le paso a una alerta en ESTA corrida."""
    batch_id: str
    signal_name: str
    category: str
    severity: str
    transition: str           # NEW | ONGOING | RESOLVED
    occurrences: int
    first_seen: str
    last_seen: str
    detail: str = ""
    thresholds_source: str = "default"
    escalated_from: Optional[str] = None

    @property
    def is_new(self) -> bool:
        return self.transition == NEW

    @property
    def severity_change(self) -> Optional[str]:
        """"escalada" si la severidad subio, "bajo" si bajo, None si no cambio."""
        return severity_change(self.escalated_from, self.severity)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "signal_name": self.signal_name,
            "category": self.category,
            "severity": self.severity,
            "transition": self.transition,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "detail": self.detail,
            "thresholds_source": self.thresholds_source,
            "escalated_from": self.escalated_from,
        }


def severity_change(previous: Optional[str], current: str) -> Optional[str]:
    """Direccion del cambio de severidad. Un ALERT que baja a WARNING no es una
    escalada, y llamarlo asi al lado de una que si escalo confunde al que lee."""
    if not previous or previous == current:
        return None
    try:
        subio = config.STATUS_ORDER.index(current) > config.STATUS_ORDER.index(previous)
    except ValueError:
        return None
    return "escalada" if subio else "bajo"


def summarize(transitions: List[AlertTransition]) -> Dict[str, int]:
    """Cuenta las transiciones por tipo. Las claves son fijas: siempre estan las tres."""
    counts = {NEW: 0, ONGOING: 0, RESOLVED: 0}
    for transition in transitions or []:
        if transition.transition in counts:
            counts[transition.transition] += 1
    return {
        "new": counts[NEW],
        "ongoing": counts[ONGOING],
        "resolved": counts[RESOLVED],
    }


def summarize_many(transitions_by_batch: Dict[str, List[AlertTransition]]) -> Dict[str, int]:
    """Resumen global a partir del detalle por lote."""
    total = {"new": 0, "ongoing": 0, "resolved": 0}
    for transitions in (transitions_by_batch or {}).values():
        for key, value in summarize(transitions).items():
            total[key] += value
    return total


# ============================================================================
# Acceso a la base
# ============================================================================

def fetch_open_alerts(engine, batch_id: str) -> Dict[str, dict]:
    """Alertas abiertas de un lote, indexadas por `signal_name`."""
    query = text(
        "SELECT id, batch_id, signal_name, category, severity, first_seen, last_seen, "
        "       occurrences, severity_history "
        "FROM alert_history WHERE batch_id = :batch_id AND state = :state"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"batch_id": batch_id, "state": STATE_OPEN}).mappings().all()
    return {row["signal_name"]: dict(row) for row in rows}


def _as_iso(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value) if value is not None else ""


# ============================================================================
# Reconciliacion
# ============================================================================

def reconcile(engine, report, model_info: Dict[str, Any] = None) -> List[AlertTransition]:
    """Compara las senales del lote contra las alertas abiertas y aplica las transiciones.

    Devuelve la lista de transiciones de ESTA corrida.

    Si la base no responde, loguea un WARNING y devuelve []. Es el mismo criterio que
    `mlflow_utils.log_artifact_safe`: el historial de alertas es valioso, pero no vale
    abortar un scoring que ya termino bien y cuyas predicciones ya estan escritas.
    """
    try:
        return _reconcile(engine, report, model_info or {})
    except Exception as exc:
        logger.warning("No se pudo actualizar el historial de alertas del lote %s: %s",
                       report.batch_id, exc)
        return []


def _reconcile(engine, report, model_info: Dict[str, Any]) -> List[AlertTransition]:
    batch_id = report.batch_id
    now = datetime.now()
    open_alerts = fetch_open_alerts(engine, batch_id)

    # Senales que HOY estan mal. Una senal en OK no entra: el historial guarda
    # problemas, no el estado completo de cada corrida (para eso esta batch_monitoring).
    firing = {
        signal.name: signal
        for signal in report.signals
        if signal.status != config.STATUS_OK
    }

    transitions: List[AlertTransition] = []
    with engine.begin() as conn:
        for name, signal in firing.items():
            existing = open_alerts.get(name)
            if existing is None:
                transitions.append(_open_alert(conn, batch_id, signal, now, model_info))
            else:
                transitions.append(_touch_alert(conn, existing, signal, now, batch_id))

        for name, existing in open_alerts.items():
            if name not in firing:
                transitions.append(_resolve_alert(conn, existing, batch_id, now))

    return transitions


def _open_alert(conn, batch_id: str, signal, now: datetime,
                model_info: Dict[str, Any]) -> AlertTransition:
    history = [{"at": now.isoformat(timespec="seconds"), "severity": signal.status}]
    conn.execute(
        text(
            "INSERT INTO alert_history ("
            "  batch_id, signal_name, category, severity, state, first_seen, last_seen, "
            "  occurrences, last_value, last_detail, thresholds_source, "
            "  model_name, model_version, mlflow_run_id, severity_history"
            ") VALUES ("
            "  :batch_id, :signal_name, :category, :severity, :state, :now, :now, "
            "  1, :value, :detail, :thresholds_source, "
            "  :model_name, :model_version, :run_id, CAST(:history AS JSONB))"
        ),
        {
            "batch_id": batch_id, "signal_name": signal.name, "category": signal.category,
            "severity": signal.status, "state": STATE_OPEN, "now": now,
            "value": _finite(signal.value), "detail": signal.detail,
            "thresholds_source": getattr(signal, "thresholds_source", "default"),
            "model_name": model_info.get("model_name"),
            "model_version": model_info.get("model_version"),
            "run_id": model_info.get("scoring_run_id") or model_info.get("run_id"),
            "history": json.dumps(history),
        },
    )
    return AlertTransition(
        batch_id=batch_id, signal_name=signal.name, category=signal.category,
        severity=signal.status, transition=NEW, occurrences=1,
        first_seen=_as_iso(now), last_seen=_as_iso(now), detail=signal.detail,
        thresholds_source=getattr(signal, "thresholds_source", "default"),
    )


def _touch_alert(conn, existing: dict, signal, now: datetime,
                 batch_id: str = None) -> AlertTransition:
    """Actualiza una alerta que ya estaba abierta. NO abre una nueva.

    Un cambio de severidad se anota como escalamiento en `severity_history`: sigue
    siendo el mismo problema, no uno nuevo, y tratarlo como nuevo volveria a notificar
    algo que ya se sabia.
    """
    # El lote viene por parametro y no de la fila: asi la transicion no depende de que
    # el SELECT de fetch_open_alerts se acuerde de incluir la columna.
    batch_id = batch_id or existing.get("batch_id") or ""
    previous = existing["severity"]
    escalated = previous != signal.status

    history = existing.get("severity_history") or []
    if isinstance(history, str):
        history = json.loads(history)
    if escalated:
        history = list(history) + [
            {"at": now.isoformat(timespec="seconds"), "severity": signal.status}
        ]

    occurrences = int(existing["occurrences"]) + 1
    conn.execute(
        text(
            "UPDATE alert_history SET last_seen = :now, occurrences = :occurrences, "
            "  severity = :severity, last_value = :value, last_detail = :detail, "
            "  thresholds_source = :thresholds_source, "
            "  severity_history = CAST(:history AS JSONB) "
            "WHERE id = :id"
        ),
        {
            "now": now, "occurrences": occurrences, "severity": signal.status,
            "value": _finite(signal.value), "detail": signal.detail,
            "thresholds_source": getattr(signal, "thresholds_source", "default"),
            "history": json.dumps(history), "id": existing["id"],
        },
    )
    if escalated:
        logger.info("Alerta %s/%s cambio de severidad: %s -> %s",
                    batch_id, signal.name, previous, signal.status)

    return AlertTransition(
        batch_id=batch_id, signal_name=signal.name,
        category=signal.category, severity=signal.status, transition=ONGOING,
        occurrences=occurrences, first_seen=_as_iso(existing["first_seen"]),
        last_seen=_as_iso(now), detail=signal.detail,
        thresholds_source=getattr(signal, "thresholds_source", "default"),
        escalated_from=previous if escalated else None,
    )


def _resolve_alert(conn, existing: dict, batch_id: str, now: datetime) -> AlertTransition:
    conn.execute(
        text("UPDATE alert_history SET state = :state, resolved_at = :now WHERE id = :id"),
        {"state": STATE_RESOLVED, "now": now, "id": existing["id"]},
    )
    return AlertTransition(
        batch_id=batch_id, signal_name=existing["signal_name"],
        category=existing["category"], severity=existing["severity"],
        transition=RESOLVED, occurrences=int(existing["occurrences"]),
        first_seen=_as_iso(existing["first_seen"]),
        last_seen=_as_iso(existing["last_seen"]),
        detail="la senal volvio a OK",
    )


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


# ============================================================================
# Consulta e informe
# ============================================================================

_TRANSITION_ICON = {NEW: "[NUEVA]   ", ONGOING: "[PERSISTE]", RESOLVED: "[RESUELTA]"}


def render_alerts_section(transitions: List[AlertTransition], indent: str = "  ") -> List[str]:
    """Bloque de alertas de un lote, para el reporte de texto."""
    if not transitions:
        return [f"{indent}sin alertas registradas"]

    orden = {NEW: 0, ONGOING: 1, RESOLVED: 2}
    lines = []
    for transition in sorted(transitions, key=lambda t: (orden[t.transition], t.signal_name)):
        icon = _TRANSITION_ICON[transition.transition]
        if transition.transition == NEW:
            contexto = "1a vez"
        elif transition.transition == ONGOING:
            contexto = f"{transition.occurrences} corridas, desde {transition.first_seen}"
            cambio = transition.severity_change
            if cambio:
                contexto += (f"; {cambio.upper()} {transition.escalated_from} -> "
                             f"{transition.severity}")
        else:
            contexto = f"estuvo {transition.occurrences} corrida(s) abierta"
        lines.append(f"{indent}{icon} {transition.signal_name:<28} ({contexto})")
    return lines


def recent_history(engine, batch_id: str = None, include_resolved: bool = False,
                   limit: int = 50) -> pd.DataFrame:
    """Historial de alertas, para consultar desde la CLI o desde un notebook."""
    where, params = [], {"limit": limit}
    if batch_id:
        where.append("batch_id = :batch_id")
        params["batch_id"] = batch_id
    if not include_resolved:
        where.append("state = :state")
        params["state"] = STATE_OPEN

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    query = (
        "SELECT batch_id, signal_name, category, severity, state, occurrences, "
        "       first_seen, last_seen, resolved_at, thresholds_source, "
        "       model_name, model_version, last_detail "
        f"FROM alert_history {clause} ORDER BY last_seen DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(text(query), params).mappings().all())


def main(argv=None) -> bool:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Historial de alertas de monitoreo.")
    parser.add_argument("--batch", help="filtrar por lote (prod1, prod2, ...)")
    parser.add_argument("--history", action="store_true",
                        help="incluir tambien las alertas ya resueltas")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    from utils import get_db_engine
    frame = recent_history(get_db_engine(), batch_id=args.batch,
                           include_resolved=args.history, limit=args.limit)

    if frame.empty:
        estado = "alertas" if args.history else "alertas ABIERTAS"
        print(f"No hay {estado}" + (f" para el lote {args.batch}." if args.batch else "."))
        return True

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False))
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
