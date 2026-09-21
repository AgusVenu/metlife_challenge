"""Tests del historial de alertas con estado.

No se usa PostgreSQL: se sustituye el engine por un doble que registra las sentencias
ejecutadas. Lo que se verifica aca es la REGLA de decision -- que senal produce que
transicion, y que una alerta que ya estaba abierta se ACTUALICE en vez de volver a
insertarse. Que el SQL corra de verdad contra Postgres se verifica ejecutando el
pipeline (ver INFORME.md, seccion 12), no en esta suite.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

import alerts
import config
import monitoring as mon


# ---------------------------------------------------------------------------
# Doble del engine
# ---------------------------------------------------------------------------

class FakeConn:
    def __init__(self, registro):
        self.registro = registro

    def execute(self, statement, params=None):
        self.registro.append((str(statement).strip().split()[0].upper(), params or {}))
        return self


class FakeEngine:
    """Engine minimo: guarda las alertas abiertas y registra lo que se ejecuta."""

    def __init__(self, open_alerts=None):
        self.open_alerts = open_alerts or {}
        self.registro = []

    @contextmanager
    def begin(self):
        yield FakeConn(self.registro)

    def verbos(self):
        return [verbo for verbo, _ in self.registro]


def _signal(name, status, category="feature_drift", value=0.5, source="default"):
    return mon.Signal(name=name, category=category, value=value, status=status,
                      detail=f"{name} = {value}", thresholds_source=source)


def _report(batch_id, signals):
    report = mon.BatchReport(batch_id=batch_id, n_rows=100, has_target=True)
    report.signals = signals
    return report


def _open_alert(name, severity=config.STATUS_ALERT, occurrences=1, category="feature_drift"):
    """Fila de una alerta abierta, tal como vuelve de la base.

    A proposito NO trae `batch_id`: el lote ya viene del contexto de la corrida, y la
    transicion no debe depender de que el SELECT se acuerde de incluir esa columna.
    Una version de este codigo lo hacia, y las transiciones ONGOING terminaban con
    batch_id vacio en el artefacto alerts_<lote>.json de MLflow.
    """
    hace_un_rato = datetime.now() - timedelta(hours=3)
    return {
        "id": 7, "signal_name": name, "category": category,
        "severity": severity, "first_seen": hace_un_rato, "last_seen": hace_un_rato,
        "occurrences": occurrences, "severity_history": [],
    }


@pytest.fixture
def sin_alertas_previas(monkeypatch):
    def _make(open_alerts=None):
        engine = FakeEngine()
        monkeypatch.setattr(alerts, "fetch_open_alerts", lambda e, b: open_alerts or {})
        return engine
    return _make


# ---------------------------------------------------------------------------
# Las tres transiciones
# ---------------------------------------------------------------------------

def test_una_senal_en_ok_no_registra_nada(sin_alertas_previas):
    engine = sin_alertas_previas()
    report = _report("prod1", [_signal("drift:bmi", config.STATUS_OK)])
    assert alerts.reconcile(engine, report) == []
    assert engine.verbos() == []


def test_una_senal_que_falla_por_primera_vez_es_NEW(sin_alertas_previas):
    engine = sin_alertas_previas()
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    transiciones = alerts.reconcile(engine, report)

    assert [t.transition for t in transiciones] == [alerts.NEW]
    assert transiciones[0].occurrences == 1
    assert transiciones[0].is_new
    assert engine.verbos() == ["INSERT"]


def test_la_misma_senal_en_la_corrida_siguiente_es_ONGOING(sin_alertas_previas):
    """Es la deduplicacion: una alerta que lleva N corridas abierta aparece UNA vez
    como ONGOING con occurrences=N, no como N alertas."""
    engine = sin_alertas_previas({"drift:bmi": _open_alert("drift:bmi", occurrences=4)})
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    transiciones = alerts.reconcile(engine, report)

    assert [t.transition for t in transiciones] == [alerts.ONGOING]
    assert transiciones[0].occurrences == 5
    assert not transiciones[0].is_new
    # y NO se inserto una alerta nueva
    assert engine.verbos() == ["UPDATE"]


def test_toda_transicion_lleva_su_batch_id(sin_alertas_previas):
    """Las transiciones se serializan al artefacto alerts_<lote>.json de MLflow, donde
    el lote ya no viene del contexto: si `batch_id` viaja vacio, ese artefacto queda
    inservible para cualquiera que lo lea despues."""
    engine = sin_alertas_previas({
        "drift:bmi": _open_alert("drift:bmi", occurrences=2),
        "target:psi": _open_alert("target:psi"),
    })
    report = _report("prod3", [
        _signal("drift:bmi", config.STATUS_ALERT),            # ONGOING
        _signal("target:psi", config.STATUS_OK),              # RESOLVED
        _signal("schema:bmi:out_of_range", config.STATUS_ALERT),   # NEW
    ])
    transiciones = alerts.reconcile(engine, report)
    assert len(transiciones) == 3
    for transicion in transiciones:
        assert transicion.batch_id == "prod3", transicion.transition
        assert transicion.to_dict()["batch_id"] == "prod3"


def test_una_senal_que_vuelve_a_ok_cierra_la_alerta(sin_alertas_previas):
    engine = sin_alertas_previas({"drift:bmi": _open_alert("drift:bmi", occurrences=2)})
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_OK)])
    transiciones = alerts.reconcile(engine, report)

    assert [t.transition for t in transiciones] == [alerts.RESOLVED]
    assert engine.verbos() == ["UPDATE"]


def test_una_senal_que_desaparece_del_reporte_tambien_cierra(sin_alertas_previas):
    """prod3 no trae target: las senales de performance directamente no se calculan.
    Si venian abiertas de un lote que si tenia target, hay que cerrarlas igual."""
    engine = sin_alertas_previas({"performance:r2_drop": _open_alert("performance:r2_drop")})
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_OK)])
    transiciones = alerts.reconcile(engine, report)
    assert [t.transition for t in transiciones] == [alerts.RESOLVED]


def test_una_alerta_resuelta_que_reaparece_vuelve_a_ser_NEW(sin_alertas_previas):
    """Las resueltas no figuran entre las abiertas, asi que un problema que vuelve
    despues de haberse arreglado SI amerita notificarse de nuevo."""
    engine = sin_alertas_previas({})     # la vieja quedo en state='resolved'
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    assert [t.transition for t in alerts.reconcile(engine, report)] == [alerts.NEW]


# ---------------------------------------------------------------------------
# Cambios de severidad
# ---------------------------------------------------------------------------

def test_un_escalamiento_no_abre_una_alerta_nueva(sin_alertas_previas):
    """WARNING -> ALERT sigue siendo el MISMO problema. Tratarlo como nuevo volveria
    a notificar algo que ya se sabia."""
    engine = sin_alertas_previas(
        {"drift:bmi": _open_alert("drift:bmi", severity=config.STATUS_WARNING, occurrences=2)}
    )
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    transicion = alerts.reconcile(engine, report)[0]

    assert transicion.transition == alerts.ONGOING
    assert transicion.escalated_from == config.STATUS_WARNING
    assert transicion.severity == config.STATUS_ALERT
    assert transicion.severity_change == "escalada"
    assert engine.verbos() == ["UPDATE"]


def test_una_baja_de_severidad_no_se_llama_escalada(sin_alertas_previas):
    """Un ALERT que baja a WARNING al lado de uno que escalo, ambos rotulados
    'ESCALADA', confunde al que lee el reporte."""
    engine = sin_alertas_previas(
        {"drift:bmi": _open_alert("drift:bmi", severity=config.STATUS_ALERT, occurrences=2)}
    )
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_WARNING)])
    transicion = alerts.reconcile(engine, report)[0]
    assert transicion.severity_change == "bajo"


def test_sin_cambio_de_severidad_no_hay_nada_que_reportar(sin_alertas_previas):
    engine = sin_alertas_previas(
        {"drift:bmi": _open_alert("drift:bmi", severity=config.STATUS_ALERT)}
    )
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    assert alerts.reconcile(engine, report)[0].severity_change is None


@pytest.mark.parametrize("previa,actual,esperado", [
    (config.STATUS_WARNING, config.STATUS_ALERT, "escalada"),
    (config.STATUS_ALERT, config.STATUS_WARNING, "bajo"),
    (config.STATUS_ALERT, config.STATUS_ALERT, None),
    (None, config.STATUS_ALERT, None),
])
def test_direccion_del_cambio_de_severidad(previa, actual, esperado):
    assert alerts.severity_change(previa, actual) == esperado


# ---------------------------------------------------------------------------
# Varias senales a la vez
# ---------------------------------------------------------------------------

def test_un_lote_puede_tener_las_tres_transiciones_en_la_misma_corrida(sin_alertas_previas):
    engine = sin_alertas_previas({
        "drift:bmi": _open_alert("drift:bmi", occurrences=3),        # sigue mal -> ONGOING
        "target:psi": _open_alert("target:psi", occurrences=1),      # se arreglo -> RESOLVED
    })
    report = _report("prod3", [
        _signal("drift:bmi", config.STATUS_ALERT),
        _signal("target:psi", config.STATUS_OK),
        _signal("schema:bmi:out_of_range", config.STATUS_WARNING),   # nueva -> NEW
    ])
    resumen = alerts.summarize(alerts.reconcile(engine, report))
    assert resumen == {"new": 1, "ongoing": 1, "resolved": 1}


def test_el_resumen_global_suma_todos_los_lotes():
    def t(kind):
        return alerts.AlertTransition(
            batch_id="x", signal_name="s", category="c", severity="ALERT",
            transition=kind, occurrences=1, first_seen="", last_seen="",
        )
    total = alerts.summarize_many({
        "prod1": [],
        "prod2": [t(alerts.NEW), t(alerts.NEW), t(alerts.RESOLVED)],
        "prod3": [t(alerts.ONGOING)],
    })
    assert total == {"new": 2, "ongoing": 1, "resolved": 1}


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------

def test_un_fallo_de_la_base_no_aborta_un_scoring_que_ya_termino_bien(caplog):
    """Mismo criterio que mlflow_utils.log_artifact_safe: el historial es valioso,
    pero las predicciones ya estan escritas y el reporte ya se puede generar."""
    class EngineRoto:
        def connect(self):
            raise RuntimeError("connection refused")
        def begin(self):
            raise RuntimeError("connection refused")

    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    with caplog.at_level("WARNING"):
        assert alerts.reconcile(EngineRoto(), report) == []
    assert any("historial de alertas" in r.message for r in caplog.records)


def test_un_valor_no_finito_no_rompe_la_insercion(sin_alertas_previas):
    engine = sin_alertas_previas()
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT, value=float("inf"))])
    alerts.reconcile(engine, report)
    _, params = engine.registro[0]
    assert params["value"] is None


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------

def test_el_reporte_distingue_visualmente_lo_nuevo_de_lo_conocido(sin_alertas_previas):
    engine = sin_alertas_previas({"drift:bmi": _open_alert("drift:bmi", occurrences=2)})
    report = _report("prod3", [
        _signal("drift:bmi", config.STATUS_ALERT),
        _signal("target:psi", config.STATUS_ALERT),
    ])
    texto = "\n".join(alerts.render_alerts_section(alerts.reconcile(engine, report)))
    assert "[NUEVA]" in texto and "[PERSISTE]" in texto
    # lo nuevo va primero: es lo unico que pide accion
    assert texto.index("[NUEVA]") < texto.index("[PERSISTE]")


def test_sin_transiciones_lo_dice_explicito():
    assert "sin alertas" in "\n".join(alerts.render_alerts_section([]))


def test_un_lote_que_no_se_pudo_reconciliar_no_dice_sin_cambios():
    """`[]` significa "se reconcilio y nada cambio"; `None` significa "no se pudo
    reconciliar". Colapsarlos haria que un lote que exploto se muestre como "sin
    cambios respecto de la corrida anterior" al lado de su badge de ALERT, que es
    justamente lo que no se puede afirmar."""
    fallado = _report("prod9", [])
    fallado.status = config.STATUS_ALERT
    consolidado = mon.consolidate([fallado], {}, {"prod9": None})

    assert consolidado["alerts"]["by_batch"]["prod9"] is None
    assert consolidado["alerts"]["summary"] == {"new": 0, "ongoing": 0, "resolved": 0}

    texto = mon.render_text_report(consolidado)
    assert "no se pudo procesar" in texto
    assert "sin cambios en el historial" not in texto

    import dashboard
    html = dashboard.render(consolidado)
    assert "no hubo reconciliacion" in html
    assert "sin cambios respecto de la corrida anterior" not in html


def test_un_lote_reconciliado_sin_novedades_si_dice_sin_cambios():
    sano = _report("prod1", [_signal("drift:bmi", config.STATUS_OK)])
    consolidado = mon.consolidate([sano], {}, {"prod1": []})
    assert consolidado["alerts"]["by_batch"]["prod1"] == []
    assert "sin cambios en el historial" in mon.render_text_report(consolidado)


def test_el_reporte_consolidado_incluye_el_bloque_de_alertas():
    report = _report("prod3", [_signal("drift:bmi", config.STATUS_ALERT)])
    transicion = alerts.AlertTransition(
        batch_id="prod3", signal_name="drift:bmi", category="feature_drift",
        severity="ALERT", transition=alerts.NEW, occurrences=1,
        first_seen="2026-09-20T10:00:00", last_seen="2026-09-20T10:00:00",
    )
    consolidado = mon.consolidate([report], {}, {"prod3": [transicion]})

    assert consolidado["alerts"]["summary"] == {"new": 1, "ongoing": 0, "resolved": 0}
    assert consolidado["alerts"]["by_batch"]["prod3"][0]["signal_name"] == "drift:bmi"

    texto = mon.render_text_report(consolidado)
    assert "NUEVAS=1" in texto
    assert "[NUEVA]" in texto

    frame = mon.to_dataframe([report], {"prod3": [transicion]})
    assert frame.iloc[0]["alerts_new"] == 1
    assert frame.iloc[0]["alerts_resolved"] == 0
