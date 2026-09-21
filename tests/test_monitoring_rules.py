"""Tests de las reglas de monitoreo por feature y por lote.

Lo que se verifica aca es la REGLA de precedencia y la tolerancia a un archivo
ausente o roto. Nada depende de MLflow ni de PostgreSQL.
"""

import json

import numpy as np
import pandas as pd
import pytest

import config
import monitoring as mon


REGLAS = {
    "features": {
        "bmi": {"psi_warn": 0.05, "psi_alert": 0.15},
        "region": {"psi_warn": 0.20, "psi_alert": 0.40},
    },
    "batches": {
        "prod3": {
            "psi_warn": 0.02,
            "psi_alert": 0.08,
            "schema_alert_pct": 50.0,
            "features": {"region": {"psi_warn": 0.30, "psi_alert": 0.60}},
        },
    },
}


# ---------------------------------------------------------------------------
# Precedencia
# ---------------------------------------------------------------------------

def test_sin_reglas_todo_es_el_default():
    """Es el caso normal: sin archivo, el monitoreo se comporta como siempre."""
    assert config.resolve_thresholds("prod1", "bmi", rules={}) is config.DEFAULT_THRESHOLDS
    assert config.resolve_thresholds(rules={}).source == "default"


def test_una_feature_sin_regla_cae_al_default():
    thresholds = config.resolve_thresholds(feature="age", rules=REGLAS)
    assert thresholds.source == "default"
    assert thresholds.psi_alert == config.DEFAULT_THRESHOLDS.psi_alert


def test_regla_de_feature():
    thresholds = config.resolve_thresholds(feature="bmi", rules=REGLAS)
    assert thresholds.source == "feature:bmi"
    assert (thresholds.psi_warn, thresholds.psi_alert) == (0.05, 0.15)


def test_regla_de_lote():
    thresholds = config.resolve_thresholds(batch_id="prod3", rules=REGLAS)
    assert thresholds.source == "batch:prod3"
    assert thresholds.schema_alert_pct == 50.0


def test_la_regla_de_lote_le_gana_a_la_de_feature():
    """Decision documentada: una regla de lote es una afirmacion deliberada sobre un
    dataset concreto; una de feature es un refinamiento que vale para todos."""
    thresholds = config.resolve_thresholds(batch_id="prod3", feature="bmi", rules=REGLAS)
    assert (thresholds.psi_warn, thresholds.psi_alert) == (0.02, 0.08)
    assert thresholds.source == "feature:bmi+batch:prod3"


def test_la_regla_de_lote_y_feature_le_gana_a_todo():
    """Es la forma inequivoca de resolver el choque entre una regla de lote y una de
    feature."""
    thresholds = config.resolve_thresholds(batch_id="prod3", feature="region", rules=REGLAS)
    assert (thresholds.psi_warn, thresholds.psi_alert) == (0.30, 0.60)
    assert thresholds.source == "feature:region+batch:prod3+batch:prod3/feature:region"


def test_cada_grupo_de_umbral_sabe_que_regla_lo_fijo():
    """Dos reglas pueden pisar claves de grupos DISTINTOS: la de feature los PSI y la
    de lote el umbral de esquema. Si la senal de drift dijera "batch:prod3", el rastro
    de auditoria estaria nombrando una regla que no aporto el umbral que disparo."""
    reglas = {
        "features": {"bmi": {"psi_alert": 0.15}},
        "batches": {"prod3": {"schema_alert_pct": 50.0}},
    }
    thresholds = config.resolve_thresholds(batch_id="prod3", feature="bmi", rules=reglas)

    assert thresholds.psi_alert == 0.15          # vino de la regla de feature
    assert thresholds.schema_alert_pct == 50.0   # vino de la regla de lote

    assert thresholds.source_for("psi") == "feature:bmi"
    assert thresholds.source_for("schema") == "batch:prod3"
    assert thresholds.source_for("perf") == "default"      # nadie lo toco
    # y `source` sigue siendo el resumen de todo lo que aporto
    assert thresholds.source == "feature:bmi+batch:prod3"


def test_dentro_de_un_mismo_grupo_manda_la_regla_mas_especifica():
    reglas = {
        "features": {"bmi": {"psi_alert": 0.15}},
        "batches": {"prod3": {"psi_alert": 0.08}},
    }
    thresholds = config.resolve_thresholds(batch_id="prod3", feature="bmi", rules=reglas)
    assert thresholds.psi_alert == 0.08
    assert thresholds.source_for("psi") == "batch:prod3"


def test_las_senales_estampan_el_origen_de_su_propio_grupo(monkeypatch, baseline_simple):
    """Una senal de drift no debe llevar el nombre de una regla que solo toco el
    umbral de esquema, ni al reves."""
    monkeypatch.setattr(config, "load_rules", lambda *a, **k: {
        "features": {"bmi": {"psi_warn": 0.0001, "psi_alert": 0.0002}},
        "batches": {"prod1": {"schema_warn_pct": 90.0, "schema_alert_pct": 95.0}},
    })
    rng = np.random.default_rng(7)
    features = pd.DataFrame({"bmi": rng.normal(31.5, 6, 1500)})
    violacion = {"column": "bmi", "kind": "out_of_range", "severity": config.STATUS_ALERT,
                 "n_rows": 15, "pct_rows": 1.0, "detail": "fuera de rango"}

    report = mon.monitor_batch("prod1", features, rng.normal(10000, 3000, 1500),
                               baseline_simple, violations=[violacion])
    por_nombre = {s.name: s for s in report.signals}

    assert por_nombre["drift:bmi"].thresholds_source == "feature:bmi"
    assert por_nombre["schema:bmi:out_of_range"].thresholds_source == "batch:prod1"
    # la regla de esquema es tan laxa que la violacion del 1% queda en OK
    assert por_nombre["schema:bmi:out_of_range"].status == config.STATUS_OK


def test_una_regla_solo_pisa_las_claves_que_nombra():
    """La regla de prod3 no dice nada de perf_alert_ratio: ese umbral sigue siendo
    el global."""
    thresholds = config.resolve_thresholds(batch_id="prod3", rules=REGLAS)
    assert thresholds.perf_alert_ratio == config.DEFAULT_THRESHOLDS.perf_alert_ratio
    assert thresholds.r2_alert_drop == config.DEFAULT_THRESHOLDS.r2_alert_drop


def test_un_lote_sin_regla_cae_al_default():
    assert config.resolve_thresholds(batch_id="prod1", rules=REGLAS).source == "default"


# ---------------------------------------------------------------------------
# Tolerancia a archivos ausentes, rotos o con basura
# ---------------------------------------------------------------------------

def test_archivo_ausente_devuelve_reglas_vacias(tmp_path):
    assert config.load_rules(tmp_path / "no-existe.json") == {}


def test_json_roto_no_tumba_el_pipeline(tmp_path, caplog):
    """Un scoring que por lo demas puede correr perfectamente con los umbrales
    globales no debe abortar por un archivo de configuracion mal escrito."""
    roto = tmp_path / "roto.json"
    roto.write_text("{ esto no es json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert config.load_rules(roto) == {}
    assert any("reglas" in r.message for r in caplog.records)


def test_json_que_no_es_un_objeto_se_ignora(tmp_path, caplog):
    lista = tmp_path / "lista.json"
    lista.write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert config.load_rules(lista) == {}


@pytest.mark.parametrize("reglas", [
    {"features": "bmi"},                        # `features` no es un objeto
    {"batches": {"prod3": 0.05}},               # el bloque del lote es un numero
    {"batches": {"prod3": {"features": []}}},   # las features del lote son una lista
    {"features": {"bmi": "estricto"}},          # el bloque de la feature es texto
    "esto no es un objeto",                     # las reglas enteras no son un dict
])
def test_un_bloque_mal_formado_cae_al_default_en_vez_de_reventar(reglas, caplog):
    """`load_rules` solo valida la RAIZ. Un bloque anidado mal escrito hacia estallar
    `resolve_thresholds` dentro de `monitor_batch`; como `run_prod_scoring` atrapa la
    excepcion por lote, el resultado era que TODOS los lotes quedaban en ALERT con
    'no pudo procesarse'. Un typo en un archivo de configuracion no debe poner en
    rojo una corrida entera."""
    with caplog.at_level("WARNING"):
        thresholds = config.resolve_thresholds(batch_id="prod3", feature="bmi", rules=reglas)
    assert thresholds is config.DEFAULT_THRESHOLDS or thresholds.source == "default"


def test_un_bloque_mal_formado_no_anula_a_los_bien_formados():
    reglas = {"features": {"bmi": {"psi_alert": 0.15}}, "batches": {"prod3": "roto"}}
    thresholds = config.resolve_thresholds(batch_id="prod3", feature="bmi", rules=reglas)
    assert thresholds.psi_alert == 0.15
    assert thresholds.source == "feature:bmi"


def test_las_claves_desconocidas_se_ignoran():
    """Los bloques traen `_comment` con la justificacion de cada regla: tienen que
    poder convivir con los umbrales sin romper nada."""
    reglas = {"features": {"bmi": {"_comment": ["porque si"], "psi_alert": 0.15,
                                   "umbral_inventado": 9.9}}}
    thresholds = config.resolve_thresholds(feature="bmi", rules=reglas)
    assert thresholds.psi_alert == 0.15
    assert not hasattr(thresholds, "umbral_inventado")


def test_un_valor_no_numerico_se_ignora_con_aviso(caplog):
    reglas = {"features": {"bmi": {"psi_alert": "mucho"}}}
    with caplog.at_level("WARNING"):
        thresholds = config.resolve_thresholds(feature="bmi", rules=reglas)
    assert thresholds.source == "default"
    assert any("no numerico" in r.message for r in caplog.records)


def test_un_bloque_solo_con_comentarios_no_cuenta_como_regla():
    reglas = {"features": {"bmi": {"_comment": "nada que cambiar"}}}
    assert config.resolve_thresholds(feature="bmi", rules=reglas).source == "default"


# ---------------------------------------------------------------------------
# Violaciones que no se negocian con un porcentaje
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["missing_column", "not_numeric", "row_count_mismatch"])
def test_las_violaciones_estructurales_no_se_ablandan_con_un_umbral(kind):
    """Una columna ausente, una que no se puede interpretar, o features y target con
    distinta cantidad de filas (o sea predicciones comparadas contra el actual
    equivocado) rompen el contrato aunque se afloje el umbral de filas invalidas.
    `row_count_mismatch` es el fallo de integridad mas grave de un lote."""
    violacion = {"column": "bmi", "kind": kind, "severity": config.STATUS_ALERT,
                 "n_rows": 10, "pct_rows": 100.0, "detail": "roto"}
    permisivo = config.DEFAULT_THRESHOLDS.__class__(
        **{**config.DEFAULT_THRESHOLDS.to_dict(),
           "schema_warn_pct": 100.0, "schema_alert_pct": 100.0, "source": "batch:x"}
    )
    senal = mon.evaluate_schema([violacion], permisivo)[0]
    assert senal.status == config.STATUS_ALERT, kind


def test_una_violacion_por_porcentaje_si_responde_al_umbral():
    violacion = {"column": "bmi", "kind": "out_of_range", "severity": config.STATUS_ALERT,
                 "n_rows": 40, "pct_rows": 3.0, "detail": "fuera de rango"}
    permisivo = config.DEFAULT_THRESHOLDS.__class__(
        **{**config.DEFAULT_THRESHOLDS.to_dict(),
           "schema_warn_pct": 5.0, "schema_alert_pct": 10.0, "source": "batch:x"}
    )
    assert mon.evaluate_schema([violacion], permisivo)[0].status == config.STATUS_OK
    assert mon.evaluate_schema([violacion])[0].status == config.STATUS_ALERT


# ---------------------------------------------------------------------------
# El archivo que se entrega
# ---------------------------------------------------------------------------

def test_el_archivo_de_reglas_del_repo_es_valido_y_esta_justificado():
    reglas = config.load_rules(config.MONITORING_RULES_FILE)
    assert reglas, "config/monitoring_rules.json deberia existir en el repo"
    for bloque in (reglas.get("features") or {}).values():
        assert "_comment" in bloque, "cada regla se entrega justificada"
    # Y el resultado de aplicarlo tiene que ser coherente con lo que documenta
    assert config.resolve_thresholds(feature="bmi", rules=reglas).psi_alert < \
        config.DEFAULT_THRESHOLDS.psi_alert
    assert config.resolve_thresholds(feature="region", rules=reglas).psi_alert > \
        config.DEFAULT_THRESHOLDS.psi_alert


# ---------------------------------------------------------------------------
# Efecto real sobre el semaforo
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline_simple():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"bmi": rng.normal(30, 6, 2000), "sex": rng.choice(["male", "female"], 2000)})
    return {
        "numeric_features": {"bmi": {"mean": 30.0, "bins": mon.make_bins(X["bmi"])}},
        "categorical_features": {},
        "predictions": {"mean": 10000.0, "bins": mon.make_bins(rng.normal(10000, 3000, 2000))},
        "target": {"mean": 10000.0, "bins": mon.make_bins(rng.normal(10000, 3000, 2000))},
        "metrics": {"val_rmse": 4000.0, "val_r2": 0.83},
    }


def test_una_regla_estricta_convierte_en_warning_lo_que_seria_ok(monkeypatch, baseline_simple):
    """Un corrimiento chico de bmi: OK con el umbral global, WARNING con la regla."""
    rng = np.random.default_rng(7)
    features = pd.DataFrame({"bmi": rng.normal(31.4, 6, 1500)})   # PSI moderado

    monkeypatch.setattr(config, "load_rules", lambda *a, **k: {})
    signals_default, _ = mon.evaluate_feature_drift(features, baseline_simple, batch_id="prod1")
    bmi_default = next(s for s in signals_default if s.name == "drift:bmi")

    monkeypatch.setattr(config, "load_rules",
                        lambda *a, **k: {"features": {"bmi": {"psi_warn": 0.0001,
                                                              "psi_alert": 999}}})
    signals_regla, _ = mon.evaluate_feature_drift(features, baseline_simple, batch_id="prod1")
    bmi_regla = next(s for s in signals_regla if s.name == "drift:bmi")

    assert bmi_default.status == config.STATUS_OK
    assert bmi_default.thresholds_source == "default"
    assert bmi_regla.status == config.STATUS_WARNING
    assert bmi_regla.thresholds_source == "feature:bmi"
    # el PSI no cambia: lo que cambia es la vara con la que se lo mide
    assert bmi_default.value == pytest.approx(bmi_regla.value)


def test_el_reporte_deja_por_escrito_que_regla_se_aplico(monkeypatch, baseline_simple):
    """Sin esto, un WARNING con umbral custom es indistinguible de uno con el global
    y el reporte deja de ser auditable."""
    rng = np.random.default_rng(7)
    features = pd.DataFrame({"bmi": rng.normal(34, 6, 1500)})
    monkeypatch.setattr(config, "load_rules",
                        lambda *a, **k: {"features": {"bmi": {"psi_warn": 0.0001,
                                                              "psi_alert": 0.0002}}})
    report = mon.monitor_batch("prod1", features, rng.normal(10000, 3000, 1500),
                               baseline_simple, violations=[])
    assert report.custom_threshold_sources == ["feature:bmi"]
    texto = mon.render_text_report(mon.consolidate([report]))
    assert "feature:bmi" in texto
    assert "Excepciones aplicadas" in texto
    # y los umbrales del pie siguen siendo los globales, marcados como tales
    assert "Umbrales por defecto" in texto
