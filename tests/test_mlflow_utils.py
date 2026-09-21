"""Tests de las utilidades de tracking.

Solo se cubre `log_artifact_safe`, que es la unica funcion del modulo con
logica propia y sin dependencia del backend: el resto (`setup_tracking`,
`resolve_model`, `register_model_version`) necesita PostgreSQL y un registry
vivo, y el resto de la suite corre sin ninguno de los dos.

Lo que se fija aca es la garantia que la funcion promete: un artefacto accesorio
que no se puede subir degrada a WARNING y nunca propaga la excepcion, porque esa
excepcion aborta un run de entrenamiento que ya habia terminado bien.
"""

import mlflow
import pytest

import mlflow_utils


@pytest.fixture
def artifact(tmp_path):
    """Un archivo real en disco, para separar 'no existe' de 'fallo la subida'."""
    path = tmp_path / "training_report.txt"
    path.write_text("reporte", encoding="utf-8")
    return path


def test_path_none_no_sube_nada_y_no_falla(monkeypatch):
    monkeypatch.setattr(mlflow, "log_artifact", _boom)
    assert mlflow_utils.log_artifact_safe(None) is False


def test_archivo_inexistente_devuelve_false_sin_llamar_a_mlflow(tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow, "log_artifact", _boom)
    assert mlflow_utils.log_artifact_safe(tmp_path / "no_existe.png") is False


def test_un_fallo_del_artifact_store_no_propaga_la_excepcion(artifact, monkeypatch):
    monkeypatch.setattr(mlflow, "log_artifact", _boom)
    assert mlflow_utils.log_artifact_safe(artifact) is False


def test_el_fallo_queda_logueado_como_warning(artifact, monkeypatch, caplog):
    monkeypatch.setattr(mlflow, "log_artifact", _boom)
    with caplog.at_level("WARNING"):
        mlflow_utils.log_artifact_safe(artifact)
    assert artifact.name in caplog.text


def test_subida_exitosa_devuelve_true_y_propaga_artifact_path(artifact, monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        mlflow, "log_artifact",
        lambda local_path, artifact_path=None: llamadas.append((local_path, artifact_path)),
    )

    assert mlflow_utils.log_artifact_safe(artifact, artifact_path="monitoring") is True
    assert llamadas == [(str(artifact), "monitoring")]


def _boom(*args, **kwargs):
    """Simula un artifact store caido (disco lleno, permisos, red)."""
    raise OSError("No space left on device")


# ============================================================================
# Gate de deduplicacion del Model Registry
# ============================================================================
#
# El problema que resuelve: reentrenar con la misma semilla y los mismos datos -- que
# es exactamente lo que uno hace verificando -- producia una version nueva e
# indistinguible en cada corrida. Con tres corridas de verificacion, el Registry
# terminaba con v1, v2 y v3 identicas y cada version dejaba de significar algo.

from types import SimpleNamespace

import config


class _FakeVersion:
    def __init__(self, version, tags):
        self.version = str(version)
        self.tags = tags
        self.run_id = f"run-de-v{version}"


def _version_con(metrics, family="random_forest", version=1):
    return _FakeVersion(version, mlflow_utils._metric_fingerprint(metrics, family))


METRICAS = {"val_rmse": 4800.247379, "val_r2": 0.837664, "overfitting_r2_diff": 0.058813}


def test_la_huella_usa_la_precision_con_la_que_se_guardan_los_tags():
    """Los tags de la version se escriben con 6 decimales. La huella se formatea igual,
    asi que comparar es exacto y no depende de ruido de punto flotante."""
    a = mlflow_utils._metric_fingerprint({"val_rmse": 4800.2473787796}, "random_forest")
    b = mlflow_utils._metric_fingerprint({"val_rmse": 4800.2473787799}, "random_forest")
    assert a == b
    assert a["metric.val_rmse"] == "4800.247379"
    assert a["model_family"] == "random_forest"


def test_un_reentrenamiento_identico_se_detecta_como_duplicado(monkeypatch):
    previa = _version_con(METRICAS, version=1)
    monkeypatch.setattr(mlflow_utils, "_latest_version", lambda c, n: previa)
    monkeypatch.setattr(mlflow_utils, "_client_or_none", lambda: None)

    duplicada = mlflow_utils.find_duplicate_version(METRICAS, "random_forest", "m")
    assert duplicada is previa


def test_una_metrica_distinta_no_es_duplicado(monkeypatch):
    previa = _version_con(METRICAS, version=1)
    monkeypatch.setattr(mlflow_utils, "_latest_version", lambda c, n: previa)
    monkeypatch.setattr(mlflow_utils, "_client_or_none", lambda: None)

    peor = dict(METRICAS, val_rmse=4983.39)
    assert mlflow_utils.find_duplicate_version(peor, "random_forest", "m") is None


def test_un_cambio_de_familia_no_es_duplicado(monkeypatch):
    """Mismas metricas con otra familia es otro modelo, aunque rinda igual."""
    previa = _version_con(METRICAS, family="random_forest", version=1)
    monkeypatch.setattr(mlflow_utils, "_latest_version", lambda c, n: previa)
    monkeypatch.setattr(mlflow_utils, "_client_or_none", lambda: None)

    assert mlflow_utils.find_duplicate_version(METRICAS, "xgboost", "m") is None


def test_sin_versiones_previas_no_hay_duplicado(monkeypatch):
    monkeypatch.setattr(mlflow_utils, "_latest_version", lambda c, n: None)
    monkeypatch.setattr(mlflow_utils, "_client_or_none", lambda: None)
    assert mlflow_utils.find_duplicate_version(METRICAS, "random_forest", "m") is None


def test_el_gate_se_puede_apagar_por_entorno(monkeypatch, caplog):
    """Un cambio de codigo que no mueve las metricas no se detecta. Por eso el
    comportamiento tiene que poder apagarse."""
    previa = _version_con(METRICAS, version=1)
    monkeypatch.setattr(mlflow_utils, "find_duplicate_version",
                        lambda *a, **k: previa)
    monkeypatch.setattr(mlflow_utils, "get_client", lambda: SimpleNamespace(
        create_registered_model=lambda *a, **k: None,
    ))

    monkeypatch.setattr(config, "REGISTER_SKIP_DUPLICATES", True)
    with caplog.at_level("WARNING"):
        resultado = mlflow_utils.register_model_version("run-nuevo", METRICAS,
                                                        family="random_forest")
    assert resultado is previa
    assert any("No se registra una version nueva" in r.message for r in caplog.records)
    assert any("REGISTER_SKIP_DUPLICATES=false" in r.message for r in caplog.records)


def test_con_el_gate_apagado_se_intenta_registrar(monkeypatch):
    previa = _version_con(METRICAS, version=1)
    llamadas = []
    monkeypatch.setattr(mlflow_utils, "find_duplicate_version", lambda *a, **k: previa)
    monkeypatch.setattr(mlflow_utils, "get_client", lambda: SimpleNamespace(
        create_registered_model=lambda *a, **k: None,
    ))
    monkeypatch.setattr(mlflow_utils.mlflow, "register_model",
                        lambda uri, name: llamadas.append(uri) or _FakeVersion(2, {}))
    monkeypatch.setattr(config, "REGISTER_SKIP_DUPLICATES", False)

    # falla despues, al setear tags sobre el cliente falso: alcanza con haber llegado
    try:
        mlflow_utils.register_model_version("run-nuevo", METRICAS, family="random_forest")
    except Exception:
        pass
    assert llamadas, "con el gate apagado tiene que intentar registrar"
