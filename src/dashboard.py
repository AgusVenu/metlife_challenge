"""Dashboard HTML de monitoreo (bonus del challenge).

Genera un archivo autocontenido - sin CDN, sin servidor, sin build - que se abre
con doble clic sobre `results/monitoring_dashboard_*.html`.

Decisiones de visualizacion
---------------------------
- El estado nunca se transmite solo con color: cada badge lleva icono + texto,
  asi el semaforo sigue siendo legible para daltonismo y en impresion B/N.
- Los colores de estado (verde/ambar/rojo) estan reservados para el semaforo y
  no se reutilizan para ninguna serie.
- Las barras de PSI son una sola magnitud, asi que usan un unico tono secuencial
  con el valor numerico impreso al lado y una marca en el umbral de ALERT.
- Todo el contenido tambien esta disponible como tabla, que es la vista que
  funciona con lector de pantalla y al copiar y pegar.
"""

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import config

logger = logging.getLogger(__name__)

# Paleta de estado: fija, reservada, nunca reutilizada para series.
STATUS_COLOR = {
    config.STATUS_OK: "#0ca30c",
    config.STATUS_WARNING: "#fab219",
    config.STATUS_ALERT: "#d03b3b",
}
STATUS_ICON = {
    config.STATUS_OK: "&#10003;",      # check
    config.STATUS_WARNING: "&#9888;",  # triangulo de aviso
    config.STATUS_ALERT: "&#10007;",   # cruz
}

# Transiciones de alerta. NO reusan la paleta de estado: lo que codifican es otra
# cosa (que CAMBIO desde la corrida anterior, no que tan grave es). Solo lo nuevo
# pide accion, asi que es lo unico que lleva un color saturado; lo que persiste va en
# gris porque ya se sabia, y lo resuelto en verde apagado porque es una buena noticia
# que no requiere hacer nada.
TRANSITION_LABEL = {"NEW": "NUEVA", "ONGOING": "PERSISTE", "RESOLVED": "RESUELTA"}
TRANSITION_COLOR = {"NEW": "#b4341f", "ONGOING": "#77756f", "RESOLVED": "#0ca30c"}
TRANSITION_ORDER = {"NEW": 0, "ONGOING": 1, "RESOLVED": 2}

_CSS = """
:root {
  color-scheme: light;
  --surface-0:#f4f3f0; --surface-1:#fcfcfb; --surface-2:#eceae5;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#77756f;
  --line:#dedcd6; --series:#2a78d6; --series-soft:#cde2fb;
  --ok:#0ca30c; --warn:#fab219; --alert:#d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#232321;
    --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8e8c84;
    --line:#333330; --series:#3987e5; --series-soft:#184f95;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#232321;
  --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8e8c84;
  --line:#333330; --series:#3987e5; --series-soft:#184f95;
}
* { box-sizing:border-box; }
body {
  margin:0; padding:0 16px 64px;
  background:var(--surface-0); color:var(--ink);
  font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.wrap { max-width:1060px; margin:0 auto; }
header { padding:40px 0 24px; border-bottom:1px solid var(--line); margin-bottom:28px; }
h1 { margin:0 0 6px; font-size:25px; letter-spacing:-.02em; }
h2 { margin:38px 0 14px; font-size:18px; letter-spacing:-.01em; }
h3 { margin:0; font-size:16px; letter-spacing:-.01em; }
.sub { color:var(--ink-2); font-size:13px; margin:0; }
.meta { display:flex; flex-wrap:wrap; gap:8px 26px; margin-top:16px;
        font-size:12.5px; color:var(--ink-2); }
.meta b { color:var(--ink); font-weight:600; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }

.badge { display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
         border-radius:999px; font-size:12px; font-weight:650; letter-spacing:.02em;
         color:#fff; white-space:nowrap; }
.badge .ic { font-size:12px; line-height:1; }

.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.tile { background:var(--surface-1); border:1px solid var(--line);
        border-radius:12px; padding:16px 18px; }
.tile .k { font-size:11.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); }
.tile .v { font-size:28px; font-weight:680; letter-spacing:-.02em; margin-top:4px; }

.card { background:var(--surface-1); border:1px solid var(--line); border-radius:14px;
        padding:22px 24px; margin-bottom:18px; }
.card-head { display:flex; align-items:center; justify-content:space-between;
             gap:14px; flex-wrap:wrap; margin-bottom:6px; }
.card-sub { color:var(--ink-3); font-size:12.5px; margin:0 0 18px; }
.sec { font-size:11.5px; text-transform:uppercase; letter-spacing:.07em;
       color:var(--ink-3); margin:22px 0 8px; }

table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
th { font-size:11.5px; text-transform:uppercase; letter-spacing:.06em;
     color:var(--ink-3); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tbody tr:last-child td { border-bottom:none; }

.psi-row { display:grid; grid-template-columns:110px 1fr 78px 72px;
           align-items:center; gap:12px; padding:5px 0; }
.psi-name { font-size:13px; color:var(--ink-2); }
.psi-track { position:relative; height:9px; background:var(--surface-2);
             border-radius:5px; overflow:hidden; }
.psi-fill { height:100%; border-radius:5px; background:var(--series); }
.psi-mark { position:absolute; top:-3px; bottom:-3px; width:2px;
            background:var(--ink-3); opacity:.5; }
.psi-val { font-size:12.5px; font-variant-numeric:tabular-nums;
           color:var(--ink-2); text-align:right; }

.diag { background:var(--surface-2); border-left:3px solid var(--series);
        border-radius:0 8px 8px 0; padding:12px 16px; margin-top:20px;
        font-size:13.5px; color:var(--ink); }
.viol { font-size:12.5px; color:var(--ink-2); padding:4px 0;
        display:flex; gap:8px; align-items:flex-start; }
.viol .dot { flex:none; width:8px; height:8px; border-radius:50%; margin-top:5px; }
.legend { color:var(--ink-3); font-size:12px; margin-top:8px; }
.alert-row { font-size:12.5px; color:var(--ink-2); padding:4px 0;
             display:flex; gap:10px; align-items:baseline; }
.alert-tag { flex:none; font-size:10.5px; font-weight:600; letter-spacing:.06em;
             text-transform:uppercase; padding:2px 7px; border-radius:3px;
             color:#fcfcfb; min-width:66px; text-align:center; }
.alert-name { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.alert-ctx { color:var(--ink-3); }
footer { margin-top:44px; padding-top:18px; border-top:1px solid var(--line);
         color:var(--ink-3); font-size:12px; }
@media (max-width:620px) {
  .psi-row { grid-template-columns:86px 1fr 64px; }
  .psi-row .psi-status { display:none; }
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _badge(status: str) -> str:
    return (f'<span class="badge" style="background:{STATUS_COLOR[status]}">'
            f'<span class="ic">{STATUS_ICON[status]}</span>{_esc(status)}</span>')


def _money(value) -> str:
    return f"${value:,.2f}" if isinstance(value, (int, float)) else "n/d"


def _num(value, decimals=4) -> str:
    return f"{value:,.{decimals}f}" if isinstance(value, (int, float)) else "n/d"


def _signals_by_name(batch: Dict[str, Any]) -> Dict[str, dict]:
    """Indexa las senales del lote por nombre.

    El dashboard LEE el estado de la senal en vez de recalcularlo. Recalcularlo con
    los umbrales globales contradiria al semaforo del lote cada vez que una regla por
    feature o por lote este en juego: la tarjeta diria OK donde el pipeline abrio una
    alerta, o al reves.
    """
    return {s["name"]: s for s in batch.get("signals", [])}


def _psi_status(value: float) -> str:
    """Fallback para un PSI sin senal asociada. Usa los umbrales globales."""
    if value > config.DEFAULT_THRESHOLDS.psi_alert:
        return config.STATUS_ALERT
    if value > config.DEFAULT_THRESHOLDS.psi_warn:
        return config.STATUS_WARNING
    return config.STATUS_OK


def _psi_rows(psi: Dict[str, float], signals: Dict[str, dict] = None) -> str:
    """Barras de PSI: una sola magnitud, un solo tono, valor siempre impreso."""
    if not psi:
        return '<p class="sub">Sin features para evaluar.</p>'

    signals = signals or {}
    base = config.DEFAULT_THRESHOLDS

    # Escala fija hasta el doble del umbral de ALERT global: mantiene comparables los
    # lotes y las features entre si, en vez de reescalar cada barra a su propio
    # umbral. El estado real de cada feature lo da el badge, no la marca.
    scale = base.psi_alert * 2
    mark_pct = 100 * base.psi_alert / scale

    rows, con_regla = [], []
    for feature, value in sorted(psi.items(), key=lambda kv: -kv[1]):
        width = min(100.0, 100.0 * value / scale) if scale else 0.0
        signal = signals.get(f"drift:{feature}", {})
        status = signal.get("status") or _psi_status(value)
        origen = signal.get("thresholds_source", "default")
        if origen != "default":
            con_regla.append(f"{feature} ({origen})")
        rows.append(
            '<div class="psi-row">'
            f'<div class="psi-name">{_esc(feature)}</div>'
            '<div class="psi-track">'
            f'<div class="psi-fill" style="width:{width:.1f}%"></div>'
            f'<div class="psi-mark" style="left:{mark_pct:.1f}%"></div>'
            '</div>'
            f'<div class="psi-val">{value:,.4f}</div>'
            f'<div class="psi-status">{_badge(status)}</div>'
            '</div>'
        )

    leyenda = (f'La marca vertical senala el umbral de ALERT por defecto '
               f'(PSI &gt; {base.psi_alert}); WARNING a partir de {base.psi_warn}.')
    if con_regla:
        leyenda += (f' Evaluadas con su propia regla, no con esa marca: '
                    f'<b>{_esc(", ".join(con_regla))}</b>.')
    rows.append(f'<p class="legend">{leyenda}</p>')
    return "".join(rows)


def _metrics_table(batch: Dict[str, Any]) -> str:
    metrics, baseline = batch.get("metrics") or {}, batch.get("baseline_metrics") or {}
    if not metrics:
        return ('<p class="sub">Este lote no trae ground truth, asi que no hay '
                'metricas de performance. El seguimiento se apoya en el contrato '
                'de datos y en el drift de features y de predicciones.</p>')

    rows = []
    for key, label, fmt in [("rmse", "RMSE", _money), ("mae", "MAE", _money),
                            ("r2", "R&sup2;", lambda v: _num(v, 4)),
                            ("mape", "MAPE", lambda v: f"{v:,.2f}%")]:
        base = baseline.get(f"val_{key}")
        delta = "n/d"
        if isinstance(base, (int, float)) and isinstance(metrics.get(key), (int, float)) and base:
            delta = f"x{metrics[key] / base:,.2f}" if key != "r2" else f"{metrics[key] - base:+,.4f}"
        rows.append(
            f"<tr><td>{label}</td><td class='num'>{fmt(metrics.get(key))}</td>"
            f"<td class='num'>{fmt(base) if base is not None else 'n/d'}</td>"
            f"<td class='num'>{delta}</td></tr>"
        )

    return (
        "<table><thead><tr><th>Metrica</th><th class='num'>Lote</th>"
        "<th class='num'>Validacion</th><th class='num'>Relacion</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _violations(batch: Dict[str, Any], signals: Dict[str, dict] = None) -> str:
    violations = batch.get("violations") or []
    if not violations:
        return '<p class="sub">Sin violaciones del contrato de datos.</p>'

    # La severidad se lee de la senal, no del campo `severity` de la violacion:
    # `data_loader` resuelve el suyo con los umbrales globales porque no sabe a que
    # lote pertenece el archivo, y `monitoring.evaluate_schema` lo recalcula con los
    # del lote. Mostrar el primero contradiria al semaforo de la misma tarjeta.
    signals = signals or {}
    items = []
    for violation in violations:
        signal = signals.get(f"schema:{violation['column']}:{violation['kind']}", {})
        status = signal.get("status") or violation["severity"]
        color = STATUS_COLOR.get(status, STATUS_COLOR[config.STATUS_ALERT])
        items.append(
            f'<div class="viol"><span class="dot" style="background:{color}"></span>'
            f'<span><b>{_esc(violation["column"])}</b> &mdash; {_esc(violation["detail"])} '
            f'({violation["n_rows"]:,} filas, {violation["pct_rows"]:.1f}%) '
            f'{_badge(status)}</span></div>'
        )
    return "".join(items)


def _alerts(transitions) -> str:
    """Bloque de alertas de un lote: separa lo nuevo de lo que ya se sabia.

    Tres estados distintos, que no hay que colapsar: sin informacion de alertas en el
    reporte, reconciliado sin novedades, y no reconciliable porque el lote fallo.
    """
    if transitions is not None and not isinstance(transitions, (list, tuple)):
        return '<p class="legend">sin historial de alertas para este lote</p>'
    if transitions is None:
        return ('<p class="legend">el lote no se pudo procesar: no hubo reconciliacion '
                'de alertas, y las previas de este lote siguen abiertas</p>')
    if not transitions:
        return '<p class="legend">sin cambios respecto de la corrida anterior</p>'

    rows = []
    for transition in sorted(transitions,
                             key=lambda x: (TRANSITION_ORDER.get(x.get("transition"), 9),
                                            x.get("signal_name", ""))):
        kind = transition.get("transition", "")
        occurrences = transition.get("occurrences", 1)
        if kind == "NEW":
            contexto = "primera vez que aparece"
        elif kind == "ONGOING":
            contexto = (f"{occurrences} corridas, desde "
                        f"{_esc(transition.get('first_seen', 'n/d'))}")
            previa, actual = transition.get("escalated_from"), transition.get("severity")
            if previa and actual and previa != actual:
                subio = (config.STATUS_ORDER.index(actual) > config.STATUS_ORDER.index(previa)
                         if actual in config.STATUS_ORDER and previa in config.STATUS_ORDER
                         else True)
                contexto += (f" &middot; {'escalada' if subio else 'bajo'} "
                             f"{_esc(previa)} &rarr; {_esc(actual)}")
        else:
            contexto = f"estuvo {occurrences} corrida(s) abierta"

        color = TRANSITION_COLOR.get(kind, "#77756f")
        rows.append(
            f'<div class="alert-row">'
            f'<span class="alert-tag" style="background:{color}">'
            f'{TRANSITION_LABEL.get(kind, kind)}</span>'
            f'<span class="alert-name">{_esc(transition.get("signal_name", ""))}</span>'
            f'<span class="alert-ctx">{contexto}</span></div>'
        )
    return "".join(rows)


def _batch_card(batch: Dict[str, Any], transitions=None) -> str:
    signals = _signals_by_name(batch)
    predictions = batch.get("prediction_summary") or {}
    pred_line = ""
    if predictions:
        pred_line = (
            f'<p class="sub">Predicciones: media {_money(predictions.get("mean"))} &middot; '
            f'desvio {_money(predictions.get("std"))} &middot; '
            f'rango {_money(predictions.get("min"))} a {_money(predictions.get("max"))}</p>'
        )

    return f"""
<section class="card">
  <div class="card-head">
    <h3>{_esc(batch['batch_id'])}</h3>
    {_badge(batch['status'])}
  </div>
  <p class="card-sub">{batch['n_rows']:,} filas &middot;
     ground truth: {'si' if batch['has_target'] else 'no'} &middot;
     scoreado {_esc(batch.get('scored_at', ''))}</p>

  <div class="sec">Performance</div>
  {_metrics_table(batch)}

  <div class="sec">Drift de features (PSI)</div>
  {_psi_rows(batch.get('psi') or {}, signals)}
  {pred_line}

  <div class="sec">Contrato de datos</div>
  {_violations(batch, signals)}

  <div class="sec">Alertas &mdash; que cambio desde la corrida anterior</div>
  {_alerts(transitions)}

  <div class="diag"><b>Diagnostico:</b> {_esc(batch.get('diagnosis', ''))}</div>
</section>"""


def _summary_table(batches, by_batch: Dict[str, Any] = None) -> str:
    by_batch = by_batch or {}
    rows = []
    for batch in batches:
        metrics = batch.get("metrics") or {}
        transitions = by_batch.get(batch["batch_id"]) or []
        nuevas = sum(1 for x in transitions if x.get("transition") == "NEW")
        celda = (f"<b style='color:{TRANSITION_COLOR['NEW']}'>{nuevas}</b>"
                 if nuevas else "0")
        rows.append(
            f"<tr><td><code>{_esc(batch['batch_id'])}</code></td>"
            f"<td>{_badge(batch['status'])}</td>"
            f"<td class='num'>{batch['n_rows']:,}</td>"
            f"<td>{'si' if batch['has_target'] else 'no'}</td>"
            f"<td class='num'>{_money(metrics.get('rmse'))}</td>"
            f"<td class='num'>{_num(metrics.get('r2'), 4)}</td>"
            f"<td class='num'>{_num(batch.get('psi_max'), 4)}</td>"
            f"<td>{_esc(batch.get('psi_max_feature') or 'n/d')}</td>"
            f"<td class='num'>{celda}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Lote</th><th>Estado</th><th class='num'>Filas</th>"
        "<th>Target</th><th class='num'>RMSE</th><th class='num'>R&sup2;</th>"
        "<th class='num'>PSI max</th><th>Feature</th>"
        "<th class='num'>Alertas nuevas</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _thresholds_footer(consolidated: Dict[str, Any]) -> str:
    """Umbrales por defecto, mas las excepciones que efectivamente se aplicaron.

    Imprimir solo los globales seria enganoso cuando un lote se midio con una regla
    propia: el lector deduciria umbrales que a ese lote no se le aplicaron.
    """
    base = config.DEFAULT_THRESHOLDS
    texto = (
        f"Umbrales por defecto &mdash; PSI: WARNING &gt; {base.psi_warn}, "
        f"ALERT &gt; {base.psi_alert} &middot; "
        f"RMSE: WARNING &gt; {base.perf_warn_ratio}x, ALERT &gt; {base.perf_alert_ratio}x &middot; "
        f"caida de R&sup2;: WARNING &gt; {base.r2_warn_drop}, ALERT &gt; {base.r2_alert_drop} &middot; "
        f"filas invalidas: ALERT &gt; {base.schema_alert_pct}%."
    )

    excepciones = {}
    for batch in consolidated.get("batches", []):
        for signal in batch.get("signals", []):
            origen = signal.get("thresholds_source", "default")
            if origen != "default":
                excepciones.setdefault((batch["batch_id"], origen), set()).add(signal["name"])

    if excepciones:
        detalle = " &middot; ".join(
            f"<code>{_esc(batch_id)}</code>: regla <code>{_esc(origen)}</code> "
            f"sobre {_esc(', '.join(sorted(nombres)))}"
            for (batch_id, origen), nombres in sorted(excepciones.items())
        )
        texto += (f"<br>Excepciones aplicadas (<code>config/monitoring_rules.json</code>) "
                  f"&mdash; {detalle}")
    return texto


def render(consolidated: Dict[str, Any]) -> str:
    """Devuelve el HTML completo del dashboard."""
    model = consolidated.get("model_info") or {}
    counts = consolidated.get("status_counts") or {}
    overall = consolidated.get("overall_status", config.STATUS_OK)

    tiles = "".join(
        f'<div class="tile"><div class="k">{status}</div>'
        f'<div class="v" style="color:{STATUS_COLOR[status]}">{counts.get(status, 0)}</div></div>'
        for status in config.STATUS_ORDER
    )
    tiles += (f'<div class="tile"><div class="k">Lotes</div>'
              f'<div class="v">{consolidated.get("n_batches", 0)}</div></div>')

    alerts_block = consolidated.get("alerts") or {}
    by_batch = alerts_block.get("by_batch") or {}
    resumen = alerts_block.get("summary") or {}
    if resumen:
        # El semaforo dice como esta cada lote; esto dice que cambio, que es lo unico
        # que amerita que alguien abra el dashboard hoy y no ayer.
        tiles += (
            f'<div class="tile"><div class="k">Alertas nuevas</div>'
            f'<div class="v" style="color:{TRANSITION_COLOR["NEW"]}">'
            f'{resumen.get("new", 0)}</div></div>'
            f'<div class="tile"><div class="k">Persisten</div>'
            f'<div class="v" style="color:{TRANSITION_COLOR["ONGOING"]}">'
            f'{resumen.get("ongoing", 0)}</div></div>'
            f'<div class="tile"><div class="k">Resueltas</div>'
            f'<div class="v" style="color:{TRANSITION_COLOR["RESOLVED"]}">'
            f'{resumen.get("resolved", 0)}</div></div>'
        )

    _SIN_INFO = object()
    cards = "".join(
        _batch_card(b, by_batch.get(b["batch_id"], _SIN_INFO))
        for b in consolidated.get("batches", [])
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitoreo de Scoring</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Monitoreo de scoring &mdash; costos de seguro medico</h1>
    <p class="sub">Estado global {_badge(overall)} &middot; generado el
       {_esc(consolidated.get('generated_at', ''))}</p>
    <div class="meta">
      <span>Modelo <b>{_esc(model.get('model_name', 'n/d'))} v{_esc(model.get('model_version', 'n/d'))}</b></span>
      <span>Origen <b>{_esc(model.get('source', 'n/d'))}</b></span>
      <span>Run de training <b><code>{_esc(model.get('run_id', 'n/d'))}</code></b></span>
      <span>RMSE de validacion <b>{_money(model.get('val_rmse'))}</b></span>
    </div>
  </header>

  <div class="tiles">{tiles}</div>

  <h2>Resumen por lote</h2>
  {_summary_table(consolidated.get('batches', []), by_batch)}

  <h2>Detalle</h2>
  {cards}

  <footer>
    {_thresholds_footer(consolidated)}
    <br>Generado por <code>src/dashboard.py</code> del pipeline de scoring.
  </footer>
</div>
</body>
</html>"""


def write_dashboard(consolidated: Dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(consolidated), encoding="utf-8")
    logger.info("Dashboard generado en: %s", output_path)
    return output_path


if __name__ == "__main__":
    # Permite regenerar el dashboard desde un JSON ya existente:
    #   python src/dashboard.py results/monitoring_report_<ts>.json
    import sys
    source = Path(sys.argv[1])
    data = json.loads(source.read_text(encoding="utf-8"))
    write_dashboard(data, source.with_suffix(".html"))
