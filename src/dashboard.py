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


def _psi_status(value: float) -> str:
    if value > config.PSI_ALERT:
        return config.STATUS_ALERT
    if value > config.PSI_WARN:
        return config.STATUS_WARNING
    return config.STATUS_OK


def _psi_rows(psi: Dict[str, float]) -> str:
    """Barras de PSI: una sola magnitud, un solo tono, valor siempre impreso."""
    if not psi:
        return '<p class="sub">Sin features para evaluar.</p>'

    # Escala fija hasta el doble del umbral de ALERT: mantiene comparables los
    # lotes entre si en vez de reescalar cada tarjeta a su propio maximo.
    scale = config.PSI_ALERT * 2
    mark_pct = 100 * config.PSI_ALERT / scale

    rows = []
    for feature, value in sorted(psi.items(), key=lambda kv: -kv[1]):
        width = min(100.0, 100.0 * value / scale) if scale else 0.0
        status = _psi_status(value)
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
    rows.append(
        f'<p class="legend">La marca vertical senala el umbral de ALERT '
        f'(PSI &gt; {config.PSI_ALERT}); WARNING a partir de {config.PSI_WARN}.</p>'
    )
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


def _violations(batch: Dict[str, Any]) -> str:
    violations = batch.get("violations") or []
    if not violations:
        return '<p class="sub">Sin violaciones del contrato de datos.</p>'
    items = []
    for violation in violations:
        color = STATUS_COLOR.get(violation["severity"], STATUS_COLOR[config.STATUS_ALERT])
        items.append(
            f'<div class="viol"><span class="dot" style="background:{color}"></span>'
            f'<span><b>{_esc(violation["column"])}</b> &mdash; {_esc(violation["detail"])} '
            f'({violation["n_rows"]:,} filas, {violation["pct_rows"]:.1f}%) '
            f'{_badge(violation["severity"])}</span></div>'
        )
    return "".join(items)


def _batch_card(batch: Dict[str, Any]) -> str:
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
  {_psi_rows(batch.get('psi') or {})}
  {pred_line}

  <div class="sec">Contrato de datos</div>
  {_violations(batch)}

  <div class="diag"><b>Diagnostico:</b> {_esc(batch.get('diagnosis', ''))}</div>
</section>"""


def _summary_table(batches) -> str:
    rows = []
    for batch in batches:
        metrics = batch.get("metrics") or {}
        rows.append(
            f"<tr><td><code>{_esc(batch['batch_id'])}</code></td>"
            f"<td>{_badge(batch['status'])}</td>"
            f"<td class='num'>{batch['n_rows']:,}</td>"
            f"<td>{'si' if batch['has_target'] else 'no'}</td>"
            f"<td class='num'>{_money(metrics.get('rmse'))}</td>"
            f"<td class='num'>{_num(metrics.get('r2'), 4)}</td>"
            f"<td class='num'>{_num(batch.get('psi_max'), 4)}</td>"
            f"<td>{_esc(batch.get('psi_max_feature') or 'n/d')}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Lote</th><th>Estado</th><th class='num'>Filas</th>"
        "<th>Target</th><th class='num'>RMSE</th><th class='num'>R&sup2;</th>"
        "<th class='num'>PSI max</th><th>Feature</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


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

    cards = "".join(_batch_card(b) for b in consolidated.get("batches", []))

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
  {_summary_table(consolidated.get('batches', []))}

  <h2>Detalle</h2>
  {cards}

  <footer>
    Umbrales &mdash; PSI: WARNING &gt; {config.PSI_WARN}, ALERT &gt; {config.PSI_ALERT} &middot;
    RMSE: WARNING &gt; {config.PERF_WARN_RATIO}x, ALERT &gt; {config.PERF_ALERT_RATIO}x &middot;
    caida de R&sup2;: WARNING &gt; {config.R2_WARN_DROP}, ALERT &gt; {config.R2_ALERT_DROP} &middot;
    filas invalidas: ALERT &gt; {config.SCHEMA_ALERT_PCT}%.
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
