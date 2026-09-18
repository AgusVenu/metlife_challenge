# Solución al ML Ops Challenge

Documento de la solución: cómo ejecutar el pipeline, qué se registra en MLflow y
cómo leer el monitoreo.

> **`README.md` es el documento original del equipo de ciencia de datos y no fue
> modificado.** Ahí está la justificación del modelo (por qué XGBoost, por qué
> `log1p`, por qué esas features derivadas) y el EDA, que siguen vigentes: este
> challenge es de MLOps y **el modelo no se tocó**. Todo lo que se agregó
> alrededor —trazabilidad, gobierno del modelo y observabilidad— está acá.
>
> Los supuestos, el análisis forense de los datos de producción y los bugs
> encontrados en el código original están en **[DECISIONS.md](DECISIONS.md)**.

```
db_setup  →  training (MLflow)  →  promote_model  →  scoring + monitoreo
```

---

## Resultado en un vistazo

Corriendo el pipeline sobre los tres lotes de `data/prod/`:

| Lote | Filas | Target | RMSE | R² | PSI máx | Estado |
|------|------:|:------:|-----:|---:|--------:|:------:|
| `prod1` | 1.338 | sí | $4.843 | 0,8455 | 0,0406 | **OK** |
| `prod2` | 1.338 | sí | $1.794.680 | −1,1221 | 0,0406 | **ALERT** |
| `prod3` | 1.338 | **no** | n/d | n/d | 11,7573 | **ALERT** |

Los tres lotes vienen del mismo padrón, pero **dos traen defectos de datos
deliberados**. El pipeline los detecta y, más importante, los **distingue**:

- **`prod2`** — features idénticas a `prod1` (mismo MD5) pero el target está
  multiplicado por 100. Como las features no driftearon, el diagnóstico
  automático concluye que el problema está en la **etiqueta**, no en el modelo:
  > *Las features son estables y el desvío está SOLO en el target: apunta a un
  > problema de CALIDAD DE DATOS en la etiqueta (unidad o escala), no a
  > degradación del modelo.*

- **`prod3`** — al `bmi` le eliminaron el punto decimal (`27.929` → `27929`) y el
  lote **no trae ground truth**, así que no hay ninguna métrica de performance que
  pueda delatarlo. Se detecta por validación de contrato (1.337 de 1.338 filas
  fuera de rango), PSI de features (11,76) y drift de la distribución de
  predicciones (0,60).

- **`prod1`** — usa coma decimal (`14700,80931`). No es corrupción sino formato,
  pero `pd.read_csv()` a secas lo parsea **mal y en silencio**: toma `14700` como
  índice y `80931` como valor, sin lanzar ninguna excepción. El pipeline lo lee
  correctamente y el lote queda en OK.

---

## Instalación y ejecución

### Requisitos

- **Python 3.11** (MLflow 3.x requiere ≥ 3.10; el proyecto original apuntaba a 3.10)
- **PostgreSQL 15+** — se usan **dos bases**: `metlife_db` (datos y resultados
  de negocio) y `mlflow_db` (backend de tracking de MLflow)
- Opcionalmente **Docker** 20.10+

### Opción A — Local *(esta es la vía verificada)*

```bash
# 1. Entorno virtual
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
# (con uv:  uv venv --python 3.11 && uv pip install -r requirements.txt)

# 2. Bases de datos (dos: la de negocio y la de tracking de MLflow)
psql -d postgres -c "CREATE ROLE metlife_user LOGIN PASSWORD 'metlife_pass';"

createdb -O metlife_user metlife_db
psql -d metlife_db -c "GRANT ALL ON SCHEMA public TO metlife_user;"

createdb -O metlife_user mlflow_db
psql -d mlflow_db -c "GRANT ALL ON SCHEMA public TO metlife_user;"

# 3. Configuración
cp .env.template .env
# editar .env:  DB_HOST=localhost   y   DB_PASSWORD=metlife_pass
```

> El archivo de plantilla se llama **`.env.template`** (el README original lo
> menciona como `.env.example`, que nunca existió en el repo).

```bash
# 4. Pipeline completo
python -m pytest tests/ -q      # 68 tests, sin necesidad de DB ni MLflow
python src/db_setup.py          # crea el esquema y carga data/dataset.csv
python src/training.py          # entrena y registra el run en MLflow
python src/promote_model.py     # promueve a production si pasa los gates
python src/scoring.py           # puntúa data/prod/ y genera el monitoreo
```

Para iterar rápido, reducí las iteraciones de la búsqueda:

```bash
HYPERPARAM_ITERATIONS=20 python src/training.py
```

### Opción B — Docker

> ⚠️ **No verificado en este entorno.** Los archivos están actualizados y son
> coherentes con el pipeline local (mismas variables, mismos pasos, mismos
> volúmenes), y el `requirements.txt` que faltaba —y que hacía fallar el build—
> ya existe. Pero la verificación end-to-end de esta solución se hizo en local:
> la máquina de desarrollo no tiene Docker instalado. Ver
> [DECISIONS.md § 5](DECISIONS.md).

```bash
cp .env.template .env          # dejar DB_HOST=postgres

docker compose up --build      # tests → db_setup → training → promote → scoring
docker compose logs -f ml_pipeline

# UI de MLflow (perfil aparte, no se levanta con `up`)
docker compose --profile ui up -d mlflow_ui     # → http://localhost:5000

docker compose down            # agregar -v para resetear también la DB
```

---

## 1. Entrenamiento con tracking

```bash
python src/training.py
```

Cada ejecución abre un run de MLflow en el experimento
`insurance-charges-training` y registra:

**Parámetros** — hiperparámetros ganadores, semillas (`random_seed`,
`split_seed`), `test_size`, folds de CV, iteraciones de la búsqueda, métrica de
scoring, features de entrada y derivadas, transformación del target, cantidad de
filas y de features antes y después del one-hot.

**Métricas** — `train_*` y `val_*` de RMSE, MAE, R², R² ajustado y MAPE, tanto en
dólares como en escala logarítmica, más `cv_best_rmse_log` y
`overfitting_r2_diff`.

**Artefactos**

| Artefacto | Para qué sirve |
|-----------|----------------|
| `model/` | El pipeline serializado, con *signature* e *input example* |
| `baseline_stats.json` | **Distribuciones de referencia del monitoreo** |
| `training_report_*.txt` | Reporte legible de la evaluación |
| `model_metadata_*.json` | Hiperparámetros y métricas en JSON |
| `cv_results_*.csv` | Resultado completo de la búsqueda |
| `feature_importance_*.json` / `.png` | Importancia con los nombres post one-hot |

Además se abren **runs anidados** con las 10 mejores combinaciones de la
búsqueda, para que quede trazada la exploración entera y no sólo el ganador.

**Criterio de mejor modelo:** `val_rmse` mínimo (configurable con
`MODEL_SELECTION_METRIC` / `MODEL_SELECTION_MODE`). Cada corrida registra una
versión nueva en el Model Registry con alias `staging`.

### Backend de tracking

MLflow guarda los metadatos de los runs en **PostgreSQL** (`mlflow_db`) y los
artefactos en el filesystem (`./mlruns`). Dos aclaraciones:

- **Base separada de la de negocio.** MLflow crea **59 tablas** propias; ponerlas
  en `metlife_db` junto a las 4 tablas de la aplicación (`training_dataset`,
  `predictions`, `batch_predictions`, `batch_monitoring`) volvería ilegible el
  esquema. Es el mismo servidor de Postgres, otra base.
- **El URI se deriva de las credenciales `DB_*`,** para no duplicar la contraseña
  en dos variables. Sólo hace falta setear `MLFLOW_TRACKING_URI` a mano para
  apuntar a otro backend (un tracking server remoto, o SQLite en un entorno sin
  Postgres); el código soporta ambos.

Como el URI contiene credenciales, **nunca se imprime en claro**: los logs
muestran la versión enmascarada (`postgresql+psycopg2://metlife_user:***@...`).

### Ver los runs

```bash
./scripts/mlflow_ui.sh          # → http://127.0.0.1:5000
./scripts/mlflow_ui.sh 5001     # en otro puerto
```

El script resuelve el tracking URI desde `src/config.py` y se lo pasa a MLflow
sin imprimirlo. Con backend PostgreSQL el URI contiene la contraseña, así que no
puede quedar en un log ni en el historial del shell.

---

## 2. Promoción del modelo

```bash
python src/promote_model.py              # evalúa la versión en staging
python src/promote_model.py --dry-run    # explica la decisión sin aplicarla
python src/promote_model.py --version 3  # evalúa una versión puntual
```

Registrar y promover son pasos distintos **a propósito**: si el entrenamiento
promoviera solo, cualquier corrida experimental pasaría a servir en producción.

**Gates aplicados:**

| Gate | Condición | Variable |
|------|-----------|----------|
| Absoluto | `val_r2 >= 0.75` | `PROMOTION_MIN_R2` |
| Absoluto | `overfitting_r2_diff < 0.15` | `PROMOTION_MAX_OVERFITTING` |
| Relativo | `val_rmse <= rmse_prod × (1 − mejora)` | `PROMOTION_MIN_IMPROVEMENT` |

La decisión se explica en el log y queda en los tags de la versión
(`promoted_at`, `promoted_by`, `promotion_reason`). Al promover, la versión que
estaba sirviendo se archiva (`stage=Archived`, `superseded_by`).

Ejemplo real de rechazo, entrenando con menos iteraciones de búsqueda:

```
Candidato:   v3   val_rmse=4,983.39  val_r2=0.8250  overfitting=0.0769
Producción:  v2   val_rmse=4,897.22  val_r2=0.8310
  OK: val_r2=0.8250 >= 0.75
  OK: overfitting=0.0769 < 0.15
  RECHAZO: val_rmse=4,983.39 > umbral 4,897.22; el candidato no mejora al modelo actual
DECISION: NO promover la v3. Se mantiene en 'staging'.
```

> **Nota sobre `Staging`/`Production`:** MLflow deprecó los *stages* en 2.9 y los
> eliminó en 3.x. Se usan **aliases** (`staging`, `production`) más un tag `stage`
> con el nombre clásico. El Model Registry necesita un backend con base de datos,
> y por eso el tracking corre sobre **PostgreSQL** y no sobre un file store.

---

## 3. Scoring sobre producción

```bash
python src/scoring.py
```

**Resolución del modelo** — explícita, logueada y propagada a todas las salidas:

```
1. models:/insurance-charges-xgb@production
2. models:/insurance-charges-xgb@staging
3. mejor run del experimento por val_rmse
4. models/best_model.pkl                    (compatibilidad, con WARNING)
```

El `baseline_stats.json` se descarga **del run que produjo ese modelo**, así que
el baseline contra el que se mide drift siempre corresponde al modelo en uso.

Los lotes se descubren por convención de nombre (`dataset_*_feats.csv*`), así que
agregar un `dataset_prod4_feats.csv` no requiere tocar código. Si un lote no tiene
archivo `_target`, se procesa como batch sin etiquetas.

### Salidas

| Salida | Ruta |
|--------|------|
| Predicciones por lote | `results/predictions/predictions_<lote>_<ts>.csv` |
| Reporte de monitoreo | `results/monitoring_report_<ts>.{json,csv,txt}` |
| Dashboard | `results/monitoring_dashboard_<ts>.html` |
| Predicciones en DB | tabla `batch_predictions` |
| Monitoreo en DB | tabla `batch_monitoring` |
| Runs de MLflow | experimento `insurance-charges-scoring` |

**Modo legacy:** `SCORING_MODE=sample python src/scoring.py` reproduce el
comportamiento original (10 filas aleatorias de `training_dataset` → tabla
`predictions`).

---

## 4. Revisión de resultados y monitoreo

```bash
# Reporte en texto, con el semáforo
cat results/monitoring_report_*.txt

# Dashboard en el navegador
open results/monitoring_dashboard_*.html

# Estado por lote, desde la base
psql -h localhost -U metlife_user -d metlife_db -c "
  SELECT batch_id, status, n_rows, has_target,
         round(rmse::numeric,2) AS rmse, round(r2::numeric,4) AS r2,
         round(psi_max::numeric,4) AS psi_max, psi_max_feature,
         model_version, model_source
  FROM batch_monitoring ORDER BY scored_at DESC, batch_id;"
```

### Señales de monitoreo

| Señal | Necesita target | Qué mide |
|-------|:---------------:|----------|
| Contrato de datos | no | Rangos, categorías, nulos, columnas faltantes |
| Drift de features (PSI) | no | Distribución de entrada vs. training |
| Drift de predicciones (PSI) | no | Distribución de `y_pred` vs. validación |
| Performance | **sí** | RMSE / MAE / R² / MAPE vs. validación |
| Desvío del target | **sí** | Media y PSI del target vs. training |

El estado de un lote es el **peor** de todas sus señales: un solo `ALERT` alcanza.
Las tres primeras señales no necesitan ground truth, y son las únicas disponibles
para un lote como `prod3`.

### Umbrales (todos configurables por entorno)

| Señal | WARNING | ALERT |
|-------|---------|-------|
| PSI por feature | > 0,10 | > 0,25 |
| RMSE lote / RMSE validación | > 1,25× | > 1,50× |
| Caída de R² vs. validación | > 0,05 | > 0,15 |
| Desvío de la media del target | > 25 % | > 100 % |
| Filas que violan el contrato | > 0 % | > 1 % |

Con `FAIL_ON_ALERT=true` el pipeline termina con exit code ≠ 0 si algún lote
queda en `ALERT` (útil como gate de CI/CD). El default es `false`: un `ALERT` es
una señal de monitoreo, no un fallo de ejecución.

---

## Estructura de la solución

Los archivos marcados con **+** son nuevos y los marcados con **~** fueron
modificados respecto del repositorio original.

```
metlife-challenge-mlops/
├── src/
│   │  + config.py          Configuración centralizada (env + contrato de datos)
│   │  ~ utils.py           Feature engineering y transformación del target
│   │  + data_loader.py     Lectura robusta de data/prod + validación de contrato
│   │  + monitoring.py      PSI, comparación vs. baseline, semáforo, diagnóstico
│   │  + mlflow_utils.py    Tracking, Model Registry, resolución del modelo
│   │  + dashboard.py       Dashboard HTML autocontenido
│   │  ~ db_setup.py        Esquema (+ batch_predictions, batch_monitoring)
│   │  ~ training.py        Entrenamiento + tracking + registro
│   │  + promote_model.py   Promoción por métricas
│   └  ~ scoring.py         Scoring batch + monitoreo
├── + tests/                68 tests (pytest, sin DB ni MLflow)
├── data/                   Sin cambios: dataset.csv y los 3 lotes de prod/
├── notebooks/              Sin cambios: EDA original
├── + scripts/              init-mlflow-db.sql y mlflow_ui.sh
├── models/  results/  mlruns/             (generados)
├── + requirements.txt      Faltaba en el repo y rompía el build de Docker
├── ~ Dockerfile            Python 3.11, directorios de MLflow
├── ~ docker-compose.yaml   Volúmenes de MLflow y data, servicio mlflow_ui
├── ~ entrypoint.sh         + tests y + promoción en la secuencia
├── ~ .env.template         Variables de MLflow y umbrales de monitoreo
├── ~ .gitignore            Excepciones para requirements.txt y documentación
├── + DECISIONS.md          Supuestos, análisis forense y bugs encontrados
├── + SOLUTION.md           Este archivo
└── README.md               Documento original del equipo de DS (sin modificar)
```

---

## Tests

```bash
python -m pytest tests/ -q          # 68 tests
python -m pytest tests/ -v          # con el detalle de cada caso
```

Cubren las funciones puras, que es donde un bug se paga caro y en silencio:

- **`test_data_loader.py`** — parseo de todas las convenciones decimales,
  detección del fallo silencioso de `pd.read_csv`, descubrimiento de lotes,
  validación del contrato contra los archivos reales del repo
- **`test_monitoring.py`** — PSI (identidad, monotonía, valores fuera de escala,
  categóricas), umbrales en los bordes exactos, serialización del baseline, y los
  tres escenarios del challenge con sus diagnósticos
- **`test_utils.py`** — feature engineering (interacciones, cuadráticos, umbrales)
  y round-trip de `log1p`, que es lo que previene el training/serving skew

---

## Sobre el modelo

**No se modificó**: el challenge es de MLOps. Se conservan el XGBoost, el grid de
búsqueda, la transformación `log1p` del target y el feature engineering
originales (`bmi_smoker`, `age_smoker`, términos cuadráticos y umbrales binarios
de obesidad y edad). La justificación de todas esas decisiones está en el
`README.md` original.

Métricas de la corrida de referencia de esta solución:

| Métrica | Train | Validación | README original (val.) |
|---------|------:|-----------:|-----------------------:|
| R² | 0,8805 | **0,8310** | 0,8353 |
| R² ajustado | 0,8789 | 0,8217 | 0,8276 |
| RMSE | $4.202 | **$4.897** | $4.835 |
| MAE | $1.869 | $2.251 | $2.102 |
| MAPE | 14,22 % | 17,75 % | 17,70 % |

Gap de overfitting (R² train − val) = **0,0494** → sin overfitting significativo.
La diferencia con las métricas del README original se explica por la cantidad de
iteraciones de la búsqueda aleatoria, no por un cambio en el modelo.
