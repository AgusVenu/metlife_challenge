# Solución al ML Ops Challenge

Documento de la solución: cómo ejecutar el pipeline, qué se registra en MLflow y
cómo leer el monitoreo.

> **`README.md` es el documento original del equipo de ciencia de datos y no fue
> modificado.** Ahí está la justificación del modelo (por qué XGBoost, por qué
> `log1p`, por qué esas features derivadas) y el EDA. El pipeline de XGBoost —su
> grid, su feature engineering, su transformación del target— **se conserva
> intacto**; lo que se agregó son **alternativas contra las cuales medirlo**, para
> que "el mejor modelo" sea el resultado de una comparación y no de una
> preferencia. Ver [§ 1.1](#11-comparación-entre-familias-de-modelos).
>
> Los supuestos, el análisis forense de los datos de producción y los bugs
> encontrados en el código original están en **[DECISIONS.md](DECISIONS.md)**.

```
db_setup  →  training (MLflow)  →  promote_model  →  scoring + monitoreo
```

---

## Resultado en un vistazo

Corriendo el pipeline sobre los tres lotes de `data/prod/`:

| Lote | Filas | Target | RMSE | R² | PSI máx | Estado | Alertas nuevas |
|------|------:|:------:|-----:|---:|--------:|:------:|---------------:|
| `prod1` | 1.338 | sí | $4.616 | 0,8596 | 0,0406 | **OK** | 0 |
| `prod2` | 1.338 | sí | $1.794.679 | −1,1221 | 0,0406 | **ALERT** | 5 |
| `prod3` | 1.338 | **no** | n/d | n/d | 11,7573 | **ALERT** | 3 |

Modelo en producción: `insurance-charges-regressor`, familia **Random Forest**,
ganadora de la comparación entre cuatro familias (`val_rmse` = $4.800,25). La
columna "alertas nuevas" corresponde a la **primera** corrida de scoring: en la
segunda las mismas ocho pasan a `PERSISTE` (ver
[§ 4.2](#42-alertas-con-estado-qué-es-nuevo-y-qué-se-resolvió)).

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

> **El proyecto no se entrega containerizado.** El `Dockerfile`, el
> `docker-compose.yaml` y el `entrypoint.sh` del repositorio original fueron
> **eliminados**, no adaptados. El motivo está en
> [DECISIONS.md § 5](DECISIONS.md): la máquina de desarrollo no tiene Docker, y
> entregar infraestructura que nunca se ejecutó es peor que no entregarla.
> Todo lo que hacía el `entrypoint.sh` son los cinco comandos de abajo, en orden.

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
python -m pytest tests/ -q      # 149 tests, sin necesidad de DB ni MLflow
python src/db_setup.py          # crea el esquema y carga data/dataset.csv
python src/training.py          # compara 4 familias y registra el ganador en MLflow
python src/promote_model.py     # promueve a production si pasa los gates
python src/scoring.py           # puntúa data/prod/ y genera el monitoreo
```

Correr `src/scoring.py` **una segunda vez** es lo que muestra el sistema de
alertas funcionando: las mismas señales pasan de `NUEVA` a `PERSISTE`
(ver [§ 4.2](#42-alertas-con-estado-qué-es-nuevo-y-qué-se-resolvió)).

Para iterar rápido, reducí el presupuesto de la búsqueda o el catálogo de familias:

```bash
HYPERPARAM_ITERATIONS=20 python src/training.py
TRAIN_MODEL_FAMILIES=xgboost python src/training.py
```

---

## 1. Entrenamiento con tracking

```bash
python src/training.py
```

Cada ejecución abre un run de MLflow en el experimento `insurance-charges` y
registra:

**Parámetros** — hiperparámetros ganadores, semillas (`random_seed`,
`split_seed`), `test_size`, folds de CV, iteraciones de la búsqueda, métrica de
scoring, features de entrada y derivadas, transformación del target, cantidad de
filas y de features antes y después del one-hot.

**Métricas** — `train_*` y `val_*` de RMSE, MAE, R², R² ajustado y MAPE, tanto en
dólares como en escala logarítmica, más `cv_best_rmse_log` y
`overfitting_r2_diff`.

Además, las **19 métricas canónicas** sin prefijo, que son exactamente las mismas
que loguea cada lote de scoring (ver abajo).

**Artefactos**

| Artefacto | Para qué sirve |
|-----------|----------------|
| `model/` | El pipeline serializado, con *signature* e *input example* |
| `baseline_stats.json` | **Distribuciones de referencia del monitoreo** |
| `training_report_*.txt` | Reporte legible de la evaluación |
| `model_metadata_*.json` | Hiperparámetros y métricas en JSON |
| `cv_results_*.csv` | Resultado completo de la búsqueda |
| `feature_importance_*.json` / `.png` | Importancia con los nombres post one-hot |

**Criterio de mejor modelo:** `val_rmse` mínimo (configurable con
`MODEL_SELECTION_METRIC` / `MODEL_SELECTION_MODE`), aplicado **primero entre
familias** dentro de la corrida y después entre corridas. Cada ejecución registra
una versión nueva en el Model Registry con alias `staging` y el tag `model_family`.

### 1.1 Comparación entre familias de modelos

El proyecto original entrenaba una sola familia y justificaba la elección de
XGBoost en un párrafo escrito a mano. Eso no es una comparación de modelos: es una
comparación de hiperparámetros con una preferencia declarada arriba.

Ahora cada ejecución entrena **cuatro familias sobre el mismo split** —el split se
calcula una sola vez, fuera del bucle, para que ninguna dependa de la semilla de
otra— y el ganador es el que gana la métrica:

| Familia | Rol en la comparación |
|---|---|
| `xgboost` | **Incumbente**, con su grid original intacto: la referencia a batir |
| `random_forest` | Bagging en vez de boosting: otro perfil de sesgo/varianza |
| `hist_gradient_boosting` | Otra implementación de boosting; separa "el boosting funciona" de "la implementación de XGBoost funciona" |
| `elasticnet` | **El piso lineal.** Sin un baseline lineal no se puede afirmar que la complejidad no lineal aporta algo: sólo suponerlo |

Las cuatro salen de `scikit-learn` + `xgboost`, ya declarados en
`requirements.txt`. **No se agregó `lightgbm` ni `catboost`**:
`HistGradientBoostingRegressor` cubre ese rol sin sumar una dependencia.

El catálogo vive en **`src/model_zoo.py`** y se recorta por entorno:

```bash
TRAIN_MODEL_FAMILIES=xgboost python src/training.py   # reproduce el pipeline previo
```

`HYPERPARAM_ITERATIONS` dejó de ser el `n_iter` absoluto y pasó a ser un
**presupuesto por familia**: cada una corre `round(presupuesto × su ratio)`
iteraciones, topeadas por la cardinalidad de su grid. Con el catálogo por defecto,
50 se traduce en 50 + 20 + 20 + 20 = **110 iteraciones en total** (550 fits contra
los 250 de antes, ~2,2×). Las cuatro familias corren en ~15 s sobre este dataset.

**Resultado de la corrida de referencia** (`results/model_comparison_*.csv`, que
además se sube como artefacto del run):

| Familia | val_RMSE | val_R² | overfitting | CV RMSE(log) | seg |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **$4.800,25** | **0,8377** | 0,0588 | 0,3599 | 6,2 |
| HistGradientBoosting | $4.821,37 | 0,8362 | 0,0438 | 0,3529 | 1,9 |
| XGBoost | $4.897,22 | 0,8310 | 0,0494 | **0,3522** | 6,8 |
| ElasticNet | $5.526,95 | 0,7848 | 0,0265 | 0,3690 | 0,3 |

Tres lecturas que sólo existen porque hay comparación:

1. **XGBoost reproduce exactamente sus métricas documentadas** ($4.897,22 /
   0,8310). Es la prueba de no-regresión del refactor.
2. **El piso lineal queda $726,70 por detrás.** La complejidad no lineal se
   justifica con una medición, no por defecto.
3. **Los rankings por validación y por CV no coinciden**: gana Random Forest por
   `val_rmse`, pero XGBoost tiene el mejor RMSE de CV. Elegir entre cuatro familias
   sobre el mismo set de validación es una comparación múltiple, así que el ganador
   se lleva algo de ventaja por azar. El pipeline **no cambia el criterio**
   —`resolve_model()` y `promote_model.py` ordenan por `val_rmse` y hay que ser
   coherente con ellos— pero **emite un WARNING explícito** cuando los dos rankings
   discrepan, para que un margen estrecho quede a la vista en vez de escondido:

   ```
   WARNING - El ranking por validacion y el ranking por CV no coinciden: gana
   'random_forest' por val_rmse pero 'xgboost' tiene mejor RMSE de CV
   (0.3522 vs 0.3599). El margen no es solido; tomar la diferencia con cautela.
   ```

El bloque **"Justificación"** del `training_report_*.txt` pasó de ser un párrafo
fijo a derivarse de los números de la corrida. Era la única forma de que no fuera
falso en cuanto ganara otra familia.

### 1.2 La línea de base del README

El `README.md` original publica un modelo con métricas concretas y sus hiperparámetros
ganadores, pero ese modelo no existía como artefacto: el `.pkl` nunca se versionó. Era
una afirmación sin nada que la respaldara.

```bash
python scripts/register_readme_baseline.py --dry-run   # reproduce y compara, sin registrar
python scripts/register_readme_baseline.py             # registra, por única vez
```

El script reconstruye ese modelo desde sus hiperparámetros publicados, sobre el mismo
split (verificado contra el commit inicial: `test_size=0.2, random_state=43`), y
**contrasta el resultado contra lo que el README declara**:

```
                         README      reproducido        delta
  --- validation ---
  r2                     0.8353           0.8353      +0.0000
  rmse                 4,835.00         4,834.74        -0.26
  mae                  2,102.00         2,101.55        -0.45
  mape                  17.7000          17.7036      +0.0036
```

Queda registrado como **`insurance-charges-baseline-ds`** con alias `reference` — un
modelo aparte, no una versión del que sirve en producción. Su run lleva
`pipeline_stage=baseline`, así que `resolve_model()` no puede resolverlo por accidente.

**Y deja ver algo incómodo:** la línea de base ($4.834,74) le gana a la familia
`xgboost` de la comparación ($4.897,22). El código original corría **350 iteraciones**
de búsqueda —por el bug del typo— y el default de hoy son 50. Se verificó qué pasa
dándole a XGBoost ese mismo presupuesto:

| Configuración | val_RMSE | val_R² |
|---|---:|---:|
| `xgboost`, 50 iteraciones (default) | $4.897,22 | 0,8310 |
| `xgboost`, 350 iteraciones (el original) | $4.834,74 | 0,8353 |
| **`random_forest`, 20 iteraciones (ganador)** | **$4.800,25** | **0,8377** |

Con 350 iteraciones XGBoost reproduce exactamente la línea de base **y sigue
perdiendo**. La conclusión de la comparación se sostiene incluso dándole al incumbente
siete veces más presupuesto que al ganador.

**Nombre del modelo registrado.** `MLFLOW_MODEL_NAME` pasó de
`insurance-charges-xgb` a **`insurance-charges-regressor`**: el nombre designa la
tarea, no el algoritmo, y un registry llamado `-xgb` sirviendo un Random Forest
sería engañoso. La familia de cada versión queda en su tag `model_family`. Para
seguir usando el registry anterior alcanza con
`MLFLOW_MODEL_NAME=insurance-charges-xgb`.

### Un solo experimento, cuatro tipos de run

Entrenamiento y scoring comparten el experimento **`insurance-charges`**, porque
loguean las mismas métricas y por lo tanto son comparables. Los runs se
distinguen por el tag `pipeline_stage`:

| `pipeline_stage` | Qué es | ¿Loguea modelo? |
|---|---|:---:|
| `training` | Run principal: el del modelo **ganador** de la comparación | **sí** |
| `training_candidate` | Una familia evaluada en la comparación (anidado) | no |
| `training_trial` | Una combinación de la búsqueda (anidado en su familia) | no |
| `scoring` | El padre (`scope=all_batches`) y uno por lote (`scope=batch`) | no |

```
train_<ts>                        pipeline_stage=training       <- el ganador
├── family_xgboost                pipeline_stage=training_candidate
│   └── xgboost_trial_rank_1..5   pipeline_stage=training_trial
├── family_random_forest          ...
├── family_hist_gradient_boosting ...
└── family_elasticnet             ...
```

Dos decisiones deliberadas sobre ese esquema:

- **Los candidatos no loguean el artefacto del modelo.** `resolve_model()` busca el
  mejor run con `pipeline_stage='training'`, y de esos hay **exactamente uno por
  ejecución**. Que los candidatos no tengan modelo hace *imposible* resolver a uno
  de ellos por accidente, aun si alguien aflojara ese filtro más adelante. Lo que
  sí queda de cada candidato es su `best_params_` y su `cv_results_` completo: son
  reproducibles desde la semilla.
- **Los candidatos no escriben las métricas sin prefijo.** `rmse`, `r2`, `mape` y
  `psi_*` son la serie que cruza etapas (validación → prod1 → prod2). Si cuatro
  candidatos las escribieran, ese gráfico dejaría de significar lo que este
  documento dice que significa. Los candidatos usan `val_*` y `train_*`.

El run padre además loguea una métrica **`family_<nombre>_val_rmse` por familia**,
así que su fila en la tabla de la UI ya muestra la comparación completa sin abrir
los hijos.

Consultas útiles para pegar en el buscador de la UI:

```
tags.monitoring_status = 'ALERT'                              # lotes con problema
tags.pipeline_stage = 'scoring' and metrics.psi_max > 0.25    # drift de entrada
tags.pipeline_stage = 'training'                              # solo los ganadores
tags.pipeline_stage = 'training_candidate'                    # comparar las familias
tags.has_new_alerts = 'true'                                  # corridas con algo nuevo
params.eval_dataset = 'prod1'                                 # la historia de un lote
```

Para separarlos de nuevo alcanza con setear `MLFLOW_EXPERIMENT_TRAINING` y
`MLFLOW_EXPERIMENT_SCORING` por separado en el `.env`.

### Métricas comparables entre entrenamiento y producción

Training y scoring loguean **las mismas 19 claves**, calculadas por una única
función compartida (`monitoring.canonical_metrics`), más un param `eval_dataset`
que da el contexto. Eso permite graficar una sola serie en la UI de MLflow y ver
la degradación de un vistazo:

| `eval_dataset` | `rmse` | `r2` | `adj_r2` | `psi_max` |
|---|---:|---:|---:|---:|
| `validation` | $4.897,22 | 0,8310 | 0,8217 | 0,0587 |
| `prod1` | $4.842,62 | 0,8455 | 0,8439 | 0,0406 |
| `prod2` | $1.794.680,19 | −1,1221 | −1,1445 | 0,0406 |
| `prod3` | n/d *(sin target)* | n/d | n/d | 11,7573 |

Detalle a tener en cuenta al leer la tabla: el `psi_max` de `validation` es el
PSI entre *train* y *validación*, o sea el **piso de ruido** del propio split.
Que valga 0,0587 y `prod1` sólo 0,0406 confirma que `prod1` es genuinamente
estable, y calibra el umbral: alertar por debajo de 0,06 sería alertar por ruido.

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
(`promoted_at`, `promoted_by`, `promotion_reason`, `model_family`). Al promover, la
versión que estaba sirviendo se archiva (`stage=Archived`, `superseded_by`).

**Un entrenamiento idéntico no registra versión nueva.** Reentrenar con la misma
semilla y los mismos datos —lo que uno hace verificando— produce el mismo modelo, y
registrarlo otra vez hace que cada versión deje de significar algo.
`register_model_version` compara la familia y las métricas de selección contra la
última versión; si coinciden, avisa y devuelve la existente:

```
WARNING - No se registra una version nueva: insurance-charges-regressor v3 ya tiene
estas metricas (familia=random_forest, val_rmse=4,800.2474). Reentrenar con la misma
semilla y los mismos datos produce el mismo modelo.
Para registrar igual: REGISTER_SKIP_DUPLICATES=false
```

Limitación asumida: un cambio de código que no mueva las métricas no se detecta. El
run igual queda trazado entero en MLflow con sus artefactos — lo único que se saltea
es la entrada en el Registry.

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
| Historial de alertas en DB | tabla `alert_history` (consultable con `python src/alerts.py`) |
| Runs de MLflow | experimento `insurance-charges` (tag `pipeline_stage=scoring`) |

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

# Alertas abiertas (y su historial completo con --history)
python src/alerts.py
python src/alerts.py --batch prod3 --history
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

Estos son los **defaults globales**; `config/monitoring_rules.json` los afina por
feature y por lote (ver [§ 4.1](#41-reglas-por-feature-y-por-lote)).

Con `FAIL_ON_ALERT=true` el pipeline termina con exit code ≠ 0 si algún lote
queda en `ALERT` (útil como gate de CI/CD). El default es `false`: un `ALERT` es
una señal de monitoreo, no un fallo de ejecución.

### 4.1 Reglas por feature y por lote

Los umbrales de la tabla anterior son el **default global**. Aplicar la misma vara
a `bmi` que a `region`, y medir `prod3` —sin ground truth, con las features
corruptas— con la misma que `prod1`, hace que el semáforo sea a la vez demasiado
ruidoso en unas señales y demasiado permisivo en otras.

**`config/monitoring_rules.json`** afina umbrales sin tocar código ni multiplicar
variables de entorno. Precedencia, del más específico al más general:

```
regla de (lote, feature)  >  regla de lote  >  regla de feature  >  default global
```

Que "lote" gane sobre "feature" es una decisión: una regla de lote es una
afirmación deliberada sobre un dataset concreto, y una de feature es un
refinamiento que vale para todos. Cuando las dos aplican y hay que ser explícito,
la forma inequívoca de resolverlo es escribir la regla de `(lote, feature)`.

Las reglas que se entregan están justificadas en el propio archivo:

| Regla | Umbral | Por qué |
|---|---|---|
| `bmi` | PSI 0,05 / 0,15 (más estricto) | Es la feature de mayor peso y entra tres veces (`bmi`, `bmi_squared`, `bmi_smoker`); además es la que efectivamente se corrompe en producción |
| `sex`, `region` | PSI 0,20 / 0,40 (más laxo) | Categóricas balanceadas de 2 y 4 niveles: su PSI oscila por muestreo y a 0,10 sería ruido, no señal |
| lote `prod3` | PSI 0,05 / 0,15 | Llega **sin ground truth**: no hay performance que pueda desmentir un drift moderado, así que la vara sobre la entrada se endurece |
| `prod3` + `sex` / `region` | PSI 0,20 / 0,40 | La regla de lote sería demasiado estricta para una categórica balanceada; esta es la única forma inequívoca de resolverlo |

**Borrar el archivo es seguro**: sin él, todo se mide con los umbrales globales y
el comportamiento es idéntico al de antes de que existieran las reglas.

Cada excepción aplicada **queda escrita en el reporte**, junto a la señal y en el
pie. Sin eso, un `WARNING` emitido con una regla propia sería indistinguible de uno
emitido con el umbral global y el reporte dejaría de ser auditable. La regla que se
imprime es la que fijó **el umbral que decidió esa señal**, no un resumen de todas
las que aplicaron al lote: una regla de feature puede gobernar el PSI mientras una de
lote gobierna el umbral de esquema, y nombrar la equivocada sería peor que no nombrar
ninguna. El dashboard lee ese mismo estado en vez de recalcularlo:

```
Drift de features (PSI):
  [ OK ] bmi          PSI=0.0049  (regla feature:bmi)
  [ OK ] region       PSI=0.0010  (regla feature:region)
  [ OK ] children     PSI=0.0001

Reglas de umbral aplicadas: feature:bmi, feature:region, feature:sex
```

### 4.2 Alertas con estado: qué es nuevo y qué se resolvió

`batch_monitoring` guarda una fila por lote por corrida. Eso responde *cómo está el
lote hoy*, pero no las dos preguntas que hacen útil a una alerta: **¿esto es
nuevo?** y **¿lo que estaba mal se arregló?**. Un reporte que dice exactamente lo
mismo en cada corrida es un informe, no un sistema de alertas.

La tabla **`alert_history`** le da memoria. La identidad de una alerta es la tupla
`(batch_id, signal_name)` —por ejemplo `("prod3", "drift:bmi")`— y sobre esa clave
hay tres transiciones:

| Situación en esta corrida | Alerta abierta previa | Transición |
|---|---|---|
| Señal en WARNING/ALERT | no existe | **`NEW`** — se abre |
| Señal en WARNING/ALERT | ya existe | **`ONGOING`** — `occurrences += 1` |
| Señal en OK, o ausente | existe | **`RESOLVED`** — se cierra con `resolved_at` |
| Señal en OK | no existe | nada: no se registra ruido |

**La deduplicación es el punto.** `NEW` es el único evento notificable: una alerta
que lleva cinco corridas abierta aparece **una vez** como `ONGOING` con
`occurrences=5`, no como cinco alertas. Un cambio de severidad
(`WARNING` → `ALERT`) se anota en `severity_history` y **no** abre una alerta
nueva: sigue siendo el mismo problema.

Ese invariante está impuesto **por la base**, no por disciplina del código:

```sql
CREATE UNIQUE INDEX idx_alert_open
    ON alert_history (batch_id, signal_name) WHERE state = 'open';
```

Dos corridas consecutivas de `src/scoring.py` sobre los mismos datos:

```
# corrida 1                              # corrida 2
Alertas: 8 nuevas | 0 persisten          Alertas: 0 nuevas | 8 persisten
  [NUEVA]    drift:bmi        (1a vez)     [PERSISTE] drift:bmi  (2 corridas, desde 23:33:00)
```

Si algo saliera `NUEVA` dos veces, la deduplicación estaría rota: ésa es la prueba.

Las transiciones también se loguean en MLflow como métricas `alerts_new`,
`alerts_ongoing` y `alerts_resolved` (por lote y agregadas) más el tag
`has_new_alerts`, así que la serie de alertas es **graficable junto a `rmse` y
`psi_bmi`** en el mismo experimento.

Si la base no responde, `alerts.reconcile()` loguea un WARNING y devuelve vacío: el
historial es valioso, pero no vale abortar un scoring que ya terminó bien y cuyas
predicciones ya están escritas.

---

## Estructura de la solución

Los archivos marcados con **+** son nuevos, los marcados con **~** fueron
modificados y los marcados con **−** fueron eliminados respecto del repositorio
original.

```
metlife-challenge-mlops/
├── src/
│   │  + config.py          Configuración centralizada (env + contrato de datos)
│   │  ~ utils.py           Feature engineering y transformación del target
│   │  + data_loader.py     Lectura robusta de data/prod + validación de contrato
│   │  + monitoring.py      PSI, comparación vs. baseline, semáforo, diagnóstico
│   │  + mlflow_utils.py    Tracking, Model Registry, resolución del modelo
│   │  + model_zoo.py       Catálogo de familias de modelos a comparar
│   │  + alerts.py          Historial de alertas con estado (+ CLI de consulta)
│   │  + dashboard.py       Dashboard HTML autocontenido
│   │  ~ db_setup.py        Esquema (+ batch_predictions, batch_monitoring, alert_history)
│   │  ~ training.py        Comparación entre familias + tracking + registro
│   │  + promote_model.py   Promoción por métricas
│   └  ~ scoring.py         Scoring batch + monitoreo + alertas
├── + tests/                149 tests (pytest, sin DB ni MLflow)
├── + config/               monitoring_rules.json: umbrales por feature y por lote
├── data/                   Sin cambios: dataset.csv y los 3 lotes de prod/
├── notebooks/              Sin cambios: EDA original
├── + scripts/              mlflow_ui.sh · register_readme_baseline.py
├── models/  results/  mlruns/             (generados)
├── + requirements.txt      No existía en el repo original
├── − Dockerfile            Eliminado (ver DECISIONS.md § 5)
├── − docker-compose.yaml   Eliminado
├── − entrypoint.sh         Eliminado
├── ~ .env.template         Variables de MLflow y umbrales de monitoreo
├── ~ .gitignore            Excepciones para requirements.txt y documentación
├── + DECISIONS.md          Supuestos, análisis forense y bugs encontrados
├── + SOLUTION.md           Este archivo
└── README.md               Documento original del equipo de DS (sin modificar)
```

---

## Tests

```bash
python -m pytest tests/ -q          # 149 tests
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
- **`test_model_zoo.py`** — que cada familia arme un pipeline con los steps del
  contrato (`preprocessor` / `model`), que `n_iter` nunca supere la cardinalidad de
  su grid, que sólo la familia lineal escale las numéricas y que un nombre de
  familia desconocido falle fuerte en vez de entrenar de menos en silencio
- **`test_training_selection.py`** — la regla de decisión del mejor modelo:
  `min` / `max`, desempate determinístico por orden de catálogo, el aviso cuando el
  ranking por CV discrepa del de validación, y la importancia de features
  tolerante a familias sin `feature_importances_`
- **`test_monitoring_rules.py`** — los cuatro niveles de precedencia de umbrales,
  la tolerancia a un archivo de reglas ausente o mal formado, y que una regla
  efectivamente cambie el estado de una señal y quede registrada
- **`test_alerts.py`** — la máquina de estados completa: `NEW` → `ONGOING` →
  `RESOLVED`, la reaparición después de resuelta, el escalamiento sin abrir alerta
  nueva, y que un fallo de la base no aborte un scoring que ya terminó bien
- **`test_mlflow_utils.py`** — que un artefacto accesorio que no se pudo subir sea
  un WARNING y no un fallo del pipeline

---

## Sobre el modelo

**El pipeline de XGBoost no se modificó**: se conservan su grid de búsqueda, la
transformación `log1p` del target y el feature engineering originales
(`bmi_smoker`, `age_smoker`, términos cuadráticos y umbrales binarios de obesidad
y edad). La justificación de todas esas decisiones está en el `README.md`
original, y la corrida de XGBoost de esta solución reproduce exactamente sus
métricas documentadas.

Lo que sí cambió es que **XGBoost ahora compite**. Antes era la única familia
entrenada y su elección estaba argumentada en un párrafo escrito a mano; hoy se
mide contra otras tres sobre el mismo split y el reporte de entrenamiento lleva la
tabla comparativa. Ver [§ 1.1](#11-comparación-entre-familias-de-modelos).

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
