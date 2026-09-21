# Informe del ML Ops Challenge

**Predicción de costos médicos — evolución a un pipeline productivo y observable**

| | |
|---|---|
| **Repositorio** | `metlife-challenge-mlops` |
| **Stack** | Python 3.11 · MLflow 3.16 · PostgreSQL 17 · scikit-learn 1.9 · XGBoost 3.2 |
| **Entregables** | `INFORME.md` (este documento) · `informe.html` (versión navegable) · `SOLUTION.md` (guía de ejecución) · `DECISIONS.md` (decisiones y hallazgos) |
| **Estado** | Pipeline completo verificado end-to-end en local · 172 tests |

---

## Índice

1. [Qué pedía el challenge](#1-qué-pedía-el-challenge)
2. [Resumen ejecutivo](#2-resumen-ejecutivo)
3. [Análisis forense de los datos de producción](#3-análisis-forense-de-los-datos-de-producción)
4. [Entrenamiento con trazabilidad](#4-entrenamiento-con-trazabilidad)
5. [Comparación entre familias de modelos](#5-comparación-entre-familias-de-modelos)
6. [Gobierno del modelo: registro y promoción](#6-gobierno-del-modelo-registro-y-promoción)
7. [Scoring batch sobre producción](#7-scoring-batch-sobre-producción)
8. [Monitoreo: señales, reglas y alertas](#8-monitoreo-señales-reglas-y-alertas)
9. [Bugs encontrados en el código original](#9-bugs-encontrados-en-el-código-original)
10. [Desviaciones deliberadas](#10-desviaciones-deliberadas)
11. [Verificación contra los criterios de aceptación](#11-verificación-contra-los-criterios-de-aceptación)
12. [Cómo reproducirlo](#12-cómo-reproducirlo)
13. [Estructura de la solución](#13-estructura-de-la-solución)

---

## 1. Qué pedía el challenge

El repositorio de partida fue escrito por un equipo de ciencia de datos: entrenaba un
XGBoost para predecir costos médicos y hacía un scoring de muestra sobre el propio set
de entrenamiento. El encargo era evolucionarlo hacia un enfoque **productivo y
observable**, con dos ejes:

**Entrenamiento.** Integrar MLflow Tracking; registrar parámetros, al menos dos métricas
de regresión y artefactos por ejecución; definir y registrar un criterio explícito de
"mejor modelo"; centralizar la configuración en variables de entorno.

**Scoring.** Consumir el mejor artefacto registrado —no un `.pkl` suelto—; procesar los
lotes de `data/prod/`, incluido uno sin etiquetas; persistir resultados reproducibles;
calcular métricas por lote cuando haya ground truth; y simular monitoreo con una señal
de drift y un reporte consolidado con estado `OK` / `WARNING` / `ALERT` por lote.

Como bonus opcional: Model Registry con etapas, script de promoción por métricas,
dashboards y tests unitarios. **Los cuatro están implementados.**

> El enunciado cierra con una instrucción que gobierna buena parte de este informe:
> *"Si detectás ambigüedades, asumí una decisión razonable y documentala explícitamente
> en la solución."* Las decisiones no obvias están justificadas acá y en `DECISIONS.md`;
> donde la solución se desvía del enunciado, la desviación se declara (§ 10).

---

## 2. Resumen ejecutivo

El pipeline quedó en cuatro etapas encadenadas:

```
db_setup  →  training (MLflow)  →  promote_model  →  scoring + monitoreo
```

**Resultado sobre los tres lotes de producción:**

| Lote | Filas | Ground truth | RMSE | R² | PSI máx | Estado | Alertas nuevas |
|------|------:|:------------:|-----:|---:|--------:|:------:|---------------:|
| `prod1` | 1.338 | sí | $4.616 | 0,8596 | 0,0406 | **OK** | 0 |
| `prod2` | 1.338 | sí | $1.794.679 | −1,1221 | 0,0406 | **ALERT** | 5 |
| `prod3` | 1.338 | **no** | n/d | n/d | 11,7573 | **ALERT** | 3 |

Modelo en producción: `insurance-charges-regressor`, familia **Random Forest**, ganadora
de una comparación entre cuatro familias (`val_rmse` = $4.800,25).

**Lo que distingue a esta solución** no es que detecte los tres lotes, sino que los
**diferencie**: `prod2` y `prod3` están los dos en `ALERT`, pero por causas distintas, y
el reporte lo dice en prosa accionable. Un monitoreo que sólo enciende una luz roja
obliga a alguien a hacer el diagnóstico a mano; éste lo entrega hecho.

Cuatro decisiones de diseño sostienen el resto:

1. **El baseline de monitoreo viaja como artefacto del run que produjo el modelo.** El
   drift se mide siempre contra la distribución con la que se entrenó *ese* modelo, no
   contra el último entrenamiento que alguien haya corrido en la máquina.
2. **Entrenamiento y scoring loguean el mismo conjunto canónico de 19 métricas.** Eso
   permite graficar validación → prod1 → prod2 como una sola serie en MLflow.
3. **Registrar y promover son dos pasos distintos.** Un entrenamiento cualquiera no puede
   pisar el modelo que está sirviendo.
4. **El monitoreo tiene memoria.** El semáforo dice cómo está cada lote hoy; el historial
   de alertas dice qué cambió desde la corrida anterior, que es lo único que amerita que
   alguien lo mire.

---

## 3. Análisis forense de los datos de producción

Antes de escribir una línea de monitoreo hubo que entender qué traían los archivos. Los
tres lotes salen del mismo padrón de 1.338 registros, pero **dos vienen con defectos
deliberados** y el tercero con una trampa de formato.

### 3.1 `prod1` — coma decimal: formato, no corrupción

```
charges
14700,80931
1540,261607
```

El target usa **coma decimal**. `pd.read_csv()` a secas no lanza ninguna excepción: toma
`14700` como índice y `80931` como valor, y devuelve un DataFrame con la forma correcta y
los números equivocados. Es el peor tipo de bug — silencioso y plausible.

`data_loader.read_single_column_numeric()` lee la columna **como texto plano** y
normaliza la convención decimal explícitamente. Hay un test que demuestra el fallo del
camino ingenuo, para que la razón de existir de ese código quede documentada en el propio
código.

### 3.2 `prod2` — el target multiplicado por 100

```
charges
1470080,931       ← el mismo valor que prod1, ×100
```

Las features de `prod2` son **idénticas** a las de `prod1` (mismo MD5). Sólo cambió el
target: está multiplicado por 100, como si el archivo viniera en centavos.

Esto es lo que hace útil el diagnóstico automático: las features **no driftearon**, así
que el problema no puede ser el modelo. El reporte concluye:

> *Las features son estables y el desvío está SOLO en el target: apunta a un problema de
> CALIDAD DE DATOS en la etiqueta (unidad o escala), no a degradación del modelo.
> Corregir el proceso que genera el archivo de target antes de considerar un
> reentrenamiento.*

Sin esa distinción, un RMSE de $1,79 M dispararía un reentrenamiento sobre datos
corruptos.

### 3.3 `prod3` — `bmi` sin punto decimal, y sin ground truth

```
age,sex,bmi,children,smoker,region
20,female,27929,0,yes,southwest      ← era 27.929
```

Al `bmi` le falta el punto decimal, con un factor que además **varía por fila** (×1000 en
1.214 filas, ×100 en 105, ×10 en 18, ×1 en 1). Y el lote **no trae target**, así que
ninguna métrica de performance puede delatarlo.

Se detecta por tres señales que no necesitan etiquetas:

- **contrato de datos**: 1.337 de 1.338 filas con `bmi` fuera del rango plausible
- **PSI de features**: 11,76 sobre `bmi`
- **drift de predicciones**: la distribución de `y_pred` se corrió respecto de validación

### 3.4 Decisión transversal: detectar y alertar, no reparar

Sería técnicamente trivial dividir `prod3.bmi` por 1000 y seguir. **No se hace**, por tres
razones:

1. El factor varía por fila: cualquier corrección automática sería una conjetura.
2. Reparar en silencio oculta el problema del proveedor de datos, que es donde hay que
   arreglarlo.
3. Un pipeline que corrige datos de entrada sin avisar produce predicciones que nadie
   puede auditar después.

El pipeline **puntúa igual** el lote —las predicciones se generan y se persisten— pero lo
marca `ALERT` y explica por qué no son confiables.

---

## 4. Entrenamiento con trazabilidad

```bash
python src/training.py
```

### 4.1 Qué se registra

**Parámetros** — hiperparámetros ganadores, semillas (`random_seed`, `split_seed`),
`test_size`, folds de CV, iteraciones de la búsqueda, métrica de scoring, features de
entrada y derivadas, transformación del target, filas y features antes y después del
one-hot, familia de modelo y familias comparadas.

**Métricas** — `train_*` y `val_*` de RMSE, MAE, R², R² ajustado y MAPE, en dólares y en
escala logarítmica, más `cv_best_rmse_log` y `overfitting_r2_diff`. Y las **19 métricas
canónicas sin prefijo**, que son exactamente las mismas que loguea cada lote de scoring.

**Artefactos**

| Artefacto | Para qué sirve |
|---|---|
| `model/` | El pipeline serializado, con *signature* e *input example* |
| `baseline_stats.json` | **Distribuciones de referencia del monitoreo** |
| `training_report_*.txt` | Reporte legible con la comparación entre familias |
| `model_comparison_*.csv` | Tabla comparativa completa |
| `model_metadata_*.json` | Hiperparámetros y métricas |
| `cv_results_*.csv` | Resultado completo de la búsqueda de la familia ganadora |
| `feature_importance_*.json` / `.png` | Importancia con los nombres post one-hot |

### 4.2 Un experimento, cuatro tipos de run

Entrenamiento y scoring comparten el experimento `insurance-charges`, porque loguean las
mismas métricas y por lo tanto son comparables. Los runs se distinguen por
`pipeline_stage`:

```
train_<ts>                        pipeline_stage=training       ← el ganador
├── family_xgboost                pipeline_stage=training_candidate
│   └── xgboost_trial_rank_1..5   pipeline_stage=training_trial
├── family_random_forest          ...
├── family_hist_gradient_boosting ...
└── family_elasticnet             ...
```

Dos decisiones deliberadas sobre ese esquema:

- **Los candidatos no loguean el artefacto del modelo.** `resolve_model()` busca el mejor
  run con `pipeline_stage='training'`, y de esos hay **exactamente uno por ejecución**.
  Que los candidatos no tengan modelo hace *imposible* resolver a uno de ellos por
  accidente, aun si alguien aflojara ese filtro más adelante.
- **Los candidatos no escriben las métricas sin prefijo.** `rmse`, `r2`, `mape` y `psi_*`
  son la serie que cruza etapas. Si cuatro candidatos las escribieran, el gráfico de
  validación → prod1 → prod2 dejaría de significar lo que este informe dice que significa.

El run padre loguea además una métrica `family_<nombre>_val_rmse` por familia, así que su
fila en la tabla de MLflow ya muestra la comparación completa sin abrir los hijos.

### 4.3 Backend de tracking

**PostgreSQL, no `file://`**, porque el Model Registry exige un backend con base de datos.
Se usa una base **separada** de la de negocio (`mlflow_db` vs `metlife_db`): MLflow crea
59 tablas propias y mezclarlas con `training_dataset` o `batch_monitoring` vuelve
ilegible el esquema de la aplicación.

El tracking URI se **deriva** de las credenciales `DB_*` para no duplicar la contraseña en
dos variables, y se **enmascara** en todo log, reporte y mensaje de consola
(`config.mask_uri`). Por eso la UI se levanta con `./scripts/mlflow_ui.sh`, que resuelve
el URI desde `src/config.py` sin que aparezca en pantalla ni en el historial del shell.

---

## 5. Comparación entre familias de modelos

### 5.1 El problema

El proyecto original entrenaba **una sola familia** y justificaba la elección de XGBoost
con un párrafo escrito a mano en el reporte de entrenamiento. Los runs anidados que
quedaban en MLflow eran variantes del mismo modelo: eso es una comparación de
**hiperparámetros**, no de modelos. El criterio de "mejor modelo" que pide el enunciado
nunca llegaba a elegir entre alternativas reales.

### 5.2 Las cuatro familias

Cada una responde una pregunta distinta, y esa pregunta está declarada en el campo
`rationale` de su especificación (`src/model_zoo.py`):

| Familia | Pregunta que responde |
|---|---|
| `xgboost` | Es el incumbente: la referencia a batir, con su grid original intacto |
| `random_forest` | ¿El problema tiene la señal secuencial que el boosting supone? |
| `hist_gradient_boosting` | ¿La ventaja es del boosting, o de la implementación de XGBoost? |
| `elasticnet` | ¿La complejidad no lineal aporta algo, o sólo lo suponíamos? |

Las cuatro salen de `scikit-learn` + `xgboost`, **ya declarados** en `requirements.txt`.
No se agregó LightGBM ni CatBoost: `HistGradientBoostingRegressor` cubre ese rol sin sumar
una dependencia.

Un detalle que no es cosmético: `ElasticNet` es la única familia que declara
`needs_scaling=True`. El preprocesador original hacía `passthrough` de las numéricas, y
entre ellas conviven `age_squared` (hasta ~10.000) y `children` (hasta 5). Con esas
escalas, la penalización L1/L2 castiga desparejo y la evaluación de la familia lineal no
sería honesta. Los modelos de árboles son invariantes a la escala y conservan el
`passthrough` original **bit a bit** — que es lo que permite comparar la corrida de
XGBoost contra sus métricas ya documentadas.

### 5.3 Resultado

| Familia | `n_iter` | val_RMSE | val_R² | overfitting | CV RMSE(log) | seg |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 20/144 | **$4.800,25** | **0,8377** | 0,0588 | 0,3599 | 6,2 |
| HistGradientBoosting | 20/432 | $4.821,37 | 0,8362 | 0,0438 | 0,3529 | 2,0 |
| XGBoost | 50/576 | $4.897,22 | 0,8310 | 0,0494 | **0,3522** | 6,8 |
| ElasticNet | 20/20 | $5.526,95 | 0,7848 | 0,0265 | 0,3690 | 0,3 |

### 5.3.1 La línea de base del README, y un hallazgo incómodo

El `README.md` original publica un modelo con métricas concretas, pero ese modelo **no
existía como artefacto**: el `.pkl` nunca se versionó.
`scripts/register_readme_baseline.py` lo reconstruye desde sus hiperparámetros
publicados, sobre el mismo split (verificado contra el commit inicial), y **contrasta
el resultado contra lo que el documento afirma**. Coincide con diferencias de centavos:
R² 0,8353 contra 0,8353; RMSE $4.834,74 contra $4.835. El README decía la verdad.

Queda registrado aparte, como `insurance-charges-baseline-ds` con alias `reference`.

Lo incómodo: **esa línea de base le gana a la familia `xgboost` de la comparación**
($4.834,74 vs $4.897,22). El original corría 350 iteraciones de búsqueda —por el bug
del typo— y el default de hoy son 50. Si XGBoost está sub-buscado, "gana Random Forest"
podría ser un artefacto del presupuesto, así que se verificó:

| Configuración | val_RMSE | val_R² |
|---|---:|---:|
| `xgboost`, 50 iteraciones (default) | $4.897,22 | 0,8310 |
| `xgboost`, 350 iteraciones (el original) | $4.834,74 | 0,8353 |
| **`random_forest`, 20 iteraciones (ganador)** | **$4.800,25** | **0,8377** |

Con 350 iteraciones XGBoost reproduce exactamente la línea de base y **sigue
perdiendo**. La conclusión se sostiene dándole al incumbente siete veces más
presupuesto que al ganador.

Tres lecturas que **sólo existen porque hay comparación**:

1. **XGBoost reproduce exactamente sus métricas documentadas** ($4.897,22 / 0,8310). Es la
   prueba de no-regresión de todo el refactor.
2. **El piso lineal queda $726,70 por detrás.** La complejidad no lineal ahora está
   justificada con una medición, no asumida.
3. **Los rankings por validación y por CV no coinciden.** Gana Random Forest por
   `val_rmse`, pero XGBoost tiene el mejor RMSE de validación cruzada.

### 5.4 El sesgo de selección, asumido y visible

Elegir entre cuatro familias por una métrica medida sobre el **mismo** conjunto de
validación es una comparación múltiple: el ganador se lleva algo de ventaja por azar. Lo
estadísticamente más prolijo sería elegir por CV y reportar validación como estimación
honesta.

**No se hizo**, y el motivo es de coherencia del sistema: `resolve_model()` ordena por
`metrics.val_rmse` y `promote_model.py` compara por `val_rmse`. Usar un criterio local
distinto rompería esa cadena por una ganancia marginal con cuatro candidatos.

La mitigación es un WARNING explícito cuando los dos rankings discrepan — y en la corrida
de referencia **ese caso se dio**:

```
WARNING - El ranking por validacion y el ranking por CV no coinciden: gana
'random_forest' por val_rmse pero 'xgboost' tiene mejor RMSE de CV
(0.3522 vs 0.3599). El margen no es solido; tomar la diferencia con cautela.
```

El bloque **"Justificación"** del reporte de entrenamiento pasó de ser un párrafo fijo a
derivarse de los números de la corrida. Era la única forma de que no fuera falso en cuanto
ganara otra familia.

### 5.5 Costo y control

`HYPERPARAM_ITERATIONS` dejó de ser el `n_iter` absoluto y pasó a ser un presupuesto **por
familia**: cada una corre `round(presupuesto × su ratio)` iteraciones, topeadas por la
cardinalidad de su grid. Con el catálogo por defecto, 50 se traduce en 50+20+20+20 = **110
iteraciones** (550 fits contra 250, ~2,2×). Las cuatro corren en **~15 s** sobre este
dataset.

`TRAIN_MODEL_FAMILIES=xgboost` reproduce exactamente el pipeline previo a la comparación.

De paso se corrigió una subscripción excesiva de cores que ya existía con una sola
familia: `RandomizedSearchCV(n_jobs=-1)` envolvía un `XGBRegressor(n_jobs=-1)`, así que los
procesos de la validación cruzada competían entre sí por los mismos cores.

---

## 6. Gobierno del modelo: registro y promoción

### 6.1 Registrar y promover son dos pasos

`training.py` registra una versión nueva y la deja en `staging`. **No la promueve.** La
promoción es una decisión aparte que toma `promote_model.py` comparando el candidato
contra la versión que hoy está sirviendo. Separar los dos pasos es lo que evita que un
entrenamiento cualquiera pise el modelo en producción.

```bash
python src/promote_model.py            # evalúa y promueve si pasa
python src/promote_model.py --dry-run  # sólo informa
```

**Gates aplicados:**

| Gate | Umbral | Tipo |
|---|---|---|
| `val_r2` | ≥ 0,75 | absoluto |
| `overfitting_r2_diff` | < 0,15 | absoluto |
| `val_rmse` | ≤ RMSE de producción × (1 − mejora mínima) | relativo |

La decisión queda escrita **en el registry**, como tags de la versión: `promoted_at`,
`promoted_by`, `promotion_reason` con el texto completo de los gates, y `model_family`. Una
versión rechazada conserva `promotion_rejected_at` y el motivo. El registry contesta *qué
modelo está sirviendo, desde cuándo, de qué familia y por qué* sin abrir ningún run.

Los gates son **agnósticos de la familia**, que es lo correcto: se promueve por métrica, no
por algoritmo. El log sí nombra la familia, para que quede escrito "se reemplaza xgboost
por random_forest".

### 6.2 Aliases en lugar de stages

El enunciado pide registrar el modelo "con etapa (`Staging`/`Production`)". MLflow deprecó
los stages en 2.9 y los eliminó de la API en 3.x; el reemplazo son los **aliases**. Se usan
aliases `staging` y `production` y **además** se escribe un tag `stage` con el nombre
clásico, para que la equivalencia con el enunciado quede explícita tanto en la UI como en
la base del registry.

### 6.3 Cadena de resolución del modelo

`scoring.py` no carga un `.pkl` del disco y espera que sea el correcto. Resuelve en un
orden explícito y **logueado**:

```
1. alias `production` del Model Registry   ← lo normal en producción
2. alias `staging`                          ← si todavía no se promovió nada
3. mejor run de training según val_rmse     ← si el registry está vacío
4. models/best_model.pkl                    ← compatibilidad, con WARNING
```

El reporte de scoring dice exactamente de dónde salió el modelo (`registry:production`,
`run:best`, `file:legacy`), y ese dato viaja hasta la tabla `batch_predictions`, fila por
fila.

---

## 7. Scoring batch sobre producción

```bash
python src/scoring.py
```

Descubre los lotes de `data/prod/` con una expresión regular tolerante a la doble
extensión `.csv.csv` del enunciado, así que agregar un `dataset_prod4_feats.csv` **no
requiere tocar código**. Un lote sin archivo `_target` se trata como batch sin etiquetas.

**Un lote que falla no aborta el resto**: se registra como `ALERT` con el motivo y el
pipeline sigue con los demás.

### 7.1 Salidas

| Salida | Ruta |
|---|---|
| Predicciones por lote | `results/predictions/predictions_<lote>_<ts>.csv` |
| Reporte de monitoreo | `results/monitoring_report_<ts>.{json,csv,txt}` |
| Dashboard | `results/monitoring_dashboard_<ts>.html` |
| Predicciones en DB | tabla `batch_predictions` (40.140 filas acumuladas) |
| Monitoreo en DB | tabla `batch_monitoring` |
| Historial de alertas en DB | tabla `alert_history` |
| Runs de MLflow | experimento `insurance-charges`, `pipeline_stage=scoring` |

Cada fila de predicción lleva su procedencia: `model_name`, `model_version`,
`model_source` y `mlflow_run_id`. Una predicción sin trazabilidad no es auditable.

`FAIL_ON_ALERT=true` hace que el pipeline termine con exit code ≠ 0 si algún lote queda en
`ALERT` — pensado como gate de CI. El default es `false`: un `ALERT` es una señal de
monitoreo, no un fallo de ejecución.

---

## 8. Monitoreo: señales, reglas y alertas

### 8.1 Las cinco señales

| Señal | ¿Necesita target? | Qué mide |
|---|:---:|---|
| Contrato de datos | no | Rangos, categorías, nulos, columnas faltantes |
| Drift de features (PSI) | no | Distribución de entrada vs. training |
| Drift de predicciones (PSI) | no | Distribución de `y_pred` vs. validación |
| Performance | **sí** | RMSE / MAE / R² / MAPE vs. validación |
| Desvío del target | **sí** | Media y PSI del target vs. training |

El estado de un lote es el **peor** de todas sus señales: un solo `ALERT` alcanza. Las tres
primeras no necesitan ground truth, y son las únicas disponibles para `prod3`.

El **drift se mide sobre las features crudas, no sobre las derivadas**: `bmi_squared`,
`bmi_smoker` y `age_smoker` son funciones determinísticas de `bmi`, así que reportarlas
como hallazgos independientes infla el ruido sin agregar información.

Los **rangos del contrato son deliberadamente más anchos que los del training**: no se
quiere alertar por un dato nuevo pero válido, sino por datos imposibles. Un BMI de 27.929
es imposible; uno de 55 es simplemente nuevo.

### 8.2 Reglas por feature y por lote

Los umbrales globales son el punto de partida. Aplicar la misma vara a `bmi` que a
`region`, y medir `prod3` —sin ground truth, con las features corruptas— con la misma que
`prod1`, hace que el semáforo sea a la vez ruidoso en unas señales y permisivo en otras.

`config/monitoring_rules.json` afina umbrales sin tocar código ni multiplicar variables de
entorno. Precedencia, del más específico al más general:

```
regla de (lote, feature)  >  regla de lote  >  regla de feature  >  default global
```

Que "lote" gane sobre "feature" es una decisión: una regla de lote es una afirmación
deliberada sobre un dataset concreto; una de feature es un refinamiento que vale para
todos. Cuando las dos aplican y hay que ser explícito, la forma inequívoca de resolverlo es
escribir la regla de `(lote, feature)`.

Las reglas que se entregan están justificadas en el propio archivo:

| Regla | Umbral | Por qué |
|---|---|---|
| `bmi` | PSI 0,05 / 0,15 (más estricto) | Es la feature de mayor peso y entra tres veces (`bmi`, `bmi_squared`, `bmi_smoker`); además es la que efectivamente se corrompe en producción |
| `sex`, `region` | PSI 0,20 / 0,40 (más laxo) | Categóricas balanceadas de 2 y 4 niveles: su PSI oscila por muestreo y a 0,10 sería ruido |
| lote `prod3` | PSI 0,05 / 0,15 | Llega **sin ground truth**: no hay performance que pueda desmentir un drift moderado, así que la vara sobre la entrada se endurece |
| `prod3` + `sex`/`region` | PSI 0,20 / 0,40 | La regla de lote sería demasiado estricta para una categórica balanceada |

**Borrar el archivo es seguro**: sin él, todo se mide con los umbrales globales. Un archivo
presente pero mal formado —en cualquier nivel de anidamiento— cae a los defaults con un
WARNING: un JSON roto no debe tumbar un pipeline de scoring que por lo demás puede correr.

**Auditabilidad.** Cada señal lleva el nombre de la regla que fijó **el umbral que la
decidió**, no un resumen de todas las que aplicaron: una regla de feature puede gobernar el
PSI mientras una de lote gobierna el umbral de esquema, y nombrar la equivocada sería peor
que no nombrar ninguna. El reporte de texto, el dashboard y la tabla `alert_history` leen
ese mismo dato.

### 8.3 El monitoreo tiene memoria

`batch_monitoring` guarda una fila por lote por corrida. Eso responde *cómo está el lote
hoy*, pero no las dos preguntas que hacen accionable a una alerta: **¿esto es nuevo?** y
**¿lo que estaba mal se arregló?**. Un reporte que dice exactamente lo mismo en cada corrida
es un informe, no un sistema de alertas: quien lo recibe deja de leerlo.

La tabla `alert_history` le da memoria. La identidad de una alerta es la tupla
`(batch_id, signal_name)` —por ejemplo `("prod3", "drift:bmi")`— y sobre esa clave hay tres
transiciones:

| Situación en esta corrida | Alerta abierta previa | Transición |
|---|---|---|
| Señal en WARNING/ALERT | no existe | **`NEW`** — se abre |
| Señal en WARNING/ALERT | ya existe | **`ONGOING`** — `occurrences += 1` |
| Señal en OK, o ausente | existe | **`RESOLVED`** — se cierra con `resolved_at` |
| Señal en OK | no existe | nada: no se registra ruido |

**La deduplicación es el punto.** `NEW` es el único evento notificable: una alerta que
lleva cinco corridas abierta aparece **una vez** como `ONGOING` con `occurrences=5`, no
como cinco alertas. Un cambio de severidad se anota en `severity_history` y **no** abre una
alerta nueva: sigue siendo el mismo problema.

Ese invariante está impuesto **por la base**, no por disciplina del código:

```sql
CREATE UNIQUE INDEX idx_alert_open
    ON alert_history (batch_id, signal_name) WHERE state = 'open';
```

Dos corridas consecutivas sobre los mismos datos:

```
# corrida 1                            # corrida 2
Alertas: 8 nuevas | 0 persisten        Alertas: 0 nuevas | 8 persisten
  [NUEVA]  drift:bmi   (1a vez)          [PERSISTE] drift:bmi (2 corridas, desde 23:52)
```

Si algo saliera `NUEVA` dos veces, la deduplicación estaría rota: ésa es la prueba.

Las transiciones se loguean en MLflow como métricas `alerts_new`, `alerts_ongoing` y
`alerts_resolved` (por lote y agregadas) más el tag `has_new_alerts`, así que la serie de
alertas es **graficable junto a `rmse` y `psi_bmi`** en el mismo experimento.

Si la base no responde, la reconciliación loguea un WARNING y devuelve vacío: el historial
es valioso, pero no vale abortar un scoring que ya terminó bien y cuyas predicciones ya
están escritas.

### 8.4 Consultar el historial

```bash
python src/alerts.py                      # alertas abiertas
python src/alerts.py --batch prod3        # de un lote
python src/alerts.py --history            # incluye las resueltas
```

---

## 9. Bugs encontrados en el código original

Once hallazgos. Los cuatro con impacto real:

| # | Ubicación | Problema | Impacto |
|---|---|---|---|
| 1 | `.gitignore` | La regla `*.txt` ignoraba `requirements.txt`, que **no existía en el repo** aunque el `Dockerfile` hacía `COPY requirements.txt .` | **El build de Docker fallaba.** Bloqueante |
| 3 | `training.py:153` | `os.getenv('HIPERPARAM_ITERATIONS', 350)` — typo: el entorno exportaba `HYPERPARAM_ITERATIONS` | La variable **nunca tenía efecto**: se entrenaba siempre con 350 × 5 = 1.750 fits en vez de 50. Reproducibilidad rota |
| 4 | `scoring.py:82` | `df['charges'].values` sin guarda | **Reventaba con `prod3`**, que no trae target |
| 6 | `training.py:224` | `adj_r2` usaba las features **antes** del `ColumnTransformer` | R² ajustado mal calculado: ignoraba las dummies del one-hot |

Los siete restantes —MAPE duplicado bajo otro nombre, trabajo muerto en el "Paso 5 beta",
un log que decía "Symlink" donde el código hacía `copy2`, negaciones faltantes en
`.gitignore`, y tres sobre el `docker-compose.yaml`— están en `DECISIONS.md § 3` con su
estado.

El bug del README (`cp .env.example .env`, archivo que nunca existió) **no se corrigió
ahí**: el README original se preserva intacto por decisión de proyecto. La instrucción
correcta está en `SOLUTION.md`.

---

## 10. Desviaciones deliberadas

### 10.1 Docker: eliminado, no adaptado

Los requisitos técnicos piden *"mantener compatibilidad con la ejecución actual del
proyecto (incluyendo Docker si ya está configurado)"*, y el repo original **sí** lo tenía.
Se eliminaron el `Dockerfile`, el `docker-compose.yaml`, el `entrypoint.sh` y el
`scripts/init-mlflow-db.sql`.

**El motivo.** La máquina de desarrollo no tiene Docker, así que el flujo containerizado
**nunca se ejecutó**. El pipeline cambió mucho —backend PostgreSQL, una segunda base, un
paso nuevo de promoción, volúmenes para artefactos— y la adaptación de esos archivos se
había hecho a ojo. Ya había aparecido una divergencia real: el `docker-compose.yaml` seguía
partiendo los runs en dos experimentos después de que la solución los unificara en uno, así
que todo lo que la documentación promete sobre comparar validación y producción en un mismo
gráfico **no se cumplía dentro del contenedor**.

Una infraestructura que no se corrió no es una garantía de reproducibilidad: es una
afirmación sin respaldo, y si falla en la máquina de quien evalúa, es peor que su ausencia.

**Qué se pierde y cómo se compensa.** Lo único que aportaba el contenedor era levantar
PostgreSQL y ejecutar cinco comandos en orden. Lo primero son dos `createdb`; lo segundo son
los cinco comandos de § 12, en el mismo orden en que los encadenaba el `entrypoint.sh`.
Toda la configuración se lee de variables de entorno, así que containerizarlo de nuevo es
directo para quien tenga con qué probarlo.

> **Nota.** El `README.md` original sigue presentando Docker como la forma principal de
> correr el proyecto. Se preserva intacto por decisión de proyecto (`DECISIONS.md § 6`),
> pero conviene saberlo: la guía vigente es `SOLUTION.md`.

### 10.2 Lo que no se hizo, a propósito

- **Sin canal de notificación** (webhook, Slack, mail) ni scheduler. El diseño los deja a
  un paso —un notificador es un consumidor de las transiciones `NEW`— pero agregarlos sin
  un destinatario real sería infraestructura sin uso, que es el mismo error que se evitó
  con Docker.
- **Sin CI/CD.** `FAIL_ON_ALERT=true` está pensado como gate de CI, pero no hay workflow
  que lo use.
- **Los umbrales no están calibrados sobre histórico.** Los cortes de PSI (0,10 / 0,25) son
  el estándar de facto en scoring de riesgo; los de performance son una elección
  conservadora. Son un punto de partida razonable, no un calibrado — y por eso todos son
  configurables.

---

## 11. Verificación contra los criterios de aceptación

| # | Criterio del enunciado | Estado | Evidencia |
|---|---|:---:|---|
| 1 | Entrenamiento y scoring corren de punta a punta sin errores | ✅ | Secuencia completa de § 12, verificada desde cero |
| 2 | Quedan registrados runs con parámetros, métricas y artefactos en MLflow | ✅ | Experimento `insurance-charges`; 4 tipos de run; 7 artefactos por run de training |
| 3 | Scoring usa el mejor artefacto entrenado, no un modelo aislado | ✅ | `resolve_model()` con cadena explícita y logueada; el origen viaja hasta `batch_predictions` |
| 4 | Se generan predicciones sobre los lotes de `data/prod/` | ✅ | 3 lotes × 1.338 filas; CSV por lote + tabla `batch_predictions` |
| 5 | Existe un reporte de monitoreo por batch con métricas y estado | ✅ | JSON + CSV + TXT + dashboard HTML, con semáforo y diagnóstico |
| 6 | La documentación permite reproducir el flujo completo | ✅ | `SOLUTION.md` § "Instalación y ejecución"; § 12 de este informe |

**Bonus del enunciado:**

| Bonus | Estado | Dónde |
|---|:---:|---|
| Model Registry con etapa (`Staging`/`Production`) | ✅ | Aliases + tag `stage` con el nombre clásico |
| Script de promoción basado en métricas | ✅ | `src/promote_model.py`, con `--dry-run` |
| Dashboards o visualizaciones de monitoreo | ✅ | `src/dashboard.py`: HTML autocontenido, sin CDN, con soporte de tema |
| Tests unitarios de tracking y monitoreo | ✅ | **172 tests**, sin necesidad de DB ni MLflow |

### 11.1 Tests

```bash
python -m pytest tests/ -q          # 172 tests
```

Cubren las funciones puras, que es donde un bug se paga caro y en silencio:

- **`test_data_loader.py`** — parseo de todas las convenciones decimales, demostración del
  fallo silencioso de `pd.read_csv`, descubrimiento de lotes, y validación del contrato
  **contra los archivos reales del repo**
- **`test_monitoring.py`** — PSI (identidad, monotonía, valores fuera de escala,
  categóricas), umbrales en los bordes exactos, y los tres escenarios del challenge con sus
  diagnósticos
- **`test_monitoring_rules.py`** — los cuatro niveles de precedencia, la tolerancia a
  archivos ausentes o mal formados en cualquier nivel, y que una regla cambie efectivamente
  el estado de una señal
- **`test_alerts.py`** — la máquina de estados completa, la reaparición después de
  resuelta, el escalamiento sin abrir alerta nueva, y que un fallo de la base no aborte un
  scoring que ya terminó bien
- **`test_model_zoo.py`** / **`test_training_selection.py`** — el contrato del pipeline de
  cada familia, el tope de `n_iter`, la regla de decisión del mejor modelo y su desempate
  determinístico
- **`test_utils.py`** — feature engineering y round-trip de `log1p`, que es lo que previene
  el training/serving skew

---

## 12. Cómo reproducirlo

**Requisitos:** Python 3.11 y PostgreSQL 15+. Se usan **dos bases**: `metlife_db` (negocio)
y `mlflow_db` (tracking).

```bash
# 1. Entorno
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 2. Bases de datos
psql -d postgres -c "CREATE ROLE metlife_user LOGIN PASSWORD 'metlife_pass';"
createdb -O metlife_user metlife_db
psql -d metlife_db -c "GRANT ALL ON SCHEMA public TO metlife_user;"
createdb -O metlife_user mlflow_db
psql -d mlflow_db -c "GRANT ALL ON SCHEMA public TO metlife_user;"

# 3. Configuración
cp .env.template .env        # ajustar DB_HOST y DB_PASSWORD

# 4. Pipeline completo
python -m pytest tests/ -q      # 172 tests, sin DB ni MLflow
python src/db_setup.py          # esquema + carga de data/dataset.csv
python src/training.py          # compara 4 familias, registra el ganador
python src/promote_model.py     # promueve si pasa los gates
python src/scoring.py           # puntúa data/prod/ y genera el monitoreo
```

**Correr `src/scoring.py` una segunda vez** es lo que muestra el sistema de alertas
funcionando: las mismas señales pasan de `NUEVA` a `PERSISTE`.

**Revisar los resultados:**

```bash
cat results/monitoring_report_*.txt          # semáforo y diagnóstico
open results/monitoring_dashboard_*.html     # dashboard
python src/alerts.py                         # alertas abiertas
./scripts/mlflow_ui.sh 5001                  # UI de MLflow
```

> En macOS el puerto 5000 lo ocupa el receptor de AirPlay, por eso el ejemplo usa 5001.

**Consultas útiles en la UI de MLflow:**

```
tags.pipeline_stage = 'training_candidate'                    # comparar las familias
tags.pipeline_stage = 'scoring' and metrics.psi_max > 0.25    # drift de entrada
tags.has_new_alerts = 'true'                                  # corridas con algo nuevo
params.eval_dataset = 'prod1'                                 # la historia de un lote
```

---

## 13. Estructura de la solución

```
src/
  config.py          Configuración centralizada + resolvedor de umbrales
  model_zoo.py       Catálogo de familias de modelos a comparar
  training.py        Comparación entre familias + tracking + registro
  promote_model.py   Promoción por métricas, con gates
  scoring.py         Scoring batch + monitoreo + alertas
  monitoring.py      PSI, señales, semáforo, diagnóstico
  alerts.py          Historial de alertas con estado (+ CLI)
  data_loader.py     Lectura robusta de data/prod + contrato de datos
  dashboard.py       Dashboard HTML autocontenido
  db_setup.py        Esquema de PostgreSQL
  mlflow_utils.py    Tracking, Model Registry, resolución del modelo
  utils.py           Feature engineering y transformación del target
config/
  monitoring_rules.json   Umbrales por feature y por lote
tests/                    172 tests (pytest, sin DB ni MLflow)
scripts/mlflow_ui.sh      Levanta la UI sin exponer la contraseña
```

**Tablas de PostgreSQL** (`metlife_db`): `training_dataset`, `predictions` (legacy),
`batch_predictions`, `batch_monitoring`, `alert_history`.

---

## Documentos relacionados

| Documento | Qué contiene |
|---|---|
| **`SOLUTION.md`** | Guía operativa: cómo instalar, ejecutar y leer cada salida |
| **`DECISIONS.md`** | Análisis forense completo, las 16 decisiones de arquitectura con su fundamento, los 11 bugs y los supuestos asumidos |
| **`README.md`** | Documento original del equipo de ciencia de datos, preservado sin modificar |
| **`challenge_ml.md`** | Enunciado del challenge |
