# Decisiones de diseño, supuestos y hallazgos

El enunciado del challenge cierra con:

> *Si detectás ambigüedades, asumí una decisión razonable y documentala explícitamente en la solución.*

Este documento es esa documentación. Cubre tres cosas: el análisis forense de
los archivos de `data/prod/`, las decisiones de arquitectura con su porqué, y
los bugs encontrados en el código original.

---

## 1. Análisis forense de `data/prod/`

Los tres lotes derivan del mismo origen que `data/dataset.csv`: 1.338 filas,
`age` incrementada en 1 y `bmi` con un jitter mínimo — simulan "el mismo padrón,
un año después". Cada uno trae un defecto distinto, y **detectarlos es el
verdadero objetivo del ejercicio**.

### 1.1 `prod1` — coma decimal (formato, no corrupción)

```
$ head -3 data/prod/dataset_prod1_target.csv.csv
charges
14700,80931
1540,261607
```

El archivo declara una sola columna (`charges`) pero cada fila trae una coma.
Con el separador por defecto de `pd.read_csv`, pandas ve **dos campos y un solo
nombre de columna**, así que usa el primero como índice y el segundo como valor:

```python
>>> pd.read_csv("data/prod/dataset_prod1_target.csv.csv").head(2)
       charges
14700    80931      # el valor real era 14700.80931
1540    261607
```

**No lanza ninguna excepción.** Devuelve un DataFrame válido y completamente
equivocado. Es exactamente el fallo que un pipeline productivo no puede
permitirse, porque se propaga hasta las métricas sin que nada avise.

**Decisión:** `data_loader.read_single_column_numeric()` lee el archivo como
texto plano y normaliza el separador explícitamente, en vez de delegarle a
pandas una decisión ambigua. Está cubierto por
`tests/test_data_loader.py::test_read_csv_directo_corrompe_el_target_en_silencio`,
que deja el fallo documentado como test de regresión.

Bien parseado, `prod1` tiene media **13.276,24** contra **13.270,42** del
dataset de entrenamiento. Es un lote sano → **OK**.

### 1.2 `prod2` — target multiplicado por 100

Las features de `prod2` son **byte a byte idénticas** a las de `prod1`:

```
$ md5 data/prod/dataset_prod{1,2}_feats.csv.csv
MD5 (dataset_prod1_feats.csv.csv) = 6612db1d55a2d7e540a82a291a2bfb60
MD5 (dataset_prod2_feats.csv.csv) = 6612db1d55a2d7e540a82a291a2bfb60
```

El target, en cambio, tiene el punto decimal corrido dos posiciones:
`14700,80931` → `1470080,931`. El ratio contra `prod1` es **exactamente 100,0 en
las 1.338 filas** (desvío estándar 7,9e-15, o sea ruido de punto flotante).

Esto genera el escenario más interesante del challenge: **features estables,
target desviado**. El modelo no se degradó; el archivo de etiquetas está mal.
Un monitoreo que sólo mire RMSE reportaría "el modelo se rompió, hay que
reentrenar", que es la conclusión opuesta a la correcta.

**Decisión:** el diagnóstico automático de `monitoring.diagnose()` cruza las
señales y distingue el caso:

> *Las features son estables y el desvío está SOLO en el target: apunta a un
> problema de CALIDAD DE DATOS en la etiqueta (unidad o escala), no a
> degradación del modelo. Corregir el proceso que genera el archivo de target
> antes de considerar un reentrenamiento.*

### 1.3 `prod3` — `bmi` sin punto decimal, y sin ground truth

```
$ head -3 data/prod/dataset_prod3_feats.csv.csv
age,sex,bmi,children,smoker,region
20,female,27929,0,yes,southwest     # el valor real era 27.929
19,male,34027,1,no,southeast        # el valor real era 34.027
```

Todas las columnas salvo `bmi` coinciden con `prod1`. Al `bmi` le eliminaron el
separador decimal, y **el factor resultante varía fila por fila** según cuántos
decimales tenía el original:

| factor | filas | ejemplo |
|--------|-------|---------|
| ×1000 | 1.214 | `27.929` → `27929` |
| ×100 | 105 | `32.68` → `3268` |
| ×10 | 18 | `45.3` → `453` |
| ×1 | 1 | `31.0` → `31` |

**Por qué esto importa para la decisión de no reparar:** como el factor depende
de la cantidad de decimales de cada valor original, "dividir por 1000" arruinaría
124 filas. Existe una regla que funcionaría — el BMI siempre tiene dos dígitos
enteros en este dominio (rango 15,96–53,13), así que `bmi / 10^(dígitos-2)`
reconstruye el valor — pero **aplicarla sería inferir sobre datos de producción
sin confirmación del proveedor del archivo**. Si la suposición "el BMI siempre
tiene dos dígitos" no valiera, el pipeline generaría predicciones con apariencia
normal sobre datos inventados, que es peor que no predecir.

Además `prod3` **no trae archivo de target**, así que no hay ninguna métrica de
performance que pueda delatar el problema. Las únicas señales disponibles son:

1. **Validación de contrato**: 1.337 de 1.338 filas con `bmi` fuera de `[10, 60]`
2. **PSI de features**: `bmi` = 11,76 (umbral de ALERT: 0,25)
3. **Drift de predicciones**: PSI = 0,60 contra la distribución de validación

Esa tercera señal es la razón por la que el baseline guarda la distribución de
`y_pred` sobre validación: es lo único que mide el comportamiento del modelo
cuando no hay etiquetas con qué compararlo.

### 1.4 Decisión transversal: detectar y alertar, **no reparar**

| Lote | Qué hace el pipeline |
|------|----------------------|
| prod1 | Parsea la coma decimal correctamente (es **formato declarado**, no corrupción) |
| prod2 | Puntúa, calcula métricas, marca `ALERT` y diagnostica que el defecto está en la etiqueta |
| prod3 | Puntúa, marca `ALERT` y avisa que las predicciones no son confiables |

La distinción es deliberada: **normalizar un formato es determinístico;
reconstruir un valor perdido es una suposición.** Lo primero se hace en
silencio, lo segundo se reporta y se deja en manos de quien es dueño del dato.

Ambos lotes defectuosos igual se puntúan, en vez de abortar, porque el reporte
de monitoreo es más útil con los números a la vista (un RMSE de $1.794.680 es
evidencia de la magnitud del problema) y porque las predicciones quedan
marcadas como no confiables en todas las salidas.

---

## 2. Decisiones de arquitectura

### 2.1 El baseline de monitoreo viaja como artefacto del run de training

**Problema:** para medir drift hacen falta las distribuciones de referencia. Si
se guardan en un archivo suelto, nada garantiza que correspondan al modelo que
scoring está usando — basta con que alguien reentrene y el baseline quede
desincronizado del modelo en producción.

**Decisión:** `training.py` sube `baseline_stats.json` como artefacto **del mismo
run** que produjo el modelo, y `scoring.py` lo baja desde el run asociado a la
versión que resolvió. El baseline y el modelo son inseparables por construcción.

El baseline contiene: bordes de bins y frecuencias por feature, estadísticos
descriptivos, distribución del target, **distribución de las predicciones sobre
validación** y las métricas de referencia.

### 2.2 Los bins del PSI guardan sólo los bordes interiores

Primera implementación: bordes con `-inf` y `+inf` en los extremos, para que los
valores fuera de rango cayeran en los bins extremos.

**Falló, y en silencio.** Al serializar el baseline a JSON, los infinitos se
convierten en `null`, y al releerlos vuelven como `NaN`; el histograma calculado
con bordes `NaN` daba conteos basura. Se detectó porque el PSI entre dos muestras
idénticamente distribuidas daba 1,6 en vez de ~0.

**Decisión:** guardar únicamente los **bordes interiores** y asignar los bins con
`np.digitize`, que trata los extremos como no acotados por definición. Es
JSON-safe y no depende de que los infinitos sobrevivan la serialización. Hay un
test que verifica explícitamente que los bordes releídos no sean `None`.

### 2.3 Cadena de resolución del modelo, explícita y logueada

El criterio de aceptación #3 pide que scoring use "efectivamente el mejor
artefacto entrenado y no un modelo aislado/manual".

```
1. models:/insurance-charges-xgb@production   <- lo normal
2. models:/insurance-charges-xgb@staging      <- si todavía no se promovió nada
3. mejor run del experimento por val_rmse     <- si el registry está vacío
4. models/best_model.pkl                      <- compatibilidad, con WARNING
```

El origen elegido se loguea y se propaga a **todas** las salidas: el reporte de
texto, el dashboard, las tablas `batch_predictions` y `batch_monitoring`, y los
params del run de scoring. Nunca hay que adivinar qué modelo produjo una
predicción. El paso 4 existe sólo por la exigencia de mantener compatibilidad
con la ejecución original, y avisa que ese modelo no tiene trazabilidad.

### 2.4 Registrar y promover son dos pasos distintos

`training.py` registra **siempre** una versión nueva y la deja en `staging`.
`promote_model.py` decide si pasa a `production`, comparando contra el modelo
vigente:

- **Gates absolutos**: `val_r2 >= 0.75` y `overfitting_r2_diff < 0.15`
- **Gate relativo**: `val_rmse <= rmse_producción * (1 - MIN_IMPROVEMENT)`

Si el entrenamiento promoviera solo, cualquier corrida experimental pasaría a
servir en producción. La decisión queda registrada en los tags de la versión
(`promoted_at`, `promoted_by`, `promotion_reason`) y explicada en el log.

### 2.4.1 Evidencia real de los gates: un candidato rechazado

Los gates no son decorativos. Durante el desarrollo, un entrenamiento produjo un
candidato **peor** que el modelo en producción y `promote_model.py` lo rechazó. El
texto quedó escrito en el tag `promotion_reason` de esa versión:

```
OK: val_r2=0.8250 >= 0.75
OK: overfitting=0.0769 < 0.15
RECHAZO: val_rmse=4,983.39 > umbral 4,897.22 (produccion=4,897.22);
         el candidato no mejora al modelo actual
```

La versión quedó en `staging` con `promotion_rejected_at`, y producción siguió
sirviendo el modelo anterior. Es el comportamiento que justifica separar *registrar*
de *promover* (§ 2.4): sin esa separación, ese entrenamiento habría pisado un modelo
mejor.

> El Registry se limpió después de la etapa de desarrollo y esa versión ya no está;
> el texto se conserva acá porque es la evidencia de que el gate relativo funciona.

### 2.5 Aliases en lugar de stages

El enunciado pide etapas `Staging`/`Production`. **MLflow deprecó los stages en
2.9 y los eliminó de la API en 3.x** (este proyecto corre sobre MLflow 3.16.1);
el reemplazo son los *aliases*, punteros con nombre a una versión concreta.

**Decisión:** usar aliases `staging` y `production` y además escribir un tag
`stage` con el nombre clásico (`Staging` / `Production`), para que la
equivalencia con el enunciado quede explícita en la UI y en la base del registry.

Esto obliga a un backend con base de datos — **el Model Registry no funciona con
file store** — y por eso el tracking corre sobre PostgreSQL (ver § 2.11).

### 2.11 PostgreSQL como backend de tracking, en una base separada

**Decisión:** los metadatos de los runs viven en PostgreSQL (`mlflow_db`) y los
artefactos en el filesystem (`./mlruns`).

Dos razones para no usar un file store: el **Model Registry no funciona sin una
base de datos**, y un file store no soporta escrituras concurrentes, así que dos
corridas simultáneas del pipeline pueden corromperlo.

**Por qué una base separada y no `metlife_db`:** MLflow crea **59 tablas** propias
(`runs`, `metrics`, `params`, `registered_models`, `traces`, `scorers`,
`webhooks`, …). La aplicación tiene 4 (`training_dataset`, `predictions`,
`batch_predictions`, `batch_monitoring`). Mezclarlas haría que el esquema de
negocio quedara enterrado, y complicaría dar permisos distintos o hacer backups
por separado. Es el mismo servidor, otra base.

**Consecuencia de seguridad que hubo que resolver:** a diferencia de un URI
`sqlite://`, el URI de Postgres **contiene la contraseña**. El código lo logueaba
en claro en tres lugares (`config.describe()`, `mlflow_utils.setup_tracking()` y
el mensaje final de training y scoring). Ahora:

- `config.mask_uri()` produce `MLFLOW_TRACKING_URI_SAFE`, y es esa la versión que
  se loguea;
- el URI se **deriva** de las credenciales `DB_*` en vez de duplicarse en otra
  variable de entorno;
- el mensaje final apunta a `./scripts/mlflow_ui.sh`, que resuelve el URI desde
  `config.py` y se lo pasa a MLflow sin que aparezca en pantalla ni en el
  historial del shell.

`mlflow_utils._check_backend()` verifica la conexión al arrancar y, si la base no
existe, devuelve el comando exacto para crearla en vez del error crudo de
psycopg2.

**Migración:** MLflow no ofrece migración entre backends. Los runs que hubiera en
un `mlflow.db` de SQLite previo no se trasladan; el backend nuevo arranca vacío y
se repuebla volviendo a correr el pipeline.

### 2.12 Métricas canónicas compartidas entre training y scoring

**El enunciado no pide MLflow en scoring.** La sección 2 pide *"registrar
resultados de scoring en un output reproducible (tabla en DB, CSV, o artefacto
versionado)"*, y eso ya lo cubren `batch_predictions`, `batch_monitoring` y los
CSV/JSON. Loguear además los runs de scoring a MLflow es un **agregado
deliberado**: deja el historial de monitoreo consultable y comparable en el
tiempo (`tags.monitoring_status = 'ALERT'` filtra los lotes con problema entre
todas las corridas), que es la base para graficar la degradación como serie.

**El problema que tenía ese agregado.** En la primera versión, training y
scoring logueaban conjuntos de métricas **sin una sola clave en común**: 24 y 18
respectivamente, cero solapamiento. Eso hacía imposible lo único que vuelve útil
el agregado — poner en un mismo gráfico el RMSE de validación y el de cada lote.

**Decisión:** una única función, `monitoring.canonical_metrics()`, que usan los
dos pipelines, exactamente igual que ya compartían `feature_engineering()`. Cada
run loguea las mismas 19 claves referidas a *su* dataset de evaluación, más un
param `eval_dataset` (`validation`, `prod1`, `prod2`, …) que da el contexto:

| Familia | Claves |
|---|---|
| Escala dólar | `rmse`, `mae`, `r2`, `adj_r2`, `mape` |
| Escala log | `rmse_log`, `mae_log`, `r2_log`, `mape_log` |
| Drift | `psi_age`, `psi_bmi`, `psi_children`, `psi_sex`, `psi_smoker`, `psi_region`, `psi_max` |
| Datos y salida | `n_violations`, `pred_mean`, `pred_std` |

Training conserva además las versiones prefijadas `train_*` / `val_*`, porque un
run de training evalúa **dos** datasets y necesita distinguirlos; las claves sin
prefijo refieren siempre al dataset de evaluación del run.

**Las métricas log se derivan de los valores en dólares** con `log1p`, y dan
exactamente los mismos números que evaluarlas en el espacio en el que entrena el
modelo, porque `log1p(expm1(x)) == x`. Eso es lo que permite una sola función:
training tiene las predicciones en escala log y scoring sólo en dólares. Hay un
test que fija esa equivalencia.

**Efecto secundario valioso.** Para poder loguear `psi_*` y `n_violations`,
training ahora calcula el PSI entre train y validación, y valida los datos de
entrenamiento contra el mismo contrato que se le exige a producción. Eso da dos
cosas que antes no existían:

- **El punto de referencia del PSI.** En la corrida actual, `psi_max` entre train
  y validación es **0,0587**, mayor que el de `prod1` contra training (**0,0406**).
  O sea: la variación entre las dos mitades del propio dataset de entrenamiento
  es mayor que la del primer lote de producción. Eso confirma que `prod1` es
  genuinamente estable y calibra el umbral: si un lote alerta con 0,06, estaría
  alertando por debajo del ruido del split.
- **Validación del dato de entrenamiento.** Un modelo entrenado sobre datos que
  violan el contrato es un problema que antes no se detectaba en ningún lado.

### 2.13 Un único experimento de MLflow

**Decisión revisada.** La primera versión usaba dos experimentos,
`insurance-charges-training` e `insurance-charges-scoring`. El argumento era que
el experimento es la unidad de comparación de la UI y que sus métricas no se
solapaban — pero esa falta de solapamiento era un defecto del diseño, no una
propiedad del problema, y se corrigió en § 2.12. Una vez que ambos lados loguean
las mismas 19 claves, mantenerlos separados impide justamente lo que vuelve útil
loguear el scoring: **ver validación y lotes de producción en el mismo gráfico.**

Hoy todo va a **`insurance-charges`**, y los runs se distinguen por el tag
`pipeline_stage`:

| `pipeline_stage` | Qué es | `scope` |
|---|---|---|
| `training` | Run principal: el modelo **ganador** de la comparación entre familias | — |
| `training_candidate` | Una familia evaluada en esa comparación (ver § 2.14) | — |
| `training_trial` | Cada combinación de la búsqueda de hiperparámetros | — |
| `scoring` | Run de scoring | `all_batches` (padre) o `batch` |

Eso habilita consultas directas en la UI, que antes requerían mirar dos
experimentos:

```
tags.pipeline_stage = 'scoring' and metrics.psi_max > 0.25
tags.monitoring_status = 'ALERT'
```

**Lo que hubo que blindar.** `resolve_model()` busca el mejor run por `val_rmse`
cuando el Model Registry está vacío. Con todo en un experimento, esa búsqueda
podía devolver un run de scoring o un trial de la búsqueda de hiperparámetros,
ninguno de los cuales tiene un modelo asociado. Ahora el filtro incluye
`tags.pipeline_stage = 'training'`, y por eso los trials anidados llevan
`training_trial` y no `training`: nunca deben competir con el run principal.

**Sigue siendo configurable.** `MLFLOW_EXPERIMENT` define el experimento único;
setear `MLFLOW_EXPERIMENT_TRAINING` y `MLFLOW_EXPERIMENT_SCORING` por separado
vuelve a dividirlos sin tocar código.

### 2.14 El mejor modelo se elige entre familias, no sólo entre hiperparámetros

**El problema.** El proyecto entrenaba una sola familia (XGBoost) y justificaba la
elección con un párrafo escrito a mano en el `training_report`. Los 10 runs
anidados que quedaban en MLflow eran variantes del mismo modelo: eso es una
comparación de **hiperparámetros**, no de modelos. "El mejor modelo" nunca se
eligió entre alternativas reales.

**La decisión.** Cada ejecución entrena cuatro familias sobre el mismo split, y el
criterio de selección que ya existía (`MODEL_SELECTION_METRIC`) pasa a aplicarse
también entre ellas. El catálogo vive en `src/model_zoo.py` como una tabla de
datos, aparte de `training.py`, para poder revisarlo de un vistazo y testearlo sin
MLflow ni PostgreSQL.

**Por qué esas cuatro.** Cada una responde una pregunta distinta, y eso está
declarado en el campo `rationale` de su spec:

| Familia | Pregunta que responde |
|---|---|
| `xgboost` | Es el incumbente: la referencia a batir, con su grid original intacto |
| `random_forest` | ¿El problema tiene la señal secuencial que el boosting supone? |
| `hist_gradient_boosting` | ¿La ventaja es del boosting o de la implementación de XGBoost? |
| `elasticnet` | ¿La complejidad no lineal aporta algo, o sólo lo suponíamos? |

**`HistGradientBoostingRegressor` en lugar de LightGBM.** Cumple el mismo rol
—otra implementación de boosting con binning de histogramas— y ya viene en
`scikit-learn`. Agregar `lightgbm` o `catboost` sumaría dependencias sin aportar
una familia conceptualmente distinta de las que ya están. El costo: no expone
`feature_importances_`, así que si llegara a ganar, esa corrida no genera gráfico
de importancia. Es una degradación explícita y documentada, no un fallo.

**El escalado no es un detalle.** `create_preprocessor()` hacía `passthrough` de
las numéricas. Entre ellas conviven `age_squared` (hasta ~10.000) y `children`
(hasta 5): con esas escalas, la penalización L1/L2 de `ElasticNet` castiga
desparejo y la evaluación de la familia lineal no sería honesta. Por eso sólo esa
familia declara `needs_scaling=True`. Los modelos de árboles son invariantes a la
escala y conservan el `passthrough` original **bit a bit**, que es lo que permite
comparar la corrida de XGBoost contra sus métricas ya documentadas.

**Jerarquía de runs: los candidatos no loguean modelo.** El run padre `training` es
el del ganador y es el único con artefacto de modelo. `resolve_model()` busca el
mejor run con `pipeline_stage='training'`; con este esquema hay **exactamente uno
por ejecución**, así que esa consulta no cambió una línea y sigue siendo correcta.
Que los candidatos no tengan modelo hace *imposible* resolver a uno de ellos por
accidente, aun si alguien aflojara ese filtro. Lo que sí queda de cada candidato es
su `best_params_` y su `cv_results_` completo: son reproducibles desde la semilla.

**Los candidatos tampoco escriben las métricas sin prefijo.** `rmse`, `r2`, `mape`
y `psi_*` son la serie que cruza etapas (§ 2.12). Si cuatro candidatos las
escribieran, el gráfico de validación → prod1 → prod2 dejaría de significar lo que
§ 2.12 y § 2.13 dicen que significa. Los candidatos usan `val_*` y `train_*`; el
run padre agrega una métrica `family_<nombre>_val_rmse` por familia, para que su
fila en la tabla de la UI muestre la comparación completa sin abrir los hijos.

**El sesgo de selección, asumido y visible.** Elegir entre cuatro familias por una
métrica medida sobre el **mismo** conjunto de validación es una comparación
múltiple: el ganador se lleva algo de ventaja por azar. Lo estadísticamente más
prolijo sería elegir por CV y reportar validación como estimación honesta. **No se
hizo**, porque `resolve_model()` ordena por `metrics.val_rmse` y `promote_model.py`
compara por `val_rmse`: usar un criterio local distinto rompería la coherencia del
sistema por una ganancia marginal con cuatro candidatos. La mitigación es un
WARNING explícito cuando el ranking por CV discrepa del ranking por validación.

En la corrida de referencia **ese caso se dio**: gana `random_forest` por
`val_rmse` ($4.800,25 vs $4.897,22) pero `xgboost` tiene mejor RMSE de CV (0,3522
vs 0,3599). El margen no es sólido, y el pipeline lo dice en vez de esconderlo.

**Un entrenamiento idéntico no registra versión nueva.** El Registry es un log de
*qué se entrenó*, y `training.py` registra en cada corrida. Pero reentrenar con la
misma semilla y los mismos datos —que es exactamente lo que uno hace verificando—
produce el mismo modelo, y registrarlo otra vez hace que cada versión deje de
significar algo: durante el desarrollo el Registry acumuló tres versiones con las 45
métricas idénticas. `register_model_version` compara ahora la familia y las métricas
de selección contra la última versión y, si coinciden, devuelve la existente con un
WARNING explícito en vez de crear una nueva.

La limitación está asumida y documentada en el código: si cambia el código pero las
métricas no se mueven, el cambio no se detecta. El run igual queda trazado entero en
MLflow con sus artefactos —lo único que se saltea es la entrada en el Registry— y el
comportamiento se apaga con `REGISTER_SKIP_DUPLICATES=false`.

**Costo de cómputo.** `HYPERPARAM_ITERATIONS` dejó de ser el `n_iter` absoluto y
pasó a ser un presupuesto **por familia** —no un total a repartir: con los ratios
del catálogo (1,0 + 0,4×3), 50 se traduce en 110 iteraciones en total—;
cada familia declara qué fracción consume,
topeada por la cardinalidad de su grid (sin ese tope, `elasticnet` —20
combinaciones— recibiría 50 iteraciones: `ParameterSampler` las recorta igual, pero
deja un número que no es el real en `search.n_iter`, y ese número termina en el
reporte y en MLflow). De 250 fits se pasó a 550, ~2,2×, y la corrida completa tarda
~15 s. `TRAIN_MODEL_FAMILIES=xgboost` reproduce exactamente el pipeline anterior.

De paso se corrigió una subscripción excesiva de cores que ya existía con una sola
familia: `RandomizedSearchCV(n_jobs=-1)` envolvía un `XGBRegressor(n_jobs=-1)`, así
que los procesos de la validación cruzada competían entre sí por los mismos cores.
Los estimadores del zoo se construyen con `n_jobs=1` y el paralelismo queda sólo en
el nivel de CV.

**El nombre del modelo registrado.** `MLFLOW_MODEL_NAME` pasó de
`insurance-charges-xgb` a `insurance-charges-regressor`. El nombre de un registered
model designa la **tarea**, no el algoritmo; con comparación entre familias, el
nombre anterior pasa de inexacto a engañoso en cuanto gana otra. La familia de cada
versión queda en su tag `model_family`. Las versiones ya registradas quedan bajo el
nombre viejo: no estorban, porque `resolve_model()` cae al paso 3 de su cadena
(mejor run de training) hasta que el nombre nuevo tenga su primer alias, y
`MLFLOW_MODEL_NAME=insurance-charges-xgb` restituye la continuidad exacta.

### 2.14.1 El modelo del README, registrado como línea de base

**El problema.** El `README.md` del equipo de ciencia de datos publica un modelo con
métricas concretas (R² = 0,8353 / RMSE = $4.835) y sus hiperparámetros ganadores. Pero
ese modelo **no estaba en ningún lado**: el `.pkl` original nunca se versionó y el repo
no traía artefactos. Era una afirmación en un documento, sin nada que la respaldara ni
contra qué comparar.

**La decisión.** `scripts/register_readme_baseline.py` lo reconstruye y lo registra
**una sola vez**, para que la comparación *"lo que había"* vs *"lo que eligió el
pipeline"* se pueda hacer dentro de MLflow y no leyendo dos documentos en paralelo.

**Es una reproducción, no el artefacto original**, y el script lo dice en su docstring.
Lo que sí se conserva del original, verificado contra el commit inicial
(`git show decb3a1:src/training.py`):

- los hiperparámetros exactos que publica el README
- el mismo split: `test_size=0.2, random_state=43` — idéntico al `SPLIT_SEED` actual
- las mismas semillas del estimador y de la búsqueda (42)
- el mismo feature engineering y la misma transformación `log1p`

Como el split y las semillas coinciden, la reproducción es determinística. **El script
imprime la comparación contra lo que el README declara**, y si no coincidiera lo diría
en vez de disimularlo. Coincide:

| Métrica (validación) | README | Reproducido | Delta |
|---|---:|---:|---:|
| R² | 0,8353 | 0,8353 | +0,0000 |
| RMSE | $4.835,00 | $4.834,74 | −0,26 |
| MAE | $2.102,00 | $2.101,55 | −0,45 |
| MAPE | 17,7000 | 17,7036 | +0,0036 |

O sea que **el README decía la verdad**, con diferencias de centavos atribuibles al
redondeo del propio documento.

**El hallazgo incómodo, y su verificación.** La línea de base del README
($4.834,74) le **gana** a la familia `xgboost` de nuestra comparación ($4.897,22). No
es misterioso: el código original corría **350 iteraciones** de búsqueda —por el bug
del typo (§ 3, hallazgo 3), que dejaba el default de 350 en vez de las 50 que se creía
configurar— y nosotros corremos 50. Con menos búsqueda, XGBoost encuentra un óptimo
peor.

Eso pone en duda la conclusión de § 2.14: si XGBoost está sub-buscado, "gana Random
Forest" podría ser un artefacto del presupuesto. **Se verificó**, corriendo la familia
`xgboost` con las 350 iteraciones del original:

| Configuración | val_RMSE | val_R² |
|---|---:|---:|
| `xgboost`, 50 iteraciones (default) | $4.897,22 | 0,8310 |
| `xgboost`, 350 iteraciones (el original) | $4.834,74 | 0,8353 |
| **`random_forest`, 20 iteraciones (ganador)** | **$4.800,25** | **0,8377** |

Con 350 iteraciones, XGBoost reproduce exactamente la línea de base del README — lo
que confirma de paso que esa línea de base es fiel— y **sigue perdiendo**. La
conclusión se sostiene incluso dándole al incumbente siete veces más presupuesto que
al ganador.

Dos cosas quedan dichas y no barridas: con el presupuesto por defecto, la familia
`xgboost` del reporte rinde peor que el modelo con el que arrancó el proyecto; y no se
hizo un barrido de presupuesto para las cuatro familias, sólo para la que tenía motivos
para sospecharse perjudicada.

**Por qué en un registro aparte.** Queda como `insurance-charges-baseline-ds` con alias
`reference`, no como una versión del modelo que sirve en producción. Es una referencia
histórica, no un candidato: si viviera en la misma cadena de versiones daría a entender
que compite por el alias `production`, y no compite. Por la misma razón su run lleva
`pipeline_stage=baseline` y no `training`, así que `resolve_model()` no puede
resolverlo por accidente.

### 2.15 Umbrales de monitoreo por feature y por lote

**El problema.** `PSI_WARN` / `PSI_ALERT` se aplicaban igual a `bmi` —la feature de
mayor peso, que además entra tres veces vía `bmi_squared` y `bmi_smoker`— que a
`region`, una categórica de cuatro niveles casi uniformes cuyo PSI oscila por
muestreo. Y `prod3`, que llega sin ground truth y con las features corruptas, se
medía con la misma vara que `prod1`, que está sano. El resultado es un semáforo a
la vez ruidoso en unas señales y permisivo en otras.

**La decisión.** Un archivo de reglas, `config/monitoring_rules.json`, con cuatro
niveles de precedencia:

```
regla de (lote, feature)  >  regla de lote  >  regla de feature  >  default global
```

**Por qué "lote" le gana a "feature"**, que es el único choque no obvio: una regla
de lote es una afirmación deliberada sobre un dataset concreto, tomada por alguien
que conoce ese dataset; una de feature es un refinamiento que vale para todos.
Cuando las dos aplican y hay que ser explícito, la forma inequívoca de resolverlo
es escribir la regla de `(lote, feature)` — y el archivo que se entrega usa
exactamente ese mecanismo para `prod3` + `sex` / `region`.

**Por qué un archivo y no más variables de entorno.** La alternativa era
`PSI_WARN_BMI`, `PSI_ALERT_BMI`, `PSI_WARN_REGION`… una variable por feature y por
umbral. Con 6 features y 5 pares de umbrales eso son 60 variables, y ninguna
combinación por lote sería expresable.

**Compatibilidad.** Sin archivo, `resolve_thresholds()` devuelve el default global y
el comportamiento es idéntico al de antes de que existieran las reglas — es el caso
normal, no una excepción. Un archivo presente pero mal formado devuelve los
defaults con un WARNING: un JSON roto no debe tumbar un pipeline de scoring que por
lo demás puede correr perfectamente.

**Auditabilidad, y por qué es por grupo de umbral.** Cada `Signal` lleva un campo
`thresholds_source` con la regla que fijó **el umbral que decidió esa señal**, no un
resumen de todas las que aplicaron. La distinción importa: una regla de feature puede
fijar los PSI y una de lote el umbral de esquema, y entonces la señal de drift y la de
contrato están gobernadas por reglas distintas aunque las dos aplicaron al mismo lote.
Por eso `Thresholds` guarda la procedencia por grupo (`psi`, `perf`, `r2`,
`target_shift`, `schema`) y cada evaluador estampa la del suyo. El reporte imprime esa
regla junto a la señal y en el pie. Sin eso, un `WARNING`
emitido con un umbral custom sería indistinguible de uno emitido con el global, y
el reporte dejaría de ser verificable. Por la misma razón, `evaluate_schema()`
**recalcula** la severidad con los umbrales del lote en vez de confiar en la que
trae `data_loader`: ese módulo resuelve la suya con los globales porque no sabe a
qué lote pertenece el archivo que está leyendo.

### 2.16 El monitoreo tiene memoria: alertas con estado

**El problema.** `batch_monitoring` guarda una fila por lote por corrida. Eso
responde *cómo está el lote hoy*, pero no las dos preguntas que hacen accionable a
una alerta: **¿esto es nuevo?** y **¿lo que estaba mal se arregló?**. Un reporte que
dice exactamente lo mismo en cada corrida es un informe, no un sistema de alertas:
quien lo recibe deja de leerlo.

**La decisión.** Una tabla `alert_history` donde la identidad de una alerta es la
tupla `(batch_id, signal_name)`, con tres transiciones: `NEW`, `ONGOING`,
`RESOLVED`. Una señal en OK que no tenía alerta abierta **no registra nada**: el
historial guarda problemas, no el estado completo de cada corrida.

**La deduplicación es el punto.** `NEW` es el único evento notificable. Una alerta
que lleva cinco corridas abierta aparece una vez como `ONGOING` con
`occurrences=5`, no como cinco alertas. Un cambio de severidad (`WARNING` →
`ALERT`) se anota en `severity_history` y **no** abre una alerta nueva: sigue siendo
el mismo problema, y tratarlo como nuevo volvería a notificar algo que ya se sabía.

**El invariante lo impone la base, no el código:**

```sql
CREATE UNIQUE INDEX idx_alert_open
    ON alert_history (batch_id, signal_name) WHERE state = 'open';
```

Un índice único parcial sobre `state='open'`. La deduplicación no queda dependiendo
de que `reconcile()` se acuerde de respetarla.

**Una alerta resuelta que reaparece vuelve a ser `NEW`**, y es correcto: las
resueltas no figuran entre las abiertas, y un problema que vuelve después de
haberse arreglado sí amerita notificarse otra vez.

**Escalamiento vs. baja de severidad.** Un `ALERT` que baja a `WARNING` no es un
escalamiento, y rotular ambos casos igual confundiría a quien lee el reporte. La
dirección del cambio se calcula con `config.STATUS_ORDER` y se imprime como
`ESCALADA` o `BAJO`.

**Un fallo de la base no aborta el scoring.** `reconcile()` loguea un WARNING y
devuelve vacío. Es el mismo criterio que `mlflow_utils.log_artifact_safe`: el historial
de alertas es valioso, pero las predicciones ya están escritas y el reporte ya se
puede generar; perder un scoring completo por no poder actualizar el historial sería
desproporcionado.

**Lo que NO se hizo, a propósito:** no hay canal de notificación (webhook, Slack,
mail) ni scheduler. El diseño los deja a un paso — un notificador es un consumidor
de las transiciones `NEW`, que es exactamente el evento notificable — pero
agregarlos sin un destinatario real sería infraestructura sin uso, que es el mismo
error que se evitó con Docker (§ 5).

### 2.6 El drift se mide sobre las features crudas, no sobre las derivadas

`bmi_squared`, `bmi_smoker`, `age_smoker`, etc. son funciones determinísticas de
las features crudas. Si driftea `bmi`, driftean las tres por construcción.
Reportarlas como hallazgos independientes infla el ruido sin agregar
información, así que el PSI se calcula sobre `age`, `bmi`, `children`, `sex`,
`smoker` y `region`.

### 2.7 Rangos del contrato de datos más anchos que los del training

`FEATURE_SPEC` usa `bmi ∈ [10, 60]` y `age ∈ [18, 100]`, más anchos que el rango
observado en entrenamiento (`bmi` 15,96–53,13). La intención es distinguir **dato
nuevo pero válido** de **dato imposible**: un BMI de 55 es simplemente un caso no
visto, y para eso está el PSI; un BMI de 27.929 es un error de datos, y para eso
está el contrato. Si el contrato fuera el rango exacto del training, alertaría
por cualquier valor nuevo y el ruido lo volvería inútil.

### 2.8 Un lote que falla no aborta el resto

Si el scoring de un lote lanza una excepción, se registra como `ALERT` con el
error en el diagnóstico y el pipeline continúa con los demás. Es preferible un
reporte con dos lotes buenos y uno fallado que ningún reporte.

### 2.9 `FAIL_ON_ALERT=false` por defecto

Un `ALERT` es una señal de monitoreo, no un fallo de ejecución: el pipeline
**hizo su trabajo** al detectar el problema. Con el default, termina en 0 y deja
el reporte. Poniendo `FAIL_ON_ALERT=true` se obtiene el comportamiento de gate
para CI/CD, donde un lote en ALERT sí debería frenar el flujo.

### 2.10 `feature_engineering` compartido entre training y scoring

Ambos pipelines importan la misma función de `utils.py`. Si cada uno derivara sus
features por separado, cualquier cambio en uno solo introduciría training/serving
skew silencioso. `tests/test_utils.py` verifica que sea determinística y que dé
el mismo resultado con `is_training=True` y `False`.

---

## 3. Bugs encontrados en el código original

| # | Ubicación | Problema | Impacto | Estado |
|---|-----------|----------|---------|--------|
| 1 | `.gitignore` | La regla `*.txt` ignoraba `requirements.txt`, que directamente **no existía en el repo**, aunque el `Dockerfile` original hacía `COPY requirements.txt .` | **El build de Docker fallaba**, y sin el archivo tampoco había forma de reproducir el entorno local. Bloqueante | Corregido: archivo creado + negaciones en `.gitignore` |
| 2 | `.gitignore` | `*.json`, `*.txt`, `*.pkl` globales sin excepciones | Ignoraba reportes, metadata y documentación | Corregido con negaciones explícitas |
| 3 | `training.py:153` | `os.getenv('HIPERPARAM_ITERATIONS', 350)` — typo; el entorno del proyecto exportaba `HYPERPARAM_ITERATIONS` | La variable **nunca tenía efecto**: se entrenaba siempre con 350 iteraciones × 5 folds = 1.750 fits en vez de 50. Reproducibilidad rota | Corregido en `config.py`, aceptando el nombre viejo como respaldo |
| 4 | `scoring.py:82` | `actual_charges = df['charges'].values` sin guarda | **Reventaba con `prod3`**, que no tiene target | Corregido: el target es opcional en todo el flujo |
| 5 | `training.py:213` | `mape_log` se calculaba con los valores en escala original | Duplicaba el MAPE en dólares bajo otro nombre | Corregido |
| 6 | `training.py:224-225` | `adj_r2` usaba `p = X_train.shape[1]`, las features **antes** del `ColumnTransformer` | R² ajustado mal calculado: ignoraba las dummies del one-hot | Corregido usando `get_feature_names_out()` |
| 7 | `training.py:419-421` | "Paso 5 beta" llamaba a `create_preprocessor()` y `define_hyperparameter_grid()` y descartaba el resultado | Trabajo muerto | Eliminado |
| 8 | `README.md:139` | Indica `cp .env.example .env`, pero el archivo se llama `.env.template` (`.env.example` nunca existió en el repo) | Instrucción de setup rota | **No corregido ahí**: el README original no se modifica (ver § 6). La instrucción correcta está en `SOLUTION.md` |
| 9 | `docker-compose.yaml` | Sin volúmenes para MLflow ni para `./data` | Los experimentos se perderían al bajar el contenedor | **Sin efecto**: el archivo se eliminó (ver § 5). Queda como hallazgo sobre el repo original |
| 10 | `training.py:312` | El log decía "Symlink actualizado" pero el código usa `shutil.copy2` | Mensaje engañoso | Corregido |
| 11 | `docker-compose.yaml` | `restart: on-failure` en un job por lotes | Un pipeline que falla se reintentaría en loop | **Sin efecto**: el archivo se eliminó (ver § 5). Queda como hallazgo sobre el repo original |

---

## 4. Supuestos asumidos

1. **`data/prod/` es inmutable.** Son los archivos que entrega el proveedor; el
   pipeline los lee y los evalúa, no los corrige ni los reescribe.
2. **La doble extensión `.csv.csv` es un accidente del enunciado**, no una
   convención. El descubrimiento de lotes tolera ambas formas (`.csv` y
   `.csv.csv`), así que agregar un `dataset_prod4_feats.csv` no requiere tocar
   código.
3. **Un lote sin archivo `_target` es un batch sin etiquetas**, tal como indica el
   enunciado para `prod3`, y no un error de configuración.
4. **Los umbrales del semáforo son un punto de partida razonable**, no un
   calibrado sobre datos históricos. Los cortes de PSI (0,10 / 0,25) son el
   estándar de facto en scoring de riesgo; los de performance (1,25× / 1,50×
   sobre el RMSE de validación) son una elección conservadora. Todos son
   configurables por variable de entorno, que es lo que permite ajustarlos
   cuando haya historia real.
5. **El pipeline de XGBoost no se toca; se le agregan competidores.** El challenge
   es de MLOps, así que se conservan intactos el grid de búsqueda, la
   transformación `log1p` y el feature engineering originales. Las métricas de la
   familia `xgboost` (R² = 0,8310, RMSE = $4.897) están en línea con las
   documentadas en el README original (0,8353 / $4.835); la diferencia se explica
   por la cantidad de iteraciones de la búsqueda aleatoria. Lo que sí se agregó
   son tres familias más para medirlo contra algo (§ 2.14): el enunciado pide un
   criterio de "mejor modelo", y un criterio que se aplica a un solo candidato no
   está eligiendo nada.
6. **Python 3.11.** El proyecto original apuntaba a 3.10; se subió a 3.11 porque
   MLflow 3.x requiere ≥ 3.10 y 3.11 es la versión con soporte más estable hoy.

---

## 5. Docker: eliminado, no adaptado

**Decisión:** se eliminaron del repositorio el `Dockerfile`, el
`docker-compose.yaml`, el `entrypoint.sh` y el `scripts/init-mlflow-db.sql`
(que sólo existía para inicializar la base de MLflow dentro del contenedor). El
proyecto se entrega para ejecución local, documentada en `SOLUTION.md`.

**Tensión con el enunciado.** Los requisitos técnicos piden *"mantener
compatibilidad con la ejecución actual del proyecto (incluyendo Docker si ya
está configurado)"*, y el repo original **sí** lo tenía configurado. Esta es una
desviación deliberada y el motivo es el siguiente.

**Por qué.** La máquina de desarrollo no tiene Docker instalado, así que el
flujo containerizado **nunca se ejecutó**. Mantenerlo obligaba a elegir entre
dos malas opciones:

1. **Entregarlo sin verificar.** El pipeline cambió mucho —backend PostgreSQL
   para MLflow, una segunda base, un paso nuevo de promoción, volúmenes para
   artefactos—, y la adaptación de los tres archivos se hizo *a ojo*. Ya había
   aparecido al menos una divergencia real respecto del flujo local: el
   `docker-compose.yaml` seguía partiendo los runs en dos experimentos
   (`insurance-charges-training` / `-scoring`) después de que la solución los
   unificara en uno solo (§ 2.13). Todo lo que `SOLUTION.md` promete sobre
   comparar validación y producción en un mismo gráfico **no se cumplía** dentro
   del contenedor. Una infraestructura que no se corrió no es una garantía de
   reproducibilidad: es una afirmación sin respaldo, y si falla en la máquina de
   quien evalúa, es peor que su ausencia.
2. **Instalar Docker y verificarlo.** Fuera del alcance de este challenge.

**Qué se pierde y cómo se compensa.** Lo único que aportaba el contenedor era
levantar PostgreSQL y ejecutar cinco comandos en orden. Lo primero son dos
`createdb`; lo segundo son los cinco comandos que `SOLUTION.md` lista
explícitamente, en el mismo orden en que los encadenaba el `entrypoint.sh`:

```
pytest → db_setup → training → promote_model → scoring
```

El proyecto no depende de Docker en ningún punto: toda la configuración se lee
de variables de entorno (`src/config.py`), así que containerizarlo de nuevo es
directo para quien tenga con qué probarlo.

**Lo que sí está verificado:** el pipeline completo corre end-to-end en local,
con venv de Python 3.11 y PostgreSQL 17 en `localhost`.

---

## 6. El `README.md` original no se modifica

**Restricción del proyecto:** `README.md` es el documento del equipo de ciencia
de datos y se deja exactamente como estaba. Toda la documentación de esta
solución vive en dos archivos nuevos:

- **`SOLUTION.md`** — cómo ejecutar el pipeline, qué se registra en MLflow y cómo
  leer el monitoreo
- **`DECISIONS.md`** — este archivo: supuestos, análisis forense y bugs

Esto crea una tensión menor con el enunciado, que pide *"documentar en
`README.md` cómo ejecutar entrenamiento, scoring y revisión de resultados"*. Se
resuelve a favor de la restricción del proyecto: el contenido que el challenge
pide existe completo, sólo que en `SOLUTION.md`, que está enlazado desde el
primer párrafo.

Consecuencia práctica: el hallazgo #8 de la tabla de bugs (el README menciona un
`.env.example` que no existe) **queda documentado pero no corregido en el
README**. La instrucción correcta —`cp .env.template .env`— está en
`SOLUTION.md`, con una nota al pie que aclara la discrepancia.
