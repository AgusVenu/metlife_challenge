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
| 1 | `.gitignore` | La regla `*.txt` ignoraba `requirements.txt`, que directamente **no existía en el repo**, aunque el `Dockerfile` hace `COPY requirements.txt .` | **El build de Docker fallaba.** Bloqueante | Corregido: archivo creado + negaciones en `.gitignore` |
| 2 | `.gitignore` | `*.json`, `*.txt`, `*.pkl` globales sin excepciones | Ignoraba reportes, metadata y documentación | Corregido con negaciones explícitas |
| 3 | `training.py:153` | `os.getenv('HIPERPARAM_ITERATIONS', 350)` — typo; `docker-compose` exporta `HYPERPARAM_ITERATIONS` | La variable **nunca tenía efecto**: se entrenaba siempre con 350 iteraciones × 5 folds = 1.750 fits en vez de 50. Reproducibilidad rota | Corregido en `config.py`, aceptando el nombre viejo como respaldo |
| 4 | `scoring.py:82` | `actual_charges = df['charges'].values` sin guarda | **Reventaba con `prod3`**, que no tiene target | Corregido: el target es opcional en todo el flujo |
| 5 | `training.py:213` | `mape_log` se calculaba con los valores en escala original | Duplicaba el MAPE en dólares bajo otro nombre | Corregido |
| 6 | `training.py:224-225` | `adj_r2` usaba `p = X_train.shape[1]`, las features **antes** del `ColumnTransformer` | R² ajustado mal calculado: ignoraba las dummies del one-hot | Corregido usando `get_feature_names_out()` |
| 7 | `training.py:419-421` | "Paso 5 beta" llamaba a `create_preprocessor()` y `define_hyperparameter_grid()` y descartaba el resultado | Trabajo muerto | Eliminado |
| 8 | `README.md:139` | Indica `cp .env.example .env`, pero el archivo se llama `.env.template` (`.env.example` nunca existió en el repo) | Instrucción de setup rota | **No corregido ahí**: el README original no se modifica (ver § 6). La instrucción correcta está en `SOLUTION.md` |
| 9 | `docker-compose.yaml` | Sin volúmenes para MLflow ni para `./data` | Los experimentos se perderían al bajar el contenedor | Corregido |
| 10 | `training.py:312` | El log decía "Symlink actualizado" pero el código usa `shutil.copy2` | Mensaje engañoso | Corregido |
| 11 | `docker-compose.yaml` | `restart: on-failure` en un job por lotes | Un pipeline que falla se reintentaría en loop | Cambiado a `restart: "no"` |

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
5. **El modelo no se toca.** El challenge es de MLOps: se conservan el XGBoost,
   el grid de búsqueda, la transformación `log1p` y el feature engineering
   originales. Las métricas obtenidas (R² = 0,8310, RMSE = $4.897) están en línea
   con las documentadas en el README original (0,8353 / $4.835); la diferencia
   se explica por la cantidad de iteraciones de la búsqueda aleatoria.
6. **Python 3.11.** El proyecto original apuntaba a 3.10; se subió a 3.11 porque
   MLflow 3.x requiere ≥ 3.10 y 3.11 es la versión con soporte más estable hoy.
   `Dockerfile` y entorno local usan la misma versión.

---

## 5. Qué quedó sin verificar

**El flujo con Docker está escrito pero no fue ejecutado**, porque la máquina de
desarrollo no tiene Docker instalado. El `Dockerfile`, el `docker-compose.yaml` y
el `entrypoint.sh` están actualizados y son coherentes con el pipeline local
(mismas variables de entorno, mismos pasos, mismos volúmenes), y el
`requirements.txt` que faltaba —y que hacía fallar el build— ya existe. Pero la
verificación end-to-end se hizo **en local**: venv con Python 3.11 y PostgreSQL 17
en `localhost`.

`SOLUTION.md` marca explícitamente qué parte está verificada y cuál no.

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
