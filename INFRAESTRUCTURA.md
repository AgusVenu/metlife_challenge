# Infraestructura — diseño propuesto

> **Este documento describe infraestructura que NO se ejecutó.** Es un diseño, no una
> entrega. Se escribe así a propósito: la razón por la que el proyecto no se entrega
> containerizado es exactamente que no había forma de probarlo, y entregar archivos de
> infra sin correrlos es afirmar algo que no se verificó. Lo que sigue es el plan para
> quien tenga con qué ejecutarlo, con los puntos flojos señalados.

| | |
|---|---|
| **Estado** | Diseño. Ningún archivo de este documento está en el repositorio |
| **Lo que sí está verificado** | La ejecución local: venv de Python 3.11 + PostgreSQL 17 |
| **Alcance** | Contenedores, orquestación, CI, programación de tareas y el camino a producción real |

---

## 1. Por qué hoy no está

El repositorio original traía `Dockerfile`, `docker-compose.yaml`, `entrypoint.sh` y un
script de inicialización de la base. **Se eliminaron en vez de adaptarse.**

El entorno de desarrollo no tenía Docker, así que el flujo containerizado nunca se
ejecutó. Mientras tanto el pipeline cambió mucho: backend PostgreSQL para MLflow, una
segunda base de datos, un paso nuevo de promoción, un archivo de reglas de monitoreo, una
tabla de historial de alertas. La adaptación de esos archivos se estaba haciendo a ojo, y
ya había aparecido una divergencia real: el compose seguía partiendo los runs en dos
experimentos de MLflow después de que la solución los unificara en uno, de modo que todo
lo que la documentación promete sobre comparar validación y producción en un mismo
gráfico **no se cumplía dentro del contenedor**.

Una infraestructura que no se corrió no es una garantía de reproducibilidad: es una
afirmación sin respaldo, y si falla en manos de quien la recibe es peor que su ausencia.

Los archivos originales siguen en el historial de git:

```bash
git show decb3a1:Dockerfile
git show decb3a1:docker-compose.yaml
git show decb3a1:entrypoint.sh
```

---

## 2. Qué cambió desde que esos archivos existían

Quien retome esto necesita saber qué hay que corregir. No alcanza con restaurarlos.

| Cambio en el pipeline | Qué implica para la infra |
|---|---|
| MLflow pasó de `file://` a **PostgreSQL** | Hace falta una **segunda base** (`mlflow_db`), creada al inicializar el contenedor de Postgres |
| Los artefactos van a `./mlruns` | Necesita volumen propio, o un artifact store remoto (§ 6) |
| **Un solo experimento** (`insurance-charges`) | El compose original exportaba `MLFLOW_EXPERIMENT_TRAINING` y `_SCORING` por separado. **Hay que borrar esas dos variables**, no adaptarlas |
| Paso nuevo: **`promote_model.py`** | El entrypoint tiene que ejecutarlo entre training y scoring, y **no** abortar si rechaza: un rechazo es una decisión válida |
| **`config/monitoring_rules.json`** | Archivo nuevo que hay que copiar a la imagen o montar |
| Tabla **`alert_history`** | La crea `db_setup.py`, no requiere cambios de infra, pero su volumen de Postgres ahora importa más: es el que da memoria al monitoreo |
| Python 3.11 | El Dockerfile original usaba `python:3.10-slim`. MLflow 3.x requiere ≥ 3.10, pero el proyecto se desarrolló y verificó sobre **3.11** |
| ~40 variables de entorno | El compose original exportaba 7. Casi todas tienen default razonable en `src/config.py`, así que el compose sólo necesita las que difieren |

---

## 3. `Dockerfile`

El original ya era bueno: multi-stage, usuario no-root, healthcheck. Los cambios son
menores.

```dockerfile
# ---------- build ----------
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# postgresql-client: el entrypoint espera a que la base responda con `psql`
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY src/     /app/src/
COPY scripts/ /app/scripts/
COPY tests/   /app/tests/
COPY config/  /app/config/        # <-- nuevo: reglas de monitoreo
COPY entrypoint.sh .

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p models results results/predictions logs mlruns \
    && chown -R appuser:appuser /app \
    && chmod +x entrypoint.sh

USER appuser

# El healthcheck original era `import sys; sys.exit(0)`, que no verifica nada.
# Este al menos confirma que las dependencias pesadas cargan.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import mlflow, sklearn, xgboost" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
```

**`data/` no se copia**: se monta como volumen de sólo lectura, así agregar un
`dataset_prod4_feats.csv` no obliga a reconstruir la imagen. El descubrimiento de lotes
ya tolera archivos nuevos sin tocar código.

---

## 4. `docker-compose.yaml`

Tres servicios. El de MLflow queda detrás de un *profile* para que no arranque con un
`up` normal.

```yaml
services:

  postgres:
    image: postgres:17-alpine
    container_name: metlife_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${DB_NAME:-metlife_db}
      POSTGRES_USER: ${DB_USER:-metlife_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD:?definir DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      # Crea la SEGUNDA base, la de tracking de MLflow.
      - ./scripts/init-mlflow-db.sql:/docker-entrypoint-initdb.d/10-init-mlflow-db.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-metlife_user} -d ${DB_NAME:-metlife_db}"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks: [metlife_network]

  ml_pipeline:
    build: { context: ., dockerfile: Dockerfile }
    container_name: metlife_ml_pipeline
    # "no", no "on-failure": es un job por lotes. Un pipeline que falla no debe
    # reintentarse en loop; debe fallar una vez y dejar el log.
    restart: "no"
    depends_on:
      postgres: { condition: service_healthy }
    environment:
      DB_HOST: postgres
      DB_PASSWORD: ${DB_PASSWORD:?definir DB_PASSWORD}
      MLFLOW_DB_NAME: mlflow_db
      MLFLOW_ARTIFACT_ROOT: ./mlruns
      # UN solo experimento. NO definir MLFLOW_EXPERIMENT_TRAINING/_SCORING:
      # separarlos rompe la serie validacion -> prod1 -> prod2.
      MLFLOW_EXPERIMENT: insurance-charges
      MLFLOW_MODEL_NAME: insurance-charges-regressor
      MONITORING_RULES_FILE: config/monitoring_rules.json
      PYTHONUNBUFFERED: 1
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      HYPERPARAM_ITERATIONS: ${HYPERPARAM_ITERATIONS:-50}
      TRAIN_MODEL_FAMILIES: ${TRAIN_MODEL_FAMILIES:-xgboost,random_forest,hist_gradient_boosting,elasticnet}
      SCORING_MODE: prod
      FAIL_ON_ALERT: ${FAIL_ON_ALERT:-false}
      RUN_TESTS: ${RUN_TESTS:-true}
    volumes:
      - ./data:/app/data:ro        # sólo lectura: el pipeline no repara datos
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
      - ./mlruns:/app/mlruns
    networks: [metlife_network]

  mlflow_ui:
    build: { context: ., dockerfile: Dockerfile }
    container_name: metlife_mlflow_ui
    profiles: ["ui"]              # no arranca con un `up` normal
    restart: unless-stopped
    depends_on:
      postgres: { condition: service_healthy }
    environment:
      DB_HOST: postgres
      DB_PASSWORD: ${DB_PASSWORD:?definir DB_PASSWORD}
      MLFLOW_DB_NAME: mlflow_db
    entrypoint: ["/bin/sh", "-c"]
    command:
      - >
        mlflow ui
        --backend-store-uri "postgresql+psycopg2://${DB_USER:-metlife_user}:$${DB_PASSWORD}@postgres:5432/mlflow_db"
        --host 0.0.0.0 --port 5000
    ports:
      - "5001:5000"               # 5001: en macOS el 5000 lo ocupa AirPlay
    volumes:
      - ./mlruns:/app/mlruns
    networks: [metlife_network]

volumes:
  postgres_data:

networks:
  metlife_network:
    driver: bridge
```

### `scripts/init-mlflow-db.sql`

```sql
CREATE DATABASE mlflow_db OWNER metlife_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO metlife_user;
```

> **Trampa conocida**: Postgres sólo ejecuta `/docker-entrypoint-initdb.d` cuando el
> volumen está vacío. Si el contenedor ya se levantó antes sin ese archivo, hay que hacer
> `docker compose down -v` (que borra los datos) o crear la base a mano.

---

## 5. `entrypoint.sh` — la secuencia

El original encadenaba tests → espera de Postgres → `db_setup` → `training` → `scoring`.
Hoy faltan dos pasos y hay un matiz sobre cuál puede fallar.

```
1. pytest                    aborta si falla   (si RUN_TESTS=true)
2. esperar a PostgreSQL      aborta a los 30 reintentos
3. python src/db_setup.py    aborta si falla
4. python src/training.py    aborta si falla
5. python src/promote_model.py   <-- NUEVO.  NO aborta si "falla"
6. python src/scoring.py     aborta si falla
7. python src/scoring.py     <-- opcional: la segunda corrida demuestra las alertas
```

Tres decisiones que conviene no perder al reescribirlo:

**Los tests van primero, antes de entrenar.** Si el parseo de los lotes de producción o
el motor de monitoreo están rotos, no tiene sentido gastar los minutos del entrenamiento
para descubrirlo después.

**El paso de promoción no aborta el pipeline.** Que un candidato sea rechazado por no
mejorar al modelo vigente es el comportamiento correcto, no un error. El scoring sigue
con el modelo que ya estaba sirviendo.

**Una segunda corrida de scoring es la demostración del sistema de alertas.** En la
primera todo sale como `NUEVA`; en la segunda, las mismas señales pasan a `PERSISTE`. Si
algo sale `NUEVA` dos veces, la deduplicación está rota. Vale como verificación
automática dentro del contenedor.

### Uso

```bash
cp .env.template .env            # y definir DB_PASSWORD
docker compose up --build        # pipeline completo
docker compose --profile ui up -d mlflow_ui    # UI en http://localhost:5001
docker compose down              # -v tambien borra la base
```

---

## 6. Lo que Docker **no** resuelve

Containerizar hace el entorno reproducible. No convierte esto en un sistema productivo.
Los huecos reales, en orden de importancia:

### 6.1 No hay servidor de tracking — y eso reparte credenciales

Hoy `MLFLOW_TRACKING_URI` apunta **directo a PostgreSQL**:

```
postgresql+psycopg2://metlife_user:PASSWORD@localhost:5432/mlflow_db
```

O sea que **todo proceso que loguea a MLflow necesita la contraseña de la base**. Por eso
existe `scripts/mlflow_ui.sh`, que resuelve el URI sin imprimirlo. Funciona para una
máquina; no escala a un equipo.

Lo correcto es un **servidor de tracking** delante:

```
MLFLOW_TRACKING_URI=http://mlflow:5000
```

El servidor tiene las credenciales; los clientes no tienen ninguna. **El código ya lo
soporta sin cambios** — `src/config.py` acepta cualquier URI, incluido `http://`. Es
cambiar una variable de entorno y agregar un servicio al compose.

### 6.2 El artifact store es local

`./mlruns` en disco. Los modelos, los `baseline_stats.json` y los reportes no sobreviven a
otra máquina. Para producción: S3, GCS o Azure Blob como `MLFLOW_ARTIFACT_ROOT`.

Consecuencia concreta hoy: el baseline de monitoreo viaja como artefacto del run que
produjo el modelo. Si el artifact store se pierde, **el drift deja de poder medirse**
contra la distribución correcta.

### 6.3 Nadie ejecuta nada

No hay tarea programada. El pipeline se dispara a mano.

| Opción | Cuándo conviene |
|---|---|
| `cron` dentro del contenedor | Lo más simple. Suficiente si los lotes llegan con frecuencia fija |
| GitHub Actions con `schedule` | Sin infra propia, pero necesita acceso a la base |
| Airflow / Prefect / Dagster | Cuando haya dependencias entre tareas, reintentos y backfill. Hoy sería desproporcionado |

El pipeline ya está preparado: cada paso es un script con exit code, y
`FAIL_ON_ALERT=true` hace que scoring termine con código ≠ 0 si algún lote queda en
`ALERT`. Eso alcanza para que un scheduler sepa que algo pasó.

### 6.4 No hay CI

No existe `.github/workflows`. Lo mínimo útil, en cada push:

```yaml
# .github/workflows/tests.yml  (propuesto)
name: tests
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -q
```

Las 172 pruebas **no necesitan ni PostgreSQL ni MLflow** — se diseñaron así
deliberadamente. Ese workflow corre en menos de un minuto y no requiere servicios.

Un segundo workflow, más ambicioso, levantaría Postgres como *service container* y
correría el pipeline completo con `FAIL_ON_ALERT=true` sobre datos de prueba.

### 6.5 Las alertas no salen a ningún lado

Quedan en la tabla `alert_history` y en los reportes. Nadie se entera salvo que mire.

El enganche ya existe y está diseñado para esto: las transiciones **`NEW`** son el único
evento notificable, precisamente porque el resto ya se sabía. Un notificador es un
consumidor de esa lista. Lo que falta es decidir el destinatario y el canal, no
arquitectura.

### 6.6 Secretos en variables de entorno

`DB_PASSWORD` viaja por el entorno y por el `.env`. Aceptable en local; en producción
corresponde un gestor de secretos (Docker secrets, AWS Secrets Manager, Vault). El
`.env.template` ya evita traer una contraseña real por defecto.

---

## 7. Camino sugerido

En orden, porque cada escalón habilita al siguiente:

| # | Paso | Qué desbloquea |
|---|---|---|
| 1 | **CI con los tests** | Lo más barato y lo que más previene. No necesita servicios |
| 2 | **Docker Compose**, con los archivos de este documento, **ejecutado y verificado** | Reproducibilidad real en cualquier máquina |
| 3 | **Servidor de tracking** en vez de conexión directa a la base | Deja de repartir credenciales de Postgres |
| 4 | **Artifact store remoto** | La trazabilidad sobrevive a la máquina |
| 5 | **Scheduler** | El monitoreo corre solo |
| 6 | **Canal de notificación** sobre las transiciones `NEW` | Alguien se entera sin mirar |
| 7 | **Gestor de secretos** | Requisito para cualquier entorno compartido |

Los pasos 1 y 2 son de horas. El 3 y el 4 son de configuración, porque el código ya los
soporta. Del 5 en adelante hay que decidir cosas de negocio —cada cuánto llegan los
lotes, quién recibe las alertas— que no son técnicas.

---

## 8. Una advertencia sobre este documento

Nada de lo de acá se ejecutó. El `Dockerfile` y el `docker-compose.yaml` propuestos están
razonados contra el estado actual del código, pero **razonar no es verificar**: es
exactamente la distinción que llevó a eliminar los archivos originales en vez de
entregarlos sin probar.

Quien los implemente debería tratarlos como un punto de partida y validar, como mínimo:

- que la imagen construya con Python 3.11 y las versiones de `requirements.txt`
- que `init-mlflow-db.sql` corra con el volumen vacío
- que el pipeline complete las siete etapas
- que la UI de MLflow muestre **un solo** experimento con los cuatro tipos de run
- que la segunda corrida de scoring reporte `0 nuevas` y no repita las alertas

Recién ahí este documento deja de ser un diseño y pasa a ser infraestructura.
