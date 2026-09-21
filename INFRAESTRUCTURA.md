# Infraestructura — diseño propuesto

> **Nada de este documento se ejecutó.** Es un diseño. La razón por la que el proyecto no
> se entrega containerizado es precisamente que no había forma de probarlo, y entregar
> infra sin correrla es afirmar algo que no se verificó.

**Verificado:** la ejecución local (venv de Python 3.11 + PostgreSQL 17).

---

## 1. Por qué hoy no está

El repositorio original traía `Dockerfile`, `docker-compose.yaml`, `entrypoint.sh` y un
script de inicialización. **Se eliminaron en vez de adaptarse**: el entorno de desarrollo
no tenía Docker, así que ese flujo nunca se ejecutó. Y ya había una divergencia real: el
compose seguía partiendo los runs en **dos** experimentos de MLflow después de que la
solución los unificara, así que lo que se promete sobre comparar validación contra
producción en un mismo gráfico **no se cumplía dentro del contenedor**.

Los originales están en el historial: `git show decb3a1:docker-compose.yaml`.

---

## 2. Qué cambió desde entonces

Restaurarlos no alcanza. Esto es lo que hay que corregir:

| Cambio en el pipeline | Qué implica |
|---|---|
| MLflow pasó de `file://` a **PostgreSQL** | Hace falta una **segunda base** (`mlflow_db`) creada al inicializar Postgres |
| **Un solo experimento** | Hay que **borrar** `MLFLOW_EXPERIMENT_TRAINING` y `_SCORING`, no adaptarlas |
| Paso nuevo **`promote_model.py`** | Va entre training y scoring, y **no** debe abortar si rechaza |
| **`config/monitoring_rules.json`** | Archivo nuevo: copiarlo a la imagen |
| Tabla **`alert_history`** | El volumen de Postgres es lo que le da memoria al monitoreo |

---

## 3. Los archivos

**`Dockerfile`** — el original ya era correcto (multi-stage, usuario no-root) y sirve con
tres cambios: base `python:3.11-slim`, `COPY config/ /app/config/`, y un healthcheck real
—era `import sys; sys.exit(0)`— con `python -c "import mlflow, sklearn, xgboost"`.
`data/` se monta de sólo lectura en vez de copiarse: agregar un lote no rehace la imagen.

**`scripts/init-mlflow-db.sql`** — `CREATE DATABASE mlflow_db OWNER metlife_user;` más su
`GRANT`. Ojo: Postgres sólo ejecuta `/docker-entrypoint-initdb.d` con el volumen vacío; si
ya se levantó antes, `docker compose down -v` o crear la base a mano.

**`docker-compose.yaml`**

```yaml
services:
  postgres:
    image: postgres:17-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${DB_NAME:-metlife_db}
      POSTGRES_USER: ${DB_USER:-metlife_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD:?definir DB_PASSWORD}
    ports: ["5432:5432"]
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-mlflow-db.sql:/docker-entrypoint-initdb.d/10-init.sql:ro
    healthcheck: { test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-metlife_user}"],
                   interval: 5s, retries: 5 }

  ml_pipeline:
    build: { context: ., dockerfile: Dockerfile }
    # "no", NO "on-failure": es un job por lotes. Debe fallar una vez y dejar
    # el log, no reintentarse en loop.
    restart: "no"
    depends_on: { postgres: { condition: service_healthy } }
    environment:
      DB_HOST: postgres
      DB_PASSWORD: ${DB_PASSWORD:?definir DB_PASSWORD}
      MLFLOW_DB_NAME: mlflow_db
      # UN solo experimento. No definir MLFLOW_EXPERIMENT_TRAINING/_SCORING.
      MLFLOW_EXPERIMENT: insurance-charges
      MLFLOW_MODEL_NAME: insurance-charges-regressor
      MONITORING_RULES_FILE: config/monitoring_rules.json
      FAIL_ON_ALERT: ${FAIL_ON_ALERT:-false}
      RUN_TESTS: ${RUN_TESTS:-true}
    # data va :ro — el pipeline detecta y alerta, no repara
    volumes: ["./data:/app/data:ro", "./models:/app/models",
              "./results:/app/results", "./mlruns:/app/mlruns"]

  mlflow_ui:                       # `docker compose --profile ui up -d mlflow_ui`
    build: { context: ., dockerfile: Dockerfile }
    profiles: ["ui"]
    depends_on: { postgres: { condition: service_healthy } }
    entrypoint: ["/bin/sh", "-c"]
    command: ["mlflow ui --backend-store-uri \"postgresql+psycopg2://${DB_USER:-metlife_user}:$${DB_PASSWORD}@postgres:5432/mlflow_db\" --host 0.0.0.0 --port 5000"]
    ports: ["5001:5000"]           # en macOS el 5000 lo ocupa AirPlay
    volumes: ["./mlruns:/app/mlruns"]

volumes:
  postgres_data:
```

---

## 4. `entrypoint.sh`

```
pytest  ->  esperar Postgres  ->  db_setup  ->  training  ->  promote_model  ->  scoring
   ^            ^                                                  ^              ^
   todos abortan el pipeline si fallan                   NO aborta         + 2a corrida
                                                                            (opcional)
```

Tres decisiones que no hay que perder al reescribirlo:

- **Los tests van primero.** Si el parseo de los lotes o el monitoreo están rotos, no
  tiene sentido gastar el entrenamiento para descubrirlo después.
- **La promoción no aborta.** Que un candidato sea rechazado por no mejorar al modelo
  vigente es el comportamiento correcto, no un error.
- **La segunda corrida de scoring verifica las alertas.** En la primera todo sale
  `NUEVA`; en la segunda, `PERSISTE`. Si algo sale `NUEVA` dos veces, la deduplicación
  está rota.

---

## 5. Lo que Docker **no** resuelve

Containerizar hace el entorno reproducible. No lo vuelve productivo.

**1. No hay servidor de tracking, y eso reparte credenciales.** `MLFLOW_TRACKING_URI`
apunta directo a PostgreSQL, así que **todo proceso que loguea a MLflow necesita la
contraseña de la base** (por eso existe `scripts/mlflow_ui.sh`). Lo correcto es un
servidor delante: `MLFLOW_TRACKING_URI=http://mlflow:5000`. **El código ya lo soporta** —
es una variable de entorno y un servicio más.

**2. El artifact store es local.** `./mlruns` no sobrevive a otra máquina. Y el baseline
de monitoreo viaja como artefacto del run que produjo el modelo: si se pierde, **el drift
deja de poder medirse** contra la distribución correcta.

**3. Nadie ejecuta nada.** Cada paso es un script con exit code y `FAIL_ON_ALERT=true`
hace que scoring termine con código ≠ 0 ante un `ALERT`. Falta quién lo dispare.

**4. No hay CI.** Las **172 pruebas no necesitan ni PostgreSQL ni MLflow**: un workflow
con `pytest tests/ -q` corre en menos de un minuto sin levantar servicios.

**5 y 6. Las alertas no salen a ningún lado** —las transiciones `NEW` son el único evento
notificable, falta decidir canal y destinatario— y los **secretos viajan por variables de
entorno**, que en producción pide un gestor.

---

## 6. Camino, y qué validar

| # | Paso | Qué desbloquea |
|---|---|---|
| 1 | CI con los tests | Lo más barato y lo que más previene |
| 2 | Compose, **ejecutado y verificado** | Reproducibilidad real |
| 3 | Servidor de tracking | Deja de repartir credenciales |
| 4 | Artifact store remoto | La trazabilidad sobrevive a la máquina |
| 5 | Scheduler + canal de aviso | El monitoreo corre y avisa solo |

Los pasos 1 y 2 son de horas; el 3 y el 4 son configuración, porque el código ya los
soporta. El 5 requiere decisiones de negocio. Y razonar no es verificar —la misma
distinción que llevó a eliminar los originales—, así que antes de dar esto por bueno: que
la imagen construya con 3.11, que `init-mlflow-db.sql` corra con el volumen vacío, que el
pipeline complete las siete etapas, que la UI muestre **un solo** experimento con los
cuatro tipos de run, y que la segunda corrida de scoring reporte `0 nuevas`.
