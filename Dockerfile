# =============================================================================
# MetLife ML Ops Challenge - imagen del pipeline
# Multi-stage: las dependencias de compilacion no llegan a la imagen final.
# =============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Dependencias de compilacion (psycopg2, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Stage 2 - runtime
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="jm.aragonpaz@gmail.com" \
      description="ML Engineering Challenge - MetLife (MLflow + scoring batch + monitoreo)" \
      version="2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY src/ /app/src/
COPY tests/ /app/tests/
COPY data/ /app/data/
COPY entrypoint.sh .

RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chmod +x entrypoint.sh

# Directorios de salida. `mlflow` guarda el backend SQLite del tracking y
# `mlruns` los artefactos; ambos se montan como volumenes en docker-compose
# para que los experimentos sobrevivan a `docker compose down`.
RUN mkdir -p models results results/predictions logs mlruns mlflow && \
    chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import mlflow, sklearn, xgboost; import sys; sys.exit(0)" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
