-- ============================================================================
-- Crea la base que usa MLflow como backend de tracking.
--
-- Postgres ejecuta los scripts de /docker-entrypoint-initdb.d una unica vez,
-- cuando el volumen de datos esta vacio. Si ya levantaste el stack antes de
-- agregar este archivo, hay que recrear el volumen:  docker compose down -v
--
-- Se usa una base separada de la de negocio a proposito: MLflow crea unas 15
-- tablas propias y mezclarlas con training_dataset / batch_predictions /
-- batch_monitoring vuelve ilegible el esquema de la aplicacion.
-- ============================================================================

CREATE DATABASE mlflow_db OWNER metlife_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO metlife_user;
