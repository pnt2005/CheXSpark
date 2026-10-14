-- ─────────────────────────────────────────────────────────────
-- CheXSpark: PostgreSQL bootstrap for MLflow tracking backend
-- Runs automatically on first container start (docker-entrypoint-initdb.d)
-- ─────────────────────────────────────────────────────────────

-- Dedicated role + database for MLflow experiment tracking & model registry
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Optional: dedicated database for future metadata catalog (Lakehouse catalog, inference cache, etc.)
CREATE DATABASE chexspark_metadata OWNER postgres;
