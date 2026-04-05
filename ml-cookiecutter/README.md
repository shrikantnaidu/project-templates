# ML Project Template

A modern, self-contained machine learning project template using the latest Python tools.

## 🚀 Features

- **📦 `uv` Package Manager** - Blazing fast dependency management
- **⚙️ Pydantic Settings** - Type-safe configuration with automatic `.env` loading
- **📝 Loguru** - Beautiful, zero-config logging
- **🔧 Ruff** - Ultra-fast linting and formatting
- **🧪 Pytest** - Modern testing with coverage
- **🌐 FastAPI** - REST API for model serving
- **📊 MLflow** - Experiment tracking and model registry
- **🏗️ FTI Architecture** - Feature, Training, Inference pipeline structure

## 📁 Project Structure

```text
ml-project/
├── src/ml_project/
│   ├── config.py            # Configuration management
│   ├── logging.py           # Logging setup
│   ├── cli.py               # Command-line interface
│   ├── api.py               # FastAPI application
│   ├── data/                # Data loading and demo dataset prep
│   ├── features/            # Feature engineering processors
│   ├── models/              # Model training and evaluation helpers
│   └── pipelines/           # FTI pipeline orchestration
├── tests/                   # Test suite
├── data/
│   ├── raw/
│   └── processed/
├── pyproject.toml
└── Makefile
```

## 🏁 Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Initialize project directories
uv run python -m ml_project.cli init

# Optional: start MLflow server in Docker
# docker compose up -d mlflow-db mlflow-server
```

## 🎬 Churn Demo Showcase (End-to-End)

This template ships with a ready-to-demo **Telco churn classification** workflow.

### 1) Download and prepare demo data

```bash
uv run python -m ml_project.cli prepare-churn-demo
```

This command downloads the public Telco churn CSV and creates
`data/processed/churn_features.parquet` with cleaned and one-hot encoded features.

### 2) Train a classifier and register in MLflow

```bash
uv run python -m ml_project.cli train \
  --data processed/churn_features.parquet \
  --target Churn \
  --experiment churn_demo
```

The command prints both a **Run ID** and a model URI: `runs:/<run_id>/model`.

### 3) Run batch predictions

```bash
uv run python -m ml_project.cli predict \
  --data processed/churn_features.parquet \
  --model-uri runs:/<run_id>/model \
  --output artifacts/churn_predictions.json
```

### 4) Serve as API

```bash
uv run python -m ml_project.cli serve
```

Then call:

- `GET /health`
- `POST /predict`

Example payload:

```json
{
  "features": [
    {
      "SeniorCitizen": 0,
      "tenure": 1,
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85,
      "gender_Male": 0,
      "Partner_Yes": 1
    }
  ]
}
```


## 🧰 Demo Assets

A ready-to-use demo pack is available at:

- `demo/churn_showcase/README.md`
- `demo/churn_showcase/talk_track.md`
- `demo/churn_showcase/sample_request.json`

Use these assets for a repeatable live walkthrough.

## 🔧 Common Commands

```bash
make help
make lint
make format
make test
make test-cov
make serve
```

## 🐳 Docker

- `docker-compose.yml` provisions:
  - PostgreSQL backend for MLflow
  - MLflow tracking server
  - Inference API service

## 📄 License

MIT License.
