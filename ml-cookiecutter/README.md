# ML Project Template

A modern, self-contained machine learning project template using the latest Python tools.

## 🚀 Features

- **📦 `uv` Package Manager** - Blazing fast dependency management (replaces pip, poetry, conda)
- **⚙️ Pydantic Settings** - Type-safe configuration with automatic `.env` loading
- **📝 Loguru** - Beautiful, zero-config logging
- **🔧 Ruff** - Ultra-fast linting and formatting (replaces flake8, black, isort)
- **🧪 Pytest** - Modern testing with coverage
- **🌐 FastAPI** - Optional REST API for model serving
- **📊 MLflow tracking** - Experiment tracking and model registration
- **📈 Optional W&B integration** - Add-on experiment visualization

## 📁 Project Structure

```
ml-project/
├── src/ml_project/          # Source code (src-layout)
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── logging.py           # Logging setup
│   ├── cli.py               # Command-line interface
│   ├── api.py               # FastAPI application
│   ├── data/                # Data loading utilities
│   ├── features/            # Feature engineering
│   ├── models/              # Model training & evaluation
│   └── pipelines/           # ML pipeline orchestration
├── tests/                   # Test suite
├── data/                    # Data directory (gitignored)
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── models/                  # Saved models (gitignored)
├── notebooks/               # Jupyter notebooks
├── pyproject.toml           # Project configuration
├── Makefile                 # Common commands
└── .env.example             # Environment template
```

## 🏁 Quick Start

### Prerequisites

Install `uv` (if not already installed):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
# Clone/copy this template
cd your-project

# Install dependencies (creates .venv automatically)
uv sync

# For local development and API serving
uv sync --extra dev --extra api

# Or with all optional dependencies
uv sync --all-extras

# Initialize project directories
make init
# Or: uv run python -m ml_project.cli init

# Copy environment file
cp .env.example .env
```

## 📖 Usage

### Command Line Interface

```bash
# Show project info
uv run python -m ml_project.cli info

# Build features while preserving the target column
uv run python -m ml_project.cli feature --data raw/train.csv --target label

# Train a model directly from raw data
uv run python -m ml_project.cli train --data raw/train.csv --target label

# Make predictions
uv run python -m ml_project.cli predict --data raw/test.csv --model models:/default_model@latest

# Serve model as API
uv run python -m ml_project.cli serve
```

### Using Make Commands

```bash
make help          # Show all commands
make install       # Install dependencies
make install-dev   # Install with dev tools
make lint          # Run linter
make format        # Format code
make test          # Run tests
make test-cov      # Run tests with coverage
make feature DATA=raw/train.csv TARGET=label
make train DATA=raw/train.csv TARGET=label
make predict DATA=raw/test.csv MODEL=models:/default_model@latest
make serve
```

### Python API

```python
from sklearn.ensemble import RandomForestClassifier

from ml_project.data import load_csv
from ml_project.features import StandardScaler, NullFiller
from ml_project.pipelines import FeaturePipeline, TrainingPipeline

data = load_csv("raw/train.csv")
features = FeaturePipeline([
    NullFiller(strategy="mean"),
    StandardScaler(),
])
trainer = TrainingPipeline(experiment_name="default")
run_id = trainer.run(
    data=data,
    target_column="target",
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    feature_pipeline=features,
)
print(f"MLflow run: {run_id}")
```

## 🔧 Configuration

Settings are loaded from environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_NAME` | ml-project | Project name |
| `ENVIRONMENT` | development | dev/staging/production |
| `DATA_DIR` | ./data | Data directory |
| `MODELS_DIR` | ./models | Models directory |
| `LOG_LEVEL` | INFO | Logging level |
| `API_HOST` | 0.0.0.0 | API host |
| `API_PORT` | 8000 | API port |
| `MODEL_NAME` | default_model | MLflow registered model |
| `MODEL_ALIAS` | latest | MLflow model alias/version |

## 🧪 Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linter
uv run ruff check src tests

# Format code
uv run ruff format src tests

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/ml_project --cov-report=html

# Type checking
uv run mypy src
```

## 📦 Optional Dependencies

Install extras as needed:

```bash
# Deep learning (PyTorch + Lightning)
uv sync --extra deep-learning

# Optional W&B integration (MLflow is included in the core install)
uv sync --extra tracking

# API serving (FastAPI + Uvicorn)
uv sync --extra api

# All optional dependencies
uv sync --all-extras
```

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src

# Install dependencies
RUN uv sync --frozen --no-dev --extra api

# Run
CMD ["uv", "run", "python", "-m", "ml_project.cli", "serve"]
```

## 📄 License

MIT License - feel free to use this template for your projects.
