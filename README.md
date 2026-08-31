# Project Templates

A small collection of reusable project starters. Each template is self-contained
and can be copied into a new project or used as a reference.

For now, this README covers:

- [`ml-cookiecutter`](ml-cookiecutter/) — a practical machine-learning project
  template with data, feature, training, inference, API, and MLOps foundations.
- [`uv-docker`](uv-docker/) — a minimal Python application template demonstrating
  reproducible `uv` dependency management with Docker.

The remaining templates will be documented separately later.

## Choosing a template

| Template | Best for | Python | Docker |
|---|---|---:|---|
| `ml-cookiecutter` | Starting an ML/MLflow project | 3.11+ | Optional; Compose includes MLflow |
| `uv-docker` | Starting a small containerized Python app | 3.12+ | Core workflow |

## `ml-cookiecutter`

`ml-cookiecutter` provides a structured ML project with:

- `src`-layout Python package and typed environment configuration
- Pandas/Polars data loading helpers
- Reusable feature processors and a leakage-safe training pipeline
- MLflow experiment tracking, model registration, and alias-based inference
- FastAPI serving support
- Pytest, Ruff, mypy, pre-commit, Make targets, and a committed `uv.lock`
- Optional Docker Compose services for MLflow, PostgreSQL, and the inference API

### Start locally

```bash
cd ml-cookiecutter
cp .env.example .env
uv sync --extra dev --extra api
uv run python -m ml_project.cli info
uv run pytest
```

Typical workflow:

```bash
# Put training data under data/raw/
uv run python -m ml_project.cli feature --data raw/train.csv --target label
uv run python -m ml_project.cli train --data raw/train.csv --target label
uv run python -m ml_project.cli predict \
  --data raw/test.csv \
  --model models:/default_model@latest
```

The default local MLflow tracking location is `./mlruns`. For the full tracking
and serving stack, use:

```bash
docker compose up --build
```

See the [`ml-cookiecutter README`](ml-cookiecutter/README.md) and its
[`docs/`](ml-cookiecutter/docs/) directory for the detailed project guide.

## `uv-docker`

`uv-docker` is intentionally small. It demonstrates:

- A committed lockfile and locked installs
- Dependency-layer caching before application source is copied
- A non-root runtime user
- A slim Python image with the virtual environment on `PATH`
- Separate local and Docker development commands

### Start locally

```bash
cd uv-docker
uv sync --dev
uv run python main.py
uv run pytest
uv run ruff check .
```

### Build and run with Docker

```bash
cd uv-docker
docker build -t uv-docker .
docker run --rm uv-docker
```

For development with the working tree mounted into the container:

```bash
docker compose run --rm app
```

See the [`uv-docker README`](uv-docker/README.md) for the project layout and
details on the Docker and Compose setup.

## Common requirements

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Python matching the selected template
- Docker Desktop or Docker Engine for container workflows

Both templates include Makefiles with shortcuts for installation, testing,
linting, formatting, and Docker operations where applicable.
