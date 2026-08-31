# uv-docker

A small, production-minded Python template showing how to use `uv` with Docker.
It includes a lockfile-first image build, a non-root runtime user, and a Compose
workflow for local development.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop or Docker Engine with BuildKit

## Local development

```bash
uv sync --dev
uv run python main.py
uv run ruff check .
uv run pytest
```

`uv.lock` is committed so local and container installs resolve the same package
versions. After changing `pyproject.toml`, refresh it with `uv lock`.

## Docker

Build and run the production-style image:

```bash
docker build -t uv-docker .
docker run --rm uv-docker
```

Or use the development service, which mounts the working tree into the container:

```bash
docker compose run --rm app
```

The Dockerfile installs dependencies before copying application code, so normal
source changes can reuse the dependency layer. The final image runs as the
`app` user and starts with the virtual environment on `PATH`.

## Project layout

```text
uv-docker/
├── Dockerfile
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

Replace `main.py` with your application entry point and update the final `CMD`
when your project grows beyond this minimal example.
