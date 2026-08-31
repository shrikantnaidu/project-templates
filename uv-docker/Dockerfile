# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12

# Keep uv separate from the Python runtime so its version is explicit and easy
# to update independently of the base image.
FROM ghcr.io/astral-sh/uv:0.12.7 AS uv
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

# Run the application without root privileges.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

# Copy only dependency metadata first so source changes keep the dependency
# layer cached.
COPY --chown=app:app pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY --chown=app:app main.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

USER app

CMD ["python", "main.py"]
