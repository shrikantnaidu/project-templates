.PHONY: help install run lint test format docker-build docker-run docker-compose

IMAGE ?= uv-docker

help:
	@echo "install         Install development dependencies"
	@echo "run             Run the app locally"
	@echo "lint            Check formatting and lint rules"
	@echo "test            Run tests"
	@echo "format          Format Python files"
	@echo "docker-build    Build the production image"
	@echo "docker-run      Run the production image"
	@echo "docker-compose  Run the development service"

install:
	uv sync --dev

run:
	uv run python main.py

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

format:
	uv run ruff format .
	uv run ruff check --fix .

docker-build:
	docker build -t $(IMAGE) .

docker-run: docker-build
	docker run --rm $(IMAGE)

docker-compose:
	docker compose run --rm app
