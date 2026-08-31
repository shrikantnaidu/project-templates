# =============================================================================
# ML Project Makefile
# =============================================================================
# Provides common commands for development, testing, and running the project.
# Uses 'uv' as the package manager (https://github.com/astral-sh/uv)
#
# Usage: make <target>
# =============================================================================

.PHONY: help install install-dev install-all sync lint format type-check test test-cov clean init feature train predict serve

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install      - Install core dependencies"
	@echo "    make install-dev  - Install with development tools"
	@echo "    make install-all  - Install all optional dependencies"
	@echo "    make sync         - Sync dependencies from lockfile"
	@echo "    make init         - Initialize project directories"
	@echo ""
	@echo "  Development:"
	@echo "    make lint         - Run linter (ruff)"
	@echo "    make format       - Format code (ruff)"
	@echo "    make test         - Run tests"
	@echo "    make test-cov     - Run tests with coverage"
	@echo "    make type-check   - Run mypy"
	@echo ""
	@echo "  ML Workflow:"
	@echo "    make feature DATA=path/to/data.csv [TARGET=column]"
	@echo "    make train        - Train model (requires DATA and TARGET)"
	@echo "    make predict      - Generate predictions (requires DATA)"
	@echo "    make serve        - Start API server"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean        - Remove build artifacts"
	@echo "    make clean-all    - Remove all generated files"

# =============================================================================
# Setup Commands
# =============================================================================

install:
	uv sync

install-dev:
	uv sync --extra dev

install-all:
	uv sync --all-extras

sync:
	uv sync --frozen

init:
	uv run python -c "from ml_project.config import settings; settings.ensure_directories()"
	@echo "Project directories initialized"

# =============================================================================
# Development Commands
# =============================================================================

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

type-check:
	uv run mypy src

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/ml_project --cov-report=html --cov-report=term

# =============================================================================
# ML Workflow Commands
# =============================================================================

# Usage: make feature DATA=raw/train.csv TARGET=label
feature:
ifndef DATA
	$(error DATA is required. Usage: make feature DATA=path/to/data.csv [TARGET=column])
endif
	uv run python -m ml_project.cli feature --data $(DATA) $(if $(TARGET),--target $(TARGET),)

# Usage: make train DATA=raw/train.csv TARGET=label
train:
ifndef DATA
	$(error DATA is required. Usage: make train DATA=path/to/data.csv TARGET=column_name)
endif
ifndef TARGET
	$(error TARGET is required. Usage: make train DATA=path/to/data.csv TARGET=column_name)
endif
	uv run python -m ml_project.cli train --data $(DATA) --target $(TARGET)

# Usage: make predict DATA=raw/test.csv [MODEL=models:/default_model@latest]
predict:
ifndef DATA
	$(error DATA is required. Usage: make predict DATA=path/to/data.csv [MODEL=model-uri])
endif
	uv run python -m ml_project.cli predict --data $(DATA) $(if $(MODEL),--model $(MODEL),)

serve:
	uv run python -m ml_project.cli serve

# =============================================================================
# Maintenance Commands
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv/
	rm -rf models/
	rm -rf artifacts/
	rm -rf mlruns/
