# Modern Tooling Decisions

This template prioritizes speed, type-safety, and developer experience by using the latest Python ecosystem tools.

## ⚡ [uv](https://github.com/astral-sh/uv)
**Role:** Package & Project Manager
- **Why?** It is 10-100x faster than `pip` and replaces `poetry`, `pip-tools`, and `venv` management.
- **Locking:** Uses `uv.lock` for deterministic builds.

## 🛠️ [Ruff](https://github.com/astral-sh/ruff)
**Role:** Linter & Formatter
- **Why?** Replaces `flake8`, `isort`, and `black` with a single tool written in Rust. It is near-instant even on large codebases.

## 🏗️ [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
**Role:** Configuration Management
- **Why?** Environment variables are validated at startup. If a required API key or path is missing, the app fails fast with a clear error.
- **Integration:** Automatically loads from `.env`.

## 📝 [Loguru](https://github.com/Delgan/loguru)
**Role:** Logging
- **Why?** Standard Python logging is verbose and complex. Loguru provides beautiful, structured, and rotate-able logs with a single import.

## 🧪 [Pytest](https://docs.pytest.org/)
**Role:** Testing
- **Why?** Industry standard for Python testing. We include fixtures and coverage tracking out of the box.
