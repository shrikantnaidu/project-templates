"""
ML Project - A modern, self-contained machine learning project template.

This package provides a clean structure for ML projects with:
- Configuration management via Pydantic
- Modular pipeline architecture
- Built-in logging and experiment tracking
- CLI interface for common tasks
"""

from ml_project.config import settings

__version__ = "0.1.0"
__all__ = ["settings", "__version__"]
