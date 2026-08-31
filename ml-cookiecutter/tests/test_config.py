"""Tests for configuration module."""

from ml_project.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    assert settings.project_name == "ml-project"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.model_name == "default_model"
    assert settings.model_alias == "latest"


def test_settings_from_env(monkeypatch):
    """Test settings from environment variables."""
    monkeypatch.setenv("PROJECT_NAME", "test-project")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()
    assert settings.project_name == "test-project"
    assert settings.environment == "production"
    assert settings.log_level == "DEBUG"


def test_is_production():
    """Test is_production property."""
    settings = Settings(environment="production")
    assert settings.is_production is True
    assert settings.is_development is False


def test_is_development():
    """Test is_development property."""
    settings = Settings(environment="development")
    assert settings.is_development is True
    assert settings.is_production is False


def test_path_resolution():
    """Test that paths are resolved correctly."""
    settings = Settings(data_dir="./data")
    assert settings.data_dir.is_absolute()


def test_ensure_directories(tmp_path, monkeypatch):
    """Test directory creation."""
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
    )
    settings.ensure_directories()

    assert settings.data_dir.exists()
    assert settings.raw_data_dir.exists()
    assert settings.processed_data_dir.exists()
    assert settings.models_dir.exists()
    assert settings.artifacts_dir.exists()
