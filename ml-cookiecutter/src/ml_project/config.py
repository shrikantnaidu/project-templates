"""
Configuration management using Pydantic Settings.

Automatically loads from environment variables and .env files.
Type-safe and validated at runtime.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Project Metadata
    # =========================================================================
    project_name: str = Field(default="ml-project", description="Project name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )

    # =========================================================================
    # Paths
    # =========================================================================
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    models_dir: Path = Field(default=Path("./models"), description="Models directory")
    artifacts_dir: Path = Field(
        default=Path("./artifacts"), description="Artifacts directory"
    )

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    # =========================================================================
    # ML Tracking
    # =========================================================================
    mlflow_tracking_uri: str | None = Field(
        default=None, description="MLflow tracking server URI"
    )
    wandb_project: str | None = Field(default=None, description="W&B project name")

    # =========================================================================
    # API (for serving)
    # =========================================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("data_dir", "models_dir", "artifacts_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v).resolve()

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw data directory."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed data directory."""
        return self.data_dir / "processed"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.artifacts_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
