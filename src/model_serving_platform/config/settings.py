"""Environment based settings for the service."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Define process and service metadata settings from environment variables.

    These settings establish the minimum startup contract for the API process.
    They keep Stage 1 simple while giving later stages stable configuration
    names for bundle loading, readiness, and runtime concerns.
    Parameters: values are loaded from environment variables or defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_SERVING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = Field(default="model-serving-platform")
    service_environment: str = Field(default="local")
    service_version: str = Field(default="0.1.0")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    bundle_path: str = Field(default="./bundles/graphsage")
    max_top_k: int = Field(default=25, gt=0)
    default_attachment_strategy: Literal["interaction", "cosine"] = Field(
        default="interaction"
    )
    external_api_timeout_seconds: float = Field(default=2.0, gt=0.0)
    external_api_retry_count: int = Field(default=2, ge=0)
    external_api_retry_backoff_seconds: float = Field(default=0.2, ge=0.0)
    restricted_network_mode: bool = Field(default=False)
    external_description_lookup_url: str | None = Field(default=None)
    external_interaction_lookup_url: str | None = Field(default=None)
