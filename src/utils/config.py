"""Configuration management using Pydantic settings."""
import os
from functools import lru_cache
from typing import List

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from YAML and environment variables."""

    # Application Settings
    app_name: str = "Customer Segmentation API"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = True
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Server Settings
    host: str = "0.0.0.0"
    port: int = Field(default=8000, env="PORT")
    workers: int = 1
    reload: bool = True

    # Model Settings
    model_path: str = "models/model.pkl"
    n_clusters: int = 5
    random_state: int = 0
    max_iter: int = 300
    init_method: str = "k-means++"

    # Data Settings
    data_path: str = "data/Mall_Customers.csv"
    feature_columns: List[str] = ["Annual Income (k$)", "Spending Score (1-100)"]
    income_min: float = 15
    income_max: float = 137
    score_min: float = 1
    score_max: float = 99

    # API Settings
    batch_max_size: int = 100
    request_timeout: int = 30
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]

    # MLflow Settings
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "customer_segmentation"
    model_registry_name: str = "kmeans_segmentation"

    # DVC Settings
    remote_name: str = "origin"
    remote_url: str = Field(default="", env="DAGSHUB_URL")

    # Monitoring Settings
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"

    # Security Settings
    api_key_header: str = "X-API-Key"
    api_key: str = Field(default="", env="API_KEY")

    @field_validator("income_min", "income_max")
    @classmethod
    def validate_income_range(cls, v, info):
        """Validate income range."""
        if info.field_name == "income_max" and hasattr(info.data, "income_min"):
            if v <= info.data.get("income_min", 0):
                raise ValueError("income_max must be greater than income_min")
        return v

    @field_validator("score_min", "score_max")
    @classmethod
    def validate_score_range(cls, v, info):
        """Validate score range."""
        if info.field_name == "score_max" and hasattr(info.data, "score_min"):
            if v <= info.data.get("score_min", 0):
                raise ValueError("score_max must be greater than score_min")
        return v

    @field_validator("batch_max_size")
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("batch_max_size must be greater than 0")
        return v

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Loads configuration from YAML file based on ENVIRONMENT variable,
    then overrides with environment variables.

    Returns:
        Settings: Cached settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development")

    # Load base config
    base_config_path = "configs/config.yaml"
    config_data = {}

    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Load environment-specific config
    env_config_path = f"configs/{environment}.yaml"
    if os.path.exists(env_config_path):
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
            config_data.update(env_config)

    # Create settings instance (env vars will override YAML)
    settings = Settings(**config_data)

    # Validate required settings in production
    if settings.environment == "production":
        if not settings.api_key:
            raise ValueError("API_KEY must be set in production environment")

    return settings
