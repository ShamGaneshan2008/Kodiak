from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Kodiak"
    environment: Literal["development", "test", "staging", "production"] = Field(
        default="development", alias="KODIAK_ENV"
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    secret_key: str = "change-me"
    database_url: str = "postgresql+asyncpg://kodiak:kodiak@localhost:5432/kodiak"
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    github_app_id: str | None = None
    github_app_private_key: str | None = None
    github_webhook_secret: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    qwen_api_key: str | None = None
    deepseek_api_key: str | None = None
    default_llm_provider: str = "local"
    default_llm_model: str = "local-dev"
    sandbox_image: str = "kodiak-sandbox:latest"
    sandbox_timeout_seconds: int = 120
    log_level: str = "INFO"

    @property
    def is_test(self) -> bool:
        return self.environment == "test"


@lru_cache
def get_settings() -> Settings:
    return Settings()
