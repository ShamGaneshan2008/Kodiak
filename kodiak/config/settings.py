from __future__ import annotations

import secrets
from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ================= DB =================
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://kodiak:kodiak@localhost:5432/kodiak"
    )

    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # ================= APP =================
    APP_NAME: str = "Kodiak"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False

    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))

    API_V1_PREFIX: str = "/api/v1"

    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[AnyHttpUrl | str] = ["http://localhost:3000"]

    # ================= REDIS =================
    REDIS_URL: RedisDsn = Field(default="redis://localhost:6379/0")
    REDIS_CELERY_DB: int = 1
    REDIS_CACHE_DB: int = 2

    # ================= AUTH =================
    JWT_ALGORITHM: str = "HS256"

    # ================= LLM =================
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    # ================= VALIDATORS =================
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_hosts(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v

    # ================= HELPERS =================
    @property
    def database_url_sync(self) -> str:
        return str(self.DATABASE_URL).replace("+asyncpg", "")

    @property
    def database_url_async(self) -> str:
        return str(self.DATABASE_URL)

    @property
    def celery_broker_url(self) -> str:
        base = str(self.REDIS_URL).rsplit("/", 1)[0]
        return f"{base}/{self.REDIS_CELERY_DB}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
