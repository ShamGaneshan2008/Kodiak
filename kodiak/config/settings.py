from __future__ import annotations

import secrets
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kodiak import __version__


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


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ================= DB =================
    DATABASE_URL: str = Field(default="postgresql+asyncpg://kodiak:kodiak@localhost:5432/kodiak")

    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # ================= APP =================
    APP_NAME: str = "Kodiak"
    APP_VERSION: str = __version__
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: Literal["console", "json"] = "console"

    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))

    API_V1_PREFIX: str = "/api/v1"

    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[AnyHttpUrl | str] = ["http://localhost:3000"]

    # ================= REDIS =================
    REDIS_URL: RedisDsn = Field(default=RedisDsn("redis://localhost:6379/0"))
    REDIS_CELERY_DB: int = 1
    REDIS_CACHE_DB: int = 2

    # ================= AUTH =================
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, gt=0)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, gt=0)

    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/github/callback"

    # ================= LLM =================
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    primary_llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_max_retries: int = Field(default=2, ge=0)
    llm_max_tokens: int = Field(default=4096, gt=0)

    # ================= FEATURE FLAGS =================
    UNLEASH_URL: str | None = None
    UNLEASH_API_TOKEN: str | None = None
    UNLEASH_APP_NAME: str = "kodiak"

    # ================= TRACING =================
    OTEL_SERVICE_NAME: str = "kodiak"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

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

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
