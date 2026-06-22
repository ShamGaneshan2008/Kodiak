"""
Central application settings using Pydantic BaseSettings.
All configuration is read from environment variables or .env file.
"""

from __future__ import annotations

import secrets
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
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

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "Kodiak"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[AnyHttpUrl | str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://kodiak:kodiak@localhost:5432/kodiak"
    )
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: RedisDsn = Field(default="redis://localhost:6379/0")
    REDIS_CELERY_DB: int = 1
    REDIS_CACHE_DB: int = 2
    REDIS_CACHE_TTL: int = 3600  # seconds

    # ── Auth ──────────────────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    API_KEY_LENGTH: int = 64

    # ── LLM Providers ────────────────────────────────────────────────────────
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_DEFAULT_MODEL: str = "gpt-4o"

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"

    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-coder"

    QWEN_API_KEY: str | None = None
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_DEFAULT_MODEL: str = "qwen2.5-coder-32b-instruct"

    LLM_REQUEST_TIMEOUT: int = 120  # seconds
    LLM_MAX_RETRIES: int = 3
    LLM_COST_BUDGET_USD_PER_TASK: float = 1.0

    # ── Vector Store (ChromaDB) ───────────────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_PREFIX: str = "kodiak"

    # ── GitHub ────────────────────────────────────────────────────────────────
    GITHUB_APP_ID: str | None = None
    GITHUB_APP_PRIVATE_KEY: str | None = None
    GITHUB_WEBHOOK_SECRET: str | None = None
    GITHUB_OAUTH_CLIENT_ID: str | None = None
    GITHUB_OAUTH_CLIENT_SECRET: str | None = None

    # ── Sandbox ───────────────────────────────────────────────────────────────
    SANDBOX_IMAGE: str = "kodiak-sandbox:latest"
    SANDBOX_CPU_LIMIT: str = "2.0"
    SANDBOX_MEMORY_LIMIT: str = "2g"
    SANDBOX_TIMEOUT_SECONDS: int = 300
    SANDBOX_NETWORK_DISABLED: bool = True

    # ── Workers / Celery ─────────────────────────────────────────────────────
    CELERY_TASK_SOFT_TIME_LIMIT: int = 600
    CELERY_TASK_TIME_LIMIT: int = 900
    CELERY_WORKER_CONCURRENCY: int = 4

    # ── Observability ────────────────────────────────────────────────────────
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "json"  # "json" | "text"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_SERVICE_NAME: str = "kodiak"
    SENTRY_DSN: str | None = None

    # ── Plugins ───────────────────────────────────────────────────────────────
    PLUGINS_DIR: str = "plugins"
    PLUGIN_MARKETPLACE_URL: str = "https://marketplace.kodiak.dev"

    # ── Feature Flags ────────────────────────────────────────────────────────
    UNLEASH_URL: str | None = None
    UNLEASH_API_TOKEN: str | None = None
    UNLEASH_APP_NAME: str = "kodiak"

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == Environment.TESTING

    @property
    def database_url_sync(self) -> str:
        """Sync DSN for Alembic migrations."""
        return str(self.DATABASE_URL).replace("+asyncpg", "")

    @property
    def celery_broker_url(self) -> str:
        base = str(self.REDIS_URL).rsplit("/", 1)[0]
        return f"{base}/{self.REDIS_CELERY_DB}"

    @property
    def celery_result_backend(self) -> str:
        return self.celery_broker_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance. Import and call this everywhere."""
    return Settings()