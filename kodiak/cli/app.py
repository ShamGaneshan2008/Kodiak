
"""CLI configuration management for the Kodiak command-line interface.

This module is responsible solely for locating, loading, saving, and
validating the CLI's own configuration file (``config.toml``). It has
no knowledge of application settings, database configuration, or
FastAPI settings — those live under ``kodiak/config/``.
"""

from __future__ import annotations

import platform
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w

CONFIG_DIR_NAME = ".kodiak"
CONFIG_FILE_NAME = "config.toml"

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_THEMES = frozenset({"dark", "light", "auto"})


@dataclass(slots=True)
class CLIConfig:
    """Represents the persisted configuration of the Kodiak CLI.

    Attributes:
        api_key: API key used by the CLI to authenticate outbound
            requests. Empty string means unset.
        provider: Name of the AI provider the CLI is configured to use.
        model: Identifier of the model the CLI is configured to use.
        default_repository: Path or identifier of the default repository
            the CLI operates against when none is specified.
        telemetry_enabled: Whether anonymous usage telemetry is enabled.
        theme: Display theme preference for CLI output.
        log_level: Minimum log level the CLI should emit.
    """

    api_key: str = ""
    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    default_repository: str = ""
    telemetry_enabled: bool = True
    theme: str = "auto"
    log_level: str = "INFO"


class ConfigError(Exception):
    """Raised when the CLI configuration is invalid or cannot be processed."""


class ConfigManager:
    """Manages the lifecycle of the Kodiak CLI configuration file.

    The manager is responsible for locating the configuration directory
    and file, creating defaults when absent, and providing typed access
    to configuration values. It performs no business logic beyond
    configuration persistence and validation.
    """

    def __init__(self, config_directory: Path | None = None) -> None:
        """Initialize the configuration manager.

        Args:
            config_directory: Optional override for the configuration
                directory. When omitted, the platform-appropriate default
                directory is used. Primarily useful for testing.
        """
        self._config_directory = config_directory or self._default_config_directory()

    @staticmethod
    def _default_config_directory() -> Path:
        """Resolve the platform-appropriate CLI configuration directory.

        Returns:
            Path to ``.kodiak`` under the current user's home directory
            on all supported platforms.
        """
        return Path.home() / CONFIG_DIR_NAME

    def config_directory(self) -> Path:
        """Return the directory in which the configuration file lives.

        Returns:
            Absolute path to the configuration directory.
        """
        return self._config_directory

    def config_path(self) -> Path:
        """Return the full path to the configuration file.

        Returns:
            Absolute path to ``config.toml``.
        """
        return self._config_directory / CONFIG_FILE_NAME

    def exists(self) -> bool:
        """Check whether a configuration file currently exists on disk.

        Returns:
            True if the configuration file exists, False otherwise.
        """
        return self.config_path().is_file()

    def create_default(self) -> CLIConfig:
        """Create and persist a default configuration file.

        The configuration directory is created if it does not already
        exist. Any existing configuration file is overwritten.

        Returns:
            The default `CLIConfig` instance that was written to disk.
        """
        config = CLIConfig()
        self.save(config)
        return config

    def load(self) -> CLIConfig:
        """Load the configuration from disk, creating a default if needed.

        Returns:
            The loaded `CLIConfig`. If no configuration file exists, a
            default one is created and returned.

        Raises:
            ConfigError: If the configuration file exists but cannot be
                parsed or contains invalid values.
        """
        if not self.exists():
            return self.create_default()

        try:
            raw_bytes = self.config_path().read_bytes()
            data = tomllib.loads(raw_bytes.decode("utf-8"))
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            raise ConfigError(
                f"Failed to read configuration file at {self.config_path()}: {exc}"
            ) from exc

        config = self._from_dict(data)
        self.validate(config)
        return config

    def save(self, config: CLIConfig) -> None:
        """Persist the given configuration to disk.

        Args:
            config: The configuration instance to write.

        Raises:
            ConfigError: If the configuration is invalid or cannot be
                written to disk.
        """
        self.validate(config)
        self._config_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.config_path().write_bytes(tomli_w.dumps(asdict(config)).encode("utf-8"))
        except OSError as exc:
            raise ConfigError(
                f"Failed to write configuration file at {self.config_path()}: {exc}"
            ) from exc

    def reset(self) -> CLIConfig:
        """Reset the configuration file to its default values.

        Returns:
            The default `CLIConfig` instance that was written to disk.
        """
        return self.create_default()

    def update(self, **values: Any) -> CLIConfig:
        """Update one or more configuration fields and persist the result.

        Args:
            **values: Field names and new values to apply to the current
                configuration.

        Returns:
            The updated and persisted `CLIConfig`.

        Raises:
            ConfigError: If an unknown field is supplied or the resulting
                configuration is invalid.
        """
        config = self.load()
        valid_fields = {f.name for f in fields(CLIConfig)}

        for key, value in values.items():
            if key not in valid_fields:
                raise ConfigError(f"Unknown configuration field: '{key}'")
            setattr(config, key, value)

        self.save(config)
        return config

    def get(self, key: str) -> Any:
        """Retrieve a single configuration value by field name.

        Args:
            key: Name of the configuration field to retrieve.

        Returns:
            The current value of the requested field.

        Raises:
            ConfigError: If the field name is unknown.
        """
        config = self.load()
        if key not in {f.name for f in fields(CLIConfig)}:
            raise ConfigError(f"Unknown configuration field: '{key}'")
        return getattr(config, key)

    def set(self, key: str, value: Any) -> CLIConfig:
        """Set a single configuration value and persist the result.

        Args:
            key: Name of the configuration field to set.
            value: New value for the field.

        Returns:
            The updated and persisted `CLIConfig`.

        Raises:
            ConfigError: If the field name is unknown or the resulting
                configuration is invalid.
        """
        return self.update(**{key: value})

    @staticmethod
    def validate(config: CLIConfig) -> None:
        """Validate that a configuration instance holds acceptable values.

        Args:
            config: The configuration instance to validate.

        Raises:
            ConfigError: If any field holds an invalid value.
        """
        if config.log_level not in _VALID_LOG_LEVELS:
            raise ConfigError(
                f"Invalid log_level '{config.log_level}'. "
                f"Expected one of: {sorted(_VALID_LOG_LEVELS)}"
            )
        if config.theme not in _VALID_THEMES:
            raise ConfigError(
                f"Invalid theme '{config.theme}'. Expected one of: {sorted(_VALID_THEMES)}"
            )
        if not isinstance(config.telemetry_enabled, bool):
            raise ConfigError("Field 'telemetry_enabled' must be a boolean.")
        if not config.provider.strip():
            raise ConfigError("Field 'provider' must not be empty.")
        if not config.model.strip():
            raise ConfigError("Field 'model' must not be empty.")

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> CLIConfig:
        """Construct a `CLIConfig` from a raw TOML-parsed dictionary.

        Unknown keys present in the dictionary are ignored; missing keys
        fall back to the dataclass defaults.

        Args:
            data: Dictionary parsed from the configuration file.

        Returns:
            A populated `CLIConfig` instance.
        """
        valid_fields = {f.name for f in fields(CLIConfig)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return CLIConfig(**filtered)


def get_platform_identifier() -> str:
    """Return a short identifier for the current operating system.

    Useful for diagnostics or platform-specific configuration decisions
    made by callers of this module.

    Returns:
        One of "windows", "darwin", "linux", or "unknown".
    """
    system = platform.system().lower()
    if system in {"windows", "darwin", "linux"}:
        return system
    return "unknown"

import typer

APP_NAME = "kodiak"
APP_HELP = "Kodiak — an autonomous AI software engineer."


def create_app() -> typer.Typer:
    app = typer.Typer(
        name=APP_NAME,
        help=APP_HELP,
        add_completion=True,
        no_args_is_help=True,
    )
    return app


app = create_app()

APP_NAME = "kodiak"
APP_HELP = "Kodiak — an autonomous AI software engineer."


def create_app() -> typer.Typer:
    app = typer.Typer(
        name=APP_NAME,
        help=APP_HELP,
        add_completion=True,
        no_args_is_help=True,
    )
    return app


app = create_app()