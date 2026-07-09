"""Public API for the Kodiak CLI utility package.

Re-exports the CLI's exception hierarchy, output formatting helpers,
filesystem path utilities, and input validators so consumers can depend
on `kodiak.cli.utils` directly instead of reaching into individual
submodules.
"""

from __future__ import annotations

from kodiak.cli.utils.errors import (
    AuthenticationError,
    CommandExecutionError,
    ConfigurationError,
    KodiakCLIError,
    RepositoryError,
    ServiceError,
    ValidationError,
)
from kodiak.cli.utils.output import (
    get_console,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from kodiak.cli.utils.paths import (
    ensure_directory_exists,
    expand_user_path,
    is_git_repository,
    normalize_path,
    resolve_repository_path,
)
from kodiak.cli.utils.validators import (
    validate_branch_name,
    validate_directory,
    validate_issue_number,
    validate_non_empty_string,
    validate_repository_path,
)

__all__ = [
    "AuthenticationError",
    "CommandExecutionError",
    "ConfigurationError",
    "KodiakCLIError",
    "RepositoryError",
    "ServiceError",
    "ValidationError",
    "get_console",
    "print_error",
    "print_info",
    "print_success",
    "print_warning",
    "ensure_directory_exists",
    "expand_user_path",
    "is_git_repository",
    "normalize_path",
    "resolve_repository_path",
    "validate_branch_name",
    "validate_directory",
    "validate_issue_number",
    "validate_non_empty_string",
    "validate_repository_path",
]