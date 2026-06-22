"""
Feature flag abstraction layer.

Priority order (highest → lowest):
  1. LOCAL_OVERRIDES dict   – forced overrides in code / tests
  2. Unleash remote server  – when UNLEASH_URL is configured
  3. DEFAULT_FLAGS dict     – static defaults defined below

Usage::

    from kodiak.config.feature_flags import is_enabled, flags

    if is_enabled("new_reflection_loop"):
        ...

    # or use the singleton for context-aware evaluation:
    result = flags.is_enabled("multi_model_consensus", context={"userId": user_id})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Static default flag values ────────────────────────────────────────────────
# Keep this in sync with your Unleash project. These values are used when
# Unleash is not configured or unreachable.

DEFAULT_FLAGS: dict[str, bool] = {
    # Core orchestration
    "reflection_loop": True,
    "approval_gate": True,
    "multi_agent_parallel": False,
    # LLM features
    "multi_model_consensus": False,
    "llm_cost_optimizer": True,
    "structured_output": True,
    # RAG / memory
    "semantic_memory": True,
    "episodic_memory": True,
    "cross_repo_rag": False,
    "call_graph_indexing": False,
    # GitHub integration
    "auto_pr_creation": True,
    "auto_code_review": False,
    "webhook_issue_triage": True,
    # Sandbox
    "sandbox_network_egress": False,
    # Learning
    "cross_repo_pattern_sync": False,
    "reward_model_training": False,
    # Plugin marketplace
    "plugin_marketplace": False,
}

# In-process overrides (highest priority). Mutate in tests or local dev.
LOCAL_OVERRIDES: dict[str, bool] = {}


@dataclass
class FlagContext:
    """Context passed to Unleash for targeting / gradual rollout."""

    user_id: str | None = None
    session_id: str | None = None
    remote_address: str | None = None
    properties: dict[str, str] = field(default_factory=dict)

    def to_unleash(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        if self.user_id:
            ctx["userId"] = self.user_id
        if self.session_id:
            ctx["sessionId"] = self.session_id
        if self.remote_address:
            ctx["remoteAddress"] = self.remote_address
        if self.properties:
            ctx["properties"] = self.properties
        return ctx


class FeatureFlags:
    """
    Thin wrapper around Unleash with local-override fallback.
    Initialise once via ``configure_feature_flags()`` and then use the
    module-level ``flags`` singleton or the ``is_enabled()`` shortcut.
    """

    def __init__(self) -> None:
        self._client: Any = None  # UnleashClient or None
        self._ready: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def configure(self, settings: Any | None = None) -> None:
        """
        Connect to Unleash if configured, otherwise operate in offline mode.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._ready:
            return

        if settings is None:
            from kodiak.config.settings import get_settings

            settings = get_settings()

        if settings.UNLEASH_URL and settings.UNLEASH_API_TOKEN:
            self._setup_unleash(settings)
        else:
            logger.info(
                "feature_flags: Unleash not configured, using defaults only"
            )

        self._ready = True

    def _setup_unleash(self, settings: Any) -> None:
        try:
            from UnleashClient import UnleashClient  # type: ignore[import]

            client = UnleashClient(
                url=settings.UNLEASH_URL,
                app_name=settings.UNLEASH_APP_NAME,
                custom_headers={"Authorization": settings.UNLEASH_API_TOKEN},
            )
            client.initialize_client()
            self._client = client
            logger.info("feature_flags: Unleash client initialised", extra={"url": settings.UNLEASH_URL})
        except ImportError:
            logger.warning(
                "feature_flags: UnleashClient package not installed; "
                "pip install UnleashClient to enable remote flags"
            )
        except Exception as exc:
            logger.warning(
                "feature_flags: Failed to connect to Unleash, using defaults",
                exc_info=exc,
            )

    def shutdown(self) -> None:
        """Flush the Unleash client. Call on application shutdown."""
        if self._client is not None:
            try:
                self._client.destroy()
            except Exception:
                pass

    # ── Evaluation ────────────────────────────────────────────────────────

    def is_enabled(
        self,
        flag_name: str,
        context: FlagContext | None = None,
        default: bool | None = None,
    ) -> bool:
        """
        Check whether a feature flag is enabled.

        Priority: LOCAL_OVERRIDES > Unleash > DEFAULT_FLAGS > *default*.

        Args:
            flag_name: The feature flag name (snake_case).
            context:   Optional targeting context for Unleash rollouts.
            default:   Fallback if the flag is not in any source. Defaults
                       to ``False`` when ``None``.

        Returns:
            bool
        """
        # 1. Local override
        if flag_name in LOCAL_OVERRIDES:
            return LOCAL_OVERRIDES[flag_name]

        # 2. Unleash remote
        if self._client is not None:
            try:
                ctx = context.to_unleash() if context else {}
                return bool(self._client.is_enabled(flag_name, ctx))
            except Exception as exc:
                logger.debug(
                    "feature_flags: Unleash evaluation error",
                    extra={"flag": flag_name, "error": str(exc)},
                )

        # 3. Static defaults
        if flag_name in DEFAULT_FLAGS:
            return DEFAULT_FLAGS[flag_name]

        # 4. Caller-provided default
        if default is not None:
            return default

        logger.debug("feature_flags: unknown flag %r, returning False", flag_name)
        return False

    def get_all(self) -> dict[str, bool]:
        """Return the resolved state of every known flag (no Unleash context)."""
        return {name: self.is_enabled(name) for name in DEFAULT_FLAGS}

    def override(self, flag_name: str, value: bool) -> None:
        """Set a local in-process override. Useful in tests."""
        LOCAL_OVERRIDES[flag_name] = value

    def clear_override(self, flag_name: str) -> None:
        """Remove a local in-process override."""
        LOCAL_OVERRIDES.pop(flag_name, None)

    def clear_all_overrides(self) -> None:
        """Remove all local in-process overrides. Useful in test teardown."""
        LOCAL_OVERRIDES.clear()


# ── Module-level singleton ────────────────────────────────────────────────────

flags = FeatureFlags()


def configure_feature_flags(settings: Any | None = None) -> None:
    """Convenience wrapper. Call once at startup."""
    flags.configure(settings)


def is_enabled(
    flag_name: str,
    context: FlagContext | None = None,
    default: bool = False,
) -> bool:
    """
    Module-level shortcut for ``flags.is_enabled(...)``.

    Usage::

        from kodiak.config.feature_flags import is_enabled

        if is_enabled("multi_model_consensus"):
            ...
    """
    return flags.is_enabled(flag_name, context=context, default=default)