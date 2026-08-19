"""Automatic discovery, validation, and registration of Kodiak agents."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import Any

import structlog

from kodiak.agents.adapters import DiscoveredAgentHandle, ManagerAgentAdapter
from kodiak.agents.base import AgentRole, BaseAgent
from kodiak.agents.capabilities import ROLE_CAPABILITIES, default_capabilities_for_role
from kodiak.agents.registry import (
    AgentAlreadyRegisteredError,
    AgentRegistry,
    InvalidAgentMetadataError,
)

logger = structlog.get_logger(__name__)

DEFAULT_PACKAGE = "kodiak.agents"
DEFAULT_EXCLUDE_MODULES = frozenset(
    {
        "base",
        "registry",
        "manager",
        "discovery",
        "selector",
        "adapters",
        "lifecycle",
    }
)


class DiscoveryRejectReason(StrEnum):
    NOT_A_CLASS = "not_a_class"
    NOT_BASE_AGENT = "not_base_agent"
    IS_BASE_AGENT = "is_base_agent"
    ABSTRACT = "abstract"
    MISSING_ROLE = "missing_role"
    INVALID_ROLE = "invalid_role"
    INVALID_AGENT_ID = "invalid_agent_id"
    MISSING_DEPENDENCIES = "missing_dependencies"
    DUPLICATE = "duplicate"
    UNRELATED = "unrelated"


@dataclass(frozen=True, slots=True)
class DiscoveryRejection:
    """A candidate that was inspected but not registered."""

    qualname: str
    reason: DiscoveryRejectReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DiscoveryImportError:
    """A module that could not be imported during discovery."""

    module: str
    error: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Outcome of a discovery pass."""

    registered: tuple[str, ...] = ()
    rejections: tuple[DiscoveryRejection, ...] = ()
    import_errors: tuple[DiscoveryImportError, ...] = ()


@dataclass(slots=True)
class _Candidate:
    qualname: str
    module_name: str
    agent_cls: type[BaseAgent]
    agent_id: str
    capabilities: tuple[str, ...]


class AgentDiscovery:
    """Discover BaseAgent implementations and register them in AgentRegistry.

    Responsibilities: find candidates, validate, register.
    Does NOT execute, select, or manage agents.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        package: str = DEFAULT_PACKAGE,
        exclude_modules: frozenset[str] | None = None,
        modules: Sequence[str] | None = None,
        dependencies: Mapping[str, Any] | None = None,
        lifecycle: Any | None = None,
    ) -> None:
        self._registry = registry
        self._package = package
        self._exclude_modules = exclude_modules or DEFAULT_EXCLUDE_MODULES
        self._modules = list(modules) if modules is not None else None
        self._dependencies = dict(dependencies or {})
        self._lifecycle = lifecycle
        self._log = logger.bind(component="agent_discovery")

    async def discover_and_register(self, *, replace: bool = False) -> DiscoveryResult:
        """Run discovery and register all valid agents."""
        registered: list[str] = []
        rejections: list[DiscoveryRejection] = []
        import_errors: list[DiscoveryImportError] = []
        seen_agent_ids: set[str] = set()

        module_names = self._resolve_module_names(import_errors)
        candidates: list[_Candidate] = []

        for module_name in module_names:
            module = self._import_module(module_name, import_errors)
            if module is None:
                continue
            self._collect_candidates(module, module_name, candidates, rejections)

        candidates.sort(key=lambda c: (c.agent_id, c.module_name, c.qualname))

        for candidate in candidates:
            if candidate.agent_id in seen_agent_ids:
                rejections.append(
                    DiscoveryRejection(
                        qualname=candidate.qualname,
                        reason=DiscoveryRejectReason.DUPLICATE,
                        detail=f"agent_id {candidate.agent_id!r} already discovered",
                    )
                )
                continue

            dep_error = self._missing_dependencies(candidate.agent_cls)
            if dep_error is not None:
                rejections.append(
                    DiscoveryRejection(
                        qualname=candidate.qualname,
                        reason=DiscoveryRejectReason.MISSING_DEPENDENCIES,
                        detail=dep_error,
                    )
                )
                continue

            try:
                await self._registry.register(
                    candidate.agent_id,
                    factory=self._make_factory(candidate.agent_cls, candidate.agent_id),
                    name=candidate.qualname.rsplit(".", 1)[-1],
                    capabilities=candidate.capabilities,
                    description=f"Discovered from {candidate.module_name}",
                    dependencies=self._dependencies,
                    replace=replace,
                )
            except AgentAlreadyRegisteredError:
                rejections.append(
                    DiscoveryRejection(
                        qualname=candidate.qualname,
                        reason=DiscoveryRejectReason.DUPLICATE,
                        detail=f"agent_id {candidate.agent_id!r} already registered",
                    )
                )
                continue
            except InvalidAgentMetadataError as exc:
                rejections.append(
                    DiscoveryRejection(
                        qualname=candidate.qualname,
                        reason=DiscoveryRejectReason.INVALID_AGENT_ID,
                        detail=str(exc),
                    )
                )
                continue

            seen_agent_ids.add(candidate.agent_id)
            registered.append(candidate.agent_id)
            if self._lifecycle is not None:
                await self._lifecycle.mark_discovered(candidate.agent_id)
            self._log.info(
                "agent_discovered",
                agent_id=candidate.agent_id,
                qualname=candidate.qualname,
            )

        registered.sort()
        rejections.sort(key=lambda r: (r.qualname, r.reason.value))
        import_errors.sort(key=lambda e: e.module)

        return DiscoveryResult(
            registered=tuple(registered),
            rejections=tuple(rejections),
            import_errors=tuple(import_errors),
        )

    async def register_with_manager(self, manager: Any) -> tuple[str, ...]:
        """Register all registry agents with an AgentManager via adapters."""
        registered: list[str] = []
        metadata_list = await self._registry.list_agents()
        for metadata in metadata_list:
            handle = await self._registry.get(metadata.agent_id)
            if not isinstance(handle, DiscoveredAgentHandle):
                continue
            adapter = ManagerAgentAdapter(
                handle,
                capabilities=frozenset(metadata.capabilities),
            )
            try:
                await manager.register(adapter)
            except Exception as exc:
                if type(exc).__name__ != "AgentAlreadyRegisteredError":
                    raise
            registered.append(metadata.agent_id)
        registered.sort()
        return tuple(registered)

    def _resolve_module_names(self, import_errors: list[DiscoveryImportError]) -> list[str]:
        if self._modules is not None:
            return sorted(self._modules)

        try:
            package = importlib.import_module(self._package)
        except ImportError as exc:
            import_errors.append(DiscoveryImportError(module=self._package, error=str(exc)))
            return []

        if not hasattr(package, "__path__"):
            return sorted([self._package])

        names: list[str] = []
        for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if module_info.name in self._exclude_modules:
                continue
            if module_info.name.startswith("_"):
                continue
            names.append(f"{self._package}.{module_info.name}")
        return names

    def _import_module(
        self,
        module_name: str,
        import_errors: list[DiscoveryImportError],
    ) -> ModuleType | None:
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            self._log.warning("agent_discovery_import_failed", module=module_name, error=str(exc))
            import_errors.append(DiscoveryImportError(module=module_name, error=str(exc)))
            return None

    def _collect_candidates(
        self,
        module: ModuleType,
        module_name: str,
        candidates: list[_Candidate],
        rejections: list[DiscoveryRejection],
    ) -> None:
        for name, obj in sorted(
            inspect.getmembers(module, inspect.isclass), key=lambda item: item[0]
        ):
            qualname = f"{module_name}.{name}"
            rejection = self._validate_candidate(obj, qualname)
            if rejection is not None:
                if rejection.reason is not DiscoveryRejectReason.UNRELATED:
                    rejections.append(rejection)
                continue

            role = obj.role
            agent_id = role.value
            caps = self._capabilities_for(obj, role)
            candidates.append(
                _Candidate(
                    qualname=qualname,
                    module_name=module_name,
                    agent_cls=obj,
                    agent_id=agent_id,
                    capabilities=caps,
                )
            )

    def _validate_candidate(
        self,
        obj: type[Any],
        qualname: str,
    ) -> DiscoveryRejection | None:
        if not inspect.isclass(obj):
            return DiscoveryRejection(qualname=qualname, reason=DiscoveryRejectReason.NOT_A_CLASS)

        if obj is BaseAgent:
            return DiscoveryRejection(qualname=qualname, reason=DiscoveryRejectReason.IS_BASE_AGENT)

        if not issubclass(obj, BaseAgent):
            return DiscoveryRejection(qualname=qualname, reason=DiscoveryRejectReason.UNRELATED)

        if inspect.isabstract(obj):
            return DiscoveryRejection(qualname=qualname, reason=DiscoveryRejectReason.ABSTRACT)

        role = getattr(obj, "role", None)
        if role is None:
            return DiscoveryRejection(qualname=qualname, reason=DiscoveryRejectReason.MISSING_ROLE)

        if not isinstance(role, AgentRole):
            return DiscoveryRejection(
                qualname=qualname,
                reason=DiscoveryRejectReason.INVALID_ROLE,
                detail=f"expected AgentRole, got {type(role).__name__}",
            )

        return None

    def _capabilities_for(self, agent_cls: type[BaseAgent], role: AgentRole) -> tuple[str, ...]:
        declared = getattr(agent_cls, "capabilities", None)
        if isinstance(declared, (frozenset, set, list, tuple)) and declared:
            return tuple(sorted(str(cap) for cap in declared))
        role_caps = default_capabilities_for_role(role)
        return tuple(sorted(role_caps))

    def _missing_dependencies(self, agent_cls: type[BaseAgent]) -> str | None:
        try:
            signature = inspect.signature(agent_cls.__init__)
        except (TypeError, ValueError):
            return None

        missing: list[str] = []
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param_name not in self._dependencies:
                missing.append(param_name)

        if not missing:
            return None
        return f"missing constructor dependencies: {', '.join(sorted(missing))}"

    def _make_factory(
        self,
        agent_cls: type[BaseAgent],
        agent_id: str,
    ) -> Any:
        dependencies = self._dependencies

        def factory(**deps: Any) -> DiscoveredAgentHandle:
            merged = {**dependencies, **deps}
            kwargs = self._factory_kwargs(agent_cls, merged)
            instance = agent_cls(**kwargs)
            return DiscoveredAgentHandle(agent_id=agent_id, _agent=instance)

        return factory

    @staticmethod
    def _factory_kwargs(agent_cls: type[BaseAgent], merged: Mapping[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(agent_cls.__init__)
        kwargs: dict[str, Any] = {}
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            if param_name in merged:
                kwargs[param_name] = merged[param_name]
            elif param.default is not inspect.Parameter.empty:
                continue
            elif param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
        return kwargs


__all__ = [
    "AgentDiscovery",
    "DiscoveryImportError",
    "DiscoveryRejectReason",
    "DiscoveryRejection",
    "DiscoveryResult",
    "DEFAULT_EXCLUDE_MODULES",
    "DEFAULT_PACKAGE",
    "ROLE_CAPABILITIES",
]
