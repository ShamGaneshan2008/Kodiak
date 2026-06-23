"""
kodiak/orchestration/context_manager.py

Responsible for assembling, pruning, and injecting context into every LLM
call made by an agent.  The ContextManager reads from TaskState and produces
a ContextPacket — a token-budget-aware bundle of information that an agent
can embed directly into its prompt.

Design goals
------------
* Token-budget enforcement: never exceed a caller-specified token ceiling.
* Priority-based inclusion: critical items (objective, current step) are
  always included; lower-priority items (history, reflections, memory) are
  included only if budget remains.
* Stateless computation: the manager holds no mutable state of its own;
  all state lives in TaskState.
* Extensible sections: new context sections can be registered via
  ``register_section`` without changing core logic.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

from pydantic import BaseModel, Field

from kodiak.orchestration.state import (
    AgentRole,
    ExecutionStep,
    ReflectionEntry,
    StepStatus,
    TaskState,
)

logger = logging.getLogger(__name__)

# Constants

# Conservative characters-per-token ratio for budget estimation.
# GPT-4 / Claude family average ~3.8 chars/token; we use 4 for safety.
_CHARS_PER_TOKEN: float = 4.0

# Default token budget for a packed context (leaves room for the response).
DEFAULT_TOKEN_BUDGET: int = 6_000

# How many completed steps to include in the "recent history" section.
DEFAULT_HISTORY_WINDOW: int = 5

# How many reflections to include.
DEFAULT_REFLECTION_WINDOW: int = 3


# Priority enum


class SectionPriority(IntEnum):
    """
    Numeric priority for context sections.

    Higher values are included first when the token budget is tight.
    """

    CRITICAL = 100   # Objective, current step — always included.
    HIGH = 75        # Repository metadata, pending approval.
    MEDIUM = 50      # Recent step history, working memory highlights.
    LOW = 25         # Reflections, full working memory.
    OPTIONAL = 10    # Extra metadata, tags.



# Section descriptor



@dataclass
class ContextSection:
    """
    A named, prioritised block of text that can be included in a context packet.

    Attributes
    ----------
    name:
        Unique identifier used for deduplication and logging.
    priority:
        Determines inclusion order when the token budget is constrained.
    content:
        The rendered text content of this section.
    token_estimate:
        Estimated token count; computed automatically if not provided.
    """

    name: str
    priority: SectionPriority
    content: str
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_estimate = _estimate_tokens(self.content)


# Context packet


class ContextPacket(BaseModel):
    """
    The assembled, token-budget-respecting context ready to be injected into
    an agent prompt.

    Attributes
    ----------
    task_id:
        The task this context belongs to.
    agent_role:
        Which agent role this packet was built for.
    sections:
        Ordered list of (name, content) tuples that were included.
    excluded_sections:
        Names of sections that were dropped due to budget constraints.
    total_tokens_estimated:
        Sum of token estimates for all included sections.
    budget_tokens:
        The token ceiling that was in effect when the packet was built.
    rendered:
        Full concatenated prompt-ready string.
    """

    task_id: str
    agent_role: AgentRole
    sections: list[tuple[str, str]] = Field(default_factory=list)
    excluded_sections: list[str] = Field(default_factory=list)
    total_tokens_estimated: int = 0
    budget_tokens: int = DEFAULT_TOKEN_BUDGET
    rendered: str = ""

    model_config = {"arbitrary_types_allowed": True}


# Section builder type alias

SectionBuilder = Callable[[TaskState, AgentRole], ContextSection | None]


# Utility helpers


def _estimate_tokens(text: str) -> int:
    """Estimate the number of LLM tokens in *text* using a fixed char ratio."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _format_step(step: ExecutionStep, *, include_output: bool = True) -> str:
    """Render a single ExecutionStep as a compact human-readable string."""
    lines = [
        f"[Step {step.index}] {step.description}",
        f"  Role   : {step.agent_role.value}",
        f"  Status : {step.status.value}",
    ]
    if step.error:
        lines.append(f"  Error  : {step.error}")
    if include_output and step.output:
        trimmed = textwrap.shorten(step.output, width=400, placeholder=" …")
        lines.append(f"  Output : {trimmed}")
    return "\n".join(lines)


# Built-in section builders

def _build_objective_section(state: TaskState, _role: AgentRole) -> ContextSection:
    """Always-included section describing the task objective."""
    content = (
        f"## Task Objective\n"
        f"Title : {state.title}\n"
        f"Goal  : {state.objective}\n"
        f"Status: {state.status.value}\n"
        f"Task ID: {state.task_id}"
    )
    return ContextSection(
        name="objective",
        priority=SectionPriority.CRITICAL,
        content=content,
    )


def _build_current_step_section(state: TaskState, _role: AgentRole) -> ContextSection | None:
    """Include the current step if one is active."""
    step = state.current_step
    if step is None:
        return None
    content = f"## Current Step\n{_format_step(step, include_output=False)}"
    return ContextSection(
        name="current_step",
        priority=SectionPriority.CRITICAL,
        content=content,
    )


def _build_repository_section(state: TaskState, _role: AgentRole) -> ContextSection | None:
    """Include repository / PR context when available."""
    parts: list[str] = []
    if state.repository_id:
        parts.append(f"Repository : {state.repository_id}")
    if state.pull_request_id is not None:
        parts.append(f"Pull Request: #{state.pull_request_id}")
    if state.issue_number is not None:
        parts.append(f"Issue      : #{state.issue_number}")
    if not parts:
        return None
    content = "## Repository Context\n" + "\n".join(parts)
    return ContextSection(
        name="repository",
        priority=SectionPriority.HIGH,
        content=content,
    )


def _build_pending_approval_section(state: TaskState, _role: AgentRole) -> ContextSection | None:
    """Highlight any active approval gate so the agent is aware of it."""
    if state.pending_approval is None:
        return None
    ap = state.pending_approval
    content = (
        f"## Pending Approval Gate\n"
        f"Approval ID : {ap.approval_id}\n"
        f"Reason      : {ap.reason}\n"
        f"Requested at: {ap.requested_at.isoformat()}"
    )
    return ContextSection(
        name="pending_approval",
        priority=SectionPriority.HIGH,
        content=content,
    )


def _build_step_history_section(
    state: TaskState,
    _role: AgentRole,
    *,
    window: int = DEFAULT_HISTORY_WINDOW,
) -> ContextSection | None:
    """Include the most recent completed/failed steps as execution history."""
    terminal_steps = [
        s for s in state.steps
        if s.status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
    ]
    if not terminal_steps:
        return None
    recent = terminal_steps[-window:]
    rendered_steps = "\n\n".join(_format_step(s) for s in recent)
    content = f"## Recent Step History (last {len(recent)})\n{rendered_steps}"
    return ContextSection(
        name="step_history",
        priority=SectionPriority.MEDIUM,
        content=content,
    )


def _build_working_memory_section(state: TaskState, _role: AgentRole) -> ContextSection | None:
    """Expose the working memory store relevant to the current agent role."""
    if not state.working_memory:
        return None
    lines = [f"  {k}: {v}" for k, v in state.working_memory.items()]
    content = "## Working Memory\n" + "\n".join(lines)
    return ContextSection(
        name="working_memory",
        priority=SectionPriority.MEDIUM,
        content=content,
    )


def _build_reflections_section(
    state: TaskState,
    role: AgentRole,
    *,
    window: int = DEFAULT_REFLECTION_WINDOW,
) -> ContextSection | None:
    """Include the most recent reflections, optionally filtered by role."""
    relevant: list[ReflectionEntry] = [
        r for r in state.reflections
        if r.agent_role == role or r.agent_role == AgentRole.REFLECTION
    ]
    if not relevant:
        return None
    recent = relevant[-window:]
    parts: list[str] = []
    for r in recent:
        actions = (
            "\n".join(f"    - {a}" for a in r.suggested_actions)
            if r.suggested_actions
            else "    (none)"
        )
        parts.append(
            f"  [{r.agent_role.value}] {r.summary}\n"
            f"  Suggested actions:\n{actions}"
        )
    content = "## Recent Reflections\n" + "\n\n".join(parts)
    return ContextSection(
        name="reflections",
        priority=SectionPriority.LOW,
        content=content,
    )


def _build_tags_section(state: TaskState, _role: AgentRole) -> ContextSection | None:
    """Include task tags when present."""
    if not state.tags:
        return None
    content = "## Tags\n" + ", ".join(state.tags)
    return ContextSection(
        name="tags",
        priority=SectionPriority.OPTIONAL,
        content=content,
    )


def _build_progress_section(state: TaskState, _role: AgentRole) -> ContextSection | None:
    """Include a compact progress summary."""
    total = len(state.steps)
    if total == 0:
        return None
    completed = len(state.completed_steps)
    failed = len(state.failed_steps)
    content = (
        f"## Execution Progress\n"
        f"Steps: {completed}/{total} completed, {failed} failed "
        f"({state.progress_pct:.1f}%)"
    )
    return ContextSection(
        name="progress",
        priority=SectionPriority.OPTIONAL,
        content=content,
    )


# Context manager


class ContextManager:
    """
    Assembles a token-budget-aware ContextPacket from a TaskState.

    The manager maintains an ordered registry of *section builders* — callables
    that accept a ``TaskState`` and ``AgentRole`` and return a
    ``ContextSection`` (or ``None`` if the section is not applicable).

    Sections are sorted by descending priority and greedily included until the
    token budget is exhausted.  The result is a ``ContextPacket`` that any
    agent can render directly into its system/user prompt.

    Usage
    -----
    ::

        manager = ContextManager(token_budget=8_000)
        packet = manager.build(state, AgentRole.CODER)
        prompt = packet.rendered

    Custom sections
    ---------------
    ::

        def my_section(state: TaskState, role: AgentRole) -> ContextSection | None:
            return ContextSection("my_section", SectionPriority.MEDIUM, "…")

        manager.register_section(my_section)
    """

    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        reflection_window: int = DEFAULT_REFLECTION_WINDOW,
    ) -> None:
        """
        Initialise the ContextManager with optional token-budget overrides.

        Parameters
        ----------
        token_budget:
            Maximum number of tokens the assembled context may use.
        history_window:
            Number of recent terminal steps to include in history.
        reflection_window:
            Number of recent reflections to include.
        """
        self._token_budget = token_budget
        self._history_window = history_window
        self._reflection_window = reflection_window

        # Ordered list of (priority, builder) tuples.
        self._builders: list[tuple[int, SectionBuilder]] = []

        self._register_defaults()

    # Registration

    def register_section(
        self,
        builder: SectionBuilder,
        priority_override: SectionPriority | None = None,
    ) -> None:
        """
        Add a custom section builder to the registry.

        Parameters
        ----------
        builder:
            A callable ``(TaskState, AgentRole) -> ContextSection | None``.
        priority_override:
            If provided, this priority is used instead of the section's own.
        """
        priority = int(priority_override) if priority_override is not None else 0
        self._builders.append((priority, builder))
        logger.debug("Registered context section builder: %s", getattr(builder, "__name__", repr(builder)))

    def _register_defaults(self) -> None:
        """Register the built-in section builders in priority order."""
        defaults: list[tuple[SectionPriority, SectionBuilder]] = [
            (SectionPriority.CRITICAL, _build_objective_section),
            (SectionPriority.CRITICAL, _build_current_step_section),
            (SectionPriority.HIGH,     _build_repository_section),
            (SectionPriority.HIGH,     _build_pending_approval_section),
            (SectionPriority.MEDIUM,   self._history_builder),
            (SectionPriority.MEDIUM,   _build_working_memory_section),
            (SectionPriority.LOW,      self._reflection_builder),
            (SectionPriority.OPTIONAL, _build_tags_section),
            (SectionPriority.OPTIONAL, _build_progress_section),
        ]
        for prio, builder in defaults:
            self._builders.append((int(prio), builder))

    # Closure builders (capture window config)

    def _history_builder(self, state: TaskState, role: AgentRole) -> ContextSection | None:
        return _build_step_history_section(state, role, window=self._history_window)

    def _reflection_builder(self, state: TaskState, role: AgentRole) -> ContextSection | None:
        return _build_reflections_section(state, role, window=self._reflection_window)

    # Core assembly

    def build(
        self,
        state: TaskState,
        role: AgentRole,
        *,
        extra_context: dict[str, Any] | None = None,
        token_budget: int | None = None,
    ) -> ContextPacket:
        """
        Assemble and return a ContextPacket for *state* and *role*.

        The method:
        1. Invokes all registered section builders.
        2. Sorts sections by descending priority.
        3. Greedily includes sections until the token budget is exhausted.
        4. Renders the final context string and stores it in the packet.
        5. Updates ``state.context_snapshot`` with a lightweight summary.

        Parameters
        ----------
        state:
            The current TaskState to read from.
        role:
            The AgentRole that will consume this context.
        extra_context:
            Ad-hoc key-value pairs to append as an extra LOW-priority section.
        token_budget:
            Override the instance-level budget for this single call.

        Returns
        -------
        ContextPacket
            Ready-to-use context bundle.
        """
        budget = token_budget if token_budget is not None else self._token_budget

        # 1. Collect candidate sections.
        candidates: list[ContextSection] = []
        for prio, builder in self._builders:
            try:
                section = builder(state, role)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section builder %r raised: %s", builder, exc)
                section = None
            if section is not None:
                # Allow the registration priority to override the section's own.
                if prio > 0:
                    section.priority = SectionPriority(
                        min(prio, max(p.value for p in SectionPriority))
                    )
                candidates.append(section)

        # Inject extra_context as a synthetic section.
        if extra_context:
            lines = "\n".join(f"  {k}: {v}" for k, v in extra_context.items())
            candidates.append(
                ContextSection(
                    name="extra_context",
                    priority=SectionPriority.LOW,
                    content=f"## Extra Context\n{lines}",
                )
            )

        # 2. Sort by descending priority (stable sort preserves registration order
        #    for equal priorities).
        candidates.sort(key=lambda s: s.priority, reverse=True)

        # 3. Greedy token-budget inclusion.
        included: list[ContextSection] = []
        excluded_names: list[str] = []
        tokens_used = 0

        for section in candidates:
            if tokens_used + section.token_estimate <= budget:
                included.append(section)
                tokens_used += section.token_estimate
            else:
                excluded_names.append(section.name)
                logger.debug(
                    "Context section '%s' excluded (budget %d, used %d, needed %d).",
                    section.name,
                    budget,
                    tokens_used,
                    section.token_estimate,
                )

        # 4. Render.
        rendered = "\n\n".join(s.content for s in included)

        # 5. Update the state snapshot (lightweight — avoids storing large blobs).
        state.context_snapshot = {
            "sections_included": [s.name for s in included],
            "sections_excluded": excluded_names,
            "tokens_estimated": tokens_used,
            "budget_tokens": budget,
            "agent_role": role.value,
        }

        packet = ContextPacket(
            task_id=state.task_id,
            agent_role=role,
            sections=[(s.name, s.content) for s in included],
            excluded_sections=excluded_names,
            total_tokens_estimated=tokens_used,
            budget_tokens=budget,
            rendered=rendered,
        )

        logger.debug(
            "Built context for task=%s role=%s sections=%d tokens~%d/%d",
            state.task_id,
            role.value,
            len(included),
            tokens_used,
            budget,
        )

        return packet

    # Convenience helpers

    def render(
        self,
        state: TaskState,
        role: AgentRole,
        *,
        extra_context: dict[str, Any] | None = None,
        token_budget: int | None = None,
    ) -> str:
        """
        Shorthand that returns only the rendered string from ``build``.

        Useful when callers do not need the full ``ContextPacket`` metadata.
        """
        return self.build(
            state,
            role,
            extra_context=extra_context,
            token_budget=token_budget,
        ).rendered

    def estimate_tokens(self, state: TaskState, role: AgentRole) -> int:
        """
        Return the estimated token count for a fully assembled context.

        This builds the packet internally and discards it, so it should only
        be used for planning/quota-checking purposes.
        """
        return self.build(state, role).total_tokens_estimated

    def section_names(self) -> list[str]:
        """Return the names of all registered section builders, in order."""
        return [
            getattr(builder, "__name__", f"builder_{i}")
            for i, (_, builder) in enumerate(self._builders)
        ]