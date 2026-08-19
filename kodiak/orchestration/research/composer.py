"""Strategy composition engine.

Combines successful ideas from different strategies to create composite
strategies.  Does NOT assume that combining good ideas automatically
creates a good strategy — the combination must be tested.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from kodiak.orchestration.strategy import EngineeringStrategy, ProblemClass

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CompositionPlan:
    """A plan for composing multiple strategies into one.

    Records which strategies are being combined and the proposed
    composite approach.
    """

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    component_strategy_ids: tuple[str, ...] = ()
    component_names: tuple[str, ...] = ()
    composite_name: str = ""
    composite_approach: str = ""
    rationale: str = ""
    expected_benefit: str = ""
    expected_cost: float = 0.5
    expected_risk: float = 0.5
    assumptions: tuple[str, ...] = ()
    composition_type: str = ""  # e.g. "sequential", "parallel", "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "component_strategy_ids": list(self.component_strategy_ids),
            "component_names": list(self.component_names),
            "composite_name": self.composite_name,
            "composite_approach": self.composite_approach,
            "rationale": self.rationale,
            "expected_benefit": self.expected_benefit,
            "expected_cost": self.expected_cost,
            "expected_risk": self.expected_risk,
            "assumptions": list(self.assumptions),
            "composition_type": self.composition_type,
        }


class StrategyComposer:
    """Combines successful strategies into composite strategies.

    The composer creates composition plans and derived strategies, but
    does NOT automatically accept the composite — it must be validated
    through experimentation.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="strategy_composer")

    def compose(
        self,
        strategies: list[EngineeringStrategy],
        *,
        composite_name: str = "",
        composite_approach: str = "",
        rationale: str = "",
        problem_class: ProblemClass = ProblemClass.UNKNOWN,
    ) -> tuple[CompositionPlan, EngineeringStrategy]:
        """Create a composition plan and derived strategy from components.

        Args:
            strategies: Strategies to combine.  Must have at least 2.
            composite_name: Name for the composite strategy.
            composite_approach: Description of the composite approach.
            rationale: Why combining these strategies makes sense.
            problem_class: Problem class for the composite.

        Returns:
            Tuple of (CompositionPlan, EngineeringStrategy).

        Raises:
            ValueError: If fewer than 2 strategies are provided.
        """
        if len(strategies) < 2:
            raise ValueError("At least 2 strategies are required for composition")

        # Build composite approach from components
        if not composite_approach:
            composite_approach = self._build_composite_approach(strategies)

        if not composite_name:
            composite_name = self._build_composite_name(strategies)

        plan = CompositionPlan(
            component_strategy_ids=tuple(s.strategy_id for s in strategies),
            component_names=tuple(s.name for s in strategies),
            composite_name=composite_name,
            composite_approach=composite_approach,
            rationale=rationale
            or f"Combining {len(strategies)} strategies to leverage their individual strengths.",
            expected_benefit="Combined strengths may outperform individual strategies.",
            expected_cost=max(s.expected_cost for s in strategies),
            expected_risk=self._estimate_composition_risk(strategies),
            assumptions=self._identify_assumptions(strategies),
            composition_type=self._infer_composition_type(strategies),
        )

        # Create the composite strategy
        composite = EngineeringStrategy(
            name=composite_name,
            problem_class=problem_class,
            approach=composite_approach,
            required_capabilities=tuple(
                sorted(
                    {
                        cap
                        for s in strategies
                        for cap in s.required_capabilities
                    }
                )
            ),
            expected_cost=plan.expected_cost,
            expected_risk=plan.expected_risk,
            expected_success_probability=0.5,  # Unknown until tested
            verification_method=self._best_verification_method(strategies),
            tags=tuple(
                sorted({tag for s in strategies for tag in s.tags})
            ),
            provenance=f"composed_from:{','.join(s.name for s in strategies)}",
            metadata={
                "composition_plan_id": plan.plan_id,
                "component_strategy_ids": list(plan.component_strategy_ids),
                "component_names": list(plan.component_names),
                "composition_type": plan.composition_type,
            },
        )

        self._log.info(
            "strategy_composed",
            composite_name=composite_name,
            components=len(strategies),
            composition_type=plan.composition_type,
        )

        return plan, composite

    def suggest_compositions(
        self,
        strategies: list[EngineeringStrategy],
        *,
        problem_class: ProblemClass | None = None,
        min_success_rate: float = 0.6,
    ) -> list[CompositionPlan]:
        """Suggest possible compositions from a set of strategies.

        Only suggests compositions from strategies that meet a minimum
        success rate threshold.
        """
        candidates = [
            s
            for s in strategies
            if s.success_rate >= min_success_rate and not s.is_deprecated
        ]

        if len(candidates) < 2:
            return []

        plans: list[CompositionPlan] = []

        # Suggest pairwise compositions
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                s1, s2 = candidates[i], candidates[j]

                # Only compose strategies that address different aspects
                if s1.problem_class == s2.problem_class and s1.name == s2.name:
                    continue

                plan, _ = self.compose(
                    [s1, s2],
                    problem_class=problem_class or s1.problem_class,
                    rationale=(
                        f"Combining '{s1.name}' (success_rate={s1.success_rate:.2f}) "
                        f"with '{s2.name}' (success_rate={s2.success_rate:.2f})."
                    ),
                )
                plans.append(plan)

        self._log.info(
            "compositions_suggested",
            count=len(plans),
            candidate_strategies=len(candidates),
        )

        return plans

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_composite_approach(strategies: list[EngineeringStrategy]) -> str:
        parts = []
        for i, s in enumerate(strategies, 1):
            parts.append(f"Component {i} ({s.name}):\n{s.approach}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_composite_name(strategies: list[EngineeringStrategy]) -> str:
        names = [s.name for s in strategies]
        return "_plus_".join(names[:3])

    @staticmethod
    def _estimate_composition_risk(strategies: list[EngineeringStrategy]) -> float:
        """Composition risk increases with the number of components."""
        base_risk = max(s.expected_risk for s in strategies)
        component_penalty = len(strategies) * 0.05
        return min(base_risk + component_penalty, 1.0)

    @staticmethod
    def _identify_assumptions(strategies: list[EngineeringStrategy]) -> tuple[str, ...]:
        assumptions: list[str] = []
        for s in strategies:
            assumptions.append(
                f"Strategy '{s.name}' works independently (success_rate={s.success_rate:.2f})."
            )
        assumptions.append("Combining independent strategies produces a coherent approach.")
        assumptions.append("No negative interactions between component strategies.")
        return tuple(assumptions)

    @staticmethod
    def _infer_composition_type(strategies: list[EngineeringStrategy]) -> str:
        """Infer whether composition is sequential, parallel, or hybrid."""
        if len(strategies) <= 2:
            return "sequential"
        return "hybrid"

    @staticmethod
    def _best_verification_method(strategies: list[EngineeringStrategy]) -> str:
        methods = [s.verification_method for s in strategies if s.verification_method]
        return methods[0] if methods else "verification_engine"


__all__ = ["CompositionPlan", "StrategyComposer"]
