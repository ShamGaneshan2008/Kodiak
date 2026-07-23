import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kodiak.llm.base import LLMProvider

from kodiak.db.models import TaskType

from .metrics import MetricsCollector


@dataclass
class EvaluationCase:
    id: str
    name: str
    description: str
    task_type: TaskType
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    case_id: str
    case_name: str
    passed: bool
    output: Any
    execution_time: float
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EvaluationHarness:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.metrics = MetricsCollector()
        self.results: list[EvaluationResult] = []
        self.validators: dict[str, Callable] = {}

    def register_validator(self, task_type: str, validator: Callable):
        self.validators[task_type] = validator

    async def run_case(
        self,
        case: EvaluationCase,
        agent_execute_fn: Callable,
    ) -> EvaluationResult:
        start_time = time.time()
        error = None
        output = None
        passed = False

        try:
            output = await asyncio.wait_for(
                agent_execute_fn(case.input_data),
                timeout=case.timeout_seconds,
            )

            if case.task_type.value in self.validators:
                validator = self.validators[case.task_type.value]
                passed = validator(output, case.expected_output)
            else:
                passed = output is not None

        except TimeoutError:
            error = f"Execution timeout after {case.timeout_seconds}s"
        except Exception as e:
            error = str(e)

        execution_time = time.time() - start_time

        result = EvaluationResult(
            case_id=case.id,
            case_name=case.name,
            passed=passed,
            output=output,
            execution_time=execution_time,
            error=error,
            metrics=self.metrics.get_snapshot(),
        )

        self.results.append(result)
        return result

    async def run_suite(
        self,
        cases: list[EvaluationCase],
        agent_execute_fn: Callable,
    ) -> dict[str, Any]:
        results = []
        for case in cases:
            result = await self.run_case(case, agent_execute_fn)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        avg_execution_time = sum(r.execution_time for r in results) / len(results) if results else 0

        return {
            "total_cases": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": pass_rate,
            "avg_execution_time": avg_execution_time,
            "results": results,
        }

    def get_results(self) -> list[EvaluationResult]:
        return self.results

    def clear_results(self):
        self.results = []
        self.metrics.reset()
