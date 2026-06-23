import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from .harness import EvaluationResult


class EvaluationReporter:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./evaluation_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports: List[Dict[str, Any]] = []

    def generate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        if not results:
            return {}

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0

        execution_times = [r.execution_time for r in results]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 2),
            "avg_execution_time": round(avg_time, 3),
            "min_execution_time": round(min(execution_times), 3) if execution_times else 0,
            "max_execution_time": round(max(execution_times), 3) if execution_times else 0,
        }

    def generate_json_report(
        self,
        results: List[EvaluationResult],
        filename: Optional[str] = None,
    ) -> str:
        if not filename:
            filename = f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        summary = self.generate_summary(results)
        report_data = {
            "summary": summary,
            "results": [
                {
                    "case_id": r.case_id,
                    "case_name": r.case_name,
                    "passed": r.passed,
                    "execution_time": round(r.execution_time, 3),
                    "error": r.error,
                    "metrics": r.metrics,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results
            ],
        }

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2)

        self.reports.append(report_data)
        return str(filepath)

    def generate_markdown_report(
        self,
        results: List[EvaluationResult],
        filename: Optional[str] = None,
    ) -> str:
        if not filename:
            filename = f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"

        summary = self.generate_summary(results)

        lines = [
            "# Evaluation Report",
            "",
            f"**Generated:** {summary['timestamp']}",
            "",
            "## Summary",
            "",
            f"- **Total Cases:** {summary['total_cases']}",
            f"- **Passed:** {summary['passed']}",
            f"- **Failed:** {summary['failed']}",
            f"- **Pass Rate:** {summary['pass_rate']}%",
            f"- **Avg Execution Time:** {summary['avg_execution_time']}s",
            "",
            "## Results",
            "",
            "| Case | Status | Time (s) | Error |",
            "|------|--------|----------|-------|",
        ]

        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            error_text = r.error[:50] + "..." if r.error and len(r.error) > 50 else r.error or "-"
            lines.append(
                f"| {r.case_name} | {status} | {r.execution_time:.3f} | {error_text} |"
            )

        lines.extend(["", "## Detailed Results", ""])

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.extend([
                f"### {r.case_name} [{status}]",
                "",
                f"- **Execution Time:** {r.execution_time:.3f}s",
            ])
            if r.error:
                lines.append(f"- **Error:** {r.error}")
            if r.metrics:
                lines.append("- **Metrics:**")
                for metric_name, metric_value in r.metrics.items():
                    lines.append(f"  - {metric_name}: {metric_value}")
            lines.append("")

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return str(filepath)

    def generate_csv_report(
        self,
        results: List[EvaluationResult],
        filename: Optional[str] = None,
    ) -> str:
        if not filename:
            filename = f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write("case_id,case_name,passed,execution_time,error\n")
            for r in results:
                error_escaped = f'"{r.error}"' if r.error else ""
                f.write(f"{r.case_id},{r.case_name},{r.passed},{r.execution_time:.3f},{error_escaped}\n")

        return str(filepath)

    def compare_reports(
        self,
        results1: List[EvaluationResult],
        results2: List[EvaluationResult],
    ) -> Dict[str, Any]:
        summary1 = self.generate_summary(results1)
        summary2 = self.generate_summary(results2)

        return {
            "baseline": summary1,
            "current": summary2,
            "differences": {
                "pass_rate_delta": round(summary2["pass_rate"] - summary1["pass_rate"], 2),
                "avg_time_delta": round(summary2["avg_execution_time"] - summary1["avg_execution_time"], 3),
            },
        }

    def get_reports(self) -> List[Dict[str, Any]]:
        return self.reports