from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior software engineer inside the Kodiak autonomous engineering system.
Your job is to implement production-ready code changes.

Rules:
- Write complete, working code — never truncate with "..." or TODOs.
- Follow the conventions and patterns you see in the codebase context exactly.
- Use Python 3.12 type hints throughout.
- Do not add unnecessary comments; only comment what is genuinely non-obvious.
- Never change files that are not listed in the implementation plan.
- Output ONLY valid JSON — no prose, no markdown fences.

Output schema:
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "action": "create | modify | delete",
      "content": "<full file content as a string>"
    }
  ],
  "explanation": "<brief explanation of what was done and why>"
}
"""


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def __init__(self, llm_client: Any, sandbox: Any | None = None) -> None:
        super().__init__()
        self._llm = llm_client
        self._sandbox = sandbox

    async def _run(self, input_: AgentInput) -> AgentOutput:
        subtask: dict = input_.context.get("subtask", {})
        design: dict = input_.context.get("design", {})
        rag_context: str = input_.context.get("rag_context", "")
        work_dir: str = input_.context.get("work_dir", "")
        reviewer_feedback: str = input_.context.get("reviewer_feedback", "")

        user_message = self._build_message(
            instruction=input_.instruction,
            subtask=subtask,
            design=design,
            rag_context=rag_context,
            reviewer_feedback=reviewer_feedback,
        )

        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
            max_tokens=8192,
        )

        raw = response.get("content", "")
        token_usage = response.get("usage", {})

        file_changes = self._parse_changes(raw)
        if file_changes is None:
            return self._make_error(input_, f"Failed to parse code output: {raw[:300]}")

        if work_dir:
            errors = self._write_files(work_dir, file_changes["files"])
            if errors:
                return self._make_error(input_, f"File write errors: {'; '.join(errors)}")

        verification: dict = {}
        if self._sandbox and work_dir:
            verification = await self._verify(work_dir, subtask)

        return self._make_output(
            input_,
            result={
                "files": file_changes["files"],
                "explanation": file_changes.get("explanation", ""),
                "verification": verification,
            },
            token_usage=token_usage,
        )

    def _build_message(
        self,
        instruction: str,
        subtask: dict,
        design: dict,
        rag_context: str,
        reviewer_feedback: str,
    ) -> str:
        parts = [f"## Task\n{instruction}"]
        if subtask:
            parts.append(f"## Current subtask\n{json.dumps(subtask, indent=2)}")
        if design:
            parts.append(f"## Architecture design\n{json.dumps(design, indent=2)}")
        if rag_context:
            parts.append(f"## Relevant codebase\n{rag_context}")
        if reviewer_feedback:
            parts.append(f"## Reviewer feedback to address\n{reviewer_feedback}")
        return "\n\n".join(parts)

    def _parse_changes(self, raw: str) -> dict | None:
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
            if "files" not in data:
                return None
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("coder.parse_failed", error=str(exc))
            return None

    def _write_files(self, work_dir: str, files: list[dict]) -> list[str]:
        errors: list[str] = []
        base = Path(work_dir)
        for file_spec in files:
            path = base / file_spec["path"]
            action = file_spec.get("action", "modify")
            try:
                if action == "delete":
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(file_spec.get("content", ""), encoding="utf-8")
            except OSError as exc:
                errors.append(f"{file_spec['path']}: {exc}")
        return errors

    async def _verify(self, work_dir: str, subtask: dict) -> dict:
        if self._sandbox is None:
            return {}
        try:
            result = await self._sandbox.run(
                work_dir=work_dir,
                command=["python", "-m", "pytest", "--tb=short", "-q"],
                timeout=30,
            )
            return {
                "exit_code": result.get("exit_code", -1),
                "stdout": result.get("stdout", "")[:2000],
                "stderr": result.get("stderr", "")[:1000],
                "passed": result.get("exit_code", -1) == 0,
            }
        except Exception as exc:
            logger.warning("coder.sandbox_verify_failed", error=str(exc))
            return {"error": str(exc), "passed": False}