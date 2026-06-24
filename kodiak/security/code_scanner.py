import ast
import re
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SecurityFinding(BaseModel):
    severity: str
    rule_id: str
    message: str
    line_number: int | None = None


class ScanResult(BaseModel):
    safe: bool
    findings: list[SecurityFinding] = Field(default_factory=list)


class CodeScanner:
    def __init__(self) -> None:
        self._python_rules: dict[str, tuple[type[ast.AST], str]] = {
            "SUBPROCESS_CALL": (ast.Call, "subprocess"),
            "OS_SYSTEM": (ast.Call, "os.system"),
            "EVAL_USAGE": (ast.Call, "eval"),
            "EXEC_USAGE": (ast.Call, "exec"),
            "DANGEROUS_IMPORTS": (
                ast.Import | ast.ImportFrom,
                "os, subprocess, shutil, socket, requests",
            ),
        }
        self._shell_patterns: dict[str, re.Pattern[str]] = {
            "RM_RF": re.compile(r"\brm\s+-rf\b"),
            "DD_BURN": re.compile(r"\bdd\s+if="),
            "MKFS": re.compile(r"\bmkfs\.\w+"),
            "CHMOD777": re.compile(r"\bchmod\s+777\b"),
            "CURL_PIPE_BASH": re.compile(r"\bcurl\b.*\|\s*bash\b"),
            "WGET_PIPE_SH": re.compile(r"\bwget\b.*\|\s*sh\b"),
        }

    async def scan_python(self, code: str) -> ScanResult:
        findings: list[SecurityFinding] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ScanResult(
                safe=False,
                findings=[
                    SecurityFinding(
                        severity="high", rule_id="SYNTAX_ERROR", message=str(e)
                    )
                ],
            )

        for node in ast.walk(tree):
            for rule_id, rule_def in self._python_rules.items():
                node_type, targets = rule_def
                if isinstance(node, node_type):
                    if self._matches_node(node, targets, rule_id):
                        findings.append(
                            SecurityFinding(
                                severity="high",
                                rule_id=rule_id,
                                message=f"Detected {rule_id}",
                                line_number=getattr(node, "lineno", None),
                            )
                        )
        if findings:
            logger.warning("python_security_findings", count=len(findings))
        return ScanResult(safe=len(findings) == 0, findings=findings)

    def _matches_node(
        self, node: ast.AST, targets: str, rule_id: str
    ) -> bool:
        target_list = [t.strip() for t in targets.split(",")]
        if rule_id == "DANGEROUS_IMPORTS":
            if isinstance(node, ast.Import):
                return any(alias.name in target_list for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                return node.module in target_list
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                full_name = f"{func.value.id}.{func.attr}"
                return full_name in target_list or func.attr in target_list
            if isinstance(func, ast.Name):
                return func.id in target_list
        return False

    async def scan_shell(self, command: str) -> ScanResult:
        findings: list[SecurityFinding] = []
        for rule_id, pattern in self._shell_patterns.items():
            if pattern.search(command):
                findings.append(
                    SecurityFinding(
                        severity="critical",
                        rule_id=rule_id,
                        message=f"Detected dangerous shell pattern: {rule_id}",
                    )
                )
        if findings:
            logger.warning("shell_security_findings", count=len(findings))
        return ScanResult(safe=len(findings) == 0, findings=findings)

    async def scan_file(self, file_path: Path) -> ScanResult:
        if not file_path.exists():
            return ScanResult(
                safe=False,
                findings=[
                    SecurityFinding(
                        severity="high",
                        rule_id="FILE_NOT_FOUND",
                        message=f"File not found: {file_path}",
                    )
                ],
            )
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if file_path.suffix == ".py":
            return await self.scan_python(content)
        return await self.scan_shell(content)

    async def scan_directory(self, dir_path: Path) -> dict[str, ScanResult]:
        if not dir_path.is_dir():
            return {}
        results: dict[str, ScanResult] = {}
        for path in dir_path.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".bash"}:
                results[str(path)] = await self.scan_file(path)
        return results