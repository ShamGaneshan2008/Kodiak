from typing import Any, Callable, Coroutine

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

ToolExecutor = Callable[..., Coroutine[Any, Any, Any]]


class ToolDefinition(BaseModel):
    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    executor: ToolExecutor | None = None


class ToolRouter:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        if not tool.executor:
            raise ValueError(f"Tool {tool.name} must have an executor")
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name)

    def route(
        self, action: str, required_capability: str | None = None
    ) -> ToolDefinition | None:
        for tool in self._tools.values():
            if tool.name == action:
                if (
                    required_capability
                    and required_capability not in tool.capabilities
                ):
                    logger.warning(
                        "tool_missing_capability",
                        tool=tool.name,
                        cap=required_capability,
                    )
                    return None
                return tool

            if required_capability and required_capability in tool.capabilities:
                return tool

        logger.warning(
            "no_tool_found", action=action, capability=required_capability
        )
        return None

    def validate(self, tool_name: str, params: dict[str, Any]) -> bool:
        tool = self._tools.get(tool_name)
        if not tool:
            logger.error("validation_failed_tool_not_found", tool=tool_name)
            return False
        if not tool.executor:
            logger.error("validation_failed_no_executor", tool=tool_name)
            return False
        return True

    async def execute(self, tool_name: str, params: dict[str, Any]) -> Any:
        if not self.validate(tool_name, params):
            raise RuntimeError(f"Tool {tool_name} validation failed")

        tool = self._tools[tool_name]
        logger.info("executing_tool", name=tool_name)

        try:
            return await tool.executor(**params)  # type: ignore[misc]
        except Exception:
            logger.exception("tool_execution_failed", name=tool_name)
            raise