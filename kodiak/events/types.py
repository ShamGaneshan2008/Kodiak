from enum import StrEnum


class EventType(StrEnum):
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    AGENT_INITIALIZED = "agent.initialized"
    AGENT_THINKING = "agent.thinking"
    AGENT_ACTION = "agent.action"
    AGENT_OBSERVATION = "agent.observation"
    AGENT_ERROR = "agent.error"

    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    SESSION_PAUSED = "session.paused"
    SESSION_RESUMED = "session.resumed"

    MEMORY_UPDATED = "memory.updated"
    MEMORY_CLEARED = "memory.cleared"

    KNOWLEDGE_INDEXED = "knowledge.indexed"
    KNOWLEDGE_QUERIED = "knowledge.queried"

    CODE_GENERATED = "code.generated"
    CODE_TESTED = "code.tested"
    CODE_REVIEWED = "code.reviewed"

    GITHUB_EVENT = "github.event"
    GITHUB_ISSUE_OPENED = "github.issue.opened"
    GITHUB_PR_CREATED = "github.pr.created"
    GITHUB_PR_REVIEWED = "github.pr.reviewed"

    PLUGIN_LOADED = "plugin.loaded"
    PLUGIN_EXECUTED = "plugin.executed"
    PLUGIN_ERROR = "plugin.error"

    MONITORING_ALERT = "monitoring.alert"
    METRICS_RECORDED = "metrics.recorded"

    ERROR_OCCURRED = "error.occurred"
    WARNING_ISSUED = "warning.issued"
