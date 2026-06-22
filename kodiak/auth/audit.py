from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class AuditEvent:
    actor: str
    action: str
    created_at: datetime = datetime.utcnow()
