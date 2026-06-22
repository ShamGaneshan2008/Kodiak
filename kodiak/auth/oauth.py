from dataclasses import dataclass


@dataclass(slots=True)
class OAuthProfile:
    provider: str
    subject: str
    email: str | None = None
