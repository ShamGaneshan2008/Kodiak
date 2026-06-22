from datetime import datetime, timedelta, timezone

from jose import jwt

from kodiak.config.settings import get_settings


def create_access_token(subject: str, expires_minutes: int = 60) -> str:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    return jwt.encode({"sub": subject, "exp": expires_at}, settings.secret_key, algorithm="HS256")
