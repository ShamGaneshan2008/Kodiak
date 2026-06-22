try:
    from celery import Celery
except Exception:  # pragma: no cover
    Celery = None

from kodiak.config.settings import get_settings

settings = get_settings()
celery_app = (
    Celery("kodiak", broker=settings.redis_url, backend=settings.redis_url) if Celery else None
)
