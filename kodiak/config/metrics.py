try:
    from prometheus_client import Counter
except Exception:  # pragma: no cover
    Counter = None


REQUESTS_TOTAL = Counter("kodiak_requests_total", "Total API requests") if Counter else None
