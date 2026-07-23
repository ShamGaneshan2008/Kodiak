from typing import cast

from fastapi import APIRouter, Request

from kodiak.github.webhook_handler import handle_github_webhook  # type: ignore[attr-defined]

router = APIRouter(prefix="/github", tags=["github"])


@router.post("/webhook")
async def github_webhook(request: Request) -> dict[str, str]:
    payload = await request.json()
    event = request.headers.get("x-github-event", "unknown")
    return cast("dict[str, str]", await handle_github_webhook(event, payload))
