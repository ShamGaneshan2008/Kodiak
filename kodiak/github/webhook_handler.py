async def handle_github_webhook(event: str, payload: dict) -> dict[str, str]:
    action = payload.get("action", "received")
    return {"status": "accepted", "event": event, "action": action}
