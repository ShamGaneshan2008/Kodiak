import secrets


def generate_api_key(prefix: str = "kod") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"
