"""Tiny, dependency-free target for Kodiak's read-only alpha quickstart."""


def greet(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("Kodiak"))
