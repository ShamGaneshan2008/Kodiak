from pathlib import Path

folders = [
    ".github/workflows",
    ".github/ISSUE_TEMPLATE",
    "docker/sandbox",
    "k8s/base",
    "k8s/overlays",
    "alembic/versions",
    "kodiak/config",
    "kodiak/api/middleware",
    "kodiak/api/routers/webhooks",
    "kodiak/api/schemas",
    # Add the rest...
]

files = [
    "pyproject.toml",
    "README.md",
    "kodiak/main.py",
    "kodiak/config/settings.py",
    "kodiak/api/middleware/auth.py",
    # Add the rest...
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

print("Kodiak project structure created successfully!")
