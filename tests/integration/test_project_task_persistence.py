from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kodiak.api.main import app
from kodiak.db.base import Base
from kodiak.db.models.project import Project
from kodiak.db.models.task import Task
from kodiak.db.models.user import User
from kodiak.db.session import get_db


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    """Use a file-backed database so every request receives a new session."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def create_schema() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(
                Base.metadata.create_all,
                tables=[User.__table__, Project.__table__, Task.__table__],
            )

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    asyncio.run(create_schema())
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())


def _authorization_headers(client: TestClient, suffix: str) -> dict[str, str]:
    safe_suffix = suffix.replace("-", "_")
    email = f"persistence-{safe_suffix}@example.com"
    password = "correct-horse-battery-staple"
    response = client.post(
        "/auth/register",
        json={"email": email, "username": f"persistence_{safe_suffix}", "password": password},
    )
    assert response.status_code == 201, response.text
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_create_project_persists_across_requests(client: TestClient, monkeypatch) -> None:
    headers = _authorization_headers(client, "owner")
    monkeypatch.setattr(
        "kodiak.workers.tasks.run_task.run_task_async.delay",
        lambda *_args: type("Result", (), {"id": "queued-task"})(),
    )

    created = client.post(
        "/projects",
        json={"name": "Persistent project", "description": "survives requests"},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    project = created.json()

    listed = client.get("/projects", headers=headers)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [project["id"]]

    fetched = client.get(f"/projects/{project['id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == project["id"]

    task_created = client.post(
        f"/projects/{project['id']}/tasks",
        json={"title": "Persist task", "task_type": "implementation"},
        headers=headers,
    )
    assert task_created.status_code == 201, task_created.text
    task = task_created.json()

    task_listed = client.get(f"/projects/{project['id']}/tasks", headers=headers)
    assert task_listed.status_code == 200
    assert [item["id"] for item in task_listed.json()["items"]] == [task["id"]]

    task_fetched = client.get(f"/projects/{project['id']}/tasks/{task['id']}", headers=headers)
    assert task_fetched.status_code == 200
    assert task_fetched.json()["id"] == task["id"]


def test_project_owner_isolation(client: TestClient) -> None:
    owner_headers = _authorization_headers(client, "project-owner")
    other_headers = _authorization_headers(client, "other-user")
    created = client.post("/projects", json={"name": "Private project"}, headers=owner_headers)
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]

    listed = client.get("/projects", headers=other_headers)
    assert listed.status_code == 200
    assert listed.json()["items"] == []
    assert client.get(f"/projects/{project_id}", headers=other_headers).status_code == 404
    assert client.get(f"/projects/{project_id}/tasks", headers=other_headers).status_code == 404
