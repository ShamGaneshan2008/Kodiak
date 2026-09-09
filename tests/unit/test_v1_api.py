"""API coverage for the repository-local Kodiak v1 service layer."""

from pathlib import Path

from fastapi.testclient import TestClient

from kodiak.api.main import app
from kodiak.orchestration.approval_manager import ApprovalManager


def test_local_task_dry_run_and_lookup_share_service_layer(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    client = TestClient(app)

    response = client.post(
        "/tasks/run",
        json={
            "instruction": "Improve the README quickstart section",
            "path": str(tmp_path),
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_status"] == "dry_run"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    stored = client.get(f"/tasks/{payload['task_id']}", params={"path": str(tmp_path)})
    assert stored.status_code == 200
    assert stored.json()["task_id"] == payload["task_id"]


def test_local_task_lookup_rejects_unsafe_id(tmp_path: Path) -> None:
    response = TestClient(app).get("/tasks/bad$id", params={"path": str(tmp_path)})
    assert response.status_code == 400


def test_approval_and_history_routes_are_repository_scoped(tmp_path: Path) -> None:
    approval = ApprovalManager(tmp_path).create(
        task_id="api-proof", action="verification", reason="API test", risk_level="low"
    )
    client = TestClient(app)

    listed = client.get("/approvals", params={"path": str(tmp_path)})
    assert listed.status_code == 200
    assert listed.json()[0]["approval_id"] == approval["approval_id"]
    resolved = client.post(
        f"/approvals/{approval['approval_id']}/approve", params={"path": str(tmp_path)}
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "approved"
    history = client.get("/memory/history", params={"path": str(tmp_path)})
    assert history.status_code == 200
    assert history.json() == []


def test_history_route_reports_invalid_repository_path(tmp_path: Path) -> None:
    response = TestClient(app).get("/memory/history", params={"path": str(tmp_path / "missing")})
    assert response.status_code == 400
