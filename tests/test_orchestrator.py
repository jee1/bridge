from fastapi.testclient import TestClient

from bridge.orchestrator.app import app


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_plan_task_returns_steps():
    client = TestClient(app)
    payload = {
        "intent": "describe churn",
        "sources": ["postgresql"],
        "required_tools": ["sql_executor"],
        "context": {},
    }
    response = client.post("/tasks/plan", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "planned"
    assert len(body["steps"]) == 2
