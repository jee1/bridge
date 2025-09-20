from fastapi.testclient import TestClient

from bridge.orchestrator.app import app
from bridge.orchestrator.celery_app import celery_app


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
    assert len(body["steps"]) == 3

    queue_step = body["steps"][-1]
    assert queue_step["name"] == "queue_execution"
    details = queue_step["details"]
    assert details["job_id"]
    if celery_app.conf.task_always_eager:
        assert details["result_preview"]["status"] == "completed"
        assert details["result_preview"]["intent"] == payload["intent"]


def test_celery_is_eager_by_default():
    assert celery_app.conf.task_always_eager is True
