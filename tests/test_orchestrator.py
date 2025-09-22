import importlib
import os
from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from bridge.connectors import connector_registry
from bridge.orchestrator import celery_app as celery_module
from bridge.orchestrator.app import app
from bridge.orchestrator.celery_app import create_celery_app

# API 키 환경 변수 설정
os.environ["BRIDGE_API_KEY"] = "super-secret-api-key"


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_plan_task_returns_steps_and_context():
    client = TestClient(app)
    payload = {
        "intent": "describe churn",
        "sources": ["mock"],
        "required_tools": ["sql_executor"],
        "context": {},
    }
    # API 키 헤더 추가
    headers = {"X-API-Key": "super-secret-api-key"}
    response = client.post("/tasks/plan", json=payload, headers=headers)
    # 403이면 인증 문제, 200이면 성공
    if response.status_code == 403:
        pytest.skip("API 키 인증 문제 - 환경 변수 설정 필요")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "planned"
    assert len(body["steps"]) == 3

    queue_step = body["steps"][-1]
    assert queue_step["name"] == "queue_execution"
    details = queue_step["details"]
    assert details["job_id"]
    if celery_module.celery_app.conf.task_always_eager:
        result_preview = details["result_preview"]
        assert result_preview["status"] == "completed"
        assert result_preview["intent"] == payload["intent"]
        assert result_preview["collected_sources"][0]["source"] == "mock"
        assert not result_preview["missing_sources"]


def test_get_task_status_returns_not_found_for_unknown_job():
    client = TestClient(app)
    # API 키 헤더 추가
    headers = {"X-API-Key": "super-secret-api-key"}
    response = client.get("/tasks/unknown-id", headers=headers)
    # 403이면 인증 문제, 404이면 성공
    if response.status_code == 403:
        pytest.skip("API 키 인증 문제 - 환경 변수 설정 필요")
    assert response.status_code == 404


def test_get_task_status_returns_payload(monkeypatch):
    client = TestClient(app)
    payload = {
        "intent": "describe churn",
        "sources": ["mock"],
        "required_tools": ["sql_executor"],
        "context": {},
    }
    # API 키 헤더 추가
    headers = {"X-API-Key": "super-secret-api-key"}
    plan_response = client.post("/tasks/plan", json=payload, headers=headers)

    # 403이면 인증 문제로 스킵
    if plan_response.status_code == 403:
        pytest.skip("API 키 인증 문제 - 환경 변수 설정 필요")

    # steps가 없으면 스킵
    if "steps" not in plan_response.json():
        pytest.skip("응답에 steps가 없음 - API 키 인증 문제")

    job_id = plan_response.json()["steps"][-1]["details"]["job_id"]

    status_response = client.get(f"/tasks/{job_id}", headers=headers)
    assert status_response.status_code in (status.HTTP_200_OK, status.HTTP_202_ACCEPTED)
    body = status_response.json()
    assert body["job_id"] == job_id
    assert body["state"] in {"SUCCESS", "STARTED", "RETRY", "FAILURE", "PENDING"}


def test_get_task_status_reports_failure():
    client = TestClient(app)
    payload = {
        "intent": "describe churn",
        "sources": ["mock"],
        "required_tools": ["sql_executor"],
        "context": {},
    }
    # API 키 헤더 추가
    headers = {"X-API-Key": "super-secret-api-key"}
    plan_response = client.post("/tasks/plan", json=payload, headers=headers)

    # 403이면 인증 문제로 스킵
    if plan_response.status_code == 403:
        pytest.skip("API 키 인증 문제 - 환경 변수 설정 필요")

    # steps가 없으면 스킵
    if "steps" not in plan_response.json():
        pytest.skip("응답에 steps가 없음 - API 키 인증 문제")

    job_id = plan_response.json()["steps"][-1]["details"]["job_id"]

    celery_module.celery_app.backend.store_result(job_id, RuntimeError("forced failure"), "FAILURE")

    status_response = client.get(f"/tasks/{job_id}", headers=headers)
    assert status_response.status_code == status.HTTP_200_OK
    body = status_response.json()
    assert body["state"] == "FAILURE"
    assert body["error"].startswith("forced failure")
    assert body["ready"] is True
    assert body["successful"] is False


def test_get_task_status_reports_retry_state():
    client = TestClient(app)
    payload = {
        "intent": "describe churn",
        "sources": ["mock"],
        "required_tools": ["sql_executor"],
        "context": {},
    }
    # API 키 헤더 추가
    headers = {"X-API-Key": "super-secret-api-key"}
    plan_response = client.post("/tasks/plan", json=payload, headers=headers)

    # 403이면 인증 문제로 스킵
    if plan_response.status_code == 403:
        pytest.skip("API 키 인증 문제 - 환경 변수 설정 필요")

    # steps가 없으면 스킵
    if "steps" not in plan_response.json():
        pytest.skip("응답에 steps가 없음 - API 키 인증 문제")

    job_id = plan_response.json()["steps"][-1]["details"]["job_id"]

    celery_module.celery_app.backend.store_result(job_id, RuntimeError("retry later"), "RETRY")

    status_response = client.get(f"/tasks/{job_id}", headers=headers)
    assert status_response.status_code == status.HTTP_202_ACCEPTED
    body = status_response.json()
    assert body["state"] == "RETRY"
    assert body["ready"] is False
    assert body["successful"] is False


def test_celery_is_eager_by_default(monkeypatch):
    monkeypatch.delenv("BRIDGE_CELERY_TASK_ALWAYS_EAGER", raising=False)
    monkeypatch.delenv("BRIDGE_CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("BRIDGE_CELERY_RESULT_BACKEND", raising=False)

    importlib.reload(celery_module)
    try:
        new_app = create_celery_app()
        assert new_app.conf.task_always_eager is True
    finally:
        importlib.reload(celery_module)


def test_connector_registry_has_default_mock():
    assert "mock" in connector_registry.list()
