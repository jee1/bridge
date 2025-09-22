"""오케스트레이터 인증 테스트."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from bridge.orchestrator.app import app


class TestOrchestratorAuth:
    """오케스트레이터 인증 테스트."""

    def test_health_check_no_auth_required(self):
        """헬스 체크는 인증이 필요하지 않음."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_plan_task_requires_auth(self):
        """작업 계획은 인증이 필요함."""
        client = TestClient(app)
        payload = {
            "intent": "describe churn",
            "sources": ["mock"],
            "required_tools": ["sql_executor"],
            "context": {},
        }

        # 인증 없이 요청
        response = client.post("/tasks/plan", json=payload)
        assert response.status_code == 403  # FastAPI Security는 403을 반환
        assert "Not authenticated" in response.json()["detail"]

    def test_get_task_requires_auth(self):
        """작업 조회는 인증이 필요함."""
        client = TestClient(app)

        # 인증 없이 요청
        response = client.get("/tasks/test-job-id")
        assert response.status_code == 403  # FastAPI Security는 403을 반환
        assert "Not authenticated" in response.json()["detail"]

    @patch.dict("os.environ", {"BRIDGE_API_KEY": "test-api-key"})
    def test_plan_task_with_valid_auth(self):
        """유효한 인증으로 작업 계획 테스트."""
        client = TestClient(app)
        payload = {
            "intent": "describe churn",
            "sources": ["mock"],
            "required_tools": ["sql_executor"],
            "context": {},
        }

        headers = {"Authorization": "Bearer test-api-key"}
        response = client.post("/tasks/plan", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "planned"

    @patch.dict("os.environ", {"BRIDGE_API_KEY": "test-api-key"})
    def test_plan_task_with_invalid_auth(self):
        """잘못된 인증으로 작업 계획 테스트."""
        client = TestClient(app)
        payload = {
            "intent": "describe churn",
            "sources": ["mock"],
            "required_tools": ["sql_executor"],
            "context": {},
        }

        headers = {"Authorization": "Bearer wrong-api-key"}
        response = client.post("/tasks/plan", json=payload, headers=headers)
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    @patch.dict("os.environ", {"BRIDGE_API_KEY": "test-api-key"})
    def test_get_task_with_valid_auth(self):
        """유효한 인증으로 작업 조회 테스트."""
        client = TestClient(app)

        headers = {"Authorization": "Bearer test-api-key"}
        response = client.get("/tasks/test-job-id", headers=headers)
        # 작업이 존재하지 않아도 인증은 통과
        assert response.status_code in [200, 404]

    @patch.dict("os.environ", {"BRIDGE_API_KEY": "test-api-key"})
    def test_get_task_with_invalid_auth(self):
        """잘못된 인증으로 작업 조회 테스트."""
        client = TestClient(app)

        headers = {"Authorization": "Bearer wrong-api-key"}
        response = client.get("/tasks/test-job-id", headers=headers)
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_missing_api_key_config(self):
        """API 키 설정이 없는 경우 테스트."""
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(app)
            payload = {
                "intent": "describe churn",
                "sources": ["mock"],
                "required_tools": ["sql_executor"],
                "context": {},
            }

            headers = {"Authorization": "Bearer any-key"}
            response = client.post("/tasks/plan", json=payload, headers=headers)
            assert response.status_code == 500
            assert "API key not configured" in response.json()["detail"]

    def test_missing_authorization_header(self):
        """Authorization 헤더가 없는 경우 테스트."""
        client = TestClient(app)
        payload = {
            "intent": "describe churn",
            "sources": ["mock"],
            "required_tools": ["sql_executor"],
            "context": {},
        }

        response = client.post("/tasks/plan", json=payload)
        assert response.status_code == 403  # FastAPI Security는 403을 반환
        assert "Not authenticated" in response.json()["detail"]
