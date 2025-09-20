import importlib
import os

import pytest

from bridge.orchestrator import celery_app as celery_module
from bridge.orchestrator import tasks as tasks_module

REDIS_URL = os.getenv("BRIDGE_TEST_REDIS_URL")

redis_required = pytest.mark.skipif(
    REDIS_URL is None,
    reason="BRIDGE_TEST_REDIS_URL 환경 변수가 설정되지 않아 Redis 통합 테스트를 건너뜁니다.",
)


@redis_required
def test_execute_pipeline_with_redis(monkeypatch):
    # Celery 앱을 Redis 브로커/백엔드와 eager 비활성화 설정으로 재초기화한다.
    monkeypatch.setenv("BRIDGE_CELERY_BROKER_URL", REDIS_URL)
    monkeypatch.setenv("BRIDGE_CELERY_RESULT_BACKEND", REDIS_URL)
    monkeypatch.setenv("BRIDGE_CELERY_TASK_ALWAYS_EAGER", "false")

    importlib.reload(celery_module)
    importlib.reload(tasks_module)

    payload = {
        "intent": "redis integration test",
        "sources": ["mock"],
        "required_tools": ["sql_executor"],
        "context": {},
    }

    async_result = celery_module.celery_app.send_task(
        "bridge.execute_pipeline", args=[payload]
    )
    result = async_result.get(timeout=10)
    assert result["intent"] == payload["intent"]
    assert result["status"] in {"completed", "partial"}

    # 테스트 후 eager 모드로 되돌리기 위해 환경 변수 초기화
    monkeypatch.delenv("BRIDGE_CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("BRIDGE_CELERY_RESULT_BACKEND", raising=False)
    monkeypatch.delenv("BRIDGE_CELERY_TASK_ALWAYS_EAGER", raising=False)
    importlib.reload(celery_module)
    importlib.reload(tasks_module)
