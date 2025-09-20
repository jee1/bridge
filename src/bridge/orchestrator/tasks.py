"""오케스트레이터 Celery 태스크."""
from __future__ import annotations

from typing import Any, Dict

from .celery_app import celery_app


@celery_app.task(name="bridge.execute_pipeline")
def execute_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """컨텍스트 수집 및 도구 실행을 모사한 태스크."""

    intent = payload.get("intent", "")
    sources = payload.get("sources", [])
    tools = payload.get("required_tools", [])

    return {
        "intent": intent,
        "status": "completed",
        "collected_sources": sources,
        "triggered_tools": tools,
    }
