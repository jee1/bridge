"""오케스트레이터 Celery 태스크."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from bridge.connectors import connector_registry, ConnectorNotFoundError

from .celery_app import celery_app


def _run_maybe_async(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


@celery_app.task(name="bridge.execute_pipeline")
def execute_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """컨텍스트 수집 및 도구 실행을 모사한 태스크."""

    intent = payload.get("intent", "")
    sources: List[str] = payload.get("sources", [])
    tools = payload.get("required_tools", [])

    collected_context: List[Dict[str, Any]] = []
    missing_sources: List[str] = []

    for source_name in sources:
        try:
            connector = connector_registry.get(source_name)
        except ConnectorNotFoundError:
            missing_sources.append(source_name)
            continue

        metadata = _run_maybe_async(connector.get_metadata())
        collected_context.append({
            "source": source_name,
            "metadata": metadata,
        })

    status = "completed" if not missing_sources else "partial"

    return {
        "intent": intent,
        "status": status,
        "collected_sources": collected_context,
        "missing_sources": missing_sources,
        "triggered_tools": tools,
    }
