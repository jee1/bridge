"""Celery 결과 조회 유틸리티."""
from __future__ import annotations

from typing import Any, Dict

from celery.result import AsyncResult

from .celery_app import celery_app


def get_task_status(job_id: str) -> Dict[str, Any]:
    """작업 상태와 결과를 조회한다."""

    result = AsyncResult(job_id, app=celery_app)
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else False,
    }

    if result.failed():
        payload["error"] = str(result.result)
    elif result.ready():
        payload["result"] = result.result

    return payload
