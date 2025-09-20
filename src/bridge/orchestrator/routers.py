"""오케스트레이션 작업 라우터."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..semantic.models import TaskRequest, TaskResponse
from .tasks import execute_pipeline

router = APIRouter()


@router.post("/plan", response_model=TaskResponse)
async def plan_task(request: TaskRequest) -> TaskResponse:
    """사용자 요청을 간단한 작업 그래프로 계획하고 비동기 실행을 큐에 넣는다."""

    async_steps = [
        {"name": "collect_context", "details": {"sources": request.sources}},
        {"name": "execute_tools", "details": {"tools": request.required_tools}},
    ]

    async_result = execute_pipeline.delay(request.model_dump())

    details: dict[str, Any] = {"job_id": async_result.id}
    if async_result.successful():
        details["result_preview"] = async_result.result

    return TaskResponse(
        intent=request.intent,
        status="planned",
        steps=[
            *async_steps,
            {"name": "queue_execution", "details": details},
        ],
    )
