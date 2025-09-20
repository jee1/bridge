"""오케스트레이션 작업 라우터."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..semantic.models import TaskRequest, TaskResponse

router = APIRouter()


@router.post("/plan", response_model=TaskResponse)
async def plan_task(request: TaskRequest) -> TaskResponse:
    """사용자 요청을 간단한 작업 그래프로 계획한다."""

    # 실제 구현에서는 MCP 구성, 커넥터 선택, Celery 태스크 등록 등을 수행한다.
    return TaskResponse(
        intent=request.intent,
        status="planned",
        steps=[
            {"name": "collect_context", "details": {"sources": request.sources}},
            {"name": "execute_tools", "details": {"tools": request.required_tools}},
        ],
    )
