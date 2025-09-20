"""오케스트레이션 작업 라우터."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from ..semantic.models import TaskRequest, TaskResponse, TaskStatusResponse
from .queries import get_task_status
from .tasks import execute_pipeline

router = APIRouter()
DISPATCHED_JOB_IDS: set[str] = set()


@router.post("/plan", response_model=TaskResponse)
async def plan_task(request: TaskRequest) -> TaskResponse:
    """사용자 요청을 간단한 작업 그래프로 계획하고 비동기 실행을 큐에 넣는다."""

    async_steps = [
        {"name": "collect_context", "details": {"sources": request.sources}},
        {"name": "execute_tools", "details": {"tools": request.required_tools}},
    ]

    async_result = execute_pipeline.delay(request.model_dump())
    DISPATCHED_JOB_IDS.add(async_result.id)

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


@router.get("/{job_id}", response_model=TaskStatusResponse)
async def get_task(job_id: str, response: Response) -> TaskStatusResponse:
    """작업 상태와 결과를 조회한다."""

    task_status = get_task_status(job_id)
    if task_status["state"] in ("PENDING", "RETRY") and not task_status["ready"]:
        if job_id not in DISPATCHED_JOB_IDS:
            raise HTTPException(status_code=404, detail="Job not found or not started yet")
        response.status_code = status.HTTP_202_ACCEPTED
    else:
        DISPATCHED_JOB_IDS.discard(job_id)

    return TaskStatusResponse(**task_status)
