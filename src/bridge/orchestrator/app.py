"""FastAPI 기반 오케스트레이터 서비스 엔트리포인트."""
from __future__ import annotations

from fastapi import FastAPI

from .routers import router as tasks_router

app = FastAPI(title="Bridge Orchestrator")
app.include_router(tasks_router, prefix="/tasks", tags=["tasks"])


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
