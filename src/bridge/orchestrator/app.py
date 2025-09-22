"""FastAPI 기반 오케스트레이터 서비스 엔트리포인트."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..logging_config import get_logger, setup_logging
from .routers import router as tasks_router

# 로깅 설정
setup_logging()
logger = get_logger("orchestrator")

app = FastAPI(
    title="Bridge Orchestrator",
    description="Model Context Protocol 기반 데이터 통합 및 AI 오케스트레이션 시스템",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(tasks_router, prefix="/tasks", tags=["tasks"])


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 이벤트."""
    logger.info("Bridge Orchestrator 시작됨")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 이벤트."""
    logger.info("Bridge Orchestrator 종료됨")


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """헬스 체크 엔드포인트."""
    logger.debug("헬스 체크 요청")
    return {"status": "ok"}
