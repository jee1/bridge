"""시맨틱 레이어 데이터 모델."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SemanticEntity(BaseModel):
    name: str
    description: Optional[str] = None
    sensitivity: str = Field(default="internal", description="데이터 민감도 레벨")


class TaskRequest(BaseModel):
    intent: str
    sources: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


class TaskStep(BaseModel):
    name: str
    details: Dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    intent: str
    status: str
    steps: List[TaskStep]
