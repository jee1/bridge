"""시맨틱 레이어 데이터 모델."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SensitivityLevel(str, Enum):
    """데이터 민감도 레벨."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class SemanticEntity(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="엔터티 이름")
    description: Optional[str] = Field(None, max_length=500, description="엔터티 설명")
    sensitivity: SensitivityLevel = Field(default=SensitivityLevel.INTERNAL, description="데이터 민감도 레벨")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """이름 검증."""
        if not v.strip():
            raise ValueError('이름은 공백일 수 없습니다')
        return v.strip()


class TaskRequest(BaseModel):
    intent: str = Field(..., min_length=1, max_length=1000, description="사용자 의도")
    sources: List[str] = Field(default_factory=list, max_length=10, description="데이터 소스 목록")
    required_tools: List[str] = Field(default_factory=list, max_length=10, description="필요한 도구 목록")
    context: Dict[str, Any] = Field(default_factory=dict, max_length=50, description="추가 컨텍스트")
    
    @field_validator('intent')
    @classmethod
    def validate_intent(cls, v):
        """의도 검증."""
        if not v.strip():
            raise ValueError('의도는 공백일 수 없습니다')
        return v.strip()
    
    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        """소스 검증."""
        if v:
            for source in v:
                if not source.strip():
                    raise ValueError('소스 이름은 공백일 수 없습니다')
                if len(source) > 100:
                    raise ValueError('소스 이름은 100자를 초과할 수 없습니다')
        return v
    
    @field_validator('required_tools')
    @classmethod
    def validate_tools(cls, v):
        """도구 검증."""
        if v:
            for tool in v:
                if not tool.strip():
                    raise ValueError('도구 이름은 공백일 수 없습니다')
                if len(tool) > 100:
                    raise ValueError('도구 이름은 100자를 초과할 수 없습니다')
        return v


class TaskStep(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="단계 이름")
    details: Dict[str, Any] = Field(default_factory=dict, max_length=20, description="단계 세부사항")


class TaskStatus(str, Enum):
    """작업 상태."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResponse(BaseModel):
    intent: str = Field(..., min_length=1, max_length=1000, description="사용자 의도")
    status: TaskStatus = Field(..., description="작업 상태")
    steps: List[TaskStep] = Field(..., min_length=1, max_length=20, description="작업 단계")


class TaskState(str, Enum):
    """Celery 작업 상태."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskStatusResponse(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=100, description="작업 ID")
    state: TaskState = Field(..., description="Celery 작업 상태")
    ready: bool = Field(..., description="작업 완료 여부")
    successful: bool = Field(..., description="작업 성공 여부")
    result: Dict[str, Any] | None = Field(None, max_length=100, description="작업 결과")
    error: str | None = Field(None, max_length=1000, description="에러 메시지")
