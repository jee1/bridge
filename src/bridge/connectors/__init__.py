"""커넥터 패키지 초기화."""
from __future__ import annotations

from .base import BaseConnector
from .mock import MockConnector
from .registry import ConnectorNotFoundError, connector_registry

# 기본 Mock 커넥터 등록 (테스트 및 초기 개발용)
connector_registry.register(MockConnector())

__all__ = [
    "BaseConnector",
    "MockConnector",
    "ConnectorNotFoundError",
    "connector_registry",
]
