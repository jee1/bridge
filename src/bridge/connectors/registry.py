"""커넥터 레지스트리."""
from __future__ import annotations

from typing import Dict

from .base import BaseConnector


class ConnectorNotFoundError(KeyError):
    """등록되지 않은 커넥터를 요청했을 때 발생."""


class ConnectorRegistry:
    """커넥터 인스턴스를 중앙에서 관리한다."""

    def __init__(self) -> None:
        self._connectors: Dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector, overwrite: bool = False) -> None:
        if not overwrite and connector.name in self._connectors:
            return
        self._connectors[connector.name] = connector

    def get(self, name: str) -> BaseConnector:
        try:
            return self._connectors[name]
        except KeyError as exc:
            raise ConnectorNotFoundError(name) from exc

    def list(self) -> Dict[str, BaseConnector]:
        return dict(self._connectors)


connector_registry = ConnectorRegistry()
