"""커넥터 구현을 위한 공통 추상화."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class BaseConnector(ABC):
    """데이터 소스별 공통 인터페이스."""

    def __init__(self, name: str, settings: Dict[str, Any]):
        self.name = name
        self.settings = settings

    @abstractmethod
    def test_connection(self) -> bool:
        """연결 검사. 실패 시 예외를 던진다."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """스키마, 엔터티 정보 등을 반환한다."""

    @abstractmethod
    def run_query(self, query: str, params: Dict[str, Any] | None = None) -> Iterable[Dict[str, Any]]:
        """원본에 쿼리를 실행하고 결과를 반환한다."""

    def mask_columns(self, rows: Iterable[Dict[str, Any]], masked_fields: Iterable[str]) -> Iterable[Dict[str, Any]]:
        """민감 필드를 마스킹한다."""

        masked = set(masked_fields)
        for row in rows:
            yield {key: ("***" if key in masked else value) for key, value in row.items()}
