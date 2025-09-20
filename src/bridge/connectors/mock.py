"""테스트 및 초기 개발용 Mock 커넥터."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .base import BaseConnector


class MockConnector(BaseConnector):
    """미리 정의된 레코드를 반환하는 단순 커넥터."""

    def __init__(self, name: str = "mock", settings: Dict[str, Any] | None = None, sample_rows: List[Dict[str, Any]] | None = None):
        super().__init__(name=name, settings=settings or {})
        self._sample_rows = sample_rows or [
            {"id": 1, "metric": 42},
            {"id": 2, "metric": 17},
        ]

    def test_connection(self) -> bool:  # type: ignore[override]
        return True

    def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "name": self.name,
            "type": "mock",
            "fields": [
                {"name": key, "type": type(value).__name__}
                for key, value in self._sample_rows[0].items()
            ],
        }

    def run_query(  # type: ignore[override]
        self, query: str, params: Dict[str, Any] | None = None
    ) -> Iterable[Dict[str, Any]]:
        _ = query, params
        yield from self._sample_rows
