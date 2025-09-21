"""커넥터 구현을 위한 공통 추상화."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class BaseConnector(ABC):
    """데이터 소스별 공통 인터페이스.

    모든 데이터 커넥터는 이 클래스를 상속받아 구현해야 합니다.
    각 커넥터는 연결 테스트, 메타데이터 조회, 쿼리 실행 기능을 제공해야 합니다.

    Attributes:
        name: 커넥터의 고유 이름
        settings: 커넥터 설정 정보
    """

    def __init__(self, name: str, settings: Dict[str, Any]):
        """커넥터를 초기화합니다.

        Args:
            name: 커넥터의 고유 이름
            settings: 커넥터 설정 정보 (데이터베이스 연결 정보 등)
        """
        self.name = name
        self.settings = settings

    @abstractmethod
    def test_connection(self) -> bool:
        """데이터 소스와의 연결을 테스트합니다.

        Returns:
            bool: 연결 성공 시 True, 실패 시 예외 발생

        Raises:
            ConnectionError: 연결 실패 시
        """

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """데이터 소스의 메타데이터를 조회합니다.

        Returns:
            Dict[str, Any]: 스키마, 테이블, 컬럼 정보 등

        Raises:
            MetadataError: 메타데이터 조회 실패 시
        """

    @abstractmethod
    def run_query(
        self, query: str, params: Dict[str, Any] | None = None
    ) -> Iterable[Dict[str, Any]]:
        """데이터 소스에서 쿼리를 실행합니다.

        Args:
            query: 실행할 쿼리 문자열
            params: 쿼리 파라미터 (SQL 인젝션 방지를 위해 사용)

        Yields:
            Dict[str, Any]: 쿼리 결과 행

        Raises:
            QueryExecutionError: 쿼리 실행 실패 시
        """

    def mask_columns(
        self, rows: Iterable[Dict[str, Any]], masked_fields: Iterable[str]
    ) -> Iterable[Dict[str, Any]]:
        """민감한 필드를 마스킹합니다.

        Args:
            rows: 마스킹할 데이터 행들
            masked_fields: 마스킹할 필드 이름들

        Yields:
            Dict[str, Any]: 마스킹된 데이터 행
        """
        masked = set(masked_fields)
        for row in rows:
            yield {key: ("***" if key in masked else value) for key, value in row.items()}
