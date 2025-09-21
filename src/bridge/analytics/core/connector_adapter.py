"""커넥터 어댑터.

기존 커넥터들을 통합 레이어에 맞게 래핑하여 표준화된 데이터 처리를 가능하게 합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from bridge.connectors.base import BaseConnector

from .data_integration import UnifiedDataFrame
from .type_normalizer import TypeNormalizer

logger = logging.getLogger(__name__)


class ConnectorAdapter:
    """기존 커넥터를 통합 레이어에 맞게 래핑하는 어댑터.

    기존 커넥터의 데이터를 UnifiedDataFrame으로 변환하여
    일관된 방식으로 처리할 수 있게 합니다.
    """

    def __init__(self, connector: BaseConnector, normalize_types: bool = True):
        """ConnectorAdapter를 초기화합니다.

        Args:
            connector: 래핑할 커넥터
            normalize_types: 데이터 타입 정규화 여부
        """
        self.connector = connector
        self.normalize_types = normalize_types
        self.type_normalizer = TypeNormalizer() if normalize_types else None
        self._metadata: Dict[str, Any] = {}

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> UnifiedDataFrame:
        """쿼리를 실행하고 UnifiedDataFrame으로 반환합니다.

        Args:
            query: 실행할 쿼리
            params: 쿼리 파라미터

        Returns:
            UnifiedDataFrame: 쿼리 결과

        Raises:
            Exception: 쿼리 실행 실패 시
        """
        try:
            logger.info(f"커넥터 '{self.connector.name}'에서 쿼리 실행: {query[:100]}...")

            # 쿼리 실행
            result_iter = self.connector.run_query(query, params)

            # 결과를 리스트로 변환
            result_data = list(result_iter)

            if not result_data:
                logger.warning("쿼리 결과가 비어있습니다.")
                return UnifiedDataFrame()

            # UnifiedDataFrame 생성
            unified_df = UnifiedDataFrame(result_data)

            # 타입 정규화 적용
            if self.normalize_types and self.type_normalizer:
                normalized_table = self.type_normalizer.normalize_data(unified_df.table)
                unified_df = UnifiedDataFrame(normalized_table)

            # 메타데이터 추가
            unified_df.add_metadata("connector_name", self.connector.name)
            unified_df.add_metadata("query", query)
            unified_df.add_metadata("params", params or {})

            logger.info(f"쿼리 실행 완료: {unified_df.num_rows}행, {unified_df.num_columns}열")
            return unified_df

        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            raise

    async def get_metadata(self) -> Dict[str, Any]:
        """커넥터의 메타데이터를 조회합니다.

        Returns:
            Dict[str, Any]: 메타데이터 정보
        """
        try:
            metadata = self.connector.get_metadata()

            # 통합 레이어용 메타데이터 추가
            enhanced_metadata = {
                "connector_name": self.connector.name,
                "connector_type": type(self.connector).__name__,
                "supports_normalization": self.normalize_types,
                **metadata,
            }

            self._metadata = enhanced_metadata
            return enhanced_metadata

        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {e}")
            raise

    async def test_connection(self) -> bool:
        """커넥터 연결을 테스트합니다.

        Returns:
            bool: 연결 성공 시 True
        """
        try:
            return self.connector.test_connection()
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False

    def get_connector_info(self) -> Dict[str, Any]:
        """커넥터 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 커넥터 정보
        """
        return {
            "name": self.connector.name,
            "type": type(self.connector).__name__,
            "settings": self.connector.settings,
            "normalize_types": self.normalize_types,
            "metadata": self._metadata,
        }

    def set_type_normalization(self, enabled: bool) -> None:
        """타입 정규화 설정을 변경합니다.

        Args:
            enabled: 정규화 활성화 여부
        """
        self.normalize_types = enabled
        if enabled and not self.type_normalizer:
            self.type_normalizer = TypeNormalizer()
        elif not enabled:
            self.type_normalizer = None

        logger.info(f"타입 정규화 {'활성화' if enabled else '비활성화'}")

    def add_custom_type_mapping(self, key: str, data_type: pa.DataType) -> None:
        """커스텀 타입 매핑을 추가합니다.

        Args:
            key: 매핑 키
            data_type: 매핑할 Arrow 타입
        """
        if self.type_normalizer:
            self.type_normalizer.add_type_mapping(key, data_type)
        else:
            logger.warning("타입 정규화가 비활성화되어 있습니다.")

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"ConnectorAdapter(connector={self.connector.name}, normalize={self.normalize_types})"
        )

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
