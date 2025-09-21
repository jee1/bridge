"""통합 데이터 처리 레이어.

Arrow Table 기반으로 다양한 데이터 소스를 통합하여 처리합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)


class UnifiedDataFrame:
    """Arrow Table 기반 통합 데이터 처리 클래스.

    다양한 데이터 소스(커넥터, pandas DataFrame, 딕셔너리 리스트 등)를
    Arrow Table로 통합하여 일관된 방식으로 처리할 수 있게 합니다.
    """

    def __init__(self, data: Union[pa.Table, pd.DataFrame, Iterable[Dict[str, Any]], None] = None):
        """UnifiedDataFrame을 초기화합니다.

        Args:
            data: 변환할 데이터. Arrow Table, pandas DataFrame,
                  딕셔너리 리스트 등이 가능합니다.
        """
        self._table: pa.Table = self._convert_to_arrow(data)
        self._metadata: Dict[str, Any] = {}

    def _convert_to_arrow(
        self, data: Union[pa.Table, pd.DataFrame, Iterable[Dict[str, Any]], None]
    ) -> pa.Table:
        """다양한 데이터 형식을 Arrow Table로 변환합니다.

        Args:
            data: 변환할 데이터

        Returns:
            pa.Table: 변환된 Arrow Table

        Raises:
            ValueError: 지원하지 않는 데이터 타입인 경우
        """
        if data is None:
            return pa.table({})

        if isinstance(data, pa.Table):
            return data

        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data)

        if isinstance(data, (list, tuple)):
            if not data:
                return pa.table({})

            # 딕셔너리 리스트인지 확인
            if isinstance(data[0], dict):
                return pa.Table.from_pylist(data)

        # RecordBatchReader 처리
        if hasattr(data, "read_all"):
            return data.read_all()

        # Iterable[Dict[str, Any]] 처리
        try:
            data_list = list(data)
            if data_list and isinstance(data_list[0], dict):
                return pa.Table.from_pylist(data_list)
        except Exception as e:
            logger.warning(f"데이터 변환 중 오류 발생: {e}")

        raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")

    @property
    def table(self) -> pa.Table:
        """내부 Arrow Table을 반환합니다."""
        return self._table

    @property
    def schema(self) -> pa.Schema:
        """스키마 정보를 반환합니다."""
        return self._table.schema

    @property
    def num_rows(self) -> int:
        """행 수를 반환합니다."""
        return len(self._table)

    @property
    def num_columns(self) -> int:
        """열 수를 반환합니다."""
        return len(self._table.schema)

    @property
    def column_names(self) -> List[str]:
        """컬럼 이름 목록을 반환합니다."""
        return self._table.column_names

    def to_arrow(self) -> pa.Table:
        """Arrow Table로 변환합니다."""
        return self._table

    def to_pandas(self) -> pd.DataFrame:
        """Pandas DataFrame으로 변환합니다."""
        return self._table.to_pandas()

    def to_pylist(self) -> List[Dict[str, Any]]:
        """Python 딕셔너리 리스트로 변환합니다."""
        return self._table.to_pylist()

    def get_schema_info(self) -> Dict[str, Any]:
        """스키마 정보를 딕셔너리로 반환합니다."""
        schema_info = {
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "column_names": self.column_names,
            "column_types": {
                name: str(field.type) for name, field in zip(self.schema.names, self.schema)
            },
        }
        return schema_info

    def select_columns(self, columns: List[str]) -> UnifiedDataFrame:
        """지정된 컬럼들만 선택합니다.

        Args:
            columns: 선택할 컬럼 이름 목록

        Returns:
            UnifiedDataFrame: 선택된 컬럼들로 구성된 새로운 UnifiedDataFrame
        """
        try:
            selected_table = self._table.select(columns)
            return UnifiedDataFrame(selected_table)
        except Exception as e:
            logger.error(f"컬럼 선택 실패: {e}")
            raise ValueError(f"컬럼 선택에 실패했습니다: {e}")

    def filter_rows(self, condition: str) -> UnifiedDataFrame:
        """조건에 맞는 행들을 필터링합니다.

        Args:
            condition: 필터링 조건 (SQL WHERE 절과 유사)

        Returns:
            UnifiedDataFrame: 필터링된 결과
        """
        try:
            # 간단한 필터링 조건 처리 (향후 확장 가능)
            # 현재는 기본적인 비교 연산자만 지원
            filtered_table = self._table.filter(
                pa.compute.greater_equal(
                    self._table[condition.split()[0]], int(condition.split()[2])
                )
            )
            return UnifiedDataFrame(filtered_table)
        except Exception as e:
            logger.error(f"행 필터링 실패: {e}")
            raise ValueError(f"행 필터링에 실패했습니다: {e}")

    def add_metadata(self, key: str, value: Any) -> None:
        """메타데이터를 추가합니다.

        Args:
            key: 메타데이터 키
            value: 메타데이터 값
        """
        self._metadata[key] = value

    def get_metadata(self, key: str = None) -> Any:
        """메타데이터를 조회합니다.

        Args:
            key: 조회할 메타데이터 키. None이면 전체 메타데이터 반환

        Returns:
            메타데이터 값 또는 전체 메타데이터 딕셔너리
        """
        if key is None:
            return self._metadata
        return self._metadata.get(key)

    def __len__(self) -> int:
        """len() 함수 지원."""
        return self.num_rows

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"UnifiedDataFrame(rows={self.num_rows}, columns={self.num_columns})"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
