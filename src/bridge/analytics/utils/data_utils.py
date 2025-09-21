"""데이터 처리 유틸리티 함수들.

데이터 변환, 검증, 분석을 위한 유틸리티 함수들을 제공합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)


class DataUtils:
    """데이터 처리 유틸리티 클래스."""

    @staticmethod
    def validate_data(data: Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]) -> bool:
        """데이터 유효성을 검증합니다.

        Args:
            data: 검증할 데이터

        Returns:
            bool: 유효한 데이터인지 여부
        """
        try:
            if isinstance(data, pa.Table):
                return len(data) >= 0 and len(data.schema) >= 0
            elif isinstance(data, pd.DataFrame):
                return len(data) >= 0 and len(data.columns) >= 0
            elif isinstance(data, list):
                return len(data) >= 0
            else:
                return False
        except Exception as e:
            logger.error(f"데이터 검증 실패: {e}")
            return False

    @staticmethod
    def get_data_info(data: Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """데이터 정보를 반환합니다.

        Args:
            data: 정보를 조회할 데이터

        Returns:
            Dict[str, Any]: 데이터 정보
        """
        info = {
            "type": type(data).__name__,
            "valid": False,
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "memory_usage": 0,
        }

        try:
            if isinstance(data, pa.Table):
                info.update(
                    {
                        "valid": True,
                        "rows": len(data),
                        "columns": len(data.schema),
                        "column_names": data.column_names,
                        "memory_usage": data.nbytes,
                    }
                )
            elif isinstance(data, pd.DataFrame):
                info.update(
                    {
                        "valid": True,
                        "rows": len(data),
                        "columns": len(data.columns),
                        "column_names": data.columns.tolist(),
                        "memory_usage": data.memory_usage(deep=True).sum(),
                    }
                )
            elif isinstance(data, list):
                info.update(
                    {
                        "valid": len(data) > 0,
                        "rows": len(data),
                        "columns": len(data[0].keys()) if data else 0,
                        "column_names": list(data[0].keys()) if data else [],
                        "memory_usage": 0,  # 정확한 메모리 사용량 계산은 복잡함
                    }
                )
        except Exception as e:
            logger.error(f"데이터 정보 조회 실패: {e}")

        return info

    @staticmethod
    def convert_to_arrow(data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pa.Table:
        """데이터를 Arrow Table로 변환합니다.

        Args:
            data: 변환할 데이터

        Returns:
            pa.Table: 변환된 Arrow Table
        """
        try:
            if isinstance(data, pd.DataFrame):
                return pa.Table.from_pandas(data)
            elif isinstance(data, list):
                return pa.Table.from_pylist(data)
            else:
                raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")
        except Exception as e:
            logger.error(f"Arrow 변환 실패: {e}")
            raise

    @staticmethod
    def convert_to_pandas(data: Union[pa.Table, List[Dict[str, Any]]]) -> pd.DataFrame:
        """데이터를 Pandas DataFrame으로 변환합니다.

        Args:
            data: 변환할 데이터

        Returns:
            pd.DataFrame: 변환된 DataFrame
        """
        try:
            if isinstance(data, pa.Table):
                return data.to_pandas()
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")
        except Exception as e:
            logger.error(f"Pandas 변환 실패: {e}")
            raise

    @staticmethod
    def detect_missing_values(data: Union[pa.Table, pd.DataFrame]) -> Dict[str, int]:
        """결측값을 감지합니다.

        Args:
            data: 분석할 데이터

        Returns:
            Dict[str, int]: 컬럼별 결측값 개수
        """
        missing_counts = {}

        try:
            if isinstance(data, pa.Table):
                for i, column_name in enumerate(data.column_names):
                    column = data.column(i)
                    null_count = pa.compute.sum(pa.compute.is_null(column)).as_py()
                    missing_counts[column_name] = null_count
            elif isinstance(data, pd.DataFrame):
                missing_counts = data.isnull().sum().to_dict()
        except Exception as e:
            logger.error(f"결측값 감지 실패: {e}")

        return missing_counts

    @staticmethod
    def get_column_types(data: Union[pa.Table, pd.DataFrame]) -> Dict[str, str]:
        """컬럼 타입을 반환합니다.

        Args:
            data: 분석할 데이터

        Returns:
            Dict[str, str]: 컬럼별 타입 정보
        """
        column_types = {}

        try:
            if isinstance(data, pa.Table):
                for field in data.schema:
                    column_types[field.name] = str(field.type)
            elif isinstance(data, pd.DataFrame):
                column_types = data.dtypes.astype(str).to_dict()
        except Exception as e:
            logger.error(f"컬럼 타입 조회 실패: {e}")

        return column_types

    @staticmethod
    def sample_data(
        data: Union[pa.Table, pd.DataFrame], n: int = 5
    ) -> Union[pa.Table, pd.DataFrame]:
        """데이터 샘플을 반환합니다.

        Args:
            data: 샘플링할 데이터
            n: 샘플 개수

        Returns:
            샘플 데이터
        """
        try:
            if isinstance(data, pa.Table):
                return data.slice(0, min(n, len(data)))
            elif isinstance(data, pd.DataFrame):
                return data.head(n)
            else:
                raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")
        except Exception as e:
            logger.error(f"데이터 샘플링 실패: {e}")
            raise

    @staticmethod
    def compare_schemas(schema1: pa.Schema, schema2: pa.Schema) -> Dict[str, Any]:
        """두 스키마를 비교합니다.

        Args:
            schema1: 첫 번째 스키마
            schema2: 두 번째 스키마

        Returns:
            Dict[str, Any]: 비교 결과
        """
        comparison = {
            "identical": False,
            "common_columns": [],
            "different_columns": [],
            "missing_in_schema1": [],
            "missing_in_schema2": [],
        }

        try:
            names1 = set(schema1.names)
            names2 = set(schema2.names)

            comparison["common_columns"] = list(names1 & names2)
            comparison["missing_in_schema1"] = list(names2 - names1)
            comparison["missing_in_schema2"] = list(names1 - names2)

            # 공통 컬럼의 타입 비교
            for col_name in comparison["common_columns"]:
                field1 = schema1.field(col_name)
                field2 = schema2.field(col_name)

                if field1.type != field2.type:
                    comparison["different_columns"].append(
                        {"column": col_name, "type1": str(field1.type), "type2": str(field2.type)}
                    )

            # 완전히 동일한지 확인
            comparison["identical"] = (
                len(comparison["different_columns"]) == 0
                and len(comparison["missing_in_schema1"]) == 0
                and len(comparison["missing_in_schema2"]) == 0
            )

        except Exception as e:
            logger.error(f"스키마 비교 실패: {e}")

        return comparison
