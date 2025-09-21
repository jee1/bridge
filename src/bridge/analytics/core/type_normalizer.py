"""데이터 타입 정규화 시스템.

다양한 데이터 소스에서 오는 데이터 타입을 표준화하여 일관된 처리를 가능하게 합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)


class TypeNormalizer:
    """데이터 타입 자동 변환 시스템.

    다양한 데이터 소스에서 오는 데이터를 표준화된 타입으로 변환합니다.
    """

    def __init__(self):
        """TypeNormalizer를 초기화합니다."""
        self._type_mapping = self._get_type_mapping()

    def _get_type_mapping(self) -> Dict[str, pa.DataType]:
        """타입 매핑 규칙을 정의합니다.

        Returns:
            Dict[str, pa.DataType]: 타입 매핑 딕셔너리
        """
        return {
            # 정수 타입
            "int": pa.int64(),
            "int8": pa.int8(),
            "int16": pa.int16(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "uint8": pa.uint8(),
            "uint16": pa.uint16(),
            "uint32": pa.uint32(),
            "uint64": pa.uint64(),
            # 부동소수점 타입
            "float": pa.float64(),
            "float32": pa.float32(),
            "float64": pa.float64(),
            # 문자열 타입
            "str": pa.string(),
            "string": pa.string(),
            "text": pa.string(),
            "varchar": pa.string(),
            # 불린 타입
            "bool": pa.bool_(),
            "boolean": pa.bool_(),
            # 날짜/시간 타입
            "datetime": pa.timestamp("us"),
            "timestamp": pa.timestamp("us"),
            "date": pa.date32(),
            "time": pa.time64("us"),
            # 기타 타입
            "object": pa.string(),  # Python 객체는 문자열로 변환
            "category": pa.string(),  # 카테고리는 문자열로 변환
        }

    def detect_types(self, data: List[Dict[str, Any]]) -> Dict[str, pa.DataType]:
        """데이터에서 타입을 자동 감지합니다.

        Args:
            data: 타입을 감지할 데이터 (딕셔너리 리스트)

        Returns:
            Dict[str, pa.DataType]: 컬럼별 감지된 타입
        """
        if not data:
            return {}

        detected_types = {}

        for column in data[0].keys():
            column_data = [row.get(column) for row in data if column in row]
            detected_types[column] = self._detect_column_type(column_data)

        return detected_types

    def _detect_column_type(self, column_data: List[Any]) -> pa.DataType:
        """단일 컬럼의 타입을 감지합니다.

        Args:
            column_data: 컬럼 데이터 리스트

        Returns:
            pa.DataType: 감지된 타입
        """
        if not column_data:
            return pa.string()

        # None 값 제거
        non_null_data = [x for x in column_data if x is not None]

        if not non_null_data:
            return pa.string()

        # 타입별 우선순위로 감지
        sample = non_null_data[0]

        # 정수 타입 감지
        if all(isinstance(x, int) for x in non_null_data):
            max_val = max(non_null_data)
            min_val = min(non_null_data)

            if min_val >= 0:  # 양수만 있는 경우
                if max_val < 2**8:
                    return pa.uint8()
                elif max_val < 2**16:
                    return pa.uint16()
                elif max_val < 2**32:
                    return pa.uint32()
                else:
                    return pa.uint64()
            else:  # 음수 포함
                if min_val >= -(2**7) and max_val < 2**7:
                    return pa.int8()
                elif min_val >= -(2**15) and max_val < 2**15:
                    return pa.int16()
                elif min_val >= -(2**31) and max_val < 2**31:
                    return pa.int32()
                else:
                    return pa.int64()

        # 부동소수점 타입 감지
        if all(isinstance(x, (int, float)) for x in non_null_data):
            return pa.float64()

        # 불린 타입 감지
        if all(isinstance(x, bool) for x in non_null_data):
            return pa.bool_()

        # 문자열 타입 (기본값)
        return pa.string()

    def normalize_schema(self, schema: pa.Schema) -> pa.Schema:
        """스키마를 정규화합니다.

        Args:
            schema: 정규화할 스키마

        Returns:
            pa.Schema: 정규화된 스키마
        """
        normalized_fields = []

        for field in schema:
            normalized_type = self._normalize_type(field.type)
            normalized_field = pa.field(field.name, normalized_type)
            normalized_fields.append(normalized_field)

        return pa.schema(normalized_fields)

    def _normalize_type(self, data_type: pa.DataType) -> pa.DataType:
        """단일 타입을 정규화합니다.

        Args:
            data_type: 정규화할 타입

        Returns:
            pa.DataType: 정규화된 타입
        """
        type_str = str(data_type)

        # 타입 매핑에서 찾기
        if type_str in self._type_mapping:
            return self._type_mapping[type_str]

        # 부분 매칭 시도
        for key, normalized_type in self._type_mapping.items():
            if key in type_str.lower():
                return normalized_type

        # 매칭되지 않으면 원본 타입 반환
        return data_type

    def normalize_data(self, table: pa.Table) -> pa.Table:
        """데이터를 정규화합니다.

        Args:
            table: 정규화할 Arrow Table

        Returns:
            pa.Table: 정규화된 Arrow Table
        """
        try:
            # 스키마 정규화
            normalized_schema = self.normalize_schema(table.schema)

            # 데이터 타입 변환
            normalized_arrays = []
            for i, field in enumerate(normalized_schema):
                original_array = table.column(i)

                # 타입이 다르면 변환
                if original_array.type != field.type:
                    try:
                        converted_array = pa.compute.cast(original_array, field.type)
                        normalized_arrays.append(converted_array)
                    except Exception as e:
                        logger.warning(f"컬럼 '{field.name}' 타입 변환 실패: {e}")
                        # 변환 실패 시 원본 유지
                        normalized_arrays.append(original_array)
                else:
                    normalized_arrays.append(original_array)

            return pa.Table.from_arrays(normalized_arrays, schema=normalized_schema)

        except Exception as e:
            logger.error(f"데이터 정규화 실패: {e}")
            # 정규화 실패 시 원본 반환
            return table

    def get_type_mapping(self) -> Dict[str, pa.DataType]:
        """현재 타입 매핑 규칙을 반환합니다.

        Returns:
            Dict[str, pa.DataType]: 타입 매핑 딕셔너리
        """
        return self._type_mapping.copy()

    def add_type_mapping(self, key: str, data_type: pa.DataType) -> None:
        """새로운 타입 매핑 규칙을 추가합니다.

        Args:
            key: 매핑 키
            data_type: 매핑할 Arrow 타입
        """
        self._type_mapping[key] = data_type
        logger.info(f"새로운 타입 매핑 추가: {key} -> {data_type}")

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"TypeNormalizer(mappings={len(self._type_mapping)})"
