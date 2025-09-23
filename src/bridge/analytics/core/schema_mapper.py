"""통합 데이터 분석 레이어 - 스키마 매핑 모듈.

CA 마일스톤 3.1: 통합 데이터 분석 레이어
스키마 매핑 및 정규화를 담당하는 핵심 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ColumnMapping(BaseModel):
    """컬럼 매핑 정보를 담는 모델."""

    source_column: str = Field(..., description="원본 컬럼명")
    target_column: str = Field(..., description="대상 컬럼명")
    transformation: Optional[str] = Field(None, description="변환 규칙")
    data_type: Optional[str] = Field(None, description="대상 데이터 타입")
    is_required: bool = Field(True, description="필수 컬럼 여부")
    default_value: Optional[Any] = Field(None, description="기본값")


class SchemaMapping(BaseModel):
    """스키마 매핑 정보를 담는 모델."""

    source_name: str = Field(..., description="소스명")
    column_mappings: List[ColumnMapping] = Field(..., description="컬럼 매핑 목록")
    merge_strategy: str = Field("union", description="통합 전략")
    data_quality_rules: Dict[str, Any] = Field(default_factory=dict, description="데이터 품질 규칙")


class SchemaMapper:
    """스키마 매핑 및 정규화 클래스.

    다양한 데이터 소스의 스키마를 통합 스키마로 매핑하고
    데이터 변환 규칙을 적용합니다.
    """

    def __init__(self):
        """SchemaMapper를 초기화합니다."""
        self._mappings: Dict[str, SchemaMapping] = {}
        self._unified_schema: Optional[pa.Schema] = None
        self._column_aliases: Dict[str, str] = {}

    def create_mapping(
        self,
        source_name: str,
        source_schema: pa.Schema,
        target_schema: Optional[pa.Schema] = None,
        mapping_rules: Optional[Dict[str, str]] = None,
        auto_detect: bool = True,
    ) -> SchemaMapping:
        """스키마 매핑을 생성합니다.

        Args:
            source_name: 소스명
            source_schema: 원본 스키마
            target_schema: 대상 스키마 (None이면 자동 생성)
            mapping_rules: 매핑 규칙 {원본컬럼: 대상컬럼}
            auto_detect: 자동 매핑 감지 여부

        Returns:
            SchemaMapping: 생성된 스키마 매핑
        """
        logger.info(f"스키마 매핑 생성 시작: {source_name}")

        # 대상 스키마가 없으면 자동 생성
        if target_schema is None:
            target_schema = self._create_target_schema(source_schema, mapping_rules)

        # 컬럼 매핑 생성
        column_mappings = self._create_column_mappings(
            source_schema, target_schema, mapping_rules or {}, auto_detect
        )

        # 스키마 매핑 생성
        schema_mapping = SchemaMapping(
            source_name=source_name, column_mappings=column_mappings, merge_strategy="union"
        )

        self._mappings[source_name] = schema_mapping
        logger.info(f"스키마 매핑 생성 완료: {source_name} ({len(column_mappings)}개 컬럼)")

        return schema_mapping

    def _create_target_schema(
        self, source_schema: pa.Schema, mapping_rules: Optional[Dict[str, str]] = None
    ) -> pa.Schema:
        """대상 스키마를 생성합니다.

        Args:
            source_schema: 원본 스키마
            mapping_rules: 매핑 규칙

        Returns:
            pa.Schema: 생성된 대상 스키마
        """
        target_fields = []

        for field in source_schema:
            # 매핑 규칙이 있으면 대상 컬럼명 사용
            target_column = (
                mapping_rules.get(field.name, field.name) if mapping_rules else field.name
            )

            # 컬럼명 정규화
            normalized_name = self._normalize_column_name(target_column)

            # 데이터 타입 정규화
            normalized_type = self._normalize_data_type(field.type)

            target_field = pa.field(normalized_name, normalized_type)
            target_fields.append(target_field)

        return pa.schema(target_fields)

    def _create_column_mappings(
        self,
        source_schema: pa.Schema,
        target_schema: pa.Schema,
        mapping_rules: Dict[str, str],
        auto_detect: bool,
    ) -> List[ColumnMapping]:
        """컬럼 매핑을 생성합니다.

        Args:
            source_schema: 원본 스키마
            target_schema: 대상 스키마
            mapping_rules: 매핑 규칙
            auto_detect: 자동 매핑 감지 여부

        Returns:
            List[ColumnMapping]: 컬럼 매핑 목록
        """
        column_mappings = []

        # 명시적 매핑 규칙 적용
        for source_field in source_schema:
            source_column = source_field.name

            if source_column in mapping_rules:
                target_column = mapping_rules[source_column]
                target_field = self._find_field_by_name(target_schema, target_column)

                if target_field:
                    mapping = ColumnMapping(
                        source_column=source_column,
                        target_column=target_column,
                        data_type=str(target_field.type),
                        is_required=True,
                    )
                    column_mappings.append(mapping)
            elif auto_detect:
                # 자동 매핑 감지
                target_column = self._find_best_match(source_column, target_schema)
                if target_column:
                    target_field = self._find_field_by_name(target_schema, target_column)
                    if target_field:
                        mapping = ColumnMapping(
                            source_column=source_column,
                            target_column=target_column,
                            data_type=str(target_field.type),
                            is_required=True,
                        )
                        column_mappings.append(mapping)

        return column_mappings

    def _find_field_by_name(self, schema: pa.Schema, name: str) -> Optional[pa.Field]:
        """스키마에서 이름으로 필드를 찾습니다.

        Args:
            schema: 검색할 스키마
            name: 찾을 필드명

        Returns:
            Optional[pa.Field]: 찾은 필드 또는 None
        """
        for field in schema:
            if field.name == name:
                return field
        return None

    def _find_best_match(self, source_column: str, target_schema: pa.Schema) -> Optional[str]:
        """최적의 매칭 컬럼을 찾습니다.

        Args:
            source_column: 원본 컬럼명
            target_schema: 대상 스키마

        Returns:
            Optional[str]: 매칭된 컬럼명 또는 None
        """
        # 정확한 매칭
        for field in target_schema:
            if field.name.lower() == source_column.lower():
                return field.name

        # 부분 매칭
        source_lower = source_column.lower()
        for field in target_schema:
            if source_lower in field.name.lower() or field.name.lower() in source_lower:
                return field.name

        # 별칭 매칭
        if source_column in self._column_aliases:
            alias = self._column_aliases[source_column]
            for field in target_schema:
                if field.name.lower() == alias.lower():
                    return field.name

        return None

    def _normalize_column_name(self, column_name: str) -> str:
        """컬럼명을 정규화합니다.

        Args:
            column_name: 정규화할 컬럼명

        Returns:
            str: 정규화된 컬럼명
        """
        # 소문자 변환
        normalized = column_name.lower()

        # 특수문자 제거 및 공백을 언더스코어로 변환
        import re

        normalized = re.sub(r"[^a-z0-9_]", "_", normalized)

        # 연속된 언더스코어 제거
        normalized = re.sub(r"_+", "_", normalized)

        # 시작과 끝의 언더스코어 제거
        normalized = normalized.strip("_")

        # 빈 문자열이면 기본값 사용
        if not normalized:
            normalized = "column"

        return normalized

    def _normalize_data_type(self, data_type: pa.DataType) -> pa.DataType:
        """데이터 타입을 정규화합니다.

        Args:
            data_type: 정규화할 데이터 타입

        Returns:
            pa.DataType: 정규화된 데이터 타입
        """
        # 타입별 정규화 규칙
        type_str = str(data_type).lower()

        if "int" in type_str:
            return pa.int64()
        elif "float" in type_str or "double" in type_str:
            return pa.float64()
        elif "bool" in type_str:
            return pa.bool_()
        elif "timestamp" in type_str or "datetime" in type_str:
            return pa.timestamp("us")
        elif "date" in type_str:
            return pa.date32()
        elif "time" in type_str:
            return pa.time64("us")
        else:
            return pa.string()

    def apply_mapping(
        self, source_name: str, data: Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]
    ) -> pa.Table:
        """스키마 매핑을 적용합니다.

        Args:
            source_name: 소스명
            data: 적용할 데이터

        Returns:
            pa.Table: 매핑이 적용된 Arrow Table

        Raises:
            ValueError: 매핑이 존재하지 않거나 적용 실패 시
        """
        if source_name not in self._mappings:
            raise ValueError(f"소스 '{source_name}'에 대한 매핑이 존재하지 않습니다.")

        mapping = self._mappings[source_name]

        # 데이터를 Arrow Table로 변환
        if isinstance(data, pa.Table):
            source_table = data
        elif isinstance(data, pd.DataFrame):
            source_table = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            source_table = pa.Table.from_pylist(data)
        else:
            raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")

        # 매핑 적용
        mapped_arrays = []
        mapped_schema_fields = []

        for column_mapping in mapping.column_mappings:
            source_column = column_mapping.source_column
            target_column = column_mapping.target_column

            if source_column in source_table.column_names:
                # 원본 컬럼에서 데이터 추출
                source_array = source_table.column(source_column)

                # 데이터 타입 변환
                if column_mapping.data_type:
                    try:
                        # Arrow 타입 문자열을 실제 타입으로 변환
                        if column_mapping.data_type == "int64":
                            target_type = pa.int64()
                        elif column_mapping.data_type == "string":
                            target_type = pa.string()
                        elif column_mapping.data_type == "float64":
                            target_type = pa.float64()
                        elif column_mapping.data_type == "bool":
                            target_type = pa.bool_()
                        elif column_mapping.data_type == "timestamp":
                            target_type = pa.timestamp("us")
                        elif column_mapping.data_type == "date":
                            target_type = pa.date32()
                        else:
                            target_type = pa.string()
                    except Exception:
                        target_type = pa.string()
                else:
                    target_type = pa.string()
                if source_array.type != target_type:
                    try:
                        mapped_array = pa.compute.cast(source_array, target_type)
                    except Exception as e:
                        logger.warning(f"컬럼 '{source_column}' 타입 변환 실패: {e}")
                        mapped_array = source_array
                else:
                    mapped_array = source_array

                # 변환 규칙 적용
                if column_mapping.transformation:
                    mapped_array = self._apply_transformation(
                        mapped_array, column_mapping.transformation
                    )

            else:
                # 컬럼이 없으면 기본값 또는 null로 채움
                if column_mapping.default_value is not None:
                    mapped_array = pa.array(
                        [column_mapping.default_value] * len(source_table), type=target_type
                    )
                else:
                    target_type = (
                        pa.DataType.from_string(column_mapping.data_type)
                        if column_mapping.data_type
                        else pa.string()
                    )
                    mapped_array = pa.nulls(len(source_table), target_type)

            mapped_arrays.append(mapped_array)
            mapped_schema_fields.append(pa.field(target_column, mapped_array.type))

        # 매핑된 테이블 생성
        mapped_schema = pa.schema(mapped_schema_fields)
        mapped_table = pa.Table.from_arrays(mapped_arrays, schema=mapped_schema)

        logger.info(
            f"스키마 매핑 적용 완료: {source_name} -> {len(mapped_table)}행, {len(mapped_table.schema)}열"
        )
        return mapped_table

    def _apply_transformation(self, array: pa.Array, transformation: str) -> pa.Array:
        """데이터 변환 규칙을 적용합니다.

        Args:
            array: 변환할 배열
            transformation: 변환 규칙

        Returns:
            pa.Array: 변환된 배열
        """
        try:
            if transformation == "uppercase":
                return pa.compute.utf8_upper(array)
            elif transformation == "lowercase":
                return pa.compute.utf8_lower(array)
            elif transformation == "trim":
                return pa.compute.utf8_trim_whitespace(array)
            elif transformation == "abs":
                return pa.compute.abs(array)
            elif transformation == "round":
                return pa.compute.round(array)
            else:
                logger.warning(f"지원하지 않는 변환 규칙: {transformation}")
                return array
        except Exception as e:
            logger.warning(f"변환 규칙 적용 실패: {transformation}, {e}")
            return array

    def add_column_alias(self, source_name: str, alias: str, target_name: str) -> None:
        """컬럼 별칭을 추가합니다.

        Args:
            source_name: 소스명
            alias: 별칭
            target_name: 대상 이름
        """
        key = f"{source_name}.{alias}"
        self._column_aliases[key] = target_name
        logger.info(f"컬럼 별칭 추가: {key} -> {target_name}")

    def get_mapping(self, source_name: str) -> Optional[SchemaMapping]:
        """스키마 매핑을 조회합니다.

        Args:
            source_name: 소스명

        Returns:
            Optional[SchemaMapping]: 스키마 매핑 또는 None
        """
        return self._mappings.get(source_name)

    def list_mappings(self) -> List[str]:
        """모든 매핑된 소스명을 반환합니다.

        Returns:
            List[str]: 소스명 목록
        """
        return list(self._mappings.keys())

    def validate_mapping(self, source_name: str) -> Dict[str, Any]:
        """스키마 매핑의 유효성을 검증합니다.

        Args:
            source_name: 소스명

        Returns:
            Dict[str, Any]: 검증 결과
        """
        if source_name not in self._mappings:
            return {
                "is_valid": False,
                "error": f"소스 '{source_name}'에 대한 매핑이 존재하지 않습니다.",
            }

        mapping = self._mappings[source_name]
        validation_result = {
            "is_valid": True,
            "source_name": source_name,
            "total_mappings": len(mapping.column_mappings),
            "required_mappings": len([m for m in mapping.column_mappings if m.is_required]),
            "optional_mappings": len([m for m in mapping.column_mappings if not m.is_required]),
            "transformations": len([m for m in mapping.column_mappings if m.transformation]),
        }

        return validation_result

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"SchemaMapper(mappings={len(self._mappings)})"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
