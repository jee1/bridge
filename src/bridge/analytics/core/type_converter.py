"""통합 데이터 분석 레이어 - 고급 데이터 타입 변환 모듈.

CA 마일스톤 3.1: 통합 데이터 분석 레이어
고급 데이터 타입 변환을 담당하는 핵심 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversionRule(BaseModel):
    """데이터 타입 변환 규칙을 담는 모델."""
    
    source_type: str = Field(..., description="원본 데이터 타입")
    target_type: str = Field(..., description="대상 데이터 타입")
    conversion_function: str = Field(..., description="변환 함수명")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="변환 파라미터")
    error_handling: str = Field("skip", description="에러 처리 방식 (skip, null, raise)")


class TypeConverter:
    """고급 데이터 타입 변환 클래스.
    
    복잡한 데이터 타입 간 변환을 지원하고
    사용자 정의 변환 규칙을 적용할 수 있습니다.
    """

    def __init__(self):
        """TypeConverter를 초기화합니다."""
        self._conversion_rules: Dict[str, ConversionRule] = {}
        self._custom_functions: Dict[str, callable] = {}
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """기본 변환 규칙을 초기화합니다."""
        default_rules = [
            # 정수 타입 변환
            ConversionRule(source_type="int8", target_type="int64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="int16", target_type="int64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="int32", target_type="int64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="uint8", target_type="int64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="uint16", target_type="int64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="uint32", target_type="int64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="uint64", target_type="int64", conversion_function="cast", parameters={}),
            
            # 부동소수점 타입 변환
            ConversionRule(source_type="float32", target_type="float64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="int64", target_type="float64", conversion_function="cast", parameters={}),
            ConversionRule(source_type="int32", target_type="float64", conversion_function="cast", parameters={}),
            
            # 문자열 타입 변환
            ConversionRule(source_type="int64", target_type="string", conversion_function="to_string", parameters={}),
            ConversionRule(source_type="float64", target_type="string", conversion_function="to_string", parameters={}),
            ConversionRule(source_type="bool", target_type="string", conversion_function="to_string", parameters={}),
            ConversionRule(source_type="timestamp", target_type="string", conversion_function="to_string", parameters={}),
            
            # 날짜/시간 타입 변환
            ConversionRule(source_type="string", target_type="timestamp", conversion_function="parse_timestamp", parameters={}),
            ConversionRule(source_type="string", target_type="date", conversion_function="parse_date", parameters={}),
            ConversionRule(source_type="timestamp", target_type="date", conversion_function="extract_date", parameters={}),
            
            # 불린 타입 변환
            ConversionRule(source_type="string", target_type="bool", conversion_function="parse_bool", parameters={}),
            ConversionRule(source_type="int64", target_type="bool", conversion_function="int_to_bool", parameters={}),
            ConversionRule(source_type="float64", target_type="bool", conversion_function="float_to_bool", parameters={}),
        ]
        
        for rule in default_rules:
            self._conversion_rules[f"{rule.source_type}->{rule.target_type}"] = rule

    def convert_column(
        self,
        array: pa.Array,
        target_type: pa.DataType,
        conversion_rule: Optional[ConversionRule] = None
    ) -> pa.Array:
        """컬럼의 데이터 타입을 변환합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            conversion_rule: 변환 규칙 (None이면 자동 선택)
            
        Returns:
            pa.Array: 변환된 배열
        """
        source_type_str = str(array.type)
        target_type_str = str(target_type)
        
        # 이미 같은 타입이면 그대로 반환
        if source_type_str == target_type_str:
            return array
        
        # 변환 규칙 선택
        if conversion_rule is None:
            conversion_rule = self._get_conversion_rule(source_type_str, target_type_str)
        
        if conversion_rule is None:
            logger.warning(f"변환 규칙을 찾을 수 없습니다: {source_type_str} -> {target_type_str}")
            return self._fallback_conversion(array, target_type)
        
        # 변환 실행
        try:
            return self._apply_conversion(array, target_type, conversion_rule)
        except Exception as e:
            logger.error(f"데이터 타입 변환 실패: {e}")
            return self._handle_conversion_error(array, target_type, conversion_rule)

    def _get_conversion_rule(self, source_type: str, target_type: str) -> Optional[ConversionRule]:
        """변환 규칙을 찾습니다.
        
        Args:
            source_type: 원본 타입
            target_type: 대상 타입
            
        Returns:
            Optional[ConversionRule]: 변환 규칙 또는 None
        """
        # 정확한 매칭
        rule_key = f"{source_type}->{target_type}"
        if rule_key in self._conversion_rules:
            return self._conversion_rules[rule_key]
        
        # 부분 매칭 (타입 계층 구조 고려)
        for key, rule in self._conversion_rules.items():
            if self._is_compatible_conversion(source_type, target_type, rule):
                return rule
        
        return None

    def _is_compatible_conversion(
        self, 
        source_type: str, 
        target_type: str, 
        rule: ConversionRule
    ) -> bool:
        """변환 규칙이 호환되는지 확인합니다.
        
        Args:
            source_type: 원본 타입
            target_type: 대상 타입
            rule: 변환 규칙
            
        Returns:
            bool: 호환 여부
        """
        # 정수 타입 계층 구조
        int_types = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
        if source_type in int_types and rule.source_type in int_types:
            return True
        
        # 부동소수점 타입 계층 구조
        float_types = ["float32", "float64"]
        if source_type in float_types and rule.source_type in float_types:
            return True
        
        # 문자열 타입
        if source_type == "string" and rule.source_type == "string":
            return True
        
        return False

    def _apply_conversion(
        self, 
        array: pa.Array, 
        target_type: pa.DataType, 
        rule: ConversionRule
    ) -> pa.Array:
        """변환 규칙을 적용합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            rule: 변환 규칙
            
        Returns:
            pa.Array: 변환된 배열
        """
        function_name = rule.conversion_function
        
        if function_name == "cast":
            return pa.compute.cast(array, target_type)
        elif function_name == "to_string":
            return pa.compute.cast(array, pa.string())
        elif function_name == "parse_timestamp":
            return self._parse_timestamp(array, target_type, rule.parameters)
        elif function_name == "parse_date":
            return self._parse_date(array, target_type, rule.parameters)
        elif function_name == "extract_date":
            return self._extract_date(array, target_type)
        elif function_name == "parse_bool":
            return self._parse_bool(array, rule.parameters)
        elif function_name == "int_to_bool":
            return self._int_to_bool(array)
        elif function_name == "float_to_bool":
            return self._float_to_bool(array)
        elif function_name in self._custom_functions:
            return self._custom_functions[function_name](array, target_type, rule.parameters)
        else:
            raise ValueError(f"지원하지 않는 변환 함수: {function_name}")

    def _parse_timestamp(
        self, 
        array: pa.Array, 
        target_type: pa.DataType, 
        parameters: Dict[str, Any]
    ) -> pa.Array:
        """문자열을 타임스탬프로 변환합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            parameters: 변환 파라미터
            
        Returns:
            pa.Array: 변환된 배열
        """
        format_str = parameters.get("format", "%Y-%m-%d %H:%M:%S")
        
        try:
            # Arrow의 strptime 함수 사용
            return pa.compute.strptime(array, format_str, target_type)
        except Exception as e:
            logger.warning(f"타임스탬프 파싱 실패: {e}")
            return pa.nulls(len(array), target_type)

    def _parse_date(
        self, 
        array: pa.Array, 
        target_type: pa.DataType, 
        parameters: Dict[str, Any]
    ) -> pa.Array:
        """문자열을 날짜로 변환합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            parameters: 변환 파라미터
            
        Returns:
            pa.Array: 변환된 배열
        """
        format_str = parameters.get("format", "%Y-%m-%d")
        
        try:
            # Arrow의 strptime 함수 사용 후 날짜로 변환
            timestamp_array = pa.compute.strptime(array, format_str, pa.timestamp("us"))
            return pa.compute.cast(timestamp_array, target_type)
        except Exception as e:
            logger.warning(f"날짜 파싱 실패: {e}")
            return pa.nulls(len(array), target_type)

    def _extract_date(self, array: pa.Array, target_type: pa.DataType) -> pa.Array:
        """타임스탬프에서 날짜를 추출합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            
        Returns:
            pa.Array: 변환된 배열
        """
        try:
            return pa.compute.cast(array, target_type)
        except Exception as e:
            logger.warning(f"날짜 추출 실패: {e}")
            return pa.nulls(len(array), target_type)

    def _parse_bool(self, array: pa.Array, parameters: Dict[str, Any]) -> pa.Array:
        """문자열을 불린으로 변환합니다.
        
        Args:
            array: 변환할 배열
            parameters: 변환 파라미터
            
        Returns:
            pa.Array: 변환된 배열
        """
        true_values = parameters.get("true_values", ["true", "1", "yes", "on"])
        false_values = parameters.get("false_values", ["false", "0", "no", "off"])
        
        try:
            # 대소문자 구분 없이 변환
            lower_array = pa.compute.utf8_lower(array)
            
            # true 값 매핑
            true_mask = pa.compute.is_in(lower_array, pa.array(true_values))
            false_mask = pa.compute.is_in(lower_array, pa.array(false_values))
            
            # 결과 생성
            result = pa.nulls(len(array), pa.bool_())
            result = pa.compute.if_else(true_mask, pa.array([True] * len(array)), result)
            result = pa.compute.if_else(false_mask, pa.array([False] * len(array)), result)
            
            return result
        except Exception as e:
            logger.warning(f"불린 파싱 실패: {e}")
            return pa.nulls(len(array), pa.bool_())

    def _int_to_bool(self, array: pa.Array) -> pa.Array:
        """정수를 불린으로 변환합니다.
        
        Args:
            array: 변환할 배열
            
        Returns:
            pa.Array: 변환된 배열
        """
        try:
            # 0이 아닌 값은 True, 0은 False
            return pa.compute.not_equal(array, pa.scalar(0))
        except Exception as e:
            logger.warning(f"정수->불린 변환 실패: {e}")
            return pa.nulls(len(array), pa.bool_())

    def _float_to_bool(self, array: pa.Array) -> pa.Array:
        """부동소수점을 불린으로 변환합니다.
        
        Args:
            array: 변환할 배열
            
        Returns:
            pa.Array: 변환된 배열
        """
        try:
            # 0.0이 아닌 값은 True, 0.0은 False
            return pa.compute.not_equal(array, pa.scalar(0.0))
        except Exception as e:
            logger.warning(f"부동소수점->불린 변환 실패: {e}")
            return pa.nulls(len(array), pa.bool_())

    def _fallback_conversion(self, array: pa.Array, target_type: pa.DataType) -> pa.Array:
        """기본 변환을 시도합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            
        Returns:
            pa.Array: 변환된 배열
        """
        try:
            return pa.compute.cast(array, target_type)
        except Exception as e:
            logger.warning(f"기본 변환 실패: {e}")
            return pa.nulls(len(array), target_type)

    def _handle_conversion_error(
        self, 
        array: pa.Array, 
        target_type: pa.DataType, 
        rule: ConversionRule
    ) -> pa.Array:
        """변환 에러를 처리합니다.
        
        Args:
            array: 변환할 배열
            target_type: 대상 데이터 타입
            rule: 변환 규칙
            
        Returns:
            pa.Array: 처리된 배열
        """
        if rule.error_handling == "null":
            return pa.nulls(len(array), target_type)
        elif rule.error_handling == "skip":
            return array  # 원본 유지
        else:  # raise
            raise ValueError(f"데이터 타입 변환 실패: {array.type} -> {target_type}")

    def add_conversion_rule(self, rule: ConversionRule) -> None:
        """사용자 정의 변환 규칙을 추가합니다.
        
        Args:
            rule: 변환 규칙
        """
        rule_key = f"{rule.source_type}->{rule.target_type}"
        self._conversion_rules[rule_key] = rule
        logger.info(f"변환 규칙 추가: {rule_key}")

    def add_custom_function(self, name: str, function: callable) -> None:
        """사용자 정의 변환 함수를 추가합니다.
        
        Args:
            name: 함수명
            function: 변환 함수
        """
        self._custom_functions[name] = function
        logger.info(f"사용자 정의 함수 추가: {name}")

    def convert_table(
        self, 
        table: pa.Table, 
        target_schema: pa.Schema,
        column_mappings: Optional[Dict[str, str]] = None
    ) -> pa.Table:
        """테이블의 데이터 타입을 변환합니다.
        
        Args:
            table: 변환할 테이블
            target_schema: 대상 스키마
            column_mappings: 컬럼 매핑 {원본컬럼: 대상컬럼}
            
        Returns:
            pa.Table: 변환된 테이블
        """
        converted_arrays = []
        converted_schema_fields = []
        
        for i, target_field in enumerate(target_schema):
            target_column = target_field.name
            
            # 컬럼 매핑 확인
            source_column = target_column
            if column_mappings:
                for src_col, tgt_col in column_mappings.items():
                    if tgt_col == target_column:
                        source_column = src_col
                        break
            
            # 원본 컬럼 찾기
            if source_column in table.column_names:
                source_array = table.column(source_column)
                converted_array = self.convert_column(source_array, target_field.type)
            else:
                # 컬럼이 없으면 null로 채움
                converted_array = pa.nulls(len(table), target_field.type)
            
            converted_arrays.append(converted_array)
            converted_schema_fields.append(pa.field(target_column, converted_array.type))
        
        # 변환된 테이블 생성
        converted_schema = pa.schema(converted_schema_fields)
        return pa.Table.from_arrays(converted_arrays, schema=converted_schema)

    def get_conversion_rules(self) -> Dict[str, ConversionRule]:
        """모든 변환 규칙을 반환합니다.
        
        Returns:
            Dict[str, ConversionRule]: 변환 규칙 딕셔너리
        """
        return self._conversion_rules.copy()

    def validate_conversion(
        self, 
        source_type: str, 
        target_type: str
    ) -> Dict[str, Any]:
        """변환 가능성을 검증합니다.
        
        Args:
            source_type: 원본 타입
            target_type: 대상 타입
            
        Returns:
            Dict[str, Any]: 검증 결과
        """
        rule_key = f"{source_type}->{target_type}"
        
        if rule_key in self._conversion_rules:
            return {
                "is_convertible": True,
                "rule": self._conversion_rules[rule_key],
                "method": "exact_match"
            }
        
        # 부분 매칭 확인
        for key, rule in self._conversion_rules.items():
            if self._is_compatible_conversion(source_type, target_type, rule):
                return {
                    "is_convertible": True,
                    "rule": rule,
                    "method": "compatible_match"
                }
        
        return {
            "is_convertible": False,
            "rule": None,
            "method": "no_match"
        }

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"TypeConverter(rules={len(self._conversion_rules)}, functions={len(self._custom_functions)})"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
