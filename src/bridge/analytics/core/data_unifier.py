"""통합 데이터 분석 레이어 - 데이터 통합 모듈.

CA 마일스톤 3.1: 통합 데이터 분석 레이어
다중 소스 데이터를 표준 테이블 형태로 통합하는 핵심 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.core.type_normalizer import TypeNormalizer

logger = logging.getLogger(__name__)


class DataUnifier:
    """다중 소스 데이터 통합 클래스.
    
    다양한 데이터 소스에서 오는 데이터를 표준화된 형태로 통합하여
    일관된 분석이 가능하도록 합니다.
    """

    def __init__(self, type_normalizer: Optional[TypeNormalizer] = None):
        """DataUnifier를 초기화합니다.
        
        Args:
            type_normalizer: 타입 정규화기. None이면 기본 인스턴스 생성
        """
        self.type_normalizer = type_normalizer or TypeNormalizer()
        self._unified_schema: Optional[pa.Schema] = None
        self._source_metadata: Dict[str, Any] = {}

    def unify_data_sources(
        self, 
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]],
        schema_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        merge_strategy: str = "union"
    ) -> UnifiedDataFrame:
        """다중 데이터 소스를 통합합니다.
        
        Args:
            data_sources: 데이터 소스 딕셔너리 {소스명: 데이터}
            schema_mapping: 스키마 매핑 규칙 {소스명: {원본컬럼: 통합컬럼}}
            merge_strategy: 통합 전략 ("union", "intersection", "custom")
            
        Returns:
            UnifiedDataFrame: 통합된 데이터
            
        Raises:
            ValueError: 잘못된 통합 전략이거나 데이터 통합 실패 시
        """
        if not data_sources:
            logger.warning("통합할 데이터 소스가 없습니다.")
            return UnifiedDataFrame()

        logger.info(f"데이터 소스 통합 시작: {list(data_sources.keys())}")
        
        # 1. 각 데이터 소스를 UnifiedDataFrame으로 변환
        unified_sources = {}
        for source_name, data in data_sources.items():
            try:
                unified_df = UnifiedDataFrame(data)
                unified_sources[source_name] = unified_df
                logger.debug(f"소스 '{source_name}' 변환 완료: {unified_df.num_rows}행, {unified_df.num_columns}열")
            except Exception as e:
                logger.error(f"소스 '{source_name}' 변환 실패: {e}")
                continue

        if not unified_sources:
            raise ValueError("통합 가능한 데이터 소스가 없습니다.")

        # 2. 스키마 통합
        unified_schema = self._create_unified_schema(
            unified_sources, 
            schema_mapping or {}, 
            merge_strategy
        )

        # 3. 데이터 정규화 및 통합
        unified_data = self._merge_data_sources(unified_sources, unified_schema, schema_mapping or {})

        # 4. 메타데이터 저장
        self._store_source_metadata(unified_sources, schema_mapping or {})

        logger.info(f"데이터 통합 완료: {unified_data.num_rows}행, {unified_data.num_columns}열")
        return unified_data

    def _create_unified_schema(
        self,
        unified_sources: Dict[str, UnifiedDataFrame],
        schema_mapping: Dict[str, Dict[str, str]],
        merge_strategy: str
    ) -> pa.Schema:
        """통합 스키마를 생성합니다.
        
        Args:
            unified_sources: 통합된 데이터 소스들
            schema_mapping: 스키마 매핑 규칙
            merge_strategy: 통합 전략
            
        Returns:
            pa.Schema: 통합된 스키마
        """
        all_columns = {}
        
        for source_name, unified_df in unified_sources.items():
            source_mapping = schema_mapping.get(source_name, {})
            
            for field in unified_df.schema:
                # 매핑 규칙이 있으면 통합 컬럼명 사용, 없으면 원본 컬럼명 사용
                unified_column = source_mapping.get(field.name, field.name)
                
                if unified_column in all_columns:
                    # 이미 존재하는 컬럼의 경우 타입 통합
                    existing_type = all_columns[unified_column]
                    unified_type = self._unify_column_types(existing_type, field.type)
                    all_columns[unified_column] = unified_type
                else:
                    all_columns[unified_column] = field.type

        # 스키마 생성
        unified_fields = [
            pa.field(column_name, column_type) 
            for column_name, column_type in all_columns.items()
        ]
        
        return pa.schema(unified_fields)

    def _unify_column_types(self, type1: pa.DataType, type2: pa.DataType) -> pa.DataType:
        """두 컬럼 타입을 통합합니다.
        
        Args:
            type1: 첫 번째 타입
            type2: 두 번째 타입
            
        Returns:
            pa.DataType: 통합된 타입
        """
        # 타입이 같으면 그대로 반환
        if type1 == type2:
            return type1
            
        # 타입 우선순위: string > float64 > int64 > bool
        type_priority = {
            pa.string(): 4,
            pa.float64(): 3,
            pa.int64(): 2,
            pa.bool_(): 1
        }
        
        priority1 = type_priority.get(type1, 0)
        priority2 = type_priority.get(type2, 0)
        
        return type1 if priority1 >= priority2 else type2

    def _merge_data_sources(
        self,
        unified_sources: Dict[str, UnifiedDataFrame],
        unified_schema: pa.Schema,
        schema_mapping: Dict[str, Dict[str, str]]
    ) -> UnifiedDataFrame:
        """데이터 소스들을 통합합니다.
        
        Args:
            unified_sources: 통합된 데이터 소스들
            unified_schema: 통합 스키마
            schema_mapping: 스키마 매핑 규칙
            
        Returns:
            UnifiedDataFrame: 통합된 데이터
        """
        all_arrays = []
        
        for source_name, unified_df in unified_sources.items():
            source_mapping = schema_mapping.get(source_name, {})
            
            # 각 컬럼에 대해 데이터 추출
            source_arrays = []
            for field in unified_schema:
                unified_column = field.name
                
                # 원본 컬럼명 찾기
                original_column = None
                for orig_col, unified_col in source_mapping.items():
                    if unified_col == unified_column:
                        original_column = orig_col
                        break
                
                if original_column is None:
                    # 매핑되지 않은 컬럼은 None으로 채움
                    source_arrays.append(pa.nulls(unified_df.num_rows, field.type))
                else:
                    # 원본 컬럼에서 데이터 추출
                    if original_column in unified_df.column_names:
                        original_data = unified_df.table.column(original_column)
                        # 타입 변환
                        if original_data.type != field.type:
                            try:
                                converted_data = pa.compute.cast(original_data, field.type)
                                source_arrays.append(converted_data)
                            except Exception as e:
                                logger.warning(f"컬럼 '{original_column}' 타입 변환 실패: {e}")
                                source_arrays.append(pa.nulls(unified_df.num_rows, field.type))
                        else:
                            source_arrays.append(original_data)
                    else:
                        # 컬럼이 없으면 None으로 채움
                        source_arrays.append(pa.nulls(unified_df.num_rows, field.type))
            
            # 소스별 테이블 생성
            source_table = pa.Table.from_arrays(source_arrays, schema=unified_schema)
            all_arrays.append(source_table)

        # 모든 테이블을 하나로 결합
        if all_arrays:
            unified_table = pa.concat_tables(all_arrays)
            return UnifiedDataFrame(unified_table)
        else:
            return UnifiedDataFrame()

    def _store_source_metadata(
        self,
        unified_sources: Dict[str, UnifiedDataFrame],
        schema_mapping: Dict[str, Dict[str, str]]
    ) -> None:
        """소스 메타데이터를 저장합니다.
        
        Args:
            unified_sources: 통합된 데이터 소스들
            schema_mapping: 스키마 매핑 규칙
        """
        self._source_metadata = {
            "sources": {
                source_name: {
                    "rows": unified_df.num_rows,
                    "columns": unified_df.num_columns,
                    "column_names": unified_df.column_names
                }
                for source_name, unified_df in unified_sources.items()
            },
            "schema_mapping": schema_mapping,
            "total_sources": len(unified_sources)
        }

    def get_source_metadata(self) -> Dict[str, Any]:
        """소스 메타데이터를 반환합니다.
        
        Returns:
            Dict[str, Any]: 소스 메타데이터
        """
        return self._source_metadata.copy()

    def validate_unified_data(self, unified_data: UnifiedDataFrame) -> Dict[str, Any]:
        """통합된 데이터의 유효성을 검증합니다.
        
        Args:
            unified_data: 검증할 통합 데이터
            
        Returns:
            Dict[str, Any]: 검증 결과
        """
        validation_result = {
            "is_valid": True,
            "total_rows": unified_data.num_rows,
            "total_columns": unified_data.num_columns,
            "column_types": {
                name: str(field.type) 
                for name, field in zip(unified_data.schema.names, unified_data.schema)
            },
            "missing_values": {},
            "data_quality_score": 0.0
        }

        # 결측값 분석
        for i, field in enumerate(unified_data.schema):
            column_name = field.name
            column_data = unified_data.table.column(i)
            
            # 결측값 개수 계산
            null_count = pa.compute.sum(pa.compute.is_null(column_data)).as_py()
            total_count = len(column_data)
            missing_ratio = null_count / total_count if total_count > 0 else 0.0
            
            validation_result["missing_values"][column_name] = {
                "count": null_count,
                "ratio": missing_ratio
            }

        # 데이터 품질 점수 계산 (결측값 비율 기반)
        total_columns = len(unified_data.schema)
        if total_columns > 0:
            avg_missing_ratio = sum(
                info["ratio"] for info in validation_result["missing_values"].values()
            ) / total_columns
            validation_result["data_quality_score"] = max(0.0, 1.0 - avg_missing_ratio)

        return validation_result

    def create_data_summary(self, unified_data: UnifiedDataFrame) -> Dict[str, Any]:
        """통합된 데이터의 요약 정보를 생성합니다.
        
        Args:
            unified_data: 요약할 통합 데이터
            
        Returns:
            Dict[str, Any]: 데이터 요약 정보
        """
        summary = {
            "basic_info": {
                "total_rows": unified_data.num_rows,
                "total_columns": unified_data.num_columns,
                "column_names": unified_data.column_names
            },
            "column_types": {
                name: str(field.type) 
                for name, field in zip(unified_data.schema.names, unified_data.schema)
            },
            "source_info": self._source_metadata,
            "memory_usage": self._estimate_memory_usage(unified_data)
        }

        return summary

    def _estimate_memory_usage(self, unified_data: UnifiedDataFrame) -> Dict[str, Any]:
        """메모리 사용량을 추정합니다.
        
        Args:
            unified_data: 메모리 사용량을 추정할 데이터
            
        Returns:
            Dict[str, Any]: 메모리 사용량 정보
        """
        try:
            # Arrow Table의 메모리 사용량 계산
            memory_usage = unified_data.table.nbytes
            memory_usage_mb = memory_usage / (1024 * 1024)
            
            return {
                "bytes": memory_usage,
                "mb": round(memory_usage_mb, 2),
                "gb": round(memory_usage_mb / 1024, 2)
            }
        except Exception as e:
            logger.warning(f"메모리 사용량 추정 실패: {e}")
            return {"bytes": 0, "mb": 0, "gb": 0}

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"DataUnifier(sources={len(self._source_metadata.get('sources', {}))})"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
