"""통합 데이터 분석 레이어 - 메인 통합 클래스.

CA 마일스톤 3.1: 통합 데이터 분석 레이어
다중 소스 데이터를 통합하고 표준화된 분석을 제공하는 메인 클래스입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.core.data_unifier import DataUnifier
from bridge.analytics.core.schema_mapper import SchemaMapper
from bridge.analytics.core.streaming_processor import StreamingProcessor
from bridge.analytics.core.type_converter import TypeConverter

logger = logging.getLogger(__name__)


class IntegratedDataLayer:
    """통합 데이터 분석 레이어 메인 클래스.
    
    다중 소스 데이터를 통합하고 표준화된 분석을 제공합니다.
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        memory_limit_mb: int = 1000,
        auto_schema_mapping: bool = True
    ):
        """IntegratedDataLayer를 초기화합니다.
        
        Args:
            chunk_size: 스트리밍 처리 청크 크기
            memory_limit_mb: 메모리 제한 (MB)
            auto_schema_mapping: 자동 스키마 매핑 여부
        """
        self.data_unifier = DataUnifier()
        self.schema_mapper = SchemaMapper()
        self.type_converter = TypeConverter()
        self.streaming_processor = StreamingProcessor(chunk_size, memory_limit_mb)
        self.auto_schema_mapping = auto_schema_mapping
        
        # 통합된 데이터 저장소
        self._unified_data: Optional[UnifiedDataFrame] = None
        self._source_metadata: Dict[str, Any] = {}

    def integrate_data_sources(
        self,
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]],
        schema_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        merge_strategy: str = "union",
        enable_streaming: bool = True
    ) -> UnifiedDataFrame:
        """다중 데이터 소스를 통합합니다.
        
        Args:
            data_sources: 데이터 소스 딕셔너리 {소스명: 데이터}
            schema_mapping: 스키마 매핑 규칙
            merge_strategy: 통합 전략 ("union", "intersection", "custom")
            enable_streaming: 스트리밍 처리 활성화 여부
            
        Returns:
            UnifiedDataFrame: 통합된 데이터
        """
        logger.info(f"데이터 소스 통합 시작: {list(data_sources.keys())}")
        
        # 1. 스키마 매핑 생성 (자동 매핑이 활성화된 경우)
        if self.auto_schema_mapping and schema_mapping is None:
            schema_mapping = self._create_auto_schema_mapping(data_sources)
        
        # 2. 스트리밍 처리 여부 결정
        total_size = self._estimate_total_size(data_sources)
        use_streaming = enable_streaming and total_size > self.streaming_processor.memory_limit_bytes
        
        if use_streaming:
            logger.info("스트리밍 처리 모드로 데이터 통합")
            unified_data = self._integrate_with_streaming(data_sources, schema_mapping, merge_strategy)
        else:
            logger.info("일반 처리 모드로 데이터 통합")
            unified_data = self._integrate_without_streaming(data_sources, schema_mapping, merge_strategy)
        
        # 3. 통합된 데이터 저장
        self._unified_data = unified_data
        self._source_metadata = self.data_unifier.get_source_metadata()
        
        logger.info(f"데이터 통합 완료: {unified_data.num_rows}행, {unified_data.num_columns}열")
        return unified_data

    def _create_auto_schema_mapping(
        self, 
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]]
    ) -> Dict[str, Dict[str, str]]:
        """자동 스키마 매핑을 생성합니다.
        
        Args:
            data_sources: 데이터 소스 딕셔너리
            
        Returns:
            Dict[str, Dict[str, str]]: 자동 생성된 스키마 매핑
        """
        schema_mapping = {}
        
        for source_name, data in data_sources.items():
            # 데이터를 Arrow Table로 변환
            if isinstance(data, pa.Table):
                table = data
            elif isinstance(data, pd.DataFrame):
                table = pa.Table.from_pandas(data)
            elif isinstance(data, list):
                table = pa.Table.from_pylist(data)
            else:
                continue
            
            # 컬럼명 정규화
            source_mapping = {}
            for field in table.schema:
                normalized_name = self.schema_mapper._normalize_column_name(field.name)
                if normalized_name != field.name:
                    source_mapping[field.name] = normalized_name
            
            if source_mapping:
                schema_mapping[source_name] = source_mapping
        
        return schema_mapping

    def _estimate_total_size(
        self, 
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]]
    ) -> int:
        """전체 데이터 크기를 추정합니다.
        
        Args:
            data_sources: 데이터 소스 딕셔너리
            
        Returns:
            int: 추정된 전체 크기 (바이트)
        """
        total_size = 0
        
        for data in data_sources.values():
            if isinstance(data, pa.Table):
                total_size += data.nbytes
            elif isinstance(data, pd.DataFrame):
                total_size += data.memory_usage(deep=True).sum()
            elif isinstance(data, list):
                # 리스트의 경우 대략적인 추정
                total_size += len(data) * 100  # 행당 평균 100바이트로 추정
        
        return total_size

    def _integrate_with_streaming(
        self,
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]],
        schema_mapping: Optional[Dict[str, Dict[str, str]]],
        merge_strategy: str
    ) -> UnifiedDataFrame:
        """스트리밍 처리로 데이터를 통합합니다.
        
        Args:
            data_sources: 데이터 소스 딕셔너리
            schema_mapping: 스키마 매핑 규칙
            merge_strategy: 통합 전략
            
        Returns:
            UnifiedDataFrame: 통합된 데이터
        """
        # 스트리밍 처리로 각 소스별로 처리
        processed_sources = {}
        
        for source_name, data in data_sources.items():
            try:
                # 데이터를 Arrow Table로 변환
                if isinstance(data, pa.Table):
                    table = data
                elif isinstance(data, pd.DataFrame):
                    table = pa.Table.from_pandas(data)
                elif isinstance(data, list):
                    table = pa.Table.from_pylist(data)
                else:
                    continue
                
                # 스키마 매핑 적용
                if schema_mapping and source_name in schema_mapping:
                    mapped_table = self.schema_mapper.apply_mapping(source_name, table)
                else:
                    mapped_table = table
                
                # 스트리밍 처리
                processed_table = self.streaming_processor.process_large_table(
                    mapped_table,
                    lambda x, **kwargs: x  # 단순 통과 함수
                )
                
                processed_sources[source_name] = processed_table
                
            except Exception as e:
                logger.error(f"소스 '{source_name}' 스트리밍 처리 실패: {e}")
                continue
        
        # 통합 처리
        return self.data_unifier.unify_data_sources(
            processed_sources,
            schema_mapping,
            merge_strategy
        )

    def _integrate_without_streaming(
        self,
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]],
        schema_mapping: Optional[Dict[str, Dict[str, str]]],
        merge_strategy: str
    ) -> UnifiedDataFrame:
        """일반 처리로 데이터를 통합합니다.
        
        Args:
            data_sources: 데이터 소스 딕셔너리
            schema_mapping: 스키마 매핑 규칙
            merge_strategy: 통합 전략
            
        Returns:
            UnifiedDataFrame: 통합된 데이터
        """
        # 스키마 매핑 적용
        processed_sources = {}
        
        for source_name, data in data_sources.items():
            try:
                # 데이터를 UnifiedDataFrame으로 변환
                unified_df = UnifiedDataFrame(data)
                
                # 스키마 매핑 적용
                if schema_mapping and source_name in schema_mapping:
                    mapped_table = self.schema_mapper.apply_mapping(source_name, unified_df.table)
                    processed_sources[source_name] = mapped_table
                else:
                    processed_sources[source_name] = unified_df.table
                
            except Exception as e:
                logger.error(f"소스 '{source_name}' 처리 실패: {e}")
                continue
        
        # 통합 처리
        return self.data_unifier.unify_data_sources(
            processed_sources,
            schema_mapping,
            merge_strategy
        )

    def get_unified_data(self) -> Optional[UnifiedDataFrame]:
        """통합된 데이터를 반환합니다.
        
        Returns:
            Optional[UnifiedDataFrame]: 통합된 데이터 또는 None
        """
        return self._unified_data

    def get_data_summary(self) -> Dict[str, Any]:
        """통합된 데이터의 요약 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 데이터 요약 정보
        """
        if self._unified_data is None:
            return {"error": "통합된 데이터가 없습니다."}
        
        return self.data_unifier.create_data_summary(self._unified_data)

    def validate_data_quality(self) -> Dict[str, Any]:
        """데이터 품질을 검증합니다.
        
        Returns:
            Dict[str, Any]: 데이터 품질 검증 결과
        """
        if self._unified_data is None:
            return {"error": "통합된 데이터가 없습니다."}
        
        return self.data_unifier.validate_unified_data(self._unified_data)

    def apply_data_transformations(
        self,
        transformations: List[Dict[str, Any]]
    ) -> UnifiedDataFrame:
        """데이터 변환을 적용합니다.
        
        Args:
            transformations: 변환 규칙 목록
            
        Returns:
            UnifiedDataFrame: 변환된 데이터
        """
        if self._unified_data is None:
            raise ValueError("통합된 데이터가 없습니다.")
        
        logger.info(f"데이터 변환 적용 시작: {len(transformations)}개 변환")
        
        # 변환 적용
        transformed_table = self._unified_data.table
        
        for transformation in transformations:
            try:
                transformed_table = self._apply_single_transformation(
                    transformed_table, 
                    transformation
                )
            except Exception as e:
                logger.error(f"변환 적용 실패: {transformation}, {e}")
                continue
        
        # 변환된 데이터로 업데이트
        self._unified_data = UnifiedDataFrame(transformed_table)
        
        logger.info("데이터 변환 적용 완료")
        return self._unified_data

    def _apply_single_transformation(
        self, 
        table: pa.Table, 
        transformation: Dict[str, Any]
    ) -> pa.Table:
        """단일 변환을 적용합니다.
        
        Args:
            table: 변환할 테이블
            transformation: 변환 규칙
            
        Returns:
            pa.Table: 변환된 테이블
        """
        transformation_type = transformation.get("type")
        
        if transformation_type == "column_rename":
            return self._rename_columns(table, transformation)
        elif transformation_type == "column_type_conversion":
            return self._convert_column_types(table, transformation)
        elif transformation_type == "column_filter":
            return self._filter_columns(table, transformation)
        elif transformation_type == "row_filter":
            return self._filter_rows(table, transformation)
        else:
            logger.warning(f"지원하지 않는 변환 타입: {transformation_type}")
            return table

    def _rename_columns(self, table: pa.Table, transformation: Dict[str, Any]) -> pa.Table:
        """컬럼명을 변경합니다.
        
        Args:
            table: 변환할 테이블
            transformation: 변환 규칙
            
        Returns:
            pa.Table: 변환된 테이블
        """
        column_mapping = transformation.get("column_mapping", {})
        
        if not column_mapping:
            return table
        
        # 새로운 스키마 생성
        new_fields = []
        for field in table.schema:
            new_name = column_mapping.get(field.name, field.name)
            new_fields.append(pa.field(new_name, field.type))
        
        new_schema = pa.schema(new_fields)
        
        # 컬럼명 변경을 위해 새로운 배열 생성
        new_arrays = []
        for field in table.schema:
            new_name = column_mapping.get(field.name, field.name)
            new_field = pa.field(new_name, field.type)
            new_arrays.append(table.column(field.name))
        
        return pa.Table.from_arrays(new_arrays, schema=new_schema)

    def _convert_column_types(self, table: pa.Table, transformation: Dict[str, Any]) -> pa.Table:
        """컬럼 타입을 변환합니다.
        
        Args:
            table: 변환할 테이블
            transformation: 변환 규칙
            
        Returns:
            pa.Table: 변환된 테이블
        """
        type_mapping = transformation.get("type_mapping", {})
        
        if not type_mapping:
            return table
        
        return self.type_converter.convert_table(table, table.schema, type_mapping)

    def _filter_columns(self, table: pa.Table, transformation: Dict[str, Any]) -> pa.Table:
        """컬럼을 필터링합니다.
        
        Args:
            table: 변환할 테이블
            transformation: 변환 규칙
            
        Returns:
            pa.Table: 변환된 테이블
        """
        columns_to_keep = transformation.get("columns", [])
        
        if not columns_to_keep:
            return table
        
        # 존재하는 컬럼만 선택
        existing_columns = [col for col in columns_to_keep if col in table.column_names]
        
        if not existing_columns:
            logger.warning("선택된 컬럼이 존재하지 않습니다.")
            return table
        
        return table.select(existing_columns)

    def _filter_rows(self, table: pa.Table, transformation: Dict[str, Any]) -> pa.Table:
        """행을 필터링합니다.
        
        Args:
            table: 변환할 테이블
            transformation: 변환 규칙
            
        Returns:
            pa.Table: 변환된 테이블
        """
        # 간단한 필터링 구현 (향후 확장 가능)
        condition = transformation.get("condition")
        
        if not condition:
            return table
        
        # 기본적인 조건 처리
        try:
            # 예: "column_name > value" 형태의 조건
            parts = condition.split()
            if len(parts) == 3:
                column_name, operator, value = parts
                if column_name in table.column_names:
                    column_array = table.column(column_name)
                    
                    if operator == ">":
                        mask = pa.compute.greater(column_array, pa.scalar(float(value)))
                    elif operator == "<":
                        mask = pa.compute.less(column_array, pa.scalar(float(value)))
                    elif operator == "==":
                        mask = pa.compute.equal(column_array, pa.scalar(value))
                    else:
                        logger.warning(f"지원하지 않는 연산자: {operator}")
                        return table
                    
                    return table.filter(mask)
        except Exception as e:
            logger.error(f"행 필터링 실패: {e}")
        
        return table

    def export_data(
        self,
        format: str = "arrow",
        file_path: Optional[str] = None
    ) -> Union[pa.Table, pd.DataFrame, str]:
        """통합된 데이터를 내보냅니다.
        
        Args:
            format: 내보낼 형식 ("arrow", "pandas", "csv", "parquet")
            file_path: 파일 경로 (None이면 메모리에서 반환)
            
        Returns:
            Union[pa.Table, pd.DataFrame, str]: 내보낸 데이터
        """
        if self._unified_data is None:
            raise ValueError("통합된 데이터가 없습니다.")
        
        if format == "arrow":
            return self._unified_data.to_arrow()
        elif format == "pandas":
            return self._unified_data.to_pandas()
        elif format == "csv":
            if file_path:
                self._unified_data.to_pandas().to_csv(file_path, index=False)
                return f"CSV 파일이 저장되었습니다: {file_path}"
            else:
                return self._unified_data.to_pandas().to_csv(index=False)
        elif format == "parquet":
            if file_path:
                self._unified_data.to_arrow().to_parquet(file_path)
                return f"Parquet 파일이 저장되었습니다: {file_path}"
            else:
                return self._unified_data.to_arrow()
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량을 반환합니다.
        
        Returns:
            Dict[str, Any]: 메모리 사용량 정보
        """
        if self._unified_data is None:
            return {"error": "통합된 데이터가 없습니다."}
        
        return self.streaming_processor.get_memory_usage(self._unified_data.table)

    def __repr__(self) -> str:
        """문자열 표현."""
        data_info = f"({self._unified_data.num_rows}행, {self._unified_data.num_columns}열)" if self._unified_data else "(데이터 없음)"
        return f"IntegratedDataLayer{data_info}"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
