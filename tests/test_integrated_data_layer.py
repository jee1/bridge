"""통합 데이터 분석 레이어 테스트.

CA 마일스톤 3.1: 통합 데이터 분석 레이어 테스트
"""

import pytest
import pyarrow as pa
import pandas as pd
from typing import Dict, Any, List

from bridge.analytics.core import (
    DataUnifier,
    SchemaMapper,
    TypeConverter,
    StreamingProcessor,
    IntegratedDataLayer,
    UnifiedDataFrame,
)


class TestDataUnifier:
    """DataUnifier 테스트 클래스"""

    def test_data_unifier_initialization(self):
        """DataUnifier 초기화 테스트"""
        unifier = DataUnifier()
        assert unifier is not None
        assert unifier.type_normalizer is not None

    def test_unify_simple_data_sources(self):
        """간단한 데이터 소스 통합 테스트"""
        unifier = DataUnifier()
        
        # 테스트 데이터 준비
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ],
            "source2": [
                {"id": 3, "name": "Charlie", "age": 35},
                {"id": 4, "name": "David", "age": 40}
            ]
        }
        
        # 데이터 통합
        unified_data = unifier.unify_data_sources(data_sources)
        
        # 결과 검증
        assert unified_data.num_rows == 4
        assert unified_data.num_columns == 3
        assert "id" in unified_data.column_names
        assert "name" in unified_data.column_names
        assert "age" in unified_data.column_names

    def test_unify_with_schema_mapping(self):
        """스키마 매핑을 사용한 데이터 통합 테스트"""
        unifier = DataUnifier()
        
        # 테스트 데이터 준비
        data_sources = {
            "source1": [
                {"user_id": 1, "user_name": "Alice", "user_age": 25},
                {"user_id": 2, "user_name": "Bob", "user_age": 30}
            ],
            "source2": [
                {"id": 3, "name": "Charlie", "age": 35},
                {"id": 4, "name": "David", "age": 40}
            ]
        }
        
        # 스키마 매핑 정의
        schema_mapping = {
            "source1": {
                "user_id": "id",
                "user_name": "name",
                "user_age": "age"
            }
        }
        
        # 데이터 통합
        unified_data = unifier.unify_data_sources(data_sources, schema_mapping)
        
        # 결과 검증
        assert unified_data.num_rows == 4
        assert unified_data.num_columns == 3
        assert "id" in unified_data.column_names
        assert "name" in unified_data.column_names
        assert "age" in unified_data.column_names

    def test_validate_unified_data(self):
        """통합된 데이터 검증 테스트"""
        unifier = DataUnifier()
        
        # 테스트 데이터 준비
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ]
        }
        
        # 데이터 통합
        unified_data = unifier.unify_data_sources(data_sources)
        
        # 데이터 검증
        validation = unifier.validate_unified_data(unified_data)
        
        # 결과 검증
        assert validation["is_valid"] is True
        assert validation["total_rows"] == 2
        assert validation["total_columns"] == 3
        assert "data_quality_score" in validation


class TestSchemaMapper:
    """SchemaMapper 테스트 클래스"""

    def test_schema_mapper_initialization(self):
        """SchemaMapper 초기화 테스트"""
        mapper = SchemaMapper()
        assert mapper is not None

    def test_create_mapping(self):
        """스키마 매핑 생성 테스트"""
        mapper = SchemaMapper()
        
        # 원본 스키마 정의
        source_schema = pa.schema([
            pa.field("user_id", pa.int64()),
            pa.field("user_name", pa.string()),
            pa.field("user_age", pa.int64())
        ])
        
        # 매핑 규칙 정의
        mapping_rules = {
            "user_id": "id",
            "user_name": "name",
            "user_age": "age"
        }
        
        # 스키마 매핑 생성
        schema_mapping = mapper.create_mapping("test_source", source_schema, None, mapping_rules)
        
        # 결과 검증
        assert schema_mapping.source_name == "test_source"
        assert len(schema_mapping.column_mappings) == 3
        assert schema_mapping.column_mappings[0].source_column == "user_id"
        assert schema_mapping.column_mappings[0].target_column == "id"

    def test_apply_mapping(self):
        """스키마 매핑 적용 테스트"""
        mapper = SchemaMapper()
        
        # 원본 스키마 정의
        source_schema = pa.schema([
            pa.field("user_id", pa.int64()),
            pa.field("user_name", pa.string()),
            pa.field("user_age", pa.int64())
        ])
        
        # 매핑 규칙 정의
        mapping_rules = {
            "user_id": "id",
            "user_name": "name",
            "user_age": "age"
        }
        
        # 스키마 매핑 생성
        schema_mapping = mapper.create_mapping("test_source", source_schema, None, mapping_rules)
        
        # 테스트 데이터
        test_data = [
            {"user_id": 1, "user_name": "Alice", "user_age": 25},
            {"user_id": 2, "user_name": "Bob", "user_age": 30}
        ]
        
        # 매핑 적용
        mapped_table = mapper.apply_mapping("test_source", test_data)
        
        # 결과 검증
        assert len(mapped_table) == 2
        assert "id" in mapped_table.column_names
        assert "name" in mapped_table.column_names
        assert "age" in mapped_table.column_names


class TestTypeConverter:
    """TypeConverter 테스트 클래스"""

    def test_type_converter_initialization(self):
        """TypeConverter 초기화 테스트"""
        converter = TypeConverter()
        assert converter is not None
        assert len(converter.get_conversion_rules()) > 0

    def test_convert_column_types(self):
        """컬럼 타입 변환 테스트"""
        converter = TypeConverter()
        
        # 테스트 데이터
        test_data = [
            {"id": "1", "name": "Alice", "age": "25", "active": "true"},
            {"id": "2", "name": "Bob", "age": "30", "active": "false"}
        ]
        
        table = pa.Table.from_pylist(test_data)
        
        # 대상 스키마 정의
        target_schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("age", pa.int64()),
            pa.field("active", pa.bool_())
        ])
        
        # 타입 변환
        converted_table = converter.convert_table(table, target_schema)
        
        # 결과 검증
        assert len(converted_table) == 2
        assert converted_table.schema.field("id").type == pa.int64()
        assert converted_table.schema.field("name").type == pa.string()
        assert converted_table.schema.field("age").type == pa.int64()
        assert converted_table.schema.field("active").type == pa.bool_()

    def test_add_custom_conversion_rule(self):
        """사용자 정의 변환 규칙 추가 테스트"""
        converter = TypeConverter()
        
        # 사용자 정의 변환 규칙 추가
        from bridge.analytics.core import ConversionRule
        custom_rule = ConversionRule(
            source_type="string",
            target_type="int64",
            conversion_function="cast",
            parameters={}
        )
        
        converter.add_conversion_rule(custom_rule)
        
        # 변환 규칙 확인
        rules = converter.get_conversion_rules()
        assert "string->int64" in rules


class TestStreamingProcessor:
    """StreamingProcessor 테스트 클래스"""

    def test_streaming_processor_initialization(self):
        """StreamingProcessor 초기화 테스트"""
        processor = StreamingProcessor(chunk_size=1000, memory_limit_mb=100)
        assert processor is not None
        assert processor.chunk_size == 1000
        assert processor.memory_limit_mb == 100

    def test_process_large_table(self):
        """대용량 테이블 처리 테스트"""
        processor = StreamingProcessor(chunk_size=2, memory_limit_mb=1)
        
        # 테스트 데이터 생성
        test_data = [
            {"id": i, "name": f"User{i}", "age": 20 + i}
            for i in range(10)
        ]
        
        table = pa.Table.from_pylist(test_data)
        
        # 처리 함수 정의
        def process_func(table_chunk, **kwargs):
            return table_chunk
        
        # 스트리밍 처리
        processed_table = processor.process_large_table(table, process_func)
        
        # 결과 검증
        assert len(processed_table) == 10
        assert len(processed_table.schema) == 3

    def test_get_memory_usage(self):
        """메모리 사용량 분석 테스트"""
        processor = StreamingProcessor()
        
        # 테스트 데이터
        test_data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30}
        ]
        
        table = pa.Table.from_pylist(test_data)
        memory_usage = processor.get_memory_usage(table)
        
        # 결과 검증
        assert "total_size" in memory_usage
        assert "row_count" in memory_usage
        assert "column_count" in memory_usage
        assert memory_usage["row_count"] == 2
        assert memory_usage["column_count"] == 3


class TestIntegratedDataLayer:
    """IntegratedDataLayer 테스트 클래스"""

    def test_integrated_data_layer_initialization(self):
        """IntegratedDataLayer 초기화 테스트"""
        layer = IntegratedDataLayer(chunk_size=1000, memory_limit_mb=100)
        assert layer is not None
        assert layer.data_unifier is not None
        assert layer.schema_mapper is not None
        assert layer.type_converter is not None
        assert layer.streaming_processor is not None

    def test_integrate_data_sources(self):
        """데이터 소스 통합 테스트"""
        layer = IntegratedDataLayer()
        
        # 테스트 데이터 준비
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ],
            "source2": [
                {"id": 3, "name": "Charlie", "age": 35},
                {"id": 4, "name": "David", "age": 40}
            ]
        }
        
        # 데이터 통합
        unified_data = layer.integrate_data_sources(data_sources)
        
        # 결과 검증
        assert unified_data.num_rows == 4
        assert unified_data.num_columns == 3
        assert "id" in unified_data.column_names
        assert "name" in unified_data.column_names
        assert "age" in unified_data.column_names

    def test_apply_data_transformations(self):
        """데이터 변환 적용 테스트"""
        layer = IntegratedDataLayer()
        
        # 먼저 데이터 통합
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ]
        }
        
        unified_data = layer.integrate_data_sources(data_sources)
        
        # 변환 규칙 정의
        transformations = [
            {
                "type": "column_rename",
                "column_mapping": {"id": "user_id", "name": "user_name"}
            }
        ]
        
        # 변환 적용
        transformed_data = layer.apply_data_transformations(transformations)
        
        # 결과 검증
        assert transformed_data.num_rows == 2
        assert "user_id" in transformed_data.column_names
        assert "user_name" in transformed_data.column_names

    def test_export_data(self):
        """데이터 내보내기 테스트"""
        layer = IntegratedDataLayer()
        
        # 먼저 데이터 통합
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ]
        }
        
        unified_data = layer.integrate_data_sources(data_sources)
        
        # Arrow 형식으로 내보내기
        exported_data = layer.export_data("arrow")
        
        # 결과 검증
        assert isinstance(exported_data, pa.Table)
        assert len(exported_data) == 2

    def test_validate_data_quality(self):
        """데이터 품질 검증 테스트"""
        layer = IntegratedDataLayer()
        
        # 먼저 데이터 통합
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ]
        }
        
        unified_data = layer.integrate_data_sources(data_sources)
        
        # 데이터 품질 검증
        validation = layer.validate_data_quality()
        
        # 결과 검증
        assert "is_valid" in validation
        assert "total_rows" in validation
        assert "total_columns" in validation
        assert "data_quality_score" in validation

    def test_get_data_summary(self):
        """데이터 요약 생성 테스트"""
        layer = IntegratedDataLayer()
        
        # 먼저 데이터 통합
        data_sources = {
            "source1": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30}
            ]
        }
        
        unified_data = layer.integrate_data_sources(data_sources)
        
        # 데이터 요약 생성
        summary = layer.get_data_summary()
        
        # 결과 검증
        assert "basic_info" in summary
        assert "column_types" in summary
        assert "source_info" in summary
        assert "memory_usage" in summary


if __name__ == "__main__":
    pytest.main([__file__])
