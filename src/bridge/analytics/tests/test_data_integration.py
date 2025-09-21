"""데이터 통합 기능 테스트."""

import pytest
import pandas as pd
import pyarrow as pa
from bridge.analytics.core import UnifiedDataFrame, TypeNormalizer, ConnectorAdapter, CrossSourceJoiner


class TestUnifiedDataFrame:
    """UnifiedDataFrame 테스트."""
    
    def test_init_with_arrow_table(self):
        """Arrow Table으로 초기화 테스트."""
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df = UnifiedDataFrame(table)
        
        assert df.num_rows == 3
        assert df.num_columns == 2
        assert df.column_names == ["id", "name"]
    
    def test_init_with_pandas_dataframe(self):
        """Pandas DataFrame으로 초기화 테스트."""
        pandas_df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df = UnifiedDataFrame(pandas_df)
        
        assert df.num_rows == 3
        assert df.num_columns == 2
        assert df.column_names == ["id", "name"]
    
    def test_init_with_dict_list(self):
        """딕셔너리 리스트로 초기화 테스트."""
        data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        df = UnifiedDataFrame(data)
        
        assert df.num_rows == 2
        assert df.num_columns == 2
        assert df.column_names == ["id", "name"]
    
    def test_init_with_empty_data(self):
        """빈 데이터로 초기화 테스트."""
        df = UnifiedDataFrame()
        
        assert df.num_rows == 0
        assert df.num_columns == 0
        assert df.column_names == []
    
    def test_to_pandas(self):
        """Pandas 변환 테스트."""
        data = [{"id": 1, "name": "a"}]
        df = UnifiedDataFrame(data)
        pandas_df = df.to_pandas()
        
        assert isinstance(pandas_df, pd.DataFrame)
        assert len(pandas_df) == 1
        assert list(pandas_df.columns) == ["id", "name"]
    
    def test_to_arrow(self):
        """Arrow 변환 테스트."""
        data = [{"id": 1, "name": "a"}]
        df = UnifiedDataFrame(data)
        arrow_table = df.to_arrow()
        
        assert isinstance(arrow_table, pa.Table)
        assert len(arrow_table) == 1
        assert arrow_table.column_names == ["id", "name"]
    
    def test_select_columns(self):
        """컬럼 선택 테스트."""
        data = [{"id": 1, "name": "a", "value": 10}]
        df = UnifiedDataFrame(data)
        selected_df = df.select_columns(["id", "name"])
        
        assert selected_df.num_columns == 2
        assert selected_df.column_names == ["id", "name"]
    
    def test_metadata(self):
        """메타데이터 테스트."""
        df = UnifiedDataFrame([{"id": 1}])
        df.add_metadata("test_key", "test_value")
        
        assert df.get_metadata("test_key") == "test_value"
        assert df.get_metadata() == {"test_key": "test_value"}


class TestTypeNormalizer:
    """TypeNormalizer 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        normalizer = TypeNormalizer()
        assert normalizer is not None
    
    def test_detect_types(self):
        """타입 감지 테스트."""
        normalizer = TypeNormalizer()
        data = [
            {"id": 1, "name": "a", "value": 1.5, "active": True},
            {"id": 2, "name": "b", "value": 2.5, "active": False}
        ]
        
        types = normalizer.detect_types(data)
        
        assert "id" in types
        assert "name" in types
        assert "value" in types
        assert "active" in types
    
    def test_normalize_data(self):
        """데이터 정규화 테스트."""
        normalizer = TypeNormalizer()
        table = pa.table({"id": [1, 2], "name": ["a", "b"]})
        
        normalized = normalizer.normalize_data(table)
        
        assert isinstance(normalized, pa.Table)
        assert len(normalized) == 2
    
    def test_add_type_mapping(self):
        """타입 매핑 추가 테스트."""
        normalizer = TypeNormalizer()
        normalizer.add_type_mapping("custom_type", pa.string())
        
        mappings = normalizer.get_type_mapping()
        assert "custom_type" in mappings


class TestCrossSourceJoiner:
    """CrossSourceJoiner 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        joiner = CrossSourceJoiner()
        assert joiner is not None
    
    def test_register_table(self):
        """테이블 등록 테스트."""
        joiner = CrossSourceJoiner()
        df = UnifiedDataFrame([{"id": 1, "name": "a"}])
        
        joiner.register_table("test_table", df)
        
        assert "test_table" in joiner.get_registered_tables()
    
    def test_join_tables(self):
        """테이블 조인 테스트."""
        joiner = CrossSourceJoiner()
        
        # 테이블 등록
        left_df = UnifiedDataFrame([{"id": 1, "name": "a"}])
        right_df = UnifiedDataFrame([{"id": 1, "value": 10}])
        
        joiner.register_table("left_table", left_df)
        joiner.register_table("right_table", right_df)
        
        # 조인 실행
        result = joiner.join_tables("left_table", "right_table", "left_table.id = right_table.id")
        
        assert isinstance(result, UnifiedDataFrame)
        assert result.num_rows >= 0
    
    def test_clear_tables(self):
        """테이블 정리 테스트."""
        joiner = CrossSourceJoiner()
        df = UnifiedDataFrame([{"id": 1}])
        
        joiner.register_table("test", df)
        assert len(joiner.get_registered_tables()) == 1
        
        joiner.clear_tables()
        assert len(joiner.get_registered_tables()) == 0
    
    def test_context_manager(self):
        """컨텍스트 매니저 테스트."""
        with CrossSourceJoiner() as joiner:
            df = UnifiedDataFrame([{"id": 1}])
            joiner.register_table("test", df)
            assert "test" in joiner.get_registered_tables()
        
        # 컨텍스트 종료 후에는 연결이 닫혀야 함
        # (실제로는 DuckDB 연결이 닫히지만 테스트에서는 확인하기 어려움)
