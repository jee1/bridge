"""커넥터 단위 테스트."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from bridge.connectors.base import BaseConnector
from bridge.connectors.mock import MockConnector
from bridge.connectors.postgres import PostgresConnector
from bridge.connectors.registry import ConnectorRegistry, ConnectorNotFoundError


class TestMockConnector:
    """MockConnector 테스트."""
    
    def test_init_with_default_values(self):
        """기본값으로 초기화 테스트."""
        connector = MockConnector()
        assert connector.name == "mock"
        assert connector.settings == {}
        assert len(connector._sample_rows) == 2
    
    def test_init_with_custom_values(self):
        """사용자 정의 값으로 초기화 테스트."""
        custom_rows = [{"id": 1, "name": "test"}]
        connector = MockConnector(
            name="custom_mock",
            settings={"test": "value"},
            sample_rows=custom_rows
        )
        assert connector.name == "custom_mock"
        assert connector.settings == {"test": "value"}
        assert connector._sample_rows == custom_rows
    
    def test_test_connection(self):
        """연결 테스트."""
        connector = MockConnector()
        assert connector.test_connection() is True
    
    def test_get_metadata(self):
        """메타데이터 조회 테스트."""
        connector = MockConnector()
        metadata = connector.get_metadata()
        
        assert metadata["name"] == "mock"
        assert metadata["type"] == "mock"
        assert "fields" in metadata
        assert len(metadata["fields"]) == 2
    
    def test_run_query(self):
        """쿼리 실행 테스트."""
        connector = MockConnector()
        results = list(connector.run_query("SELECT * FROM test"))
        
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["metric"] == 42
        assert results[1]["id"] == 2
        assert results[1]["metric"] == 17
    
    def test_run_query_with_params(self):
        """파라미터가 있는 쿼리 실행 테스트."""
        connector = MockConnector()
        results = list(connector.run_query("SELECT * FROM test", {"param": "value"}))
        
        assert len(results) == 2
        # 파라미터는 무시되고 샘플 데이터가 반환됨
        assert results[0]["id"] == 1


class TestPostgresConnector:
    """PostgresConnector 테스트."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool."""
        pool = AsyncMock()
        connection = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = connection
        pool.acquire.return_value.__aexit__.return_value = None
        return pool, connection
    
    @pytest.fixture
    def connector(self):
        """PostgresConnector 인스턴스."""
        return PostgresConnector(
            name="test_postgres",
            settings={"host": "localhost", "port": 5432}
        )
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, connector, mock_pool):
        """연결 테스트 성공 케이스."""
        pool, connection = mock_pool
        connector._get_pool = AsyncMock(return_value=pool)
        
        result = await connector.test_connection()
        assert result is True
        connection.execute.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, connector, mock_pool):
        """메타데이터 조회 테스트."""
        pool, connection = mock_pool
        connector._get_pool = AsyncMock(return_value=pool)
        
        # Mock cursor results
        mock_record = {"table_schema": "public", "table_name": "test", "column_name": "id", "data_type": "integer"}
        connection.cursor.return_value.__aiter__.return_value = [mock_record]
        
        metadata = await connector.get_metadata()
        
        assert "columns" in metadata
        assert len(metadata["columns"]) == 1
        assert metadata["columns"][0]["table_schema"] == "public"
    
    @pytest.mark.asyncio
    async def test_run_query(self, connector, mock_pool):
        """쿼리 실행 테스트."""
        pool, connection = mock_pool
        connector._get_pool = AsyncMock(return_value=pool)
        
        # Mock cursor results
        mock_record = {"id": 1, "name": "test"}
        connection.cursor.return_value.__aiter__.return_value = [mock_record]
        
        results = []
        async for record in connector.run_query("SELECT * FROM test", {"param": "value"}):
            results.append(record)
        
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["name"] == "test"


class TestConnectorRegistry:
    """ConnectorRegistry 테스트."""
    
    def test_init(self):
        """레지스트리 초기화 테스트."""
        registry = ConnectorRegistry()
        assert len(registry.list()) == 0
    
    def test_register_connector(self):
        """커넥터 등록 테스트."""
        registry = ConnectorRegistry()
        connector = MockConnector(name="test_connector")
        
        registry.register(connector)
        assert "test_connector" in registry.list()
        assert registry.get("test_connector") == connector
    
    def test_register_duplicate_connector(self):
        """중복 커넥터 등록 테스트."""
        registry = ConnectorRegistry()
        connector1 = MockConnector(name="test_connector")
        connector2 = MockConnector(name="test_connector")
        
        registry.register(connector1)
        registry.register(connector2, overwrite=False)
        
        # overwrite=False이므로 첫 번째 커넥터가 유지됨
        assert registry.get("test_connector") == connector1
    
    def test_register_duplicate_connector_overwrite(self):
        """중복 커넥터 덮어쓰기 테스트."""
        registry = ConnectorRegistry()
        connector1 = MockConnector(name="test_connector")
        connector2 = MockConnector(name="test_connector")
        
        registry.register(connector1)
        registry.register(connector2, overwrite=True)
        
        # overwrite=True이므로 두 번째 커넥터로 교체됨
        assert registry.get("test_connector") == connector2
    
    def test_get_nonexistent_connector(self):
        """존재하지 않는 커넥터 조회 테스트."""
        registry = ConnectorRegistry()
        
        with pytest.raises(ConnectorNotFoundError):
            registry.get("nonexistent")
    
    def test_list_connectors(self):
        """커넥터 목록 조회 테스트."""
        registry = ConnectorRegistry()
        connector1 = MockConnector(name="connector1")
        connector2 = MockConnector(name="connector2")
        
        registry.register(connector1)
        registry.register(connector2)
        
        connectors = registry.list()
        assert len(connectors) == 2
        assert "connector1" in connectors
        assert "connector2" in connectors
