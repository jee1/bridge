"""커넥터 단위 테스트."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bridge.connectors.base import BaseConnector
from bridge.connectors.mock import MockConnector
from bridge.connectors.postgres import PostgresConnector
from bridge.connectors.registry import ConnectorNotFoundError, ConnectorRegistry


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
            name="custom_mock", settings={"test": "value"}, sample_rows=custom_rows
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
        results = list(connector.run_query("SELECT * FROM dummy"))

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["metric"] == 42
        assert results[1]["id"] == 2
        assert results[1]["metric"] == 17

    def test_mask_columns(self):
        """컬럼 마스킹 테스트."""
        connector = MockConnector()
        rows = [{"id": 1, "name": "Alice", "email": "alice@example.com"}]
        masked_rows = list(connector.mask_columns(rows, ["email"]))

        assert masked_rows[0]["email"] == "***"
        assert masked_rows[0]["name"] == "Alice"
        assert masked_rows[0]["id"] == 1


class TestPostgresConnector:
    """PostgresConnector 테스트."""

    def test_connector_initialization(self):
        """PostgresConnector 초기화 테스트."""
        connector = PostgresConnector(
            name="test_postgres",
            settings={
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "user": "test",
                "password": "test",
            },
        )
        assert connector.name == "test_postgres"
        assert connector.settings["host"] == "localhost"
        assert connector.settings["port"] == 5432

    def test_connector_missing_settings(self):
        """필수 설정 누락 테스트."""
        connector = PostgresConnector(name="test", settings={"host": "localhost"})
        # _get_pool 메서드가 ConfigurationError를 발생시키는지 확인
        import asyncio

        async def test_missing_settings():
            try:
                await connector._get_pool()
                return False
            except Exception as e:
                return "필수 설정이 누락되었습니다" in str(e)

        result = asyncio.run(test_missing_settings())
        assert result is True


class TestConnectorRegistry:
    """ConnectorRegistry 테스트."""

    def test_register_and_get_connector(self):
        """커넥터 등록 및 조회 테스트."""
        registry = ConnectorRegistry()
        connector = MockConnector(name="test_connector")

        registry.register(connector)
        retrieved = registry.get("test_connector")

        assert retrieved is connector
        assert retrieved.name == "test_connector"

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
