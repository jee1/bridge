"""Elasticsearch 커넥터 단위 테스트."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import json

from bridge.connectors.elasticsearch import ElasticsearchConnector
from bridge.connectors.exceptions import ConfigurationError, ConnectionError, MetadataError, QueryExecutionError


class TestElasticsearchConnector:
    """ElasticsearchConnector 테스트."""

    def test_init_with_required_settings(self):
        """필수 설정으로 초기화 테스트."""
        connector = ElasticsearchConnector(
            name="test_elasticsearch",
            settings={
                "host": "localhost",
                "port": 9200,
                "use_ssl": False
            }
        )
        assert connector.name == "test_elasticsearch"
        assert connector.settings["host"] == "localhost"
        assert connector.settings["port"] == 9200
        assert connector.settings["use_ssl"] is False

    def test_init_missing_required_settings(self):
        """필수 설정 누락 테스트."""
        with pytest.raises(ConfigurationError, match="필수 설정이 누락되었습니다"):
            ElasticsearchConnector(
                name="test",
                settings={"host": "localhost"}  # port 누락
            )

    def test_validate_port_string(self):
        """문자열 포트 검증 테스트."""
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": "9200"}
        )
        assert connector._validate_port("9200") == 9200

    def test_validate_port_integer(self):
        """정수 포트 검증 테스트."""
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        assert connector._validate_port(9200) == 9200

    def test_validate_port_invalid(self):
        """잘못된 포트 검증 테스트."""
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        with pytest.raises(ConfigurationError, match="Elasticsearch port 설정이 올바르지 않습니다"):
            connector._validate_port("invalid")

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_get_client_success(self, mock_elasticsearch):
        """클라이언트 생성 성공 테스트."""
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        client = await connector._get_client()
        assert client == mock_client
        mock_elasticsearch.assert_called_once()

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_get_client_with_auth(self, mock_elasticsearch):
        """인증 정보가 있는 클라이언트 생성 테스트."""
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={
                "host": "localhost",
                "port": 9200,
                "username": "user",
                "password": "pass"
            }
        )
        
        await connector._get_client()
        
        # basic_auth가 포함된 호출 확인
        call_args = mock_elasticsearch.call_args
        assert "basic_auth" in call_args.kwargs
        assert call_args.kwargs["basic_auth"] == ("user", "pass")

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_test_connection_success(self, mock_elasticsearch):
        """연결 테스트 성공."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        result = await connector.test_connection()
        assert result is True
        mock_client.ping.assert_called_once()

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_test_connection_failure(self, mock_elasticsearch):
        """연결 테스트 실패."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = False
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        with pytest.raises(ConnectionError, match="Elasticsearch ping 실패"):
            await connector.test_connection()

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_get_metadata_success(self, mock_elasticsearch):
        """메타데이터 조회 성공 테스트."""
        mock_client = AsyncMock()
        
        # cat.indices 응답 모킹
        mock_client.cat.indices.return_value = [
            {"index": "test_index_1"},
            {"index": "test_index_2"},
            {"index": ".system_index"}  # 시스템 인덱스는 제외되어야 함
        ]
        
        # get_mapping 응답 모킹
        mock_client.indices.get_mapping.return_value = {
            "test_index_1": {
                "mappings": {
                    "properties": {
                        "field1": {"type": "text"},
                        "field2": {"type": "keyword"}
                    }
                }
            },
            "test_index_2": {
                "mappings": {
                    "properties": {
                        "field3": {"type": "integer"}
                    }
                }
            }
        }
        
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        metadata = await connector.get_metadata()
        
        assert "indices" in metadata
        assert "mappings" in metadata
        assert "total_indices" in metadata
        assert len(metadata["indices"]) == 2
        assert "test_index_1" in metadata["indices"]
        assert "test_index_2" in metadata["indices"]
        assert ".system_index" not in metadata["indices"]
        assert metadata["total_indices"] == 2

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_run_query_success(self, mock_elasticsearch):
        """쿼리 실행 성공 테스트."""
        mock_client = AsyncMock()
        
        # search 응답 모킹
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_index": "test_index",
                        "_score": 1.0,
                        "_source": {"field1": "value1", "field2": "value2"}
                    },
                    {
                        "_id": "2",
                        "_index": "test_index",
                        "_score": 0.8,
                        "_source": {"field1": "value3", "field2": "value4"}
                    }
                ]
            }
        }
        
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        query = '{"query": {"match_all": {}}}'
        results = list(await connector.run_query(query).__anext__())
        
        # 첫 번째 결과 확인
        assert results["_id"] == "1"
        assert results["_index"] == "test_index"
        assert results["_score"] == 1.0
        assert results["field1"] == "value1"
        assert results["field2"] == "value2"

    @patch('bridge.connectors.elasticsearch.AsyncElasticsearch')
    async def test_run_query_with_params(self, mock_elasticsearch):
        """파라미터가 있는 쿼리 실행 테스트."""
        mock_client = AsyncMock()
        mock_client.search.return_value = {"hits": {"hits": []}}
        mock_elasticsearch.return_value = mock_client
        
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        query = '{"query": {"match_all": {}}}'
        params = {"index": "specific_index", "size": 50, "from": 10}
        
        await connector.run_query(query, params).__anext__()
        
        # search 호출 파라미터 확인
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs["index"] == "specific_index"
        assert call_args.kwargs["size"] == 50
        assert call_args.kwargs["from_"] == 10

    async def test_run_query_empty_query(self):
        """빈 쿼리 실행 테스트."""
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        with pytest.raises(QueryExecutionError, match="쿼리가 비어있습니다"):
            await connector.run_query("").__anext__()

    async def test_run_query_invalid_json(self):
        """잘못된 JSON 쿼리 테스트."""
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        # 잘못된 JSON은 match_all 쿼리로 변환되어야 함
        with patch('bridge.connectors.elasticsearch.AsyncElasticsearch') as mock_elasticsearch:
            mock_client = AsyncMock()
            mock_client.search.return_value = {"hits": {"hits": []}}
            mock_elasticsearch.return_value = mock_client
            
            await connector.run_query("invalid json").__anext__()
            
            # match_all 쿼리로 변환되었는지 확인
            call_args = mock_client.search.call_args
            query_dict = call_args.kwargs["body"]
            assert "query" in query_dict
            assert "match" in query_dict["query"]

    async def test_close(self):
        """연결 종료 테스트."""
        connector = ElasticsearchConnector(
            name="test",
            settings={"host": "localhost", "port": 9200}
        )
        
        # 클라이언트가 설정되지 않은 상태에서 close 호출
        await connector.close()
        assert connector._client is None
        
        # 클라이언트가 설정된 상태에서 close 호출
        with patch('bridge.connectors.elasticsearch.AsyncElasticsearch') as mock_elasticsearch:
            mock_client = AsyncMock()
            mock_elasticsearch.return_value = mock_client
            
            await connector._get_client()
            await connector.close()
            
            mock_client.close.assert_called_once()
            assert connector._client is None