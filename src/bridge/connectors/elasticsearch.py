"""Elasticsearch 커넥터 구현."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable
from urllib.parse import urlparse

from elasticsearch import AsyncElasticsearch

from .base import BaseConnector
from .exceptions import ConfigurationError, ConnectionError, MetadataError, QueryExecutionError

logger = logging.getLogger(__name__)


class ElasticsearchConnector(BaseConnector):
    """Elasticsearch 데이터 접근을 담당한다."""

    def __init__(self, name: str, settings: Dict[str, Any]):
        super().__init__(name, settings)
        self._client: AsyncElasticsearch | None = None

    async def _get_client(self) -> AsyncElasticsearch:
        """Elasticsearch 클라이언트를 생성한다."""
        if self._client is None:
            try:
                configured_url = str(self.settings.get("url", "")).strip()
                configured_host = str(self.settings.get("host", "")).strip()
                configured_port = self.settings.get("port")
                use_ssl = bool(self.settings.get("use_ssl", False))

                if configured_url:
                    parsed = urlparse(configured_url)
                    if not (parsed.scheme and parsed.hostname and parsed.port):
                        raise ConfigurationError(
                            "Elasticsearch URL이 잘못되었습니다. 예: https://localhost:9200"
                        )
                    host_url = configured_url
                    use_ssl = parsed.scheme == "https"
                else:
                    if not configured_host:
                        raise ConfigurationError("Elasticsearch host 설정이 비어있습니다")
                    try:
                        port = int(configured_port)
                    except (TypeError, ValueError):
                        raise ConfigurationError(
                            f"Elasticsearch port 설정이 올바르지 않습니다: {configured_port}"
                        ) from None
                    scheme = "https" if use_ssl else "http"
                    host_url = f"{scheme}://{configured_host}:{port}"

                logger.info(f"Elasticsearch URL 생성: {host_url}")

                client_kwargs: Dict[str, Any] = {
                    "hosts": [host_url],
                    "verify_certs": use_ssl,
                    "request_timeout": 30,
                    "retry_on_timeout": True,
                }

                username = str(self.settings.get("username", "")).strip()
                password = str(self.settings.get("password", "")).strip()
                if username and password:
                    client_kwargs["basic_auth"] = (username, password)

                self._client = AsyncElasticsearch(**client_kwargs)

                logger.info(f"Elasticsearch 클라이언트 생성 성공: {host_url}")

            except Exception as e:
                logger.error(f"Elasticsearch 클라이언트 생성 실패: {e}")
                raise ConnectionError(f"Elasticsearch 클라이언트 생성에 실패했습니다: {e}") from e

        return self._client

    async def test_connection(self) -> bool:  # type: ignore[override]
        """연결을 테스트한다."""
        try:
            client = await self._get_client()
            # ping 대신 info API 사용
            response = await client.info()
            if not response:
                raise ConnectionError("Elasticsearch info 조회 실패")
            logger.info(
                f"Elasticsearch 연결 테스트 성공: {response.get('cluster_name', 'unknown')}"
            )
            return True
        except Exception as e:
            logger.error(f"Elasticsearch 연결 실패: {e}")
            raise ConnectionError(f"Elasticsearch 서버에 연결할 수 없습니다: {e}") from e

    async def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        """메타데이터를 조회한다."""
        try:
            client = await self._get_client()

            # 인덱스 목록 조회
            indices_response = await client.cat.indices(format="json")
            indices = [
                index["index"] for index in indices_response if not index["index"].startswith(".")
            ]

            # 각 인덱스의 매핑 정보 조회
            mappings = {}
            for index in indices:
                try:
                    mapping_response = await client.indices.get_mapping(index=index)
                    mappings[index] = mapping_response[index]["mappings"]
                except Exception as e:
                    logger.warning(f"인덱스 {index}의 매핑 조회 실패: {e}")
                    mappings[index] = {}

            logger.info(f"Elasticsearch 메타데이터 조회 성공: {len(indices)}개 인덱스")
            return {"indices": indices, "mappings": mappings, "total_indices": len(indices)}
        except Exception as e:
            logger.error(f"Elasticsearch 메타데이터 조회 실패: {e}")
            raise MetadataError(f"메타데이터 조회에 실패했습니다: {e}") from e

    async def run_query(  # type: ignore[override]
        self, query: str, params: Dict[str, Any] | None = None
    ) -> Iterable[Dict[str, Any]]:
        """쿼리를 실행한다."""
        params = params or {}
        try:
            # 쿼리 검증
            if not query.strip():
                raise QueryExecutionError("쿼리가 비어있습니다")

            client = await self._get_client()

            # 쿼리 파싱 및 실행
            # Elasticsearch 쿼리는 JSON 형태로 전달되어야 함
            try:
                import json

                query_dict = json.loads(query)
            except json.JSONDecodeError:
                # 단순 문자열 쿼리인 경우 match_all 쿼리로 변환
                query_dict = {"query": {"match": {"_all": query}}}

            # 인덱스 지정 (params에서 가져오거나 모든 인덱스 검색)
            index = params.get("index", "_all")

            logger.info(
                f"Elasticsearch 쿼리 실행: {index} - {query[:100]}{'...' if len(query) > 100 else ''}"
            )

            # 검색 실행
            response = await client.search(
                index=index,
                body=query_dict,
                size=params.get("size", 100),
                from_=params.get("from", 0),
            )

            # 결과 처리
            hits = response.get("hits", {}).get("hits", [])
            for hit in hits:
                # Elasticsearch 결과를 표준화된 형태로 변환
                result = {
                    "_id": hit["_id"],
                    "_index": hit["_index"],
                    "_score": hit.get("_score", 0),
                    **hit["_source"],
                }
                yield result

            logger.info(f"Elasticsearch 쿼리 실행 완료: {len(hits)}개 결과")

        except Exception as e:
            logger.error(f"Elasticsearch 쿼리 실행 실패: {e}")
            raise QueryExecutionError(f"쿼리 실행에 실패했습니다: {e}") from e

    async def close(self):
        """연결을 종료한다."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Elasticsearch 클라이언트 연결 종료")
