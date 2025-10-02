"""Elasticsearch 커넥터 구현."""

from __future__ import annotations

import logging
import numbers
from typing import Any, AsyncIterator, Dict, List, cast

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
                # 필수 설정 검증
                required_settings = ["host", "port"]
                missing_settings = [
                    setting for setting in required_settings if setting not in self.settings
                ]
                if missing_settings:
                    raise ConfigurationError(f"필수 설정이 누락되었습니다: {missing_settings}")

                # Elasticsearch 클라이언트 설정
                scheme = "https" if self.settings.get("use_ssl", False) else "http"
                port = self._validate_port(self.settings.get("port"))
                host_url = f"{scheme}://{self.settings['host']}:{port}"
                logger.info(f"Elasticsearch URL 생성: {host_url}")

                # AsyncElasticsearch 클라이언트 생성
                if self.settings.get("username") and self.settings.get("password"):
                    self._client = AsyncElasticsearch(
                        hosts=[host_url],
                        basic_auth=(self.settings["username"], self.settings["password"]),
                        verify_certs=self.settings.get("use_ssl", False),
                        request_timeout=30,
                        retry_on_timeout=True,
                    )
                else:
                    self._client = AsyncElasticsearch(
                        hosts=[host_url],
                        verify_certs=self.settings.get("use_ssl", False),
                        request_timeout=30,
                        retry_on_timeout=True,
                    )

                logger.info(f"Elasticsearch 클라이언트 생성 성공: {host_url}")

            except Exception as e:
                logger.error(f"Elasticsearch 클라이언트 생성 실패: {e}")
                raise ConnectionError(f"Elasticsearch 클라이언트 생성에 실패했습니다: {e}") from e

        return self._client

    async def test_connection(self) -> bool:  # type: ignore[override]
        """연결을 테스트한다."""
        try:
            client = await self._get_client()
            response = await client.ping()
            if not response:
                raise ConnectionError("Elasticsearch ping 실패")
            logger.info("Elasticsearch 연결 테스트 성공")
            return True
        except Exception as e:
            logger.error(f"Elasticsearch 연결 실패: {e}")
            raise ConnectionError(f"Elasticsearch 서버에 연결할 수 없습니다: {e}") from e

    async def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        """메타데이터를 조회한다."""
        try:
            client = await self._get_client()

            # 인덱스 목록 조회
            indices_response_raw = await client.cat.indices(format="json")
            indices_response = cast(List[Dict[str, Any]], indices_response_raw)
            indices = [
                index_info["index"]
                for index_info in indices_response
                if isinstance(index_info, dict)
                and isinstance(index_info.get("index"), str)
                and not index_info["index"].startswith(".")
            ]

            # 각 인덱스의 매핑 정보 조회
            mappings: Dict[str, Dict[str, Any]] = {}
            for index_name in indices:
                try:
                    mapping_response_raw = await client.indices.get_mapping(index=index_name)
                    mapping_response = cast(Dict[str, Dict[str, Any]], mapping_response_raw)
                    index_mapping = mapping_response.get(index_name, {})
                    mapping_data = {}
                    if isinstance(index_mapping, dict):
                        mapping_entry = index_mapping.get("mappings", {})
                        if isinstance(mapping_entry, dict):
                            mapping_data = mapping_entry
                    mappings[index_name] = mapping_data
                except Exception as e:
                    logger.warning(f"인덱스 {index_name}의 매핑 조회 실패: {e}")
                    mappings[index_name] = {}

            logger.info(f"Elasticsearch 메타데이터 조회 성공: {len(indices)}개 인덱스")
            return {"indices": indices, "mappings": mappings, "total_indices": len(indices)}
        except Exception as e:
            logger.error(f"Elasticsearch 메타데이터 조회 실패: {e}")
            raise MetadataError(f"메타데이터 조회에 실패했습니다: {e}") from e

    async def run_query(  # type: ignore[override]
        self, query: str, params: Dict[str, Any] | None = None
    ) -> AsyncIterator[Dict[str, Any]]:
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
            index_param = params.get("index", "_all")
            index = str(index_param)

            logger.info(
                f"Elasticsearch 쿼리 실행: {index} - {query[:100]}{'...' if len(query) > 100 else ''}"
            )

            size_param = params.get("size", 100)
            from_param = params.get("from", 0)
            try:
                size = int(size_param)
                from_value = int(from_param)
            except (TypeError, ValueError) as exc:
                raise QueryExecutionError("size/from 파라미터는 정수여야 합니다") from exc

            # 검색 실행
            response = await client.search(
                index=index,
                body=query_dict,
                size=size,
                from_=from_value,
            )

            # 결과 처리
            hits_container = response.get("hits", {})
            hits = cast(List[Dict[str, Any]], hits_container.get("hits", []))
            for hit in hits:
                # Elasticsearch 결과를 표준화된 형태로 변환
                source = cast(Dict[str, Any], hit.get("_source", {}))
                result = {
                    "_id": hit["_id"],
                    "_index": hit["_index"],
                    "_score": hit.get("_score", 0),
                    **source,
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

    def _validate_port(self, raw_port: Any) -> int:
        """설정에서 전달된 포트를 정수로 검증한다."""
        if isinstance(raw_port, str):
            port_str = raw_port.strip()
            if not port_str.isdigit():
                raise ConfigurationError(f"Elasticsearch port 설정이 올바르지 않습니다: {raw_port}")
            return int(port_str)
        if isinstance(raw_port, numbers.Integral):
            return int(raw_port)
        raise ConfigurationError(f"Elasticsearch port 설정이 올바르지 않습니다: {raw_port}")