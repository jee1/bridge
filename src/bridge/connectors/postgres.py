"""PostgreSQL 커넥터 예시 구현."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict

import asyncpg

from .base import BaseConnector
from .exceptions import ConfigurationError, ConnectionError, MetadataError, QueryExecutionError

logger = logging.getLogger(__name__)


class PostgresConnector(BaseConnector):
    """PostgreSQL 데이터 접근을 담당한다."""

    async def _get_pool(self) -> asyncpg.Pool:
        """연결 풀을 생성한다."""
        try:
            # 필수 설정 검증
            required_settings = ["host", "port", "database", "user", "password"]
            missing_settings = [
                setting for setting in required_settings if setting not in self.settings
            ]
            if missing_settings:
                raise ConfigurationError(f"필수 설정이 누락되었습니다: {missing_settings}")

            logger.info(
                f"PostgreSQL 연결 풀 생성 중: {self.settings.get('host')}:{self.settings.get('port')}"
            )
            return await asyncpg.create_pool(**self.settings)
        except asyncpg.InvalidAuthorizationSpecificationError as e:
            logger.error(f"PostgreSQL 인증 실패: {e}")
            raise ConnectionError(f"데이터베이스 인증에 실패했습니다: {e}") from e
        except asyncpg.InvalidCatalogNameError as e:
            logger.error(f"PostgreSQL 데이터베이스 없음: {e}")
            raise ConnectionError(
                f"데이터베이스 '{self.settings.get('database')}'를 찾을 수 없습니다: {e}"
            ) from e
        except asyncpg.ConnectionDoesNotExistError as e:
            logger.error(f"PostgreSQL 연결 실패: {e}")
            raise ConnectionError(f"데이터베이스 서버에 연결할 수 없습니다: {e}") from e
        except Exception as e:
            logger.error(f"PostgreSQL 연결 풀 생성 실패: {e}")
            raise ConnectionError(f"연결 풀 생성에 실패했습니다: {e}") from e

    async def test_connection(self) -> bool:  # type: ignore[override]
        """연결을 테스트한다."""
        try:
            async with await self._get_pool() as pool:
                async with pool.acquire() as connection:
                    await connection.execute("SELECT 1")
            logger.info("PostgreSQL 연결 테스트 성공")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL 연결 테스트 실패: {e}")
            raise ConnectionError(f"연결 테스트에 실패했습니다: {e}") from e

    async def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        """메타데이터를 조회한다."""
        query = """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position
        """
        try:
            rows = []
            async with await self._get_pool() as pool:
                async with pool.acquire() as connection:
                    async with connection.transaction():
                        async for record in connection.cursor(query):
                            rows.append(dict(record))
            logger.info(f"PostgreSQL 메타데이터 조회 성공: {len(rows)}개 컬럼")
            return {"columns": rows}
        except Exception as e:
            logger.error(f"PostgreSQL 메타데이터 조회 실패: {e}")
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

            # SQL 인젝션 방지를 위한 안전한 파라미터 바인딩
            # asyncpg는 $1, $2 형태의 플레이스홀더를 사용하여 파라미터를 안전하게 바인딩
            # 파라미터는 순서대로 전달되어야 함
            param_values = list(params.values())

            logger.info(f"PostgreSQL 쿼리 실행: {query[:100]}{'...' if len(query) > 100 else ''}")

            async with await self._get_pool() as pool:
                async with pool.acquire() as connection:
                    async with connection.transaction():
                        async for record in connection.cursor(query, *param_values):
                            yield dict(record)

            logger.info("PostgreSQL 쿼리 실행 완료")
        except asyncpg.PostgresSyntaxError as e:
            logger.error(f"PostgreSQL 구문 오류: {e}")
            raise QueryExecutionError(f"SQL 구문 오류: {e}") from e
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL 실행 오류: {e}")
            raise QueryExecutionError(f"쿼리 실행 오류: {e}") from e
        except Exception as e:
            logger.error(f"PostgreSQL 쿼리 실행 실패: {e}")
            raise QueryExecutionError(f"쿼리 실행에 실패했습니다: {e}") from e
