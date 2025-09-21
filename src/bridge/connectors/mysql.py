"""MySQL 커넥터 구현."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict

import aiomysql

from .base import BaseConnector
from .exceptions import ConfigurationError, ConnectionError, MetadataError, QueryExecutionError

logger = logging.getLogger(__name__)


class MySQLConnector(BaseConnector):
    """MySQL 데이터 접근을 담당한다."""

    async def _get_pool(self) -> aiomysql.Pool:
        """연결 풀을 생성한다."""
        try:
            # 필수 설정 검증
            required_settings = ["host", "port", "user", "password", "db"]
            missing_settings = [
                setting for setting in required_settings if setting not in self.settings
            ]
            if missing_settings:
                raise ConfigurationError(f"필수 설정이 누락되었습니다: {missing_settings}")

            logger.info(
                f"MySQL 연결 풀 생성 중: {self.settings.get('host')}:{self.settings.get('port')}"
            )

            # aiomysql 설정
            pool_config = {
                "host": self.settings["host"],
                "port": self.settings["port"],
                "user": self.settings["user"],
                "password": self.settings["password"],
                "db": self.settings["db"],
                "minsize": 1,
                "maxsize": 10,
                "autocommit": True,
                "charset": "utf8mb4",
                "use_unicode": True,
            }

            return await aiomysql.create_pool(**pool_config)
        except aiomysql.OperationalError as e:
            logger.error(f"MySQL 연결 실패: {e}")
            raise ConnectionError(f"MySQL 서버에 연결할 수 없습니다: {e}") from e
        except aiomysql.ProgrammingError as e:
            logger.error(f"MySQL 인증 실패: {e}")
            raise ConnectionError(f"MySQL 인증에 실패했습니다: {e}") from e
        except Exception as e:
            logger.error(f"MySQL 연결 풀 생성 실패: {e}")
            raise ConnectionError(f"연결 풀 생성에 실패했습니다: {e}") from e

    async def test_connection(self) -> bool:  # type: ignore[override]
        """연결을 테스트한다."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    if result[0] != 1:
                        raise ConnectionError("연결 테스트 쿼리 결과가 예상과 다릅니다")
            pool.close()
            await pool.wait_closed()
            logger.info("MySQL 연결 테스트 성공")
            return True
        except Exception as e:
            logger.error(f"MySQL 연결 테스트 실패: {e}")
            raise ConnectionError(f"연결 테스트에 실패했습니다: {e}") from e

    async def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        """메타데이터를 조회한다."""
        query = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COLUMN_KEY,
            EXTRA
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        try:
            pool = await self._get_pool()
            rows = []
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, (self.settings["db"],))
                    async for row in cursor:
                        rows.append(dict(row))
            pool.close()
            await pool.wait_closed()
            logger.info(f"MySQL 메타데이터 조회 성공: {len(rows)}개 컬럼")
            return {"columns": rows}
        except Exception as e:
            logger.error(f"MySQL 메타데이터 조회 실패: {e}")
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
            # aiomysql은 %s 형태의 플레이스홀더를 사용하여 파라미터를 안전하게 바인딩
            param_values = list(params.values())

            logger.info(f"MySQL 쿼리 실행: {query[:100]}{'...' if len(query) > 100 else ''}")

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, param_values)
                    async for row in cursor:
                        yield dict(row)
            pool.close()
            await pool.wait_closed()

            logger.info("MySQL 쿼리 실행 완료")
        except aiomysql.ProgrammingError as e:
            logger.error(f"MySQL 구문 오류: {e}")
            raise QueryExecutionError(f"SQL 구문 오류: {e}") from e
        except aiomysql.OperationalError as e:
            logger.error(f"MySQL 실행 오류: {e}")
            raise QueryExecutionError(f"쿼리 실행 오류: {e}") from e
        except Exception as e:
            logger.error(f"MySQL 쿼리 실행 실패: {e}")
            raise QueryExecutionError(f"쿼리 실행에 실패했습니다: {e}") from e
