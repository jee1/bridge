"""PostgreSQL 커넥터 예시 구현."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import asyncpg

from .base import BaseConnector


class PostgresConnector(BaseConnector):
    """PostgreSQL 데이터 접근을 담당한다."""

    async def _get_pool(self) -> asyncpg.Pool:
        return await asyncpg.create_pool(**self.settings)

    async def test_connection(self) -> bool:  # type: ignore[override]
        async with await self._get_pool() as pool:
            async with pool.acquire() as connection:
                await connection.execute("SELECT 1")
        return True

    async def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        query = """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position
        """
        rows = []
        async with await self._get_pool() as pool:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    async for record in connection.cursor(query):
                        rows.append(dict(record))
        return {"columns": rows}

    async def run_query(  # type: ignore[override]
        self, query: str, params: Dict[str, Any] | None = None
    ) -> Iterable[Dict[str, Any]]:
        params = params or {}
        async with await self._get_pool() as pool:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    async for record in connection.cursor(query, *params.values()):
                        yield dict(record)
