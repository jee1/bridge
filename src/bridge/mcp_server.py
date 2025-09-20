"""Bridge MCP 서버 구현."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp import types
from mcp import stdio_server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions

from .connectors import connector_registry, ConnectorNotFoundError
from .connectors.exceptions import ConnectionError, QueryExecutionError, MetadataError
from .semantic.models import TaskRequest

logger = logging.getLogger(__name__)


class BridgeMCPServer:
    """Bridge MCP 서버."""
    
    def __init__(self):
        self.server = Server("bridge-mcp")
        self.connectors = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """MCP 핸들러 설정."""
        
        @self.server.list_tools()
        async def list_tools() -> types.ListToolsResult:
            """사용 가능한 도구 목록 반환."""
            logger.info("list_tools() called")
            tools = [
                types.Tool(
                    name="query_database",
                    description="데이터베이스 쿼리 실행",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL 쿼리"},
                            "database": {"type": "string", "description": "데이터베이스 이름"},
                            "params": {"type": "object", "description": "쿼리 파라미터"}
                        },
                        "required": ["query", "database"]
                    }
                ),
                types.Tool(
                    name="get_schema",
                    description="데이터베이스 스키마 정보 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {"type": "string", "description": "데이터베이스 이름"}
                        },
                        "required": ["database"]
                    }
                ),
                types.Tool(
                    name="analyze_data",
                    description="데이터 분석 및 인사이트 생성",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string", "description": "분석 의도"},
                            "sources": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "object"}
                        },
                        "required": ["intent"]
                    }
                ),
                types.Tool(
                    name="list_connectors",
                    description="사용 가능한 커넥터 목록 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
            logger.info(f"Returning {len(tools)} tools: {[tool.name for tool in tools]}")
            return types.ListToolsResult(
                tools=tools,
                meta=None,
                nextCursor=None
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
            """도구 실행."""
            logger.info(f"call_tool() called with name={name}, arguments={arguments}")
            try:
                if name == "query_database":
                    return await self._execute_query(arguments)
                elif name == "get_schema":
                    return await self._get_schema(arguments)
                elif name == "analyze_data":
                    return await self._analyze_data(arguments)
                elif name == "list_connectors":
                    return await self._list_connectors(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"도구 실행 실패: {name}, {e}")
                return types.CallToolResult(
                    content=[types.TextContent(
                        type="text",
                        text=json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
                    )]
                )
    
    async def _execute_query(self, args: Dict[str, Any]) -> types.CallToolResult:
        """데이터베이스 쿼리 실행."""
        database = args["database"]
        query = args["query"]
        params = args.get("params", {})
        
        try:
            # 커넥터 가져오기
            connector = await self._get_connector(database)
            
            # 쿼리 실행
            results = []
            async for row in connector.run_query(query, params):
                results.append(dict(row))
            
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "database": database,
                        "query": query,
                        "results": results,
                        "count": len(results)
                    }, ensure_ascii=False, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "database": database,
                        "error": str(e)
                    }, ensure_ascii=False, indent=2)
                )]
            )
    
    async def _get_schema(self, args: Dict[str, Any]) -> types.CallToolResult:
        """스키마 정보 조회."""
        database = args["database"]
        
        try:
            connector = await self._get_connector(database)
            metadata = await connector.get_metadata()
            
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "database": database,
                        "metadata": metadata
                    }, ensure_ascii=False, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"스키마 조회 실패: {e}")
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "database": database,
                        "error": str(e)
                    }, ensure_ascii=False, indent=2)
                )]
            )
    
    async def _analyze_data(self, args: Dict[str, Any]) -> types.CallToolResult:
        """데이터 분석 실행."""
        intent = args["intent"]
        sources = args.get("sources", [])
        context = args.get("context", {})
        
        try:
            # TaskRequest 생성
            task_request = TaskRequest(
                intent=intent,
                sources=sources,
                context=context
            )
            
            # 간단한 분석 시뮬레이션
            analysis_result = {
                "intent": intent,
                "sources_analyzed": sources,
                "context": context,
                "analysis": f"'{intent}'에 대한 분석이 요청되었습니다.",
                "recommendations": [
                    "데이터 소스 연결을 확인하세요",
                    "필요한 도구를 지정하세요",
                    "컨텍스트 정보를 추가하세요"
                ],
                "status": "completed"
            }
            
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps(analysis_result, ensure_ascii=False, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"분석 실행 실패: {e}")
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "intent": intent,
                        "error": str(e)
                    }, ensure_ascii=False, indent=2)
                )]
            )
    
    async def _list_connectors(self, args: Dict[str, Any]) -> types.CallToolResult:
        """사용 가능한 커넥터 목록 조회."""
        try:
            available_connectors = list(connector_registry.list_connectors())
            
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "connectors": available_connectors,
                        "count": len(available_connectors)
                    }, ensure_ascii=False, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"커넥터 목록 조회 실패: {e}")
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e)
                    }, ensure_ascii=False, indent=2)
                )]
            )
    
    async def _get_connector(self, database: str):
        """커넥터 가져오기 또는 생성."""
        if database not in self.connectors:
            try:
                connector = connector_registry.get(database)
                self.connectors[database] = connector
            except ConnectorNotFoundError:
                # Mock 커넥터로 폴백
                from .connectors.mock import MockConnector
                connector = MockConnector(name=database)
                self.connectors[database] = connector
                logger.warning(f"커넥터 '{database}'를 찾을 수 없어 Mock 커넥터를 사용합니다.")
        
        return self.connectors[database]


async def main():
    """MCP 서버 실행."""
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting Bridge MCP Server...")
    
    bridge_server = BridgeMCPServer()
    logger.info("Bridge MCP Server initialized")
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            await bridge_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="bridge-mcp",
                    server_version="0.1.0",
                    capabilities=bridge_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
        except Exception as e:
            logger.error(f"MCP Server error: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
