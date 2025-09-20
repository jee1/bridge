#!/usr/bin/env python3
"""최소한의 MCP 서버 구현"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

from mcp.server import Server
from mcp import types
from mcp import stdio_server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

class MinimalBridgeMCPServer:
    def __init__(self):
        self.server = Server("bridge-mcp")
        self.setup_handlers()

    def setup_handlers(self):
        """MCP 핸들러 설정"""
        
        @self.server.list_tools()
        async def list_tools():
            """사용 가능한 도구 목록 반환"""
            logger.info("list_tools() called")
            
            # 간단한 도구 목록 생성
            tools = [
                {
                    "name": "query_database",
                    "description": "데이터베이스 쿼리 실행",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "database": {"type": "string", "description": "데이터베이스 이름"},
                            "query": {"type": "string", "description": "실행할 SQL 쿼리"}
                        },
                        "required": ["database", "query"]
                    }
                },
                {
                    "name": "get_schema",
                    "description": "데이터베이스 스키마 정보 조회",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "database": {"type": "string", "description": "데이터베이스 이름"}
                        },
                        "required": ["database"]
                    }
                },
                {
                    "name": "analyze_data",
                    "description": "Bridge 오케스트레이터를 통해 데이터 분석",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string", "description": "분석 의도 설명"}
                        },
                        "required": ["intent"]
                    }
                },
                {
                    "name": "list_connectors",
                    "description": "사용 가능한 데이터 커넥터 목록 조회",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
            
            logger.info(f"Returning {len(tools)} tools")
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            """도구 실행"""
            logger.info(f"call_tool() called with name={name}, arguments={arguments}")
            
            try:
                if name == "query_database":
                    result = await self._execute_query(arguments)
                elif name == "get_schema":
                    result = await self._get_schema(arguments)
                elif name == "analyze_data":
                    result = await self._analyze_data(arguments)
                elif name == "list_connectors":
                    result = await self._list_connectors(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return result
            except Exception as e:
                logger.error(f"도구 실행 실패: {name}, {e}")
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
                    }]
                }

    async def _execute_query(self, args: Dict[str, Any]):
        """데이터베이스 쿼리 실행"""
        database = args.get("database", "unknown")
        query = args.get("query", "")
        
        result = {
            "success": True,
            "database": database,
            "query": query,
            "message": "쿼리가 성공적으로 실행되었습니다 (모의 응답)",
            "results": [{"id": 1, "name": "Sample Data"}]
        }
        
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        )

    async def _get_schema(self, args: Dict[str, Any]):
        """스키마 정보 조회"""
        database = args.get("database", "unknown")
        
        result = {
            "success": True,
            "database": database,
            "message": "스키마 정보를 성공적으로 조회했습니다 (모의 응답)",
            "tables": ["users", "orders", "products"]
        }
        
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        )

    async def _analyze_data(self, args: Dict[str, Any]):
        """데이터 분석 실행"""
        intent = args.get("intent", "")
        
        result = {
            "success": True,
            "intent": intent,
            "message": "분석이 성공적으로 완료되었습니다 (모의 응답)",
            "insights": ["데이터 품질이 양호합니다", "추가 분석이 필요할 수 있습니다"]
        }
        
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        )

    async def _list_connectors(self, args: Dict[str, Any]):
        """커넥터 목록 조회"""
        result = {
            "success": True,
            "message": "커넥터 목록을 성공적으로 조회했습니다 (모의 응답)",
            "connectors": ["postgres", "mongodb", "elasticsearch", "mock"]
        }
        
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        )


async def main():
    """MCP 서버 실행"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting Minimal Bridge MCP Server...")

    bridge_server = MinimalBridgeMCPServer()
    logger.info("Minimal Bridge MCP Server initialized")

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


def run():
    """동기 래퍼 함수 - 콘솔 스크립트 진입점"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
