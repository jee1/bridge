#!/usr/bin/env python3
"""간단한 MCP 서버 구현 - 직접 JSON-RPC"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

class SimpleMCPServer:
    def __init__(self):
        self.tools = [
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

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """JSON-RPC 요청 처리"""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})

        logger.info(f"Handling request: {method}")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "bridge-mcp",
                        "version": "0.1.0"
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": self.tools
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "list_connectors":
                result = {
                    "success": True,
                    "message": "커넥터 목록을 성공적으로 조회했습니다",
                    "connectors": ["postgres", "mongodb", "elasticsearch", "mock"]
                }
            elif tool_name == "query_database":
                result = {
                    "success": True,
                    "database": arguments.get("database", "unknown"),
                    "query": arguments.get("query", ""),
                    "message": "쿼리가 성공적으로 실행되었습니다 (모의 응답)",
                    "results": [{"id": 1, "name": "Sample Data"}]
                }
            elif tool_name == "get_schema":
                result = {
                    "success": True,
                    "database": arguments.get("database", "unknown"),
                    "message": "스키마 정보를 성공적으로 조회했습니다 (모의 응답)",
                    "tables": ["users", "orders", "products"]
                }
            elif tool_name == "analyze_data":
                result = {
                    "success": True,
                    "intent": arguments.get("intent", ""),
                    "message": "분석이 성공적으로 완료되었습니다 (모의 응답)",
                    "insights": ["데이터 품질이 양호합니다", "추가 분석이 필요할 수 있습니다"]
                }
            else:
                result = {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, indent=2)
                    }]
                }
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    async def run(self):
        """MCP 서버 실행"""
        logger.info("Starting Simple MCP Server...")
        
        while True:
            try:
                # stdin에서 JSON-RPC 요청 읽기
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # JSON 파싱
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue
                
                # 요청 처리
                response = await self.handle_request(request)
                
                # 응답 전송
                print(json.dumps(response, ensure_ascii=False))
                sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                continue

def run():
    """동기 래퍼 함수 - 콘솔 스크립트 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = SimpleMCPServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    run()