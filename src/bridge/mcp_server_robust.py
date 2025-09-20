#!/usr/bin/env python3
"""강화된 MCP 서버 구현 - 재시작 안정성 개선"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

class RobustMCPServer:
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
        
        # 서버 상태 관리
        self.is_initialized = False
        self.is_running = False
        self.request_count = 0
        
        # 시그널 핸들러 설정
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.is_running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _cleanup_resources(self):
        """리소스 정리"""
        logger.info("Cleaning up resources...")
        self.is_initialized = False
        self.is_running = False
        self.request_count = 0

    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """JSON-RPC 요청 처리"""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        self.request_count += 1
        logger.info(f"Handling request #{self.request_count}: {method}")

        try:
            if method == "initialize":
                self.is_initialized = True
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": False}
                        },
                        "serverInfo": {
                            "name": "bridge-mcp-robust",
                            "version": "0.2.0"
                        }
                    }
                }

            elif method == "notifications/initialized":
                logger.info("Client initialized notification received")
                return None  # Notifications don't get responses

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

                result = await self._execute_tool(tool_name, arguments)
                
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

        except Exception as e:
            logger.error(f"Error handling request {method}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행"""
        try:
            if tool_name == "list_connectors":
                return {
                    "success": True,
                    "message": "커넥터 목록을 성공적으로 조회했습니다",
                    "connectors": ["postgres", "mongodb", "elasticsearch", "mock"],
                    "server_info": {
                        "name": "bridge-mcp-robust",
                        "version": "0.2.0",
                        "request_count": self.request_count
                    }
                }
            elif tool_name == "query_database":
                return {
                    "success": True,
                    "database": arguments.get("database", "unknown"),
                    "query": arguments.get("query", ""),
                    "message": "쿼리가 성공적으로 실행되었습니다 (모의 응답)",
                    "results": [{"id": 1, "name": "Sample Data", "timestamp": time.time()}]
                }
            elif tool_name == "get_schema":
                return {
                    "success": True,
                    "database": arguments.get("database", "unknown"),
                    "message": "스키마 정보를 성공적으로 조회했습니다 (모의 응답)",
                    "tables": ["users", "orders", "products"],
                    "columns": {
                        "users": ["id", "name", "email", "created_at"],
                        "orders": ["id", "user_id", "product_id", "amount", "created_at"],
                        "products": ["id", "name", "price", "category"]
                    }
                }
            elif tool_name == "analyze_data":
                return {
                    "success": True,
                    "intent": arguments.get("intent", ""),
                    "message": "분석이 성공적으로 완료되었습니다 (모의 응답)",
                    "insights": [
                        "데이터 품질이 양호합니다",
                        "추가 분석이 필요할 수 있습니다",
                        "고객 세분화 분석을 권장합니다"
                    ],
                    "recommendations": [
                        "정기적인 데이터 품질 검사",
                        "고급 분석 도구 도입 검토"
                    ]
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [tool["name"] for tool in self.tools]
                }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    async def run(self):
        """MCP 서버 실행"""
        logger.info("Starting Robust MCP Server...")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    # stdin에서 JSON-RPC 요청 읽기
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    
                    if not line:
                        logger.info("EOF received, shutting down...")
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
                    
                    # 응답 전송 (None이 아닌 경우만)
                    if response is not None:
                        print(json.dumps(response, ensure_ascii=False))
                        sys.stdout.flush()
                    
                except asyncio.CancelledError:
                    logger.info("Server cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    # 에러가 발생해도 서버는 계속 실행
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self._cleanup_resources()
            logger.info("Robust MCP Server stopped")

def run():
    """동기 래퍼 함수 - 콘솔 스크립트 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = RobustMCPServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    run()
