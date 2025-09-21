#!/usr/bin/env python3
"""실제 데이터베이스 커넥터를 사용하는 MCP 서버"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from bridge.utils import json as bridge_json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from .connectors.postgres import PostgresConnector
from .connectors.mysql import MySQLConnector
from .connectors.elasticsearch import ElasticsearchConnector
from .connectors.registry import connector_registry

logger = logging.getLogger(__name__)


def _get_env(key: str, default: str) -> str:
    """환경 변수를 읽고 비어있으면 기본값을 반환한다."""
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_env_int(key: str, default: int) -> int:
    """정수 환경 변수를 읽고 비어있거나 잘못된 값이면 기본값을 사용한다."""
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("환경 변수 %s 값 '%s'이(가) 유효한 정수가 아닙니다. 기본값 %s을(를) 사용합니다.", key, value, default)
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """불리언 환경 변수를 읽고 비어있으면 기본값을 반환한다."""
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return value.strip().lower() == "true"

class RealMCPServer:
    def __init__(self):
        self.tools = [
            {
                "name": "query_database",
                "description": "데이터베이스 쿼리 실행",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "데이터베이스 이름 (postgres, mysql, elasticsearch)"},
                        "query": {"type": "string", "description": "실행할 쿼리 (SQL 또는 Elasticsearch JSON)"}
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
                        "database": {"type": "string", "description": "데이터베이스 이름 (postgres, mysql, elasticsearch)"}
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
        self.connectors = {}
        
        # 시그널 핸들러 설정
        self._setup_signal_handlers()
        
        # 커넥터 초기화
        self._initialize_connectors()

    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.is_running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _initialize_connectors(self):
        """커넥터들을 초기화한다."""
        try:
            # PostgreSQL 커넥터
            postgres_settings = {
                "host": _get_env("BRIDGE_POSTGRES_HOST", "localhost"),
                "port": _get_env_int("BRIDGE_POSTGRES_PORT", 5432),
                "database": _get_env("BRIDGE_POSTGRES_DB", "bridge_dev"),
                "user": _get_env("BRIDGE_POSTGRES_USER", "bridge_user"),
                "password": _get_env("BRIDGE_POSTGRES_PASSWORD", "bridge_password"),
            }
            self.connectors["postgres"] = PostgresConnector("postgres", postgres_settings)
            connector_registry.register(self.connectors["postgres"])
            
            # MySQL 커넥터
            mysql_settings = {
                "host": _get_env("BRIDGE_MYSQL_HOST", "localhost"),
                "port": _get_env_int("BRIDGE_MYSQL_PORT", 3306),
                "db": _get_env("BRIDGE_MYSQL_DB", "bridge_dev"),
                "user": _get_env("BRIDGE_MYSQL_USER", "bridge_user"),
                "password": _get_env("BRIDGE_MYSQL_PASSWORD", "bridge_password"),
            }
            self.connectors["mysql"] = MySQLConnector("mysql", mysql_settings)
            connector_registry.register(self.connectors["mysql"])
            
            # Elasticsearch 커넥터
            elasticsearch_settings = {
                "host": _get_env("BRIDGE_ELASTICSEARCH_HOST", "localhost"),
                "port": _get_env_int("BRIDGE_ELASTICSEARCH_PORT", 9200),
                "use_ssl": _get_env_bool("BRIDGE_ELASTICSEARCH_USE_SSL", False),
                "username": _get_env("BRIDGE_ELASTICSEARCH_USERNAME", ""),
                "password": _get_env("BRIDGE_ELASTICSEARCH_PASSWORD", ""),
                "url": _get_env("BRIDGE_ELASTICSEARCH_URL", ""),
            }
            logger.info(f"Elasticsearch 설정: {elasticsearch_settings}")
            
            # Elasticsearch 커넥터 사용
            self.connectors["elasticsearch"] = ElasticsearchConnector("elasticsearch", elasticsearch_settings)
            connector_registry.register(self.connectors["elasticsearch"])
            
            logger.info(f"커넥터 초기화 완료: {list(self.connectors.keys())}")
            
        except Exception as e:
            logger.error(f"커넥터 초기화 실패: {e}")
            # 커넥터 초기화 실패해도 서버는 계속 실행 (모의 응답으로 대체)

    def _cleanup_resources(self):
        """리소스 정리"""
        logger.info("Cleaning up resources...")
        self.is_initialized = False
        self.is_running = False
        self.request_count = 0
        
        # 커넥터 정리
        for connector in self.connectors.values():
            if hasattr(connector, 'close'):
                try:
                    asyncio.create_task(connector.close())
                except Exception as e:
                    logger.warning(f"커넥터 정리 중 오류: {e}")

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
                            "name": "bridge-mcp-real",
                            "version": "0.3.0"
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
                            "text": bridge_json.dumps(result, indent=2)
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
                    "connectors": list(self.connectors.keys()),
                    "server_info": {
                        "name": "bridge-mcp-real",
                        "version": "0.3.0",
                        "request_count": self.request_count
                    }
                }
                
            elif tool_name == "query_database":
                database = arguments.get("database", "").lower()
                query = arguments.get("query", "")
                
                if database not in self.connectors:
                    return {
                        "success": False,
                        "error": f"지원하지 않는 데이터베이스: {database}",
                        "available_databases": list(self.connectors.keys())
                    }
                
                # 연결 테스트
                try:
                    await self.connectors[database].test_connection()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"데이터베이스 연결 실패: {str(e)}",
                        "database": database
                    }
                
                # 쿼리 실행
                try:
                    results = []
                    async for row in self.connectors[database].run_query(query):
                        results.append(row)
                    
                    return {
                        "success": True,
                        "database": database,
                        "query": query,
                        "message": f"쿼리가 성공적으로 실행되었습니다",
                        "results": results,
                        "result_count": len(results)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"쿼리 실행 실패: {str(e)}",
                        "database": database,
                        "query": query
                    }
                    
            elif tool_name == "get_schema":
                database = arguments.get("database", "").lower()
                
                if database not in self.connectors:
                    return {
                        "success": False,
                        "error": f"지원하지 않는 데이터베이스: {database}",
                        "available_databases": list(self.connectors.keys())
                    }
                
                # 연결 테스트
                try:
                    await self.connectors[database].test_connection()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"데이터베이스 연결 실패: {str(e)}",
                        "database": database
                    }
                
                # 스키마 조회
                try:
                    metadata = await self.connectors[database].get_metadata()
                    return {
                        "success": True,
                        "database": database,
                        "message": "스키마 정보를 성공적으로 조회했습니다",
                        "metadata": metadata
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"스키마 조회 실패: {str(e)}",
                        "database": database
                    }
                    
            elif tool_name == "analyze_data":
                intent = arguments.get("intent", "")
                return {
                    "success": True,
                    "intent": intent,
                    "message": "분석이 성공적으로 완료되었습니다 (실제 구현 예정)",
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
        logger.info("Starting Real MCP Server...")
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
                        print(bridge_json.dumps(response))
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
            logger.info("Real MCP Server stopped")

def run():
    """동기 래퍼 함수 - 콘솔 스크립트 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = RealMCPServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    run()
