#!/usr/bin/env python3
"""통합된 Bridge MCP 서버 - 환경 변수 기반 모드 지원"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Union

from bridge.utils import json as bridge_json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class UnifiedBridgeMCPServer:
    """통합된 Bridge MCP 서버 - 다양한 모드 지원"""
    
    def __init__(self):
        # 환경 변수 기반 설정
        self.mode = os.getenv("BRIDGE_MCP_MODE", "development").lower()
        self.use_sdk = os.getenv("BRIDGE_MCP_USE_SDK", "true").lower() == "true"
        self.error_recovery = os.getenv("BRIDGE_MCP_ERROR_RECOVERY", "false").lower() == "true"
        self.server_name = os.getenv("BRIDGE_MCP_SERVER_NAME", "bridge-mcp-unified")
        self.server_version = os.getenv("BRIDGE_MCP_VERSION", "1.0.0")
        
        # 서버 상태 관리
        self.is_initialized = False
        self.is_running = False
        self.request_count = 0
        self.connectors = {}
        
        # MCP SDK 관련
        self.server = None
        if self.use_sdk:
            self._init_mcp_sdk()
        
        # 시그널 핸들러 설정
        if self.error_recovery:
            self._setup_signal_handlers()
        
        # 모드별 초기화
        self._init_mode_specific()
        
        # 도구 정의
        self.tools = self._define_tools()
    
    def _init_mcp_sdk(self):
        """MCP SDK 초기화"""
        try:
            from mcp.server import Server
            from mcp import types
            from mcp import stdio_server
            from mcp.server.models import InitializationOptions
            from mcp.server.lowlevel.server import NotificationOptions
            
            self.server = Server(self.server_name)
            self.stdio_server = stdio_server
            self.types = types
            self.InitializationOptions = InitializationOptions
            self.NotificationOptions = NotificationOptions
            
            if self.use_sdk:
                self._setup_sdk_handlers()
                
        except ImportError as e:
            logger.warning(f"MCP SDK를 사용할 수 없습니다: {e}")
            self.use_sdk = False
    
    def _setup_sdk_handlers(self):
        """MCP SDK 핸들러 설정"""
        if not self.server:
            return
            
        @self.server.list_tools()
        async def list_tools():
            """사용 가능한 도구 목록 반환"""
            logger.info("list_tools() called")
            return self.types.ListToolsResult(
                tools=[self.types.Tool(**tool) for tool in self.tools]
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            """도구 실행"""
            logger.info(f"call_tool() called with name={name}, arguments={arguments}")
            try:
                result = await self._execute_tool(name, arguments)
                return self.types.CallToolResult(
                    content=[self.types.TextContent(
                        type="text",
                        text=bridge_json.dumps(result, indent=2)
                    )]
                )
            except Exception as e:
                logger.error(f"도구 실행 실패: {name}, {e}")
                return self.types.CallToolResult(
                    content=[self.types.TextContent(
                        type="text",
                        text=bridge_json.dumps({"error": str(e)}, indent=2)
                    )]
                )
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.is_running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _init_mode_specific(self):
        """모드별 초기화"""
        if self.mode == "real":
            self._init_real_connectors()
        elif self.mode == "production":
            self._init_production_connectors()
        else:  # development, mock
            self._init_mock_connectors()
    
    def _init_real_connectors(self):
        """실제 데이터베이스 커넥터 초기화"""
        try:
            from .connectors.postgres import PostgresConnector
            from .connectors.mysql import MySQLConnector
            from .connectors.elasticsearch import ElasticsearchConnector
            from .connectors.registry import connector_registry
            
            # PostgreSQL 커넥터
            postgres_settings = {
                "host": self._get_env("BRIDGE_POSTGRES_HOST", "localhost"),
                "port": self._get_env_int("BRIDGE_POSTGRES_PORT", 5432),
                "database": self._get_env("BRIDGE_POSTGRES_DB", "bridge_dev"),
                "user": self._get_env("BRIDGE_POSTGRES_USER", "bridge_user"),
                "password": self._get_env("BRIDGE_POSTGRES_PASSWORD", "bridge_password"),
            }
            self.connectors["postgres"] = PostgresConnector("postgres", postgres_settings)
            connector_registry.register(self.connectors["postgres"])
            
            # MySQL 커넥터
            mysql_settings = {
                "host": self._get_env("BRIDGE_MYSQL_HOST", "localhost"),
                "port": self._get_env_int("BRIDGE_MYSQL_PORT", 3306),
                "db": self._get_env("BRIDGE_MYSQL_DB", "bridge_dev"),
                "user": self._get_env("BRIDGE_MYSQL_USER", "bridge_user"),
                "password": self._get_env("BRIDGE_MYSQL_PASSWORD", "bridge_password"),
            }
            self.connectors["mysql"] = MySQLConnector("mysql", mysql_settings)
            connector_registry.register(self.connectors["mysql"])
            
            # Elasticsearch 커넥터
            elasticsearch_settings = {
                "host": self._get_env("BRIDGE_ELASTICSEARCH_HOST", "localhost"),
                "port": self._get_env_int("BRIDGE_ELASTICSEARCH_PORT", 9200),
                "use_ssl": self._get_env_bool("BRIDGE_ELASTICSEARCH_USE_SSL", False),
                "username": self._get_env("BRIDGE_ELASTICSEARCH_USERNAME", ""),
                "password": self._get_env("BRIDGE_ELASTICSEARCH_PASSWORD", ""),
                "url": self._get_env("BRIDGE_ELASTICSEARCH_URL", ""),
            }
            self.connectors["elasticsearch"] = ElasticsearchConnector("elasticsearch", elasticsearch_settings)
            connector_registry.register(self.connectors["elasticsearch"])
            
            logger.info(f"실제 커넥터 초기화 완료: {list(self.connectors.keys())}")
            
        except Exception as e:
            logger.error(f"실제 커넥터 초기화 실패: {e}")
            self._init_mock_connectors()
    
    def _init_production_connectors(self):
        """프로덕션 커넥터 초기화 (실제 + 에러 복구)"""
        self._init_real_connectors()
        # 프로덕션 특별 설정 추가 가능
    
    def _init_mock_connectors(self):
        """모의 커넥터 초기화"""
        try:
            from .connectors.mock import MockConnector
            self.connectors["mock"] = MockConnector("mock")
            logger.info("모의 커넥터 초기화 완료")
        except Exception as e:
            logger.warning(f"모의 커넥터 초기화 실패: {e}")
    
    def _define_tools(self):
        """도구 정의"""
        return [
            {
                "name": "query_database",
                "description": "데이터베이스 쿼리 실행",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "데이터베이스 이름"},
                        "query": {"type": "string", "description": "실행할 쿼리 (SQL 또는 Elasticsearch JSON)"},
                        "params": {"type": "object", "description": "쿼리 파라미터"}
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
                        "intent": {"type": "string", "description": "분석 의도 설명"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "object"}
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
    
    def _get_env(self, key: str, default: str) -> str:
        """환경 변수를 읽고 비어있으면 기본값을 반환"""
        value = os.getenv(key)
        if value is None or not value.strip():
            return default
        return value.strip()
    
    def _get_env_int(self, key: str, default: int) -> int:
        """정수 환경 변수를 읽고 비어있거나 잘못된 값이면 기본값을 사용"""
        value = os.getenv(key)
        if value is None or not value.strip():
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"환경 변수 {key} 값 '{value}'이(가) 유효한 정수가 아닙니다. 기본값 {default}을(를) 사용합니다.")
            return default
    
    def _get_env_bool(self, key: str, default: bool) -> bool:
        """불리언 환경 변수를 읽고 비어있으면 기본값을 반환"""
        value = os.getenv(key)
        if value is None or not value.strip():
            return default
        return value.strip().lower() == "true"
    
    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """JSON-RPC 요청 처리 (SDK 미사용 모드)"""
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
                            "name": self.server_name,
                            "version": self.server_version
                        }
                    }
                }

            elif method == "notifications/initialized":
                logger.info("Client initialized notification received")
                return None

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
                return await self._list_connectors(arguments)
            elif tool_name == "query_database":
                return await self._execute_query(arguments)
            elif tool_name == "get_schema":
                return await self._get_schema(arguments)
            elif tool_name == "analyze_data":
                return await self._analyze_data(arguments)
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
    
    async def _list_connectors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """커넥터 목록 조회"""
        try:
            if self.mode in ["real", "production"] and self.connectors:
                available_connectors = list(self.connectors.keys())
            else:
                available_connectors = ["postgres", "mongodb", "elasticsearch", "mock"]
            
            return {
                "success": True,
                "message": "커넥터 목록을 성공적으로 조회했습니다",
                "connectors": available_connectors,
                "mode": self.mode,
                "server_info": {
                    "name": self.server_name,
                    "version": self.server_version,
                    "request_count": self.request_count
                }
            }
        except Exception as e:
            logger.error(f"커넥터 목록 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스 쿼리 실행"""
        database = args.get("database", "").lower()
        query = args.get("query", "")
        params = args.get("params", {})
        
        try:
            if self.mode in ["real", "production"] and database in self.connectors:
                # 실제 커넥터 사용
                connector = self.connectors[database]
                
                # 연결 테스트
                try:
                    await connector.test_connection()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"데이터베이스 연결 실패: {str(e)}",
                        "database": database
                    }
                
                # 쿼리 실행
                try:
                    results = []
                    async for row in connector.run_query(query, params):
                        results.append(row)
                    
                    return {
                        "success": True,
                        "database": database,
                        "query": query,
                        "message": f"쿼리가 성공적으로 실행되었습니다",
                        "results": results,
                        "result_count": len(results),
                        "mode": self.mode
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"쿼리 실행 실패: {str(e)}",
                        "database": database,
                        "query": query
                    }
            else:
                # 모의 응답
                return {
                    "success": True,
                    "database": database,
                    "query": query,
                    "message": "쿼리가 성공적으로 실행되었습니다 (모의 응답)",
                    "results": [{"id": 1, "name": "Sample Data", "timestamp": time.time()}],
                    "mode": self.mode
                }
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "database": database
            }
    
    async def _get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """스키마 정보 조회"""
        database = args.get("database", "").lower()
        
        try:
            if self.mode in ["real", "production"] and database in self.connectors:
                # 실제 커넥터 사용
                connector = self.connectors[database]
                
                # 연결 테스트
                try:
                    await connector.test_connection()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"데이터베이스 연결 실패: {str(e)}",
                        "database": database
                    }
                
                # 스키마 조회
                try:
                    metadata = await connector.get_metadata()
                    return {
                        "success": True,
                        "database": database,
                        "message": "스키마 정보를 성공적으로 조회했습니다",
                        "metadata": metadata,
                        "mode": self.mode
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"스키마 조회 실패: {str(e)}",
                        "database": database
                    }
            else:
                # 모의 응답
                return {
                    "success": True,
                    "database": database,
                    "message": "스키마 정보를 성공적으로 조회했습니다 (모의 응답)",
                    "tables": ["users", "orders", "products"],
                    "columns": {
                        "users": ["id", "name", "email", "created_at"],
                        "orders": ["id", "user_id", "product_id", "amount", "created_at"],
                        "products": ["id", "name", "price", "category"]
                    },
                    "mode": self.mode
                }
        except Exception as e:
            logger.error(f"스키마 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "database": database
            }
    
    async def _analyze_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 분석 실행"""
        intent = args.get("intent", "")
        sources = args.get("sources", [])
        context = args.get("context", {})
        
        try:
            if self.mode in ["real", "production"]:
                # 실제 분석 로직 (향후 구현)
                analysis_result = {
                    "intent": intent,
                    "sources_analyzed": sources,
                    "context": context,
                    "analysis": f"'{intent}'에 대한 실제 분석이 요청되었습니다 (구현 예정)",
                    "recommendations": [
                        "실제 데이터 소스 연결을 확인하세요",
                        "필요한 도구를 지정하세요",
                        "컨텍스트 정보를 추가하세요"
                    ],
                    "status": "completed",
                    "mode": self.mode
                }
            else:
                # 모의 분석 응답
                analysis_result = {
                    "intent": intent,
                    "sources_analyzed": sources,
                    "context": context,
                    "analysis": f"'{intent}'에 대한 분석이 요청되었습니다 (모의 응답)",
                    "insights": [
                        "데이터 품질이 양호합니다",
                        "추가 분석이 필요할 수 있습니다",
                        "고객 세분화 분석을 권장합니다"
                    ],
                    "recommendations": [
                        "정기적인 데이터 품질 검사",
                        "고급 분석 도구 도입 검토"
                    ],
                    "status": "completed",
                    "mode": self.mode
                }
            
            return analysis_result
        except Exception as e:
            logger.error(f"분석 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "intent": intent
            }
    
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
    
    async def run_sdk_mode(self):
        """MCP SDK 모드로 실행"""
        logger.info(f"Starting Unified Bridge MCP Server (SDK mode) - Mode: {self.mode}")
        self.is_running = True
        
        try:
            async with self.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.InitializationOptions(
                        server_name=self.server_name,
                        server_version=self.server_version,
                        capabilities=self.server.get_capabilities(
                            notification_options=self.NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        except Exception as e:
            logger.error(f"MCP Server error: {e}")
            raise
        finally:
            self._cleanup_resources()
            logger.info("Unified Bridge MCP Server stopped")
    
    async def run_direct_mode(self):
        """직접 JSON-RPC 모드로 실행"""
        logger.info(f"Starting Unified Bridge MCP Server (Direct mode) - Mode: {self.mode}")
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
                    if self.error_recovery:
                        # 에러 복구 모드에서는 계속 실행
                        continue
                    else:
                        raise
                        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self._cleanup_resources()
            logger.info("Unified Bridge MCP Server stopped")
    
    async def run(self):
        """MCP 서버 실행"""
        if self.use_sdk:
            await self.run_sdk_mode()
        else:
            await self.run_direct_mode()


def run():
    """동기 래퍼 함수 - 콘솔 스크립트 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = UnifiedBridgeMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    run()
