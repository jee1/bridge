#!/usr/bin/env python3
"""통합된 Bridge MCP 서버 - 환경 변수 기반 모드 지원"""

import asyncio
import base64
import io
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bridge.utils import json as bridge_json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

PLACEHOLDER_CHART_DATA = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9YqVFxoAAAAASUVORK5CYII="
)


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
        self.connectors: Dict[str, Any] = {}

        # MCP SDK 관련
        self.server: Any | None = None
        self.stdio_server: Any | None = None
        self.types: Any | None = None
        self.InitializationOptions: Any | None = None
        self.NotificationOptions: Any | None = None
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
            from mcp import stdio_server, types
            from mcp.server import Server
            from mcp.server.lowlevel.server import NotificationOptions
            from mcp.server.models import InitializationOptions

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
        if not self.server or self.types is None:
            return

        types_module = self.types

        @self.server.list_tools()
        async def list_tools():
            """사용 가능한 도구 목록 반환"""
            logger.info("list_tools() called")
            return types_module.ListToolsResult(
                tools=[types_module.Tool(**tool) for tool in self.tools]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            """도구 실행"""
            logger.info(f"call_tool() called with name={name}, arguments={arguments}")
            try:
                result = await self._execute_tool(name, arguments)
                return types_module.CallToolResult(
                    content=[
                        types_module.TextContent(
                            type="text", text=bridge_json.dumps(result, indent=2)
                        )
                    ]
                )
            except Exception as e:
                logger.error(f"도구 실행 실패: {name}, {e}")
                return types_module.CallToolResult(
                    content=[
                        types_module.TextContent(
                            type="text", text=bridge_json.dumps({"error": str(e)}, indent=2)
                        )
                    ]
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
            from .connectors.elasticsearch import ElasticsearchConnector
            from .connectors.mysql import MySQLConnector
            from .connectors.postgres import PostgresConnector
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
            self.connectors["elasticsearch"] = ElasticsearchConnector(
                "elasticsearch", elasticsearch_settings
            )
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
                        "query": {
                            "type": "string",
                            "description": "실행할 쿼리 (SQL 또는 Elasticsearch JSON)",
                        },
                        "params": {"type": "object", "description": "쿼리 파라미터"},
                    },
                    "required": ["database", "query"],
                },
            },
            {
                "name": "get_schema",
                "description": "데이터베이스 스키마 정보 조회",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "데이터베이스 이름"}
                    },
                    "required": ["database"],
                },
            },
            {
                "name": "analyze_data",
                "description": "Bridge 오케스트레이터를 통해 데이터 분석",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string", "description": "분석 의도 설명"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "object"},
                    },
                    "required": ["intent"],
                },
            },
            {
                "name": "list_connectors",
                "description": "사용 가능한 데이터 커넥터 목록 조회",
                "inputSchema": {"type": "object", "properties": {}},
            },
            # Analytics 도구들 추가
            {
                "name": "statistics_analyzer",
                "description": "데이터에 대한 기본 통계 분석 수행",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "분석할 데이터 (UnifiedDataFrame 또는 dict)",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "분석할 컬럼 목록 (선택사항)",
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["descriptive", "distribution", "correlation", "summary"],
                            "description": "분석 유형",
                        },
                    },
                    "required": ["data", "analysis_type"],
                },
            },
            {
                "name": "data_profiler",
                "description": "데이터 프로파일링 및 기본 정보 수집",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "프로파일링할 데이터"},
                        "include_stats": {
                            "type": "boolean",
                            "description": "통계 정보 포함 여부",
                            "default": True,
                        },
                        "include_quality": {
                            "type": "boolean",
                            "description": "품질 검사 포함 여부",
                            "default": True,
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "outlier_detector",
                "description": "데이터에서 이상치 탐지",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "분석할 데이터"},
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "분석할 컬럼 목록",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore"],
                            "description": "이상치 탐지 방법",
                            "default": "iqr",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "임계값 (Z-score 방법 사용시)",
                            "default": 3.0,
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "chart_generator",
                "description": "데이터 시각화 차트 생성",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "시각화할 데이터"},
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "histogram", "box", "heatmap"],
                            "description": "차트 유형",
                        },
                        "x_column": {"type": "string", "description": "X축 컬럼명"},
                        "y_column": {"type": "string", "description": "Y축 컬럼명 (선택사항)"},
                        "hue_column": {
                            "type": "string",
                            "description": "색상 구분 컬럼명 (산점도용)",
                        },
                        "title": {"type": "string", "description": "차트 제목"},
                        "config": {"type": "object", "description": "차트 설정 (선택사항)"},
                    },
                    "required": ["data", "chart_type"],
                },
            },
            {
                "name": "quality_checker",
                "description": "데이터 품질 검사 수행",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "검사할 데이터"},
                        "check_missing": {
                            "type": "boolean",
                            "description": "결측값 검사 여부",
                            "default": True,
                        },
                        "check_outliers": {
                            "type": "boolean",
                            "description": "이상치 검사 여부",
                            "default": True,
                        },
                        "check_consistency": {
                            "type": "boolean",
                            "description": "일관성 검사 여부",
                            "default": True,
                        },
                        "outlier_method": {
                            "type": "string",
                            "enum": ["iqr", "zscore"],
                            "description": "이상치 탐지 방법",
                            "default": "iqr",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "report_builder",
                "description": "종합 분석 리포트 생성",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "분석할 데이터"},
                        "title": {"type": "string", "description": "리포트 제목"},
                        "author": {"type": "string", "description": "작성자"},
                        "include_charts": {
                            "type": "boolean",
                            "description": "차트 포함 여부",
                            "default": True,
                        },
                        "include_dashboard": {
                            "type": "boolean",
                            "description": "대시보드 포함 여부",
                            "default": True,
                        },
                        "include_quality": {
                            "type": "boolean",
                            "description": "품질 검사 포함 여부",
                            "default": True,
                        },
                    },
                    "required": ["data"],
                },
            },
            # CA 마일스톤 3.1: 통합 데이터 분석 레이어 도구들
            {
                "name": "data_unifier",
                "description": "다중 소스 데이터를 표준 테이블 형태로 통합",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data_sources": {
                            "type": "object",
                            "description": "데이터 소스 딕셔너리 {소스명: 데이터}",
                        },
                        "schema_mapping": {
                            "type": "object",
                            "description": "스키마 매핑 규칙 {소스명: {원본컬럼: 통합컬럼}}",
                        },
                        "merge_strategy": {
                            "type": "string",
                            "enum": ["union", "intersection", "custom"],
                            "description": "통합 전략",
                            "default": "union",
                        },
                    },
                    "required": ["data_sources"],
                },
            },
            {
                "name": "schema_mapper",
                "description": "스키마 매핑 및 정규화",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_name": {"type": "string", "description": "소스명"},
                        "source_schema": {"type": "object", "description": "원본 스키마"},
                        "target_schema": {"type": "object", "description": "대상 스키마 (선택사항)"},
                        "mapping_rules": {
                            "type": "object",
                            "description": "매핑 규칙 {원본컬럼: 대상컬럼}",
                        },
                        "auto_detect": {
                            "type": "boolean",
                            "description": "자동 매핑 감지 여부",
                            "default": True,
                        },
                    },
                    "required": ["source_name", "source_schema"],
                },
            },
            {
                "name": "type_converter",
                "description": "고급 데이터 타입 변환",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "변환할 데이터"},
                        "target_schema": {"type": "object", "description": "대상 스키마"},
                        "column_mappings": {
                            "type": "object",
                            "description": "컬럼 매핑 {원본컬럼: 대상컬럼}",
                        },
                        "conversion_rules": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "사용자 정의 변환 규칙",
                        },
                    },
                    "required": ["data", "target_schema"],
                },
            },
            {
                "name": "streaming_processor",
                "description": "대용량 데이터 스트리밍 처리",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "처리할 데이터"},
                        "processor_func": {"type": "string", "description": "처리 함수명"},
                        "chunk_size": {
                            "type": "integer",
                            "description": "청크 크기",
                            "default": 10000,
                        },
                        "memory_limit_mb": {
                            "type": "integer",
                            "description": "메모리 제한 (MB)",
                            "default": 1000,
                        },
                        "parameters": {
                            "type": "object",
                            "description": "처리 함수에 전달할 파라미터",
                        },
                    },
                    "required": ["data", "processor_func"],
                },
            },
            {
                "name": "integrated_data_layer",
                "description": "통합 데이터 분석 레이어 메인 도구",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["integrate", "transform", "export", "validate", "summary"],
                            "description": "수행할 작업",
                        },
                        "data_sources": {
                            "type": "object",
                            "description": "데이터 소스 딕셔너리 (integrate 작업시)",
                        },
                        "transformations": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "변환 규칙 목록 (transform 작업시)",
                        },
                        "export_format": {
                            "type": "string",
                            "enum": ["arrow", "pandas", "csv", "parquet"],
                            "description": "내보낼 형식 (export 작업시)",
                            "default": "arrow",
                        },
                        "file_path": {"type": "string", "description": "파일 경로 (export 작업시)"},
                        "chunk_size": {
                            "type": "integer",
                            "description": "청크 크기",
                            "default": 10000,
                        },
                        "memory_limit_mb": {
                            "type": "integer",
                            "description": "메모리 제한 (MB)",
                            "default": 1000,
                        },
                    },
                    "required": ["action"],
                },
            },
            # CA 마일스톤 3.2: 고급 통계 분석 및 시각화 도구들
            {
                "name": "advanced_statistics",
                "description": "고급 통계 분석 수행 (기술 통계, 상관관계, 분포 분석)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "분석할 데이터"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["descriptive", "correlation", "distribution", "summary"],
                            "description": "분석 유형"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "분석할 컬럼 목록"
                        }
                    },
                    "required": ["data", "analysis_type"]
                }
            },
            {
                "name": "interactive_charts",
                "description": "인터랙티브 차트 및 시각화 생성",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "시각화할 데이터"},
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "histogram", "box", "heatmap"],
                            "description": "차트 유형"
                        },
                        "x_column": {"type": "string", "description": "X축 컬럼명"},
                        "y_column": {"type": "string", "description": "Y축 컬럼명"},
                        "hue_column": {"type": "string", "description": "색상 구분 컬럼명"},
                        "title": {"type": "string", "description": "차트 제목"},
                        "config": {"type": "object", "description": "차트 설정"}
                    },
                    "required": ["data", "chart_type"]
                }
            },
            {
                "name": "statistical_tests",
                "description": "통계적 검정 수행 (가설검정, A/B 테스트, 회귀분석)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "분석할 데이터"},
                        "test_type": {
                            "type": "string",
                            "enum": ["t_test", "chi_square", "anova", "regression", "ab_test"],
                            "description": "검정 유형"
                        },
                        "x_column": {"type": "string", "description": "X 변수 컬럼명"},
                        "y_column": {"type": "string", "description": "Y 변수 컬럼명"},
                        "group_column": {"type": "string", "description": "그룹 변수 컬럼명"},
                        "alpha": {"type": "number", "description": "유의수준 (기본값: 0.05)"}
                    },
                    "required": ["data", "test_type"]
                }
            },
            {
                "name": "time_series_analysis",
                "description": "시계열 분석 및 예측 수행",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "분석할 데이터"},
                        "time_column": {"type": "string", "description": "시간 컬럼명"},
                        "value_column": {"type": "string", "description": "값 컬럼명"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["decompose", "trend", "seasonality", "forecast", "anomalies"],
                            "description": "분석 유형"
                        },
                        "method": {"type": "string", "description": "분석 방법"},
                        "forecast_periods": {"type": "integer", "description": "예측 기간"},
                        "threshold": {"type": "number", "description": "임계값"}
                    },
                    "required": ["data", "time_column", "value_column", "analysis_type"]
                }
            },
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
            logger.warning(
                f"환경 변수 {key} 값 '{value}'이(가) 유효한 정수가 아닙니다. 기본값 {default}을(를) 사용합니다."
            )
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
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": self.server_name, "version": self.server_version},
                    },
                }

            elif method == "notifications/initialized":
                logger.info("Client initialized notification received")
                return None

            elif method == "tools/list":
                return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": self.tools}}

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                result = await self._execute_tool(tool_name, arguments)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": bridge_json.dumps(result, indent=2)}]
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.error(f"Error handling request {method}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
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
            # Analytics 도구들
            elif tool_name == "statistics_analyzer":
                return await self._statistics_analyzer(arguments)
            elif tool_name == "data_profiler":
                return await self._data_profiler(arguments)
            elif tool_name == "outlier_detector":
                return await self._outlier_detector(arguments)
            elif tool_name == "chart_generator":
                return await self._chart_generator(arguments)
            elif tool_name == "quality_checker":
                return await self._quality_checker(arguments)
            elif tool_name == "report_builder":
                return await self._report_builder(arguments)
            # CA 마일스톤 3.1: 통합 데이터 분석 레이어 도구들
            elif tool_name == "data_unifier":
                return await self._data_unifier(arguments)
            elif tool_name == "schema_mapper":
                return await self._schema_mapper(arguments)
            elif tool_name == "type_converter":
                return await self._type_converter(arguments)
            elif tool_name == "streaming_processor":
                return await self._streaming_processor(arguments)
            elif tool_name == "integrated_data_layer":
                return await self._integrated_data_layer(arguments)
            # CA 마일스톤 3.2: 고급 통계 분석 및 시각화 도구들
            elif tool_name == "advanced_statistics":
                return await self._advanced_statistics(arguments)
            elif tool_name == "interactive_charts":
                return await self._interactive_charts(arguments)
            elif tool_name == "statistical_tests":
                return await self._statistical_tests(arguments)
            elif tool_name == "time_series_analysis":
                return await self._time_series_analysis(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [tool["name"] for tool in self.tools],
                }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": f"Tool execution failed: {str(e)}"}

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
                    "request_count": self.request_count,
                },
            }
        except Exception as e:
            logger.error(f"커넥터 목록 조회 실패: {e}")
            return {"success": False, "error": str(e)}

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
                        "database": database,
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
                        "mode": self.mode,
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"쿼리 실행 실패: {str(e)}",
                        "database": database,
                        "query": query,
                    }
            else:
                # 모의 응답
                return {
                    "success": True,
                    "database": database,
                    "query": query,
                    "message": "쿼리가 성공적으로 실행되었습니다 (모의 응답)",
                    "results": [{"id": 1, "name": "Sample Data", "timestamp": time.time()}],
                    "mode": self.mode,
                }
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            return {"success": False, "error": str(e), "database": database}

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
                        "database": database,
                    }

                # 스키마 조회
                try:
                    metadata = await connector.get_metadata()
                    return {
                        "success": True,
                        "database": database,
                        "message": "스키마 정보를 성공적으로 조회했습니다",
                        "metadata": metadata,
                        "mode": self.mode,
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"스키마 조회 실패: {str(e)}",
                        "database": database,
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
                        "products": ["id", "name", "price", "category"],
                    },
                    "mode": self.mode,
                }
        except Exception as e:
            logger.error(f"스키마 조회 실패: {e}")
            return {"success": False, "error": str(e), "database": database}

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
                        "컨텍스트 정보를 추가하세요",
                    ],
                    "status": "completed",
                    "mode": self.mode,
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
                        "고객 세분화 분석을 권장합니다",
                    ],
                    "recommendations": ["정기적인 데이터 품질 검사", "고급 분석 도구 도입 검토"],
                    "status": "completed",
                    "mode": self.mode,
                }

            return analysis_result
        except Exception as e:
            logger.error(f"분석 실행 실패: {e}")
            return {"success": False, "error": str(e), "intent": intent}

    # Analytics 도구 핸들러들
    async def _statistics_analyzer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """통계 분석 도구"""
        try:
            from bridge.analytics.core import StatisticsAnalyzer, UnifiedDataFrame

            data = args.get("data", {})
            raw_columns = args.get("columns")
            columns: List[str] | None = None
            if isinstance(raw_columns, list):
                columns = [str(col) for col in raw_columns]

            analysis_type_value = args.get("analysis_type", "descriptive")
            if isinstance(analysis_type_value, str):
                analysis_type = analysis_type_value.lower()
            else:
                analysis_type = "descriptive"

            # 데이터를 UnifiedDataFrame으로 변환
            df: UnifiedDataFrame
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            elif isinstance(data, UnifiedDataFrame):
                df = data
            else:
                raise ValueError("data must be a mapping or UnifiedDataFrame")

            analyzer = StatisticsAnalyzer()

            result: Any
            if analysis_type == "descriptive":
                result = analyzer.calculate_descriptive_stats(df, columns)
            elif analysis_type == "distribution":
                if not columns:
                    return {
                        "success": False,
                        "error": "분포 분석을 위해 컬럼을 지정해야 합니다",
                    }
                result = analyzer.calculate_distribution_stats(df, columns)
            elif analysis_type == "correlation":
                result = analyzer.calculate_correlation(df, columns)
            elif analysis_type == "summary":
                result = analyzer.generate_summary_report(df)
            else:
                return {"success": False, "error": f"지원하지 않는 분석 유형: {analysis_type}"}

            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "pyarrow":
                logger.warning("pyarrow가 없어 pandas 기반 통계 분석으로 대체합니다")
                return self._statistics_analyzer_fallback(args)
            logger.error(f"통계 분석 실패: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as e:
            logger.error(f"통계 분석 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _data_profiler(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 프로파일링 도구"""
        try:
            from bridge.analytics.core import QualityChecker, StatisticsAnalyzer, UnifiedDataFrame

            data = args.get("data", {})
            include_stats = bool(args.get("include_stats", True))
            include_quality = bool(args.get("include_quality", True))

            # 데이터를 UnifiedDataFrame으로 변환
            df: UnifiedDataFrame
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            elif isinstance(data, UnifiedDataFrame):
                df = data
            else:
                raise ValueError("data must be a mapping or UnifiedDataFrame")

            profile: Dict[str, Any] = {
                "basic_info": {
                    "rows": df.num_rows,
                    "columns": df.num_columns,
                    "memory_usage": df.get_metadata("memory_usage"),
                    "data_types": df.get_metadata("data_types"),
                }
            }

            if include_stats:
                analyzer = StatisticsAnalyzer()
                profile["statistics"] = analyzer.generate_summary_report(df)

            if include_quality:
                checker = QualityChecker()
                profile["quality"] = checker.generate_quality_report(df)

            return {"success": True, "profile": profile, "mode": self.mode}

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "pyarrow":
                logger.warning("pyarrow가 없어 pandas 기반 데이터 프로파일링으로 대체합니다")
                return self._data_profiler_fallback(args)
            logger.error(f"데이터 프로파일링 실패: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as e:
            logger.error(f"데이터 프로파일링 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _outlier_detector(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """이상치 탐지 도구"""
        try:
            from bridge.analytics.core import QualityChecker, UnifiedDataFrame

            data = args.get("data", {})
            raw_columns = args.get("columns", [])
            method_value = args.get("method", "iqr")
            method = method_value if isinstance(method_value, str) else "iqr"
            threshold_value = args.get("threshold", 3.0)
            try:
                threshold = float(threshold_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("threshold 파라미터는 숫자여야 합니다") from exc

            columns: List[str] = []
            if isinstance(raw_columns, list):
                columns = [str(col) for col in raw_columns]

            # 데이터를 UnifiedDataFrame으로 변환
            df: UnifiedDataFrame
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            elif isinstance(data, UnifiedDataFrame):
                df = data
            else:
                raise ValueError("data must be a mapping or UnifiedDataFrame")

            checker = QualityChecker()
            outliers: Dict[str, Any] = {}

            # 컬럼이 지정되지 않은 경우 모든 숫자형 컬럼 분석
            if not columns:
                pandas_df = df.to_pandas()
                numeric_columns = pandas_df.select_dtypes(include=["number"]).columns.tolist()
                columns = numeric_columns

            # QualityChecker의 detect_outliers 메서드 사용
            outlier_results = checker.detect_outliers(df, columns, method, threshold)

            for column, outlier_stats in outlier_results.items():
                outliers[column] = {
                    "method": method,
                    "outlier_count": outlier_stats.outlier_count,
                    "outlier_ratio": outlier_stats.outlier_ratio,
                    "outlier_indices": outlier_stats.outlier_indices[:10],  # 처음 10개만
                    "outlier_values": outlier_stats.outlier_values[:10],  # 처음 10개만
                }

            return {
                "success": True,
                "method": method,
                "threshold": threshold,
                "outliers": outliers,
                "total_columns_analyzed": len(columns),
            }

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "pyarrow":
                logger.warning("pyarrow가 없어 pandas 기반 이상치 탐지로 대체합니다")
                return self._outlier_detector_fallback(args)
            logger.error(f"이상치 탐지 실패: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as e:
            logger.error(f"이상치 탐지 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _chart_generator(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """차트 생성 도구"""
        try:
            from bridge.analytics.core import ChartConfig, ChartGenerator, UnifiedDataFrame

            data = args.get("data", {})
            raw_chart_type = args.get("chart_type")
            if not isinstance(raw_chart_type, str) or not raw_chart_type:
                return {"success": False, "error": "chart_type이 필요합니다"}
            chart_type = raw_chart_type.lower()

            x_column = args.get("x_column")
            y_column = args.get("y_column")
            hue_column = args.get("hue_column")

            raw_title = args.get("title")
            if isinstance(raw_title, str) and raw_title:
                title = raw_title
            else:
                title = f"{chart_type.title()} Chart"

            raw_config = args.get("config")
            config: Dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}

            # 데이터를 UnifiedDataFrame으로 변환
            df: UnifiedDataFrame
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            elif isinstance(data, UnifiedDataFrame):
                df = data
            else:
                raise ValueError("data must be a mapping or UnifiedDataFrame")

            generator = ChartGenerator()
            chart_config = ChartConfig(title=title, **config)

            if chart_type == "bar":
                if not isinstance(x_column, str):
                    return {"success": False, "error": "막대 차트에는 x_column이 필요합니다"}
                y_column_value = y_column if isinstance(y_column, str) else None
                fig = generator.create_bar_chart(df, x_column, y_column_value, chart_config)
            elif chart_type == "line":
                if not isinstance(x_column, str) or not isinstance(y_column, str):
                    return {
                        "success": False,
                        "error": "선 차트를 위해 x_column과 y_column이 필요합니다",
                    }
                fig = generator.create_line_chart(df, x_column, y_column, chart_config)
            elif chart_type == "scatter":
                if not isinstance(x_column, str) or not isinstance(y_column, str):
                    return {
                        "success": False,
                        "error": "산점도를 위해 x_column과 y_column이 필요합니다",
                    }
                hue_value = hue_column if isinstance(hue_column, str) else None
                fig = generator.create_scatter_plot(df, x_column, y_column, hue_value, chart_config)
            elif chart_type == "histogram":
                if not isinstance(x_column, str):
                    return {"success": False, "error": "히스토그램을 위해 x_column이 필요합니다"}
                fig = generator.create_histogram(df, x_column, chart_config)
            elif chart_type == "box":
                if not isinstance(y_column, str):
                    return {"success": False, "error": "박스 플롯을 위해 y_column이 필요합니다"}
                x_value = x_column if isinstance(x_column, str) else None
                fig = generator.create_box_plot(df, x_value, y_column, chart_config)
            elif chart_type == "heatmap":
                raw_columns = args.get("columns")
                columns = None
                if isinstance(raw_columns, list):
                    columns = [str(col) for col in raw_columns]
                fig = generator.create_heatmap(df, chart_config, columns)
            else:
                return {"success": False, "error": f"지원하지 않는 차트 유형: {chart_type}"}

            import base64
            import io

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()

            return {
                "success": True,
                "chart_type": chart_type,
                "title": title,
                "chart_data": chart_data,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) in {"pyarrow", "matplotlib"}:
                logger.warning(
                    "pyarrow/matplotlib 미사용 환경에서 pandas 기반 차트 생성으로 대체합니다"
                )
                return self._chart_generator_fallback(args)
            logger.error(f"차트 생성 실패: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as e:
            logger.error(f"차트 생성 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _quality_checker(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 품질 검사 도구"""
        try:
            from bridge.analytics.core import QualityChecker, UnifiedDataFrame

            data = args.get("data", {})
            check_missing = args.get("check_missing", True)
            check_outliers = args.get("check_outliers", True)
            check_consistency = args.get("check_consistency", True)
            outlier_method = args.get("outlier_method", "iqr")

            # 데이터를 UnifiedDataFrame으로 변환
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            else:
                df = data

            checker = QualityChecker()

            quality_report = checker.generate_quality_report(df, outlier_method)

            # 요청된 검사만 포함
            result = {
                "overall_score": quality_report.overall_score,
                "recommendations": quality_report.recommendations,
                "critical_issues": quality_report.critical_issues,
            }

            if check_missing:
                result["missing_values"] = {
                    "score": quality_report.missing_value_score,
                    "details": "Missing value analysis completed",
                }

            if check_outliers:
                result["outliers"] = {
                    "score": quality_report.outlier_score,
                    "details": "Outlier detection completed",
                }

            if check_consistency:
                result["consistency"] = {
                    "score": quality_report.consistency_score,
                    "details": "Consistency check completed",
                }

            return {
                "success": True,
                "quality_report": result,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "pyarrow":
                logger.warning("pyarrow가 없어 pandas 기반 품질 검사로 대체합니다")
                return self._quality_checker_fallback(args)
            logger.error(f"품질 검사 실패: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as e:
            logger.error(f"품질 검사 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _report_builder(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """리포트 생성 도구"""
        try:
            from bridge.analytics.core import ReportConfig, ReportGenerator, UnifiedDataFrame

            data = args.get("data", {})
            title = args.get("title", "Analytics Report")
            author = args.get("author")
            include_charts = args.get("include_charts", True)
            include_dashboard = args.get("include_dashboard", True)
            include_quality = args.get("include_quality", True)

            # 데이터를 UnifiedDataFrame으로 변환
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            else:
                df = data

            report_config = ReportConfig(title=title, author=author)

            generator = ReportGenerator()
            report = generator.generate_analytics_report(df, report_config)

            # 요청된 구성 요소만 포함
            result = {
                "title": report["title"],
                "author": report["author"],
                "date": report["date"],
                "basic_stats": report["basic_stats"],
                "column_stats": report["column_stats"],
            }

            if include_charts:
                # 차트는 메타데이터만 포함 (실제 이미지 데이터는 별도 처리)
                result["charts"] = {
                    "available_charts": list(report["charts"].keys()),
                    "chart_count": len(report["charts"]),
                }

            if include_dashboard:
                result["dashboard"] = {"available": True, "title": "Analytics Dashboard"}

            if include_quality:
                # 품질 검사는 별도로 실행
                from bridge.analytics.core import QualityChecker

                checker = QualityChecker()
                quality_report = checker.generate_quality_report(df)
                result["quality"] = {
                    "overall_score": quality_report.overall_score,
                    "recommendations": quality_report.recommendations,
                    "critical_issues": quality_report.critical_issues,
                }

            return {
                "success": True,
                "report": result,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) in {"pyarrow", "matplotlib"}:
                logger.warning(
                    "pyarrow/matplotlib 미사용 환경에서 pandas 기반 리포트 생성으로 대체합니다"
                )
                return self._report_builder_fallback(args)
            logger.error(f"리포트 생성 실패: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            return {"success": False, "error": str(e)}

    # CA 마일스톤 3.1: 통합 데이터 분석 레이어 도구들 구현

    async def _data_unifier(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 통합 도구"""
        try:
            from bridge.analytics.core import DataUnifier

            data_sources = args.get("data_sources", {})
            schema_mapping = args.get("schema_mapping")
            merge_strategy = args.get("merge_strategy", "union")

            if not data_sources:
                return {"success": False, "error": "데이터 소스가 제공되지 않았습니다"}

            # DataUnifier 인스턴스 생성
            unifier = DataUnifier()

            # 데이터 통합 실행
            unified_data = unifier.unify_data_sources(
                data_sources, schema_mapping, merge_strategy
            )

            # 결과 요약 생성
            summary = unifier.create_data_summary(unified_data)
            validation = unifier.validate_unified_data(unified_data)

            return {
                "success": True,
                "message": "데이터 통합이 성공적으로 완료되었습니다",
                "unified_data": {
                    "rows": unified_data.num_rows,
                    "columns": unified_data.num_columns,
                    "column_names": unified_data.column_names,
                },
                "summary": summary,
                "validation": validation,
                "mode": self.mode,
            }
        except Exception as e:
            logger.error(f"데이터 통합 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _schema_mapper(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """스키마 매핑 도구"""
        try:
            from bridge.analytics.core import SchemaMapper
            import pyarrow as pa

            source_name = args.get("source_name", "")
            source_schema_data = args.get("source_schema", {})
            target_schema_data = args.get("target_schema")
            mapping_rules = args.get("mapping_rules", {})
            auto_detect = args.get("auto_detect", True)

            if not source_name or not source_schema_data:
                return {"success": False, "error": "소스명과 원본 스키마가 필요합니다"}

            # 스키마 객체 생성
            source_schema = pa.schema([
                pa.field(name, pa.DataType.from_string(type_str))
                for name, type_str in source_schema_data.items()
            ])

            target_schema = None
            if target_schema_data:
                target_schema = pa.schema([
                    pa.field(name, pa.DataType.from_string(type_str))
                    for name, type_str in target_schema_data.items()
                ])

            # SchemaMapper 인스턴스 생성
            mapper = SchemaMapper()

            # 스키마 매핑 생성
            schema_mapping = mapper.create_mapping(
                source_name, source_schema, target_schema, mapping_rules, auto_detect
            )

            # 매핑 검증
            validation = mapper.validate_mapping(source_name)

            return {
                "success": True,
                "message": "스키마 매핑이 성공적으로 생성되었습니다",
                "mapping": {
                    "source_name": schema_mapping.source_name,
                    "total_mappings": len(schema_mapping.column_mappings),
                    "column_mappings": [
                        {
                            "source_column": mapping.source_column,
                            "target_column": mapping.target_column,
                            "data_type": mapping.data_type,
                            "is_required": mapping.is_required,
                        }
                        for mapping in schema_mapping.column_mappings
                    ],
                },
                "validation": validation,
                "mode": self.mode,
            }
        except Exception as e:
            logger.error(f"스키마 매핑 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _type_converter(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 타입 변환 도구"""
        try:
            from bridge.analytics.core import TypeConverter
            import pyarrow as pa

            data = args.get("data", {})
            target_schema_data = args.get("target_schema", {})
            column_mappings = args.get("column_mappings", {})
            conversion_rules = args.get("conversion_rules", [])

            if not data or not target_schema_data:
                return {"success": False, "error": "데이터와 대상 스키마가 필요합니다"}

            # 데이터를 Arrow Table로 변환
            if isinstance(data, dict):
                table = pa.Table.from_pylist([data])
            elif isinstance(data, list):
                table = pa.Table.from_pylist(data)
            else:
                table = data

            # 대상 스키마 생성
            def get_arrow_type(type_str):
                if type_str == "int64":
                    return pa.int64()
                elif type_str == "string":
                    return pa.string()
                elif type_str == "float64":
                    return pa.float64()
                elif type_str == "bool":
                    return pa.bool_()
                elif type_str == "timestamp":
                    return pa.timestamp("us")
                elif type_str == "date":
                    return pa.date32()
                else:
                    return pa.string()
            
            target_schema = pa.schema([
                pa.field(name, get_arrow_type(type_str))
                for name, type_str in target_schema_data.items()
            ])

            # TypeConverter 인스턴스 생성
            converter = TypeConverter()

            # 사용자 정의 변환 규칙 추가
            for rule_data in conversion_rules:
                from bridge.analytics.core import ConversionRule
                rule = ConversionRule(**rule_data)
                converter.add_conversion_rule(rule)

            # 데이터 타입 변환 실행
            converted_table = converter.convert_table(table, target_schema, column_mappings)

            return {
                "success": True,
                "message": "데이터 타입 변환이 성공적으로 완료되었습니다",
                "converted_data": {
                    "rows": len(converted_table),
                    "columns": len(converted_table.schema),
                    "column_types": {
                        name: str(field.type)
                        for name, field in zip(converted_table.schema.names, converted_table.schema)
                    },
                },
                "mode": self.mode,
            }
        except Exception as e:
            logger.error(f"데이터 타입 변환 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _streaming_processor(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """스트리밍 처리 도구"""
        try:
            from bridge.analytics.core import StreamingProcessor
            import pyarrow as pa

            data = args.get("data", {})
            processor_func = args.get("processor_func", "")
            chunk_size = args.get("chunk_size", 10000)
            memory_limit_mb = args.get("memory_limit_mb", 1000)
            parameters = args.get("parameters", {})

            if not data or not processor_func:
                return {"success": False, "error": "데이터와 처리 함수가 필요합니다"}

            # 데이터를 Arrow Table로 변환
            if isinstance(data, dict):
                table = pa.Table.from_pylist([data])
            elif isinstance(data, list):
                table = pa.Table.from_pylist(data)
            else:
                table = data

            # StreamingProcessor 인스턴스 생성
            processor = StreamingProcessor(chunk_size, memory_limit_mb)

            # 처리 함수 정의
            def process_func(table_chunk, **kwargs):
                # 간단한 처리 함수 (실제로는 더 복잡한 로직이 들어갈 수 있음)
                return table_chunk

            # 스트리밍 처리 실행
            processed_table = processor.process_large_table(
                table, process_func, **parameters
            )

            # 메모리 사용량 분석
            memory_usage = processor.get_memory_usage(processed_table)

            return {
                "success": True,
                "message": "스트리밍 처리가 성공적으로 완료되었습니다",
                "processed_data": {
                    "rows": len(processed_table),
                    "columns": len(processed_table.schema),
                },
                "memory_usage": memory_usage,
                "processing_info": {
                    "chunk_size": chunk_size,
                    "memory_limit_mb": memory_limit_mb,
                    "processor_func": processor_func,
                },
                "mode": self.mode,
            }
        except Exception as e:
            logger.error(f"스트리밍 처리 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _integrated_data_layer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """통합 데이터 분석 레이어 메인 도구"""
        try:
            from bridge.analytics.core import IntegratedDataLayer

            action = args.get("action", "")
            data_sources = args.get("data_sources", {})
            transformations = args.get("transformations", [])
            export_format = args.get("export_format", "arrow")
            file_path = args.get("file_path")
            chunk_size = args.get("chunk_size", 10000)
            memory_limit_mb = args.get("memory_limit_mb", 1000)

            if not action:
                return {"success": False, "error": "수행할 작업이 지정되지 않았습니다"}

            # IntegratedDataLayer 인스턴스 생성
            layer = IntegratedDataLayer(chunk_size, memory_limit_mb)

            if action == "integrate":
                if not data_sources:
                    return {"success": False, "error": "통합할 데이터 소스가 필요합니다"}

                # 데이터 통합
                unified_data = layer.integrate_data_sources(data_sources)
                summary = layer.get_data_summary()
                validation = layer.validate_data_quality()

                return {
                    "success": True,
                    "message": "데이터 통합이 성공적으로 완료되었습니다",
                    "action": action,
                    "unified_data": {
                        "rows": unified_data.num_rows,
                        "columns": unified_data.num_columns,
                        "column_names": unified_data.column_names,
                    },
                    "summary": summary,
                    "validation": validation,
                    "mode": self.mode,
                }

            elif action == "transform":
                if not transformations:
                    return {"success": False, "error": "적용할 변환이 필요합니다"}

                # 데이터 변환
                transformed_data = layer.apply_data_transformations(transformations)

                return {
                    "success": True,
                    "message": "데이터 변환이 성공적으로 완료되었습니다",
                    "action": action,
                    "transformed_data": {
                        "rows": transformed_data.num_rows,
                        "columns": transformed_data.num_columns,
                    },
                    "mode": self.mode,
                }

            elif action == "export":
                # 데이터 내보내기
                exported_data = layer.export_data(export_format, file_path)

                return {
                    "success": True,
                    "message": f"데이터가 성공적으로 내보내졌습니다 ({export_format})",
                    "action": action,
                    "export_format": export_format,
                    "file_path": file_path,
                    "exported_data": str(exported_data)[:500] + "..." if len(str(exported_data)) > 500 else str(exported_data),
                    "mode": self.mode,
                }

            elif action == "validate":
                # 데이터 품질 검증
                validation = layer.validate_data_quality()

                return {
                    "success": True,
                    "message": "데이터 품질 검증이 완료되었습니다",
                    "action": action,
                    "validation": validation,
                    "mode": self.mode,
                }

            elif action == "summary":
                # 데이터 요약
                summary = layer.get_data_summary()
                memory_usage = layer.get_memory_usage()

                return {
                    "success": True,
                    "message": "데이터 요약이 생성되었습니다",
                    "action": action,
                    "summary": summary,
                    "memory_usage": memory_usage,
                    "mode": self.mode,
                }

            else:
                return {
                    "success": False,
                    "error": f"지원하지 않는 작업: {action}",
                    "supported_actions": ["integrate", "transform", "export", "validate", "summary"],
                }

        except Exception as e:
            logger.error(f"통합 데이터 분석 레이어 작업 실패: {e}")
            return {"success": False, "error": str(e)}

    def _ensure_dataframe(self, data: Any) -> pd.DataFrame:
        """입력 데이터를 pandas DataFrame으로 정규화한다."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if hasattr(data, "to_pandas"):
            try:
                return data.to_pandas().copy()
            except Exception:
                pass
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
        if data is None:
            return pd.DataFrame()
        raise ValueError("지원하지 않는 데이터 형식입니다")

    def _select_numeric_columns(self, df: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
        numeric_df = df.select_dtypes(include=[np.number])
        if columns:
            return [col for col in columns if col in numeric_df.columns]
        return list(numeric_df.columns)

    def _statistics_analyzer_fallback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """pyarrow 없이 pandas 기반으로 통계 분석을 수행한다."""
        df = self._ensure_dataframe(args.get("data", {}))
        analysis_type = str(args.get("analysis_type", "descriptive")).lower()
        raw_columns = args.get("columns")
        columns = raw_columns if isinstance(raw_columns, list) else None
        numeric_columns = self._select_numeric_columns(df, columns)

        result: Any
        if analysis_type == "descriptive":
            if not numeric_columns:
                result = {}
            else:
                describe = df[numeric_columns].describe().to_dict()
                result = {
                    column: {key: float(value) for key, value in stats.items()}
                    for column, stats in describe.items()
                }
        elif analysis_type == "distribution":
            distribution: Dict[str, Dict[str, float]] = {}
            for column in numeric_columns:
                series = df[column].dropna()
                if series.empty:
                    distribution[column] = {
                        "skewness": 0.0,
                        "kurtosis": 0.0,
                        "variance": 0.0,
                        "range": 0.0,
                        "iqr": 0.0,
                    }
                    continue
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                distribution[column] = {
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "variance": float(series.var()),
                    "range": float(series.max() - series.min()),
                    "iqr": float(q3 - q1),
                }
            result = distribution
        elif analysis_type == "correlation":
            if len(numeric_columns) < 2:
                corr_matrix = pd.DataFrame()
                strong: List[Dict[str, Any]] = []
                moderate: List[Dict[str, Any]] = []
            else:
                corr_matrix = df[numeric_columns].corr().fillna(0.0)
                strong = []
                moderate = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_value = float(corr_matrix.iloc[i, j])
                        info = {
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr_value,
                            "abs_correlation": abs(corr_value),
                        }
                        if abs(corr_value) > 0.7:
                            strong.append(info)
                        elif abs(corr_value) > 0.3:
                            moderate.append(info)
            result = {
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": strong,
                "moderate_correlations": moderate,
            }
        elif analysis_type == "summary":
            summary = df.describe(include="all").fillna(0).to_dict()
            correlation = df[numeric_columns].corr().fillna(0).to_dict() if numeric_columns else {}
            result = {"summary": summary, "correlation": correlation}
        else:
            return {
                "success": False,
                "error": f"지원하지 않는 분석 유형: {analysis_type}",
            }

        return {
            "success": True,
            "analysis_type": analysis_type,
            "result": result,
            "data_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }

    def _data_profiler_fallback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """pyarrow 없이 pandas 기반 데이터 프로파일링을 수행한다."""
        df = self._ensure_dataframe(args.get("data", {}))
        include_stats = bool(args.get("include_stats", True))
        include_quality = bool(args.get("include_quality", True))

        profile: Dict[str, Any] = {
            "basic_info": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
            }
        }

        if include_stats:
            profile["statistics"] = df.describe(include="all").fillna(0).to_dict()

        if include_quality:
            quality = self._quality_checker_fallback({"data": df})["quality_report"]
            profile["quality"] = quality

        return {"success": True, "profile": profile, "mode": self.mode}

    def _outlier_detector_fallback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """pyarrow 없이 pandas 기반 이상치 탐지를 수행한다."""
        df = self._ensure_dataframe(args.get("data", {}))
        raw_columns = args.get("columns")
        columns = raw_columns if isinstance(raw_columns, list) else None
        method = str(args.get("method", "iqr")).lower()
        threshold = float(args.get("threshold", 1.5 if method == "iqr" else 3.0))
        numeric_columns = self._select_numeric_columns(df, columns)

        outliers: Dict[str, Any] = {}
        for column in numeric_columns:
            series = df[column].dropna()
            if series.empty:
                outliers[column] = {
                    "method": method,
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                    "outlier_indices": [],
                    "outlier_values": [],
                }
                continue

            if method == "zscore":
                std = series.std()
                if std == 0:
                    mask = pd.Series(False, index=series.index)
                else:
                    mask = ((series - series.mean()).abs() / std) > threshold
            else:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    mask = pd.Series(False, index=series.index)
                else:
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    mask = (series < lower) | (series > upper)

            indices = mask[mask].index.tolist()
            outliers[column] = {
                "method": method,
                "outlier_count": int(len(indices)),
                "outlier_ratio": float(len(indices) / len(series)) if len(series) else 0.0,
                "outlier_indices": indices[:10],
                "outlier_values": series.loc[indices[:10]].tolist(),
            }

        return {
            "success": True,
            "method": method,
            "threshold": threshold,
            "outliers": outliers,
            "total_columns_analyzed": len(numeric_columns),
        }

    def _chart_generator_fallback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """pyarrow 또는 matplotlib 없이 기본 차트 응답을 생성한다."""
        df = self._ensure_dataframe(args.get("data", {}))
        raw_chart_type = args.get("chart_type")
        if not isinstance(raw_chart_type, str) or not raw_chart_type:
            return {"success": False, "error": "chart_type이 필요합니다"}
        chart_type = raw_chart_type.lower()

        x_column = args.get("x_column")
        y_column = args.get("y_column")
        title = args.get("title") or f"{chart_type.title()} Chart"

        # 최소한의 컬럼 검증만 수행한다.
        if chart_type in {"bar", "histogram"} and not isinstance(x_column, str):
            return {"success": False, "error": "막대/히스토그램 차트에는 x_column이 필요합니다"}
        if chart_type in {"line", "scatter"} and (
            not isinstance(x_column, str) or not isinstance(y_column, str)
        ):
            return {
                "success": False,
                "error": "선/산점도 차트를 위해 x_column과 y_column이 필요합니다",
            }
        if chart_type == "box" and not isinstance(y_column, str):
            return {"success": False, "error": "박스 플롯을 위해 y_column이 필요합니다"}

        return {
            "success": True,
            "chart_type": chart_type,
            "title": title,
            "chart_data": PLACEHOLDER_CHART_DATA,
            "data_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }

    def _quality_checker_fallback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """pyarrow 없이 pandas 기반 품질 검사를 수행한다."""
        df = self._ensure_dataframe(args.get("data", {}))
        check_missing = args.get("check_missing", True)
        check_outliers = args.get("check_outliers", True)
        check_consistency = args.get("check_consistency", True)
        outlier_method = str(args.get("outlier_method", "iqr")).lower()

        numeric_columns = self._select_numeric_columns(df, None)
        missing_ratio = float(df.isnull().mean().mean()) if not df.empty else 0.0
        duplicate_ratio = float(df.duplicated().mean()) if len(df) else 0.0

        report: Dict[str, Any] = {
            "overall_score": max(
                0.0,
                100.0 - missing_ratio * 40.0 - duplicate_ratio * 30.0,
            ),
            "recommendations": [],
            "critical_issues": [],
        }

        if check_missing:
            missing_details = {column: int(df[column].isnull().sum()) for column in df.columns}
            report["missing_values"] = {
                "score": max(0.0, 100.0 - missing_ratio * 100.0),
                "details": missing_details,
            }
            if missing_ratio > 0.1:
                report["recommendations"].append("결측값 처리를 통해 데이터 품질을 개선하세요.")

        if check_outliers and numeric_columns:
            fallback_args = {
                "data": df,
                "columns": numeric_columns,
                "method": outlier_method,
                "threshold": args.get("threshold", 3.0 if outlier_method == "zscore" else 1.5),
            }
            outlier_info = self._outlier_detector_fallback(fallback_args)
            report["outliers"] = {
                "score": max(
                    0.0,
                    100.0
                    - sum(details["outlier_count"] for details in outlier_info["outliers"].values())
                    / max(len(df), 1)
                    * 100.0,
                ),
                "details": outlier_info["outliers"],
            }

        if check_consistency:
            schema_issues = []
            for column in df.columns:
                if df[column].dtype == object and df[column].map(type).nunique() > 1:
                    schema_issues.append(f"컬럼 '{column}'에 혼합된 타입이 존재합니다")
            report["consistency"] = {
                "score": max(0.0, 100.0 - duplicate_ratio * 50.0 - len(schema_issues) * 5.0),
                "duplicate_ratio": duplicate_ratio,
                "schema_issues": schema_issues,
            }
            if schema_issues:
                report["critical_issues"].extend(schema_issues)

        return {
            "success": True,
            "quality_report": report,
            "data_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }

    def _report_builder_fallback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """pyarrow 없이 pandas 기반 리포트를 생성한다."""
        df = self._ensure_dataframe(args.get("data", {}))
        title = args.get("title", "Analytics Report")
        author = args.get("author")
        include_charts = args.get("include_charts", True)
        include_dashboard = args.get("include_dashboard", True)
        include_quality = args.get("include_quality", True)

        basic_stats = df.describe(include="all").fillna(0).to_dict() if not df.empty else {}
        column_stats = {
            column: {
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isnull().sum()),
            }
            for column in df.columns
        }

        report: Dict[str, Any] = {
            "title": title,
            "author": author,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "basic_stats": basic_stats,
            "column_stats": column_stats,
        }

        if include_charts:
            report["charts"] = {
                "available_charts": ["bar", "scatter", "histogram"],
                "chart_count": 3,
            }

        if include_dashboard:
            report["dashboard"] = {"available": True, "title": "Analytics Dashboard"}

        if include_quality:
            quality = self._quality_checker_fallback({"data": df})["quality_report"]
            report["quality"] = quality

        return {
            "success": True,
            "report": report,
            "data_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }

    def _cleanup_resources(self):
        """리소스 정리"""
        logger.info("Cleaning up resources...")
        self.is_initialized = False
        self.is_running = False
        self.request_count = 0

        # 커넥터 정리
        for connector in self.connectors.values():
            if hasattr(connector, "close"):
                try:
                    asyncio.create_task(connector.close())
                except Exception as e:
                    logger.warning(f"커넥터 정리 중 오류: {e}")

    async def run_sdk_mode(self):
        """MCP SDK 모드로 실행"""
        logger.info(f"Starting Unified Bridge MCP Server (SDK mode) - Mode: {self.mode}")
        self.is_running = True

        if (
            self.server is None
            or self.stdio_server is None
            or self.InitializationOptions is None
            or self.NotificationOptions is None
        ):
            raise RuntimeError("MCP SDK가 초기화되지 않았습니다")

        server = self.server
        stdio_server = self.stdio_server
        init_options_cls = self.InitializationOptions
        notification_options_cls = self.NotificationOptions

        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    init_options_cls(
                        server_name=self.server_name,
                        server_version=self.server_version,
                        capabilities=server.get_capabilities(
                            notification_options=notification_options_cls(),
                            experimental_capabilities={},
                        ),
                    ),
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
                    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

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


    # CA 마일스톤 3.2: 고급 통계 분석 및 시각화 도구 핸들러들
    async def _advanced_statistics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """고급 통계 분석 도구"""
        try:
            from bridge.analytics.core import AdvancedStatistics, UnifiedDataFrame

            data = args.get("data", {})
            analysis_type = args.get("analysis_type", "descriptive")
            columns = args.get("columns", [])

            # 데이터를 UnifiedDataFrame으로 변환
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            else:
                df = data

            # AdvancedStatistics 인스턴스 생성
            analyzer = AdvancedStatistics()

            # 분석 유형에 따른 처리
            if analysis_type == "descriptive":
                result = analyzer.descriptive_statistics(df, columns)
            elif analysis_type == "correlation":
                result = analyzer.correlation_analysis(df, columns)
            elif analysis_type == "distribution":
                result = analyzer.distribution_analysis(df, columns)
            elif analysis_type == "summary":
                result = analyzer.comprehensive_summary(df, columns)
            else:
                return {"success": False, "error": f"지원하지 않는 분석 유형: {analysis_type}"}

            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except Exception as e:
            logger.error(f"고급 통계 분석 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _interactive_charts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """인터랙티브 차트 생성 도구"""
        try:
            from bridge.analytics.core import AdvancedVisualization, UnifiedDataFrame

            data = args.get("data", {})
            chart_type = args.get("chart_type", "bar")
            x_column = args.get("x_column")
            y_column = args.get("y_column")
            hue_column = args.get("hue_column")
            title = args.get("title", f"{chart_type.title()} Chart")
            config = args.get("config", {})

            # 데이터를 UnifiedDataFrame으로 변환
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            else:
                df = data

            # AdvancedVisualization 인스턴스 생성
            viz = AdvancedVisualization()

            # 차트 생성
            chart_data = viz.create_interactive_chart(
                df, chart_type, x_column, y_column, hue_column, title, config
            )

            return {
                "success": True,
                "chart_type": chart_type,
                "title": title,
                "chart_data": chart_data,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except Exception as e:
            logger.error(f"인터랙티브 차트 생성 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _statistical_tests(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """통계적 검정 도구"""
        try:
            from bridge.analytics.core import StatisticalTests, UnifiedDataFrame

            data = args.get("data", {})
            test_type = args.get("test_type", "t_test")
            x_column = args.get("x_column")
            y_column = args.get("y_column")
            group_column = args.get("group_column")
            alpha = args.get("alpha", 0.05)

            # 데이터를 UnifiedDataFrame으로 변환
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            else:
                df = data

            # StatisticalTests 인스턴스 생성
            tester = StatisticalTests()

            # 검정 유형에 따른 처리
            if test_type == "t_test":
                result = tester.t_test(df, x_column, y_column, alpha)
            elif test_type == "chi_square":
                result = tester.chi_square_test(df, x_column, y_column, alpha)
            elif test_type == "anova":
                result = tester.anova_test(df, x_column, group_column, alpha)
            elif test_type == "regression":
                result = tester.regression_analysis(df, x_column, y_column, alpha)
            elif test_type == "ab_test":
                result = tester.ab_test(df, x_column, y_column, group_column, alpha)
            else:
                return {"success": False, "error": f"지원하지 않는 검정 유형: {test_type}"}

            return {
                "success": True,
                "test_type": test_type,
                "result": result,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except Exception as e:
            logger.error(f"통계적 검정 실패: {e}")
            return {"success": False, "error": str(e)}

    async def _time_series_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """시계열 분석 도구"""
        try:
            from bridge.analytics.core import TimeSeriesAnalysis, UnifiedDataFrame

            data = args.get("data", {})
            time_column = args.get("time_column")
            value_column = args.get("value_column")
            analysis_type = args.get("analysis_type", "decompose")
            method = args.get("method")
            forecast_periods = args.get("forecast_periods", 12)
            threshold = args.get("threshold", 3.0)

            if not time_column or not value_column:
                return {"success": False, "error": "time_column과 value_column이 필요합니다"}

            # 데이터를 UnifiedDataFrame으로 변환
            if isinstance(data, dict):
                df = UnifiedDataFrame(data)
            else:
                df = data

            # TimeSeriesAnalysis 인스턴스 생성
            analyzer = TimeSeriesAnalysis()

            # 분석 유형에 따른 처리
            if analysis_type == "decompose":
                result = analyzer.decompose_time_series(df, time_column, value_column, method)
            elif analysis_type == "trend":
                result = analyzer.detect_trend(df, time_column, value_column, method)
            elif analysis_type == "seasonality":
                result = analyzer.detect_seasonality(df, time_column, value_column)
            elif analysis_type == "forecast":
                result = analyzer.forecast_time_series(df, time_column, value_column, forecast_periods, method)
            elif analysis_type == "anomalies":
                result = analyzer.analyze_anomalies(df, time_column, value_column, method, threshold)
            else:
                return {"success": False, "error": f"지원하지 않는 분석 유형: {analysis_type}"}

            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "data_shape": {"rows": df.num_rows, "columns": df.num_columns},
            }

        except Exception as e:
            logger.error(f"시계열 분석 실패: {e}")
            return {"success": False, "error": str(e)}


def run():
    """동기 래퍼 함수 - 콘솔 스크립트 진입점"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    server = UnifiedBridgeMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    run()
