# Bridge MCP 서버 구현 계획

## 개요

Bridge 프로젝트를 Cursor IDE에서 사용할 수 있는 MCP(Model Context Protocol) 서버로 변환하는 계획을 설명합니다. **현재는 구현되지 않은 상태**이며, 향후 개발 예정입니다.

## 현재 상태 vs MCP 서버 요구사항

### 현재 Bridge 구조
```
FastAPI App → REST API → Celery Tasks → Connectors → Databases
```

### MCP 서버 구조 (목표)
```
Cursor IDE → JSON-RPC → MCP Server → Bridge Services → Databases
```

## MCP 서버 구현 계획 (향후 개발)

### 1. MCP 서버 래퍼 생성 (계획)

```python
# src/bridge/mcp_server.py
import asyncio
import json
from typing import Any, Dict, List, Optional
from mcp import Server, types
from mcp.server import stdio
from mcp.server.models import InitializationOptions

from .orchestrator.app import app as fastapi_app
from .connectors.postgres import PostgresConnector
from .semantic.models import TaskRequest, TaskResponse

class BridgeMCPServer:
    def __init__(self):
        self.server = Server("bridge-mcp")
        self.connectors = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """MCP 핸들러 설정"""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """사용 가능한 도구 목록 반환"""
            return [
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
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """도구 실행"""
            if name == "query_database":
                return await self._execute_query(arguments)
            elif name == "get_schema":
                return await self._get_schema(arguments)
            elif name == "analyze_data":
                return await self._analyze_data(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def _execute_query(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """데이터베이스 쿼리 실행"""
        database = args["database"]
        query = args["query"]
        params = args.get("params", {})
        
        # 커넥터 가져오기 또는 생성
        if database not in self.connectors:
            self.connectors[database] = PostgresConnector(
                name=database,
                settings={"database": database}
            )
        
        connector = self.connectors[database]
        results = []
        
        async for row in connector.run_query(query, params):
            results.append(dict(row))
        
        return [types.TextContent(
            type="text",
            text=json.dumps(results, indent=2, ensure_ascii=False)
        )]
    
    async def _get_schema(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """스키마 정보 조회"""
        database = args["database"]
        
        if database not in self.connectors:
            self.connectors[database] = PostgresConnector(
                name=database,
                settings={"database": database}
            )
        
        connector = self.connectors[database]
        metadata = await connector.get_metadata()
        
        return [types.TextContent(
            type="text",
            text=json.dumps(metadata, indent=2, ensure_ascii=False)
        )]
    
    async def _analyze_data(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """데이터 분석 실행"""
        intent = args["intent"]
        sources = args.get("sources", [])
        context = args.get("context", {})
        
        # TaskRequest 생성
        task_request = TaskRequest(
            intent=intent,
            sources=sources,
            context=context
        )
        
        # FastAPI 앱을 통해 작업 실행
        # 실제로는 Celery 태스크를 호출
        from .orchestrator.tasks import execute_pipeline
        result = execute_pipeline.delay(task_request.model_dump())
        
        return [types.TextContent(
            type="text",
            text=f"분석 작업이 큐에 추가되었습니다. Job ID: {result.id}"
        )]

async def main():
    """MCP 서버 실행"""
    bridge_server = BridgeMCPServer()
    
    async with stdio.stdio_server() as (read_stream, write_stream):
        await bridge_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="bridge-mcp",
                server_version="0.1.0",
                capabilities=bridge_server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 의존성 추가 (계획)

```toml
# pyproject.toml에 추가 예정
dependencies = [
    # ... 기존 의존성들
    "mcp>=1.0.0",  # MCP 프로토콜 라이브러리
    "mcp-server-stdio>=1.0.0",  # stdio 서버
]
```

**참고**: 현재 `pyproject.toml`에는 MCP 관련 의존성이 포함되어 있지 않습니다.

### 3. Cursor IDE 설정 (계획)

```json
// .cursor/settings.json (향후 설정)
{
  "mcp": {
    "servers": {
      "bridge": {
        "command": "python",
        "args": ["-m", "src.bridge.mcp_server"],
        "env": {
          "BRIDGE_DATABASE_URL": "postgresql://user:pass@localhost/db"
        }
      }
    }
  }
}
```

### 4. 실행 스크립트 (계획)

```python
# scripts/run_mcp_server.py (향후 생성)
#!/usr/bin/env python3
"""Bridge MCP 서버 실행 스크립트"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge.mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())
```

## 사용 예시 (계획)

### Cursor IDE에서 사용 (향후 구현)

1. **데이터베이스 쿼리 실행**
```python
# Cursor IDE에서 MCP 도구 호출 (계획)
result = await mcp.call_tool("query_database", {
    "database": "analytics",
    "query": "SELECT * FROM users WHERE created_at > %s",
    "params": {"created_at": "2024-01-01"}
})
```

2. **스키마 정보 조회**
```python
schema = await mcp.call_tool("get_schema", {
    "database": "analytics"
})
```

3. **데이터 분석**
```python
analysis = await mcp.call_tool("analyze_data", {
    "intent": "고객 세그먼트 분석",
    "sources": ["analytics"],
    "context": {"time_range": "2024-01-01 to 2024-12-31"}
})
```

## 장점

1. **기존 Bridge 기능 활용**: 커넥터, 오케스트레이터, 시맨틱 모델 재사용
2. **Cursor IDE 통합**: AI 코딩 어시스턴트와 직접 통신
3. **표준화된 인터페이스**: MCP 프로토콜을 통한 일관된 도구 사용
4. **확장성**: 새로운 도구와 리소스 쉽게 추가

## 다음 단계 (구현 계획)

1. **MCP 서버 구현**: 위의 코드를 기반으로 실제 구현
2. **의존성 추가**: `pyproject.toml`에 MCP 관련 의존성 추가
3. **테스트**: Cursor IDE와의 통합 테스트
4. **문서화**: 사용자 가이드 및 API 문서 작성
5. **최적화**: 성능 및 안정성 개선

**현재 상태**: MCP 서버는 구현되지 않았으며, 향후 개발 예정입니다.
