# Bridge MCP 설치 및 사용 가이드

## 📖 개요

이 문서는 AI 에이전트에서 Bridge MCP(Model Context Protocol) 서버를 설치하고 사용하는 방법을 설명합니다. Claude Desktop, Cursor, 그리고 기타 MCP 호환 클라이언트에서 Bridge의 데이터 분석 기능을 활용할 수 있습니다.

## 🔧 MCP 서버 설치

### 1. Bridge MCP 서버 다운로드

```bash
# Bridge 저장소 클론
git clone https://github.com/your-org/bridge.git
cd bridge

# 의존성 설치
make install
```

### 2. 환경 설정

```bash
# 환경 변수 설정
export BRIDGE_MCP_MODE=production  # 또는 development, real, mock
export BRIDGE_DATABASE_URL=postgresql://user:password@localhost:5432/bridge
export BRIDGE_REDIS_URL=redis://localhost:6379/0
```

### 3. MCP 서버 실행

```bash
# 통합 MCP 서버 실행
python -m bridge.mcp_server_unified

# 또는 특정 모드로 실행
BRIDGE_MCP_MODE=development python -m bridge.mcp_server_unified
```

## 🤖 AI 클라이언트 설정

### Claude Desktop 설정

#### 1. Claude Desktop 설정 파일 위치

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/claude/claude_desktop_config.json
```

#### 2. 설정 파일 구성

```json
{
  "mcpServers": {
    "bridge": {
      "command": "python",
      "args": [
        "-m", "bridge.mcp_server_unified"
      ],
      "env": {
        "BRIDGE_MCP_MODE": "production",
        "BRIDGE_DATABASE_URL": "postgresql://user:password@localhost:5432/bridge",
        "BRIDGE_REDIS_URL": "redis://localhost:6379/0"
      },
      "cwd": "/path/to/bridge"
    }
  }
}
```

#### 3. Claude Desktop 재시작

설정 파일을 저장한 후 Claude Desktop을 재시작하면 Bridge MCP 서버가 연결됩니다.

### Cursor 설정

#### 1. Cursor 설정 파일 위치

**Windows:**
```
%APPDATA%\Cursor\User\globalStorage\cursor.mcp\config.json
```

**macOS:**
```
~/Library/Application Support/Cursor/User/globalStorage/cursor.mcp/config.json
```

**Linux:**
```
~/.config/Cursor/User/globalStorage/cursor.mcp/config.json
```

#### 2. 설정 파일 구성

```json
{
  "mcpServers": {
    "bridge": {
      "command": "python",
      "args": [
        "-m", "bridge.mcp_server_unified"
      ],
      "env": {
        "BRIDGE_MCP_MODE": "production",
        "BRIDGE_DATABASE_URL": "postgresql://user:password@localhost:5432/bridge",
        "BRIDGE_REDIS_URL": "redis://localhost:6379/0"
      },
      "cwd": "/path/to/bridge"
    }
  }
}
```

### 기타 MCP 클라이언트 설정

#### MCP Inspector 사용

```bash
# MCP Inspector 설치
npm install -g @modelcontextprotocol/inspector

# Bridge MCP 서버 연결
npx @modelcontextprotocol/inspector python -m src.bridge.mcp_server_unified
```

#### Python MCP 클라이언트

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # MCP 서버 연결
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "bridge.mcp_server_unified"],
        env={
            "BRIDGE_MCP_MODE": "production",
            "BRIDGE_DATABASE_URL": "postgresql://user:password@localhost:5432/bridge"
        }
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 서버 초기화
            await session.initialize()
            
            # 사용 가능한 도구 목록 조회
            tools = await session.list_tools()
            print("사용 가능한 도구:", tools)
            
            # 통계 분석 도구 사용
            result = await session.call_tool(
                "statistics_analyzer",
                {
                    "data_source": "postgresql://analytics_db",
                    "table_name": "sales",
                    "columns": ["amount", "profit"],
                    "analysis_type": "descriptive"
                }
            )
            print("분석 결과:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ 사용 가능한 MCP 도구

### 1. 데이터 분석 도구

#### statistics_analyzer
기술 통계, 분포 분석, 상관관계 분석을 수행합니다.

```json
{
  "name": "statistics_analyzer",
  "description": "데이터의 기술 통계, 분포 분석, 상관관계를 계산합니다",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_source": {
        "type": "string",
        "description": "데이터 소스 URI (예: postgresql://analytics_db)"
      },
      "table_name": {
        "type": "string",
        "description": "분석할 테이블명"
      },
      "columns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "분석할 컬럼 목록"
      },
      "analysis_type": {
        "type": "string",
        "enum": ["descriptive", "distribution", "correlation"],
        "description": "분석 유형"
      }
    },
    "required": ["data_source", "table_name", "columns"]
  }
}
```

#### data_profiler
데이터 품질 검사 및 기본 정보를 수집합니다.

```json
{
  "name": "data_profiler",
  "description": "데이터 품질 검사 및 기본 정보를 수집합니다",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_source": {
        "type": "string",
        "description": "데이터 소스 URI"
      },
      "table_name": {
        "type": "string",
        "description": "프로파일링할 테이블명"
      },
      "include_stats": {
        "type": "boolean",
        "description": "통계 정보 포함 여부",
        "default": true
      },
      "include_quality": {
        "type": "boolean",
        "description": "품질 검사 포함 여부",
        "default": true
      }
    },
    "required": ["data_source", "table_name"]
  }
}
```

#### outlier_detector
IQR, Z-score 방법을 통한 이상치를 탐지합니다.

```json
{
  "name": "outlier_detector",
  "description": "데이터에서 이상치를 탐지합니다",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_source": {
        "type": "string",
        "description": "데이터 소스 URI"
      },
      "table_name": {
        "type": "string",
        "description": "분석할 테이블명"
      },
      "columns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "이상치 탐지할 컬럼 목록"
      },
      "method": {
        "type": "string",
        "enum": ["iqr", "zscore"],
        "description": "이상치 탐지 방법",
        "default": "iqr"
      },
      "threshold": {
        "type": "number",
        "description": "임계값 (Z-score 방법 사용시)",
        "default": 3
      }
    },
    "required": ["data_source", "table_name", "columns"]
  }
}
```

### 2. 시각화 도구

#### chart_generator
다양한 차트를 생성합니다.

```json
{
  "name": "chart_generator",
  "description": "데이터를 시각화하는 차트를 생성합니다",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_source": {
        "type": "string",
        "description": "데이터 소스 URI"
      },
      "table_name": {
        "type": "string",
        "description": "차트 데이터 테이블명"
      },
      "chart_type": {
        "type": "string",
        "enum": ["bar", "line", "scatter", "histogram", "box", "heatmap"],
        "description": "차트 유형"
      },
      "x_axis": {
        "type": "string",
        "description": "X축 컬럼명"
      },
      "y_axis": {
        "type": "string",
        "description": "Y축 컬럼명"
      },
      "title": {
        "type": "string",
        "description": "차트 제목"
      },
      "width": {
        "type": "integer",
        "description": "차트 너비",
        "default": 800
      },
      "height": {
        "type": "integer",
        "description": "차트 높이",
        "default": 600
      }
    },
    "required": ["data_source", "table_name", "chart_type"]
  }
}
```

### 3. 품질 검사 도구

#### quality_checker
데이터 품질을 검사합니다.

```json
{
  "name": "quality_checker",
  "description": "데이터 품질을 검사합니다",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_source": {
        "type": "string",
        "description": "데이터 소스 URI"
      },
      "table_name": {
        "type": "string",
        "description": "검사할 테이블명"
      },
      "checks": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["missing_values", "outliers", "consistency"]
        },
        "description": "수행할 품질 검사 목록",
        "default": ["missing_values", "outliers", "consistency"]
      }
    },
    "required": ["data_source", "table_name"]
  }
}
```

### 4. 리포트 생성 도구

#### report_builder
종합 분석 리포트를 생성합니다.

```json
{
  "name": "report_builder",
  "description": "종합 분석 리포트를 생성합니다",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_source": {
        "type": "string",
        "description": "데이터 소스 URI"
      },
      "table_name": {
        "type": "string",
        "description": "분석할 테이블명"
      },
      "title": {
        "type": "string",
        "description": "리포트 제목"
      },
      "author": {
        "type": "string",
        "description": "작성자"
      },
      "include_charts": {
        "type": "boolean",
        "description": "차트 포함 여부",
        "default": true
      },
      "include_dashboard": {
        "type": "boolean",
        "description": "대시보드 포함 여부",
        "default": true
      },
      "include_quality": {
        "type": "boolean",
        "description": "품질 검사 포함 여부",
        "default": true
      }
    },
    "required": ["data_source", "table_name"]
  }
}
```

## 📚 사용 예시

### 1. Claude Desktop에서 사용

```
사용자: "sales 테이블의 매출 데이터를 분석해주세요"

Claude: Bridge MCP를 사용하여 sales 테이블을 분석하겠습니다.

먼저 데이터 프로파일링을 수행하겠습니다:
- statistics_analyzer 도구를 사용하여 기술 통계를 계산
- quality_checker 도구를 사용하여 데이터 품질을 검사
- chart_generator 도구를 사용하여 시각화 생성

결과를 종합하여 리포트를 생성하겠습니다.
```

### 2. Cursor에서 사용

```
사용자: "고객 데이터의 이상치를 찾아주세요"

Cursor: Bridge MCP의 outlier_detector 도구를 사용하여 고객 데이터에서 이상치를 탐지하겠습니다.

분석 결과:
- IQR 방법으로 탐지된 이상치: 15개
- Z-score 방법으로 탐지된 이상치: 12개
- 이상치가 발견된 컬럼: age, income, purchase_amount

이상치에 대한 상세 분석과 권장사항을 제공하겠습니다.
```

### 3. Python 스크립트에서 사용

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def analyze_sales_data():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "bridge.mcp_server_unified"],
        env={"BRIDGE_MCP_MODE": "production"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. 데이터 프로파일링
            profile_result = await session.call_tool(
                "data_profiler",
                {
                    "data_source": "postgresql://analytics_db",
                    "table_name": "sales",
                    "include_stats": True,
                    "include_quality": True
                }
            )
            print("데이터 프로파일링 결과:", profile_result)
            
            # 2. 통계 분석
            stats_result = await session.call_tool(
                "statistics_analyzer",
                {
                    "data_source": "postgresql://analytics_db",
                    "table_name": "sales",
                    "columns": ["amount", "profit", "quantity"],
                    "analysis_type": "descriptive"
                }
            )
            print("통계 분석 결과:", stats_result)
            
            # 3. 차트 생성
            chart_result = await session.call_tool(
                "chart_generator",
                {
                    "data_source": "postgresql://analytics_db",
                    "table_name": "sales",
                    "chart_type": "bar",
                    "x_axis": "region",
                    "y_axis": "amount",
                    "title": "지역별 매출"
                }
            )
            print("차트 생성 결과:", chart_result)

if __name__ == "__main__":
    asyncio.run(analyze_sales_data())
```

## 🔧 고급 설정

### 1. 커스텀 데이터 소스 설정

```json
{
  "mcpServers": {
    "bridge": {
      "command": "python",
      "args": ["-m", "src.bridge.mcp_server_unified"],
      "env": {
        "BRIDGE_MCP_MODE": "production",
        "BRIDGE_DATABASE_URL": "postgresql://user:password@localhost:5432/analytics",
        "BRIDGE_REDIS_URL": "redis://localhost:6379/0",
        "BRIDGE_MYSQL_URL": "mysql://user:password@localhost:3306/mysql_db",
        "BRIDGE_MONGODB_URL": "mongodb://user:password@localhost:27017/mongo_db",
        "BRIDGE_ELASTICSEARCH_URL": "http://localhost:9200"
      }
    }
  }
}
```

### 2. 보안 설정

```json
{
  "mcpServers": {
    "bridge": {
      "command": "python",
      "args": ["-m", "src.bridge.mcp_server_unified"],
      "env": {
        "BRIDGE_MCP_MODE": "production",
        "BRIDGE_DATABASE_URL": "postgresql://user:password@localhost:5432/bridge",
        "BRIDGE_REDIS_URL": "redis://localhost:6379/0",
        "BRIDGE_API_KEY": "your-secure-api-key",
        "BRIDGE_ENCRYPTION_KEY": "your-encryption-key"
      }
    }
  }
}
```

### 3. 로깅 설정

```json
{
  "mcpServers": {
    "bridge": {
      "command": "python",
      "args": ["-m", "src.bridge.mcp_server_unified"],
      "env": {
        "BRIDGE_MCP_MODE": "production",
        "BRIDGE_DATABASE_URL": "postgresql://user:password@localhost:5432/bridge",
        "BRIDGE_LOG_LEVEL": "INFO",
        "BRIDGE_LOG_FILE": "/path/to/bridge.log",
        "BRIDGE_AUDIT_LOG": "/path/to/audit.log"
      }
    }
  }
}
```

## 🚨 문제 해결

### 1. 연결 문제

**문제**: MCP 서버에 연결할 수 없습니다.

**해결방법**:
1. Bridge MCP 서버가 실행 중인지 확인
2. 환경 변수가 올바르게 설정되었는지 확인
3. 포트가 사용 중이지 않은지 확인
4. 방화벽 설정 확인

```bash
# 서버 상태 확인
ps aux | grep mcp_server_unified

# 포트 사용 확인
netstat -tlnp | grep :8000

# 로그 확인
tail -f logs/bridge.log
```

### 2. 인증 문제

**문제**: API 키 인증에 실패합니다.

**해결방법**:
1. API 키가 올바른지 확인
2. 환경 변수에서 API 키 설정 확인
3. 데이터베이스 연결 확인

```bash
# 환경 변수 확인
echo $BRIDGE_API_KEY

# 데이터베이스 연결 테스트
python -c "
import asyncio
from src.bridge.connectors.postgres import PostgresConnector
async def test():
    conn = PostgresConnector('postgresql://user:password@localhost:5432/bridge')
    result = await conn.test_connection()
    print('연결 상태:', result)
asyncio.run(test())
"
```

### 3. 성능 문제

**문제**: MCP 도구 실행이 느립니다.

**해결방법**:
1. 데이터베이스 인덱스 최적화
2. 캐시 설정 확인
3. 메모리 사용량 모니터링
4. 쿼리 최적화

```bash
# 성능 모니터링
python -c "
from src.bridge.dashboard.monitoring_dashboard import MonitoringDashboard
monitor = MonitoringDashboard()
monitor.start_monitoring()
metrics = monitor.get_current_metrics()
print('시스템 메트릭:', metrics)
"
```

## 📞 지원

- **이슈 트래커**: [GitHub Issues](https://github.com/your-org/bridge/issues)
- **문서**: [Bridge 문서](https://github.com/your-org/bridge/docs)
- **MCP 사양**: [Model Context Protocol](https://modelcontextprotocol.io/)

## 🔄 업데이트

Bridge MCP 서버를 최신 버전으로 업데이트하려면:

```bash
# 저장소 업데이트
git pull origin main

# 의존성 업데이트
make install

# 서버 재시작
python -m bridge.mcp_server_unified
```

이 가이드를 따라 Bridge MCP를 AI 클라이언트에 설치하고 사용할 수 있습니다. 추가 질문이나 문제가 있으면 언제든 문의하세요!
