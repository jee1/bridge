# Bridge 시스템 아키텍처 및 기능 가이드

## 프로젝트 개요

Bridge는 Model Context Protocol(MCP) 기반의 데이터 통합 및 AI 오케스트레이션 시스템입니다. 다양한 데이터 소스(PostgreSQL, MongoDB, Elasticsearch 등)에 대한 표준화된 접근을 제공하고, AI 에이전트가 엔터프라이즈 데이터를 안전하고 투명하게 활용할 수 있도록 지원합니다.

## 시스템 아키텍처

### 전체 시스템 구조

```mermaid
graph TB
    subgraph "Frontend Channels"
        A[챗봇] --> B[API Gateway]
        C[API 클라이언트] --> B
        D[워크플로 엔진] --> B
    end
    
    subgraph "Orchestration Layer"
        B --> E[Bridge Orchestrator<br/>FastAPI + Celery]
        E --> F[Task Queue<br/>Redis]
    end
    
    subgraph "Data Access Layer"
        E --> G[Connector Pipeline]
        G --> H[PostgreSQL Connector]
        G --> I[MongoDB Connector]
        G --> J[Elasticsearch Connector]
    end
    
    subgraph "Semantic & Governance Layer"
        E --> K[Semantic Models<br/>Pydantic]
        E --> L[RBAC Controller]
        E --> M[Audit Logger]
    end
    
    subgraph "Execution Runtime"
        E --> N[SQL Executor]
        E --> O[Spark/Databricks Jobs]
        E --> P[Python UDF Sandbox]
        E --> Q[Vector Search Engine]
    end
    
    subgraph "MCP Server Layer"
        E --> R[MCP Server<br/>1개 통합 + 7개 개별]
        R --> S[Bridge MCP Unified<br/>환경 변수 기반 모드]
        R --> T[Bridge MCP Robust<br/>개발/테스트용]
        R --> U[Bridge MCP Real<br/>개발/테스트용]
        R --> V[Bridge MCP Working<br/>개발/테스트용]
    end
    
    subgraph "Data Sources"
        H --> Y[(PostgreSQL)]
        I --> Z[(MongoDB)]
        J --> AA[(Elasticsearch)]
    end
    
    subgraph "Storage"
        M --> BB[Audit Logs<br/>JSONL]
        K --> CC[Schema Registry]
    end
```

### 컴포넌트 상호작용 다이어그램

```mermaid
sequenceDiagram
    participant U as 사용자
    participant AG as API Gateway
    participant O as Orchestrator
    participant C as Connector
    participant DB as Database
    participant A as Audit Logger
    
    U->>AG: 분석 요청
    AG->>O: TaskRequest 전달
    O->>A: 요청 로깅
    O->>O: 작업 계획 수립
    O->>C: 데이터 소스 접근
    C->>DB: 쿼리 실행
    DB-->>C: 결과 반환
    C-->>O: 처리된 데이터
    O->>A: 실행 결과 로깅
    O-->>AG: TaskResponse 반환
    AG-->>U: 분석 결과
```

## 핵심 컴포넌트

### 1. 커넥터 레이어 (`/src/bridge/connectors/`)

데이터 소스별 표준화된 인터페이스를 제공합니다.

#### BaseConnector (추상 클래스)
```python
class BaseConnector(ABC):
    def test_connection(self) -> bool
    def get_metadata(self) -> Dict[str, Any]
    def run_query(self, query: str, params: Dict[str, Any]) -> Iterable[Dict[str, Any]]
    def mask_columns(self, rows: Iterable[Dict[str, Any]], masked_fields: Iterable[str])
```

#### PostgresConnector (구현체)
- PostgreSQL 데이터베이스 연결 및 쿼리 실행
- 비동기 처리 지원 (asyncpg)
- 스키마 메타데이터 자동 수집
- 민감 데이터 마스킹 기능

#### ConnectorRegistry (커넥터 관리)
- 커넥터 인스턴스의 중앙 관리
- 동적 커넥터 등록 및 조회
- 커넥터별 설정 및 상태 관리

### 2. 오케스트레이터 레이어 (`/src/bridge/orchestrator/`)

사용자 요청을 처리하고 작업을 조율하는 핵심 서비스입니다.

#### CLI 인터페이스 (`cli.py`)
- 명령행 인터페이스를 통한 작업 제출 및 상태 조회
- argparse를 사용한 사용자 친화적 인터페이스
- 실시간 작업 상태 모니터링 및 폴링

#### FastAPI 애플리케이션 (`app.py`)
- RESTful API 엔드포인트 제공
- 헬스 체크 및 상태 모니터링
- 라우터 기반 모듈화된 구조

#### 작업 라우터 (`routers.py`)
- `/tasks/plan` 엔드포인트: 사용자 요청을 작업으로 변환
- 비동기 작업 큐에 실행 요청 전달
- 작업 상태 추적 및 결과 반환

#### Celery 태스크 (`tasks.py`)
- 백그라운드 작업 처리
- 컨텍스트 수집 및 도구 실행
- 확장 가능한 파이프라인 구조

### 3. 시맨틱 레이어 (`/src/bridge/semantic/`)

데이터 구조와 비즈니스 로직을 정의합니다.

#### 데이터 모델 (`models.py`)
- `SemanticEntity`: 데이터 엔터티 정의 (민감도 레벨 포함)
- `TaskRequest`: 사용자 요청 구조
- `TaskResponse`: 작업 응답 구조
- `TaskStep`: 작업 단계별 상세 정보
- `TaskStatusResponse`: 작업 상태 및 결과 조회 응답

### 4. 워크스페이스 관리 (`/src/bridge/workspaces/`)

사용자 권한 및 접근 제어를 담당합니다.

#### RBAC 시스템 (`rbac.py`)
- 역할 기반 접근 제어
- 사용자별 권한 관리
- 동적 권한 검사

### 5. MCP 서버 레이어 (`/src/bridge/mcp_server*.py`)

Cursor IDE와의 통합을 위한 MCP 서버들을 제공합니다.

#### MCP 서버 구현체들
- **mcp_server_unified.py**: **통합된 MCP 서버** (환경 변수 기반 모드 지원)
  - **개발용**: `BRIDGE_MCP_MODE=development` (모의 응답)
  - **프로덕션용**: `BRIDGE_MCP_MODE=production` (실제 DB + 에러 복구)
  - **실제 DB 연동**: `BRIDGE_MCP_MODE=real` (직접 JSON-RPC)
  - **간단한 모드**: `BRIDGE_MCP_MODE=mock` (모의 응답 + 직접 JSON-RPC)
- **mcp_server.py**: 기본 MCP 서버 (통합 서버로 리다이렉트)
- **mcp_server_robust.py**: 견고한 MCP 서버 (개발/테스트용)
- **mcp_server_real.py**: 실제 데이터베이스 연동 서버 (개발/테스트용)
- **mcp_server_working.py**: 작동하는 버전 (개발/테스트용)
- **mcp_server_minimal.py**: 최소 기능 버전 (개발/테스트용)
- **mcp_server_simple.py**: 단순 버전 (개발/테스트용)
- **mcp_server_fixed.py**: 수정된 버전 (개발/테스트용)

#### MCP 서버 기능
- JSON-RPC 프로토콜 지원
- 데이터베이스 쿼리 실행 도구
- 스키마 정보 조회 도구
- 데이터 분석 도구
- 커넥터 목록 조회 도구

### 6. 감사 로깅 (`/src/bridge/audit/`)

시스템 활동을 추적하고 기록합니다.

#### 감사 로거 (`logger.py`)
- 구조화된 JSON 로그 저장
- 타임스탬프 및 메타데이터 포함
- 일별 로그 파일 자동 생성

## 데이터 플로우

### 1. 요청 처리 플로우

```mermaid
flowchart TD
    A[사용자 요청] --> B[API Gateway]
    B --> C[인증/권한 검사]
    C --> D[Orchestrator]
    D --> E[의도 분석]
    E --> F[작업 계획 수립]
    F --> G[Celery 큐에 작업 추가]
    G --> H[커넥터 선택]
    H --> I[데이터 소스 접근]
    I --> J[쿼리 실행]
    J --> K[결과 처리]
    K --> L[감사 로깅]
    L --> M[응답 반환]
```

### 2. MCP 컨텍스트 패키징

```mermaid
flowchart LR
    A[사용자 의도] --> B[시맨틱 카탈로그 조회]
    B --> C[관련 엔터티 식별]
    C --> D[커넥터별 쿼리 생성]
    D --> E[데이터 수집]
    E --> F[컨텍스트 패키징]
    F --> G[AI 모델 전달]
```

## API 엔드포인트

### 시스템 상태
- `GET /health` - 시스템 상태 확인

### 작업 관리
- `POST /tasks/plan` - 작업 계획 및 실행 요청
- `GET /tasks/{job_id}` - 작업 상태 및 결과 조회

### CLI 인터페이스
- `python cli.py <intent>` - 작업 제출 및 상태 모니터링
- `--sources` - 사용할 데이터 소스 지정
- `--tools` - 필요한 도구 목록 지정
- `--base-url` - 서버 URL 지정
- `--poll-interval` - 상태 조회 간격 조정

### MCP 서버 인터페이스
- `make mcp-server` - Makefile을 통한 통합 서버 실행 (권장)
- `python -m src.bridge.mcp_server_unified` - 통합 서버 직접 실행
- `BRIDGE_MCP_MODE=* python -m src.bridge.mcp_server_unified` - 환경 변수로 모드 지정
- `bridge-mcp` - 견고한 MCP 서버 실행 (개발/테스트용)
- `bridge-mcp-real` - 실제 데이터베이스 연동 MCP 서버 실행 (개발/테스트용)
- `python -m src.bridge.mcp_server_*` - 개별 서버 직접 실행 (개발/테스트용)
- `python scripts/run_mcp_server.py` - 스크립트 파일을 통한 실행

#### 요청 예시
```bash
curl -X POST "http://localhost:8000/tasks/plan" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "intent": "프리미엄 고객 세그먼트 분석",
    "sources": ["postgres://analytics_db"],
    "required_tools": ["sql_executor", "statistics_analyzer"],
    "context": {
      "time_range": "2024-01-01 to 2024-12-31",
      "customer_tier": "premium"
    }
  }'
```

#### 응답 예시
```json
{
  "intent": "프리미엄 고객 세그먼트 분석",
  "status": "planned",
  "steps": [
    {
      "name": "collect_context",
      "details": {"sources": ["postgres://analytics_db"]}
    },
    {
      "name": "execute_tools",
      "details": {"tools": ["sql_executor", "statistics_analyzer"]}
    },
    {
      "name": "queue_execution",
      "details": {"job_id": "abc123-def456-ghi789"}
    }
  ]
}
```

#### 작업 상태 조회 응답 예시
```bash
# 202 응답 예시 (큐에 대기 중)
curl -H "Authorization: Bearer YOUR_API_KEY" "http://localhost:8000/tasks/{job_id}"
{
  "job_id": "abc123-def456-ghi789",
  "state": "PENDING",
  "ready": false,
  "successful": false
}

# 200 응답 예시 (성공)
{
  "job_id": "abc123-def456-ghi789",
  "state": "SUCCESS",
  "ready": true,
  "successful": true,
  "result": {
    "status": "completed",
    "intent": "프리미엄 고객 세그먼트 분석",
    "collected_sources": [
      {
        "source": "postgres://analytics_db",
        "metadata": {"tables": ["customers", "orders"]}
      }
    ],
    "missing_sources": []
  },
  "error": null
}

# 200 응답 예시 (실패)
{
  "job_id": "abc123-def456-ghi789",
  "state": "FAILURE",
  "ready": true,
  "successful": false,
  "error": "에러 메시지"
}
```

## 보안 기능

### API 인증
- **API 키 인증**: FastAPI 의존성 주입을 통한 안전한 API 접근
- **Authorization 헤더**: Bearer 토큰을 통한 인증
- **권한 검사**: 각 엔드포인트별 접근 권한 확인

### 데이터 보호
- **SQL 인젝션 방지**: 파라미터 바인딩을 통한 안전한 쿼리 실행
- **입력 검증**: Pydantic 모델을 통한 엄격한 데이터 검증
- **데이터 마스킹**: 민감한 컬럼 자동 마스킹
- **에러 처리**: 안전한 예외 처리 및 로깅

### 감사 및 모니터링
- **감사 로깅**: 모든 활동 추적 및 구조화된 로그 저장
- **RBAC**: 역할 기반 접근 제어
- **세션 추적**: 사용자 세션 및 활동 모니터링

## 개발 가이드라인

### 1. 새로운 커넥터 추가

```python
from src.bridge.connectors.base import BaseConnector

class NewDatabaseConnector(BaseConnector):
    async def test_connection(self) -> bool:
        # 연결 테스트 로직
        pass
    
    async def get_metadata(self) -> Dict[str, Any]:
        # 메타데이터 수집 로직
        pass
    
    async def run_query(self, query: str, params: Dict[str, Any] = None):
        # 쿼리 실행 로직
        pass
```

### 2. 새로운 시맨틱 모델 추가

```python
from pydantic import BaseModel, Field

class NewSemanticModel(BaseModel):
    name: str
    description: str = Field(default="")
    sensitivity: str = Field(default="internal")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 3. 새로운 API 엔드포인트 추가

```python
from fastapi import APIRouter
from ..semantic.models import TaskRequest, TaskResponse

router = APIRouter()

@router.post("/new-endpoint", response_model=TaskResponse)
async def new_endpoint(request: TaskRequest) -> TaskResponse:
    # 엔드포인트 로직
    pass
```

## 보안 및 거버넌스

### 1. 데이터 마스킹
- 민감한 컬럼 자동 감지 및 마스킹
- 역할별 데이터 접근 제어
- 쿼리 리라이팅을 통한 데이터 보호

### 2. 감사 추적
- 모든 데이터 접근 로깅
- 사용자 행동 추적
- 컴플라이언스 지원

### 3. 접근 제어
- RBAC 기반 권한 관리
- 프로젝트별 데이터 격리
- API 키 기반 인증

## 모니터링 및 관측성

### 1. 로그 관리
- 구조화된 JSON 로그
- 일별 로그 파일 자동 생성
- 중앙화된 로그 수집

### 2. 메트릭 수집
- API 응답 시간
- 쿼리 실행 성능
- 에러율 모니터링

### 3. 알림 시스템
- 시스템 장애 알림
- 성능 임계값 초과 알림
- 보안 이벤트 알림

## 확장성 고려사항

### 1. 수평적 확장
- Celery 워커 노드 추가
- 커넥터 인스턴스 분산
- 캐시 레이어 추가

### 2. 수직적 확장
- 메모리 및 CPU 리소스 증설
- 데이터베이스 연결 풀 최적화
- 비동기 처리 최적화

### 3. 새로운 데이터 소스 지원
- 플러그인 아키텍처
- 커넥터 표준화
- 자동 스키마 감지

## 다음 단계

1. **커넥터 확장**: MongoDB, PostgreSQL 커넥터 구현 완료
2. **AI 통합**: LangChain 및 OpenAI SDK 통합 완료
3. **MCP 서버**: 1개 통합 서버 + 7개 개별 서버 구현 완료
4. **모니터링 강화**: Prometheus, Grafana 대시보드 구축
5. **테스트 커버리지**: 단위 테스트 및 통합 테스트 확장
6. **문서화**: API 문서 및 사용자 가이드 작성
