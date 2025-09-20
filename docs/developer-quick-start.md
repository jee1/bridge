# Bridge 개발자 빠른 시작 가이드

## 🚀 프로젝트 개요

Bridge는 **Model Context Protocol(MCP)** 기반의 데이터 통합 및 AI 오케스트레이션 시스템입니다. 다양한 데이터 소스에 대한 표준화된 접근을 제공하고, AI 에이전트가 엔터프라이즈 데이터를 안전하게 활용할 수 있도록 지원합니다.

## 📁 프로젝트 구조

```
src/bridge/
├── connectors/          # 데이터 소스 커넥터
│   ├── base.py         # BaseConnector 추상 클래스
│   ├── postgres.py     # PostgreSQL 커넥터
│   ├── mock.py         # Mock 커넥터 (테스트용)
│   └── registry.py     # 커넥터 레지스트리
├── orchestrator/        # FastAPI 오케스트레이터
│   ├── app.py          # FastAPI 애플리케이션
│   ├── routers.py      # API 라우터
│   ├── tasks.py        # Celery 태스크
│   ├── celery_app.py   # Celery 설정
│   └── queries.py      # Celery 결과 조회 유틸리티
├── semantic/           # 시맨틱 모델
│   └── models.py       # Pydantic 데이터 모델
├── workspaces/         # 워크스페이스 관리
│   └── rbac.py         # RBAC 시스템
├── audit/              # 감사 로깅
│   └── logger.py       # 감사 로거
└── cli.py              # CLI 인터페이스
```

## 🛠️ 개발 환경 설정

### 1. 의존성 설치
```bash
make install
```

### 2. 코드 포맷팅
```bash
make fmt
```

### 3. 테스트 실행
```bash
make test
```

### 4. 개발 서버 실행
```bash
make dev
```

### 5. CLI 사용법
```bash
# 기본 사용법
python cli.py "고객 세그먼트 분석"

# 특정 데이터 소스와 도구 지정
python cli.py "프리미엄 고객 분석" --sources postgres://analytics_db --tools sql_executor,statistics_analyzer

# 다른 서버 URL 지정
python cli.py "데이터 분석" --base-url http://staging.example.com:8000

# 폴링 간격 조정
python cli.py "분석 작업" --poll-interval 5.0
```

상태 출력은 HTTP 코드와 함께 제공됩니다. `202`는 작업이 큐에 남아 있음을 의미하며, `200`은 완료(성공/실패)를 뜻합니다. 실패 시 `error` 필드를 확인하고 재시도하거나 로그를 확인하세요.

### 6. Docker Compose 개발 환경 (선택사항)
```bash
# Redis와 함께 전체 개발 환경 실행
docker-compose -f docker-compose.dev.yml up -d

# 테스트 실행
docker-compose -f docker-compose.dev.yml run --rm test
```

## 🔧 핵심 컴포넌트

### 1. 커넥터 시스템
```python
# 새로운 커넥터 추가 예시
from src.bridge.connectors.base import BaseConnector

class MyDatabaseConnector(BaseConnector):
    async def test_connection(self) -> bool:
        # 연결 테스트
        return True
    
    async def get_metadata(self) -> Dict[str, Any]:
        # 메타데이터 수집
        return {"tables": []}
    
    async def run_query(self, query: str, params: Dict[str, Any] = None):
        # 쿼리 실행
        yield {}
```

### 2. API 엔드포인트
```python
# 새로운 엔드포인트 추가
@router.post("/my-endpoint")
async def my_endpoint(request: TaskRequest) -> TaskResponse:
    # 비즈니스 로직
    return TaskResponse(...)
```

### 3. 시맨틱 모델
```python
# 새로운 데이터 모델 추가
class MySemanticModel(BaseModel):
    name: str
    description: str = ""
    sensitivity: str = "internal"
```

## 📊 API 사용 예시

### 작업 계획 요청
```bash
curl -X POST "http://localhost:8000/tasks/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "고객 세그먼트 분석",
    "sources": ["postgres://analytics_db"],
    "required_tools": ["sql_executor"],
    "context": {
      "time_range": "2024-01-01 to 2024-12-31"
    }
  }'
```

### 헬스 체크
```bash
curl "http://localhost:8000/health"
```

## 🔒 보안 기능

- **데이터 마스킹**: 민감한 컬럼 자동 마스킹
- **RBAC**: 역할 기반 접근 제어
- **감사 로깅**: 모든 활동 추적
- **쿼리 리라이팅**: 데이터 보호

## 📈 모니터링

- **구조화된 로그**: JSON 형식 로그 저장
- **메트릭 수집**: 성능 및 에러 모니터링
- **감사 추적**: 컴플라이언스 지원

## 🚀 다음 단계

1. **커넥터 확장**: MongoDB, Elasticsearch 지원
2. **AI 통합**: LangChain, OpenAI SDK 통합
3. **모니터링**: Prometheus, Grafana 대시보드
4. **테스트**: 단위/통합 테스트 확장
5. **문서화**: API 문서 자동 생성

## 📚 추가 자료

- [상세 아키텍처 문서](./bridge-system-architecture.md)
- [MCP 사양](./bridge-model-context-protocol.md)
- [Python 아키텍처 가이드](./python-architecture-tech-stack.md)

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. 기능 브랜치 생성
3. 코드 작성 및 테스트
4. PR 생성 및 리뷰 요청

## 📞 지원

- 이슈 트래커: GitHub Issues
- 문서: `/docs` 디렉토리
- 코드 예시: `/assets/samples` 디렉토리
