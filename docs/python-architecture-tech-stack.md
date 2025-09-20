# Bridge Python 아키텍처 & 기술 스택

## 개요
Bridge는 다양한 데이터 소스와 AI 오케스트레이션을 파이썬 기반으로 통합하여, Model Context Protocol(MCP)에 맞춘 컨텍스트 패키징과 도구 실행을 자동화합니다. 본 문서는 핵심 구성 요소와 권장 기술 스택을 정리해 신규 기여자의 설계를 돕기 위해 작성되었습니다.

## 아키텍처 요약
- **프런트 채널**: 챗봇, API, 워크플로 엔진이 공통 게이트웨이(`api-gateway`)를 통해 진입합니다.
- **오케스트레이션 레이어**: FastAPI 기반 `bridge-orchestrator` 서비스가 사용자 의도를 해석하고 Task Graph를 구성합니다.
- **데이터 액세스 레이어**: 커넥터 파이프라인이 각 데이터베이스에 맞춘 SQL/NoSQL 어댑터와 캐시 전략을 제공합니다.
- **시맨틱 & 거버넌스 레이어**: pydantic 모델로 정의된 데이터 계약과 RBAC 정책이 적용됩니다.
- **실행 런타임**: SQL 실행기, Spark/Databricks 잡 트리거, Python UDF 샌드박스, 벡터 검색 엔진을 툴킷 형태로 묶어 MCP가 호출합니다.

## 핵심 모듈
### 데이터 커넥터 레이어
- `src/bridge/connectors/`에 저장하며, SQLAlchemy, asyncpg, pymongo, elasticsearch-py 등을 조합합니다.
- 커넥터는 `BaseConnector`를 상속하고, 스키마 메타데이터 수집과 정책 기반 필터링을 구현합니다.

### 시맨틱 카탈로그 & 데이터 계약
- `src/bridge/semantic/`에서 pydantic 모델로 엔터티, 지표, 민감도 태그를 선언합니다.
- `/schema/` 디렉터리에 JSON/YAML 계약을 버전 관리하고, 변경 사항은 마이그레이션 스크립트로 반영합니다.

### AI 오케스트레이션 레이어
- `src/bridge/orchestrator/`는 FastAPI + Celery 조합으로 비동기 워크플로를 구동합니다.
- `celery_app.py`에서 브로커/백엔드를 환경 변수(`BRIDGE_CELERY_BROKER_URL`, `BRIDGE_CELERY_RESULT_BACKEND`)로 설정하고, 기본값은 메모리 전송을 사용해 로컬에서도 워커 없이 실행됩니다.
- LangChain/OpenAI SDK를 이용해 MCP 컨텍스트를 생성하며, 프롬프트 템플릿은 `/assets/prompts/`에 저장합니다.

### 워크스페이스 & 거버넌스
- RBAC, 감사 로그, 세션 이력을 `src/bridge/workspaces/`와 `src/bridge/audit/` 모듈에서 처리합니다.
- 감사 로그는 구조화된 JSON으로 `/logs/audit/`에 저장하고, OpenTelemetry로 메트릭을 수집합니다.

## 주요 기술 스택
- **언어 & 런타임**: Python 3.11, 표준 Python 프로젝트 설정.
- **웹/API**: FastAPI, pydantic v2, Uvicorn, Celery/Redis(비동기 작업 큐).
- **데이터 액세스**: SQLAlchemy 2.x, asyncpg, pymongo, elasticsearch-py, databricks-sql-connector.
- **AI & ML**: LangChain, OpenAI SDK, sentence-transformers(유사도 분석), pandas/Polars(데이터 처리).
- **인프라 & 배포**: Docker, Docker Compose(로컬), Helm/Kubernetes(프로덕션), GitHub Actions CI.
- **관측성**: OpenTelemetry, Prometheus, Grafana, Sentry(에러 추적).

## 데이터 플로우 요약
1. 사용자가 챗/HTTP를 통해 분석 요청을 전송합니다.
2. API 게이트웨이가 요청을 `bridge-orchestrator`로 전달하고 인증/정책 검사를 수행합니다.
3. 오케스트레이터가 시맨틱 카탈로그를 조회해 관련 엔터티와 규칙을 선정합니다.
4. MCP 컨텍스트 패키지를 생성하고 필요한 커넥터/툴 실행을 스케줄링합니다.
5. 커넥터가 원본 데이터베이스에 접근해 쿼리 또는 작업을 수행하고, 결과를 캐시/버전 관리합니다.
6. 후처리 서비스가 품질 검사를 거친 뒤 응답을 요약해 사용자 채널로 반환합니다.

## 개발 및 배포 고려 사항
- 로컬 개발은 `make install` 후 `make dev`로 API와 샌드박스 에이전트를 동시 기동합니다.
- 비동기 태스크 실행은 Celery 워커를 기동하며, 로컬 환경에서는 기본적으로 eager 모드이므로 별도 워커 없이도 동작합니다.
- Redis 브로커 연동이 필요한 경우 `docker-compose -f docker-compose.redis.yml up -d`로 Redis를 실행하고 `.env`에 브로커/백엔드 URL을 설정합니다.
- 실제 비동기 동작을 검증하려면 `BRIDGE_CELERY_TASK_ALWAYS_EAGER=false`를 설정하고 `make worker`를 별도 터미널에서 실행한 다음 통합 테스트(`pytest tests/test_celery_integration.py`)를 수행합니다.
- 테스트는 pytest + coverage(`make test -- --cov`)를 실행하고, 통합 테스트 시 Docker Compose로 의존 DB를 띄웁니다.
- 환경 변수는 `BRIDGE_<DOMAIN>_<PURPOSE>` 규칙을 따르며, `.env.example`을 기반으로 설정합니다.
- CICD 파이프라인에서 fmt → lint → test → build 순으로 검증하고, 이미지 태그는 `bridge-api:<git-sha>` 패턴을 사용합니다.

## 확장 로드맵 메모
- 고가용성을 위해 오케스트레이터를 멀티 인스턴스로 구성하고, Redis Cluster를 도입합니다.
- 벡터 검색은 초기엔 OpenSearch로 시작하고, 대규모 확장 시 Milvus/Weaviate를 평가합니다.
- MCP 컨텍스트 캐싱을 위해 Feature Store(Feast, Tecton 등) 연동을 검토합니다.
