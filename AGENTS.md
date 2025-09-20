# Repository Guidelines

## Project Structure & Module Organization
- `/docs/`에는 제품 기획, MCP 사양, 온보딩 자료를 보관합니다. 새 문서는 파일명에 하이픈을 사용하고, 영문 슬러그를 유지하세요.
- 애플리케이션 코드는 `/src/bridge/` 아래 도메인별 하위 폴더(`connectors/`, `orchestrator/`, `workflows/`)로 분리합니다.
- 공용 스키마와 데이터 계약은 `/schema/`(JSON/YAML)에서 관리하고, 예제 데이터는 `/assets/samples/`에 둡니다.
- 테스트 코드는 소스 구조를 반영해 `/tests/<module>/`에 배치하고, 픽스처는 `/tests/fixtures/`에서 재사용합니다.

## Build, Test, and Development Commands
- `make install` : 의존성을 설치하고 로컬 가상환경을 초기화합니다.
- `make fmt` : 포매터와 린터를 실행해 코드 스타일을 맞춥니다.
- `make test` : 전체 단위 및 통합 테스트를 실행합니다. 실패 시 리포트를 첨부하세요.
- `make dev` : 로컬 개발 서버와 샌드박스 에이전트를 함께 기동합니다.
- `make worker` : Celery 워커를 실행합니다. 실제 비동기 처리를 확인할 때 Redis 브로커와 함께 사용하세요.
- Docker 환경이 필요할 때는 `make docker-build`, `make docker-up`, `make docker-down`, `make docker-test`를 활용해 컨테이너 기반 개발/테스트를 진행합니다.

## Coding Style & Naming Conventions
- Python 코드는 PEP 8을 준수하며, Black(라인 길이 100)과 isort를 사용합니다.
- 커넥터 클래스는 `DatabaseConnector`, `ElasticsearchConnector`처럼 명시적 이름을 사용합니다.
- 환경 변수는 `BRIDGE_<DOMAIN>_<PURPOSE>` 패턴을 따릅니다.
- 모듈 간 계약은 타입 힌트와 pydantic 모델로 명시합니다.

## Testing Guidelines
- pytest를 기본 프레임워크로 사용하고, 파일명은 `test_<feature>.py`로 지정합니다.
- 새 기능은 최소 80% 라인 커버리지를 유지해야 하며, 측정은 `make test -- --cov`로 확인합니다.
- 데이터베이스 통합 테스트는 테스트 전용 스키마나 Docker 컨테이너를 활용하고, 테스트 종료 후 정리를 보장하세요.
- Redis 통합 테스트(`tests/test_celery_integration.py`)는 `BRIDGE_TEST_REDIS_URL`이 설정된 환경에서 실행되며, 사전에 `docker-compose -f docker-compose.redis.yml up -d`와 `make worker`를 수행해야 합니다.
- Docker 기반 통합 테스트는 `make docker-up`으로 서비스를 올린 뒤 `make docker-test`를 실행하고, 종료 시 `make docker-down`으로 정리합니다.

## Commit & Pull Request Guidelines
- 커밋 메시지는 `type(scope): summary` 패턴을 사용합니다. 예) `feat(connectors): add mysql profiling step`.
- PR 설명에는 목적, 주요 변경점, 테스트 결과, 영향 받는 데이터 소스를 포함합니다.
- 관련 이슈 번호를 `Closes #123` 형식으로 연결하고, UI나 문서 변경 시 스크린샷 또는 미리보기를 첨부합니다.

## Security & Configuration Tips
- 민감 자격 증명은 `.env` 대신 시크릿 매니저에 저장하고, 로컬 테스트는 `.env.example`을 복제해 사용합니다.
- 외부 쿼리 실행 로직에는 파라미터 바인딩을 강제하고, 감사 로그는 `/logs/audit/`에 구조화된 JSON으로 남기세요.
