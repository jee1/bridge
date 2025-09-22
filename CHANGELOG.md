# Changelog

모든 주요 변경사항은 이 파일에 문서화됩니다.

이 프로젝트는 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)을 따릅니다.

## [Unreleased]

### Added

- CD 마일스톤 2.3: 통합 대시보드 및 모니터링 시스템
- CD 마일스톤 2.2: 자동화 파이프라인 구현
- CD 마일스톤 2.1: 데이터 거버넌스 시스템
- C1 마일스톤 1.4: 기본 시각화 도구
- C1 마일스톤 1.3: 데이터 품질 검사 시스템
- C1 마일스톤 1.2: 기본 통계 분석 도구
- MCP 도구 확장 구현
- 개발/QA/PR/배포 단계별 테스트 스크립트
- Bridge 프로젝트 명령어 도움말 시스템

### Changed

- MCP 서버 및 통계 분석 관련 테스트 스크립트 개선
- mypy 검사 범위 최적화

### Fixed

- MCP 서버 및 통계 분석 관련 테스트 스크립트 수정
- mypy 검사 범위 변경으로 인한 문제 해결

### Documentation

- Bridge 문서 업데이트 및 사용자 가이드 추가
- MCP 서버 문서 내용 수정 및 통합 서버 설명 업데이트
- Bridge Analytics MVP 문서 및 마일스톤 로드맵 작성

## [0.1.0] - Unreleased

### Features Added

- **Core Architecture**: Bridge MCP 오케스트레이터 기본 구조
- **Data Connectors**:
  - PostgreSQL 커넥터
  - MySQL 커넥터
  - MongoDB 커넥터
  - Elasticsearch 커넥터
  - Mock 커넥터 (테스트용)
- **AI Orchestration**:
  - LangChain과 OpenAI SDK 통합
  - 사용자 의도를 구조화된 작업으로 변환
  - 최적의 실행 플랜 선정
  - MCP 컨텍스트 재패키징
- **Analytics Tools**:
  - 통계 분석 (기술 통계, 분포 분석, 상관관계 분석)
  - 데이터 프로파일링
  - 이상치 탐지 (IQR, Z-score 방법)
  - 차트 생성 (막대, 선, 산점도, 히스토그램, 박스 플롯, 히트맵)
  - 품질 검사 (결측값, 이상치, 일관성 검사)
  - 리포트 생성
- **Enterprise Security**:
  - RBAC (역할 기반 접근 제어)
  - 데이터 마스킹
  - 감사 로깅
  - API 키 인증
  - SQL 인젝션 방지
- **Semantic Modeling**:
  - Pydantic 기반 데이터 계약
  - 비즈니스 정의와 민감도 태그
  - 자동 프로파일링
- **Async Processing**:
  - FastAPI + Celery 통합
  - Redis 브로커
  - 실시간 상태 추적
- **Observability & Monitoring**:
  - OpenTelemetry 통합
  - Prometheus + Grafana 지원
  - 구조화된 로깅
  - 에러 추적 (Sentry 통합)
- **MCP Server Implementation**:
  - 통합된 MCP 서버 (`mcp_server_unified.py`)
  - 환경 변수 기반 모드 지원 (development, production, real, mock)
  - 7개 개별 MCP 서버 구현체
- **CLI Interface**: 명령줄 인터페이스
- **Docker Support**: Docker 및 Docker Compose 설정
- **Testing Framework**: pytest 기반 테스트 시스템

### Technical Details

- **Python 3.11+** 지원
- **FastAPI 0.111.0** 웹 프레임워크
- **Pydantic 2.7.0** 데이터 검증
- **SQLAlchemy 2.0** ORM
- **Celery 5.3.6** 비동기 작업 큐
- **Redis** 브로커
- **Docker** 컨테이너화

### Project Structure

```text
src/bridge/
├── connectors/          # 데이터 소스 커넥터
├── orchestrator/        # FastAPI 오케스트레이터
├── semantic/           # 시맨틱 모델
├── workspaces/         # 워크스페이스 관리
├── audit/              # 감사 로깅
├── analytics/          # 분석 도구
├── governance/         # 데이터 거버넌스
├── automation/         # 자동화 시스템
├── dashboard/          # 대시보드 관리
└── mcp_server*.py      # MCP 서버 구현체들
```

## [0.0.1] - 2025-09-20

### Initial Setup

- 초기 프로젝트 설정
- 기본 프로젝트 구조
- 의존성 관리 (pyproject.toml)
- Git 저장소 초기화

---

## Links

- [Unreleased]: https://github.com/your-org/bridge/compare/v0.1.0...HEAD
- [0.1.0]: https://github.com/your-org/bridge/compare/v0.0.1...v0.1.0
- [0.0.1]: https://github.com/your-org/bridge/releases/tag/v0.0.1

---

## Changelog Format

이 CHANGELOG는 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/) 형식을 따릅니다.

### Categories

- **Added**: 새로운 기능
- **Changed**: 기존 기능의 변경사항
- **Deprecated**: 곧 제거될 기능
- **Removed**: 이번 릴리스에서 제거된 기능
- **Fixed**: 버그 수정
- **Security**: 보안 관련 변경사항

### Versioning

- **Major**: 호환되지 않는 API 변경
- **Minor**: 하위 호환성을 유지하는 기능 추가
- **Patch**: 하위 호환성을 유지하는 버그 수정

### Date Format

- **YYYY-MM-DD**: ISO 8601 형식 사용
