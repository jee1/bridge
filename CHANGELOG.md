# Changelog

모든 주요 변경사항은 이 파일에 문서화됩니다.

이 프로젝트는 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)을 따릅니다.

## [Unreleased]

### Added

- **CA 마일스톤 3.4: 워크플로 및 자동화 시스템 구현 완료** ✅
  - **AnalysisTemplateManager**: 분석 템플릿 관리 시스템
  - **CustomerAnalysisTemplate**: 고객 분석 템플릿 (RFM 분석, 세그멘테이션)
  - **SalesAnalysisTemplate**: 매출 분석 템플릿 (트렌드, 계절성 분석)
  - **ABTestAnalysisTemplate**: A/B 테스트 분석 템플릿 (통계적 유의성 검정)
  - **WorkflowAutomationTools**: 워크플로 자동화 및 DAG 생성
  - **MCP 도구 확장**: execute_analysis_template, list_analysis_templates, get_template_info, validate_data_for_template, create_workflow_dag, optimize_workflow_performance
- **CA 마일스톤 3.3: 데이터 품질 관리 시스템 구현 완료** ✅
  - **ComprehensiveQualityMetrics**: 종합 품질 메트릭 계산 (완전성, 정확성, 일관성, 유효성, 최신성, 유니크성)
  - **AdvancedOutlierDetector**: 고급 이상치 탐지 (Isolation Forest, LOF, One-Class SVM, 앙상블)
  - **DataCleaningPipeline**: 자동화된 데이터 정제 파이프라인
  - **QualityTrendAnalyzer**: 품질 트렌드 분석 및 예측
  - **MCP 도구 확장**: comprehensive_quality_metrics, advanced_outlier_detection, data_cleaning_pipeline, quality_trend_analysis, set_quality_thresholds
- **CA 마일스톤 3.2: 고급 통계 분석 및 시각화 구현 완료** ✅
  - **AdvancedStatistics**: 고급 통계 분석 모듈 (기술 통계, 상관관계, 분포 분석)
  - **AdvancedVisualization**: 인터랙티브 차트 및 시각화 모듈
  - **StatisticalTests**: 통계적 검정 모듈 (가설검정, A/B 테스트, 회귀분석)
  - **TimeSeriesAnalysis**: 시계열 분석 및 예측 모듈
  - **MCP 도구 확장**: advanced_statistics, interactive_charts, statistical_tests, time_series_analysis
- **CA 마일스톤 3.1: 통합 데이터 분석 레이어 구현 완료** ✅
  - **DataUnifier**: 다중 소스 데이터를 표준 테이블 형태로 통합
  - **SchemaMapper**: 스키마 매핑 및 정규화 시스템
  - **TypeConverter**: 고급 데이터 타입 변환 도구
  - **StreamingProcessor**: 대용량 데이터 스트리밍 처리
  - **IntegratedDataLayer**: 통합 데이터 분석 레이어 메인 클래스
  - **MCP 도구 확장**: data_unifier, schema_mapper, type_converter, streaming_processor, integrated_data_layer
- CD 마일스톤 2.3: 통합 대시보드 및 모니터링 시스템
- CD 마일스톤 2.2: 자동화 파이프라인 구현
- CD 마일스톤 2.1: 데이터 거버넌스 시스템
- C1 마일스톤 1.4: 기본 시각화 도구
- C1 마일스톤 1.3: 데이터 품질 검사 시스템
- C1 마일스톤 1.2: 기본 통계 분석 도구
- MCP 도구 확장 구현
- 개발/QA/PR/배포 단계별 테스트 스크립트
- Bridge 프로젝트 명령어 도움말 시스템
- **ML 알고리즘 모듈**: 시계열 분석, 이상치 탐지, 클러스터링, 차원 축소
- **ML 모델 관리**: 모델 레지스트리, 버전 관리, 추론 서비스
- **거버넌스 계약**: 데이터 계약, 모델 계약, 품질 규칙 관리

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
- **ML 관련 Cursor Rules 추가**: ml-algorithms.mdc, ml-models.mdc, governance-contracts.mdc
- **데이터 품질 관리 Cursor Rules 추가**: data-quality-management.mdc
- **워크플로 자동화 Cursor Rules 추가**: workflow-automation.mdc
- **README.md 갱신**: CA 마일스톤 3.3 & 3.4 완료 내용 반영, 프로젝트 구조 업데이트
- **CHANGELOG.md 갱신**: CA 마일스톤 3.3 & 3.4 완료 내용 추가

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
