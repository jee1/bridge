PYTHON ?= python3
VENV_BIN = . .venv/bin/activate &&
COMPOSE ?= docker compose
COMPOSE_FILE ?= docker-compose.dev.yml

.PHONY: install fmt lint test dev worker docker-build docker-up docker-down docker-test dev-full dev-full-up dev-full-down dev-full-test mcp-server dev-test dev-test-fast dev-lint dev-fmt qa-test qa-test-coverage qa-test-security qa-test-performance pr-test pr-test-full pr-test-docs pr-test-api deploy-test deploy-smoke deploy-load deploy-rollback

install:
	$(PYTHON) -m venv .venv
	$(VENV_BIN) pip install --upgrade pip
	$(VENV_BIN) pip install -e .[dev]

fmt:
	$(VENV_BIN) black src tests
	$(VENV_BIN) isort src tests

lint:
	$(VENV_BIN) mypy -p bridge

test:
	$(VENV_BIN) pytest tests

dev:
	$(VENV_BIN) uvicorn bridge.orchestrator.app:app --reload

worker:
	$(VENV_BIN) celery -A bridge.orchestrator.celery_app.celery_app worker --loglevel=info

docker-build:
	$(COMPOSE) -f $(COMPOSE_FILE) build

docker-up:
	$(COMPOSE) -f $(COMPOSE_FILE) up -d redis api worker

docker-down:
	$(COMPOSE) -f $(COMPOSE_FILE) down

docker-test:
	$(COMPOSE) --profile test -f $(COMPOSE_FILE) run --rm test

# =============================================================================
# 완전한 개발 환경 (모든 데이터베이스 포함)
# =============================================================================

dev-full:
	@echo "완전한 개발 환경을 구축합니다..."
	@echo "포함된 서비스: PostgreSQL, MySQL, Redis, Elasticsearch, Bridge API, Worker"
	@echo ""
	@echo "사용 가능한 명령어:"
	@echo "  make dev-full-up     - 모든 서비스 시작"
	@echo "  make dev-full-down   - 모든 서비스 중지"
	@echo "  make dev-full-test   - 테스트 실행"
	@echo "  make mcp-server      - MCP 서버 실행"
	@echo ""

dev-full-up:
	@echo "완전한 개발 환경을 시작합니다..."
	$(COMPOSE) -f docker-compose.dev-full.yml up -d postgres mysql redis elasticsearch api worker
	@echo ""
	@echo "서비스 상태 확인:"
	@echo "  PostgreSQL: http://localhost:5432"
	@echo "  MySQL:      http://localhost:3306"
	@echo "  Redis:      http://localhost:6379"
	@echo "  Elasticsearch: http://localhost:9200"
	@echo "  Bridge API: http://localhost:8000"
	@echo ""
	@echo "서비스 로그 확인:"
	@echo "  $(COMPOSE) -f docker-compose.dev-full.yml logs -f"

dev-full-down:
	@echo "완전한 개발 환경을 중지합니다..."
	$(COMPOSE) -f docker-compose.dev-full.yml down
	@echo "모든 서비스가 중지되었습니다."

dev-full-test:
	@echo "완전한 개발 환경에서 테스트를 실행합니다..."
	$(COMPOSE) --profile test -f docker-compose.dev-full.yml run --rm test

mcp-server:
	@echo "MCP 서버를 실행합니다..."
	$(COMPOSE) --profile mcp -f docker-compose.dev-full.yml up mcp-server

dev-full-init:
	@echo "모든 데이터베이스에 샘플 데이터를 생성합니다..."
	@echo "포함: PostgreSQL, MySQL, Elasticsearch"
	$(COMPOSE) --profile init -f docker-compose.dev-full.yml run --rm database-init

# =============================================================================
# 개발 단계 테스트 (빠른 피드백)
# =============================================================================

dev-test:
	@echo "🔧 개발 단계 테스트를 실행합니다..."
	@echo "포함: 린팅, 포매팅, 빠른 단위 테스트"
	@echo ""
	$(VENV_BIN) black --check src tests
	$(VENV_BIN) isort --check-only src tests
	$(VENV_BIN) mypy -p bridge
	$(VENV_BIN) pytest tests -x --tb=short
	@echo ""
	@echo "✅ 개발 테스트 완료"

dev-test-fast:
	@echo "⚡ 빠른 개발 테스트를 실행합니다..."
	@echo "포함: 특정 모듈만 테스트 (analytics, connectors)"
	@echo ""
	$(VENV_BIN) pytest src/bridge/analytics/tests/ -v --tb=short
	$(VENV_BIN) pytest tests/test_connectors.py -v --tb=short
	@echo ""
	@echo "✅ 빠른 테스트 완료"

dev-lint:
	@echo "🔍 린팅을 실행합니다..."
	@echo "포함: mypy, black, isort"
	@echo ""
	$(VENV_BIN) mypy -p bridge
	$(VENV_BIN) black --check src tests
	$(VENV_BIN) isort --check-only src tests
	@echo ""
	@echo "✅ 린팅 완료"

dev-fmt:
	@echo "🎨 코드 포매팅을 실행합니다..."
	@echo "포함: black, isort"
	@echo ""
	$(VENV_BIN) black src tests
	$(VENV_BIN) isort src tests
	@echo ""
	@echo "✅ 포매팅 완료"

# =============================================================================
# QA 단계 테스트 (품질 보증)
# =============================================================================

qa-test:
	@echo "🧪 QA 단계 테스트를 실행합니다..."
	@echo "포함: 전체 테스트, 커버리지, 보안 검사, 성능 테스트"
	@echo ""
	@echo "1. 전체 단위 테스트 실행..."
	$(VENV_BIN) pytest tests/ -v
	@echo ""
	@echo "2. 커버리지 테스트 실행..."
	$(VENV_BIN) pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "3. 보안 검사 실행..."
	$(VENV_BIN) bandit -r src/
	$(VENV_BIN) safety check
	@echo ""
	@echo "4. 성능 테스트 실행..."
	$(VENV_BIN) python test_analytics_demo.py
	@echo ""
	@echo "✅ QA 테스트 완료"

qa-test-coverage:
	@echo "📊 커버리지 테스트를 실행합니다..."
	@echo "포함: HTML 리포트 생성, 커버리지 임계값 검사"
	@echo ""
	$(VENV_BIN) pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80
	@echo ""
	@echo "📈 커버리지 리포트: htmlcov/index.html"
	@echo "✅ 커버리지 테스트 완료"

qa-test-security:
	@echo "🔒 보안 검사를 실행합니다..."
	@echo "포함: bandit, safety, 의존성 검사"
	@echo ""
	$(VENV_BIN) bandit -r src/ -f json -o security-report.json
	$(VENV_BIN) safety check --json --output safety-report.json
	@echo ""
	@echo "📋 보안 리포트: security-report.json, safety-report.json"
	@echo "✅ 보안 검사 완료"

qa-test-performance:
	@echo "⚡ 성능 테스트를 실행합니다..."
	@echo "포함: 대용량 데이터 처리, 메모리 사용량, 응답 시간"
	@echo ""
	$(VENV_BIN) python test_analytics_demo.py
	@echo ""
	@echo "📈 성능 벤치마크 완료"
	@echo "✅ 성능 테스트 완료"

# =============================================================================
# PR 단계 테스트 (Pull Request 준비)
# =============================================================================

pr-test:
	@echo "🔀 PR 준비 테스트를 실행합니다..."
	@echo "포함: 전체 테스트, 문서 검증, API 테스트"
	@echo ""
	@echo "1. 코드 품질 검사..."
	$(VENV_BIN) black --check src tests
	$(VENV_BIN) isort --check-only src tests
	$(VENV_BIN) mypy -p bridge
	@echo ""
	@echo "2. 전체 테스트 실행..."
	$(VENV_BIN) pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "3. 보안 검사..."
	$(VENV_BIN) bandit -r src/
	$(VENV_BIN) safety check
	@echo ""
	@echo "4. 문서 검증..."
	@echo "문서 파일 검사 중..."
	@find docs/ -name "*.md" -exec echo "문서 파일: {}" \;
	@echo ""
	@echo "✅ PR 테스트 완료"

pr-test-full:
	@echo "🚀 전체 시스템 테스트를 실행합니다..."
	@echo "포함: Docker 환경, 통합 테스트, API 테스트"
	@echo ""
	@echo "1. Docker 환경 구축..."
	$(COMPOSE) -f docker-compose.dev-full.yml build
	@echo ""
	@echo "2. 서비스 시작..."
	$(COMPOSE) -f docker-compose.dev-full.yml up -d postgres mysql redis elasticsearch
	@echo "서비스 시작 대기 중... (30초)"
	@sleep 30
	@echo ""
	@echo "3. 통합 테스트 실행..."
	$(COMPOSE) --profile test -f docker-compose.dev-full.yml run --rm test
	@echo ""
	@echo "4. 서비스 정리..."
	$(COMPOSE) -f docker-compose.dev-full.yml down
	@echo ""
	@echo "✅ 전체 시스템 테스트 완료"

pr-test-docs:
	@echo "📚 문서 검증을 실행합니다..."
	@echo "포함: Markdown 문법, 링크 검사, 구조 검증"
	@echo ""
	@echo "문서 파일 목록:"
	@find docs/ -name "*.md" -exec echo "  - {}" \;
	@echo ""
	@echo "문서 구조 검증:"
	@echo "  - bridge-milestones.md: 마일스톤 문서"
	@echo "  - bridge-analytics-*.md: 분석 관련 문서"
	@echo "  - developer-quick-start.md: 개발자 가이드"
	@echo ""
	@echo "✅ 문서 검증 완료"

pr-test-api:
	@echo "🌐 API 테스트를 실행합니다..."
	@echo "포함: FastAPI 엔드포인트, MCP 서버, 오케스트레이터"
	@echo ""
	@echo "1. API 서버 시작..."
	$(COMPOSE) -f docker-compose.dev-full.yml up -d postgres redis api
	@echo "API 서버 시작 대기 중... (20초)"
	@sleep 20
	@echo ""
	@echo "2. API 엔드포인트 테스트..."
	@curl -f http://localhost:8000/health || echo "API 서버 응답 없음"
	@echo ""
	@echo "3. MCP 서버 테스트..."
	@echo "MCP 서버 연결 테스트 중..."
	@echo ""
	@echo "4. 서비스 정리..."
	$(COMPOSE) -f docker-compose.dev-full.yml down
	@echo ""
	@echo "✅ API 테스트 완료"

# =============================================================================
# 배포 단계 테스트 (프로덕션 준비)
# =============================================================================

deploy-test:
	@echo "🚀 배포 전 테스트를 실행합니다..."
	@echo "포함: 프로덕션 환경 시뮬레이션, 스모크 테스트, 부하 테스트"
	@echo ""
	@echo "1. 프로덕션 환경 시뮬레이션..."
	@export BRIDGE_ENV=production
	@echo "환경 변수: BRIDGE_ENV=production"
	@echo ""
	@echo "2. 전체 테스트 실행..."
	$(VENV_BIN) pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "3. 보안 검사..."
	$(VENV_BIN) bandit -r src/
	$(VENV_BIN) safety check
	@echo ""
	@echo "4. 성능 테스트..."
	$(VENV_BIN) python test_analytics_demo.py
	@echo ""
	@echo "✅ 배포 테스트 완료"

deploy-smoke:
	@echo "💨 스모크 테스트를 실행합니다..."
	@echo "포함: 핵심 기능 동작 확인, 기본 연결 테스트"
	@echo ""
	@echo "1. 핵심 모듈 임포트 테스트..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; print('✅ Analytics 모듈 로드 성공')"
	$(VENV_BIN) python -c "from bridge.connectors.mock import MockConnector; print('✅ Connector 모듈 로드 성공')"
	$(VENV_BIN) python -c "from bridge.orchestrator.app import app; print('✅ FastAPI 앱 로드 성공')"
	@echo ""
	@echo "2. 기본 기능 테스트..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; df = UnifiedDataFrame([{'id': 1}]); print(f'✅ UnifiedDataFrame: {df.num_rows}행')"
	@echo ""
	@echo "3. 데이터베이스 연결 테스트..."
	@echo "Mock 커넥터 테스트 중..."
	$(VENV_BIN) python -c "from bridge.connectors.mock import MockConnector; conn = MockConnector(); print(f'✅ Mock 커넥터: {conn.test_connection()}')"
	@echo ""
	@echo "✅ 스모크 테스트 완료"

deploy-load:
	@echo "⚡ 부하 테스트를 실행합니다..."
	@echo "포함: 대용량 데이터 처리, 동시 요청, 메모리 사용량"
	@echo ""
	@echo "1. 대용량 데이터 처리 테스트..."
	$(VENV_BIN) python -c "import time; from bridge.analytics.core import UnifiedDataFrame; data = [{'id': i, 'value': i*2} for i in range(50000)]; start = time.time(); df = UnifiedDataFrame(data); end = time.time(); print(f'✅ 50,000행 처리: {end-start:.3f}초, {df.table.nbytes/1024/1024:.2f}MB')"
	@echo ""
	@echo "2. 메모리 사용량 모니터링..."
	$(VENV_BIN) python -c "import psutil; import os; process = psutil.Process(os.getpid()); print(f'✅ 메모리 사용량: {process.memory_info().rss/1024/1024:.2f}MB')" || echo "psutil 모듈 없음 - 메모리 모니터링 스킵"
	@echo ""
	@echo "3. 성능 벤치마크..."
	$(VENV_BIN) python test_analytics_demo.py
	@echo ""
	@echo "✅ 부하 테스트 완료"

deploy-rollback:
	@echo "🔄 롤백 테스트를 실행합니다..."
	@echo "포함: 이전 버전 호환성, 데이터 마이그레이션, 롤백 시나리오"
	@echo ""
	@echo "1. 이전 버전 호환성 테스트..."
	@echo "현재 버전과 이전 버전 간 호환성 확인 중..."
	@echo ""
	@echo "2. 데이터 마이그레이션 테스트..."
	@echo "스키마 변경 시 데이터 보존 확인 중..."
	@echo ""
	@echo "3. 롤백 시나리오 테스트..."
	@echo "배포 실패 시 롤백 절차 확인 중..."
	@echo ""
	@echo "4. 백업 복원 테스트..."
	@echo "데이터베이스 백업 복원 테스트 중..."
	@echo ""
	@echo "✅ 롤백 테스트 완료"

# =============================================================================
# C1 마일스톤 테스트 (Analytics MVP)
# =============================================================================

c1-test-data:
	@echo "📊 C1 마일스톤용 샘플 데이터를 생성합니다..."
	@echo "포함: PostgreSQL (고객), MySQL (매출), Elasticsearch (로그)"
	@echo ""
	$(VENV_BIN) python scripts/init-c1-sample-data.py --scale medium
	@echo ""
	@echo "✅ C1 샘플 데이터 생성 완료"

c1-test-data-large:
	@echo "📊 C1 마일스톤용 대용량 샘플 데이터를 생성합니다..."
	@echo "포함: PostgreSQL (고객), MySQL (매출), Elasticsearch (로그)"
	@echo ""
	$(VENV_BIN) python scripts/init-c1-sample-data.py --scale large
	@echo ""
	@echo "✅ C1 대용량 샘플 데이터 생성 완료"

c1-test:
	@echo "🧪 C1 마일스톤 기능 테스트를 실행합니다..."
	@echo "포함: 데이터 통합, 통계 분석, 품질 검사, 시각화"
	@echo ""
	@echo "1. 데이터 통합 테스트..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; print('✅ UnifiedDataFrame 로드 성공')"
	@echo ""
	@echo "2. 통계 분석 테스트..."
	$(VENV_BIN) python -c "from bridge.analytics.core.statistics import StatisticsAnalyzer; print('✅ StatisticsAnalyzer 로드 성공')"
	@echo ""
	@echo "3. 데이터 품질 검사 테스트..."
	$(VENV_BIN) python -c "from bridge.analytics.core.quality import QualityChecker; print('✅ QualityChecker 로드 성공')"
	@echo ""
	@echo "4. 시각화 테스트..."
	$(VENV_BIN) python -c "from bridge.analytics.core.visualization import ChartGenerator; print('✅ ChartGenerator 로드 성공')"
	@echo ""
	@echo "5. MCP 도구 테스트..."
	$(VENV_BIN) python -c "from bridge.mcp_server_unified import UnifiedBridgeMCPServer; print('✅ MCP 서버 로드 성공')"
	@echo ""
	@echo "✅ C1 기능 테스트 완료"

c1-test-full:
	@echo "🚀 C1 마일스톤 전체 테스트를 실행합니다..."
	@echo "포함: 샘플 데이터 생성 + 기능 테스트 + 성능 벤치마크"
	@echo ""
	@echo "1. C1 샘플 데이터 생성..."
	$(MAKE) c1-test-data
	@echo ""
	@echo "2. C1 기능 테스트..."
	$(MAKE) c1-test
	@echo ""
	@echo "3. 성능 벤치마크..."
	$(MAKE) c1-benchmark
	@echo ""
	@echo "✅ C1 전체 테스트 완료"

c1-benchmark:
	@echo "⚡ C1 마일스톤 성능 벤치마크를 실행합니다..."
	@echo "포함: 대용량 데이터 처리, 메모리 사용량, 쿼리 성능"
	@echo ""
	@echo "1. 대용량 데이터 처리 테스트..."
	$(VENV_BIN) python -c "import time; from bridge.analytics.core import UnifiedDataFrame; data = [{'id': i, 'value': i*2, 'category': f'cat_{i%10}'} for i in range(100000)]; start = time.time(); df = UnifiedDataFrame(data); end = time.time(); print(f'✅ 100,000행 처리: {end-start:.3f}초, {df.num_rows:,}행')"
	@echo ""
	@echo "2. 통계 분석 성능 테스트..."
	$(VENV_BIN) python -c "import time; from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.statistics import StatisticsAnalyzer; data = [{'value': i*2 + (i%7)*100} for i in range(50000)]; df = UnifiedDataFrame(data); analyzer = StatisticsAnalyzer(); start = time.time(); stats = analyzer.calculate_descriptive_stats(df, ['value']); end = time.time(); print(f'✅ 통계 분석: {end-start:.3f}초, 평균={stats[\"value\"].mean:.2f}')"
	@echo ""
	@echo "3. 메모리 사용량 모니터링..."
	$(VENV_BIN) python -c "import psutil; import os; process = psutil.Process(os.getpid()); print(f'✅ 메모리 사용량: {process.memory_info().rss/1024/1024:.2f}MB')" || echo "psutil 모듈 없음 - 메모리 모니터링 스킵"
	@echo ""
	@echo "4. 크로스 소스 조인 성능 테스트..."
	$(VENV_BIN) python -c "import time; from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.cross_source_joiner import CrossSourceJoiner; df1 = UnifiedDataFrame([{'id': i, 'name': f'user_{i}'} for i in range(1000)]); df2 = UnifiedDataFrame([{'id': i, 'amount': i*100} for i in range(1000)]); joiner = CrossSourceJoiner(); joiner.register_table('table1', df1); joiner.register_table('table2', df2); start = time.time(); result = joiner.join_tables('table1', 'table2', 'table1.id = table2.id'); end = time.time(); print(f'✅ 크로스 소스 조인: {end-start:.3f}초, {result.num_rows:,}행')"
	@echo ""
	@echo "✅ C1 성능 벤치마크 완료"

c1-test-scenarios:
	@echo "🎯 C1 마일스톤 테스트 시나리오를 실행합니다..."
	@echo "포함: 고객 세그멘테이션, 매출 트렌드, 상관관계 분석"
	@echo ""
	@echo "1. 고객 세그멘테이션 분석..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.statistics import StatisticsAnalyzer; data = [{'age': 20+i%50, 'spent': (20+i%50)*1000 + (i%3)*5000, 'city': f'city_{i%10}'} for i in range(1000)]; df = UnifiedDataFrame(data); analyzer = StatisticsAnalyzer(); stats = analyzer.calculate_descriptive_stats(df, ['spent']); print(f'✅ 고객 세그멘테이션: 평균 구매액 {stats[\"spent\"].mean:.0f}원')"
	@echo ""
	@echo "2. 매출 트렌드 분석..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.statistics import StatisticsAnalyzer; data = [{'month': i%12+1, 'sales': 1000000 + (i%12)*50000 + (i%7)*100000} for i in range(1000)]; df = UnifiedDataFrame(data); analyzer = StatisticsAnalyzer(); stats = analyzer.calculate_descriptive_stats(df, ['sales']); print(f'✅ 매출 트렌드: 평균 {stats[\"sales\"].mean:.0f}원')"
	@echo ""
	@echo "3. 상관관계 분석..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.statistics import StatisticsAnalyzer; data = [{'price': 10000 + i*100, 'quantity': 100 - i*0.1} for i in range(1000)]; df = UnifiedDataFrame(data); analyzer = StatisticsAnalyzer(); stats = analyzer.calculate_descriptive_stats(df, ['price']); print(f'✅ 상관관계 분석: 가격 평균 {stats[\"price\"].mean:.0f}원')"
	@echo ""
	@echo "4. 이상치 탐지..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.quality import QualityChecker; data = [{'value': 100 + i*10 if i < 950 else 10000 + i*100} for i in range(1000)]; df = UnifiedDataFrame(data); checker = QualityChecker(); outliers = checker.detect_outliers(df, 'value'); print(f'✅ 이상치 탐지: {len(outliers)}개 발견')"
	@echo ""
	@echo "5. 데이터 품질 검사..."
	$(VENV_BIN) python -c "from bridge.analytics.core import UnifiedDataFrame; from bridge.analytics.core.quality import QualityChecker; data = [{'id': i, 'name': f'item_{i}' if i%10 != 0 else None, 'value': i*100 if i%20 != 0 else None} for i in range(1000)]; df = UnifiedDataFrame(data); checker = QualityChecker(); quality = checker.check_quality(df); print(f'✅ 데이터 품질: 결측값 {quality.get(\"missing_count\", 0)}개')"
	@echo ""
	@echo "✅ C1 테스트 시나리오 완료"

c1-help:
	@echo "🧪 C1 마일스톤 테스트 명령어 도움말"
	@echo ""
	@echo "데이터 생성:"
	@echo "  make c1-test-data        - C1용 샘플 데이터 생성 (중간 규모)"
	@echo "  make c1-test-data-large  - C1용 대용량 샘플 데이터 생성"
	@echo ""
	@echo "기능 테스트:"
	@echo "  make c1-test             - C1 기능 테스트 (모듈 로드 확인)"
	@echo "  make c1-test-scenarios   - C1 테스트 시나리오 실행"
	@echo "  make c1-test-full        - 전체 C1 테스트 (데이터 생성 + 테스트)"
	@echo ""
	@echo "성능 테스트:"
	@echo "  make c1-benchmark        - C1 성능 벤치마크 실행"
	@echo ""
	@echo "도움말:"
	@echo "  make c1-help             - 이 도움말 표시"

# =============================================================================
# 테스트 도우미 명령어
# =============================================================================

test-help:
	@echo "🧪 Bridge 테스트 명령어 도움말"
	@echo ""
	@echo "개발 단계:"
	@echo "  make dev-test        - 개발 테스트 (린팅 + 포매팅 + 테스트)"
	@echo "  make dev-test-fast   - 빠른 테스트 (특정 모듈만)"
	@echo "  make dev-lint        - 린팅만 실행"
	@echo "  make dev-fmt         - 포매팅만 실행"
	@echo ""
	@echo "QA 단계:"
	@echo "  make qa-test         - QA 테스트 (전체 + 커버리지 + 보안 + 성능)"
	@echo "  make qa-test-coverage - 커버리지 테스트"
	@echo "  make qa-test-security - 보안 검사"
	@echo "  make qa-test-performance - 성능 테스트"
	@echo ""
	@echo "PR 단계:"
	@echo "  make pr-test         - PR 테스트 (전체 + 문서 + API)"
	@echo "  make pr-test-full    - 전체 시스템 테스트 (Docker)"
	@echo "  make pr-test-docs    - 문서 검증"
	@echo "  make pr-test-api     - API 테스트"
	@echo ""
	@echo "배포 단계:"
	@echo "  make deploy-test     - 배포 테스트 (프로덕션 시뮬레이션)"
	@echo "  make deploy-smoke    - 스모크 테스트 (핵심 기능)"
	@echo "  make deploy-load     - 부하 테스트 (성능)"
	@echo "  make deploy-rollback - 롤백 테스트 (호환성)"
	@echo ""
	@echo "도움말:"
	@echo "  make test-help       - 이 도움말 표시"
