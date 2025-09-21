PYTHON ?= python3
VENV_BIN = . .venv/bin/activate &&
COMPOSE ?= docker compose
COMPOSE_FILE ?= docker-compose.dev.yml

.PHONY: install fmt lint test dev worker docker-build docker-up docker-down docker-test dev-full dev-full-up dev-full-down dev-full-test mcp-server

install:
	$(PYTHON) -m venv .venv
	$(VENV_BIN) pip install --upgrade pip
	$(VENV_BIN) pip install -e .[dev]

fmt:
	$(VENV_BIN) black src tests
	$(VENV_BIN) isort src tests

lint:
	$(VENV_BIN) mypy src

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
