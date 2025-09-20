PYTHON ?= python3
VENV_BIN = . .venv/bin/activate &&

.PHONY: install fmt lint test dev

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
