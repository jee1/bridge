"""Celery 애플리케이션 초기화."""
from __future__ import annotations

import os
from typing import Any

from celery import Celery


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def create_celery_app(**overrides: Any) -> Celery:
    broker_url = overrides.get(
        "broker_url",
        os.getenv("BRIDGE_CELERY_BROKER_URL", "memory://"),
    )
    result_backend = overrides.get(
        "result_backend",
        os.getenv("BRIDGE_CELERY_RESULT_BACKEND", "cache+memory://"),
    )

    app = Celery("bridge-orchestrator", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_always_eager=overrides.get(
            "task_always_eager",
            _bool_env("BRIDGE_CELERY_TASK_ALWAYS_EAGER", True),
        ),
        task_eager_propagates=True,
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
    )
    app.autodiscover_tasks(["bridge.orchestrator"])
    return app


celery_app = create_celery_app()
