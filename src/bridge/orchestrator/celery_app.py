"""Celery 애플리케이션 초기화."""
from __future__ import annotations

import os

from celery import Celery


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


BROKER_URL = os.getenv("BRIDGE_CELERY_BROKER_URL", "memory://")
RESULT_BACKEND = os.getenv("BRIDGE_CELERY_RESULT_BACKEND", "cache+memory://")

celery_app = Celery("bridge-orchestrator", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_always_eager=_bool_env("BRIDGE_CELERY_TASK_ALWAYS_EAGER", True),
    task_eager_propagates=True,
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
)

celery_app.autodiscover_tasks(["bridge.orchestrator"])
