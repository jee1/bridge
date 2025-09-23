"""Minimal croniter stub for local testing.

This simplified implementation only supports incrementing the provided
``start_time`` by one minute on every ``get_next`` call.  It is sufficient for
the scheduler unit tests, which only validate that a datetime object is
returned.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Type


class croniter:
    def __init__(self, expression: str, start_time: datetime) -> None:
        self.expression = expression
        self.current = start_time

    def get_next(self, ret_type: Type[datetime] = datetime) -> datetime:
        self.current = self.current + timedelta(minutes=1)
        return self.current


__all__ = ["croniter"]
