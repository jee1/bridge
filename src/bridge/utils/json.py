"""Custom JSON helpers for Bridge."""
from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable
from uuid import UUID
import json

JsonDefault = Callable[[Any], Any]


def default_serializer(value: Any) -> Any:
    """Convert non-serializable objects into JSON-friendly values."""
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(
    obj: Any,
    *,
    ensure_ascii: bool = False,
    default: JsonDefault | None = None,
    **kwargs: Any,
) -> str:
    """Dump JSON with sensible defaults for Bridge."""
    serializer = default or default_serializer
    return json.dumps(obj, ensure_ascii=ensure_ascii, default=serializer, **kwargs)


def loads(s: str | bytes, *, encoding: str | None = None, **kwargs: Any) -> Any:
    """Wrapper for json.loads to keep symmetry with dumps."""
    if isinstance(s, bytes):
        if encoding is None:
            encoding = "utf-8"
        s = s.decode(encoding)
    return json.loads(s, **kwargs)

