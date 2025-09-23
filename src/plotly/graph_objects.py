"""Simplified Graph Objects subset for offline testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _BaseTrace:
    trace_type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.trace_type}
        result.update(self.data)
        return result


class Scatter(_BaseTrace):
    def __init__(
        self,
        x: Optional[List[Any]] = None,
        y: Optional[List[Any]] = None,
        mode: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            "scatter",
            {"x": x, "y": y, "mode": mode, "name": name},
        )


class Bar(_BaseTrace):
    def __init__(
        self,
        x: Optional[List[Any]] = None,
        y: Optional[List[Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__("bar", {"x": x, "y": y, "name": name})


class Figure:
    def __init__(self) -> None:
        self._data: List[Dict[str, Any]] = []
        self._layout: Dict[str, Any] = {}

    def add_trace(self, trace: Any, row: Optional[int] = None, col: Optional[int] = None) -> None:
        if hasattr(trace, "to_dict"):
            trace_dict = trace.to_dict()
        else:
            trace_dict = dict(trace)

        if row is not None or col is not None:
            trace_dict.update({"subplot_row": row, "subplot_col": col})

        self._data.append(trace_dict)

    def update_layout(self, **kwargs: Any) -> None:
        self._layout.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self._data, "layout": self._layout}


__all__ = ["Figure", "Scatter", "Bar"]
