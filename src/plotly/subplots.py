"""Subset of ``plotly.subplots`` for offline usage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .graph_objects import Figure


def make_subplots(
    rows: int = 1,
    cols: int = 1,
    subplot_titles: Optional[Iterable[str]] = None,
    **_: Any,
) -> Figure:
    fig = Figure()
    fig.update_layout(rows=rows, cols=cols, subplot_titles=list(subplot_titles or []))
    return fig


__all__ = ["make_subplots"]
