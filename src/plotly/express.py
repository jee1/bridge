"""Subset of ``plotly.express`` API for test environments.

The real Plotly distribution is not available in this environment, so this
module provides minimal stand-ins that behave like Plotly figures from the
perspective of the project tests.  The figures return simple dictionaries
containing the supplied data and configuration.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from .graph_objects import Figure


def scatter_matrix(
    df: Any,
    dimensions: Optional[Iterable[str]] = None,
    title: str | None = None,
) -> Figure:
    fig = Figure()
    fig.add_trace(
        {
            "type": "scatter_matrix",
            "dimensions": list(dimensions or []),
            "title": title or "",
        }
    )
    if title:
        fig.update_layout(title=title)
    return fig


def density_heatmap(
    df: Any,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: str | None = None,
    **_: Any,
) -> Figure:
    fig = Figure()
    fig.add_trace({"type": "density_heatmap", "x": x, "y": y, "title": title or ""})
    if title:
        fig.update_layout(title=title)
    return fig


__all__ = ["scatter_matrix", "density_heatmap"]
