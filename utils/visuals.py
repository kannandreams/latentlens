"""Plotly visual helpers for the Vector Debugger."""
from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


COLOR_MAP = {
    "background": "#b0b0b0",
    "result": "#1f77b4",
    "query": "#d62728",
}

SYMBOL_MAP = {
    "background": "circle",
    "result": "circle",
    "query": "star",
}


def build_scatter(df: pd.DataFrame, ruler_target_id: Optional[str] = None) -> go.Figure:
    """Create a 3D scatter plot with optional distance ruler to the query point."""
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        color_discrete_map=COLOR_MAP,
        symbol="label",
        symbol_map=SYMBOL_MAP,
        hover_data=["id", "score", "cosine_sim_to_query"],
        custom_data=["id"],
    )

    # Enlarge query marker.
    query_idx = df["label"] == "query"
    if query_idx.any():
        fig.update_traces(
            selector=lambda trace: trace.name == "query",
            marker=dict(size=10),
        )

    # Optional distance ruler.
    if ruler_target_id:
        query_point = df[query_idx]
        target_point = df[df["id"] == ruler_target_id]
        if not query_point.empty and not target_point.empty:
            qx, qy, qz = query_point[["x", "y", "z"]].iloc[0]
            tx, ty, tz = target_point[["x", "y", "z"]].iloc[0]
            fig.add_trace(
                go.Scatter3d(
                    x=[qx, tx],
                    y=[qy, ty],
                    z=[qz, tz],
                    mode="lines+text",
                    text=[None, "distance"],
                    textposition="top center",
                    line=dict(color="#ff7f0e", width=4),
                    name="distance",
                    hoverinfo="skip",
                )
            )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
