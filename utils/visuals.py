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
    "history": "#fbafe4",
}

SYMBOL_MAP = {
    "background": "circle",
    "result": "circle",
    "query": "diamond",
    "history": "circle",
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

    # Style traces.
    fig.update_traces(
        selector=lambda trace: trace.name == "query",
        marker=dict(size=10),
    )
    fig.update_traces(
        selector=lambda trace: trace.name == "background",
        opacity=0.3,
        marker=dict(size=4),
    )
    fig.update_traces(
        selector=lambda trace: trace.name == "history",
        opacity=0.6,
        marker=dict(size=6),
    )

    # Add trajectory line.
    history_df = df[df["label"] == "history"]
    query_points = df[df["label"] == "query"]
    if not history_df.empty and not query_points.empty:
        query_point = query_points.iloc[0]
        path_x = history_df["x"].tolist() + [query_point["x"]]
        path_y = history_df["y"].tolist() + [query_point["y"]]
        path_z = history_df["z"].tolist() + [query_point["z"]]
        fig.add_trace(
            go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode="lines",
                line=dict(color="#d62728", width=3, dash="dot"),
                name="trajectory",
                hoverinfo="skip",
            )
        )

    # Optional distance ruler.
    if ruler_target_id:
        query_points = df[df["label"] == "query"]
        target_points = df[df["id"] == ruler_target_id]
        if not query_points.empty and not target_points.empty:
            qp = query_points.iloc[0]
            tp = target_points.iloc[0]
            fig.add_trace(
                go.Scatter3d(
                    x=[qp["x"], tp["x"]],
                    y=[qp["y"], tp["y"]],
                    z=[qp["z"], tp["z"]],
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
