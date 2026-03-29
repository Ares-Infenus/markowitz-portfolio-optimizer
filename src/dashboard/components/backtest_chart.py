"""Cumulative returns + drawdown subplot — premium version."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_BACKTEST_PATH = Path("data/results/backtest_results.parquet")

_METHOD_COLORS = {
    "mean_variance": "#00D4FF",
    "risk_parity": "#FF6B6B",
    "cvar": "#FFD93D",
    "black_litterman": "#6BCB77",
    "equal_weight": "#4B5563",
}
_METHOD_LABELS = {
    "mean_variance": "Mean-Variance",
    "risk_parity": "Risk Parity",
    "cvar": "CVaR",
    "black_litterman": "Black-Litterman",
    "equal_weight": "Equal Weight",
}

_EVENTS = [
    {
        "name": "COVID Crash",
        "start": "2020-02-20",
        "end": "2020-04-01",
        "color": "rgba(239,68,68,0.08)",
    },
    {
        "name": "Bear Market 2022",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "color": "rgba(245,158,11,0.07)",
    },
]


def render(selected_methods: list[str]) -> None:
    if not _BACKTEST_PATH.exists():
        st.warning("No backtest data found.")
        return

    df = pd.read_parquet(_BACKTEST_PATH)
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    for method in df["method"].unique():
        if method not in selected_methods:
            continue

        sub = df[df["method"] == method].sort_values("date")
        start = pd.DataFrame(
            [
                {
                    "date": sub["date"].min() - pd.offsets.MonthBegin(1),
                    "portfolio_value": 1.0,
                }
            ]
        )
        sub_full = pd.concat([start, sub[["date", "portfolio_value"]]], ignore_index=True)

        is_bench = method == "equal_weight"
        color = _METHOD_COLORS.get(method, "#888")
        label = _METHOD_LABELS.get(method, method)
        width = 1.5 if is_bench else 2.5
        dash = "dot" if is_bench else "solid"

        # ── Cumulative return ──────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=sub_full["date"],
                y=sub_full["portfolio_value"],
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>Value: <b>%{{y:.3f}}</b><extra></extra>",
                legendgroup=method,
            ),
            row=1,
            col=1,
        )

        # ── Drawdown ───────────────────────────────────────────────────
        vals = sub_full["portfolio_value"].values
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak * 100

        fig.add_trace(
            go.Scatter(
                x=sub_full["date"],
                y=dd,
                mode="lines",
                name=label,
                line=dict(color=color, width=1.2, dash=dash),
                fill="tozeroy",
                fillcolor=(
                    color.replace(")", ",0.07)").replace("rgb", "rgba")
                    if color.startswith("rgb")
                    else _hex_to_rgba(color, 0.07)
                ),
                hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>Drawdown: <b>%{{y:.1f}}%</b><extra></extra>",
                legendgroup=method,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # ── Event shading ──────────────────────────────────────────────────────
    for ev in _EVENTS:
        for r in [1, 2]:
            fig.add_vrect(
                x0=ev["start"],
                x1=ev["end"],
                fillcolor=ev["color"],
                line_width=0,
                row=r,
                col=1,
            )
        # Label only on top chart
        fig.add_annotation(
            x=pd.Timestamp(ev["start"]) + (pd.Timestamp(ev["end"]) - pd.Timestamp(ev["start"])) / 2,
            y=1.01,
            yref="paper",
            text=ev["name"],
            showarrow=False,
            font=dict(size=9, color="#4B5563"),
            xanchor="center",
        )

    # ── Breakeven line ─────────────────────────────────────────────────────
    fig.add_hline(y=1.0, line_dash="dash", line_color="#1E2738", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#1E2738", line_width=1, row=2, col=1)

    fig.update_layout(
        **_dark_layout(height=560),
        legend=dict(
            orientation="h",
            x=0,
            y=1.08,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        hovermode="x unified",
        margin=dict(t=30, b=10, l=10, r=10),
    )
    fig.update_yaxes(
        title_text="Portfolio Value",
        title_font_size=11,
        row=1,
        col=1,
        gridcolor="#111827",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Drawdown %",
        title_font_size=11,
        row=2,
        col=1,
        gridcolor="#111827",
        ticksuffix="%",
        zeroline=False,
    )
    fig.update_xaxes(gridcolor="#0D1220", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _dark_layout(height: int = 500) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        font=dict(color="#9CA3AF", family="Inter, sans-serif"),
        height=height,
    )
