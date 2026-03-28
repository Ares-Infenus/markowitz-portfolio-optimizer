"""Cumulative returns line chart with COVID and bear-market shaded regions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_BACKTEST_PATH = Path("data/results/backtest_results.parquet")

_METHOD_COLORS = {
    "mean_variance": "#00D4FF",
    "risk_parity": "#FF6B6B",
    "cvar": "#FFD93D",
    "black_litterman": "#6BCB77",
    "equal_weight": "#888888",
}

# Annotated market events
_EVENTS = [
    {"name": "COVID Crash", "start": "2020-02-20", "end": "2020-04-01", "color": "rgba(255,100,100,0.12)"},
    {"name": "Bear Market 2022", "start": "2022-01-01", "end": "2022-12-31", "color": "rgba(255,180,0,0.10)"},
]


def render(selected_methods: list[str]) -> None:
    st.subheader("Cumulative Portfolio Returns")

    if not _BACKTEST_PATH.exists():
        st.warning("Run the pipeline first to generate backtest data.")
        return

    df = pd.read_parquet(_BACKTEST_PATH)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    # ── Shaded event regions ──────────────────────────────────────────────
    for event in _EVENTS:
        fig.add_vrect(
            x0=event["start"],
            x1=event["end"],
            fillcolor=event["color"],
            line_width=0,
            annotation_text=event["name"],
            annotation_position="top left",
            annotation=dict(font_size=11, font_color="#AAAAAA"),
        )

    # ── Portfolio lines ───────────────────────────────────────────────────
    for method in df["method"].unique():
        if method not in selected_methods and method != "equal_weight":
            continue

        sub = df[df["method"] == method].sort_values("date")
        # Prepend starting value of 1.0
        start_row = pd.DataFrame(
            [{"date": sub["date"].min() - pd.offsets.MonthBegin(1), "portfolio_value": 1.0}]
        )
        sub = pd.concat([start_row, sub[["date", "portfolio_value"]]], ignore_index=True)

        is_benchmark = method == "equal_weight"
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["portfolio_value"],
                mode="lines",
                name=method.replace("_", " ").title(),
                line=dict(
                    color=_METHOD_COLORS.get(method, "#AAAAAA"),
                    width=1.5 if is_benchmark else 2.5,
                    dash="dot" if is_benchmark else "solid",
                ),
                hovertemplate="%{x|%Y-%m}<br>Value: %{y:.3f}<extra>" + method + "</extra>",
            )
        )

    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Date",
        yaxis_title="Portfolio Value (start = 1.0)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig, use_container_width=True)


def _dark_layout() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#CCCCCC"),
        height=500,
    )
