"""Portfolio weights heatmap — tickers × rebalance months."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_WEIGHTS_PATH = Path("data/results/weights_store.parquet")
_TICKERS = ["AAPL", "MSFT", "JPM", "GS", "JNJ", "UNH", "XOM", "NEE", "AMZN", "PG"]


def render(selected_methods: list[str]) -> None:
    st.subheader("Portfolio Weights Over Time")

    if not _WEIGHTS_PATH.exists():
        st.warning("Run the pipeline first to generate weights data.")
        return

    df = pd.read_parquet(_WEIGHTS_PATH)
    df["date"] = pd.to_datetime(df["date"])

    methods_available = [m for m in df["method"].unique() if m in selected_methods]
    if not methods_available:
        st.info("Select at least one method in the sidebar.")
        return

    method = st.selectbox(
        "Select method",
        options=methods_available,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    sub = df[df["method"] == method].sort_values("date")
    ticker_cols = [c for c in _TICKERS if c in sub.columns]
    pivot = sub.set_index("date")[ticker_cols].T

    # Format column labels as YYYY-MM
    pivot.columns = [d.strftime("%Y-%m") for d in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values * 100,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[
                [0.0, "#0E1117"],
                [0.3, "#003355"],
                [0.7, "#006699"],
                [1.0, "#00D4FF"],
            ],
            zmin=0,
            zmax=20,
            colorbar=dict(title="Weight (%)", ticksuffix="%"),
            hovertemplate="Date: %{x}<br>Ticker: %{y}<br>Weight: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        **_dark_layout(),
        xaxis=dict(title="Rebalance Month", tickangle=-45),
        yaxis=dict(title="Ticker"),
    )

    st.plotly_chart(fig, use_container_width=True)


def _dark_layout() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#CCCCCC"),
        height=480,
    )
