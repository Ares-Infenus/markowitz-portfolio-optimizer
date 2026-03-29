"""Portfolio weights heatmap — premium version."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_WEIGHTS_PATH = Path("data/results/weights_store.parquet")
_TICKERS = ["AAPL", "MSFT", "JPM", "GS", "JNJ", "UNH", "XOM", "NEE", "AMZN", "PG"]

_METHOD_LABELS = {
    "mean_variance": "Mean-Variance",
    "risk_parity": "Risk Parity",
    "cvar": "CVaR",
    "black_litterman": "Black-Litterman",
}


def render(selected: list[str]) -> None:
    if not _WEIGHTS_PATH.exists():
        st.warning("No weights data — run the pipeline first.")
        return

    df = pd.read_parquet(_WEIGHTS_PATH)
    df["date"] = pd.to_datetime(df["date"])

    methods_available = [m for m in selected if m in df["method"].unique()]
    if not methods_available:
        st.info("Select at least one method.")
        return

    method = st.selectbox(
        "Strategy",
        options=methods_available,
        format_func=lambda x: _METHOD_LABELS.get(x, x),
        label_visibility="collapsed",
    )

    sub = df[df["method"] == method].sort_values("date")
    ticker_cols = [c for c in _TICKERS if c in sub.columns]
    pivot = sub.set_index("date")[ticker_cols].T
    pivot.columns = [d.strftime("%b %y") for d in pivot.columns]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values * 100,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[
                [0.00, "#080C14"],
                [0.10, "#0A1628"],
                [0.40, "#003D66"],
                [0.70, "#0077CC"],
                [1.00, "#00D4FF"],
            ],
            zmin=0,
            zmax=20,
            colorbar=dict(
                title=dict(text="Weight %", font=dict(size=10, color="#6B7280")),
                ticksuffix="%",
                tickfont=dict(size=9, color="#6B7280"),
                thickness=10,
                len=0.8,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#1E2738",
            ),
            hovertemplate="<b>%{y}</b> — %{x}<br>Weight: <b>%{z:.1f}%</b><extra></extra>",
            xgap=1,
            ygap=1,
        )
    )

    # Min/max weight lines
    fig.add_hline(y=-0.5, line_color="#1E2738", line_width=1)

    fig.update_layout(
        **_dark_layout(height=360),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=9),
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
        ),
        margin=dict(t=10, b=10, l=10, r=60),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<p style='color:#374151;font-size:0.72rem;margin-top:-8px'>"
        "Constraints: min 2% · max 20% per asset. "
        "Darker = underweight, cyan = near max. Monthly rebalance with expanding-window optimisation."
        "</p>",
        unsafe_allow_html=True,
    )


def _dark_layout(height: int = 380) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        font=dict(color="#9CA3AF", family="Inter, sans-serif"),
        height=height,
    )
