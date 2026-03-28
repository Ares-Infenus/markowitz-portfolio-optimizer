"""Efficient Frontier scatter — coloured by Sharpe, star markers for key portfolios."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_COLORS = {
    "mean_variance": "#00D4FF",
    "risk_parity": "#FF6B6B",
    "cvar": "#FFD93D",
    "black_litterman": "#6BCB77",
    "Min Variance": "#FFFFFF",
    "Max Sharpe": "#FF8C00",
}

_FRONTIER_PATH = Path("data/results/efficient_frontier.parquet")
_SPECIALS_PATH = Path("data/results/frontier_specials.parquet")
_PERF_PATH = Path("data/results/performance_summary.parquet")


def render(selected_methods: list[str]) -> None:
    st.subheader("Efficient Frontier")

    if not _FRONTIER_PATH.exists():
        st.warning("Run the pipeline first to generate efficient frontier data.")
        return

    frontier = pd.read_parquet(_FRONTIER_PATH)
    perf = pd.read_parquet(_PERF_PATH) if _PERF_PATH.exists() else pd.DataFrame()

    fig = go.Figure()

    # ── Frontier curve ────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=frontier["annualized_vol"],
            y=frontier["annualized_return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#444", width=1.5, dash="dash"),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    # ── Frontier scatter coloured by Sharpe ───────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=frontier["annualized_vol"],
            y=frontier["annualized_return"],
            mode="markers",
            name="λ sweep",
            marker=dict(
                color=frontier["sharpe"],
                colorscale="Viridis",
                size=5,
                colorbar=dict(title="Sharpe", x=1.02),
                showscale=True,
            ),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        )
    )

    # ── Special portfolios from frontier_specials.parquet ─────────────────
    if _SPECIALS_PATH.exists():
        specials = pd.read_parquet(_SPECIALS_PATH)
        for _, row in specials.iterrows():
            label = row["label"]
            fig.add_trace(
                go.Scatter(
                    x=[row["annualized_vol"]],
                    y=[row["annualized_return"]],
                    mode="markers",
                    name=label,
                    marker=dict(
                        symbol="star",
                        size=16,
                        color=_COLORS.get(label, "#FFFFFF"),
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate=f"{label}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {row['sharpe']:.2f}<extra></extra>",
                )
            )

    # ── Optimized method portfolios ───────────────────────────────────────
    if not perf.empty:
        for _, row in perf.iterrows():
            method = row["method"]
            if method == "equal_weight" or method not in selected_methods:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[row["annualized_vol"]],
                    y=[row["annualized_return"]],
                    mode="markers",
                    name=method.replace("_", " ").title(),
                    marker=dict(
                        symbol="star",
                        size=18,
                        color=_COLORS.get(method, "#AAAAAA"),
                        line=dict(color="white", width=1.5),
                    ),
                    hovertemplate=f"{method}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {row['sharpe_ratio']:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(tickformat=".1%"),
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
