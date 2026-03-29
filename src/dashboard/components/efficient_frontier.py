"""Efficient Frontier — annotated, premium version."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_FRONTIER_PATH = Path("data/results/efficient_frontier.parquet")
_SPECIALS_PATH = Path("data/results/frontier_specials.parquet")
_PERF_PATH = Path("data/results/performance_summary.parquet")

_METHOD_COLORS = {
    "mean_variance": "#00D4FF",
    "risk_parity": "#FF6B6B",
    "cvar": "#FFD93D",
    "black_litterman": "#6BCB77",
}
_METHOD_LABELS = {
    "mean_variance": "Mean-Variance",
    "risk_parity": "Risk Parity",
    "cvar": "CVaR",
    "black_litterman": "Black-Litterman",
}


def render(selected: list[str]) -> None:
    if not _FRONTIER_PATH.exists():
        st.warning("No frontier data — run the pipeline first.")
        return

    frontier = pd.read_parquet(_FRONTIER_PATH)
    perf = pd.read_parquet(_PERF_PATH) if _PERF_PATH.exists() else pd.DataFrame()

    fig = go.Figure()

    # ── Gradient frontier cloud (colour = Sharpe) ──────────────────────────
    fig.add_trace(
        go.Scatter(
            x=frontier["annualized_vol"],
            y=frontier["annualized_return"],
            mode="markers",
            name="λ sweep (100 pts)",
            marker=dict(
                color=frontier["sharpe"],
                colorscale=[
                    [0.0, "#1E0A40"],
                    [0.35, "#4B0082"],
                    [0.65, "#0066CC"],
                    [1.0, "#00D4FF"],
                ],
                size=7,
                opacity=0.75,
                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(size=11, color="#6B7280")),
                    tickfont=dict(size=10, color="#6B7280"),
                    thickness=10,
                    len=0.6,
                    x=1.02,
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="#1E2738",
                ),
                showscale=True,
            ),
            hovertemplate="Vol: <b>%{x:.1%}</b><br>Return: <b>%{y:.1%}</b><br>Sharpe: <b>%{marker.color:.2f}</b><extra></extra>",
        )
    )

    # ── Frontier boundary curve ────────────────────────────────────────────
    frontier_s = frontier.sort_values("annualized_vol")
    fig.add_trace(
        go.Scatter(
            x=frontier_s["annualized_vol"],
            y=frontier_s["annualized_return"],
            mode="lines",
            name="Frontier boundary",
            line=dict(color="#1E2E45", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ── Special portfolios (Min Var, Max Sharpe) ───────────────────────────
    if _SPECIALS_PATH.exists():
        specials = pd.read_parquet(_SPECIALS_PATH)
        special_styles = {
            "Min Variance": dict(symbol="diamond", color="#E5E7EB", size=14),
            "Max Sharpe": dict(symbol="star", color="#F59E0B", size=18),
        }
        for _, row in specials.iterrows():
            s = special_styles.get(row["label"], dict(symbol="star", color="#FFF", size=14))
            fig.add_trace(
                go.Scatter(
                    x=[row["annualized_vol"]],
                    y=[row["annualized_return"]],
                    mode="markers+text",
                    name=row["label"],
                    text=[row["label"]],
                    textposition="top center",
                    textfont=dict(size=9, color="#9CA3AF"),
                    marker=dict(
                        symbol=s["symbol"],
                        color=s["color"],
                        size=s["size"],
                        line=dict(color="#080C14", width=1.5),
                    ),
                    hovertemplate=f"<b>{row['label']}</b><br>Vol: {row['annualized_vol']:.1%}<br>Return: {row['annualized_return']:.1%}<br>Sharpe: {row['sharpe']:.2f}<extra></extra>",
                )
            )

    # ── Optimised method portfolios ────────────────────────────────────────
    if not perf.empty:
        for _, row in perf.iterrows():
            method = row["method"]
            if method not in selected or method == "equal_weight":
                continue
            color = _METHOD_COLORS.get(method, "#AAA")
            label = _METHOD_LABELS.get(method, method)

            fig.add_trace(
                go.Scatter(
                    x=[row["annualized_vol"]],
                    y=[row["annualized_return"]],
                    mode="markers+text",
                    name=label,
                    text=[label],
                    textposition="bottom center",
                    textfont=dict(size=9, color=color),
                    marker=dict(
                        symbol="star",
                        color=color,
                        size=20,
                        line=dict(color="#080C14", width=2),
                    ),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Vol: {row['annualized_vol']:.1%}<br>"
                        f"Return: {row['annualized_return']:.1%}<br>"
                        f"Sharpe: {row['sharpe_ratio']:.2f}<extra></extra>"
                    ),
                )
            )

    # ── Capital Market Line annotation ─────────────────────────────────────
    if not perf.empty and not perf[perf["method"].isin(selected)].empty:
        best = perf[perf["method"].isin(selected)].loc[
            perf[perf["method"].isin(selected)]["sharpe_ratio"].idxmax()
        ]
        rf = 0.05
        x_cml = [0, best["annualized_vol"] * 1.3]
        y_cml = [rf, rf + best["sharpe_ratio"] * best["annualized_vol"] * 1.3]
        fig.add_trace(
            go.Scatter(
                x=x_cml,
                y=y_cml,
                mode="lines",
                name="Capital Market Line",
                line=dict(color="#F59E0B", width=1.2, dash="dot"),
                hoverinfo="skip",
            )
        )
        fig.add_annotation(
            x=0,
            y=rf,
            text="Rf = 5%",
            showarrow=False,
            font=dict(size=9, color="#6B7280"),
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        **_dark_layout(height=480),
        xaxis=dict(
            title="Annualised Volatility",
            tickformat=".0%",
            gridcolor="#0D1220",
            zeroline=False,
            title_font_size=11,
        ),
        yaxis=dict(
            title="Annualised Return",
            tickformat=".0%",
            gridcolor="#0D1220",
            zeroline=False,
            title_font_size=11,
        ),
        legend=dict(
            x=0.01,
            y=0.01,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        hovermode="closest",
        margin=dict(t=20, b=10, l=10, r=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Concept note ───────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#374151;font-size:0.72rem;margin-top:-8px'>"
        "★ Each star = a portfolio optimised by one strategy. "
        "Points above the dashed CML offer better risk-adjusted return than the market equilibrium. "
        "Sharpe colour: blue–purple = low, cyan = high."
        "</p>",
        unsafe_allow_html=True,
    )


def _dark_layout(height: int = 500) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        font=dict(color="#9CA3AF", family="Inter, sans-serif"),
        height=height,
    )
