"""Portfolio allocation bar chart — weights % per asset, grouped by method."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_WEIGHTS_PATH = Path("data/results/weights_store.parquet")
_TICKERS = ["AAPL", "MSFT", "JPM", "GS", "JNJ", "UNH", "XOM", "NEE", "AMZN", "PG"]

_METHOD_COLORS = {
    "mean_variance": "#00D4FF",
    "risk_parity": "#FF6B6B",
    "cvar": "#FFD93D",
    "black_litterman": "#6BCB77",
}

_SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology",
    "JPM": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare",
    "XOM": "Energy", "NEE": "Utilities",
    "AMZN": "Consumer Disc.", "PG": "Consumer Staples",
}


def render(selected_methods: list[str]) -> None:
    st.subheader("Portfolio Allocation by Asset")

    if not _WEIGHTS_PATH.exists():
        st.warning("Run the pipeline first to generate weights data.")
        return

    df = pd.read_parquet(_WEIGHTS_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # ── Date selector ─────────────────────────────────────────────────────
    available_dates = sorted(df["date"].unique())
    selected_date = st.select_slider(
        "Rebalance date",
        options=available_dates,
        value=available_dates[-1],
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
    )

    snapshot = df[df["date"] == selected_date]
    ticker_cols = [c for c in _TICKERS if c in snapshot.columns]

    methods_in_data = [m for m in selected_methods if m in snapshot["method"].values]
    if not methods_in_data:
        st.info("No data for selected methods at this date.")
        return

    # ── View toggle ───────────────────────────────────────────────────────
    view = st.radio(
        "View",
        ["Grouped bars", "Stacked bars", "Pie charts"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view == "Pie charts":
        _render_pies(snapshot, methods_in_data, ticker_cols)
    else:
        _render_bars(snapshot, methods_in_data, ticker_cols, stacked=(view == "Stacked bars"))

    # ── Table below chart ─────────────────────────────────────────────────
    with st.expander("Raw weights table"):
        rows = []
        for method in methods_in_data:
            row = snapshot[snapshot["method"] == method][ticker_cols].iloc[0]
            rows.append({"Method": method.replace("_", " ").title(), **{t: f"{v:.1%}" for t, v in row.items()}})
        st.dataframe(pd.DataFrame(rows).set_index("Method"), use_container_width=True)


def _render_bars(snapshot, methods, ticker_cols, stacked: bool) -> None:
    fig = go.Figure()

    for method in methods:
        row = snapshot[snapshot["method"] == method][ticker_cols]
        if row.empty:
            continue
        weights_pct = row.iloc[0] * 100

        fig.add_trace(
            go.Bar(
                name=method.replace("_", " ").title(),
                x=ticker_cols,
                y=weights_pct.values,
                marker_color=_METHOD_COLORS.get(method, "#AAAAAA"),
                text=[f"{v:.1f}%" for v in weights_pct.values],
                textposition="outside" if not stacked else "inside",
                hovertemplate="%{x}<br>Weight: %{y:.1f}%<extra>" + method + "</extra>",
            )
        )

    # Equal weight reference line
    eq_w = 100.0 / len(ticker_cols)
    fig.add_hline(
        y=eq_w,
        line_dash="dot",
        line_color="#666",
        annotation_text=f"Equal weight ({eq_w:.1f}%)",
        annotation_position="top right",
        annotation_font_color="#888",
    )

    fig.update_layout(
        **_dark_layout(),
        barmode="stack" if stacked else "group",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        yaxis=dict(ticksuffix="%", range=[0, 25] if not stacked else [0, 102]),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    # Sector color bands on x-axis
    sectors = list(dict.fromkeys(_SECTOR_MAP[t] for t in ticker_cols))
    sector_colors = ["#1A2030", "#0E1820", "#162030", "#0E1A28", "#141E2C"]
    for i, sector in enumerate(sectors):
        tickers_in_sector = [t for t in ticker_cols if _SECTOR_MAP.get(t) == sector]
        if tickers_in_sector:
            x0 = ticker_cols.index(tickers_in_sector[0]) - 0.5
            x1 = ticker_cols.index(tickers_in_sector[-1]) + 0.5
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=sector_colors[i % len(sector_colors)],
                line_width=0,
                layer="below",
                annotation_text=sector,
                annotation_position="bottom",
                annotation_font=dict(size=9, color="#666"),
            )

    st.plotly_chart(fig, use_container_width=True)


def _render_pies(snapshot, methods, ticker_cols) -> None:
    cols = st.columns(len(methods))
    for col, method in zip(cols, methods):
        row = snapshot[snapshot["method"] == method][ticker_cols]
        if row.empty:
            continue
        weights = row.iloc[0]

        fig = go.Figure(
            go.Pie(
                labels=ticker_cols,
                values=(weights * 100).values,
                hole=0.4,
                texttemplate="%{label}<br>%{value:.1f}%",
                textposition="outside",
                marker=dict(
                    colors=[
                        "#00D4FF", "#0099BB", "#FF6B6B", "#CC4444",
                        "#6BCB77", "#4A9955", "#FFD93D", "#AAAAAA",
                        "#FF8C00", "#BB6600",
                    ]
                ),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(
                text=method.replace("_", " ").title(),
                font=dict(color="#00D4FF", size=13),
                x=0.5,
            ),
            **_dark_layout(),
            height=340,
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        col.plotly_chart(fig, use_container_width=True)


def _dark_layout() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#CCCCCC"),
        height=420,
    )
