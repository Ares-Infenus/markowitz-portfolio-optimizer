"""Portfolio allocation chart — premium version with sector bands."""

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
_METHOD_LABELS = {
    "mean_variance": "Mean-Variance",
    "risk_parity": "Risk Parity",
    "cvar": "CVaR",
    "black_litterman": "Black-Litterman",
}

_SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "GS": "Financials",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "XOM": "Energy",
    "NEE": "Utilities",
    "AMZN": "Consumer Disc.",
    "PG": "Consumer Staples",
}
_SECTOR_COLORS = {
    "Technology": "#1E3A5F",
    "Financials": "#2D1B4E",
    "Healthcare": "#1A3A2A",
    "Energy": "#3A2A10",
    "Utilities": "#1A2A3A",
    "Consumer Disc.": "#3A1A2A",
    "Consumer Staples": "#2A3A1A",
}


def render(selected: list[str]) -> None:
    if not _WEIGHTS_PATH.exists():
        st.warning("No weights data — run the pipeline first.")
        return

    df = pd.read_parquet(_WEIGHTS_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # ── Controls row ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        available_dates = sorted(df["date"].unique())
        selected_date = st.select_slider(
            "Rebalance date",
            options=available_dates,
            value=available_dates[-1],
            format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),
            label_visibility="collapsed",
        )
    with c2:
        view = st.selectbox(
            "View",
            ["Grouped bars", "Stacked bars", "Donut charts"],
            label_visibility="collapsed",
        )
    with c3:
        st.markdown(
            f"<p style='color:#4B5563;font-size:0.75rem;padding-top:8px'>"
            f"📅 {pd.Timestamp(selected_date).strftime('%B %Y')}</p>",
            unsafe_allow_html=True,
        )

    snapshot = df[df["date"] == selected_date]
    ticker_cols = [c for c in _TICKERS if c in snapshot.columns]
    methods = [m for m in selected if m in snapshot["method"].values]

    if not methods:
        st.info("No data for selected methods at this date.")
        return

    if view == "Donut charts":
        _render_donuts(snapshot, methods, ticker_cols)
    else:
        _render_bars(snapshot, methods, ticker_cols, stacked=(view == "Stacked bars"))

    # ── Weights table ─────────────────────────────────────────────────────
    with st.expander("Exact weights"):
        rows = []
        for m in methods:
            sub = snapshot[snapshot["method"] == m]
            if sub.empty:
                continue
            w = sub[ticker_cols].iloc[0]
            rows.append(
                {
                    "Strategy": _METHOD_LABELS.get(m, m),
                    **{t: f"{v:.1%}" for t, v in w.items()},
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)


def _render_bars(snapshot, methods, ticker_cols, stacked: bool) -> None:
    fig = go.Figure()

    for method in methods:
        row = snapshot[snapshot["method"] == method][ticker_cols]
        if row.empty:
            continue
        pct = row.iloc[0] * 100
        color = _METHOD_COLORS.get(method, "#AAA")
        label = _METHOD_LABELS.get(method, method)

        fig.add_trace(
            go.Bar(
                name=label,
                x=ticker_cols,
                y=pct.values,
                marker=dict(
                    color=color,
                    opacity=0.85,
                    line=dict(color="#080C14", width=1),
                ),
                text=[f"{v:.1f}%" for v in pct.values],
                textposition="outside" if not stacked else "inside",
                textfont=dict(size=9, color="#9CA3AF"),
                hovertemplate=f"<b>{label}</b><br>%{{x}} — <b>%{{y:.1f}}%</b><extra></extra>",
            )
        )

    # Equal-weight guideline
    eq_w = 100.0 / len(ticker_cols)
    fig.add_hline(
        y=eq_w,
        line_dash="dot",
        line_color="#2D3748",
        line_width=1.5,
        annotation_text=f"EW {eq_w:.0f}%",
        annotation_font=dict(size=9, color="#4B5563"),
        annotation_position="top right",
    )

    # Sector shading
    _add_sector_bands(fig, ticker_cols)

    fig.update_layout(
        **_dark_layout(height=440),
        barmode="stack" if stacked else "group",
        xaxis=dict(title="", gridcolor="#0D1220"),
        yaxis=dict(
            title="Weight (%)",
            ticksuffix="%",
            range=[0, 26] if not stacked else [0, 102],
            gridcolor="#0D1220",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=1.06,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        margin=dict(t=40, b=30, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_donuts(snapshot, methods, ticker_cols) -> None:
    cols = st.columns(len(methods))
    ticker_colors = [
        "#00D4FF",
        "#0099BB",
        "#FF6B6B",
        "#CC4444",
        "#6BCB77",
        "#4A9955",
        "#FFD93D",
        "#888888",
        "#FF8C00",
        "#BB6600",
    ]
    for col, method in zip(cols, methods):
        row = snapshot[snapshot["method"] == method][ticker_cols]
        if row.empty:
            continue
        weights = row.iloc[0] * 100
        label = _METHOD_LABELS.get(method, method)
        color = _METHOD_COLORS.get(method, "#AAA")

        fig = go.Figure(
            go.Pie(
                labels=ticker_cols,
                values=weights.values,
                hole=0.52,
                texttemplate="%{label}<br><b>%{value:.1f}%</b>",
                textposition="outside",
                textfont=dict(size=9),
                marker=dict(colors=ticker_colors, line=dict(color="#080C14", width=2)),
                hovertemplate="%{label}: <b>%{value:.1f}%</b><extra></extra>",
            )
        )
        fig.add_annotation(
            text=f"<b>{label}</b>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=11, color=color),
        )
        fig.update_layout(
            paper_bgcolor="#080C14",
            plot_bgcolor="#080C14",
            font=dict(color="#9CA3AF", family="Inter,sans-serif"),
            showlegend=False,
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        col.plotly_chart(fig, use_container_width=True)


def _add_sector_bands(fig, ticker_cols) -> None:
    sectors_seen = []
    for t in ticker_cols:
        s = _SECTOR_MAP.get(t, "")
        if s not in sectors_seen:
            sectors_seen.append(s)

    for sector in sectors_seen:
        tickers_in = [t for t in ticker_cols if _SECTOR_MAP.get(t) == sector]
        if not tickers_in:
            continue
        x0 = ticker_cols.index(tickers_in[0]) - 0.5
        x1 = ticker_cols.index(tickers_in[-1]) + 0.5
        bg = _SECTOR_COLORS.get(sector, "#111827")
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=bg,
            opacity=0.4,
            line_width=0,
            layer="below",
            annotation_text=sector,
            annotation_position="bottom",
            annotation_font=dict(size=8, color="#374151"),
        )


def _dark_layout(height: int = 460) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        font=dict(color="#9CA3AF", family="Inter, sans-serif"),
        height=height,
    )
