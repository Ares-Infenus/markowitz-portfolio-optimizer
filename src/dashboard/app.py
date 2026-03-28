"""Streamlit dashboard — dark theme, Plotly interactive charts."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Markowitz Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for dark professional theme ────────────────────────────────────
st.markdown(
    """
    <style>
    body, .stApp { background-color: #0E1117; color: #CCCCCC; }
    .stApp header { background-color: #0E1117; }
    section[data-testid="stSidebar"] { background-color: #161B27; }
    h1, h2, h3 { color: #00D4FF; }
    .metric-label { color: #888; }
    .stSelectbox label, .stMultiSelect label { color: #AAAAAA; }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.dashboard.components import (  # noqa: E402
    allocation_chart,
    backtest_chart,
    efficient_frontier,
    performance_table,
    weights_heatmap,
)

_METHODS = ["mean_variance", "risk_parity", "cvar", "black_litterman"]
_PERF_PATH = Path("data/results/performance_summary.parquet")
_BACKTEST_PATH = Path("data/results/backtest_results.parquet")


def main() -> None:
    t_start = time.perf_counter()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown("---")
        st.markdown("**Select methods to display:**")

        selected = []
        for method in _METHODS:
            label = method.replace("_", " ").title()
            if st.checkbox(label, value=True, key=f"cb_{method}"):
                selected.append(method)

        st.markdown("---")
        st.markdown("**Benchmark:** Equal Weight (always shown)")
        st.markdown("---")
        st.caption(f"Dashboard v1.0 | Python 3.11")

    # ── Header ────────────────────────────────────────────────────────────
    st.title("📈 Markowitz Portfolio Optimizer")
    st.markdown(
        "**4 quantitative methods** | **6-year backtest** (2019–2024) | "
        "Ledoit-Wolf covariance | 0.1% transaction costs"
    )

    if not _BACKTEST_PATH.exists():
        st.error(
            "No backtest data found. Run `python src/pipeline/run_pipeline.py` first."
        )
        return

    # ── KPI strip ─────────────────────────────────────────────────────────
    _render_kpis(selected)

    st.markdown("---")

    # ── Row 1: Efficient Frontier + Cumulative Returns ────────────────────
    col1, col2 = st.columns(2)
    with col1:
        efficient_frontier.render(selected)
    with col2:
        backtest_chart.render(selected + ["equal_weight"])

    st.markdown("---")

    # ── Row 2: Allocation breakdown ───────────────────────────────────────
    allocation_chart.render(selected)

    st.markdown("---")

    # ── Row 3: Weights Heatmap + Performance Table ────────────────────────
    col3, col4 = st.columns([1, 1])
    with col3:
        weights_heatmap.render(selected)
    with col4:
        performance_table.render(selected)

    # ── Load time ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    st.caption(f"Page rendered in {elapsed:.2f}s")


def _render_kpis(selected: list[str]) -> None:
    if not _PERF_PATH.exists():
        return

    import pandas as pd

    perf = pd.read_parquet(_PERF_PATH)
    perf = perf[perf["method"].isin(selected)]

    if perf.empty:
        return

    best_sharpe_row = perf.loc[perf["sharpe_ratio"].idxmax()]
    best_return_row = perf.loc[perf["annualized_return"].idxmax()]
    best_dd_row = perf.loc[perf["max_drawdown"].idxmax()]  # least negative

    cols = st.columns(4)
    cols[0].metric(
        "Best Sharpe",
        f"{best_sharpe_row['sharpe_ratio']:.2f}",
        best_sharpe_row["method"].replace("_", " ").title(),
    )
    cols[1].metric(
        "Best Ann. Return",
        f"{best_return_row['annualized_return']:.1%}",
        best_return_row["method"].replace("_", " ").title(),
    )
    cols[2].metric(
        "Min Max Drawdown",
        f"{best_dd_row['max_drawdown']:.1%}",
        best_dd_row["method"].replace("_", " ").title(),
    )
    cols[3].metric(
        "Methods Active",
        str(len(selected)),
        f"of {len(_METHODS)} total",
    )


if __name__ == "__main__":
    main()
