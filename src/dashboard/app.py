"""Streamlit dashboard — premium dark theme, tabbed layout."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Markowitz Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #080C14; color: #D1D5DB; }
.stApp header { background-color: #080C14 !important; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1220 0%, #080C14 100%);
    border-right: 1px solid #1E2738;
}

/* ── Typography ── */
h1 { background: linear-gradient(135deg, #00D4FF 0%, #7C3AED 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3 { color: #E5E7EB !important; font-weight: 600 !important; }
p, li { color: #9CA3AF; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #0D1525 100%);
    border: 1px solid #1E2E45;
    border-radius: 12px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00D4FF, #7C3AED);
}
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #F9FAFB; line-height: 1.1; }
.kpi-label { font-size: 0.72rem; color: #6B7280; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-sub { font-size: 0.78rem; color: #00D4FF; margin-top: 4px; font-weight: 500; }
.kpi-pos { color: #10B981; }
.kpi-neg { color: #EF4444; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1220; border-bottom: 1px solid #1E2738; gap: 4px; padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #6B7280; font-weight: 500; font-size: 0.85rem;
    padding: 10px 20px; border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #1E2E45 0%, #0D1525 100%) !important;
    color: #00D4FF !important; border-bottom: 2px solid #00D4FF !important;
}

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 8px 0 18px 0; padding-bottom: 10px;
    border-bottom: 1px solid #1E2738;
}
.section-title { color: #E5E7EB; font-size: 0.95rem; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.05em; }
.section-badge {
    background: #1E2E45; color: #00D4FF; font-size: 0.68rem;
    padding: 2px 8px; border-radius: 20px; font-weight: 600;
}

/* ── Sidebar method pills ── */
.method-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-bottom: 4px;
}

/* ── Dividers ── */
hr { border-color: #1E2738 !important; margin: 24px 0 !important; }

/* ── Metric overrides ── */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #F9FAFB !important; }
[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: #111827 !important; border-radius: 8px !important;
                            color: #9CA3AF !important; }

/* ── Radio ── */
.stRadio [data-testid="stMarkdownContainer"] p { color: #9CA3AF; font-size: 0.82rem; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Selectbox / slider ── */
.stSelectbox label, .stSlider label { color: #9CA3AF !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

from src.dashboard.components import (  # noqa: E402
    allocation_chart,
    backtest_chart,
    efficient_frontier,
    performance_table,
    radar_chart,
    weights_heatmap,
)

_METHODS = ["mean_variance", "risk_parity", "cvar", "black_litterman"]
_METHOD_LABELS = {
    "mean_variance":    "Mean-Variance",
    "risk_parity":      "Risk Parity",
    "cvar":             "CVaR",
    "black_litterman":  "Black-Litterman",
}
_METHOD_COLORS = {
    "mean_variance":    "#00D4FF",
    "risk_parity":      "#FF6B6B",
    "cvar":             "#FFD93D",
    "black_litterman":  "#6BCB77",
}
_PERF_PATH  = Path("data/results/performance_summary.parquet")
_BT_PATH    = Path("data/results/backtest_results.parquet")


def main() -> None:
    t0 = time.perf_counter()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Portfolio Methods")
        st.markdown("<p style='font-size:0.75rem;color:#6B7280;margin-bottom:12px'>Toggle which strategies appear in all charts</p>", unsafe_allow_html=True)

        selected = []
        for m in _METHODS:
            col_dot, col_cb = st.columns([0.15, 0.85])
            with col_dot:
                st.markdown(
                    f"<div style='width:10px;height:10px;border-radius:50%;background:{_METHOD_COLORS[m]};margin-top:10px'></div>",
                    unsafe_allow_html=True,
                )
            with col_cb:
                if st.checkbox(_METHOD_LABELS[m], value=True, key=f"cb_{m}"):
                    selected.append(m)

        st.markdown("---")
        st.markdown(
            "<div style='background:#111827;border:1px solid #1E2738;border-radius:8px;padding:12px'>"
            "<p style='color:#6B7280;font-size:0.72rem;margin:0 0 6px 0;text-transform:uppercase;letter-spacing:0.05em'>Benchmark</p>"
            "<p style='color:#888;font-size:0.82rem;margin:0'>⬜ Equal Weight <span style='color:#555'>(10% each)</span></p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Concept legend
        st.markdown("<p style='color:#6B7280;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em'>Method summary</p>", unsafe_allow_html=True)
        _sidebar_legend()

    # ── Header ────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<h1>Markowitz Portfolio Optimizer</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#6B7280;margin-top:-8px;font-size:0.9rem'>"
            "4 quantitative strategies &nbsp;·&nbsp; 6-year backtest 2019–2024 &nbsp;·&nbsp; "
            "Ledoit-Wolf covariance &nbsp;·&nbsp; 10 assets / 5 sectors &nbsp;·&nbsp; 0.1% tx costs"
            "</p>",
            unsafe_allow_html=True,
        )
    with c2:
        if _BT_PATH.exists():
            elapsed_run = _pipeline_age()
            st.markdown(
                f"<div style='text-align:right;padding-top:12px'>"
                f"<p style='color:#374151;font-size:0.72rem;margin:0'>LAST RUN</p>"
                f"<p style='color:#4B5563;font-size:0.82rem;margin:0'>{elapsed_run}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if not _BT_PATH.exists():
        st.markdown(
            "<div style='background:#1A0A0A;border:1px solid #7F1D1D;border-radius:10px;"
            "padding:20px;margin-top:20px'>"
            "<p style='color:#FCA5A5;margin:0'>⚠️ No backtest data found. "
            "Run <code>python src/pipeline/run_pipeline.py</code> first.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── KPI strip ─────────────────────────────────────────────────────────
    _render_kpis(selected)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊  Performance & Returns",
        "⚖️  Risk Analysis",
        "🥧  Portfolio Composition",
    ])

    with tab1:
        _section("Cumulative Returns", "vs equal-weight benchmark · shaded: COVID crash & 2022 bear market")
        backtest_chart.render(selected + ["equal_weight"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _section("Strategy Performance Scorecard", "all metrics annualised · green = best · red = worst")
        performance_table.render(selected)

    with tab2:
        col_ef, col_rd = st.columns([1.1, 0.9])
        with col_ef:
            _section("Efficient Frontier", "mean-variance λ sweep · stars = optimised portfolios")
            efficient_frontier.render(selected)
        with col_rd:
            _section("Risk Profile Radar", "normalised scores across 5 dimensions")
            radar_chart.render(selected)

    with tab3:
        _section("Asset Allocation Breakdown", "portfolio weights at selected rebalance date")
        allocation_chart.render(selected)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _section("Weight Evolution Over Time", "monthly rebalance history — select method")
        weights_heatmap.render(selected)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='color:#374151;font-size:0.72rem;text-align:center'>"
        f"Rendered in {time.perf_counter()-t0:.2f}s &nbsp;·&nbsp; "
        f"Data: 2019-01-01 → 2024-12-31 &nbsp;·&nbsp; "
        f"Rf = 5% &nbsp;·&nbsp; Rebalance: monthly expanding window"
        f"</p>",
        unsafe_allow_html=True,
    )


# ── Helper components ─────────────────────────────────────────────────────────

def _section(title: str, subtitle: str = "") -> None:
    sub_html = f"<span style='color:#4B5563;font-size:0.75rem;font-weight:400'>{subtitle}</span>" if subtitle else ""
    st.markdown(
        f"<div class='section-header'>"
        f"<span class='section-title'>{title}</span>"
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_kpis(selected: list[str]) -> None:
    if not _PERF_PATH.exists() or not selected:
        return

    perf = pd.read_parquet(_PERF_PATH)
    perf = perf[perf["method"].isin(selected)]
    if perf.empty:
        return

    cols = st.columns(len(selected))
    for col, (_, row) in zip(cols, perf.iterrows()):
        method = row["method"]
        color  = _METHOD_COLORS.get(method, "#AAAAAA")
        ret    = row["annualized_return"]
        sharpe = row["sharpe_ratio"]
        mdd    = row["max_drawdown"]
        vol    = row["annualized_vol"]
        ret_cls  = "kpi-pos" if ret > 0 else "kpi-neg"
        mdd_cls  = "kpi-neg"

        col.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>{_METHOD_LABELS.get(method, method)}</div>"
            f"<div class='kpi-value' style='color:{color}'>{ret:.1%}</div>"
            f"<div style='display:flex;gap:16px;margin-top:10px'>"
            f"  <div><span style='color:#4B5563;font-size:0.68rem'>SHARPE</span>"
            f"       <span style='color:#D1D5DB;font-size:0.85rem;font-weight:600;margin-left:4px'>{sharpe:.2f}</span></div>"
            f"  <div><span style='color:#4B5563;font-size:0.68rem'>VOL</span>"
            f"       <span style='color:#D1D5DB;font-size:0.85rem;font-weight:600;margin-left:4px'>{vol:.1%}</span></div>"
            f"  <div><span style='color:#4B5563;font-size:0.68rem'>MDD</span>"
            f"       <span class='{mdd_cls}' style='font-size:0.85rem;font-weight:600;margin-left:4px'>{mdd:.1%}</span></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _sidebar_legend() -> None:
    items = [
        ("MV",  "#00D4FF", "Maximize Sharpe via QP solver"),
        ("RP",  "#FF6B6B", "Equal risk contribution"),
        ("CVaR","#FFD93D", "Minimize tail loss at 95%"),
        ("BL",  "#6BCB77", "Bayesian: prior + momentum views"),
    ]
    for abbr, color, desc in items:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:8px'>"
            f"  <span style='background:{color}22;color:{color};font-size:0.65rem;font-weight:700;"
            f"         padding:2px 6px;border-radius:4px;flex-shrink:0;margin-top:1px'>{abbr}</span>"
            f"  <span style='color:#4B5563;font-size:0.75rem;line-height:1.4'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _pipeline_age() -> str:
    import datetime
    mtime = _BT_PATH.stat().st_mtime
    dt = datetime.datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    main()
