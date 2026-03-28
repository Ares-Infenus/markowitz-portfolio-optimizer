"""Styled performance comparison table — green = best, red = worst per column."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_PERF_PATH = Path("data/results/performance_summary.parquet")

_DISPLAY_COLS = {
    "method": "Method",
    "annualized_return": "Ann. Return",
    "annualized_vol": "Ann. Vol",
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "max_drawdown": "Max DD",
    "cvar_95": "CVaR 95%",
    "calmar_ratio": "Calmar",
}

# For these columns, lower is better
_LOWER_BETTER = {"annualized_vol", "max_drawdown", "cvar_95"}


def render(selected_methods: list[str]) -> None:
    st.subheader("Performance Comparison")

    if not _PERF_PATH.exists():
        st.warning("Run the pipeline first to generate performance data.")
        return

    df = pd.read_parquet(_PERF_PATH)
    mask = df["method"].isin(selected_methods + ["equal_weight"])
    df = df[mask].copy()

    if df.empty:
        st.info("No data for selected methods.")
        return

    df = df.rename(columns=_DISPLAY_COLS)
    df["Method"] = df["Method"].str.replace("_", " ").str.title()

    numeric_cols = [v for k, v in _DISPLAY_COLS.items() if k != "method"]

    # Format percentages
    pct_cols = {"Ann. Return", "Ann. Vol", "Max DD", "CVaR 95%"}
    for col in numeric_cols:
        if col in pct_cols:
            df[col] = df[col].map("{:.2%}".format)
        else:
            df[col] = df[col].map("{:.2f}".format)

    styled = df.style.set_properties(**{"text-align": "center"})
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1A1F2C"),
                    ("color", "#00D4FF"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border-bottom", "1px solid #333"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#0E1117"),
                    ("color", "#CCCCCC"),
                    ("border-bottom", "1px solid #1E2530"),
                ],
            },
        ]
    )

    st.write(styled.to_html(index=False), unsafe_allow_html=True)

    st.caption(
        "Green = best in column | Red = worst in column | "
        "CVaR and Max DD: lower magnitude is better"
    )
