"""Performance comparison table — real green/red per-column highlighting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_PERF_PATH = Path("data/results/performance_summary.parquet")

_COLS = {
    "method":           "Strategy",
    "annualized_return":"Ann. Return",
    "annualized_vol":   "Ann. Vol",
    "sharpe_ratio":     "Sharpe",
    "sortino_ratio":    "Sortino",
    "max_drawdown":     "Max DD",
    "cvar_95":          "CVaR 95%",
    "calmar_ratio":     "Calmar",
}
# Lower raw value = better for these columns
_LOWER_BETTER = {"annualized_vol", "max_drawdown", "cvar_95"}

_METHOD_COLORS = {
    "mean_variance":    "#00D4FF",
    "risk_parity":      "#FF6B6B",
    "cvar":             "#FFD93D",
    "black_litterman":  "#6BCB77",
    "equal_weight":     "#4B5563",
}
_METHOD_LABELS = {
    "mean_variance":    "Mean-Variance",
    "risk_parity":      "Risk Parity",
    "cvar":             "CVaR",
    "black_litterman":  "Black-Litterman",
    "equal_weight":     "Equal Weight ⊘",
}

_COL_TOOLTIPS = {
    "Ann. Return":  "Compound annual growth rate of net returns",
    "Ann. Vol":     "Standard deviation of monthly returns × √12",
    "Sharpe":       "(Return − 5% Rf) / Volatility",
    "Sortino":      "(Return − 5% Rf) / Downside volatility",
    "Max DD":       "Worst peak-to-trough decline",
    "CVaR 95%":     "Average loss in the worst 5% of months",
    "Calmar":       "Annual return / |Max drawdown|",
}


def render(selected: list[str]) -> None:
    if not _PERF_PATH.exists():
        st.warning("No performance data — run the pipeline first.")
        return

    df = pd.read_parquet(_PERF_PATH)
    df = df[df["method"].isin(selected + ["equal_weight"])].copy()
    if df.empty:
        return

    num_cols = [k for k in _COLS if k != "method"]

    # ── Build HTML table manually for full styling control ─────────────────
    thead = _build_header()
    tbody = _build_body(df, num_cols)

    table_html = f"""
    <div style="overflow-x:auto;border-radius:10px;border:1px solid #1E2738">
    <table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-family:Inter,sans-serif">
      {thead}
      {tbody}
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Legend ─────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='display:flex;gap:16px;margin-top:8px'>"
        "<span style='font-size:0.7rem;color:#6B7280'>🟢 Best in column</span>"
        "<span style='font-size:0.7rem;color:#6B7280'>🔴 Worst in column</span>"
        "<span style='font-size:0.7rem;color:#6B7280'>↓ lower is better for Vol, MDD, CVaR</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _build_header() -> str:
    cells = "<th style='background:#0D1220;color:#4B5563;font-size:0.7rem;text-transform:uppercase;"
    cells += "letter-spacing:0.06em;padding:12px 14px;text-align:left;border-bottom:1px solid #1E2738'>Strategy</th>"

    for col_key, col_label in _COLS.items():
        if col_key == "method":
            continue
        tip   = _COL_TOOLTIPS.get(col_label, "")
        arrow = " ↓" if col_key in _LOWER_BETTER else ""
        cells += (
            f"<th title='{tip}' style='background:#0D1220;color:#4B5563;font-size:0.7rem;"
            f"text-transform:uppercase;letter-spacing:0.06em;padding:12px 14px;"
            f"text-align:right;border-bottom:1px solid #1E2738;cursor:help'>{col_label}{arrow}</th>"
        )
    return f"<thead><tr>{cells}</tr></thead>"


def _build_body(df: pd.DataFrame, num_cols: list) -> str:
    # Pre-compute best/worst per numeric column
    ranks: dict[str, dict] = {}
    for col in num_cols:
        vals = df[col].values.astype(float)
        lower_better = col in _LOWER_BETTER
        best_idx  = int(np.argmin(vals)) if lower_better else int(np.argmax(vals))
        worst_idx = int(np.argmax(vals)) if lower_better else int(np.argmin(vals))
        ranks[col] = {"best": best_idx, "worst": worst_idx}

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        method  = row["method"]
        color   = _METHOD_COLORS.get(method, "#4B5563")
        label   = _METHOD_LABELS.get(method, method.replace("_", " ").title())
        is_bench = method == "equal_weight"
        row_bg   = "#080C14" if i % 2 == 0 else "#0B0F1A"

        # Method cell with colour dot
        method_cell = (
            f"<td style='padding:11px 14px;border-bottom:1px solid #0D1220;background:{row_bg}'>"
            f"  <div style='display:flex;align-items:center;gap:8px'>"
            f"    <div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0'></div>"
            f"    <span style='color:{'#6B7280' if is_bench else '#D1D5DB'};font-weight:{'400' if is_bench else '500'}'>{label}</span>"
            f"  </div>"
            f"</td>"
        )

        value_cells = ""
        for j, col in enumerate(num_cols):
            raw_val = float(row[col])
            r       = ranks[col]
            is_best  = (j == r["best"] or i == r["best"]) and (r["best"] == i)
            is_worst = (i == r["worst"])

            if is_best:
                val_color = "#10B981"
                bg_extra  = "background:rgba(16,185,129,0.06);"
                badge     = "<span style='font-size:0.6rem;margin-left:4px;color:#10B981'>▲</span>"
            elif is_worst:
                val_color = "#EF4444"
                bg_extra  = "background:rgba(239,68,68,0.06);"
                badge     = "<span style='font-size:0.6rem;margin-left:4px;color:#EF4444'>▼</span>"
            else:
                val_color = "#9CA3AF" if is_bench else "#D1D5DB"
                bg_extra  = ""
                badge     = ""

            fmt = _fmt(col, raw_val)

            value_cells += (
                f"<td style='padding:11px 14px;text-align:right;border-bottom:1px solid #0D1220;"
                f"background:{row_bg};{bg_extra}'>"
                f"  <span style='color:{val_color};font-weight:600;font-variant-numeric:tabular-nums'>{fmt}</span>"
                f"  {badge}"
                f"</td>"
            )

        rows_html += f"<tr>{method_cell}{value_cells}</tr>"

    return f"<tbody>{rows_html}</tbody>"


def _fmt(col: str, val: float) -> str:
    pct_cols = {"annualized_return", "annualized_vol", "max_drawdown", "cvar_95"}
    if col in pct_cols:
        return f"{val:.2%}"
    return f"{val:.2f}"
