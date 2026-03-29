"""Risk Profile Radar — 5-dimension spider chart comparing all strategies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PERF_PATH = Path("data/results/performance_summary.parquet")

_METHOD_COLORS = {
    "mean_variance": "#00D4FF",
    "risk_parity": "#FF6B6B",
    "cvar": "#FFD93D",
    "black_litterman": "#6BCB77",
    "equal_weight": "#4B5563",
}
_METHOD_LABELS = {
    "mean_variance": "Mean-Variance",
    "risk_parity": "Risk Parity",
    "cvar": "CVaR",
    "black_litterman": "Black-Litterman",
    "equal_weight": "Equal Weight",
}

# (column, label, higher_is_better)
_DIMENSIONS = [
    ("sharpe_ratio", "Sharpe", True),
    ("sortino_ratio", "Sortino", True),
    ("calmar_ratio", "Calmar", True),
    ("max_drawdown", "Low Drawdown", False),  # inverted
    ("cvar_95", "Low Tail Risk", False),  # inverted
]


def render(selected: list[str]) -> None:
    if not _PERF_PATH.exists():
        st.warning("No performance data — run the pipeline first.")
        return

    perf = pd.read_parquet(_PERF_PATH)
    methods_to_show = [m for m in selected if m in perf["method"].values]
    if not methods_to_show:
        st.info("Select at least one method.")
        return

    # ── Normalise each dimension to [0, 1] across all methods ─────────────
    all_methods = perf["method"].values
    scores: dict[str, list[float]] = {}
    for col, _, higher_better in _DIMENSIONS:
        vals = perf.set_index("method")[col]
        mn, mx = vals.min(), vals.max()
        for m in all_methods:
            v = float(vals[m]) if m in vals.index else 0.0
            norm = (v - mn) / (mx - mn) if mx > mn else 0.5
            if not higher_better:
                norm = 1.0 - norm  # invert: lower raw = higher score
            scores.setdefault(m, []).append(round(norm, 4))

    fig = go.Figure()
    dim_labels = [d[1] for d in _DIMENSIONS]
    # Close the polygon
    categories = dim_labels + [dim_labels[0]]

    for method in methods_to_show:
        vals = scores.get(method, [0] * len(_DIMENSIONS))
        vals_closed = vals + [vals[0]]
        color = _METHOD_COLORS.get(method, "#888")
        label = _METHOD_LABELS.get(method, method)

        # Raw values for hover
        raw = []
        for col, _, _ in _DIMENSIONS:
            v = perf[perf["method"] == method][col].values
            raw.append(float(v[0]) if len(v) else 0.0)
        raw_closed = raw + [raw[0]]

        customdata = [[r] for r in raw_closed]

        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=categories,
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(color=color, width=2),
                name=label,
                customdata=customdata,
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<br>Raw: %{customdata[0]:.3f}<extra>"
                + label
                + "</extra>",
            )
        )

    # ── Equal-weight reference ─────────────────────────────────────────────
    if "equal_weight" in perf["method"].values and "equal_weight" not in methods_to_show:
        ew_vals = scores.get("equal_weight", [0.5] * len(_DIMENSIONS))
        ew_closed = ew_vals + [ew_vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=ew_closed,
                theta=categories,
                mode="lines",
                line=dict(color="#1E2738", width=1.5, dash="dot"),
                name="Equal Weight ⊘",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#080C14",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
                tickfont=dict(size=8, color="#374151"),
                gridcolor="#1E2738",
                linecolor="#1E2738",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#9CA3AF"),
                gridcolor="#1E2738",
                linecolor="#1E2738",
            ),
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        font=dict(color="#9CA3AF", family="Inter, sans-serif"),
        legend=dict(
            x=0.5,
            y=-0.12,
            xanchor="center",
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        height=420,
        margin=dict(t=20, b=50, l=40, r=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Concept note ───────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#374151;font-size:0.72rem;margin-top:-4px'>"
        "Scores normalised 0→1 across all strategies. "
        "Drawdown & Tail Risk are inverted (larger area = better overall profile). "
        "Dashed ring = equal-weight benchmark."
        "</p>",
        unsafe_allow_html=True,
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
