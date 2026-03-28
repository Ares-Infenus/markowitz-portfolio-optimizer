"""Tests for PerformanceAnalytics metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import PerformanceAnalytics


def _make_result(monthly_returns: list[float]) -> pd.DataFrame:
    """Build a minimal backtest result DataFrame."""
    dates = pd.date_range("2023-02-01", periods=len(monthly_returns), freq="MS")
    portfolio_values = np.cumprod([1.0] + [1 + r for r in monthly_returns])[1:]
    return pd.DataFrame(
        {
            "method": "test",
            "portfolio_value": portfolio_values,
            "monthly_return": monthly_returns,
            "net_monthly_return": monthly_returns,
            "turnover": 0.0,
            "tx_cost_applied": 0.0,
        },
        index=dates,
    )


@pytest.fixture()
def analytics(settings):
    profitable = _make_result([0.02] * 24)  # 2% per month = ~26% annual
    return PerformanceAnalytics({"test": profitable}, settings)


def test_sharpe_positive_when_return_exceeds_rfr(analytics):
    """Sharpe must be > 0 when annualised return > risk-free rate."""
    summary = analytics.compute()
    row = summary[summary["method"] == "test"].iloc[0]
    assert row["annualized_return"] > analytics.settings.risk_free_rate
    assert row["sharpe_ratio"] > 0


def test_max_drawdown_nonpositive(settings):
    """Max drawdown should always be ≤ 0."""
    # Mix of gains and a drawdown
    monthly = [0.05, -0.10, 0.02, 0.03, -0.05, 0.01] * 4
    analytics = PerformanceAnalytics({"test": _make_result(monthly)}, settings)
    summary = analytics.compute()
    assert summary["max_drawdown"].iloc[0] <= 0


def test_cvar_worse_than_worst_single_loss(settings):
    """CVaR(95%) should be ≤ the worst individual monthly loss."""
    monthly = [0.01, -0.15, 0.02, -0.20, 0.01, 0.03] * 4
    analytics = PerformanceAnalytics({"test": _make_result(monthly)}, settings)
    summary = analytics.compute()
    worst_loss = min(monthly)
    cvar = summary["cvar_95"].iloc[0]
    assert cvar <= worst_loss + 1e-6, f"CVaR {cvar:.4f} > worst single loss {worst_loss:.4f}"


def test_calmar_ratio_sign(settings):
    """Calmar ratio should be positive when annual return > 0 and drawdown < 0."""
    monthly = [0.02] * 24
    analytics = PerformanceAnalytics({"test": _make_result(monthly)}, settings)
    summary = analytics.compute()
    assert summary["calmar_ratio"].iloc[0] > 0


def test_benchmark_equal_weight_shape(synthetic_returns):
    """benchmark_equal_weight should return a DataFrame with portfolio_value column."""
    bench = PerformanceAnalytics.benchmark_equal_weight(synthetic_returns)
    assert "portfolio_value" in bench.columns
    assert "net_monthly_return" in bench.columns
    assert len(bench) >= 1


def test_sortino_ratio_greater_than_sharpe_for_positive_skew(settings):
    """When losses are small, Sortino > Sharpe (downside vol < total vol)."""
    monthly = [0.03, 0.04, 0.02, -0.005, 0.03, 0.04] * 4
    analytics = PerformanceAnalytics({"test": _make_result(monthly)}, settings)
    summary = analytics.compute()
    row = summary.iloc[0]
    assert row["sortino_ratio"] >= row["sharpe_ratio"] - 0.01
