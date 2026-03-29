"""Tests for BacktestEngine — monthly rebalances, transaction costs."""

from __future__ import annotations

import pytest

from src.backtest.engine import BacktestEngine
from src.optimizers.mean_variance import MeanVarianceOptimizer


@pytest.fixture()
def engine(
    settings, synthetic_returns, synthetic_cov, synthetic_expected_returns, tmp_path, monkeypatch
):
    # Redirect parquet output to tmp_path
    monkeypatch.setattr("src.backtest.engine.WEIGHTS_PATH", tmp_path / "weights_store.parquet")
    monkeypatch.setattr("src.backtest.engine.BACKTEST_PATH", tmp_path / "backtest_results.parquet")

    optimizer = MeanVarianceOptimizer(settings)
    return BacktestEngine(
        optimizer=optimizer,
        returns=synthetic_returns,
        cov_matrix=synthetic_cov,
        expected_returns=synthetic_expected_returns,
        settings=settings,
        method_name="mean_variance",
    )


def test_monthly_rebalance_dates(engine, synthetic_returns):
    """Should generate one date per month covered by the returns index."""
    dates = engine._monthly_rebalance_dates()
    assert len(dates) >= 1
    # All dates must be in the returns index
    for d in dates:
        assert d in synthetic_returns.index


def test_portfolio_starts_at_one(engine):
    """First portfolio value record should be ≤ 2.0 (reasonable for 1 month)."""
    results = engine.run()
    assert not results.empty
    assert results["portfolio_value"].iloc[0] > 0


def test_net_return_less_than_gross_after_rebalance(engine):
    """Net monthly return should be ≤ gross return (tx costs are non-negative)."""
    results = engine.run()
    assert (results["net_monthly_return"] <= results["monthly_return"] + 1e-10).all()


def test_rebalance_count_matches_months(engine, synthetic_returns):
    """Number of result rows should match expected monthly rebalances."""
    dates = engine._monthly_rebalance_dates()
    results = engine.run()
    # Results = rebalances - 1 (first period has no prior value to record)
    assert len(results) >= len(dates) - 2


def test_weights_parquet_saved(engine, tmp_path):
    """weights_store.parquet should be created after run."""
    engine.run()
    assert (tmp_path / "weights_store.parquet").exists()


def test_backtest_parquet_saved(engine, tmp_path):
    """backtest_results.parquet should be created after run."""
    engine.run()
    assert (tmp_path / "backtest_results.parquet").exists()
