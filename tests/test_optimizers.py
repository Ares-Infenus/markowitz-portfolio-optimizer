"""Critical optimizer tests — weight constraints, solver correctness."""
from __future__ import annotations

import numpy as np
import pytest

from src.optimizers.black_litterman import BlackLittermanOptimizer
from src.optimizers.cvar import CVaROptimizer
from src.optimizers.mean_variance import MeanVarianceOptimizer
from src.optimizers.risk_parity import RiskParityOptimizer

TOL = 1e-5


# ── Shared weight-constraint parametrisation ─────────────────────────────────

@pytest.fixture(
    params=["mean_variance", "risk_parity", "cvar", "black_litterman"],
    ids=["mean_variance", "risk_parity", "cvar", "black_litterman"],
)
def optimizer_instance(request, settings):
    mapping = {
        "mean_variance": MeanVarianceOptimizer,
        "risk_parity": RiskParityOptimizer,
        "cvar": CVaROptimizer,
        "black_litterman": BlackLittermanOptimizer,
    }
    return mapping[request.param](settings)


def test_weights_sum_to_one(optimizer_instance, synthetic_returns, synthetic_cov, synthetic_expected_returns):
    """Weights must sum to 1.0 within tolerance for ALL methods."""
    weights = optimizer_instance.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)
    assert abs(weights.sum() - 1.0) < TOL, f"Weights sum={weights.sum():.8f} for {type(optimizer_instance).__name__}"


def test_no_weight_exceeds_max(optimizer_instance, synthetic_returns, synthetic_cov, synthetic_expected_returns, settings):
    """No single weight should exceed MAX_WEIGHT."""
    weights = optimizer_instance.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)
    assert (weights <= settings.max_weight + TOL).all(), (
        f"Weight exceeds max: {weights.max():.4f} > {settings.max_weight} in {type(optimizer_instance).__name__}"
    )


def test_no_weight_below_min(optimizer_instance, synthetic_returns, synthetic_cov, synthetic_expected_returns, settings):
    """No single weight should be below MIN_WEIGHT."""
    weights = optimizer_instance.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)
    assert (weights >= settings.min_weight - TOL).all(), (
        f"Weight below min: {weights.min():.4f} < {settings.min_weight} in {type(optimizer_instance).__name__}"
    )


def test_output_is_series_indexed_by_tickers(optimizer_instance, synthetic_returns, synthetic_cov, synthetic_expected_returns, settings):
    """Output must be a pd.Series indexed by the tickers."""
    import pandas as pd

    weights = optimizer_instance.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)
    assert isinstance(weights, pd.Series), f"Expected pd.Series, got {type(weights)}"
    assert set(weights.index) == set(settings.tickers), f"Unexpected index: {list(weights.index)}"


# ── Method-specific tests ─────────────────────────────────────────────────────

def test_risk_parity_equal_contributions(settings, synthetic_returns, synthetic_cov, synthetic_expected_returns):
    """Risk contributions should be nearly equal (within 1%)."""
    optimizer = RiskParityOptimizer(settings)
    weights = optimizer.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)

    Sigma = synthetic_cov.values
    w = weights.values
    portfolio_var = w @ Sigma @ w
    rc = w * (Sigma @ w) / portfolio_var

    assert rc.max() - rc.min() < 0.01, (
        f"Risk contributions not equal — range={rc.max() - rc.min():.4f}"
    )


def test_cvar_lower_tail_risk(settings, synthetic_returns, synthetic_cov, synthetic_expected_returns):
    """CVaR portfolio's realised CVaR should be ≤ equal-weight CVaR on the test set."""
    from src.optimizers.base import BaseOptimizer

    cvar_opt = CVaROptimizer(settings)
    cvar_weights = cvar_opt.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)

    ew_weights = cvar_opt._equal_weight(list(synthetic_cov.columns))

    R = synthetic_returns.values
    cvar_returns = R @ cvar_weights.values
    ew_returns = R @ ew_weights.values

    threshold_cvar = np.percentile(cvar_returns, 5)
    threshold_ew = np.percentile(ew_returns, 5)

    cvar_cvar95 = float(np.mean(cvar_returns[cvar_returns < threshold_cvar]))
    ew_cvar95 = float(np.mean(ew_returns[ew_returns < threshold_ew]))

    # CVaR optimizer should achieve ≤ or comparable CVaR vs naive equal weight
    assert cvar_cvar95 <= ew_cvar95 + 0.005, (
        f"CVaR optimizer CVaR ({cvar_cvar95:.4f}) worse than equal weight ({ew_cvar95:.4f})"
    )


def test_black_litterman_top_momentum_higher_weight(settings, synthetic_returns, synthetic_cov, synthetic_expected_returns):
    """Top-momentum asset should have ≥ weight than bottom-momentum asset."""
    optimizer = BlackLittermanOptimizer(settings)
    weights = optimizer.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)

    window = min(len(synthetic_returns), 126)
    momentum = synthetic_returns.iloc[-window:].sum()
    top_ticker = momentum.idxmax()
    bot_ticker = momentum.idxmin()

    assert weights[top_ticker] >= weights[bot_ticker] - 0.05, (
        f"Top momentum {top_ticker}={weights[top_ticker]:.4f} < bottom {bot_ticker}={weights[bot_ticker]:.4f}"
    )


def test_mean_variance_validate_weights(settings, synthetic_returns, synthetic_cov, synthetic_expected_returns):
    """validate_weights() should pass on optimizer output."""
    optimizer = MeanVarianceOptimizer(settings)
    weights = optimizer.optimize(synthetic_returns, synthetic_cov, synthetic_expected_returns)
    assert optimizer.validate_weights(weights)


def test_efficient_frontier_shape(settings, synthetic_returns, synthetic_cov, synthetic_expected_returns):
    """Efficient frontier should return 100 points with correct columns."""
    optimizer = MeanVarianceOptimizer(settings)
    frontier = optimizer.efficient_frontier(synthetic_returns, synthetic_cov, synthetic_expected_returns, n_points=10)
    assert len(frontier) == 10
    assert {"lambda", "annualized_return", "annualized_vol", "sharpe"}.issubset(frontier.columns)
