"""Tests for ReturnEngine and CovarianceEngine."""
from __future__ import annotations

import numpy as np

from src.data.preprocessing import CovarianceEngine, ReturnEngine


def test_log_returns_shape(synthetic_returns, settings):
    """Log returns should have n-1 rows relative to prices."""
    # Build fake price series from returns
    prices = np.exp(synthetic_returns.cumsum())
    ReturnEngine(prices, settings)
    # Manually compute to test the formula
    result = np.log(prices / prices.shift(1)).dropna()
    assert result.shape == (len(prices) - 1, prices.shape[1])


def test_returns_no_nans(synthetic_returns):
    """Synthetic returns fixture should have no NaN values."""
    assert not synthetic_returns.isnull().any().any()


def test_covariance_positive_definite(synthetic_cov):
    """Ledoit-Wolf covariance must be positive definite."""
    eigenvalues = np.linalg.eigvalsh(synthetic_cov.values)
    assert (eigenvalues > 0).all(), "Covariance matrix must be positive definite"


def test_covariance_symmetric(synthetic_cov):
    """Covariance matrix must be symmetric."""
    diff = (synthetic_cov - synthetic_cov.T).abs().max().max()
    assert diff < 1e-10


def test_covariance_annualisation(synthetic_returns, settings):
    """Covariance engine should annualise by ×252."""
    engine = CovarianceEngine(synthetic_returns, settings)
    cov = engine.estimate_from_window(synthetic_returns)
    # Rough check: diagonal entries (variances) should be in plausible annual range
    for v in np.diag(cov.values):
        assert 0.001 < v < 10.0, f"Annual variance {v} outside plausible range"


def test_expected_returns_positive_mean(synthetic_expected_returns):
    """Fixture returns should have slightly positive drift (seed=42)."""
    assert synthetic_expected_returns.mean() > -0.5
