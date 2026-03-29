"""Shared fixtures for all tests — deterministic synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.utils.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    s = Settings()
    s.tickers = ["AAPL", "MSFT", "JPM", "GS", "JNJ"]
    s.min_weight = 0.02
    s.max_weight = 0.40
    s.risk_free_rate = 0.05
    s.cvar_confidence = 0.95
    s.bl_risk_aversion = 2.5
    s.bl_tau = 0.05
    s.transaction_cost = 0.001
    s.sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
        "GS": "Financials",
        "JNJ": "Healthcare",
    }
    return s


@pytest.fixture(scope="session")
def synthetic_returns(settings: Settings) -> pd.DataFrame:
    """250 days × 5 assets, deterministic (seed=42)."""
    rng = np.random.default_rng(42)
    n_days, n_assets = 250, len(settings.tickers)
    # Slight positive drift with realistic vol
    data = rng.normal(0.0005, 0.015, (n_days, n_assets))
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    return pd.DataFrame(data, index=dates, columns=settings.tickers)


@pytest.fixture(scope="session")
def synthetic_cov(synthetic_returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(synthetic_returns.values)
    return pd.DataFrame(
        lw.covariance_ * 252,
        index=synthetic_returns.columns,
        columns=synthetic_returns.columns,
    )


@pytest.fixture(scope="session")
def synthetic_expected_returns(synthetic_returns: pd.DataFrame) -> pd.Series:
    return synthetic_returns.mean() * 252
