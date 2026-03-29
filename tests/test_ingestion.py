"""Tests for DataIngestion — cache hit/miss, cleaning logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import DataIngestion
from src.utils.config import Settings


@pytest.fixture()
def settings(tmp_path):
    s = Settings()
    s.tickers = ["AAPL", "MSFT"]
    s.start_date = "2023-01-01"
    s.end_date = "2023-06-30"
    s.log_dir = tmp_path / "logs"
    return s


@pytest.fixture()
def sample_prices():
    dates = pd.date_range("2023-01-02", periods=50, freq="B")
    data = {"AAPL": np.random.uniform(150, 180, 50), "MSFT": np.random.uniform(250, 300, 50)}
    return pd.DataFrame(data, index=dates)


def test_cache_hit(settings, sample_prices, tmp_path, monkeypatch):
    """If prices.parquet exists, should load from cache without calling yfinance."""
    cache_path = tmp_path / "data" / "raw" / "prices.parquet"
    cache_path.parent.mkdir(parents=True)
    sample_prices.index.name = "date"
    sample_prices.to_parquet(cache_path)

    monkeypatch.setattr("src.data.ingestion.PRICES_PATH", cache_path)

    with patch("yfinance.download") as mock_dl:
        ingestion = DataIngestion(settings)
        result = ingestion.run()
        mock_dl.assert_not_called()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["AAPL", "MSFT"]


def test_forward_fill_limit(settings):
    """Small gaps (≤5 days) should be forward-filled."""
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    data = pd.DataFrame(
        {"AAPL": np.random.uniform(150, 180, 20), "MSFT": np.random.uniform(250, 300, 20)},
        index=dates,
    )
    # Introduce a 3-day gap
    data.iloc[5:8, 0] = np.nan

    ingestion = DataIngestion(settings)
    result = ingestion._clean(data, ["AAPL", "MSFT"])

    assert result["AAPL"].isna().sum() == 0, "Small gap should be forward-filled"


def test_missing_ticker_excluded(settings):
    """Tickers absent from download should be excluded and logged."""
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    data = pd.DataFrame({"AAPL": np.random.uniform(150, 180, 10)}, index=dates)

    ingestion = DataIngestion(settings)
    result = ingestion._clean(data, ["AAPL", "MSFT"])

    assert "AAPL" in result.columns
    assert "MSFT" not in result.columns


def test_download_retry_raises(settings, monkeypatch):
    """After max_attempts failures, RuntimeError should be raised."""
    monkeypatch.setattr("src.data.ingestion.PRICES_PATH", Path("/nonexistent/prices.parquet"))

    with patch("yfinance.download", side_effect=ConnectionError("network")):
        with patch("time.sleep"):
            ingestion = DataIngestion(settings)
            with pytest.raises(RuntimeError, match="failed after"):
                ingestion._download_with_retry(["AAPL"], "2023-01-01", "2023-06-30", max_attempts=2)
