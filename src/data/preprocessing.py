"""ReturnEngine and CovarianceEngine — transforms prices into features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.utils.config import Settings
from src.utils.logger import get_logger

RETURNS_PATH = Path("data/processed/returns.parquet")
COV_PATH = Path("data/processed/cov_matrix.parquet")


class ReturnEngine:
    """Computes log-returns from daily adjusted-close prices."""

    def __init__(self, prices: pd.DataFrame, settings: Settings | None = None) -> None:
        self.prices = prices
        self.logger = get_logger(
            __name__,
            settings.log_dir if settings else Path("logs/"),
            settings.log_level if settings else "INFO",
        )

    def compute(self) -> pd.DataFrame:
        if RETURNS_PATH.exists():
            cached = pd.read_parquet(RETURNS_PATH)
            if not cached.empty:
                self.logger.info("Cache hit — loading returns from %s", RETURNS_PATH)
                return cached
            self.logger.warning("Cached returns.parquet is empty — recomputing")
            RETURNS_PATH.unlink()

        returns = np.log(self.prices / self.prices.shift(1)).dropna()
        returns.index.name = "date"

        RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        returns.to_parquet(RETURNS_PATH)
        self.logger.info(
            "Computed %d daily log-returns for %d assets — saved to %s",
            len(returns),
            returns.shape[1],
            RETURNS_PATH,
        )
        return returns


class CovarianceEngine:
    """Estimates annualised covariance matrix via Ledoit-Wolf shrinkage."""

    TRADING_DAYS = 252

    def __init__(self, returns: pd.DataFrame, settings: Settings | None = None) -> None:
        self.returns = returns
        self.logger = get_logger(
            __name__,
            settings.log_dir if settings else Path("logs/"),
            settings.log_level if settings else "INFO",
        )

    def compute(self) -> pd.DataFrame:
        if COV_PATH.exists():
            self.logger.info("Cache hit — loading cov_matrix from %s", COV_PATH)
            return pd.read_parquet(COV_PATH)

        return self._estimate_and_cache(self.returns)

    def estimate_from_window(self, returns_window: pd.DataFrame) -> pd.DataFrame:
        """Estimate without touching cache — used inside backtest loop."""
        lw = LedoitWolf().fit(returns_window.values)
        cov = pd.DataFrame(
            lw.covariance_ * self.TRADING_DAYS,
            index=returns_window.columns,
            columns=returns_window.columns,
        )
        return cov

    def _estimate_and_cache(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = self.estimate_from_window(returns)

        COV_PATH.parent.mkdir(parents=True, exist_ok=True)
        cov.to_parquet(COV_PATH)
        self.logger.info(
            "Estimated Ledoit-Wolf covariance (%dx%d) — saved to %s",
            cov.shape[0],
            cov.shape[1],
            COV_PATH,
        )
        return cov
