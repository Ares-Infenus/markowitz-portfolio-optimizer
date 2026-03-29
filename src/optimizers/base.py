"""BaseOptimizer — Strategy Pattern ABC for all portfolio optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.utils.config import Settings


class BaseOptimizer(ABC):
    """All optimizers implement this interface."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series,
    ) -> pd.Series:
        """Return a pd.Series of weights indexed by ticker, summing to 1.0."""
        ...

    def validate_weights(self, weights: pd.Series) -> bool:
        """Post-optimisation sanity checks."""
        tol = 1e-5
        if abs(weights.sum() - 1.0) > tol:
            return False
        if (weights < self.settings.min_weight - tol).any():
            return False
        if (weights > self.settings.max_weight + tol).any():
            return False
        return True

    def _clip_and_renormalize(self, weights: pd.Series) -> pd.Series:
        """Clip to [min_weight, max_weight] and re-normalise to sum=1."""
        w = weights.clip(self.settings.min_weight, self.settings.max_weight)
        return w / w.sum()

    def _equal_weight(self, tickers: list[str]) -> pd.Series:
        n = len(tickers)
        return pd.Series(1.0 / n, index=tickers)

    def _sector_weights(self, weights: pd.Series) -> dict[str, float]:
        sector_map = self.settings.sector_map
        result: dict[str, float] = {}
        for ticker, w in weights.items():
            sector = sector_map.get(str(ticker), "Unknown")
            result[sector] = result.get(sector, 0.0) + w
        return result

    @staticmethod
    def _annualised_returns(returns: pd.DataFrame) -> pd.Series:
        return returns.mean() * 252

    @staticmethod
    def _to_numpy(cov_matrix: pd.DataFrame) -> np.ndarray:
        return cov_matrix.values.astype(float)
