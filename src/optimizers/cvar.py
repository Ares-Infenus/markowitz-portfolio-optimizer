"""CVaR optimizer — Rockafellar-Uryasev LP via cvxpy."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from src.optimizers.base import BaseOptimizer
from src.utils.config import Settings
from src.utils.logger import get_logger


class CVaROptimizer(BaseOptimizer):
    """Minimize CVaR at confidence level α using historical scenarios."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.alpha = settings.cvar_confidence
        self.logger = get_logger(__name__, settings.log_dir, settings.log_level)

    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series,
    ) -> pd.Series:
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        # T × N scenario matrix
        R = returns[tickers].values.astype(float)
        T = R.shape[0]

        w = cp.Variable(n)
        zeta = cp.Variable()          # VaR auxiliary
        u = cp.Variable(T)            # exceedance variables

        cvar = zeta + (1.0 / (1.0 - self.alpha)) * (1.0 / T) * cp.sum(u)

        constraints = [
            u >= -R @ w - zeta,
            u >= 0,
            cp.sum(w) == 1,
            w >= self.settings.min_weight,
            w <= self.settings.max_weight,
        ]

        prob = cp.Problem(cp.Minimize(cvar), constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.logger.warning(
                "CVaR solver status: %s — falling back to equal weight", prob.status
            )
            return self._equal_weight(tickers)

        weights = pd.Series(w.value, index=tickers)
        weights = self._clip_and_renormalize(weights)
        portfolio_returns = R @ weights.values
        cvar_val = float(np.mean(portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, (1 - self.alpha) * 100)]))

        self.logger.info(
            "CVaR optimized — CVaR(%.0f%%)=%.4f  status=%s",
            self.alpha * 100,
            cvar_val,
            prob.status,
        )
        return weights
