"""Mean-Variance optimizer — Markowitz QP via cvxpy."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from src.optimizers.base import BaseOptimizer
from src.utils.config import Settings
from src.utils.logger import get_logger


class MeanVarianceOptimizer(BaseOptimizer):
    """Maximize μᵀw - λ·wᵀΣw subject to weight constraints."""

    def __init__(self, settings: Settings, risk_aversion: float = 1.0) -> None:
        super().__init__(settings)
        self.risk_aversion = risk_aversion
        self.logger = get_logger(__name__, settings.log_dir, settings.log_level)

    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series,
    ) -> pd.Series:
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        mu = expected_returns[tickers].values.astype(float)
        Sigma = self._to_numpy(cov_matrix)

        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - self.risk_aversion * cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 1,
            w >= self.settings.min_weight,
            w <= self.settings.max_weight,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.logger.warning(
                "MeanVariance solver status: %s — falling back to equal weight", prob.status
            )
            return self._equal_weight(tickers)

        weights = pd.Series(w.value, index=tickers)
        weights = self._clip_and_renormalize(weights)
        self.logger.info(
            "MeanVariance optimized — Sharpe proxy=%.4f  status=%s",
            float(mu @ weights.values) / float(np.sqrt(weights.values @ Sigma @ weights.values)),
            prob.status,
        )
        return weights

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """Generate frontier by sweeping risk_aversion from 0.1 to 10.0."""
        lambdas = np.linspace(0.1, 10.0, n_points)
        records = []
        Sigma = self._to_numpy(cov_matrix)

        for lam in lambdas:
            opt = MeanVarianceOptimizer(self.settings, risk_aversion=float(lam))
            w = opt.optimize(returns, cov_matrix, expected_returns)
            ret = float(expected_returns[list(cov_matrix.columns)].values @ w.values)
            vol = float(np.sqrt(w.values @ Sigma @ w.values))
            sharpe = (ret - self.settings.risk_free_rate) / vol if vol > 0 else 0.0
            records.append(
                {"lambda": lam, "annualized_return": ret, "annualized_vol": vol, "sharpe": sharpe}
            )

        return pd.DataFrame(records)
