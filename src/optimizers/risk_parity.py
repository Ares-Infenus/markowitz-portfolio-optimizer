"""Risk Parity optimizer — scipy SLSQP (non-convex objective)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.optimizers.base import BaseOptimizer
from src.utils.config import Settings
from src.utils.logger import get_logger


class RiskParityOptimizer(BaseOptimizer):
    """Equalise risk contributions across all assets."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.logger = get_logger(__name__, settings.log_dir, settings.log_level)

    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series,
    ) -> pd.Series:
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        Sigma = self._to_numpy(cov_matrix)

        def _risk_contributions(w: np.ndarray) -> np.ndarray:
            portfolio_var = w @ Sigma @ w
            marginal = Sigma @ w
            rc = w * marginal / portfolio_var
            return rc

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contributions(w)
            # Sum of squared pairwise differences
            diff = rc[:, None] - rc[None, :]
            return float(np.sum(diff**2))

        def _gradient(w: np.ndarray) -> np.ndarray:
            """Numerical gradient — scipy also uses it for SLSQP."""
            eps = 1e-6
            grad = np.zeros(n)
            base = _objective(w)
            for i in range(n):
                w_eps = w.copy()
                w_eps[i] += eps
                grad[i] = (_objective(w_eps) - base) / eps
            return grad

        w0 = np.full(n, 1.0 / n)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.settings.min_weight, self.settings.max_weight)] * n

        result = minimize(
            _objective,
            w0,
            jac=_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not result.success:
            self.logger.warning(
                "RiskParity solver did not converge: %s — falling back to equal weight",
                result.message,
            )
            return self._equal_weight(tickers)

        weights = pd.Series(result.x, index=tickers)
        weights = self._clip_and_renormalize(weights)

        rc = _risk_contributions(weights.values)
        self.logger.info(
            "RiskParity optimized — RC range=[%.4f, %.4f]  iterations=%d",
            rc.min(),
            rc.max(),
            result.nit,
        )
        return weights
