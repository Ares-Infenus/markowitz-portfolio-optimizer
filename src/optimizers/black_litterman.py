"""Black-Litterman optimizer — PyPortfolioOpt + momentum-driven views."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

from src.optimizers.base import BaseOptimizer
from src.utils.config import Settings
from src.utils.logger import get_logger

_MOMENTUM_WINDOW = 126  # ~6 trading months


class BlackLittermanOptimizer(BaseOptimizer):
    """Combine market equilibrium with momentum-based views."""

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
        Sigma = cov_matrix.loc[tickers, tickers]

        # ── Market-cap proxy: equal weights as market portfolio ───────────
        market_weights = pd.Series(1.0 / n, index=tickers)

        # ── Equilibrium returns (CAPM reverse-optimisation) ───────────────
        delta = self.settings.bl_risk_aversion
        Pi = (delta * Sigma @ market_weights).rename("equilibrium")

        # ── Generate views from 6-month momentum ─────────────────────────
        P, Q, Omega = self._build_views(returns[tickers], Pi)

        # ── Black-Litterman posterior ─────────────────────────────────────
        bl = BlackLittermanModel(
            Sigma,
            pi=Pi,
            absolute_views=None,
            P=P,
            Q=Q,
            omega=Omega,
            tau=self.settings.bl_tau,
        )
        posterior_returns = bl.bl_returns()
        posterior_cov = bl.bl_cov()

        # ── Mean-Variance on posterior ────────────────────────────────────
        ef = EfficientFrontier(posterior_returns, posterior_cov)
        ef.add_constraint(lambda w: w >= self.settings.min_weight)
        ef.add_constraint(lambda w: w <= self.settings.max_weight)

        try:
            ef.max_sharpe(risk_free_rate=self.settings.risk_free_rate)
            raw = ef.clean_weights()
        except Exception as exc:
            self.logger.warning("BL EfficientFrontier failed (%s) — using min-vol", exc)
            ef2 = EfficientFrontier(posterior_returns, posterior_cov)
            ef2.add_constraint(lambda w: w >= self.settings.min_weight)
            ef2.add_constraint(lambda w: w <= self.settings.max_weight)
            ef2.min_volatility()
            raw = ef2.clean_weights()

        weights = pd.Series(raw, index=tickers)
        weights = self._clip_and_renormalize(weights)

        self.logger.info(
            "BlackLitterman optimized — top_weight=%.4f  ticker=%s",
            weights.max(),
            weights.idxmax(),
        )
        return weights

    # ── View construction ─────────────────────────────────────────────────────

    def _build_views(
        self,
        returns: pd.DataFrame,
        Pi: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate P, Q, Omega from 6-month momentum."""
        tickers = list(returns.columns)
        n = len(tickers)

        # Use last MOMENTUM_WINDOW rows or all if shorter
        window = min(len(returns), _MOMENTUM_WINDOW)
        momentum = returns.iloc[-window:].sum()
        ranked = momentum.rank()

        top3 = ranked.nlargest(3).index.tolist()
        bot3 = ranked.nsmallest(3).index.tolist()

        views_list = []  # (P_row, q_scalar)
        for t in top3:
            row = np.zeros(n)
            row[tickers.index(t)] = 1.0
            q = float(Pi[t]) + 0.02  # +2% above equilibrium
            views_list.append((row, q))

        for t in bot3:
            row = np.zeros(n)
            row[tickers.index(t)] = 1.0
            q = float(Pi[t]) - 0.01  # -1% below equilibrium
            views_list.append((row, q))

        P = np.vstack([v[0] for v in views_list])  # (K, N)
        Q = np.array([v[1] for v in views_list])  # (K,)

        # Diagonal uncertainty: proportional to variance of each view asset
        Sigma_arr = np.diag(returns.var().values * 252)
        tau = self.settings.bl_tau
        Omega = np.diag([float(tau * P[i] @ Sigma_arr @ P[i]) for i in range(len(views_list))])

        return P, Q, Omega
