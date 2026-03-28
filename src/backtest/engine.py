"""BacktestEngine — monthly rebalance with expanding window and transaction costs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import CovarianceEngine
from src.optimizers.base import BaseOptimizer
from src.utils.config import Settings
from src.utils.logger import get_logger

WEIGHTS_PATH = Path("data/results/weights_store.parquet")
BACKTEST_PATH = Path("data/results/backtest_results.parquet")


class BacktestEngine:
    """Runs an expanding-window monthly backtest for a single optimizer."""

    TRADING_DAYS = 252

    def __init__(
        self,
        optimizer: BaseOptimizer,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,  # full-period — used as fallback
        expected_returns: pd.Series,
        settings: Settings,
        method_name: str = "unknown",
    ) -> None:
        self.optimizer = optimizer
        self.returns = returns
        self.full_cov = cov_matrix
        self.full_expected_returns = expected_returns
        self.settings = settings
        self.method_name = method_name
        self.logger = get_logger(__name__, settings.log_dir, settings.log_level)
        self._cov_engine = CovarianceEngine(returns, settings)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run backtest and return daily portfolio value DataFrame."""
        rebalance_dates = self._monthly_rebalance_dates()
        tickers = list(self.returns.columns)

        weights: pd.Series | None = None
        prev_weights: pd.Series | None = None
        portfolio_value = 1.0
        records = []
        weight_records = []

        self.logger.info(
            "%s backtest started — %d rebalances over %s→%s",
            self.method_name,
            len(rebalance_dates),
            self.returns.index[0].date(),
            self.returns.index[-1].date(),
        )

        for i, (rebal_date, next_rebal_date) in enumerate(
            zip(rebalance_dates, rebalance_dates[1:] + [None])
        ):
            # ── Expanding window: all data up to rebalance date ───────────
            window = self.returns.loc[:rebal_date]
            if len(window) < 20:
                weights = self.optimizer._equal_weight(tickers)
            else:
                cov = self._cov_engine.estimate_from_window(window)
                exp_ret = window.mean() * self.TRADING_DAYS
                try:
                    weights = self.optimizer.optimize(window, cov, exp_ret)
                except Exception as exc:
                    self.logger.error(
                        "%s optimization failed at %s: %s — using equal weight",
                        self.method_name,
                        rebal_date.date(),
                        exc,
                    )
                    weights = self.optimizer._equal_weight(tickers)

            # ── Turnover & transaction costs ──────────────────────────────
            if prev_weights is None:
                turnover = 0.0
                tx_cost = 0.0
            else:
                turnover = float((weights - prev_weights).abs().sum() / 2)
                tx_cost = turnover * self.settings.transaction_cost

            # ── Store weights ─────────────────────────────────────────────
            weight_records.append(
                {
                    "date": rebal_date,
                    "method": self.method_name,
                    **weights.to_dict(),
                    "turnover": turnover,
                }
            )

            # ── Simulate returns until next rebalance ─────────────────────
            if next_rebal_date is not None:
                period_returns = self.returns.loc[
                    (self.returns.index > rebal_date)
                    & (self.returns.index <= next_rebal_date)
                ]
            else:
                period_returns = self.returns.loc[self.returns.index > rebal_date]

            if period_returns.empty:
                prev_weights = weights
                continue

            # Monthly log-return for the portfolio
            daily_log_returns = period_returns[tickers] @ weights.values
            monthly_log_return = daily_log_returns.sum()
            monthly_return = float(np.exp(monthly_log_return) - 1)
            net_monthly_return = monthly_return - tx_cost

            prev_value = portfolio_value
            portfolio_value *= 1 + net_monthly_return

            records.append(
                {
                    "date": next_rebal_date if next_rebal_date else period_returns.index[-1],
                    "method": self.method_name,
                    "portfolio_value": portfolio_value,
                    "monthly_return": monthly_return,
                    "net_monthly_return": net_monthly_return,
                    "turnover": turnover,
                    "tx_cost_applied": tx_cost,
                }
            )

            self.logger.info(
                "%s | %s | weights_ok=%s | turnover=%.2f%% | net_ret=%.2f%%",
                self.method_name,
                rebal_date.date(),
                self.optimizer.validate_weights(weights),
                turnover * 100,
                net_monthly_return * 100,
            )

            prev_weights = weights

        results_df = pd.DataFrame(records)
        results_df["date"] = pd.to_datetime(results_df["date"])
        results_df = results_df.set_index("date")

        weights_df = pd.DataFrame(weight_records)
        weights_df["date"] = pd.to_datetime(weights_df["date"])

        self._save(results_df, weights_df)
        return results_df

    # ── Private ───────────────────────────────────────────────────────────────

    def _monthly_rebalance_dates(self) -> list[pd.Timestamp]:
        """First business day of each month within the returns index."""
        idx = self.returns.index
        monthly = idx.to_series().resample("MS").first()
        dates = [d for d in monthly if d in idx]
        return dates

    def _save(self, results_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
        BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing stores or create new ones
        if WEIGHTS_PATH.exists():
            existing = pd.read_parquet(WEIGHTS_PATH)
            existing = existing[existing["method"] != self.method_name]
            weights_df = pd.concat([existing, weights_df], ignore_index=True)
        weights_df.to_parquet(WEIGHTS_PATH, index=False)

        if BACKTEST_PATH.exists():
            existing = pd.read_parquet(BACKTEST_PATH)
            existing = existing[existing["method"] != self.method_name]
            results_df_save = results_df.reset_index()
            results_df_save["method"] = self.method_name
            combined = pd.concat([existing.reset_index(drop=True), results_df_save], ignore_index=True)
        else:
            combined = results_df.reset_index()
            combined["method"] = self.method_name

        combined.to_parquet(BACKTEST_PATH, index=False)
        self.logger.info(
            "%s — saved %d monthly results to %s",
            self.method_name,
            len(results_df),
            BACKTEST_PATH,
        )
