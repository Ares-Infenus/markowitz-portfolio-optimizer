"""PerformanceAnalytics — computes all risk/return metrics from backtest results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import Settings
from src.utils.logger import get_logger

PERF_PATH = Path("data/results/performance_summary.parquet")


class PerformanceAnalytics:
    """Compute and save performance summary for all methods."""

    def __init__(self, all_results: dict[str, pd.DataFrame], settings: Settings) -> None:
        self.all_results = all_results
        self.settings = settings
        self.logger = get_logger(__name__, settings.log_dir, settings.log_level)

    # ── Public ────────────────────────────────────────────────────────────────

    def compute(self) -> pd.DataFrame:
        records = []
        for method, df in self.all_results.items():
            if df.empty:
                continue
            record = {"method": method}
            record.update(self._metrics_for(df))
            records.append(record)
            self.logger.info(
                "%s — Ann.Return=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%",
                method,
                record["annualized_return"] * 100,
                record["sharpe_ratio"],
                record["max_drawdown"] * 100,
            )
        return pd.DataFrame(records)

    def save(self, summary: pd.DataFrame) -> None:
        PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(PERF_PATH, index=False)
        self.logger.info("Performance summary saved to %s", PERF_PATH)

    # ── Private ───────────────────────────────────────────────────────────────

    def _metrics_for(self, df: pd.DataFrame) -> dict:
        returns = df["net_monthly_return"].dropna()
        rf = self.settings.risk_free_rate

        ann_return = float((1 + returns.mean()) ** 12 - 1)
        ann_vol = float(returns.std() * np.sqrt(12))
        sharpe = float((ann_return - rf) / ann_vol) if ann_vol > 0 else 0.0

        downside = returns[returns < 0]
        downside_vol = float(downside.std() * np.sqrt(12)) if len(downside) > 1 else ann_vol
        sortino = float((ann_return - rf) / downside_vol) if downside_vol > 0 else 0.0

        max_dd = self._max_drawdown(df["portfolio_value"])

        cvar_threshold = returns.quantile(0.05)
        cvar_95 = float(returns[returns < cvar_threshold].mean()) if (returns < cvar_threshold).any() else float(cvar_threshold)

        calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

        return {
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "cvar_95": cvar_95,
            "calmar_ratio": calmar,
        }

    @staticmethod
    def _max_drawdown(portfolio_values: pd.Series) -> float:
        values = portfolio_values.values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def benchmark_equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
        """Build equal-weight benchmark DataFrame matching backtest format."""
        n = returns.shape[1]
        ew_weights = np.full(n, 1.0 / n)

        monthly_idx = returns.index.to_series().resample("MS").first()
        monthly_dates = [d for d in monthly_idx if d in returns.index]

        records = []
        portfolio_value = 1.0

        for i, (date, next_date) in enumerate(
            zip(monthly_dates, monthly_dates[1:] + [None])
        ):
            if next_date is not None:
                period = returns.loc[(returns.index > date) & (returns.index <= next_date)]
            else:
                period = returns.loc[returns.index > date]

            if period.empty:
                continue

            daily_log_ret = period.values @ ew_weights
            monthly_log_ret = daily_log_ret.sum()
            monthly_return = float(np.exp(monthly_log_ret) - 1)
            portfolio_value *= 1 + monthly_return

            records.append(
                {
                    "date": next_date if next_date else period.index[-1],
                    "method": "equal_weight",
                    "portfolio_value": portfolio_value,
                    "monthly_return": monthly_return,
                    "net_monthly_return": monthly_return,
                    "turnover": 0.0,
                    "tx_cost_applied": 0.0,
                }
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
