"""End-to-end pipeline orchestrator."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceAnalytics
from src.data.ingestion import DataIngestion
from src.data.preprocessing import CovarianceEngine, ReturnEngine
from src.optimizers.black_litterman import BlackLittermanOptimizer
from src.optimizers.cvar import CVaROptimizer
from src.optimizers.mean_variance import MeanVarianceOptimizer
from src.optimizers.risk_parity import RiskParityOptimizer
from src.utils.config import Settings
from src.utils.logger import get_logger

OPTIMIZERS = {
    "mean_variance": MeanVarianceOptimizer,
    "risk_parity": RiskParityOptimizer,
    "cvar": CVaROptimizer,
    "black_litterman": BlackLittermanOptimizer,
}


def main() -> None:
    settings = Settings()
    logger = get_logger("pipeline", settings.log_dir, settings.log_level)

    logger.info("=" * 60)
    logger.info("Markowitz Portfolio Optimizer — Pipeline Starting")
    logger.info("=" * 60)
    t0 = time.time()

    # ── Stage 1: Data ingestion ───────────────────────────────────────────
    logger.info("[1/4] Data ingestion")
    prices = DataIngestion(settings).run()

    if prices.empty:
        logger.error("Price data is empty — cannot continue. Check network connectivity.")
        raise RuntimeError("Price data download returned empty DataFrame.")

    logger.info("Prices loaded: %d rows × %d tickers", len(prices), prices.shape[1])

    # ── Stage 2: Feature engineering ─────────────────────────────────────
    logger.info("[2/4] Computing returns and covariance")
    returns = ReturnEngine(prices, settings).compute()

    if returns.empty or len(returns) < 20:
        logger.error("Returns DataFrame is empty or too short (%d rows).", len(returns))
        raise RuntimeError("Insufficient return data to run optimizers.")

    cov_matrix = CovarianceEngine(returns, settings).compute()
    expected_returns = returns.mean() * 252

    # ── Stage 3: Optimize + backtest each method ──────────────────────────
    logger.info("[3/4] Running optimizers and backtests")
    all_results: dict = {}

    # Equal-weight benchmark
    benchmark = PerformanceAnalytics.benchmark_equal_weight(returns)
    all_results["equal_weight"] = benchmark

    for name, OptimizerClass in OPTIMIZERS.items():
        t_start = time.time()
        try:
            optimizer = OptimizerClass(settings)
            engine = BacktestEngine(
                optimizer=optimizer,
                returns=returns,
                cov_matrix=cov_matrix,
                expected_returns=expected_returns,
                settings=settings,
                method_name=name,
            )
            result = engine.run()
            all_results[name] = result
            elapsed = time.time() - t_start
            logger.info("✓ %s completed in %.1fs", name, elapsed)
        except Exception as exc:
            logger.error("✗ %s failed: %s", name, exc, exc_info=True)
            continue

    # ── Stage 4: Performance analytics ───────────────────────────────────
    logger.info("[4/4] Computing performance metrics")
    analytics = PerformanceAnalytics(all_results, settings)
    summary = analytics.compute()
    analytics.save(summary)

    # ── Also generate efficient frontier data ─────────────────────────────
    _save_efficient_frontier(returns, cov_matrix, expected_returns, settings, logger)

    elapsed_total = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline completed in %.1fs", elapsed_total)
    logger.info("=" * 60)

    # Print summary table
    print("\n" + summary.to_string(index=False))


def _save_efficient_frontier(returns, cov_matrix, expected_returns, settings, logger):
    """Compute and save efficient frontier data for the dashboard."""
    try:
        mv = MeanVarianceOptimizer(settings)
        frontier = mv.efficient_frontier(returns, cov_matrix, expected_returns, n_points=100)

        # Add special portfolios
        import numpy as np

        # Min variance (highest lambda)
        min_var_optimizer = MeanVarianceOptimizer(settings, risk_aversion=100.0)
        min_var_w = min_var_optimizer.optimize(returns, cov_matrix, expected_returns)
        Sigma = cov_matrix.values
        mu = expected_returns[list(cov_matrix.columns)].values

        min_var_ret = float(mu @ min_var_w.values)
        min_var_vol = float(np.sqrt(min_var_w.values @ Sigma @ min_var_w.values))

        # Max Sharpe (close to lambda=1)
        max_sharpe_optimizer = MeanVarianceOptimizer(settings, risk_aversion=1.0)
        max_sharpe_w = max_sharpe_optimizer.optimize(returns, cov_matrix, expected_returns)
        max_sharpe_ret = float(mu @ max_sharpe_w.values)
        max_sharpe_vol = float(np.sqrt(max_sharpe_w.values @ Sigma @ max_sharpe_w.values))

        specials = pd.DataFrame(
            [
                {
                    "label": "Min Variance",
                    "annualized_return": min_var_ret,
                    "annualized_vol": min_var_vol,
                    "sharpe": (min_var_ret - settings.risk_free_rate) / min_var_vol,
                },
                {
                    "label": "Max Sharpe",
                    "annualized_return": max_sharpe_ret,
                    "annualized_vol": max_sharpe_vol,
                    "sharpe": (max_sharpe_ret - settings.risk_free_rate) / max_sharpe_vol,
                },
            ]
        )

        out_path = Path("data/results/efficient_frontier.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frontier.to_parquet(out_path, index=False)

        specials_path = Path("data/results/frontier_specials.parquet")
        specials.to_parquet(specials_path, index=False)
        logger.info("Efficient frontier saved — %d points", len(frontier))
    except Exception as exc:
        logger.warning("Efficient frontier generation failed: %s", exc)


if __name__ == "__main__":
    main()
