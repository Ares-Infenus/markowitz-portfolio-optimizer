# Markowitz Portfolio Optimizer

[![CI Pipeline](https://github.com/Ares-Infenus/markowitz-portfolio-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Ares-Infenus/markowitz-portfolio-optimizer/actions)
[![Coverage](https://codecov.io/gh/Ares-Infenus/markowitz-portfolio-optimizer/branch/main/graph/badge.svg)](https://codecov.io/gh/Ares-Infenus/markowitz-portfolio-optimizer)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade portfolio optimization system implementing **4 quantitative methods** with a **6-year backtest** (2019–2024) and an interactive Streamlit dashboard.

---

## Methods

| Method | Solver | Objective |
|--------|--------|-----------|
| **Mean-Variance (Markowitz)** | cvxpy QP | Maximize μᵀw − λ·wᵀΣw |
| **Risk Parity** | scipy SLSQP | Equalize risk contributions |
| **CVaR Optimization** | cvxpy LP (Rockafellar-Uryasev) | Minimize tail loss at 95% |
| **Black-Litterman** | PyPortfolioOpt + momentum views | Posterior mean-variance |

All methods use **Ledoit-Wolf shrinkage covariance**, respect weight constraints (2%–20% per asset, 40% per sector), and face **0.1% transaction costs** on actual portfolio turnover.

---

## Architecture

```
Data (yfinance) → Returns + Cov → Optimize × 4 → Backtest → Dashboard
```

- **Strategy Pattern** — all optimizers implement `BaseOptimizer.optimize()`
- **Pipeline idempotency** — Parquet cache prevents redundant downloads
- **Repository Pattern** — data source abstracted behind `DataIngestion`
- **Circuit breaker** — one optimizer failing never stops the others

---

## Quickstart

### Docker (recommended)

```bash
git clone https://github.com/Ares-Infenus/markowitz-portfolio-optimizer.git
cd markowitz-portfolio-optimizer
cp .env.example .env

# Run pipeline (downloads data, optimizes, backtests)
docker compose up optimizer

# Launch dashboard at http://localhost:8501
docker compose up dashboard
```

### Local

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python src/pipeline/run_pipeline.py
streamlit run src/dashboard/app.py
```

---

## Dashboard

Four interactive Plotly visualizations:

1. **Efficient Frontier** — scatter colored by Sharpe Ratio, star markers for each method
2. **Cumulative Returns** — lines per method + equal-weight benchmark, COVID/2022 bear market shaded
3. **Weights Heatmap** — allocation over time (tickers × rebalance months)
4. **Performance Table** — Sharpe, Sortino, Max Drawdown, CVaR, Calmar for all methods

---

## Universe

10 assets across 5 sectors: Technology (AAPL, MSFT), Financials (JPM, GS), Healthcare (JNJ, UNH), Energy (XOM), Utilities (NEE), Consumer Discretionary (AMZN), Consumer Staples (PG).

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest --cov=src -v
```

Target coverage: ≥85% on core modules (data, optimizers, backtest, utils).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Optimization | cvxpy 1.4.3 (QP/LP), scipy SLSQP |
| Black-Litterman | PyPortfolioOpt 1.5.5 |
| Covariance | scikit-learn LedoitWolf |
| Data | yfinance 0.2.40 → Parquet (pyarrow) |
| Dashboard | Streamlit 1.35.0 + Plotly 5.x |
| Infrastructure | Docker + GitHub Actions CI |

---

## License

MIT © Sebastian
