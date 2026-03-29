"""DataIngestion — downloads adjusted close prices via yfinance with retry + local cache."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import Settings
from src.utils.logger import get_logger

PRICES_PATH = Path("data/raw/prices.parquet")


class DataIngestion:
    """Downloads and caches daily adjusted-close prices."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, settings.log_dir, settings.log_level)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Return prices DataFrame, using cache if available and non-empty."""
        if PRICES_PATH.exists():
            prices = pd.read_parquet(PRICES_PATH)
            if not prices.empty:
                self.logger.info(
                    "Cache hit — loaded %d rows x %d tickers from %s",
                    len(prices),
                    prices.shape[1],
                    PRICES_PATH,
                )
                return prices
            self.logger.warning("Cached prices.parquet is empty — deleting and re-downloading")
            PRICES_PATH.unlink()

        return self._download_and_cache()

    # ── Private ───────────────────────────────────────────────────────────────

    def _download_and_cache(self) -> pd.DataFrame:
        """Download with exponential backoff, fill small gaps, save to Parquet."""
        tickers = self.settings.tickers
        start = self.settings.start_date
        end = self.settings.end_date

        self.logger.info("Downloading prices — tickers=%s  period=%s→%s", tickers, start, end)

        raw = self._download_with_retry(tickers, start, end)
        prices = self._clean(raw, tickers)

        PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(PRICES_PATH)
        self.logger.info(
            "Downloaded %d rows for %d tickers — saved to %s",
            len(prices),
            prices.shape[1],
            PRICES_PATH,
        )
        return prices

    def _download_with_retry(
        self, tickers: list[str], start: str, end: str, max_attempts: int = 3
    ) -> pd.DataFrame:
        for attempt in range(max_attempts):
            try:
                df = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                # yfinance returns MultiIndex columns when multiple tickers
                if isinstance(df.columns, pd.MultiIndex):
                    df = df["Close"]

                if not df.empty:
                    return df

                # Batch download returned empty — fall back to per-ticker
                self.logger.warning(
                    "Batch download returned empty on attempt %d — trying per-ticker fallback",
                    attempt + 1,
                )
                return self._download_per_ticker(tickers, start, end)

            except Exception as exc:
                wait = 2**attempt
                self.logger.warning(
                    "Attempt %d/%d failed (%s) — retrying in %ds",
                    attempt + 1,
                    max_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Data download failed after {max_attempts} attempts for tickers={tickers}"
        )

    def _download_per_ticker(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download each ticker individually and concat — more robust in some network environments."""
        frames = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df["Close"]
                    col = ticker
                else:
                    col = ticker
                if not df.empty:
                    frames[col] = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                else:
                    self.logger.warning("Per-ticker download empty for %s", ticker)
            except Exception as exc:
                self.logger.warning("Per-ticker download failed for %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("All per-ticker downloads failed")

        result = pd.DataFrame(frames)
        result.index.name = "date"
        return result

    def _clean(self, df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Handle missing tickers and fill small gaps."""
        # Ensure expected tickers are present
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            self.logger.error("Tickers not found in download, excluding: %s", missing)
            df = df[[c for c in tickers if c in df.columns]]

        # Forward-fill gaps ≤ 5 days; warn on longer gaps
        for col in df.columns:
            consec_nan = df[col].isna().astype(int).groupby((~df[col].isna()).cumsum()).cumsum()
            long_gaps = consec_nan[consec_nan > 5]
            if not long_gaps.empty:
                self.logger.warning(
                    "Ticker %s has gaps > 5 consecutive days — not forward-filled", col
                )

        df = df.ffill(limit=5)
        df.index.name = "date"
        return df
