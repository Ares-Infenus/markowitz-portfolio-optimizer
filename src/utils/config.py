"""Typed configuration dataclass loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── Data ─────────────────────────────────────────────────
    tickers: List[str] = field(
        default_factory=lambda: os.getenv(
            "TICKERS", "AAPL,MSFT,JPM,GS,JNJ,UNH,XOM,NEE,AMZN,PG"
        ).split(",")
    )
    start_date: str = field(default_factory=lambda: os.getenv("START_DATE", "2019-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("END_DATE", "2024-12-31"))

    # ── Optimization Constraints ──────────────────────────────
    max_weight: float = field(default_factory=lambda: float(os.getenv("MAX_WEIGHT", "0.20")))
    min_weight: float = field(default_factory=lambda: float(os.getenv("MIN_WEIGHT", "0.02")))
    sector_limit: float = field(default_factory=lambda: float(os.getenv("SECTOR_LIMIT", "0.40")))
    max_turnover: float = field(default_factory=lambda: float(os.getenv("MAX_TURNOVER", "0.15")))
    transaction_cost: float = field(
        default_factory=lambda: float(os.getenv("TRANSACTION_COST", "0.001"))
    )
    risk_free_rate: float = field(
        default_factory=lambda: float(os.getenv("RISK_FREE_RATE", "0.05"))
    )
    cvar_confidence: float = field(
        default_factory=lambda: float(os.getenv("CVAR_CONFIDENCE", "0.95"))
    )

    # ── Black-Litterman ───────────────────────────────────────
    bl_risk_aversion: float = field(
        default_factory=lambda: float(os.getenv("BL_RISK_AVERSION", "2.5"))
    )
    bl_tau: float = field(default_factory=lambda: float(os.getenv("BL_TAU", "0.05")))

    # ── Dashboard ─────────────────────────────────────────────
    streamlit_port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501")))

    # ── Logging ───────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs/")))

    # ── Sector map ────────────────────────────────────────────
    sector_map: dict = field(
        default_factory=lambda: {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "GS": "Financials",
            "JNJ": "Healthcare",
            "UNH": "Healthcare",
            "XOM": "Energy",
            "NEE": "Utilities",
            "AMZN": "Consumer Discretionary",
            "PG": "Consumer Staples",
        }
    )

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
