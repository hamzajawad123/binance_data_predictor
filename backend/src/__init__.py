"""Shared configuration loaded from `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/src → backend → repo root (data/, models/, .env live here)
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")


def env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def env_int(key: str, default: int) -> int:
    return int(env(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(env(key, str(default)))


def env_path(key: str, default: str) -> Path:
    raw = env(key, default)
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


SYMBOLS = [
    s.strip().upper()
    for s in env("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
    if s.strip()
]
SYMBOL_ID = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}

INTERVAL = env("INTERVAL", "1h")
LOOKBACK_DAYS = env_int("LOOKBACK_DAYS", 540)
BINANCE_BASE_URL = env("BINANCE_BASE_URL", "https://api.binance.com").rstrip("/")

RAW_PATH = env_path("RAW_PATH", "data/raw.parquet")
FEATURES_PATH = env_path("FEATURES_PATH", "data/features.parquet")
MODELS_DIR = env_path("MODELS_DIR", "models")
CLOUD_DIR = ROOT / "cloud"

def _prefer_existing(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


RAW_PATH = _prefer_existing(RAW_PATH, CLOUD_DIR / "raw.parquet")
FEATURES_PATH = _prefer_existing(FEATURES_PATH, CLOUD_DIR / "features.parquet")
if not (MODELS_DIR / "model.joblib").exists() and (CLOUD_DIR / "model.joblib").exists():
    MODELS_DIR = CLOUD_DIR
MLFLOW_TRACKING_URI = env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = env("MLFLOW_EXPERIMENT", "crypto-vol-24h")
MODEL_NAME = env("MODEL_NAME", "crypto_vol_24h")
HOLDOUT_DAYS = env_int("HOLDOUT_DAYS", 60)
VOL_ALERT_PERCENTILE = env_float("VOL_ALERT_PERCENTILE", 90.0)

# Selected model inputs after EDA + MLDLC feature selection.
# Raw OHLC levels are stored for charts only — they are not model features.
FEATURE_COLUMNS = [
    "log_return",
    "abs_return",
    "vol_6h",
    "vol_24h_hist",
    "vol_72h",
    "vol_term_structure",
    "candle_range",
    "bb_width",
    "bb_pct",
    "rsi_14",
    "sma_ratio",
    "ema_ratio",
    "log_volume",
    "volume_z",
    "trades_z",
    "taker_buy_ratio",
    "shock_volume",
    "hour",
    "dow",
    "symbol_id",
]

TARGET_COLUMN = "vol_24h"
ENTITY_COLUMN = "symbol"
TIMESTAMP_COLUMN = "event_timestamp"
