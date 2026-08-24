"""Leakage-safe feature engineering for 24h realized-vol regression.

EDA lives only in notebooks/01_eda.ipynb. This module implements MLDLC
feature transformation, construction, and selection from those findings.

Not applied (on purpose, given EDA + tree models):
- Scaling / standardization (trees split on thresholds)
- Winsorizing / dropping IQR outliers (fat tails are signal)
- PCA (small selected set; keep interpretability)
- Gap fill (EDA: missing_hours = 0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backend.src import (
    ENTITY_COLUMN,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    RAW_PATH,
    ROOT,
    SYMBOL_ID,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from backend.src.validate_data import validate_ohlcv

HORIZON = 24
CHART_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}. Run `python -m backend.src.data_collection` first.")
    frame = pd.read_parquet(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _rolling_z(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def add_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """Transformation + construction at time t using only information known at t."""
    close = group["close"]
    volume = group["volume"]
    high = group["high"]
    low = group["low"]
    trades = group["trades"] if "trades" in group.columns else pd.Series(np.nan, index=group.index)
    taker = group["taker_buy_base"] if "taker_buy_base" in group.columns else pd.Series(np.nan, index=group.index)

    group = group.copy()

    # --- Transformation ---
    # Missing: none in raw; NaNs appear only as indicator warmup and are dropped later.
    # Outliers: kept (EDA kurtosis ~11–13; 5–8% IQR tails are expected).
    # Categorical encoding: fixed symbol_id map, not data-order codes.
    group["log_volume"] = np.log1p(volume.clip(lower=0.0))
    group["symbol_id"] = group["symbol"].map(SYMBOL_ID).astype("Int64")
    group["hour"] = group["open_time"].dt.hour.astype(int)
    group["dow"] = group["open_time"].dt.dayofweek.astype(int)

    # --- Construction (domain: volatility, not price level) ---
    group["log_return"] = np.log(close / close.shift(1))
    group["abs_return"] = group["log_return"].abs()

    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    group["sma_ratio"] = sma_10 / sma_50.replace(0.0, np.nan)
    group["ema_ratio"] = ema_12 / ema_26.replace(0.0, np.nan)
    group["rsi_14"] = _rsi(close, 14)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    group["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    group["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)

    group["vol_6h"] = group["log_return"].rolling(6).std()
    group["vol_24h_hist"] = group["log_return"].rolling(24).std()
    group["vol_72h"] = group["log_return"].rolling(72).std()
    group["vol_term_structure"] = group["vol_6h"] / group["vol_24h_hist"].replace(0.0, np.nan)

    group["volume_z"] = _rolling_z(volume, 20)
    group["trades_z"] = _rolling_z(trades.astype(float), 20)
    group["taker_buy_ratio"] = taker.astype(float) / volume.replace(0.0, np.nan)
    group["candle_range"] = (high - low) / close.replace(0.0, np.nan)
    group["shock_volume"] = group["candle_range"] * group["volume_z"]

    # Target: std(log_return[t+1], ..., log_return[t+24]) — not used as a feature
    group[TARGET_COLUMN] = group["log_return"].rolling(HORIZON).std().shift(-HORIZON)
    return group


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    parts = [add_indicators(group) for _, group in raw.groupby("symbol", sort=False)]
    features = pd.concat(parts, ignore_index=True)
    features[TIMESTAMP_COLUMN] = features["open_time"]
    keep = [ENTITY_COLUMN, TIMESTAMP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    keep = [c for c in keep if c in features.columns]
    extra = [c for c in CHART_COLUMNS if c in features.columns]
    features = features[keep + extra]
    return features.sort_values([ENTITY_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


def drop_warmup(features: pd.DataFrame) -> pd.DataFrame:
    """Drop indicator warmup rows; keep rows that still have a target for training."""
    needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    cleaned = features.dropna(subset=[c for c in needed if c in features.columns])
    return cleaned.reset_index(drop=True)


def run(raw_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    raw = load_raw(raw_path)
    validate_ohlcv(raw).raise_if_invalid()
    features = build_features(raw)
    output_path = output_path or FEATURES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    trainable = drop_warmup(features)
    print(f"Wrote {len(features):,} feature rows ({len(trainable):,} trainable) -> {output_path}")
    print("Selected features:", ", ".join(FEATURE_COLUMNS))
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write leakage-safe feature parquet (EDA is in notebooks/01_eda.ipynb).")
    parser.add_argument("--raw", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw) if args.raw else RAW_PATH
    output_path = Path(args.output) if args.output else FEATURES_PATH
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    run(raw_path=raw_path, output_path=output_path)


if __name__ == "__main__":
    main()
