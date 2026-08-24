"""Pipeline unit tests using synthetic OHLCV (no live Binance calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.src import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN
from backend.src.eda_feature_eng import add_indicators, build_features, drop_warmup
from backend.src.evaluate import compute_metrics, directional_accuracy, time_split


def _synthetic_ohlcv(symbol: str = "BTCUSDT", n: int = 400, start_price: float = 50_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    returns = rng.normal(0, 0.004, size=n)
    close = start_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.002, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, size=n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(100, 500, size=n)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "open_time": idx,
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": volume,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "quote_volume": volume * close,
            "trades": rng.integers(100, 1000, size=n),
            "taker_buy_base": volume * 0.5,
            "taker_buy_quote": volume * close * 0.5,
        }
    )


def test_feature_schema_and_target_horizon():
    raw = pd.concat(
        [_synthetic_ohlcv("BTCUSDT"), _synthetic_ohlcv("ETHUSDT", start_price=3000.0)],
        ignore_index=True,
    )
    features = build_features(raw)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, TIMESTAMP_COLUMN, "symbol"]:
        assert col in features.columns

    btc = features.loc[features["symbol"] == "BTCUSDT"].reset_index(drop=True)
    # Target at t uses returns t+1..t+24, so the last 24 rows cannot have a target.
    assert btc[TARGET_COLUMN].iloc[-24:].isna().all()

    row = add_indicators(raw.loc[raw["symbol"] == "BTCUSDT"].copy()).reset_index(drop=True)
    t = 200
    future_rets = row["log_return"].iloc[t + 1 : t + 25]
    expected = float(future_rets.std(ddof=1))
    actual = float(row.loc[t, TARGET_COLUMN])
    assert np.isclose(expected, actual, rtol=1e-6, atol=1e-8)


def test_features_do_not_use_future_close():
    raw = _synthetic_ohlcv()
    base = add_indicators(raw.copy())
    mutated = raw.copy()
    mutated.loc[mutated.index[-1], "close"] = mutated.loc[mutated.index[-1], "close"] * 10
    mutated.loc[mutated.index[-1], "high"] = mutated.loc[mutated.index[-1], "close"]
    future_only = add_indicators(mutated)
    mid = 150
    # Changing the last close must not change SMA/RSI at an earlier bar.
    assert np.isclose(base.loc[mid, "vol_24h_hist"], future_only.loc[mid, "vol_24h_hist"])
    assert np.isclose(base.loc[mid, "rsi_14"], future_only.loc[mid, "rsi_14"])


def test_selected_features_exclude_price_levels():
    for banned in ("open", "high", "low", "close", "sma_10", "sma_50", "ema_12", "ema_26"):
        assert banned not in FEATURE_COLUMNS


def test_warmup_drop():
    raw = pd.concat(
        [_synthetic_ohlcv("BTCUSDT"), _synthetic_ohlcv("ETHUSDT", start_price=2000)],
        ignore_index=True,
    )
    features = drop_warmup(build_features(raw))
    assert features[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum() == 0


def test_time_split_and_metrics():
    raw = _synthetic_ohlcv(n=900)
    features = drop_warmup(build_features(raw))
    train_df, test_df = time_split(features, holdout_days=10)
    assert train_df[TIMESTAMP_COLUMN].max() < test_df[TIMESTAMP_COLUMN].min()
    y = test_df[TARGET_COLUMN].to_numpy()
    pred = y + 0.0001
    metrics = compute_metrics(y, pred)
    assert metrics["rmse"] > 0
    assert 0 <= metrics["directional_accuracy"] <= 1
    assert np.isnan(directional_accuracy(np.array([1.0]), np.array([1.0])))


def test_target_is_not_a_model_feature():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_target_at_t_ignores_returns_beyond_horizon():
    raw = _synthetic_ohlcv(n=400)
    base = add_indicators(raw.copy())
    mutated = raw.copy()
    mutated.loc[mutated.index[-1], "close"] = mutated.loc[mutated.index[-1], "close"] * 5
    future = add_indicators(mutated)
    t = 200
    assert np.isclose(base.loc[t, TARGET_COLUMN], future.loc[t, TARGET_COLUMN])
