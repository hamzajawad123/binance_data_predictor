"""Data-quality and walk-forward helper tests (no live Binance, no full model fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.src.validate_data import validate_ohlcv
from backend.src.walkforward import _bootstrap_ci, _diebold_mariano, _fold_bounds, _fmt_p, _pairwise_dm


def _ohlcv(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "open_time": idx,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
        }
    )


def test_validate_clean_ohlcv():
    report = validate_ohlcv(_ohlcv())
    assert report.ok
    assert report.n_rows == 48


def test_validate_detects_gap_and_bad_high():
    frame = _ohlcv()
    frame = pd.concat(
        [frame.iloc[:10], frame.iloc[12:]],
        ignore_index=True,
    )
    report = validate_ohlcv(frame)
    assert report.ok
    assert report.warnings
    assert any("gap" in item.lower() for item in report.warnings)

    bad = _ohlcv(5)
    bad.loc[2, "high"] = bad.loc[2, "low"] - 1
    report = validate_ohlcv(bad)
    assert not report.ok
    assert any("high < low" in issue for issue in report.issues)


def test_fold_bounds_are_causal():
    idx = pd.date_range("2022-01-01", periods=800, freq="D", tz="UTC")
    folds = _fold_bounds(idx, test_days=90, min_train_days=365)
    assert folds
    for start, train_end, test_end in folds:
        assert start < train_end < test_end
        assert (train_end - start).days >= 365


def test_diebold_mariano_detects_worse_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=500)
    good = y + rng.normal(0, 0.1, size=500)
    bad = y + rng.normal(0, 1.0, size=500)
    result = _diebold_mariano(y - bad, y - good)
    assert result["p_value"] < 0.05
    assert result["mean_d"] > 0


def test_pairwise_dm_ignores_nan_rows():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, size=400)
    good = y + rng.normal(0, 0.1, size=400)
    bad = y + rng.normal(0, 1.0, size=400)
    bad[:20] = np.nan
    result = _pairwise_dm(y, bad, good)
    assert result["n"] == 380
    assert result["p_value"] < 0.05
    assert result["mean_d"] > 0


def test_fmt_p_underflow():
    assert _fmt_p(0.0) == "<1e-300"
    assert _fmt_p(0.01) == "0.01"


def test_bootstrap_ci_covers_point():
    y = np.linspace(0.01, 0.02, 200)
    pred = y + 0.0001
    ci = _bootstrap_ci(y, pred, n_boot=80)
    assert ci["mae"]["low"] <= ci["mae"]["high"]


def test_binom_two_sided_p_extreme():
    from backend.src.walkforward import _binom_two_sided_p, _error_distribution

    assert _binom_two_sided_p(5000, 10000, 0.5) > 0.05
    assert _binom_two_sided_p(9000, 10000, 0.5) < 1e-6
    y = np.ones(20)
    pred = y + 0.1
    dist = _error_distribution(y, pred)
    assert dist["mean_error"] > 0
    assert 0 <= dist["pct_overpredict"] <= 1


def test_prefer_existing_falls_back(tmp_path):
    from backend.src import _prefer_existing

    missing = tmp_path / "missing.parquet"
    present = tmp_path / "present.parquet"
    present.write_bytes(b"ok")
    assert _prefer_existing(missing, present) == present
    missing.write_bytes(b"local")
    assert _prefer_existing(missing, present) == missing


def test_native_treeshap_ranks_vol_feature():
    import lightgbm as lgb

    from backend.src.explain import _native_tree_shap, mean_abs_shap

    rng = np.random.default_rng(0)
    n = 400
    vol = np.abs(rng.normal(0.01, 0.004, n))
    hour = rng.integers(0, 24, n).astype(float)
    x = pd.DataFrame({"vol_24h_hist": vol, "hour": hour})
    y = 0.9 * vol + rng.normal(0, 0.0005, n)
    model = lgb.LGBMRegressor(n_estimators=40, verbosity=-1, random_state=0)
    model.fit(x, y)
    ranking = mean_abs_shap(_native_tree_shap(model, x), list(x.columns))
    assert ranking[0]["feature"] == "vol_24h_hist"
    assert ranking[0]["share"] > ranking[1]["share"]
