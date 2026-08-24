"""FastAPI serving layer: Feast (or parquet fallback) + trained volatility model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.src import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    MODELS_DIR,
    RAW_PATH,
    ROOT,
    SYMBOLS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    VOL_ALERT_PERCENTILE,
)
from backend.src.evaluate import load_model_bundle
from backend.src.shadow import (
    FALLBACK_SIGMA,
    INTERVAL_Z,
    SHADOW_LOG_PATH,
    append_shadow_forecast,
    artifact_fingerprint,
    band,
    estimator_names,
    freshness,
    git_commit,
    interval_sigma,
    score_artifact,
    score_shadow_log,
)

app = FastAPI(
    title="Crypto Volatility Forecast API",
    version="0.4.0",
    description="24h-ahead realized volatility forecasts for BTC, ETH, BNB, and SOL.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_columns: list[str] = list(FEATURE_COLUMNS)
_model_info: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    symbol: str = Field(..., examples=["BTCUSDT"])


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'. Choose from {SYMBOLS}.")
    return symbol


def get_model():
    global _model, _columns
    if _model is None:
        try:
            _model, _columns = load_model_bundle(MODELS_DIR)
        except FileNotFoundError:
            _model = False
    return None if _model is False else _model


def get_model_info() -> dict[str, Any]:
    global _model_info
    model_path = MODELS_DIR / "model.joblib"
    fingerprint = artifact_fingerprint(model_path)
    cache_key = fingerprint.get("sha256_12")
    if _model_info and _model_info.get("sha256_12") == cache_key:
        return _model_info
    model = get_model()
    sigma, sigma_source = interval_sigma(MODELS_DIR)
    info = {
        **fingerprint,
        "estimators": estimator_names(model),
        "n_features": len(_columns),
        "git_commit": git_commit(ROOT),
        "interval_sigma": sigma,
        "interval_source": sigma_source,
        "interval_z": INTERVAL_Z,
        "claimed_coverage": 0.95,
        "app_version": app.version,
    }
    _model_info = info
    return info


def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise HTTPException(status_code=503, detail="Feature parquet is missing. Run data + feature pipelines.")
    frame = pd.read_parquet(FEATURES_PATH)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    return frame.sort_values([TIMESTAMP_COLUMN])


def _parquet_latest_row(symbol: str) -> dict[str, Any]:
    frame = _load_features()
    subset = frame.loc[frame["symbol"] == symbol]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No features for {symbol}.")
    usable = subset.dropna(subset=[c for c in FEATURE_COLUMNS if c in subset.columns])
    if usable.empty:
        usable = subset
    row = usable.iloc[-1]
    payload = {col: _json_value(row[col]) for col in FEATURE_COLUMNS if col in row.index}
    payload["symbol"] = symbol
    payload["event_timestamp"] = _json_value(row[TIMESTAMP_COLUMN])
    if TARGET_COLUMN in row.index and pd.notna(row[TARGET_COLUMN]):
        payload["vol_24h_realized"] = _json_value(row[TARGET_COLUMN])
    if "close" in row.index:
        payload["close"] = _json_value(row["close"])
    return payload


def _feast_row(symbol: str) -> dict[str, Any] | None:
    try:
        from backend.feature_repo.features import get_online_feature_row

        row = get_online_feature_row(symbol)
        if not row:
            return None
        payload = {col: _json_value(row.get(col)) for col in FEATURE_COLUMNS}
        payload["symbol"] = symbol
        payload["source"] = "feast"
        return payload
    except Exception:
        return None


def get_feature_snapshot(symbol: str) -> dict[str, Any]:
    feast_row = _feast_row(symbol)
    if feast_row and feast_row.get("vol_24h_hist") is not None:
        parquet_row = _parquet_latest_row(symbol)
        feast_row["event_timestamp"] = parquet_row.get("event_timestamp")
        feast_row["close"] = parquet_row.get("close")
        return feast_row
    snapshot = _parquet_latest_row(symbol)
    snapshot["source"] = "parquet"
    return snapshot


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _alert_threshold(symbol: str, percentile: float = VOL_ALERT_PERCENTILE) -> float | None:
    frame = _load_features()
    hist = frame.loc[frame["symbol"] == symbol, "vol_24h_hist"].dropna()
    if hist.empty:
        return None
    window = hist.tail(24 * 30)
    return float(np.nanpercentile(window.to_numpy(), percentile))


def _predict_vol(snapshot: dict[str, Any]) -> tuple[float, str]:
    model = get_model()
    vector = pd.DataFrame([{col: snapshot.get(col) for col in _columns}])
    if vector.isna().any().any() and snapshot.get("vol_24h_hist") is not None:
        return float(snapshot["vol_24h_hist"]), "baseline_persistence"
    if model is None:
        hist = snapshot.get("vol_24h_hist")
        if hist is None:
            raise HTTPException(status_code=503, detail="No trained model and no persistence fallback.")
        return float(hist), "baseline_persistence"
    pred = float(np.asarray(model.predict(vector), dtype=float)[0])
    return pred, "model"


@app.get("/health")
def health() -> dict[str, Any]:
    info = get_model_info()
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "symbols": SYMBOLS,
        "model_loaded": get_model() is not None,
        "model_version": info.get("version"),
        "features_exist": FEATURES_PATH.exists(),
        "raw_exist": RAW_PATH.exists(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return get_model_info()


@app.get("/features")
def features(symbol: str = Query(..., examples=["BTCUSDT"])) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    return get_feature_snapshot(symbol)


@app.get("/candles")
def candles(
    symbol: str = Query(..., examples=["BTCUSDT"]),
    limit: int = Query(168, ge=24, le=2000),
) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    if not RAW_PATH.exists():
        raise HTTPException(status_code=503, detail="Raw parquet is missing.")
    raw = pd.read_parquet(RAW_PATH)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    subset = raw.loc[raw["symbol"] == symbol].sort_values("open_time").tail(limit)
    records = [
        {
            "time": _json_value(row.open_time),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in subset.itertuples(index=False)
    ]
    vol_hist = None
    if FEATURES_PATH.exists():
        feats = pd.read_parquet(FEATURES_PATH)
        feats[TIMESTAMP_COLUMN] = pd.to_datetime(feats[TIMESTAMP_COLUMN], utc=True)
        vol = (
            feats.loc[feats["symbol"] == symbol, [TIMESTAMP_COLUMN, "vol_24h_hist"]]
            .dropna()
            .sort_values(TIMESTAMP_COLUMN)
            .tail(limit)
        )
        vol_hist = [
            {"time": _json_value(row.event_timestamp), "vol_24h_hist": float(row.vol_24h_hist)}
            for row in vol.itertuples(index=False)
        ]
    return {"symbol": symbol, "candles": records, "realized_vol": vol_hist or []}


def _predict_payload(symbol: str) -> dict[str, Any]:
    snapshot = get_feature_snapshot(symbol)
    predicted, source = _predict_vol(snapshot)
    daily = predicted * float(np.sqrt(24.0))
    threshold = _alert_threshold(symbol)
    alert = bool(threshold is not None and predicted >= threshold)
    info = get_model_info()
    sigma = float(info.get("interval_sigma") or FALLBACK_SIGMA)
    lo, hi = band(predicted, sigma)
    fresh = freshness(snapshot.get("event_timestamp"))
    payload = {
        "symbol": symbol,
        "predicted_vol_24h": predicted,
        "predicted_vol_24h_daily": daily,
        "predicted_vol_24h_low": lo,
        "predicted_vol_24h_high": hi,
        "predicted_vol_24h_daily_low": lo * float(np.sqrt(24.0)),
        "predicted_vol_24h_daily_high": hi * float(np.sqrt(24.0)),
        "interval_sigma": sigma,
        "interval_source": info.get("interval_source"),
        "claimed_coverage": info.get("claimed_coverage"),
        "event_timestamp": snapshot.get("event_timestamp"),
        "stale": fresh["stale"],
        "hours_lag": fresh["hours_lag"],
        "stale_reason": fresh["reason"],
        "prediction_source": source,
        "feature_source": snapshot.get("source"),
        "model_version": info.get("version"),
        "alert": alert,
        "alert_threshold": threshold,
        "alert_reason": (
            f"Predicted vol is at/above the {VOL_ALERT_PERCENTILE:.0f}th percentile of the last 30d realized vol."
            if alert
            else None
        ),
        "features": {k: snapshot.get(k) for k in FEATURE_COLUMNS},
        "close": snapshot.get("close"),
    }
    append_shadow_forecast(
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "event_timestamp": snapshot.get("event_timestamp"),
            "predicted_vol_24h": predicted,
            "model_version": info.get("version"),
            "prediction_source": source,
        },
        path=SHADOW_LOG_PATH,
    )
    return payload


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    symbol = _normalize_symbol(payload.symbol)
    return _predict_payload(symbol)


@app.get("/predict/all")
def predict_all() -> dict[str, Any]:
    results = []
    errors = []
    for symbol in SYMBOLS:
        try:
            results.append(_predict_payload(symbol))
        except HTTPException as exc:
            errors.append({"symbol": symbol, "detail": exc.detail})
    return {"predictions": results, "errors": errors}


@app.get("/score")
def score(
    symbol: str = Query(..., examples=["BTCUSDT"]),
    limit: int = Query(168, ge=24, le=2000),
) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    frame = _load_features()
    info = get_model_info()
    sigma = float(info.get("interval_sigma") or FALLBACK_SIGMA)
    try:
        scored = score_artifact(
            frame,
            get_model(),
            _columns,
            symbol,
            limit,
            sigma,
            z=INTERVAL_Z,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    fresh = freshness(scored.get("as_of"))
    scored.update(
        {
            "stale": fresh["stale"],
            "hours_lag": fresh["hours_lag"],
            "stale_reason": fresh["reason"],
            "model_version": info.get("version"),
            "interval_sigma": sigma,
            "interval_source": info.get("interval_source"),
            "interval_z": INTERVAL_Z,
            "claimed_coverage": info.get("claimed_coverage"),
            "shadow_log": score_shadow_log(SHADOW_LOG_PATH, frame, symbol),
        }
    )
    return scored
