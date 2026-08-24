"""API contract tests using a tiny synthetic feature table and dummy model."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from backend.src import FEATURE_COLUMNS, SYMBOLS, TARGET_COLUMN, TIMESTAMP_COLUMN


def _write_fixture(tmp_path):
    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    for symbol_id, symbol in enumerate(SYMBOLS):
        close = 100 + np.arange(n)
        row = {
            "symbol": symbol,
            TIMESTAMP_COLUMN: idx,
            "open_time": idx,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
            TARGET_COLUMN: 0.012,
        }
        for col in FEATURE_COLUMNS:
            if col == "hour":
                row[col] = idx.hour
            elif col == "dow":
                row[col] = idx.dayofweek
            elif col == "symbol_id":
                row[col] = symbol_id
            else:
                row[col] = 0.01
        frame = pd.DataFrame(row)
        rows.append(frame)
    features = pd.concat(rows, ignore_index=True)
    raw = features[
        ["symbol", "open_time", "open", "high", "low", "close", "volume"]
    ].copy()
    raw["close_time"] = raw["open_time"]
    raw["quote_volume"] = raw["volume"]
    raw["trades"] = 1
    raw["taker_buy_base"] = 1.0
    raw["taker_buy_quote"] = 1.0
    features_path = tmp_path / "features.parquet"
    raw_path = tmp_path / "raw.parquet"
    features.to_parquet(features_path, index=False)
    raw.to_parquet(raw_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = DummyRegressor(strategy="constant", constant=0.015)
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COLUMN])
    joblib.dump(model, models_dir / "model.joblib")
    (models_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    return raw_path, features_path, models_dir


def test_health_and_predict(tmp_path, monkeypatch):
    raw_path, features_path, models_dir = _write_fixture(tmp_path)
    monkeypatch.setattr("backend.app.main.RAW_PATH", raw_path)
    monkeypatch.setattr("backend.app.main.FEATURES_PATH", features_path)
    monkeypatch.setattr("backend.app.main.MODELS_DIR", models_dir)
    monkeypatch.setattr("backend.app.main.SHADOW_LOG_PATH", tmp_path / "shadow.jsonl")
    monkeypatch.setattr("backend.app.main._model", None)
    monkeypatch.setattr("backend.app.main._model_info", None)
    monkeypatch.setattr("backend.app.main._feast_row", lambda symbol: None)

    from backend.app.main import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "BTCUSDT" in body["symbols"]

    bad = client.post("/predict", json={"symbol": "DOGEUSDT"})
    assert bad.status_code == 400

    pred = client.post("/predict", json={"symbol": "BTCUSDT"})
    assert pred.status_code == 200
    payload = pred.json()
    assert payload["symbol"] == "BTCUSDT"
    assert isinstance(payload["predicted_vol_24h"], float)
    assert "features" in payload

    features = client.get("/features", params={"symbol": "ETHUSDT"})
    assert features.status_code == 200
    assert features.json()["symbol"] == "ETHUSDT"

    all_preds = client.get("/predict/all")
    assert all_preds.status_code == 200
    assert len(all_preds.json()["predictions"]) == len(SYMBOLS)

    candles = client.get("/candles", params={"symbol": "BTCUSDT", "limit": 48})
    assert candles.status_code == 200
    assert len(candles.json()["candles"]) > 0

    assert payload["stale"] is True
    assert "predicted_vol_24h_low" in payload
    assert payload["predicted_vol_24h_low"] <= payload["predicted_vol_24h"] <= payload["predicted_vol_24h_high"]
    assert payload.get("model_version")

    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["loaded"] is True

    scored = client.get("/score", params={"symbol": "BTCUSDT", "limit": 48})
    assert scored.status_code == 200
    body = scored.json()
    assert body["symbol"] == "BTCUSDT"
    assert len(body["points"]) >= 24
    assert body["metrics"]["mae"] >= 0
    assert 0.0 <= body["metrics"]["coverage_95"] <= 1.0


def test_freshness_and_band():
    from backend.src.shadow import band, freshness

    lo, hi = band(0.01, 0.002)
    assert lo < 0.01 < hi
    assert band(-0.01, 0.001)[0] == 0.0
    fresh = freshness("2099-01-01T00:00:00+00:00")
    assert fresh["stale"] is True
    assert fresh["reason"] == "future_timestamp"
    old = freshness("2020-01-01T00:00:00+00:00")
    assert old["stale"] is True
