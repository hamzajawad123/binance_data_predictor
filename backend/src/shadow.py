"""Phase 4 helpers: model fingerprint, freshness, residual interval, predicted vs actual."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.src import FEATURE_COLUMNS, ROOT, TARGET_COLUMN, TIMESTAMP_COLUMN
from backend.src.evaluate import compute_metrics

STALE_HOURS = 2.5
INTERVAL_Z = 1.96
FALLBACK_SIGMA = 0.00365
SHADOW_LOG_PATH = ROOT / "data" / "shadow_forecasts.jsonl"


def git_commit(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo or ROOT),
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def artifact_fingerprint(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        return {"loaded": False, "path": str(model_path)}
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
    return {
        "loaded": True,
        "path": str(model_path),
        "sha256_12": digest,
        "modified_utc": mtime.isoformat(),
        "version": f"joblib-{digest}",
    }


def estimator_names(model: object | None) -> list[str]:
    if model is None:
        return []
    named = getattr(model, "named_estimators_", None)
    if named:
        return [str(name) for name in named]
    return [type(model).__name__]


def interval_sigma(models_dir: Path) -> tuple[float, str]:
    path = models_dir / "walkforward_metrics.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sigma = data.get("error_distribution", {}).get("std_error")
            if sigma is not None and float(sigma) > 0:
                return float(sigma), "walkforward_residual_std"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return FALLBACK_SIGMA, "fallback_voting_rmse"


def band(predicted: float, sigma: float, z: float = INTERVAL_Z) -> tuple[float, float]:
    half = z * sigma
    lo = max(0.0, float(predicted) - half)
    hi = float(predicted) + half
    return lo, hi


def freshness(event_timestamp: Any, now: datetime | None = None, stale_hours: float = STALE_HOURS) -> dict[str, Any]:
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    if event_timestamp is None or event_timestamp == "":
        return {"stale": True, "hours_lag": None, "reason": "missing_timestamp"}
    ts = pd.to_datetime(event_timestamp, utc=True)
    lag_hours = float((now_ts - ts).total_seconds() / 3600.0)
    if lag_hours < -1.0:
        return {"stale": True, "hours_lag": round(lag_hours, 2), "reason": "future_timestamp"}
    if lag_hours > stale_hours:
        return {"stale": True, "hours_lag": round(lag_hours, 2), "reason": "older_than_stale_hours"}
    return {"stale": False, "hours_lag": round(lag_hours, 2), "reason": None}


def append_shadow_forecast(record: dict[str, Any], path: Path | None = None) -> None:
    log_path = path or SHADOW_LOG_PATH
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        return


def score_artifact(
    frame: pd.DataFrame,
    model: object | None,
    columns: list[str],
    symbol: str,
    limit: int,
    sigma: float,
    z: float = INTERVAL_Z,
) -> dict[str, Any]:
    subset = frame.loc[frame["symbol"] == symbol].sort_values(TIMESTAMP_COLUMN)
    if subset.empty:
        raise ValueError(f"No features for {symbol}")
    needed = [c for c in columns if c in subset.columns]
    usable = subset.dropna(subset=needed).tail(limit).copy()
    if usable.empty:
        raise ValueError(f"No finite feature rows for {symbol}")

    x = usable[needed]
    if model is None:
        predicted = usable["vol_24h_hist"].to_numpy(dtype=float) if "vol_24h_hist" in usable.columns else np.full(len(usable), np.nan)
        source = "baseline_persistence"
    else:
        predicted = np.asarray(model.predict(x), dtype=float)
        source = "model"

    actual = (
        usable[TARGET_COLUMN].to_numpy(dtype=float)
        if TARGET_COLUMN in usable.columns
        else np.full(len(usable), np.nan)
    )
    pending = ~np.isfinite(actual)
    lo = np.maximum(0.0, predicted - z * sigma)
    hi = predicted + z * sigma

    settled = np.isfinite(actual) & np.isfinite(predicted)
    metrics = compute_metrics(actual[settled], predicted[settled]) if settled.sum() >= 2 else {}
    coverage = None
    if settled.sum():
        inside = (actual[settled] >= lo[settled]) & (actual[settled] <= hi[settled])
        coverage = float(inside.mean())
        metrics["coverage_95"] = coverage
        metrics["n_settled"] = int(settled.sum())

    points = []
    times = usable[TIMESTAMP_COLUMN].tolist()
    for i, ts in enumerate(times):
        ts_val = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        points.append(
            {
                "time": ts_val,
                "predicted": _finite_or_none(predicted[i]),
                "actual": None if pending[i] else _finite_or_none(actual[i]),
                "lo": _finite_or_none(lo[i]),
                "hi": _finite_or_none(hi[i]),
                "pending": bool(pending[i]),
            }
        )

    as_of = points[-1]["time"] if points else None
    return {
        "symbol": symbol,
        "prediction_source": source,
        "n_points": len(points),
        "n_pending": int(pending.sum()),
        "as_of": as_of,
        "metrics": metrics,
        "points": points,
    }


def score_shadow_log(path: Path, features: pd.DataFrame, symbol: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("symbol") == symbol:
                    rows.append(rec)
    except (OSError, json.JSONDecodeError):
        return None
    if not rows:
        return None

    feats = features.loc[features["symbol"] == symbol, [TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    feats[TIMESTAMP_COLUMN] = pd.to_datetime(feats[TIMESTAMP_COLUMN], utc=True)
    lookup = {
        pd.Timestamp(ts).isoformat(): val
        for ts, val in zip(feats[TIMESTAMP_COLUMN], feats[TARGET_COLUMN], strict=False)
    }
    preds, actuals = [], []
    for rec in rows:
        key = rec.get("event_timestamp")
        if not key:
            continue
        ts = pd.to_datetime(key, utc=True).isoformat()
        actual = lookup.get(ts)
        pred = rec.get("predicted_vol_24h")
        if actual is None or not np.isfinite(float(actual)) or pred is None:
            continue
        preds.append(float(pred))
        actuals.append(float(actual))
    if len(preds) < 2:
        return {"n_logged": len(rows), "n_settled": len(preds)}
    metrics = compute_metrics(np.asarray(actuals), np.asarray(preds))
    metrics["n_logged"] = len(rows)
    return metrics


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
