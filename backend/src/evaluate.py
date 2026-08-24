"""Holdout metrics: RMSE, MAE, and directional accuracy of volatility changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.src import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    HOLDOUT_DAYS,
    MODELS_DIR,
    ROOT,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from backend.src.eda_feature_eng import drop_warmup


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of hours where the predicted vol change matches the actual vol change."""
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    mask = true_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float((true_dir[mask] == pred_dir[mask]).mean())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "correlation": corr,
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def time_split(frame: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame[TIMESTAMP_COLUMN].max() - pd.Timedelta(days=holdout_days)
    train_df = frame.loc[frame[TIMESTAMP_COLUMN] < cutoff].copy()
    test_df = frame.loc[frame[TIMESTAMP_COLUMN] >= cutoff].copy()
    return train_df, test_df


def per_symbol_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    scored = frame.copy()
    scored["_y_true"] = y_true
    scored["_y_pred"] = y_pred
    for symbol, group in scored.groupby("symbol"):
        out[str(symbol)] = compute_metrics(group["_y_true"].to_numpy(), group["_y_pred"].to_numpy())
    return out


def load_model_bundle(models_dir: Path | None = None) -> tuple[object, list[str]]:
    models_dir = models_dir or MODELS_DIR
    model_path = models_dir / "model.joblib"
    columns_path = models_dir / "feature_columns.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run `python -m backend.src.train` first.")
    model = joblib.load(model_path)
    columns = FEATURE_COLUMNS
    if columns_path.exists():
        columns = json.loads(columns_path.read_text(encoding="utf-8"))
    return model, columns


def evaluate(
    features_path: Path | None = None,
    models_dir: Path | None = None,
    holdout_days: int = HOLDOUT_DAYS,
) -> dict:
    features_path = features_path or FEATURES_PATH
    models_dir = models_dir or MODELS_DIR
    frame = pd.read_parquet(features_path)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = drop_warmup(frame)
    _, test_df = time_split(frame, holdout_days=holdout_days)
    if test_df.empty:
        raise RuntimeError("Holdout split is empty. Collect more history or lower HOLDOUT_DAYS.")

    model, columns = load_model_bundle(models_dir)
    x_test = test_df[columns]
    y_true = test_df[TARGET_COLUMN].to_numpy()
    y_pred = np.asarray(model.predict(x_test), dtype=float)
    overall = compute_metrics(y_true, y_pred)
    by_symbol = per_symbol_metrics(test_df, y_true, y_pred)

    report = {
        "n_holdout": int(len(test_df)),
        "holdout_days": holdout_days,
        "overall": overall,
        "per_symbol": by_symbol,
    }
    metrics_path = models_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Holdout evaluation ===")
    print(
        f"n={report['n_holdout']}  RMSE={overall['rmse']:.6f}  "
        f"MAE={overall['mae']:.6f}  DirAcc={overall['directional_accuracy']:.3f}"
    )
    for symbol, stats in by_symbol.items():
        print(
            f"  {symbol}: RMSE={stats['rmse']:.6f} MAE={stats['mae']:.6f} "
            f"DirAcc={stats['directional_accuracy']:.3f}"
        )
    print(f"Wrote {metrics_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved volatility model on the time holdout.")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = Path(args.features) if args.features else FEATURES_PATH
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    if not features.is_absolute():
        features = ROOT / features
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    evaluate(features_path=features, models_dir=models_dir)


if __name__ == "__main__":
    main()
