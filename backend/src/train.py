"""Train LightGBM and XGBoost, log to MLflow, and persist the better model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from backend.src import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    HOLDOUT_DAYS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODELS_DIR,
    ROOT,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from backend.src.eda_feature_eng import drop_warmup
from backend.src.evaluate import compute_metrics, per_symbol_metrics, time_split


def _load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    return drop_warmup(frame)


def _fit_models(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    lgbm = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    xgbr = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    lgbm.fit(x_train, y_train)
    xgbr.fit(x_train, y_train)
    return {"lightgbm": lgbm, "xgboost": xgbr}


def train(
    features_path: Path | None = None,
    models_dir: Path | None = None,
    holdout_days: int = HOLDOUT_DAYS,
    register: bool = True,
) -> dict:
    features_path = features_path or FEATURES_PATH
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_features(features_path)
    train_df, test_df = time_split(frame, holdout_days=holdout_days)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Need enough history for a 60-day holdout. Collect more data.")

    x_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    x_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    fitted = _fit_models(x_train, y_train)
    leaderboard: dict[str, dict] = {}
    artifact_paths = {}

    for name, model in fitted.items():
        y_pred = model.predict(x_test)
        metrics = compute_metrics(y_test.to_numpy(), y_pred)
        by_symbol = per_symbol_metrics(test_df, y_test.to_numpy(), y_pred)
        leaderboard[name] = {"overall": metrics, "per_symbol": by_symbol}
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        artifact_paths[name] = str(model_path)

        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model": name,
                    "horizon": "24h",
                    "interval": "1h",
                    "symbols": ",".join(sorted(frame["symbol"].unique())),
                    "n_estimators": 300,
                    "holdout_days": holdout_days,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                }
            )
            mlflow.log_metrics(metrics)
            for symbol, stats in by_symbol.items():
                mlflow.log_metrics({f"{symbol.lower()}_{k}": v for k, v in stats.items()})
            if name == "lightgbm":
                mlflow.lightgbm.log_model(model, artifact_path="model")
            else:
                mlflow.xgboost.log_model(model, artifact_path="model")

    winner_name = min(leaderboard, key=lambda n: leaderboard[n]["overall"]["rmse"])
    winner = fitted[winner_name]
    joblib.dump(winner, models_dir / "model.joblib")
    (models_dir / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8"
    )
    summary = {
        "winner": winner_name,
        "leaderboard": leaderboard,
        "n_train": int(len(train_df)),
        "n_holdout": int(len(test_df)),
    }
    (models_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if register:
        try:
            with mlflow.start_run(run_name=f"register-{winner_name}"):
                mlflow.log_params(
                    {
                        "winner": winner_name,
                        "horizon": "24h",
                        "interval": "1h",
                        "model_name": MODEL_NAME,
                    }
                )
                mlflow.log_metrics(leaderboard[winner_name]["overall"])
                if winner_name == "lightgbm":
                    model_info = mlflow.lightgbm.log_model(
                        winner,
                        artifact_path="model",
                        registered_model_name=MODEL_NAME,
                    )
                else:
                    model_info = mlflow.xgboost.log_model(
                        winner,
                        artifact_path="model",
                        registered_model_name=MODEL_NAME,
                    )
                print(f"Registered {MODEL_NAME} from {model_info.model_uri}")
        except Exception as exc:  # registry is nice-to-have; joblib is the serving source of truth
            print(f"MLflow registry skipped: {exc}")

    print(f"Winner={winner_name} RMSE={leaderboard[winner_name]['overall']['rmse']:.6f}")
    print(f"Saved serving bundle -> {models_dir / 'model.joblib'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train volatility models and log them to MLflow.")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = Path(args.features) if args.features else FEATURES_PATH
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    if not features.is_absolute():
        features = ROOT / features
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    train(features_path=features, models_dir=models_dir, register=not args.no_register)


if __name__ == "__main__":
    main()
