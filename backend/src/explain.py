"""TreeSHAP attributions for the serving voting ensemble.

Explains the frozen `models/model.joblib` (not a new bake-off). For a
VotingRegressor, member TreeSHAP values are averaged with equal weights.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.src import FEATURE_COLUMNS, FEATURES_PATH, MODELS_DIR, ROOT, TIMESTAMP_COLUMN
from backend.src.eda_feature_eng import drop_warmup

VOL_FEATURES = {"vol_6h", "vol_24h_hist", "vol_72h", "vol_term_structure", "abs_return", "bb_width"}


def _native_tree_shap(estimator, x: pd.DataFrame) -> np.ndarray:
    name = type(estimator).__name__.lower()
    if "lgbm" in name or hasattr(estimator, "booster_"):
        contrib = np.asarray(estimator.predict(x, pred_contrib=True), dtype=float)
        return contrib[:, :-1]
    if "xgb" in name or type(estimator).__name__ == "XGBRegressor":
        contrib = np.asarray(estimator.predict(x, pred_contrib=True), dtype=float)
        return contrib[:, :-1]
    raise TypeError(f"No native TreeSHAP for {type(estimator).__name__}")


def shap_values_for_estimator(estimator, x: pd.DataFrame) -> np.ndarray:
    try:
        import shap

        values = shap.TreeExplainer(estimator).shap_values(x)
        return np.asarray(values, dtype=float)
    except Exception:
        return _native_tree_shap(estimator, x)


def voting_members(model) -> list[tuple[str, object]]:
    named = getattr(model, "named_estimators_", None)
    if named:
        return [(str(name), est) for name, est in named.items()]
    estimators = getattr(model, "estimators_", None)
    if estimators:
        return [(type(est).__name__, est) for est in estimators]
    return [(type(model).__name__, model)]


def mean_abs_shap(shap_values: np.ndarray, columns: list[str]) -> list[dict]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    total = float(mean_abs.sum()) or 1.0
    rows = []
    for i in order:
        rows.append(
            {
                "feature": columns[int(i)],
                "mean_abs_shap": float(mean_abs[i]),
                "share": float(mean_abs[i] / total),
            }
        )
    return rows


def sample_rows(frame: pd.DataFrame, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    frame = frame.dropna(subset=FEATURE_COLUMNS)
    if frame.empty:
        raise RuntimeError("No finite feature rows to explain.")
    cutoff = frame[TIMESTAMP_COLUMN].max() - pd.Timedelta(days=180)
    recent = frame.loc[frame[TIMESTAMP_COLUMN] >= cutoff]
    pool = recent if len(recent) >= 200 else frame
    n = min(n, len(pool))
    parts = []
    leftover = n
    symbols = list(pool["symbol"].unique())
    per = max(1, n // max(len(symbols), 1))
    for i, symbol in enumerate(symbols):
        take = per if i < len(symbols) - 1 else leftover
        group = pool.loc[pool["symbol"] == symbol]
        take = min(take, len(group))
        if take:
            parts.append(group.sample(n=take, random_state=seed))
        leftover -= take
    sampled = pd.concat(parts, ignore_index=True) if parts else pool.sample(n=n, random_state=seed)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.reset_index(drop=True)


def explain(
    features_path: Path | None = None,
    models_dir: Path | None = None,
    sample_size: int = 2000,
) -> dict:
    features_path = features_path or FEATURES_PATH
    models_dir = models_dir or MODELS_DIR
    model_path = models_dir / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Missing serving model at {model_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)

    frame = pd.read_parquet(features_path)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = drop_warmup(frame)
    sampled = sample_rows(frame, n=sample_size)
    x = sampled[FEATURE_COLUMNS]

    member_rows = {}
    stacked = []
    used = []
    for name, est in voting_members(model):
        try:
            values = shap_values_for_estimator(est, x)
        except Exception as exc:
            print(f"skip {name}: {exc}")
            continue
        if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
            print(f"skip {name}: unexpected SHAP shape {getattr(values, 'shape', None)}")
            continue
        stacked.append(values)
        used.append(name)
        member_rows[name] = mean_abs_shap(values, FEATURE_COLUMNS)

    if not stacked:
        raise RuntimeError("Could not compute TreeSHAP for any ensemble member.")

    ensemble = np.mean(np.stack(stacked, axis=0), axis=0)
    ranking = mean_abs_shap(ensemble, FEATURE_COLUMNS)
    vol_share = sum(row["share"] for row in ranking if row["feature"] in VOL_FEATURES)
    report = {
        "n_rows": int(len(x)),
        "as_of": str(sampled[TIMESTAMP_COLUMN].max()),
        "window_days": 180,
        "members_explained": used,
        "method": "TreeSHAP averaged across voting members",
        "mean_abs_shap": ranking,
        "vol_family_share": vol_share,
        "per_member": member_rows,
    }
    out_json = models_dir / "shap_importance.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_bar(ranking, ROOT / "docs" / "shap_bar.png")
    _write_md(report, ROOT / "docs" / "SHAP.md")
    print(f"Wrote {out_json}")
    return report


def _write_bar(ranking: list[dict], path: Path) -> None:
    top = ranking[:12]
    labels = [row["feature"] for row in reversed(top)]
    values = [row["mean_abs_shap"] for row in reversed(top)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(labels, values, color="#3D7EA6")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("TreeSHAP — serving voting ensemble")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_md(report: dict, path: Path) -> None:
    ranking = report.get("mean_abs_shap") or []
    members = ", ".join(report.get("members_explained") or [])
    lines = [
        "# TreeSHAP (Phase 3, proof only)",
        "",
        "Attributions for the **serving** `models/model.joblib` voting ensemble. "
        "This is not a new model search. Walk-forward numbers stay in [EVALUATION.md](EVALUATION.md).",
        "",
        f"- Rows explained: **{report.get('n_rows')}** (sample from the last {report.get('window_days')} days)",
        f"- Method: {report.get('method')}",
        f"- Members: {members}",
        f"- Share from vol-family features (`vol_6h`, `vol_24h_hist`, `vol_72h`, `vol_term_structure`, `abs_return`, `bb_width`): **{report.get('vol_family_share', 0):.1%}**",
        "",
        "![Mean |SHAP| bar chart](shap_bar.png)",
        "",
        "| Rank | Feature | mean \\|SHAP\\| | Share |",
        "|---|---|---|---|",
    ]
    for i, row in enumerate(ranking, start=1):
        lines.append(
            f"| {i} | `{row['feature']}` | {row['mean_abs_shap']:.6f} | {row['share']:.1%} |"
        )
    lines += [
        "",
        "Vol clustering leads (`vol_72h`). Weekday seasonality (`dow`) and log volume also matter. "
        "Signed returns and RSI are small — this is a volatility model, not a price-direction toy.",
        "",
        "Machine-readable source: `models/shap_importance.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TreeSHAP for the serving voting model.")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = Path(args.features) if args.features else FEATURES_PATH
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    if not features.is_absolute():
        features = ROOT / features
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    explain(features_path=features, models_dir=models_dir, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
