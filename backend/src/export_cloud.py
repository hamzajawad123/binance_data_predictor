"""Build a small Streamlit Cloud bundle: recent parquet + serving model.

Writes to cloud/ so GitHub can host a demo without the full history.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from backend.src import CLOUD_DIR, FEATURES_PATH, MODELS_DIR, RAW_PATH, ROOT, TIMESTAMP_COLUMN


def export_cloud(days: int = 90, dest: Path | None = None) -> Path:
    dest = dest or CLOUD_DIR
    dest.mkdir(parents=True, exist_ok=True)
    cutoff = None

    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Missing {FEATURES_PATH}. Run feature engineering first.")
    feats = pd.read_parquet(FEATURES_PATH)
    feats[TIMESTAMP_COLUMN] = pd.to_datetime(feats[TIMESTAMP_COLUMN], utc=True)
    cutoff = feats[TIMESTAMP_COLUMN].max() - pd.Timedelta(days=days)
    feats = feats.loc[feats[TIMESTAMP_COLUMN] >= cutoff]
    feats.to_parquet(dest / "features.parquet", index=False)

    if not RAW_PATH.exists():
        raise RuntimeError(f"Missing {RAW_PATH}. Run data collection first.")
    raw = pd.read_parquet(RAW_PATH)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    raw = raw.loc[raw["open_time"] >= cutoff]
    raw.to_parquet(dest / "raw.parquet", index=False)

    model_src = MODELS_DIR / "model.joblib"
    dest_model = dest / "model.joblib"
    if not model_src.exists():
        raise RuntimeError(f"Missing {model_src}.")
    if model_src.resolve() != dest_model.resolve():
        shutil.copy2(model_src, dest_model)

    cols_src = MODELS_DIR / "feature_columns.json"
    if cols_src.exists():
        shutil.copy2(cols_src, dest / "feature_columns.json")

    wf_src = MODELS_DIR / "walkforward_metrics.json"
    if wf_src.exists():
        data = json.loads(wf_src.read_text(encoding="utf-8"))
        slim = {
            "error_distribution": data.get("error_distribution"),
            "gate": data.get("gate"),
            "n_folds": data.get("n_folds"),
        }
        (dest / "walkforward_metrics.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    (dest / "README.md").write_text(
        "\n".join(
            [
                "# Streamlit Cloud bundle",
                "",
                f"Last **{days}** days of hourly features/candles plus the serving `model.joblib`.",
                "Regenerate with `python -m backend.src.export_cloud`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {dest}  features={len(feats):,}  raw={len(raw):,}  cutoff={cutoff}")
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a slim Streamlit Cloud data/model bundle.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest) if args.dest else CLOUD_DIR
    if not dest.is_absolute():
        dest = ROOT / dest
    export_cloud(days=args.days, dest=dest)


if __name__ == "__main__":
    main()
