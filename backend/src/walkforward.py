"""Walk-forward evaluation of the production voting family vs volatility baselines.

Does not re-run Optuna or the old linear/RF/stacking bake-off.
Each fold refits LightGBM + XGBoost voting on the train window only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

from backend.src import FEATURE_COLUMNS, FEATURES_PATH, MODELS_DIR, ROOT, TARGET_COLUMN, TIMESTAMP_COLUMN
from backend.src.eda_feature_eng import drop_warmup
from backend.src.evaluate import compute_metrics

HAR_FEATURES = ["vol_6h", "vol_24h_hist", "vol_72h"]


def _voting_model() -> VotingRegressor:
    lgbm = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    xgbr = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    return VotingRegressor([("lightgbm", lgbm), ("xgboost", xgbr)])


def _garch_constant(train_returns: np.ndarray, test_len: int) -> np.ndarray:
    """GARCH(1,1) 24h-vol forecast from the end of train, held constant on the test fold."""
    try:
        from arch import arch_model
    except ImportError:
        return np.full(test_len, np.nan)
    series = np.asarray(train_returns, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 200:
        return np.full(test_len, np.nan)
    scaled = series * 100.0
    try:
        fitted = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
        forecasts = fitted.forecast(horizon=24, reindex=False)
        var = forecasts.variance.values[-1]
        hourly_std = float(np.sqrt(np.mean(var))) / 100.0
    except Exception:
        return np.full(test_len, np.nan)
    return np.full(test_len, hourly_std)


def _pairwise_dm(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, lag: int = 24) -> dict[str, float]:
    """DM on rows where both forecasts are finite (EWMA/GARCH can drop NaNs)."""
    mask = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    return _diebold_mariano(y_true[mask] - pred_a[mask], y_true[mask] - pred_b[mask], lag=lag)


def _diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, lag: int = 24) -> dict[str, float]:
    """DM test on squared errors. Positive mean_d means A is worse than B (A has larger errors)."""
    d = np.asarray(err_a, dtype=float) ** 2 - np.asarray(err_b, dtype=float) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return {"n": n, "mean_d": float("nan"), "statistic": float("nan"), "p_value": float("nan")}
    mean_d = float(np.mean(d))
    lag = max(1, min(lag, n - 1))
    gamma0 = float(np.var(d, ddof=1))
    gamma = 0.0
    centered = d - mean_d
    for k in range(1, lag + 1):
        cov = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1)
        gamma += 2.0 * weight * cov
    var = (gamma0 + gamma) / n
    if var <= 0:
        return {"n": n, "mean_d": mean_d, "statistic": float("nan"), "p_value": float("nan")}
    stat = mean_d / np.sqrt(var)
    from math import erfc

    p_value = float(erfc(abs(stat) / np.sqrt(2.0)))
    return {"n": n, "mean_d": mean_d, "statistic": float(stat), "p_value": p_value}


def _direction_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    true_dir = np.sign(np.diff(np.asarray(y_true, dtype=float)))
    pred_dir = np.sign(np.diff(np.asarray(y_pred, dtype=float)))
    mask = true_dir != 0
    n = int(mask.sum())
    k = int((true_dir[mask] == pred_dir[mask]).sum()) if n else 0
    return k, n


def _binom_two_sided_p(k: int, n: int, p0: float = 0.5) -> float:
    if n < 30:
        return float("nan")
    mean = n * p0
    sd = np.sqrt(n * p0 * (1.0 - p0))
    if sd == 0:
        return float("nan")
    z = (k - mean) / sd
    from math import erfc

    return float(erfc(abs(z) / np.sqrt(2.0)))


def _error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 10:
        return {}
    centered = resid - np.mean(resid)
    std = float(np.std(resid, ddof=1))
    skew = float(np.mean(centered**3) / std**3) if std else float("nan")
    kurt = float(np.mean(centered**4) / std**4 - 3.0) if std else float("nan")
    return {
        "n": int(len(resid)),
        "mean_error": float(np.mean(resid)),
        "std_error": std,
        "skew": skew,
        "excess_kurtosis": kurt,
        "pct_overpredict": float(np.mean(resid > 0)),
        "p50_abs_error": float(np.median(np.abs(resid))),
        "p95_abs_error": float(np.percentile(np.abs(resid), 95)),
    }


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 400, seed: int = 42) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ("rmse", "mae", "r2", "correlation")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        for key in keys:
            val = metrics[key]
            if np.isfinite(val):
                draws[key].append(float(val))
    out: dict[str, dict] = {}
    for key, values in draws.items():
        if len(values) < 20:
            out[key] = {"low": float("nan"), "high": float("nan")}
            continue
        out[key] = {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
    return out


def _regime_masks(y_true: np.ndarray) -> dict[str, np.ndarray]:
    q25, q75, q90 = np.nanpercentile(y_true, [25, 75, 90])
    return {
        "low": y_true < q25,
        "normal": (y_true >= q25) & (y_true < q75),
        "high": (y_true >= q75) & (y_true < q90),
        "spike": y_true >= q90,
    }


def _fold_bounds(index: pd.DatetimeIndex, test_days: int, min_train_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = index.min()
    end = index.max()
    first_test = start + pd.Timedelta(days=min_train_days)
    folds = []
    cursor = first_test
    step = pd.Timedelta(days=test_days)
    while cursor + step <= end:
        train_end = cursor
        test_end = cursor + step
        folds.append((start, train_end, test_end))
        cursor = test_end
    return folds


def walkforward(
    features_path: Path | None = None,
    models_dir: Path | None = None,
    test_days: int = 90,
    min_train_days: int = 365,
) -> dict:
    features_path = features_path or FEATURES_PATH
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(features_path)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = drop_warmup(frame).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No trainable rows. Run feature engineering first.")

    global_index = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN].unique()).sort_values()
    folds = _fold_bounds(global_index, test_days=test_days, min_train_days=min_train_days)
    if not folds:
        raise RuntimeError("Not enough history for walk-forward. Pull full Binance history first.")

    method_names = ["voting", "persistence", "vol_72h", "ewma_24", "har_rv", "garch11"]
    collected: dict[str, dict[str, list]] = {name: {"y_true": [], "y_pred": [], "symbol": []} for name in method_names}
    fold_summaries = []

    for fold_i, (_hist_start, train_end, test_end) in enumerate(folds, start=1):
        train_df = frame.loc[frame[TIMESTAMP_COLUMN] < train_end]
        test_df = frame.loc[(frame[TIMESTAMP_COLUMN] >= train_end) & (frame[TIMESTAMP_COLUMN] < test_end)]
        if len(train_df) < 2000 or len(test_df) < 200:
            continue
        y_true_all = test_df[TARGET_COLUMN].to_numpy(dtype=float)
        finite_y = np.isfinite(y_true_all)
        y_true = y_true_all[finite_y]
        symbols = test_df["symbol"].to_numpy()[finite_y]
        preds: dict[str, np.ndarray] = {}

        model = _voting_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        preds["voting"] = np.asarray(model.predict(test_df[FEATURE_COLUMNS]), dtype=float)
        preds["persistence"] = test_df["vol_24h_hist"].to_numpy(dtype=float)
        preds["vol_72h"] = test_df["vol_72h"].to_numpy(dtype=float)

        ewma_rows = []
        for symbol, tes in test_df.groupby("symbol", sort=False):
            tr = train_df.loc[train_df["symbol"] == symbol]
            hist = frame.loc[frame["symbol"] == symbol, [TIMESTAMP_COLUMN, "log_return"]].sort_values(TIMESTAMP_COLUMN)
            ewma_map = hist.set_index(TIMESTAMP_COLUMN)["log_return"].ewm(span=24, adjust=False).std()
            part = tes[[TIMESTAMP_COLUMN, "symbol"]].copy()
            part["_ewma"] = ewma_map.reindex(tes[TIMESTAMP_COLUMN]).to_numpy(dtype=float)
            part["_garch"] = _garch_constant(tr["log_return"].to_numpy(dtype=float), len(tes))
            ewma_rows.append(part)
        extra = pd.concat(ewma_rows, ignore_index=True)
        merged = test_df.merge(extra, on=[TIMESTAMP_COLUMN, "symbol"], how="left")
        preds["ewma_24"] = merged["_ewma"].to_numpy(dtype=float)
        preds["garch11"] = merged["_garch"].to_numpy(dtype=float)

        har = LinearRegression()
        har.fit(train_df[HAR_FEATURES], train_df[TARGET_COLUMN])
        preds["har_rv"] = np.asarray(har.predict(test_df[HAR_FEATURES]), dtype=float)

        fold_row = {
            "fold": fold_i,
            "train_end": train_end.isoformat(),
            "test_end": test_end.isoformat(),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "methods": {},
        }
        for name, y_pred in preds.items():
            y_pred = np.asarray(y_pred, dtype=float)[finite_y]
            score_mask = np.isfinite(y_pred)
            fold_row["methods"][name] = compute_metrics(y_true[score_mask], y_pred[score_mask])
            collected[name]["y_true"].append(y_true)
            collected[name]["y_pred"].append(y_pred)
            collected[name]["symbol"].append(symbols)
        fold_summaries.append(fold_row)
        print(
            f"Fold {fold_i}/{len(folds)} train_end={train_end.date()} n_test={len(test_df)} "
            f"voting_rmse={fold_row['methods']['voting']['rmse']:.5f} "
            f"persist_rmse={fold_row['methods']['persistence']['rmse']:.5f}"
        )

    overall = {}
    y_all = np.concatenate(collected["voting"]["y_true"]) if collected["voting"]["y_true"] else np.array([])
    pred_all = np.concatenate(collected["voting"]["y_pred"]) if collected["voting"]["y_pred"] else np.array([])
    vote_ok = np.isfinite(pred_all) if len(pred_all) else np.array([], dtype=bool)
    y_vote = y_all[vote_ok]
    pred_vote = pred_all[vote_ok]

    for name, store in collected.items():
        if not store["y_true"]:
            continue
        y_t = np.concatenate(store["y_true"])
        y_p = np.concatenate(store["y_pred"])
        score_mask = np.isfinite(y_t) & np.isfinite(y_p)
        overall[name] = compute_metrics(y_t[score_mask], y_p[score_mask])
        scored = pd.DataFrame(
            {
                "symbol": np.concatenate(store["symbol"])[score_mask],
                "_y_true": y_t[score_mask],
                "_y_pred": y_p[score_mask],
            }
        )
        overall[name]["per_symbol"] = {
            str(sym): compute_metrics(g["_y_true"].to_numpy(), g["_y_pred"].to_numpy())
            for sym, g in scored.groupby("symbol")
        }

    regimes = {}
    if len(y_vote):
        for label, mask in _regime_masks(y_vote).items():
            if mask.sum() < 50:
                continue
            regimes[label] = compute_metrics(y_vote[mask], pred_vote[mask])

    dm = {}
    if len(y_all) and collected["persistence"]["y_pred"]:
        persist = np.concatenate(collected["persistence"]["y_pred"])
        har_p = np.concatenate(collected["har_rv"]["y_pred"]) if collected["har_rv"]["y_pred"] else None
        dm["voting_vs_persistence"] = _pairwise_dm(y_all, pred_all, persist)
        if har_p is not None:
            dm["voting_vs_har_rv"] = _pairwise_dm(y_all, pred_all, har_p)
        for other in ("ewma_24", "vol_72h", "garch11"):
            if not collected[other]["y_pred"]:
                continue
            dm[f"voting_vs_{other}"] = _pairwise_dm(y_all, pred_all, np.concatenate(collected[other]["y_pred"]))

        vote_sym = np.concatenate(collected["voting"]["symbol"])
        dm["per_symbol_vs_persistence"] = {}
        for symbol in sorted(set(vote_sym.tolist())):
            m = vote_sym == symbol
            if m.sum() < 50:
                continue
            dm["per_symbol_vs_persistence"][str(symbol)] = _pairwise_dm(y_all[m], pred_all[m], persist[m])

    ci = _bootstrap_ci(y_vote, pred_vote) if len(y_vote) else {}

    direction = {}
    if len(y_vote):
        vote_sym = np.concatenate(collected["voting"]["symbol"])
        direction["note"] = (
            "Directional accuracy is the sign of the change in volatility, not price. "
            "Pooled figures are misleading; use per-coin tests only."
        )
        direction["per_symbol"] = {}
        for symbol in sorted(set(vote_sym.tolist())):
            m = vote_sym == symbol
            k, n = _direction_counts(y_vote[m], pred_vote[m])
            acc = (k / n) if n else float("nan")
            direction["per_symbol"][str(symbol)] = {
                "hits": k,
                "n": n,
                "accuracy": acc,
                "p_value_vs_50pct": _binom_two_sided_p(k, n, 0.5),
                "better_than_chance": bool(acc > 0.5) if n else False,
            }

    errors = _error_distribution(y_vote, pred_vote) if len(y_vote) else {}

    gate = {
        "beats_persistence_rmse": bool(
            overall.get("voting", {}).get("rmse", 1) < overall.get("persistence", {}).get("rmse", 0)
        ),
        "beats_har_rv_rmse": bool(
            overall.get("voting", {}).get("rmse", 1) < overall.get("har_rv", {}).get("rmse", 0)
        ),
    }
    gate["pass"] = bool(gate["beats_persistence_rmse"] and gate["beats_har_rv_rmse"])

    report = {
        "n_folds": len(fold_summaries),
        "test_days": test_days,
        "min_train_days": min_train_days,
        "overall": overall,
        "regimes": regimes,
        "diebold_mariano": dm,
        "voting_bootstrap_95ci": ci,
        "directional_tests": direction,
        "error_distribution": errors,
        "gate": gate,
        "folds": fold_summaries,
    }
    out_path = models_dir / "walkforward_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_evaluation_md(report, ROOT / "docs" / "EVALUATION.md")
    print(f"Gate pass={gate['pass']}  wrote {out_path}")
    return report


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_p(value) -> str:
    if value is None:
        return "n/a"
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.3g}"


def write_evaluation_md(report: dict, path: Path) -> None:
    """Official Phase 2 write-up. Do not quote pooled directional accuracy as skill."""
    overall = report.get("overall", {})
    vote = overall.get("voting", {})
    persist = overall.get("persistence", {})
    har = overall.get("har_rv", {})
    ewma = overall.get("ewma_24", {})
    garch = overall.get("garch11", {})
    vol72 = overall.get("vol_72h", {})
    dm = report.get("diebold_mariano", {})
    ci = report.get("voting_bootstrap_95ci", {})
    direction = report.get("directional_tests", {})
    errors = report.get("error_distribution", {})
    regimes = report.get("regimes", {})
    gate = report.get("gate", {})

    def row(name: str, metrics: dict) -> str:
        if not metrics:
            return f"| {name} | n/a | n/a | n/a | n/a |"
        return (
            f"| {name} | {_fmt(metrics.get('rmse'))} | {_fmt(metrics.get('mae'))} | "
            f"{_fmt(metrics.get('r2'), 3)} | {_fmt(metrics.get('correlation'), 3)} |"
        )

    lines = [
        "# Official evaluation (Phase 2)",
        "",
        "This is the **only** public performance write-up. It replaces any older 60-day holdout headline and **must not** quote pooled ~80% directional accuracy as model skill.",
        "",
        f"- Walk-forward folds: **{report.get('n_folds')}**",
        f"- Min train / test window: **{report.get('min_train_days')}d / {report.get('test_days')}d**",
        f"- Target: `vol_24h` = std of the next 24 hourly log returns (not annualized, not price direction)",
        f"- Production candidate: LightGBM + XGBoost **voting** (refit each fold; not a new Optuna bake-off)",
        f"- Gate pass: **{gate.get('pass')}**",
        "",
        "## Overall walk-forward (pooled hours, all coins)",
        "",
        "| Model | RMSE | MAE | R² | Corr |",
        "|---|---|---|---|---|",
        row("Voting", vote),
        row("HAR-RV", har),
        row("EWMA-24", ewma),
        row("72h realized vol", vol72),
        row("Persistence", persist),
        row("GARCH(1,1)", garch),
        "",
        "## 95% bootstrap CI (voting)",
        "",
    ]
    for key in ("rmse", "mae", "r2", "correlation"):
        band = ci.get(key, {})
        lines.append(f"- **{key}:** {_fmt(vote.get(key), 4)}  [{_fmt(band.get('low'), 4)}, {_fmt(band.get('high'), 4)}]")
    lines += [
        "",
        "## Diebold–Mariano (squared errors)",
        "",
        "Negative mean_d and small p-value: voting has **smaller** errors than the comparator.",
        "",
    ]
    for key, label in (
        ("voting_vs_persistence", "Voting vs persistence"),
        ("voting_vs_har_rv", "Voting vs HAR-RV"),
        ("voting_vs_ewma_24", "Voting vs EWMA-24"),
        ("voting_vs_vol_72h", "Voting vs 72h vol"),
        ("voting_vs_garch11", "Voting vs GARCH(1,1)"),
    ):
        stats = dm.get(key, {})
        if not stats:
            continue
        n = stats.get("n")
        ntxt = f" (n={int(n)})" if n is not None else ""
        ptxt = _fmt_p(stats.get("p_value"))
        lines.append(
            f"- **{label}:** DM = {_fmt(stats.get('statistic'), 2)}, p = {ptxt}{ntxt}"
            if ptxt != "n/a"
            else f"- **{label}:** n/a{ntxt}"
        )
    lines += [
        "",
        "DM vs EWMA or GARCH uses the intersection of hours where both forecasts are finite.",
        "",
        "### Per coin vs persistence",
        "",
    ]
    for symbol, stats in (dm.get("per_symbol_vs_persistence") or {}).items():
        lines.append(
            f"- **{symbol}:** DM = {_fmt(stats.get('statistic'), 2)}, p = {_fmt_p(stats.get('p_value'))}"
        )

    lines += [
        "",
        "## Per-coin voting RMSE / MAE / R²",
        "",
        "| Coin | RMSE | MAE | R² | Corr | Dir. acc. (vol change) | vs 50% p-value |",
        "|---|---|---|---|---|---|---|",
    ]
    per = vote.get("per_symbol") or {}
    dir_map = (direction.get("per_symbol") or {})
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"):
        m = per.get(symbol, {})
        d = dir_map.get(symbol, {})
        ptxt = _fmt_p(d.get("p_value_vs_50pct"))
        lines.append(
            f"| {symbol} | {_fmt(m.get('rmse'))} | {_fmt(m.get('mae'))} | {_fmt(m.get('r2'), 3)} | "
            f"{_fmt(m.get('correlation'), 3)} | {_fmt(d.get('accuracy'), 3)} | {ptxt} |"
        )
    lines += [
        "",
        direction.get("note", ""),
        "",
        "Per-coin directional accuracy near 0.42 is **not** evidence of price-timing skill, and it is not better than a coin flip. Magnitude (RMSE/MAE) is the success metric.",
        "",
        "## Error distribution (voting residual = predicted − actual)",
        "",
        f"- Mean error (bias): {_fmt(errors.get('mean_error'), 6)} (positive = over-predict vol)",
        f"- Residual std: {_fmt(errors.get('std_error'))}",
        f"- Skew / excess kurtosis: {_fmt(errors.get('skew'), 2)} / {_fmt(errors.get('excess_kurtosis'), 2)}",
        f"- Share of over-predictions: {_fmt(errors.get('pct_overpredict'), 3)}",
        f"- Median |error|: {_fmt(errors.get('p50_abs_error'))}; 95th |error|: {_fmt(errors.get('p95_abs_error'))}",
        "",
        "## Volatility regimes (voting, by actual vol_24h)",
        "",
        "R² can be negative inside a narrow regime even when overall R² is positive — the model is ranked on RMSE vs baselines, not regime R².",
        "",
        "| Regime | n | RMSE | MAE | Bias | Corr |",
        "|---|---|---|---|---|---|",
    ]
    for label in ("low", "normal", "high", "spike"):
        r = regimes.get(label, {})
        if not r:
            continue
        lines.append(
            f"| {label} | {r.get('n', '')} | {_fmt(r.get('rmse'))} | {_fmt(r.get('mae'))} | "
            f"{_fmt(r.get('bias'), 6)} | {_fmt(r.get('correlation'), 3)} |"
        )
    lines += [
        "",
        "## What we will claim",
        "",
        "- Voting beats persistence and HAR-RV on walk-forward RMSE, with significant Diebold–Mariano tests.",
        "- Forecasts are 24h realized **volatility**, not price direction.",
        "",
        "## What we will not claim",
        "",
        "- Pooled directional accuracy (~80%) as user-facing skill.",
        "- Per-coin ability to call whether volatility will rise or fall (near or below 50%).",
        "- That the original 60-day holdout RMSE of 0.00173 is the production number (that window was calmer).",
        "",
        "Feature attributions (TreeSHAP of the serving ensemble, not a new bake-off): [docs/SHAP.md](SHAP.md).",
        "",
        "Machine-readable source: `models/walkforward_metrics.json`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward voting vs volatility baselines.")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--min-train-days", type=int, default=365)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = Path(args.features) if args.features else FEATURES_PATH
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    if not features.is_absolute():
        features = ROOT / features
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    walkforward(features_path=features, models_dir=models_dir, test_days=args.test_days, min_train_days=args.min_train_days)


if __name__ == "__main__":
    main()
