"""Raw OHLCV quality checks. Fail closed: bad data must not look like a fresh forecast."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = [
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass
class ValidationReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    symbols: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Data validation failed:\n- " + "\n- ".join(self.issues))


def validate_ohlcv(frame: pd.DataFrame, max_gap_hours: float = 1.5) -> ValidationReport:
    issues: list[str] = []
    if frame.empty:
        return ValidationReport(ok=False, issues=["OHLCV frame is empty"])

    absent = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if absent:
        issues.append(f"Missing columns: {absent}")
        return ValidationReport(ok=False, issues=issues, n_rows=len(frame))

    work = frame.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], utc=True)
    n_null = int(work[REQUIRED_COLUMNS].isna().sum().sum())
    if n_null:
        issues.append(f"{n_null} null cells in required columns")

    dup = int(work.duplicated(subset=["symbol", "open_time"]).sum())
    if dup:
        issues.append(f"{dup} duplicate (symbol, open_time) rows")

    bad_hl = work["high"] < work["low"]
    if bool(bad_hl.any()):
        issues.append(f"{int(bad_hl.sum())} rows with high < low")

    nonpos = (work[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bool(nonpos.any()):
        issues.append(f"{int(nonpos.sum())} rows with non-positive OHLC")

    if bool((work["volume"] < 0).any()):
        issues.append("Negative volume present")

    symbols = sorted(work["symbol"].astype(str).unique().tolist())
    warnings: list[str] = []
    for symbol, group in work.groupby("symbol"):
        ordered = group.sort_values("open_time")
        deltas = ordered["open_time"].diff().dt.total_seconds() / 3600.0
        gaps = deltas.dropna()
        if gaps.empty:
            continue
        extra = gaps[gaps > max_gap_hours]
        if len(extra):
            warnings.append(
                f"{symbol}: {len(extra)} hourly gaps > {max_gap_hours}h "
                f"(max gap {float(extra.max()):.1f}h)"
            )

    return ValidationReport(
        ok=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        n_rows=len(work),
        symbols=symbols,
    )
