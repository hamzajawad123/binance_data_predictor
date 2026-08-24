"""Fetch hourly OHLCV candles from the Binance public klines API."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from backend.src import INTERVAL, LOOKBACK_DAYS, RAW_PATH, ROOT, SYMBOLS, env

BINANCE_BASE_URL = env("BINANCE_BASE_URL", "https://api.binance.com").rstrip("/")
KLINES_LIMIT = 1000
REQUEST_PAUSE_SEC = 0.2
MAX_RETRIES = 5

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-vol-forecast/0.1"})
    return session


def _get_json(session: requests.Session, url: str, params: dict) -> list:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code in {418, 429}:
                wait = min(2**attempt, 30)
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "code" in payload:
                raise RuntimeError(f"Binance error: {payload}")
            return payload
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries: {last_error}")


def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Paginate Binance klines between `start` and `end` (UTC, inclusive of start)."""
    session = session or _session()
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []

    while cursor_ms < end_ms:
        batch = _get_json(
            session,
            url,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": KLINES_LIMIT,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        last_open_ms = int(batch[-1][0])
        cursor_ms = last_open_ms + 1
        if len(batch) < KLINES_LIMIT:
            break
        time.sleep(REQUEST_PAUSE_SEC)

    if not rows:
        return pd.DataFrame(columns=["symbol", *KLINE_COLUMNS[:-1]])

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame = frame.drop(columns=["ignore"])
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    frame[numeric_cols] = frame[numeric_cols].astype(float)
    frame["trades"] = frame["trades"].astype(int)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame["symbol"] = symbol
    frame = frame.drop_duplicates(subset=["symbol", "open_time"]).sort_values("open_time")
    return frame.reset_index(drop=True)


def _last_open_time(existing: pd.DataFrame, symbol: str) -> datetime | None:
    subset = existing.loc[existing["symbol"] == symbol, "open_time"]
    if subset.empty:
        return None
    return pd.Timestamp(subset.max()).to_pydatetime()


LISTING_START = datetime(2017, 8, 1, tzinfo=timezone.utc)


def collect(
    symbols: list[str] | None = None,
    interval: str = INTERVAL,
    lookback_days: int = LOOKBACK_DAYS,
    incremental: bool = False,
    full_history: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Backfill or incrementally update `data/raw.parquet`."""
    symbols = symbols or SYMBOLS
    output_path = output_path or RAW_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if output_path.exists():
        existing = pd.read_parquet(output_path)

    end = datetime.now(timezone.utc)
    default_start = LISTING_START if full_history else end - timedelta(days=lookback_days)
    session = _session()
    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        start = default_start
        if incremental and not existing.empty and not full_history:
            last = _last_open_time(existing, symbol)
            if last is not None:
                start = last + timedelta(milliseconds=1)
        if start >= end:
            print(f"{symbol}: already up to date")
            continue
        print(f"{symbol}: fetching {interval} candles from {start.isoformat()} to {end.isoformat()}")
        batch = fetch_klines(symbol, interval, start, end, session=session)
        print(f"{symbol}: received {len(batch)} candles")
        frames.append(batch)

    if frames:
        fresh = pd.concat(frames, ignore_index=True)
        combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
        combined = combined.drop_duplicates(subset=["symbol", "open_time"]).sort_values(
            ["symbol", "open_time"]
        )
    else:
        combined = existing

    if combined.empty:
        raise RuntimeError("No OHLCV rows collected. Check network access to Binance.")

    combined.to_parquet(output_path, index=False)
    from backend.src.validate_data import validate_ohlcv

    report = validate_ohlcv(combined)
    if report.warnings:
        print("Data validation warnings:", "; ".join(report.warnings))
    if report.issues:
        print("Data validation issues:", "; ".join(report.issues))
    if not report.ok:
        print("Warning: raw data has quality issues (predictions should be treated as stale).")
    print(f"Wrote {len(combined):,} rows -> {output_path}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Binance hourly OHLCV into a parquet file.")
    parser.add_argument("--incremental", action="store_true", help="Fetch only candles after the last stored timestamp.")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Fetch from each pair's Binance listing (2017+) and merge with any existing parquet.",
    )
    parser.add_argument("--days", type=int, default=None, help="Lookback window in days (default LOOKBACK_DAYS).")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default from .env).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else SYMBOLS
    lookback = args.days if args.days is not None else LOOKBACK_DAYS
    output = Path(args.output) if args.output else RAW_PATH
    if not output.is_absolute():
        output = ROOT / output
    collect(
        symbols=symbols,
        lookback_days=lookback,
        incremental=args.incremental,
        full_history=args.full_history,
        output_path=output,
    )


if __name__ == "__main__":
    main()
