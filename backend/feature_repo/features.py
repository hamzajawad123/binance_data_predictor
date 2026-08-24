"""Feast entity + feature view over the offline features parquet."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

from backend.src import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PARQUET = ROOT / "data" / "features.parquet"

crypto_source = FileSource(
    name="crypto_features_source",
    path=str(FEATURES_PARQUET),
    timestamp_field="event_timestamp",
)

symbol = Entity(
    name="symbol",
    join_keys=["symbol"],
    value_type=ValueType.STRING,
    description="Binance USDT trading pair",
)

_INT_FIELDS = {"hour", "dow", "symbol_id"}
_SCHEMA = [
    Field(name=name, dtype=Int64 if name in _INT_FIELDS else Float64) for name in FEATURE_COLUMNS
]

crypto_tech_indicators = FeatureView(
    name="crypto_tech_indicators",
    entities=[symbol],
    ttl=timedelta(days=7),
    schema=_SCHEMA,
    source=crypto_source,
    online=True,
    description="EDA-selected leakage-safe features for 24h volatility forecasts.",
)

FEATURE_REFS = [f"crypto_tech_indicators:{name}" for name in FEATURE_COLUMNS]


def materialize_online(days: int = 14) -> None:
    """Apply the repo and push recent features into the SQLite online store."""
    if not FEATURES_PARQUET.exists():
        print(f"Skip Feast materialize; missing {FEATURES_PARQUET}")
        return
    repo = Path(__file__).resolve().parent
    store = FeatureStore(repo_path=str(repo))
    store.apply([crypto_source, symbol, crypto_tech_indicators])
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    store.materialize(start, end)
    print(f"Feast online store materialized {start.isoformat()} -> {end.isoformat()}")


def get_online_feature_row(symbol_name: str) -> dict | None:
    repo = Path(__file__).resolve().parent
    store = FeatureStore(repo_path=str(repo))
    result = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"symbol": symbol_name}],
    ).to_dict()
    if not result:
        return None
    row = {key: values[0] if values else None for key, values in result.items()}
    if row.get("vol_24h_hist") is None:
        return None
    return row


if __name__ == "__main__":
    materialize_online()
