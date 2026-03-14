from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(exchange_name: str, api_key: str = "", api_secret: str = "", password: str = ""):
    import ccxt

    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "password": password,
            "enableRateLimit": True,
        }
    )
    return exchange


def _milliseconds(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_history(
    exchange_name: str,
    symbol: str,
    timeframe: str = "5m",
    years: int = 2,
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    exchange = make_exchange(exchange_name)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)
    since = _milliseconds(start)
    end_ms = _milliseconds(end)

    all_rows: List[list] = []

    while since < end_ms:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        next_since = last_ts + 1
        if next_since <= since:
            break
        since = next_since

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df.reset_index(drop=True)


def fetch_and_cache_symbols(
    exchange_name: str,
    symbols: list[str],
    timeframe: str,
    years: int,
    out_dir: str = "data",
) -> pd.DataFrame:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        clean_symbol = symbol.replace("/", "_")
        out_path = Path(out_dir) / f"{exchange_name}_{clean_symbol}_{timeframe}_{years}y.csv"
        df = fetch_ohlcv_history(exchange_name, symbol, timeframe=timeframe, years=years)
        df.to_csv(out_path, index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS + ["datetime", "symbol"])
    return combined
