"""
ai/data_collector.py
Fetches maximum historical data across multiple timeframes and saves to CSV.
Run this once to build your dataset, then retrain anytime without re-downloading.

Usage:
    python -m ai.data_collector                    # fetch all symbols + timeframes
    python -m ai.data_collector --symbols BTCUSDT  # just BTC
    python -m ai.data_collector --days 90          # last 90 days of 1m data

Data saved to: ai/data/{SYMBOL}_{TIMEFRAME}.csv
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("ai/data")
BASE_URL = "https://api.binance.us/api/v3/klines"

# How many 1m bars fit in each timeframe (for calculating equivalent history)
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


async def fetch_bars(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
) -> list:
    """Fetch one page of bars (max 1000) between start and end timestamps."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": 1000,
    }
    async with session.get(BASE_URL, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_full_history(
    symbol: str,
    interval: str,
    days: int,
) -> pd.DataFrame:
    """
    Fetch `days` worth of data for a symbol/interval by paginating
    forward from (now - days) to now.
    
    Forward pagination is more reliable than backward — you always
    know where you started and can detect gaps.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    interval_ms = TIMEFRAME_MINUTES[interval] * 60 * 1000
    
    all_rows = []
    current_start = start_ms
    request_count = 0
    
    logger.info(f"  Fetching {symbol} {interval} from {start_dt.date()} to {end_dt.date()}...")

    async with aiohttp.ClientSession() as session:
        while current_start < end_ms:
            try:
                bars = await fetch_bars(
                    session, symbol, interval,
                    current_start,
                    min(current_start + 1000 * interval_ms, end_ms),
                )
                if not bars:
                    break
                    
                all_rows.extend(bars)
                current_start = bars[-1][0] + interval_ms
                request_count += 1
                
                # Progress update every 10 requests
                if request_count % 10 == 0:
                    fetched_dt = datetime.utcfromtimestamp(bars[-1][0] / 1000)
                    pct = (fetched_dt - start_dt) / (end_dt - start_dt) * 100
                    logger.info(f"    {pct:.0f}% — up to {fetched_dt.strftime('%Y-%m-%d %H:%M')} ({len(all_rows):,} bars)")
                
                # Rate limit: 1200 requests/min allowed, be conservative
                await asyncio.sleep(0.05)
                
            except aiohttp.ClientResponseError as e:
                logger.warning(f"    Request failed: {e}, retrying in 5s...")
                await asyncio.sleep(5)
                continue

    if not all_rows:
        logger.warning(f"  No data returned for {symbol} {interval}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume", "n_trades", "quote_volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    
    logger.info(f"  Done: {len(df):,} bars for {symbol} {interval}")
    return df


async def collect_all(symbols: list[str], timeframes: list[str], days_per_tf: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        for tf in timeframes:
            days = days_per_tf.get(tf, 30)
            output_path = DATA_DIR / f"{symbol}_{tf}.csv"
            
            # If file exists, only fetch new data since last bar
            if output_path.exists():
                existing = pd.read_csv(output_path, index_col=0, parse_dates=True)
                last_date = existing.index[-1]
                days_needed = (datetime.utcnow() - last_date).days + 1
                
                if days_needed <= 1:
                    logger.info(f"  {symbol} {tf}: already up to date ({last_date.date()})")
                    continue
                    
                logger.info(f"  {symbol} {tf}: updating {days_needed} days since {last_date.date()}")
                new_df = await fetch_full_history(symbol, tf, days=days_needed)
                
                if not new_df.empty:
                    combined = pd.concat([existing, new_df])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    combined.to_csv(output_path)
                    logger.info(f"  Saved {len(combined):,} total bars → {output_path}")
            else:
                df = await fetch_full_history(symbol, tf, days=days)
                if not df.empty:
                    df.to_csv(output_path)
                    logger.info(f"  Saved {len(df):,} bars → {output_path}")


def load_dataset(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load a saved dataset. Used by the trainer."""
    path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No data file at {path}. Run: python -m ai.data_collector first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df):,} bars for {symbol} {timeframe} from {path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical market data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--days", type=int, default=60,
                        help="Days of 1m history to fetch (default 60)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    
    # Timeframes to fetch and how many days of each
    # More days for higher timeframes (they have fewer bars per day)
    days_per_tf = {
        "1m":  args.days,        # e.g. 60 days = ~86,400 bars
        "5m":  args.days * 3,    # 180 days
        "15m": args.days * 6,    # 360 days
        "1h":  args.days * 12,   # 720 days (~2 years)
        "4h":  args.days * 20,   # 1200 days (~3 years)
        "1d":  args.days * 30,   # 1800 days (~5 years)
    }
    
    timeframes = list(days_per_tf.keys())
    
    logger.info(f"Collecting data for: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Base days (1m): {args.days}")
    logger.info(f"Data will be saved to: {DATA_DIR.absolute()}")
    
    asyncio.run(collect_all(symbols, timeframes, days_per_tf))
    
    logger.info("\nDone! Now train with:")
    logger.info("  python -m ai.train_enhanced")