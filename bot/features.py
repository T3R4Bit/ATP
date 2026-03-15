from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    work["ret_1"] = work.groupby("symbol")["close"].pct_change(1)
    work["ret_3"] = work.groupby("symbol")["close"].pct_change(3)
    work["ret_12"] = work.groupby("symbol")["close"].pct_change(12)

    work["sma_12"] = work.groupby("symbol")["close"].transform(lambda s: s.rolling(12).mean())
    work["sma_48"] = work.groupby("symbol")["close"].transform(lambda s: s.rolling(48).mean())
    work["sma_ratio"] = work["sma_12"] / work["sma_48"]

    work["volatility_12"] = work.groupby("symbol")["ret_1"].transform(lambda s: s.rolling(12).std())
    work["rsi_14"] = work.groupby("symbol")["close"].transform(lambda s: _rsi(s, 14))

    work["next_return"] = work.groupby("symbol")["close"].shift(-1) / work["close"] - 1

    work["symbol_id"] = work["symbol"].map({"BTC/USDT": 0, "ETH/USDT": 1}).fillna(2)

    return work.dropna().reset_index(drop=True)


def feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_3",
        "ret_12",
        "sma_ratio",
        "volatility_12",
        "rsi_14",
        "volume",
        "symbol_id",
    ]
