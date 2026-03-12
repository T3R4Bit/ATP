import ccxt
import pandas as pd

exchange = ccxt.binanceus()

def fetch_data(symbol, timeframe="5m", limit=200):

    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(candles, columns=[
        "timestamp","open","high","low","close","volume"
    ])

    return df