def generate_signal(df):

    last_rsi = df["RSI"].iloc[-1]

    if last_rsi < 30:
        return "BUY"

    if last_rsi > 70:
        return "SELL"

    return "HOLD"