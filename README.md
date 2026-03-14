# ARG Crypto Trade Bot

A Python crypto trading bot that can:

- Train an AI model on **5-minute BTC/USDT and ETH/USDT candles** for the last 2 years.
- Predict the next market move and decide to **BUY / SELL / HOLD**.
- Run in:
  - **Paper trading mode** (simulated balance, no real orders)
  - **Live trading mode** (real exchange orders through API keys)
- Track **portfolio PnL %** and **profit/loss ratio** in paper mode.
- Show a local dashboard for current actions and bot performance.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Train the AI model (market predictor)

This command downloads historical candles and trains the model used for trade decisions:

```bash
python -m bot.main train --exchange binance --symbols BTC/USDT ETH/USDT --timeframe 5m --years 2
```

- Use `--years 2` and `--timeframe 5m` for your required setup.
- The trained model is stored at `models/model.joblib`.

## Trade in paper mode with custom starting balance

```bash
python -m bot.main trade \
  --mode paper \
  --exchange binance \
  --symbols BTC/USDT ETH/USDT \
  --timeframe 5m \
  --starting-balance 25000
```

## Run dashboard on localhost

```bash
python -m bot.main trade \
  --mode paper \
  --exchange binance \
  --symbols BTC/USDT ETH/USDT \
  --timeframe 5m \
  --starting-balance 10000 \
  --dashboard \
  --dashboard-host 127.0.0.1 \
  --dashboard-port 8000
```

Open: `http://127.0.0.1:8000`

Dashboard includes:
- current mode
- starting balance
- portfolio value
- overall PnL %
- profit/loss ratio
- recent market actions and bot decisions

## Live mode (use caution)

```bash
python -m bot.main trade --mode live --exchange binance --symbols BTC/USDT ETH/USDT --timeframe 5m
```

## Notes

- Live trading uses market orders and is risky.
- Start with paper mode and validate behavior first.
- This is a baseline architecture; add stronger risk controls before production.
