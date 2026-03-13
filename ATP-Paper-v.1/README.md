# ATP---trade-bot
An AI Crpyto Trading Bot (algorithmic trade predictor)

# PaperBot v1.2
## **PaperBot v1.2** is a simple Python paper trading bot for cryptocurrencies.  
It fetches live data, calculates indicators, generates trade signals, and simulates trades — no real money required.

---

# Features

- Fetches OHLCV data from Binance  
- Calculates **RSI** and **Bollinger Bands**  
- Generates **BUY / SELL / HOLD signals**  
- Simulates trades with a starting balance  

---

# Setup

1. Clone the repo:

```bash
git clone https://github.com/YourUsername/PaperBot-v1.2.git
cd PaperBot-v1.2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate # Mac/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

# Config
Edit config.py:

```py
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
START_BALANCE = 20

RSI_BUY = 30
RSI_SELL = 70
```

# Run the Bot
```bash
python run_bot.py
```
# The bot will:

Fetch market data
Calculate indicators
Generate a signal
Simulate a trade
Strategy

Mean Reversion:
BUY: RSI < 30 and price below lower Bollinger Band
SELL: RSI > 70 and price above upper Bollinger Band
HOLD otherwise