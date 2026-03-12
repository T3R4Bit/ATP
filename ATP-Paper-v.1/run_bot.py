print("Bot starting...")
from data_fetcher import fetch_data
from indicators import calculate_rsi
from strategy import generate_signal
from paper_trader import PaperTrader
import config

bot = PaperTrader(config.START_BALANCE)

df = fetch_data(config.SYMBOL)
df = calculate_rsi(df)

signal = generate_signal(df)

price = df["close"].iloc[-1]

bot.execute(signal, price)

print("Signal:", signal)
print("Balance:", bot.balance)