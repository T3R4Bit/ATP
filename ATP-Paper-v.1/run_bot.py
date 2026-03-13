import time

from data_fetcher import fetch_data
from indicators import calculate_rsi
from strategy import generate_signal
from paper_trader import PaperTrader
from database import log_trade
import config
import dashboard
import datetime
import threading



bot = PaperTrader(config.START_BALANCE)

def start_dashboard():
    print("\nDashboard running at:")
    print("http://127.0.0.1:5000\n")
    
    dashboard.app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )

dashboard_thread = threading.Thread(target=start_dashboard)
dashboard_thread.daemon = True
dashboard_thread.start()

while True:

    print("\n--- Checking Market ---")

    df = fetch_data(config.SYMBOL)
    df = calculate_rsi(df)

    rsi = df["RSI"].iloc[-1]
    price = df["close"].iloc[-1]

    signal = generate_signal(df)

    print(f"Price: {price}")
    print(f"RSI: {round(rsi, 2)}")
    print(f"Signal: {signal}")

    # Execute trade
    bot.execute(signal, price)

    #log trade to database
    timestamp = datetime.datetime.now()
    log_trade(timestamp, price, rsi, signal, bot.balance)

    # Print action taken
    if signal == "BUY":
        print(f"Action: BOUGHT at {price}")

    elif signal == "SELL":
        print(f"Action: SOLD at {price}")

    else:
        print("Action: HOLD (no trade)")

    print(f"Balance: {bot.balance}")
    print("Next check in 5 minutes...")

    time.sleep(300)