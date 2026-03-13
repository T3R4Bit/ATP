import time
import threading
import sqlite3
from datetime import datetime

from data_fetcher import fetch_data
from indicators import calculate_rsi
from strategy import generate_signal
from paper_trader import PaperTrader
import config
import dashboard
from config import MARKET_CHECK_INTERVAL, TRADE_INTERVAL

time.sleep(MARKET_CHECK_INTERVAL)

TRADE_INTERVAL = TRADE_INTERVAL

DB_FILE = "trades.db"

# --------------------------
# Log trades / market snapshots
# --------------------------
def log_trade(time_str, price, rsi, signal, balance):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            time TEXT,
            price REAL,
            rsi REAL,
            signal TEXT,
            balance REAL
        )
    """)

    # Insert row
    cursor.execute("""
        INSERT INTO trades (time, price, rsi, signal, balance)
        VALUES (?, ?, ?, ?, ?)
    """, (time_str, price, rsi, signal, balance))

    conn.commit()
    conn.close()


# --------------------------
# Initialize bot
# --------------------------
bot = PaperTrader(config.START_BALANCE)

# --------------------------
# Start dashboard in background
# --------------------------
def start_dashboard():
    print("\nDashboard running at http://127.0.0.1:5000\n")
    dashboard.app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )

dashboard_thread = threading.Thread(target=start_dashboard)
dashboard_thread.daemon = True
dashboard_thread.start()


# --------------------------
# Helper: determines if it's time to run strategy (every 5 min)
# --------------------------
last_trade_time = None
TRADE_INTERVAL = 300  # seconds

def time_to_trade():
    global last_trade_time
    now = time.time()
    if (last_trade_time is None) or (now - last_trade_time >= TRADE_INTERVAL):
        last_trade_time = now
        return True
    return False


# --------------------------
# Main loop: market snapshots every 5 seconds
# --------------------------
while True:
    print("\n--- Checking Market ---")

    # Fetch current price and calculate RSI
    df = fetch_data(config.SYMBOL)
    df = calculate_rsi(df)

    price = df["close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    # Default signal: HOLD
    signal = "HOLD"

    # --------------------------
    # Execute strategy every 5 min
    # --------------------------
    if time_to_trade():
        signal = generate_signal(df)
        if signal in ["BUY", "SELL"]:
            bot.execute(signal, price)
            print(f"Executed {signal} at {price}")

    # --------------------------
    # Log snapshot / trade to database
    # --------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_trade(timestamp, price, rsi, signal, bot.balance)

    # --------------------------
    # Print status
    # --------------------------
    print(f"Price: {price}")
    print(f"RSI: {round(rsi, 2)}")
    print(f"Signal: {signal}")
    if signal == "BUY":
        print(f"Action: BOUGHT at {price}")
    elif signal == "SELL":
        print(f"Action: SOLD at {price}")
    else:
        print("Action: HOLD (no trade)")
    print(f"Balance: {bot.balance}")
    print("Next market check in 5 seconds...")

    # Wait 5 seconds for next snapshot
    time.sleep(5)