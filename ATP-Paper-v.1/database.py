import sqlite3

conn = sqlite3.connect("trades.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS trades (
    timestamp TEXT,
    price REAL,
    rsi REAL,
    signal TEXT,
    balance REAL
)
""")

conn.commit()


def log_trade(timestamp, price, rsi, signal, balance):
    cursor.execute(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
        (timestamp, price, rsi, signal, balance)
    )
    conn.commit()