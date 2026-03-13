from flask import Flask, render_template, jsonify
import sqlite3
import os

app = Flask(__name__)
app.debug = True

DB_FILE = "trades.db"

# Ensure database & table exist
def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE trades (
                time TEXT,
                price REAL,
                rsi REAL,
                signal TEXT,
                balance REAL
            )
        """)
        conn.commit()
        conn.close()

# Main page
@app.route("/")
def index():
    init_db()
    return render_template("dashboard_dynamic.html")

# JSON endpoint for trades
@app.route("/api/trades")
def get_trades():
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY time ASC")
        trades = cursor.fetchall()
        conn.close()

        # Convert to list of dicts
        trades_list = [
            {
                "time": t[0] or "",
                "price": t[1] or 0,
                "rsi": t[2] or 0,
                "signal": t[3] or "",
                "balance": t[4] or 0
            }
            for t in trades
        ]

        return jsonify(trades_list)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)