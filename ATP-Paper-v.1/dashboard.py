from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

@app.route("/")
def index():

    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM trades")
    trades = cursor.fetchall()

    conn.close()

    return render_template("dashboard.html", trades=trades)