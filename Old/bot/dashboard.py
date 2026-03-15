from __future__ import annotations

import threading


def start_dashboard(state, host: str = "127.0.0.1", port: int = 8000) -> None:
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route("/api/status")
    def api_status():
        return jsonify(state.snapshot())

    @app.route("/")
    def index():
        return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>ARG Trade Bot Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .grid { display: grid; grid-template-columns: repeat(2,minmax(200px,1fr)); gap: 12px; max-width: 900px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    code { background: #f5f5f5; padding: 2px 5px; border-radius: 4px; }
  </style>
</head>
<body>
  <h2>ARG Trade Bot Dashboard</h2>
  <div class=\"grid\">
    <div class=\"card\">Mode: <strong id=\"mode\">-</strong></div>
    <div class=\"card\">Start Balance: <strong id=\"start\">-</strong></div>
    <div class=\"card\">Portfolio Value: <strong id=\"value\">-</strong></div>
    <div class=\"card\">PnL %: <strong id=\"pnl\">-</strong></div>
    <div class=\"card\">Profit/Loss Ratio: <strong id=\"pl_ratio\">-</strong></div>
    <div class=\"card\">Last Action: <strong id=\"last_action\">-</strong></div>
  </div>
  <h3>Recent Actions</h3>
  <ul id=\"actions\"></ul>

  <script>
    async function refresh() {
      const resp = await fetch('/api/status');
      const d = await resp.json();
      document.getElementById('mode').textContent = d.mode;
      document.getElementById('start').textContent = Number(d.starting_balance).toFixed(2);
      document.getElementById('value').textContent = Number(d.portfolio_value).toFixed(2);
      document.getElementById('pnl').textContent = (Number(d.pnl_pct) * 100).toFixed(2) + '%';
      document.getElementById('pl_ratio').textContent = d.profit_loss_ratio === null ? 'N/A' : Number(d.profit_loss_ratio).toFixed(4);
      const last = d.actions.length ? d.actions[d.actions.length - 1] : null;
      document.getElementById('last_action').textContent = last ? `${last.symbol} ${last.action} @ ${last.price}` : '-';

      const ul = document.getElementById('actions');
      ul.innerHTML = '';
      d.actions.slice(-20).reverse().forEach(a => {
        const li = document.createElement('li');
        li.textContent = `${a.time} | ${a.symbol} | pred=${a.prediction.toFixed(5)} | action=${a.action} | price=${a.price.toFixed(2)}`;
        ul.appendChild(li);
      });
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""

    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
