# ATB — Algorithmic Trading Board

> Modular paper trading engine with AI price prediction, real-time dashboard, and strategy experimentation framework.

```
Python + FastAPI + XGBoost + WebSocket + Chart.js
Paper Trading · v0.1 · Binance.US
```

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [All Commands](#all-commands)
- [API Endpoints](#api-endpoints)
- [Strategies](#strategies)
- [AI Pipeline](#ai-pipeline)
- [Dashboard](#dashboard)
- [Risk Management](#risk-management)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Overview

ATB is a modular algorithmic trading engine built for strategy experimentation and AI-driven market prediction. It runs in **paper trading mode by default** — all trades are simulated with realistic slippage and commissions. No real money at risk.

**Data flow:** Market data → Feature engineering → Strategy signals → Risk checks → Paper execution → Portfolio tracking

| Component | Role |
|---|---|
| Python 3.14+ | Core engine, strategies, AI |
| FastAPI + Uvicorn | REST API + WebSocket server |
| XGBoost + scikit-learn | AI price direction prediction |
| aiohttp + websockets | Binance.US market data feed |
| Chart.js | Dashboard charts (no build step needed) |

---

## Project Structure

```
ATB-V1/
├── main.py                        # Engine entry point
├── requirements.txt
│
├── core/
│   ├── types.py                   # Shared dataclasses: Bar, Signal, Order, Fill, Position
│   └── events.py                  # Async event bus (pub/sub between all components)
│
├── data/
│   └── feeds/
│       └── binance.py             # Binance.US WebSocket feed + REST historical data
│
├── features/
│   └── pipeline.py                # RSI, EMA, ATR, VWAP, Bollinger Band indicators
│
├── strategies/
│   └── strategy.py                # RSIReversal + EMACrossover + strategy registry
│
├── execution/
│   └── engine.py                  # Paper broker, risk manager, ATR position sizer
│
├── portfolio/
│   └── portfolio.py               # PnL tracking, Sharpe ratio, max drawdown, trade log
│
├── backtesting/
│   └── runner.py                  # Historical simulation using same strategy code as live
│
├── ai/
│   ├── predictor.py               # XGBoost model, 25-feature engineering, AIStrategy class
│   ├── train.py                   # Basic training script
│   ├── train_enhanced.py          # Multi-timeframe + walk-forward validation training
│   ├── data_collector.py          # Paginated historical data downloader (saves CSV)
│   ├── progressive_trainer.py     # Continuous random-chunk training loop
│   ├── data/                      # Saved CSV datasets (BTCUSDT_1m.csv, etc.)
│   └── models/                    # Saved trained models (.pkl) + training history
│
├── api/
│   └── main.py                    # FastAPI app: REST endpoints + WebSocket live stream
│
├── dashboard/
│   └── index.html                 # Live trading dashboard (open directly in browser)
│
└── tests/
    └── unit/
        └── test_indicators.py
```

---

## Installation

```powershell
# 1. Navigate to project folder
cd K:\ATP\ATB-V1

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn websockets aiohttp pydantic
pip install xgboost scikit-learn pandas numpy

# 4. Create package __init__.py files
$pkgs = @("core","data","data\feeds","features","strategies","execution","portfolio","backtesting","ai","api","tests","tests\unit")
foreach ($p in $pkgs) { New-Item -ItemType File -Force "$p\__init__.py" }
```

---

## Quick Start

### API + Dashboard (no live data needed)

```powershell
# Terminal 1 — start engine
python main.py --mode api

# Open dashboard in browser
start K:\ATP\ATB-V1\dashboard\index.html
```

### Paper Trading (live Binance.US data)

```powershell
python main.py --mode paper
```

### Paper Trading + AI Predictions

```powershell
# One-time setup: collect data and train models
python -m ai.data_collector --days 365
python -m ai.train

# Run engine with AI enabled
python main.py --mode paper --ai
```

### Full Setup: Engine + Progressive Training

```powershell
# Terminal 1 — engine
python main.py --mode paper --ai

# Terminal 2 — progressive trainer (runs forever, improves model continuously)
python -m ai.progressive_trainer --interval 30 --strategy recent_biased
```

---

## All Commands

### Engine Modes

| Command | Description |
|---|---|
| `python main.py --mode api` | API server only — good for dashboard development |
| `python main.py --mode paper` | Live paper trading with Binance.US WebSocket |
| `python main.py --mode paper --ai` | Paper trading with AI prediction strategies |
| `python main.py --mode train` | Train AI models then exit |
| `python main.py --mode backtest` | Backtest against historical data |

### Engine Flags

| Flag | Default | Description |
|---|---|---|
| `--symbols BTCUSDT ETHUSDT` | BTCUSDT ETHUSDT | Symbols to trade |
| `--timeframe 5m` | 5m | Candle timeframe: 1m, 5m, 15m, 1h |
| `--capital 100000` | 100000 | Starting paper capital in USD |
| `--api-port 8000` | 8000 | API server port |
| `--ai` | off | Enable AI prediction strategies |
| `--train-bars 5000` | 5000 | Historical bars for AI training |

### AI Data Collection

```powershell
# Fetch 60 days for BTC + ETH (default)
python -m ai.data_collector

# Fetch full year
python -m ai.data_collector --days 365

# One symbol only
python -m ai.data_collector --symbols BTCUSDT

# Update existing dataset (fetches only new bars since last run)
python -m ai.data_collector
```

### AI Model Training

```powershell
# Basic training (fast, ~30 seconds)
python -m ai.train

# Train on more data
python -m ai.train --bars 20000

# Enhanced training: multi-timeframe + walk-forward validation + feature selection
python -m ai.train_enhanced

# Skip feature selection step
python -m ai.train_enhanced --no-feature-selection
```

### Progressive Trainer

```powershell
# Default: trains every 30 minutes with recent-biased sampling
python -m ai.progressive_trainer

# Custom interval and chunk size
python -m ai.progressive_trainer --interval 60 --chunk-size 8000

# Sampling strategies
python -m ai.progressive_trainer --strategy uniform        # equal chance any period
python -m ai.progressive_trainer --strategy recent_biased  # recent 3x more likely (default)
python -m ai.progressive_trainer --strategy regime_aware   # balanced across volatility regimes
```

### Testing

```powershell
pytest tests/ -v
pytest tests/unit/test_indicators.py -v
pytest tests/ --cov=features --cov=strategies
```

---

## API Endpoints

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health + WebSocket client count |
| GET | `/portfolio` | Full portfolio snapshot |
| GET | `/portfolio/trades` | Closed trade history (`?limit=50&strategy_id=...`) |
| GET | `/portfolio/strategy-pnl` | PnL breakdown per strategy |
| GET | `/strategies` | All strategies with status and PnL |
| POST | `/strategies/{id}/toggle` | Enable or disable a strategy |
| GET | `/market/prices` | Latest prices for all watched symbols |
| GET | `/ai/predictions` | Latest AI direction predictions |
| GET | `/ai/training-history` | Progressive training iteration history |
| GET | `/ai/model-status` | Model metadata (accuracy, features, age) |
| WS | `/ws/live` | Live stream: prices, fills, PnL, AI predictions |

### WebSocket Message Types

```json
{ "type": "price_update", "symbol": "BTCUSDT", "close": 84231.5, "timeframe": "1m", "timestamp": "..." }
{ "type": "fill", "symbol": "BTCUSDT", "side": "buy", "quantity": 0.00118, "price": 84232.1, "strategy_id": "..." }
{ "type": "portfolio_update", "portfolio_value": 100145.2, "realized_pnl": 145.2, "win_rate": 0.62 }
{ "type": "ai_prediction", "symbol": "BTCUSDT", "direction": 1, "direction_label": "UP", "probability_up": 0.67, "confidence": "high" }
{ "type": "signal", "symbol": "BTCUSDT", "side": "buy", "strategy_id": "rsi_reversal_btcusdt", "reason": "RSI oversold (28.3 < 30)" }
{ "type": "ping" }
```

> Respond to `ping` with `{ "type": "pong" }` to keep the connection alive.

---

## Strategies

### RSI Reversal `rsi_reversal_{symbol}`

**Type:** Mean reversion · **Indicators:** RSI-14, VWAP

- BUY when RSI < 30 (oversold) AND price is above VWAP (trend filter)
- SELL when RSI > 70 (overbought) OR position held 20+ bars
- Signal strength: RSI < 20 = STRONG, < 25 = MEDIUM, < 30 = WEAK

Best in ranging/choppy markets. Underperforms in strong trends.

### EMA Crossover `ema_cross_{symbol}`

**Type:** Trend following · **Indicators:** EMA-9, EMA-21

- BUY on golden cross: EMA-9 crosses above EMA-21
- SELL on death cross: EMA-9 crosses below EMA-21

Best in trending markets. Whipsaws in choppy conditions.

### AI Predictor `ai_predictor_{symbol}`

**Type:** ML classification · **Timeframe:** 1m · **Model:** XGBoost

- Predicts next 1m candle direction (UP / DOWN)
- Only fires when confidence = HIGH (probability > 65%)
- Model updated continuously by progressive trainer — no restart needed

Expected test accuracy: 52–57%. Above 60% likely indicates overfitting.

### Adding a New Strategy

```python
# In strategies/strategy.py

@registry.register("my_strategy")
class MyStrategy(AbstractStrategy):
    def on_bar(self, features: dict) -> Signal | None:
        # Available features: rsi_14, ema_9, ema_21, ema_200,
        # close, high, low, volume, vwap, atr_14, bar_count,
        # ema_cross_bullish, above_vwap, rsi_oversold, rsi_overbought
        if features.get("rsi_14") and features["rsi_14"] < 25:
            return self._make_signal(
                side=Side.BUY,
                price=features["close"],
                reason="Custom oversold signal"
            )
        return None
```

Then add it in `main.py` inside `setup()`:

```python
self.strategy_manager.add(MyStrategy(
    strategy_id=f"my_strategy_{symbol.lower()}",
    symbol=symbol,
    timeframe=timeframe,
))
```

---

## AI Pipeline

### Feature Engineering (25 features)

| Feature | Description |
|---|---|
| `return_1/3/5/10/30` | Price returns over N bars |
| `body_size` | Candle body as % of price |
| `upper_wick / lower_wick` | Wick ratios |
| `rsi_14 / rsi_7` | RSI at two periods |
| `rsi_momentum` | 3-bar RSI rate of change |
| `price_vs_ema9/21` | Normalized distance to EMAs |
| `ema9_vs_ema21` | EMA spread (trend direction) |
| `price_vs_vwap` | Price vs volume-weighted average |
| `atr_normalized` | ATR as % of price |
| `bb_position` | Position within Bollinger Bands (0–1) |
| `volume_ma_ratio` | Volume vs 20-bar average |
| `volume_spike` | Binary: volume > 1.5× average |
| `momentum_5/10` | Price momentum over N bars |
| `return_zscore` | Rolling z-score of returns |

### Sampling Strategies

| Strategy | Description |
|---|---|
| `recent_biased` | Recent bars 3× more likely. Best default — adapts to current regime. |
| `uniform` | Fully random from all history. Best for regime robustness. |
| `regime_aware` | Balanced sampling across high/mid/low volatility periods. |

The trainer automatically switches strategy after 24h of no improvement.

### Recommended Workflow

```powershell
# 1. Collect history (run once — ~10 min for 365 days)
python -m ai.data_collector --days 365

# 2. Train initial model
python -m ai.train_enhanced

# 3. Run progressive trainer in background (Terminal 2)
python -m ai.progressive_trainer --interval 30

# 4. Check training history anytime
# ai/models/BTCUSDT_training_history.csv
# ai/models/ETHUSDT_training_history.csv

# 5. Update dataset weekly (only fetches new bars)
python -m ai.data_collector
python -m ai.train_enhanced
```

---

## Dashboard

Open `dashboard/index.html` directly in any browser while the engine is running.

| Tab | Contents |
|---|---|
| **Sidebar** | Portfolio stats, live prices, AI forecasts with probability bars, strategy toggles |
| **Overview** | Metric cards + equity curve + strategy PnL chart + win/loss donut |
| **Charts** | Live price chart (BTC/ETH switch), volume bars, RSI-14 |
| **Trades** | Full trade history table + per-trade PnL bar chart |
| **AI Model** | Validation accuracy over training iterations + model status |
| **Event Log** | Live stream of all WebSocket events color-coded by type |

The dashboard auto-reconnects on disconnect with exponential backoff. The top-right status dot shows green when the WebSocket is live.

---

## Risk Management

All orders pass through the `RiskManager` before execution.

| Rule | Default |
|---|---|
| Max position value | 15% of capital |
| Max total exposure | 80% of capital |
| Max concurrent positions | 5 |
| Daily loss limit | 5% of capital (halts trading for the day) |
| Minimum account balance | $1,000 |
| Max single order value | $20,000 |

**Position sizing** uses ATR-based risk:

```
quantity = (capital × 1%) / (ATR × 2.0)
```

This automatically reduces size in volatile markets. Signal strength scales it further: STRONG = 100%, MEDIUM = 50%, WEAK = 25%.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Complete | Paper trading engine, RSI + EMA strategies, API, dashboard |
| 2 | ✅ Complete | AI prediction layer, progressive training, multi-timeframe features |
| 3 | 🔄 In Progress | Full backtesting with parameter optimizer, HTML reports |
| 4 | 📋 Planned | LSTM / Transformer model, RL environment (Gymnasium) |
| 5 | 📋 Planned | PostgreSQL persistence, Redis feature cache, Alembic migrations |
| 6 | 📋 Planned | Live trading via Binance.US API, full order book simulation |
| 7 | 📋 Planned | Docker Compose deployment, distributed multi-symbol scaling |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'core.events'` | Files not in subfolders. All `.py` files must be in `core/`, `ai/`, etc. — not in the root. |
| `cannot import name 'create_app'` | Old `api/main.py`. Re-download the latest version. |
| `HTTP 451 from Binance` | US geo-restriction. Must use `api.binance.us` — not `api.binance.com`. |
| `add_signal_handler NotImplementedError` | Windows doesn't support this. Use updated `main.py` with `platform.system()` check. |
| `XGBoostError: Input data contai...` | NaN/Inf in features. Re-download latest `predictor.py` with data cleaning. |
| `Label distribution ~26% bullish` | Class imbalance — fixed in latest `predictor.py` with `scale_pos_weight`. |
| `Dashboard shows disconnected` | Engine not running. Start `python main.py --mode api` first. |
| `pip Fatal error in launcher` | Venv pointing to old folder path. Recreate: `python -m venv venv` in the new folder. |
| `unrecognized arguments: --ai` | Old `main.py` without AI support. Re-download updated version. |

---

## File Checklist

Verify all required files are present:

```powershell
Get-ChildItem K:\ATP\ATB-V1\ -Recurse -Name |
  Where-Object { $_ -match "\.py$" -and $_ -notmatch "venv" } |
  Sort-Object
```

Required files:

```
main.py
core/__init__.py          core/types.py             core/events.py
data/__init__.py          data/feeds/__init__.py    data/feeds/binance.py
features/__init__.py      features/pipeline.py
strategies/__init__.py    strategies/strategy.py
execution/__init__.py     execution/engine.py
portfolio/__init__.py     portfolio/portfolio.py
backtesting/__init__.py   backtesting/runner.py
ai/__init__.py            ai/predictor.py           ai/train.py
ai/data_collector.py      ai/train_enhanced.py      ai/progressive_trainer.py
api/__init__.py           api/main.py
dashboard/index.html
```

---

<div align="center">
<sub>ATB · Paper Trading · v0.1 · Built with Python, FastAPI, XGBoost</sub>
</div>