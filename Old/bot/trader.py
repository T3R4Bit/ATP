from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bot.config import TradingConfig


@dataclass
class TradeAction:
    time: str
    symbol: str
    action: str
    prediction: float
    price: float


@dataclass
class PaperAccount:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    avg_cost: dict[str, float] = field(default_factory=dict)
    realized_profit: float = 0.0
    realized_loss: float = 0.0

    def buy(self, symbol: str, price: float, amount: float) -> None:
        cost = price * amount
        if cost <= self.cash and amount > 0:
            old_qty = self.positions.get(symbol, 0.0)
            new_qty = old_qty + amount
            old_avg = self.avg_cost.get(symbol, 0.0)
            if new_qty > 0:
                self.avg_cost[symbol] = ((old_qty * old_avg) + (amount * price)) / new_qty
            self.cash -= cost
            self.positions[symbol] = new_qty

    def sell(self, symbol: str, price: float, amount: float) -> None:
        held = self.positions.get(symbol, 0.0)
        qty = min(held, max(amount, 0.0))
        if qty <= 0:
            return

        avg = self.avg_cost.get(symbol, price)
        pnl = (price - avg) * qty
        if pnl >= 0:
            self.realized_profit += pnl
        else:
            self.realized_loss += abs(pnl)

        self.positions[symbol] = held - qty
        if self.positions[symbol] <= 1e-12:
            self.positions[symbol] = 0.0
            self.avg_cost[symbol] = 0.0

        self.cash += qty * price


def profit_loss_ratio(account: PaperAccount) -> float | None:
    if account.realized_loss == 0:
        if account.realized_profit > 0:
            return float("inf")
        return None
    return account.realized_profit / account.realized_loss


def decide_action(pred_return: float, cfg: TradingConfig) -> str:
    if pred_return > cfg.buy_threshold:
        return "BUY"
    if pred_return < cfg.sell_threshold:
        return "SELL"
    return "HOLD"


def _latest_features(exchange_name: str, symbol: str, timeframe: str):
    from bot.data import fetch_ohlcv_history
    from bot.features import build_training_frame

    candles = fetch_ohlcv_history(exchange_name, symbol, timeframe=timeframe, years=1)
    frame = build_training_frame(candles)
    return frame.tail(1)


def _position_size_usdt(cash: float, cfg: TradingConfig) -> float:
    return cash * cfg.risk_per_trade


@dataclass
class RuntimeState:
    mode: str
    starting_balance: float
    portfolio_value: float
    pnl_pct: float
    profit_loss_ratio: float | None
    actions: list[TradeAction] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_action(self, action: TradeAction) -> None:
        with self._lock:
            self.actions.append(action)
            self.actions = self.actions[-200:]

    def update_metrics(self, portfolio_value: float, pnl_pct: float, pl_ratio: float | None) -> None:
        with self._lock:
            self.portfolio_value = portfolio_value
            self.pnl_pct = pnl_pct
            self.profit_loss_ratio = pl_ratio

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "starting_balance": self.starting_balance,
                "portfolio_value": self.portfolio_value,
                "pnl_pct": self.pnl_pct,
                "profit_loss_ratio": (None if self.profit_loss_ratio is None else (999999999.0 if self.profit_loss_ratio == float("inf") else self.profit_loss_ratio)),
                "actions": [a.__dict__ for a in self.actions],
            }


def _portfolio_value_paper(account: PaperAccount, latest_prices: dict[str, float]) -> float:
    value = account.cash
    for symbol, qty in account.positions.items():
        value += qty * latest_prices.get(symbol, 0.0)
    return value


def run_trading_loop(
    model,
    mode: str,
    exchange_name: str,
    symbols: list[str],
    timeframe: str,
    cfg: TradingConfig,
    dashboard_state: RuntimeState | None = None,
) -> None:
    from bot.data import make_exchange
    from bot.features import feature_columns

    exchange = make_exchange(exchange_name, cfg.api_key, cfg.api_secret, cfg.api_password)
    paper = PaperAccount(cash=cfg.starting_cash_usdt)

    features = feature_columns()
    latest_prices: dict[str, float] = {}

    while True:
        for symbol in symbols:
            latest = _latest_features(exchange_name, symbol, timeframe)
            if latest.empty:
                continue

            pred = float(model.predict(latest[features])[0])
            action = decide_action(pred, cfg)
            price = float(latest.iloc[-1]["close"])
            latest_prices[symbol] = price

            if mode == "paper":
                spend = _position_size_usdt(paper.cash, cfg)
                amount = spend / price if price else 0.0
                if action == "BUY":
                    paper.buy(symbol, price, amount)
                elif action == "SELL":
                    paper.sell(symbol, price, amount)

                portfolio_value = _portfolio_value_paper(paper, latest_prices)
                pnl_pct = (portfolio_value - cfg.starting_cash_usdt) / cfg.starting_cash_usdt
                pl_ratio = profit_loss_ratio(paper)

                print(
                    f"[PAPER] {symbol} pred={pred:.5f} action={action} value={portfolio_value:.2f} "
                    f"pnl={pnl_pct*100:.2f}% pl_ratio={pl_ratio}"
                )

                if dashboard_state:
                    dashboard_state.push_action(
                        TradeAction(
                            time=datetime.now(timezone.utc).isoformat(),
                            symbol=symbol,
                            action=action,
                            prediction=pred,
                            price=price,
                        )
                    )
                    dashboard_state.update_metrics(portfolio_value, pnl_pct, pl_ratio)

            elif mode == "live":
                balance = exchange.fetch_balance()
                usdt = float(balance.get("USDT", {}).get("free", 0.0))
                spend = _position_size_usdt(usdt, cfg)
                amount = spend / price if price else 0.0

                if action == "BUY" and amount > 0:
                    exchange.create_market_buy_order(symbol, amount)
                elif action == "SELL" and amount > 0:
                    exchange.create_market_sell_order(symbol, amount)

                print(f"[LIVE] {symbol} pred={pred:.5f} action={action} amount={amount:.6f}")

                if dashboard_state:
                    dashboard_state.push_action(
                        TradeAction(
                            time=datetime.now(timezone.utc).isoformat(),
                            symbol=symbol,
                            action=action,
                            prediction=pred,
                            price=price,
                        )
                    )
                    dashboard_state.update_metrics(usdt, 0.0, None)
            else:
                raise ValueError("mode must be 'paper' or 'live'")

        time.sleep(cfg.poll_seconds)
