"""
portfolio/portfolio.py
Portfolio management — tracks all fills, computes PnL and performance metrics.

Design: the portfolio is the source of truth for trading outcomes.
It receives Fill events from the broker and maintains:
- Current positions and their unrealized PnL
- Closed trade history with realized PnL
- Performance metrics (Sharpe, drawdown, win rate)

All metrics are computed on-demand — no background threads.
For real-time dashboard: call get_snapshot() and push it over WebSocket.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from core.events import EventType, Event, bus
from core.types import Fill, Side

logger = logging.getLogger(__name__)


@dataclass
class ClosedTrade:
    symbol: str
    strategy_id: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    side: Side
    entry_time: datetime
    exit_time: datetime
    commission_paid: Decimal
    realized_pnl: Decimal

    @property
    def return_pct(self) -> float:
        cost = float(self.entry_price * self.quantity)
        return float(self.realized_pnl) / cost if cost != 0 else 0.0

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: Decimal
    portfolio_value: Decimal
    total_realized_pnl: Decimal
    total_unrealized_pnl: Decimal
    total_commission: Decimal
    n_open_positions: int
    n_closed_trades: int
    win_rate: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    daily_pnl: Decimal


class Portfolio:
    """
    Tracks the complete state of the trading account.

    PnL calculation:
    - Realized PnL: locked in when a position is closed
    - Unrealized PnL: computed from current price vs average cost
    - Total PnL: realized + unrealized

    Attribution: PnL is tracked per strategy so you can compare them.
    """

    def __init__(self, initial_cash: Decimal = Decimal("100000")) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash

        # Open fills waiting to be matched (entry fills)
        self._open_fills: Dict[str, List[Fill]] = {}  # symbol -> [fills]

        # Closed trades (for metrics)
        self.closed_trades: List[ClosedTrade] = []

        # Portfolio value history for Sharpe / drawdown
        self._value_history: List[Decimal] = [initial_cash]
        self._peak_value: Decimal = initial_cash

        # Per-strategy PnL
        self._strategy_pnl: Dict[str, Decimal] = {}

        # Totals
        self.total_commission: Decimal = Decimal("0")
        self.total_realized_pnl: Decimal = Decimal("0")

        # Current prices (updated from BAR events)
        self._current_prices: Dict[str, Decimal] = {}

        # Subscribe to events
        bus.subscribe_handler(EventType.ORDER_FILLED, self._on_fill)
        bus.subscribe_handler(EventType.BAR, self._on_bar)

    async def _on_bar(self, event: Event) -> None:
        bar = event.data
        self._current_prices[bar.symbol] = bar.close

    async def _on_fill(self, event: Event) -> None:
        fill: Fill = event.data
        self.total_commission += fill.commission
        self._strategy_pnl.setdefault(fill.strategy_id, Decimal("0"))

        if fill.side == Side.BUY:
            self._open_fills.setdefault(fill.symbol, []).append(fill)
            # Cash was already deducted by broker; just log
            logger.debug(f"Portfolio: opened {fill.symbol} @ {fill.price}")

        elif fill.side == Side.SELL:
            entry_fills = self._open_fills.get(fill.symbol, [])
            if not entry_fills:
                logger.warning(f"Portfolio: sell fill for {fill.symbol} but no open position")
                return

            # FIFO matching — first lot in, first lot out
            remaining_sell_qty = fill.quantity
            realized_pnl = Decimal("0")
            total_entry_commission = Decimal("0")

            while remaining_sell_qty > 0 and entry_fills:
                entry_fill = entry_fills[0]

                if entry_fill.quantity <= remaining_sell_qty:
                    # Close this entire entry lot
                    lot_pnl = (fill.price - entry_fill.price) * entry_fill.quantity
                    realized_pnl += lot_pnl
                    total_entry_commission += entry_fill.commission
                    remaining_sell_qty -= entry_fill.quantity
                    entry_fills.pop(0)
                else:
                    # Partial close of this entry lot
                    lot_pnl = (fill.price - entry_fill.price) * remaining_sell_qty
                    realized_pnl += lot_pnl
                    total_entry_commission += entry_fill.commission * (
                        remaining_sell_qty / entry_fill.quantity
                    )
                    entry_fill.quantity -= remaining_sell_qty
                    remaining_sell_qty = Decimal("0")

            # Deduct exit commission from realized PnL
            realized_pnl -= fill.commission
            realized_pnl -= total_entry_commission  # include entry commission in PnL

            self.total_realized_pnl += realized_pnl
            self._strategy_pnl[fill.strategy_id] += realized_pnl

            # Record the closed trade
            self.closed_trades.append(ClosedTrade(
                symbol=fill.symbol,
                strategy_id=fill.strategy_id,
                entry_price=entry_fills[0].price if entry_fills else fill.price,
                exit_price=fill.price,
                quantity=fill.quantity,
                side=Side.BUY,  # we exited a long
                entry_time=entry_fills[0].timestamp if entry_fills else fill.timestamp,
                exit_time=fill.timestamp,
                commission_paid=fill.commission + total_entry_commission,
                realized_pnl=realized_pnl,
            ))

            # Update portfolio value history for metrics
            current_value = self.portfolio_value
            self._value_history.append(current_value)
            if current_value > self._peak_value:
                self._peak_value = current_value

            # Notify risk manager of PnL change
            await bus.publish(Event(
                type=EventType.PNL_UPDATE,
                data={
                    "realized_pnl": realized_pnl,
                    "total_realized_pnl": self.total_realized_pnl,
                    "strategy_id": fill.strategy_id,
                },
                source="Portfolio",
            ))

            pnl_sign = "+" if realized_pnl >= 0 else ""
            logger.info(
                f"Portfolio: closed {fill.symbol} "
                f"PnL={pnl_sign}{realized_pnl:.2f} | "
                f"Total realized={self.total_realized_pnl:.2f}"
            )

    @property
    def unrealized_pnl(self) -> Decimal:
        total = Decimal("0")
        for symbol, fills in self._open_fills.items():
            current_price = self._current_prices.get(symbol)
            if not current_price:
                continue
            for fill in fills:
                total += (current_price - fill.price) * fill.quantity
        return total

    @property
    def portfolio_value(self) -> Decimal:
        return self.cash + self.unrealized_pnl

    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        winners = sum(1 for t in self.closed_trades if t.realized_pnl > 0)
        return winners / len(self.closed_trades)

    def sharpe_ratio(self, risk_free_rate: float = 0.04) -> Optional[float]:
        """
        Annualized Sharpe ratio from portfolio value history.
        Requires at least 30 data points to be meaningful.
        """
        if len(self._value_history) < 30:
            return None

        values = [float(v) for v in self._value_history]
        returns = [
            (values[i] - values[i-1]) / values[i-1]
            for i in range(1, len(values))
        ]
        if not returns:
            return None

        n = len(returns)
        mean_return = sum(returns) / n
        variance = sum((r - mean_return) ** 2 for r in returns) / n
        std_return = math.sqrt(variance) if variance > 0 else 0

        if std_return == 0:
            return None

        # Annualize (assuming ~252 trading days, bars_per_day varies by timeframe)
        annualized_return = mean_return * 252
        annualized_std = std_return * math.sqrt(252)
        return (annualized_return - risk_free_rate) / annualized_std

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough decline as a percentage."""
        if len(self._value_history) < 2:
            return 0.0

        values = [float(v) for v in self._value_history]
        peak = values[0]
        max_dd = 0.0

        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def get_snapshot(self) -> PortfolioSnapshot:
        """Full portfolio state — call this for API responses and dashboard updates."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_trades = [t for t in self.closed_trades if t.exit_time.strftime("%Y-%m-%d") == today]
        daily_pnl = sum((t.realized_pnl for t in today_trades), Decimal("0"))

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            total_realized_pnl=self.total_realized_pnl,
            total_unrealized_pnl=self.unrealized_pnl,
            total_commission=self.total_commission,
            n_open_positions=len(self._open_fills),
            n_closed_trades=len(self.closed_trades),
            win_rate=self.win_rate(),
            sharpe_ratio=self.sharpe_ratio(),
            max_drawdown=self.max_drawdown(),
            daily_pnl=daily_pnl,
        )

    def get_strategy_pnl(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self._strategy_pnl.items()}
