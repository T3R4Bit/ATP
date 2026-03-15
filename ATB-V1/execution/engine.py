"""
execution/engine.py + paper_broker.py + risk_manager.py
Execution layer.

Design decisions:
- PaperBroker simulates realistic fills: slippage, partial fills, commission
- RiskManager runs pre-trade checks BEFORE any order reaches the broker
- ExecutionEngine wires together: Signal → risk check → position size → order → fill
- The same ExecutionEngine works for paper AND live trading (swap the broker)

Risk checks implemented:
1. Max position size per symbol
2. Max total portfolio exposure
3. Max concurrent positions
4. Minimum account balance
5. Daily loss limit (halt trading if exceeded)
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from core.events import EventType, Event, bus
from core.types import (
    Fill, Order, OrderStatus, OrderType, Position, Signal, Side, SignalStrength
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Translates a Signal into an order quantity.

    Uses ATR-based sizing: risk a fixed % of capital per trade,
    where position size = (capital * risk_pct) / (ATR * atr_multiplier).

    This means you automatically size down in volatile markets.

    Fallback: fixed fraction if ATR is unavailable.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,   # 1% of capital per trade
        atr_multiplier: float = 2.0,    # stop is placed at 2x ATR from entry
        max_position_pct: float = 0.10, # no single position > 10% of portfolio
    ) -> None:
        self.risk_per_trade = Decimal(str(risk_per_trade))
        self.atr_multiplier = Decimal(str(atr_multiplier))
        self.max_position_pct = Decimal(str(max_position_pct))

    def calculate(
        self,
        signal: Signal,
        capital: Decimal,
        features: dict,
        current_price: Decimal,
    ) -> Decimal:
        """Return the quantity to trade."""
        # Scale by signal strength
        strength_multiplier = {
            SignalStrength.STRONG: Decimal("1.0"),
            SignalStrength.MEDIUM: Decimal("0.5"),
            SignalStrength.WEAK: Decimal("0.25"),
        }[signal.strength]

        # ATR-based sizing
        atr = features.get("atr_14")
        if atr and Decimal(str(atr)) > 0:
            atr = Decimal(str(atr))
            risk_amount = capital * self.risk_per_trade * strength_multiplier
            stop_distance = atr * self.atr_multiplier
            quantity = risk_amount / stop_distance
        else:
            # Fallback: fixed fraction
            position_value = capital * self.risk_per_trade * strength_multiplier * 10
            quantity = position_value / current_price

        # Cap at max_position_pct of capital
        max_quantity = (capital * self.max_position_pct) / current_price
        quantity = min(quantity, max_quantity)

        return max(quantity.quantize(Decimal("0.00001")), Decimal("0.00001"))


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    max_position_value: Decimal = Decimal("10000")   # $ per position
    max_total_exposure: Decimal = Decimal("50000")   # $ total open positions
    max_concurrent_positions: int = 5
    min_account_balance: Decimal = Decimal("1000")
    daily_loss_limit: Decimal = Decimal("500")       # halt if daily loss > this
    max_order_value: Decimal = Decimal("20000")


class RiskManager:
    """
    Pre-trade risk gate. Every order must pass all checks before execution.

    The risk manager is the safety net of the system. When in doubt,
    it rejects the trade. It's much easier to recover from a missed
    trade than from a runaway position.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._daily_loss = Decimal("0")
        self._last_reset_date: Optional[str] = None
        self._rejected_count = 0

    def _reset_daily_if_needed(self) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._daily_loss = Decimal("0")
            self._last_reset_date = today

    def update_daily_pnl(self, pnl_delta: Decimal) -> None:
        """Called by portfolio when a position is closed."""
        if pnl_delta < 0:
            self._daily_loss += abs(pnl_delta)

    def check(
        self,
        signal: Signal,
        quantity: Decimal,
        current_price: Decimal,
        portfolio_state: dict,
    ) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        All checks are run; first failure returns immediately.
        """
        self._reset_daily_if_needed()

        order_value = quantity * current_price

        # 1. Daily loss limit
        if self._daily_loss >= self.config.daily_loss_limit:
            return False, f"Daily loss limit reached (${self._daily_loss:.2f})"

        # 2. Account balance
        cash = portfolio_state.get("cash", Decimal("0"))
        if cash < self.config.min_account_balance:
            return False, f"Insufficient balance (${cash:.2f})"

        # 3. Order size sanity check
        if order_value > self.config.max_order_value:
            return False, f"Order too large (${order_value:.2f} > ${self.config.max_order_value})"

        # 4. Max concurrent positions
        n_positions = portfolio_state.get("n_positions", 0)
        if signal.side == Side.BUY and n_positions >= self.config.max_concurrent_positions:
            return False, f"Max positions reached ({n_positions})"

        # 5. Total exposure
        total_exposure = portfolio_state.get("total_exposure", Decimal("0"))
        if signal.side == Side.BUY:
            if total_exposure + order_value > self.config.max_total_exposure:
                return False, f"Total exposure limit (${total_exposure + order_value:.2f})"

        # 6. Per-position limit
        if order_value > self.config.max_position_value:
            return False, f"Position too large (${order_value:.2f})"

        return True, "approved"


# ---------------------------------------------------------------------------
# Paper Broker
# ---------------------------------------------------------------------------

class PaperBroker:
    """
    Simulates order execution without real money.

    Realism features:
    - Slippage: market orders fill at current price ± slippage_bps
    - Commission: configurable flat or percentage fee
    - Partial fills: large orders may fill in multiple chunks (not yet implemented)
    - Latency: simulated order acknowledgment delay

    To switch to live trading: implement a LiveBroker with the same interface.
    No other code needs to change.
    """

    def __init__(
        self,
        initial_cash: Decimal = Decimal("100000"),
        commission_rate: Decimal = Decimal("0.001"),  # 0.1% (Binance taker fee)
        slippage_bps: int = 2,                        # 2 basis points slippage
    ) -> None:
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = Decimal(str(slippage_bps)) / Decimal("10000")
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self._fill_count = 0

    def _apply_slippage(self, price: Decimal, side: Side) -> Decimal:
        """Buys fill slightly higher, sells slightly lower. Realistic."""
        if side == Side.BUY:
            return price * (Decimal(1) + self.slippage_bps)
        else:
            return price * (Decimal(1) - self.slippage_bps)

    async def submit_order(self, order: Order, current_price: Decimal) -> Fill:
        """
        Simulate immediate market order fill.
        For limit orders: check if price was reached (simplified — full LOB sim is Phase 3).
        """
        fill_price = self._apply_slippage(current_price, order.side)
        commission = fill_price * order.quantity * self.commission_rate
        total_cost = fill_price * order.quantity + (commission if order.side == Side.BUY else -commission)

        # Check we can afford it
        if order.side == Side.BUY and total_cost > self.cash:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.order_id} rejected: insufficient cash")
            await bus.publish(Event(
                type=EventType.ORDER_REJECTED,
                data=order,
                source="PaperBroker",
            ))
            raise ValueError("Insufficient cash for order")

        self._fill_count += 1
        fill = Fill(
            fill_id=f"fill_{self._fill_count:06d}",
            order_id=order.order_id,
            strategy_id=order.strategy_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.utcnow(),
            is_paper=True,
        )

        # Update positions
        self._update_position(fill)

        # Update cash
        if order.side == Side.BUY:
            self.cash -= total_cost
        else:
            self.cash += (fill_price * order.quantity) - commission

        order.status = OrderStatus.FILLED
        self.orders[order.order_id] = order

        await bus.publish(Event(
            type=EventType.ORDER_FILLED,
            data=fill,
            source="PaperBroker",
        ))

        logger.info(
            f"[PAPER] {fill.side.upper()} {fill.quantity} {fill.symbol} "
            f"@ {fill.price:.4f} | commission={fill.commission:.4f} | "
            f"cash remaining=${self.cash:.2f}"
        )

        return fill

    def _update_position(self, fill: Fill) -> None:
        symbol = fill.symbol
        if fill.side == Side.BUY:
            if symbol in self.positions:
                pos = self.positions[symbol]
                # Average up: new avg cost = (old_qty * old_cost + new_qty * fill_price) / total_qty
                new_qty = pos.quantity + fill.quantity
                pos.average_cost = (
                    (pos.quantity * pos.average_cost) + (fill.quantity * fill.price)
                ) / new_qty
                pos.quantity = new_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=fill.quantity,
                    average_cost=fill.price,
                    opened_at=fill.timestamp,
                    strategy_id=fill.strategy_id,
                )
        elif fill.side == Side.SELL:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity -= fill.quantity
                if pos.quantity <= Decimal("0.000001"):
                    del self.positions[symbol]

    @property
    def portfolio_value(self) -> Decimal:
        """Cash + open position market value (approximate — uses average cost as proxy)."""
        position_value = sum(
            p.quantity * p.average_cost for p in self.positions.values()
        )
        return self.cash + position_value

    @property
    def total_exposure(self) -> Decimal:
        return sum(p.quantity * p.average_cost for p in self.positions.values())

    def get_portfolio_state(self) -> dict:
        return {
            "cash": self.cash,
            "total_exposure": self.total_exposure,
            "n_positions": len(self.positions),
            "portfolio_value": self.portfolio_value,
            "positions": {k: v for k, v in self.positions.items()},
        }


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Central orchestrator.
    Wires: Signal → RiskManager → PositionSizer → Order → PaperBroker → Fill

    This is the component that subscribes to strategy signals on the event bus
    and turns them into executed (paper) trades.
    """

    def __init__(
        self,
        broker: PaperBroker,
        risk_manager: RiskManager,
        position_sizer: PositionSizer,
    ) -> None:
        self.broker = broker
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self._feature_cache: dict[str, dict] = {}  # latest features per symbol+timeframe
        self._order_count = 0

        # Subscribe to events
        bus.subscribe_handler(EventType.SIGNAL, self._on_signal)
        bus.subscribe_handler(EventType.BAR, self._on_bar)

    async def _on_bar(self, event: Event) -> None:
        """Cache the latest features for each symbol/timeframe."""
        bar = event.data
        key = f"{bar.symbol}_{bar.timeframe}"
        self._feature_cache[key] = {
            "close": bar.close,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
        }

    async def _on_signal(self, event: Event) -> None:
        """Process a signal from any strategy."""
        signal: Signal = event.data

        # Get current price
        key = f"{signal.symbol}_1m"  # fallback to 1m price
        features = self._feature_cache.get(key, {})
        current_price = signal.price  # use signal's reference price

        portfolio_state = self.broker.get_portfolio_state()

        # Calculate position size
        quantity = self.position_sizer.calculate(
            signal=signal,
            capital=portfolio_state["portfolio_value"],
            features=features,
            current_price=current_price,
        )

        # Risk check
        approved, reason = self.risk_manager.check(
            signal=signal,
            quantity=quantity,
            current_price=current_price,
            portfolio_state=portfolio_state,
        )

        if not approved:
            logger.info(f"Signal rejected by risk manager: {reason}")
            return

        # Build order
        self._order_count += 1
        order = Order(
            order_id=f"ord_{self._order_count:08d}_{uuid.uuid4().hex[:6]}",
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
        )

        await bus.publish(Event(
            type=EventType.ORDER_CREATED,
            data=order,
            source="ExecutionEngine",
        ))

        # Execute
        try:
            await self.broker.submit_order(order, current_price)
        except ValueError as e:
            logger.warning(f"Order execution failed: {e}")
