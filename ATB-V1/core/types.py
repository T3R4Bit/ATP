"""
core/types.py
Shared domain types. Every layer imports from here — nothing else shares types.
These are plain dataclasses (no ORM, no serialization logic) so they move fast
between components without any framework overhead.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalStrength(str, Enum):
    STRONG = "strong"    # full position
    MEDIUM = "medium"    # half position
    WEAK = "weak"        # quarter position


@dataclass
class Bar:
    """OHLCV candle. The fundamental unit of market data in the system."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timeframe: str = "1m"   # "1m", "5m", "15m", "1h", "1d"

    @property
    def mid(self) -> Decimal:
        return (self.high + self.low) / 2

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class Signal:
    """
    Output of a strategy. Tells the execution engine *what* to do.
    The strategy never talks to the broker — it only emits signals.
    This separation is what makes strategies independently testable.
    """
    strategy_id: str
    symbol: str
    side: Side
    strength: SignalStrength
    timestamp: datetime
    price: Decimal                    # reference price at signal time
    reason: str = ""                  # human-readable explanation
    metadata: dict = field(default_factory=dict)  # strategy-specific data


@dataclass
class Order:
    """
    Instruction sent to the broker (paper or live).
    Created by the execution engine from a Signal + risk checks.
    """
    order_id: str
    strategy_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None   # None = market order
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Fill:
    """
    Confirmed execution. Created by the broker after an order is filled.
    This is what the portfolio manager uses to update state.
    """
    fill_id: str
    order_id: str
    strategy_id: str
    symbol: str
    side: Side
    quantity: Decimal
    price: Decimal                    # actual fill price (may differ from order)
    commission: Decimal
    timestamp: datetime
    is_paper: bool = True


@dataclass
class Position:
    """Current open position in a symbol."""
    symbol: str
    quantity: Decimal                 # positive = long, negative = short
    average_cost: Decimal
    opened_at: datetime
    strategy_id: str

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        return (current_price - self.average_cost) * self.quantity

    def market_value(self, current_price: Decimal) -> Decimal:
        return abs(self.quantity) * current_price
