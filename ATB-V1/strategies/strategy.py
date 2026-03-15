"""
strategies/base.py + rsi_reversal.py
Strategy layer.

Design decisions:
- AbstractStrategy defines a minimal interface: on_bar(features) -> Signal | None
- Strategies are pure functions of feature dicts — no I/O, no broker calls
- Each strategy has a unique ID (used for PnL attribution and comparison)
- Parameters are passed at construction time for easy grid search / optimization
- The StrategyRegistry makes strategies discoverable without hardcoding names

Adding a new strategy:
1. Create a class that extends AbstractStrategy
2. Implement on_bar() and optionally on_fill()
3. Register it with @registry.register("my_strategy_name")
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Type

from core.types import Fill, Signal, Side, SignalStrength

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractStrategy(ABC):
    """
    Base class for all strategies.

    Strategies receive feature dicts (from FeaturePipeline) and emit Signals.
    They never interact with the broker, portfolio, or any I/O system directly.
    """

    def __init__(self, strategy_id: str, symbol: str, timeframe: str) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.timeframe = timeframe
        self._active = True

    @abstractmethod
    def on_bar(self, features: Dict[str, Any]) -> Optional[Signal]:
        """
        Called on every closed bar.
        Return a Signal to trade, or None to do nothing.
        """
        ...

    def on_fill(self, fill: Fill) -> None:
        """
        Called when one of this strategy's orders is filled.
        Override to update internal state (e.g. track position, adjust stops).
        Default: no-op.
        """
        pass

    def on_start(self, historical_features: list[Dict[str, Any]]) -> None:
        """
        Called at startup with historical features for warm-up.
        Override to pre-populate any internal buffers.
        Default: replay each bar through on_bar.
        """
        for features in historical_features:
            self.on_bar(features)

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def _make_signal(
        self,
        side: Side,
        price: Decimal,
        strength: SignalStrength = SignalStrength.MEDIUM,
        reason: str = "",
        metadata: dict | None = None,
    ) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            side=side,
            strength=strength,
            timestamp=datetime.utcnow(),
            price=price,
            reason=reason,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

class StrategyRegistry:
    """
    Central registry for strategy classes.
    Allows dynamic instantiation by name — useful for config-driven setup
    and the API endpoint that lists available strategies.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Type[AbstractStrategy]] = {}

    def register(self, name: str):
        """Decorator: @registry.register("rsi_reversal")"""
        def decorator(cls: Type[AbstractStrategy]) -> Type[AbstractStrategy]:
            self._strategies[name] = cls
            logger.debug(f"Registered strategy: {name}")
            return cls
        return decorator

    def create(
        self,
        name: str,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        **kwargs: Any,
    ) -> AbstractStrategy:
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(self._strategies)}")
        cls = self._strategies[name]
        return cls(strategy_id=strategy_id, symbol=symbol, timeframe=timeframe, **kwargs)

    def list_strategies(self) -> list[str]:
        return list(self._strategies.keys())


registry = StrategyRegistry()


# ---------------------------------------------------------------------------
# RSI Reversal Strategy
# ---------------------------------------------------------------------------

@registry.register("rsi_reversal")
class RSIReversalStrategy(AbstractStrategy):
    """
    RSI Mean Reversion strategy.

    Logic:
    - BUY when RSI < oversold_threshold AND close > VWAP (trend filter)
    - SELL when RSI > overbought_threshold OR stop loss hit
    - Position size scaled by signal strength (strong = RSI < 20, medium = < 30)

    This is a classic mean-reversion setup: buy when price is statistically
    stretched to the downside while still above VWAP (so we're not catching
    a falling knife against the daily trend).

    Parameters:
    - oversold: RSI level to trigger buy signal (default 30)
    - overbought: RSI level to trigger exit (default 70)
    - use_vwap_filter: require price > VWAP for longs (default True)
    """

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        oversold: float = 30.0,
        overbought: float = 70.0,
        use_vwap_filter: bool = True,
        min_bars: int = 20,
    ) -> None:
        super().__init__(strategy_id, symbol, timeframe)
        self.oversold = Decimal(str(oversold))
        self.overbought = Decimal(str(overbought))
        self.use_vwap_filter = use_vwap_filter
        self.min_bars = min_bars
        self._in_position = False
        self._entry_price: Optional[Decimal] = None
        self._bars_in_position = 0

    def on_bar(self, features: Dict[str, Any]) -> Optional[Signal]:
        # Filter to our symbol/timeframe
        if features.get("symbol") != self.symbol:
            return None
        if features.get("timeframe") != self.timeframe:
            return None

        # Wait for warm-up
        if features.get("bar_count", 0) < self.min_bars:
            return None

        rsi = features.get("rsi_14")
        close = features.get("close")
        vwap = features.get("vwap")

        # Can't trade without required indicators
        if rsi is None or close is None:
            return None

        close = Decimal(str(close))
        rsi = Decimal(str(rsi))

        # --- Exit logic (check before entry) ---
        if self._in_position:
            self._bars_in_position += 1

            # Exit on overbought RSI
            if rsi > self.overbought:
                self._in_position = False
                self._bars_in_position = 0
                return self._make_signal(
                    side=Side.SELL,
                    price=close,
                    strength=SignalStrength.STRONG,
                    reason=f"RSI overbought ({rsi:.1f} > {self.overbought})",
                    metadata={"rsi": float(rsi), "bars_held": self._bars_in_position},
                )

            # Time-based exit: don't hold longer than 20 bars
            if self._bars_in_position >= 20:
                self._in_position = False
                self._bars_in_position = 0
                return self._make_signal(
                    side=Side.SELL,
                    price=close,
                    strength=SignalStrength.MEDIUM,
                    reason=f"Max hold period reached ({self._bars_in_position} bars)",
                )

        # --- Entry logic ---
        if not self._in_position and rsi < self.oversold:
            # VWAP trend filter
            if self.use_vwap_filter and vwap is not None:
                if close < Decimal(str(vwap)):
                    return None  # Below VWAP — skip this entry

            # Scale strength based on how oversold
            if rsi < Decimal("20"):
                strength = SignalStrength.STRONG
            elif rsi < Decimal("25"):
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK

            self._in_position = True
            self._entry_price = close
            self._bars_in_position = 0

            return self._make_signal(
                side=Side.BUY,
                price=close,
                strength=strength,
                reason=f"RSI oversold ({rsi:.1f} < {self.oversold})",
                metadata={
                    "rsi": float(rsi),
                    "vwap": float(vwap) if vwap else None,
                    "above_vwap": vwap is not None and close > Decimal(str(vwap)),
                },
            )

        return None

    def on_fill(self, fill: Fill) -> None:
        if fill.strategy_id != self.strategy_id:
            return
        if fill.side == Side.SELL:
            self._in_position = False
            self._entry_price = None
            self._bars_in_position = 0


# ---------------------------------------------------------------------------
# EMA Crossover Strategy
# ---------------------------------------------------------------------------

@registry.register("ema_crossover")
class EMACrossoverStrategy(AbstractStrategy):
    """
    Classic dual EMA crossover — trend following.

    Logic:
    - BUY when fast EMA crosses above slow EMA (golden cross)
    - SELL when fast EMA crosses below slow EMA (death cross)

    Complementary to RSI reversal: run both and compare their performance
    to understand how trend-following vs mean-reversion compare on your assets.
    """

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        fast_period: int = 9,
        slow_period: int = 21,
    ) -> None:
        super().__init__(strategy_id, symbol, timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prev_fast: Optional[Decimal] = None
        self._prev_slow: Optional[Decimal] = None
        self._in_position = False

    def on_bar(self, features: Dict[str, Any]) -> Optional[Signal]:
        if features.get("symbol") != self.symbol:
            return None

        ema_fast = features.get("ema_9")
        ema_slow = features.get("ema_21")
        close = features.get("close")

        if ema_fast is None or ema_slow is None or close is None:
            return None

        ema_fast = Decimal(str(ema_fast))
        ema_slow = Decimal(str(ema_slow))
        close = Decimal(str(close))
        signal = None

        # Golden cross: fast crosses above slow
        if (self._prev_fast is not None
                and self._prev_slow is not None
                and self._prev_fast <= self._prev_slow
                and ema_fast > ema_slow
                and not self._in_position):
            self._in_position = True
            signal = self._make_signal(
                side=Side.BUY,
                price=close,
                strength=SignalStrength.MEDIUM,
                reason=f"EMA golden cross (fast={ema_fast:.2f} > slow={ema_slow:.2f})",
                metadata={"ema_fast": float(ema_fast), "ema_slow": float(ema_slow)},
            )

        # Death cross: fast crosses below slow
        elif (self._prev_fast is not None
              and self._prev_slow is not None
              and self._prev_fast >= self._prev_slow
              and ema_fast < ema_slow
              and self._in_position):
            self._in_position = False
            signal = self._make_signal(
                side=Side.SELL,
                price=close,
                strength=SignalStrength.MEDIUM,
                reason=f"EMA death cross (fast={ema_fast:.2f} < slow={ema_slow:.2f})",
                metadata={"ema_fast": float(ema_fast), "ema_slow": float(ema_slow)},
            )

        self._prev_fast = ema_fast
        self._prev_slow = ema_slow
        return signal
