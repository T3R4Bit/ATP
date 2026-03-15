"""
features/pipeline.py + indicators
Feature engineering layer.

Design decisions:
- Indicators are stateful objects that maintain their own rolling window
- FeaturePipeline composes indicators and produces a flat feature dict
- Strategies consume feature dicts — they never compute indicators themselves
- All indicators handle the warm-up period (returns None until enough data)

This separation means you can:
- Unit test indicators in isolation with synthetic data
- Swap indicator implementations without touching strategies
- Add new features without modifying existing strategies
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from decimal import Decimal
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Base indicator
# ---------------------------------------------------------------------------

class BaseIndicator(ABC):
    """
    All indicators maintain their own internal state (a rolling deque of values).
    They return None during warm-up so strategies can handle insufficient data gracefully.
    """

    def __init__(self, period: int, name: str) -> None:
        self.period = period
        self.name = name
        self._values: deque[Decimal] = deque(maxlen=period + 50)  # buffer for calculations

    @abstractmethod
    def update(self, value: Decimal) -> Optional[Decimal]:
        """Feed a new value. Returns the indicator value or None if still warming up."""
        ...

    @property
    def is_ready(self) -> bool:
        return len(self._values) >= self.period

    def reset(self) -> None:
        self._values.clear()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class RSI(BaseIndicator):
    """
    Wilder's RSI using exponential smoothing for gains/losses.
    More accurate than simple average RSI, matches TradingView's default.
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__(period, f"rsi_{period}")
        self._prev_close: Optional[Decimal] = None
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None
        self._count = 0

    def update(self, close: Decimal) -> Optional[Decimal]:
        if self._prev_close is None:
            self._prev_close = close
            return None

        change = close - self._prev_close
        gain = max(change, Decimal(0))
        loss = max(-change, Decimal(0))
        self._prev_close = close
        self._count += 1

        if self._count < self.period:
            self._values.append(close)
            # Accumulate for initial average
            if not hasattr(self, '_gains'):
                self._gains: list[Decimal] = []
                self._losses: list[Decimal] = []
            self._gains.append(gain)
            self._losses.append(loss)
            return None

        if self._count == self.period:
            # First average — simple mean
            self._avg_gain = sum(self._gains) / Decimal(self.period)
            self._avg_loss = sum(self._losses) / Decimal(self.period)
        else:
            # Wilder's smoothing
            alpha = Decimal(1) / Decimal(self.period)
            self._avg_gain = (self._avg_gain * (1 - alpha)) + (gain * alpha)
            self._avg_loss = (self._avg_loss * (1 - alpha)) + (loss * alpha)

        if self._avg_loss == 0:
            return Decimal(100)

        rs = self._avg_gain / self._avg_loss
        return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA(BaseIndicator):
    """
    Exponential Moving Average.
    k = 2 / (period + 1) is the standard smoothing factor.
    """

    def __init__(self, period: int) -> None:
        super().__init__(period, f"ema_{period}")
        self._ema: Optional[Decimal] = None
        self._k = Decimal(2) / Decimal(period + 1)
        self._seed_values: list[Decimal] = []

    def update(self, close: Decimal) -> Optional[Decimal]:
        self._values.append(close)

        if len(self._values) < self.period:
            self._seed_values.append(close)
            return None

        if self._ema is None:
            # Seed with SMA
            self._ema = sum(self._seed_values) / Decimal(len(self._seed_values))
            return self._ema

        self._ema = (close * self._k) + (self._ema * (Decimal(1) - self._k))
        return self._ema


# ---------------------------------------------------------------------------
# ATR — Average True Range
# ---------------------------------------------------------------------------

class ATR(BaseIndicator):
    """
    Average True Range — measures volatility.
    Used for position sizing and stop placement.
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__(period, f"atr_{period}")
        self._prev_close: Optional[Decimal] = None
        self._atr: Optional[Decimal] = None
        self._tr_buffer: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[Decimal]:
        if self._prev_close is None:
            self._prev_close = close
            return None

        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )
        self._prev_close = close
        self._tr_buffer.append(tr)

        if len(self._tr_buffer) < self.period:
            return None

        if self._atr is None:
            self._atr = sum(self._tr_buffer[:self.period]) / Decimal(self.period)
            return self._atr

        # Wilder's smoothing
        self._atr = (self._atr * Decimal(self.period - 1) + tr) / Decimal(self.period)
        return self._atr


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

class VWAP(BaseIndicator):
    """
    Volume-Weighted Average Price.
    Resets at the start of each trading session (daily by default).
    """

    def __init__(self) -> None:
        super().__init__(period=1, name="vwap")
        self._cumulative_pv = Decimal(0)  # price * volume
        self._cumulative_volume = Decimal(0)
        self._session_date: Optional[str] = None

    def update(
        self,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        date_str: str,  # "YYYY-MM-DD" for session detection
    ) -> Optional[Decimal]:
        # Reset at session boundary
        if self._session_date != date_str:
            self._cumulative_pv = Decimal(0)
            self._cumulative_volume = Decimal(0)
            self._session_date = date_str

        typical_price = (high + low + close) / Decimal(3)
        self._cumulative_pv += typical_price * volume
        self._cumulative_volume += volume

        if self._cumulative_volume == 0:
            return None

        return self._cumulative_pv / self._cumulative_volume


# ---------------------------------------------------------------------------
# Feature Pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Composes multiple indicators into a single feature dict.

    Each strategy gets its own pipeline instance so indicators
    don't share state across strategies.

    Usage:
        pipeline = FeaturePipeline(symbol="BTCUSDT", timeframe="5m")
        features = pipeline.process(bar)
        # features = {"rsi_14": 45.2, "ema_20": 42100.5, "atr_14": 850.3, ...}
    """

    def __init__(self, symbol: str, timeframe: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe

        # Default indicator set — override by passing custom config
        self._rsi = RSI(period=14)
        self._ema_fast = EMA(period=9)
        self._ema_slow = EMA(period=21)
        self._ema_200 = EMA(period=200)
        self._atr = ATR(period=14)
        self._vwap = VWAP()

        self._bar_count = 0

    def process(self, bar: Any) -> dict[str, Any]:
        """
        Feed a Bar and return the current feature dict.
        Values are None for indicators that haven't warmed up yet.
        Strategies must check for None before using indicator values.
        """
        if bar.symbol != self.symbol or bar.timeframe != self.timeframe:
            return {}

        self._bar_count += 1
        date_str = bar.timestamp.strftime("%Y-%m-%d")

        rsi = self._rsi.update(bar.close)
        ema_fast = self._ema_fast.update(bar.close)
        ema_slow = self._ema_slow.update(bar.close)
        ema_200 = self._ema_200.update(bar.close)
        atr = self._atr.update(bar.high, bar.low, bar.close)
        vwap = self._vwap.update(bar.high, bar.low, bar.close, bar.volume, date_str)

        features: dict[str, Any] = {
            # Raw bar data
            "symbol": bar.symbol,
            "timeframe": bar.timeframe,
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,

            # Indicators
            "rsi_14": rsi,
            "ema_9": ema_fast,
            "ema_21": ema_slow,
            "ema_200": ema_200,
            "atr_14": atr,
            "vwap": vwap,

            # Derived features (computed only when base indicators are ready)
            "ema_cross_bullish": (
                ema_fast is not None
                and ema_slow is not None
                and ema_fast > ema_slow
            ),
            "above_vwap": (
                vwap is not None and bar.close > vwap
            ),
            "rsi_oversold": rsi is not None and rsi < 30,
            "rsi_overbought": rsi is not None and rsi > 70,
            "bar_count": self._bar_count,
        }

        return features

    def is_ready(self, min_bars: int = 200) -> bool:
        """Returns True once we have enough bars for all indicators to be reliable."""
        return self._bar_count >= min_bars

    def reset(self) -> None:
        self._rsi.reset()
        self._ema_fast.reset()
        self._ema_slow.reset()
        self._ema_200.reset()
        self._atr.reset()
        self._vwap.reset()
        self._bar_count = 0
