"""
tests/unit/test_indicators.py
Unit tests for indicators and strategies.

Run with: pytest tests/ -v --cov=features --cov=strategies

Design: tests are pure Python — no async, no event bus, no broker.
This is only possible because indicators and strategies are pure functions
of their inputs. If you find yourself needing to mock the broker in a
strategy test, the strategy is doing too much.
"""

import pytest
from decimal import Decimal
from datetime import datetime

from features.pipeline import RSI, EMA, ATR, FeaturePipeline
from strategies.strategy import RSIReversalStrategy, EMACrossoverStrategy
from core.types import Bar, Side


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bar(close: float, high: float = None, low: float = None,
             volume: float = 1000.0, symbol: str = "BTCUSDT",
             timeframe: str = "5m") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        open=Decimal(str(close)),
        high=Decimal(str(high or close * 1.001)),
        low=Decimal(str(low or close * 0.999)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
        timeframe=timeframe,
    )


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestRSI:

    def test_returns_none_before_warmup(self):
        rsi = RSI(period=14)
        for i in range(13):
            result = rsi.update(Decimal(str(100 + i)))
        assert result is None

    def test_returns_value_after_warmup(self):
        rsi = RSI(period=14)
        for i in range(20):
            result = rsi.update(Decimal(str(100 + i)))
        assert result is not None
        assert 0 <= float(result) <= 100

    def test_rsi_100_on_all_gains(self):
        """If every candle is higher, RSI should approach 100."""
        rsi = RSI(period=14)
        results = []
        for i in range(50):
            r = rsi.update(Decimal(str(100 + i)))
            if r is not None:
                results.append(float(r))
        assert results[-1] > 90  # should be very high

    def test_rsi_0_on_all_losses(self):
        """If every candle is lower, RSI should approach 0."""
        rsi = RSI(period=14)
        results = []
        for i in range(50):
            r = rsi.update(Decimal(str(100 - i * 0.5)))
            if r is not None:
                results.append(float(r))
        assert results[-1] < 10  # should be very low

    def test_rsi_50_on_equal_ups_downs(self):
        """Alternating up/down bars of equal size should give RSI near 50."""
        rsi = RSI(period=14)
        price = Decimal("100")
        results = []
        for i in range(100):
            if i % 2 == 0:
                price += Decimal("1")
            else:
                price -= Decimal("1")
            r = rsi.update(price)
            if r is not None:
                results.append(float(r))
        # Should converge near 50
        assert abs(results[-1] - 50) < 5


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestEMA:

    def test_returns_none_before_warmup(self):
        ema = EMA(period=9)
        for i in range(8):
            result = ema.update(Decimal("100"))
        assert result is None

    def test_returns_value_after_warmup(self):
        ema = EMA(period=9)
        for i in range(10):
            result = ema.update(Decimal("100"))
        assert result is not None

    def test_ema_follows_price(self):
        """EMA should be closer to recent prices than old ones."""
        ema = EMA(period=5)
        # Warm up at 100
        for _ in range(10):
            ema.update(Decimal("100"))
        # Shift to 200
        for _ in range(10):
            result = ema.update(Decimal("200"))
        # EMA should be well above 100 now
        assert float(result) > 150


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestRSIReversalStrategy:

    def setup_method(self):
        self.strategy = RSIReversalStrategy(
            strategy_id="test_rsi",
            symbol="BTCUSDT",
            timeframe="5m",
            oversold=30.0,
            overbought=70.0,
            use_vwap_filter=False,  # disable for cleaner unit tests
            min_bars=5,
        )

    def _make_features(self, rsi: float, close: float, bar_count: int = 10) -> dict:
        return {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "rsi_14": Decimal(str(rsi)),
            "close": Decimal(str(close)),
            "vwap": Decimal(str(close * 0.99)),
            "bar_count": bar_count,
        }

    def test_no_signal_on_normal_rsi(self):
        features = self._make_features(rsi=50.0, close=40000)
        signal = self.strategy.on_bar(features)
        assert signal is None

    def test_buy_signal_on_oversold(self):
        features = self._make_features(rsi=25.0, close=40000)
        signal = self.strategy.on_bar(features)
        assert signal is not None
        assert signal.side == Side.BUY

    def test_no_buy_if_already_in_position(self):
        # First bar: enter
        self.strategy.on_bar(self._make_features(rsi=25.0, close=40000))
        # Second bar: still oversold — should NOT double up
        signal = self.strategy.on_bar(self._make_features(rsi=28.0, close=39000))
        assert signal is None

    def test_sell_signal_on_overbought(self):
        # Enter
        self.strategy.on_bar(self._make_features(rsi=25.0, close=40000))
        # Now overbought
        signal = self.strategy.on_bar(self._make_features(rsi=75.0, close=45000))
        assert signal is not None
        assert signal.side == Side.SELL

    def test_strong_signal_when_very_oversold(self):
        from core.types import SignalStrength
        features = self._make_features(rsi=15.0, close=40000)
        signal = self.strategy.on_bar(features)
        assert signal is not None
        assert signal.strength == SignalStrength.STRONG

    def test_no_signal_before_warmup(self):
        features = self._make_features(rsi=15.0, close=40000, bar_count=3)
        signal = self.strategy.on_bar(features)
        assert signal is None

    def test_wrong_symbol_ignored(self):
        features = self._make_features(rsi=15.0, close=40000)
        features["symbol"] = "ETHUSDT"  # different symbol
        signal = self.strategy.on_bar(features)
        assert signal is None


class TestFeaturePipeline:

    def test_pipeline_warms_up(self):
        pipeline = FeaturePipeline(symbol="BTCUSDT", timeframe="5m")
        bar = make_bar(40000.0)
        for _ in range(250):
            pipeline.process(bar)
        assert pipeline.is_ready()

    def test_pipeline_returns_features(self):
        pipeline = FeaturePipeline(symbol="BTCUSDT", timeframe="5m")
        bar = make_bar(40000.0)
        features = pipeline.process(bar)
        assert "close" in features
        assert "rsi_14" in features
        assert "ema_9" in features
        assert "vwap" in features

    def test_pipeline_filters_wrong_symbol(self):
        pipeline = FeaturePipeline(symbol="BTCUSDT", timeframe="5m")
        bar = make_bar(40000.0, symbol="ETHUSDT")
        features = pipeline.process(bar)
        assert features == {}
