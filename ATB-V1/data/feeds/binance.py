"""
data/feeds/binance.py
Binance WebSocket market data feed.

Design decisions:
- Extends AbstractDataFeed so any feed (Coinbase, Kraken, CSV) is swappable
- Reconnects automatically with exponential backoff
- Emits BAR events on the global event bus — nothing downstream is coupled to Binance
- Aggregates raw kline data into Bar objects before publishing

To use a different exchange: subclass AbstractDataFeed, implement _connect()
and _parse_message(), and swap it in core/config.py.
"""

from __future__ import annotations
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import List

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.events import EventType, Event, bus
from core.types import Bar

logger = logging.getLogger(__name__)


class AbstractDataFeed(ABC):
    """
    Interface all data feeds must implement.
    The engine only knows about this interface — never about Binance directly.
    """

    def __init__(self, symbols: List[str], timeframes: List[str]) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.timeframes = timeframes
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Connect and begin streaming. Should run indefinitely until stop()."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully disconnect."""
        ...

    @abstractmethod
    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> List[Bar]:
        """Fetch recent historical candles for warm-up."""
        ...


class BinanceFeed(AbstractDataFeed):
    """
    Streams live klines from Binance via WebSocket.
    Publishes EventType.BAR for each closed candle.

    Binance kline stream fires on every tick but marks candles as closed
    only when the interval completes — we filter to closed candles only,
    so strategies always receive complete bars.
    """

    WS_BASE = "wss://stream.binance.us:9443/stream"
    REST_BASE = "https://api.binance.us/api/v3"
    MAX_RECONNECT_ATTEMPTS = 10
    BASE_RECONNECT_DELAY = 1.0   # seconds, doubles each retry

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        super().__init__(symbols, timeframes)
        self._api_key = api_key
        self._api_secret = api_secret
        self._ws = None
        self._reconnect_delay = self.BASE_RECONNECT_DELAY

    def _build_stream_url(self) -> str:
        """
        Build combined stream URL.
        e.g. /stream?streams=btcusdt@kline_1m/ethusdt@kline_1m/btcusdt@kline_5m
        """
        streams = [
            f"{sym.lower()}@kline_{tf}"
            for sym in self.symbols
            for tf in self.timeframes
        ]
        return f"{self.WS_BASE}?streams={'/'.join(streams)}"

    def _parse_kline(self, data: dict) -> Bar | None:
        """
        Parse Binance kline message into a Bar.
        Only returns a Bar if the candle is closed (k.x == True).
        """
        stream = data.get("stream", "")
        kline = data.get("data", {}).get("k", {})

        if not kline.get("x"):  # x = is candle closed
            return None

        symbol = kline["s"]
        timeframe = kline["i"]

        return Bar(
            symbol=symbol,
            timestamp=datetime.utcfromtimestamp(kline["t"] / 1000),
            open=Decimal(kline["o"]),
            high=Decimal(kline["h"]),
            low=Decimal(kline["l"]),
            close=Decimal(kline["c"]),
            volume=Decimal(kline["v"]),
            timeframe=timeframe,
        )

    async def start(self) -> None:
        self._running = True
        attempt = 0

        while self._running and attempt < self.MAX_RECONNECT_ATTEMPTS:
            try:
                url = self._build_stream_url()
                logger.info(f"Connecting to Binance WebSocket: {url[:80]}...")

                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = self.BASE_RECONNECT_DELAY
                    attempt = 0  # reset on successful connection
                    logger.info("Binance WebSocket connected")

                    await bus.publish(Event(
                        type=EventType.ENGINE_STARTED,
                        data={"feed": "binance", "symbols": self.symbols},
                        source="BinanceFeed",
                    ))

                    async for raw_message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw_message)
                            bar = self._parse_kline(data)
                            if bar:
                                await bus.publish(Event(
                                    type=EventType.BAR,
                                    data=bar,
                                    source="BinanceFeed",
                                ))
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse message: {e}")

            except (ConnectionClosed, WebSocketException) as e:
                if not self._running:
                    break
                attempt += 1
                logger.warning(
                    f"WebSocket disconnected ({e}). "
                    f"Reconnecting in {self._reconnect_delay}s "
                    f"(attempt {attempt}/{self.MAX_RECONNECT_ATTEMPTS})"
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

            except Exception as e:
                logger.error(f"Unexpected feed error: {e}", exc_info=True)
                if not self._running:
                    break
                await asyncio.sleep(self._reconnect_delay)

        if attempt >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error("Max reconnect attempts reached. Feed stopped.")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("BinanceFeed stopped")

    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[Bar]:
        """
        Fetch historical klines via REST for strategy warm-up.
        Called once at startup so indicators have enough data before live trading.
        """
        import aiohttp

        url = f"{self.REST_BASE}/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": limit,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                raw = await resp.json()

        bars = []
        for k in raw:
            bars.append(Bar(
                symbol=symbol.upper(),
                timestamp=datetime.utcfromtimestamp(k[0] / 1000),
                open=Decimal(str(k[1])),
                high=Decimal(str(k[2])),
                low=Decimal(str(k[3])),
                close=Decimal(str(k[4])),
                volume=Decimal(str(k[5])),
                timeframe=timeframe,
            ))
        return bars


class HistoricalFeed(AbstractDataFeed):
    """
    Replays historical bars from a list for backtesting.
    Publishes the same BAR events as live feeds — strategies see no difference.

    This is the key design insight: backtesting and live trading
    use the exact same strategy code because both feed through the event bus.
    """

    def __init__(
        self,
        bars: list[Bar],
        speed: float = 0.0,  # 0 = as fast as possible, >0 = simulated real time
    ) -> None:
        symbols = list({b.symbol for b in bars})
        timeframes = list({b.timeframe for b in bars})
        super().__init__(symbols, timeframes)
        self._bars = sorted(bars, key=lambda b: b.timestamp)
        self._speed = speed

    async def start(self) -> None:
        self._running = True
        logger.info(f"HistoricalFeed: replaying {len(self._bars)} bars")

        for bar in self._bars:
            if not self._running:
                break
            await bus.publish(Event(
                type=EventType.BAR,
                data=bar,
                source="HistoricalFeed",
            ))
            if self._speed > 0:
                await asyncio.sleep(self._speed)

        logger.info("HistoricalFeed: replay complete")

    async def stop(self) -> None:
        self._running = False

    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[Bar]:
        return [
            b for b in self._bars
            if b.symbol == symbol and b.timeframe == timeframe
        ][-limit:]