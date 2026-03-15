"""
core/events.py
Async event bus. Components publish events; other components subscribe.
This is what keeps layers decoupled — the data feed doesn't know who
is consuming bars, and the strategy doesn't know who is listening to signals.

Design: simple in-process pub/sub using asyncio.Queue.
For distributed scaling (Phase 6), swap the backend for Redis Streams
or Kafka without changing any subscriber code.
"""

from __future__ import annotations
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # Market data events
    BAR = "bar"                        # new OHLCV candle
    TICK = "tick"                      # raw price tick
    ORDERBOOK_UPDATE = "orderbook"

    # Strategy events
    SIGNAL = "signal"                  # strategy emitted a signal
    SIGNAL_CANCELLED = "signal_cancelled"

    # Execution events
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Portfolio events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    PNL_UPDATE = "pnl_update"

    # System events
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""


# Type alias for async event handlers
Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async publish/subscribe event bus.

    Usage:
        bus = EventBus()

        @bus.subscribe(EventType.BAR)
        async def on_bar(event: Event):
            bar = event.data
            print(f"New bar: {bar.symbol} close={bar.close}")

        await bus.publish(Event(type=EventType.BAR, data=bar))
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(self, *event_types: EventType):
        """Decorator to register an async handler for one or more event types."""
        def decorator(handler: Handler) -> Handler:
            for et in event_types:
                self._subscribers[et].append(handler)
                logger.debug(f"Subscribed {handler.__name__} to {et}")
            return handler
        return decorator

    def subscribe_handler(self, event_type: EventType, handler: Handler) -> None:
        """Programmatic (non-decorator) subscription."""
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        """Enqueue an event for dispatch."""
        await self._queue.put(event)

    async def publish_sync(self, event: Event) -> None:
        """Dispatch an event immediately (bypasses queue, for high-frequency paths)."""
        await self._dispatch(event)

    async def _dispatch(self, event: Event) -> None:
        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            return
        results = await asyncio.gather(
            *[h(event) for h in handlers],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Handler error on {event.type}: {r}", exc_info=r)

    async def run(self) -> None:
        """
        Main dispatch loop. Run this as a background task.
        Processes queued events one by one — sequential within each event type
        but all handlers for a single event run concurrently.
        """
        self._running = True
        logger.info("EventBus running")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"EventBus dispatch error: {e}", exc_info=True)

    async def stop(self) -> None:
        self._running = False
        # Drain remaining events
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch(event)
            except asyncio.QueueEmpty:
                break
        logger.info("EventBus stopped")

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


# Module-level singleton — import this everywhere
bus = EventBus()
