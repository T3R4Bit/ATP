"""
main.py
Trading engine entry point.

This is the composition root — where all components are instantiated and wired.
Nothing else does dependency injection here; all other modules are pure logic.

Run modes:
  python main.py --mode paper    # live paper trading (Binance data, fake execution)
  python main.py --mode backtest # backtest against historical data
  python main.py --mode api      # API only (for dashboard development)

Usage:
  pip install -r requirements.txt
  python main.py --mode paper --symbols BTCUSDT ETHUSDT --timeframe 5m
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import os
import signal
import sys
from decimal import Decimal

import uvicorn

# Configure logging before any imports that might log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_engine.log"),
    ],
)

logger = logging.getLogger("engine.main")


class StrategyManager:
    """Thin wrapper that holds all active strategy instances."""

    def __init__(self) -> None:
        self.strategies = []

    def add(self, strategy) -> None:
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.strategy_id} ({strategy.__class__.__name__})")

    def get(self, strategy_id: str):
        return next((s for s in self.strategies if s.strategy_id == strategy_id), None)


class TradingEngine:
    """
    Main orchestrator. Wires all components and runs the event loop.

    Startup sequence:
    1. Initialize components (broker, portfolio, risk, strategies)
    2. Warm up indicators with historical data
    3. Connect market data feed
    4. Start event bus dispatch loop
    5. Start API server (runs concurrently with trading loop)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def setup(self) -> None:
        from core.events import bus
        from execution.engine import ExecutionEngine, PositionSizer, RiskConfig, RiskManager
        from execution.engine import PaperBroker
        from features.pipeline import FeaturePipeline
        from portfolio.portfolio import Portfolio
        from strategies.strategy import RSIReversalStrategy, EMACrossoverStrategy

        cfg = self.config
        symbols: list[str] = cfg["symbols"]
        timeframe: str = cfg["timeframe"]

        logger.info("=" * 60)
        logger.info("Trading Engine initializing")
        logger.info(f"  Mode:      {cfg['mode']}")
        logger.info(f"  Symbols:   {symbols}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Capital:   ${cfg['initial_cash']:,.2f}")
        logger.info("=" * 60)

        # --- Core components ---
        self.broker = PaperBroker(
            initial_cash=Decimal(str(cfg["initial_cash"])),
            commission_rate=Decimal("0.001"),
        )

        self.portfolio = Portfolio(
            initial_cash=Decimal(str(cfg["initial_cash"])),
        )

        risk_config = RiskConfig(
            max_position_value=Decimal(str(cfg["initial_cash"] * 0.15)),
            max_total_exposure=Decimal(str(cfg["initial_cash"] * 0.80)),
            max_concurrent_positions=5,
            daily_loss_limit=Decimal(str(cfg["initial_cash"] * 0.05)),
        )
        self.risk_manager = RiskManager(config=risk_config)

        self.position_sizer = PositionSizer(
            risk_per_trade=0.01,
            atr_multiplier=2.0,
        )

        self.execution_engine = ExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk_manager,
            position_sizer=self.position_sizer,
        )

        # --- Feature pipelines (one per symbol/timeframe) ---
        self.pipelines: dict[str, FeaturePipeline] = {}
        for symbol in symbols:
            key = f"{symbol}_{timeframe}"
            self.pipelines[key] = FeaturePipeline(symbol=symbol, timeframe=timeframe)

        # --- Strategies ---
        self.strategy_manager = StrategyManager()
        for symbol in symbols:
            rsi = RSIReversalStrategy(
                strategy_id=f"rsi_reversal_{symbol.lower()}",
                symbol=symbol,
                timeframe=timeframe,
                oversold=30.0,
                overbought=70.0,
                use_vwap_filter=True,
            )
            ema = EMACrossoverStrategy(
                strategy_id=f"ema_cross_{symbol.lower()}",
                symbol=symbol,
                timeframe=timeframe,
                fast_period=9,
                slow_period=21,
            )
            self.strategy_manager.add(rsi)
            self.strategy_manager.add(ema)

        # --- Connect bar events to feature pipeline + strategy dispatch ---
        from core.events import EventType, Event

        @bus.subscribe(EventType.BAR)
        async def on_bar(event: Event) -> None:
            bar = event.data
            key = f"{bar.symbol}_{bar.timeframe}"
            pipeline = self.pipelines.get(key)
            if not pipeline:
                return

            features = pipeline.process(bar)
            if not features:
                return

            # Dispatch to all active strategies
            for strategy in self.strategy_manager.strategies:
                if not strategy.is_active:
                    continue
                signal = strategy.on_bar(features)
                if signal:
                    logger.info(
                        f"Signal [{signal.strategy_id}] "
                        f"{signal.side.upper()} {signal.symbol} @ {signal.price:.4f} "
                        f"| strength={signal.strength} | {signal.reason}"
                    )
                    await bus.publish(Event(
                        type=EventType.SIGNAL,
                        data=signal,
                        source=signal.strategy_id,
                    ))

        # Notify strategies of fills
        @bus.subscribe(EventType.ORDER_FILLED)
        async def on_fill_dispatch(event: Event) -> None:
            fill = event.data
            for strategy in self.strategy_manager.strategies:
                strategy.on_fill(fill)

        self._bus = bus
        logger.info("Engine setup complete")

    async def warm_up(self) -> None:
        """
        Fetch historical bars and replay them through the feature pipelines.
        Ensures all indicators are ready before live trading begins.
        """
        from data.feeds.binance import BinanceFeed

        cfg = self.config
        symbols = cfg["symbols"]
        timeframe = cfg["timeframe"]
        warmup_bars = cfg.get("warmup_bars", 250)

        logger.info(f"Warming up indicators with {warmup_bars} historical bars...")

        feed = BinanceFeed(
            symbols=symbols,
            timeframes=[timeframe],
        )

        for symbol in symbols:
            try:
                bars = await feed.get_historical(symbol, timeframe, limit=warmup_bars)
                key = f"{symbol}_{timeframe}"
                pipeline = self.pipelines.get(key)
                if pipeline:
                    for bar in bars:
                        pipeline.process(bar)
                    logger.info(f"  {symbol}: warmed up with {len(bars)} bars, "
                                f"ready={pipeline.is_ready()}")
            except Exception as e:
                logger.warning(f"  {symbol}: warm-up failed ({e}), continuing...")

    async def run_paper_trading(self) -> None:
        """Run live paper trading with real Binance market data."""
        from data.feeds.binance import BinanceFeed

        cfg = self.config
        self.feed = BinanceFeed(
            symbols=cfg["symbols"],
            timeframes=[cfg["timeframe"]],
        )

        # Start the event bus dispatch loop
        bus_task = asyncio.create_task(self._bus.run(), name="event_bus")
        self._tasks.append(bus_task)

        # Start the data feed
        feed_task = asyncio.create_task(self.feed.start(), name="data_feed")
        self._tasks.append(feed_task)

        logger.info("Paper trading started. Waiting for market data...")
        self._running = True

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Engine tasks cancelled")

    async def run_api_server(self) -> None:
        """Run the FastAPI server alongside the trading loop."""
        from api.main import create_app

        app = create_app(
            portfolio=self.portfolio,
            broker=self.broker,
            strategies=self.strategy_manager,
            feed=getattr(self, "feed", None),
        )

        server_config = uvicorn.Config(
            app=app,
            host=self.config.get("api_host", "0.0.0.0"),
            port=self.config.get("api_port", 8000),
            log_level="warning",  # don't spam uvicorn logs
        )
        server = uvicorn.Server(server_config)
        await server.serve()

    async def run(self) -> None:
        """Main entry point. Sets up and runs all components concurrently."""
        await self.setup()

        if self.config["mode"] == "paper":
            await self.warm_up()
            await asyncio.gather(
                self.run_paper_trading(),
                self.run_api_server(),
            )
        elif self.config["mode"] == "api":
            await self.run_api_server()

    async def shutdown(self) -> None:
        logger.info("Shutting down engine...")
        self._running = False
        for task in self._tasks:
            task.cancel()
        if hasattr(self, "feed"):
            await self.feed.stop()
        await self._bus.stop()

        # Print final stats
        if hasattr(self, "portfolio"):
            snap = self.portfolio.get_snapshot()
            logger.info("=" * 60)
            logger.info("Final Portfolio Summary")
            logger.info(f"  Portfolio value:  ${snap.portfolio_value:.2f}")
            logger.info(f"  Realized PnL:     ${snap.total_realized_pnl:.2f}")
            logger.info(f"  Total trades:     {snap.n_closed_trades}")
            logger.info(f"  Win rate:         {snap.win_rate:.1%}")
            if snap.sharpe_ratio:
                logger.info(f"  Sharpe ratio:     {snap.sharpe_ratio:.3f}")
            logger.info(f"  Max drawdown:     {snap.max_drawdown:.1%}")
            logger.info("=" * 60)


async def main(config: dict) -> None:
    engine = TradingEngine(config)

    # Graceful shutdown — Windows-compatible (add_signal_handler is Unix only)
    import platform
    if platform.system() != "Windows":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(engine.shutdown()),
            )

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        await engine.shutdown()
    except Exception as e:
        logger.error(f"Engine error: {e}", exc_info=True)
        await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Engine")
    parser.add_argument(
        "--mode",
        choices=["paper", "backtest", "api"],
        default="paper",
        help="Run mode",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Trading symbols",
    )
    parser.add_argument(
        "--timeframe",
        default="5m",
        help="Candle timeframe (1m, 5m, 15m, 1h)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Starting capital in USD",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port",
    )

    args = parser.parse_args()

    config = {
        "mode": args.mode,
        "symbols": [s.upper() for s in args.symbols],
        "timeframe": args.timeframe,
        "initial_cash": args.capital,
        "warmup_bars": 250,
        "api_host": "0.0.0.0",
        "api_port": args.api_port,
    }

    asyncio.run(main(config))
