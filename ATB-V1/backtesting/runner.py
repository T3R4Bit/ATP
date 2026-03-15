"""
backtesting/runner.py
Backtesting system.

Runs strategies against historical data with the SAME code as live trading.
This is only possible because:
1. Strategies only see feature dicts — not raw data or the broker
2. HistoricalFeed emits the same BAR events as BinanceFeed
3. The event bus is synchronous during backtesting (no real-time delays)

Usage:
    runner = BacktestRunner(
        bars=historical_bars,
        strategies=[RSIReversalStrategy(...)],
        initial_cash=100_000,
    )
    results = await runner.run()
    report = results.to_report()
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List

from core.events import EventType, Event, bus
from core.types import Bar
from data.feeds.binance import HistoricalFeed
from execution.engine import (
    ExecutionEngine, PaperBroker, PositionSizer, RiskConfig, RiskManager
)
from features.pipeline import FeaturePipeline
from portfolio.portfolio import Portfolio
from strategies.strategy import AbstractStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    strategy_id: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_value: float
    total_return_pct: float
    total_realized_pnl: float
    total_commission: float
    n_trades: int
    win_rate: float
    sharpe_ratio: float | None
    max_drawdown: float
    trades: list

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "period": f"{self.start_date.date()} → {self.end_date.date()}",
            "initial_cash": f"${self.initial_cash:,.2f}",
            "final_value": f"${self.final_value:,.2f}",
            "total_return": f"{self.total_return_pct:+.2f}%",
            "realized_pnl": f"${self.total_realized_pnl:+,.2f}",
            "commission": f"${self.total_commission:.2f}",
            "n_trades": self.n_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "sharpe": f"{self.sharpe_ratio:.3f}" if self.sharpe_ratio else "N/A",
            "max_drawdown": f"{self.max_drawdown:.1%}",
        }

    def print_report(self) -> None:
        d = self.to_dict()
        print("\n" + "=" * 50)
        print(f"BACKTEST REPORT: {d['strategy_id']}")
        print("=" * 50)
        for k, v in d.items():
            if k not in ("strategy_id", "trades"):
                print(f"  {k:<22} {v}")
        print("=" * 50)


class BacktestRunner:
    """
    Runs a backtest for one strategy against one dataset.
    For multi-strategy comparison, run multiple BacktestRunner instances
    with the same bars and compare their BacktestResults.
    """

    def __init__(
        self,
        bars: List[Bar],
        strategy: AbstractStrategy,
        initial_cash: float = 100_000,
    ) -> None:
        self.bars = sorted(bars, key=lambda b: b.timestamp)
        self.strategy = strategy
        self.initial_cash = Decimal(str(initial_cash))

        # Fresh components for each backtest — no state leaks between runs
        self.broker = PaperBroker(
            initial_cash=self.initial_cash,
            commission_rate=Decimal("0.001"),
            slippage_bps=2,
        )
        self.portfolio = Portfolio(initial_cash=self.initial_cash)
        self.risk_manager = RiskManager(config=RiskConfig())
        self.position_sizer = PositionSizer()
        self.execution_engine = ExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk_manager,
            position_sizer=self.position_sizer,
        )
        self.pipeline = FeaturePipeline(
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
        )

    async def run(self) -> BacktestResults:
        """
        Replay all bars through the pipeline + strategy + execution engine.
        Returns a BacktestResults object with all metrics.
        """
        # Reset event bus subscriptions to avoid contamination from other runs
        # In production: use a fresh EventBus per backtest (not the singleton)
        # For simplicity here, we process events synchronously

        if not self.bars:
            raise ValueError("No bars provided for backtest")

        start_date = self.bars[0].timestamp
        end_date = self.bars[-1].timestamp

        logger.info(
            f"Backtest: {self.strategy.strategy_id} | "
            f"{start_date.date()} → {end_date.date()} | "
            f"{len(self.bars)} bars"
        )

        for bar in self.bars:
            # Update portfolio with current prices
            await bus.publish_sync(Event(type=EventType.BAR, data=bar, source="backtest"))

            # Compute features
            features = self.pipeline.process(bar)
            if not features:
                continue

            # Ask the strategy for a signal
            signal = self.strategy.on_bar(features)
            if signal:
                await bus.publish_sync(Event(
                    type=EventType.SIGNAL,
                    data=signal,
                    source=self.strategy.strategy_id,
                ))

        # Gather final metrics
        snap = self.portfolio.get_snapshot()
        final_value = float(snap.portfolio_value)
        total_return_pct = (final_value - float(self.initial_cash)) / float(self.initial_cash) * 100

        results = BacktestResults(
            strategy_id=self.strategy.strategy_id,
            symbol=self.strategy.symbol,
            timeframe=self.strategy.timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_cash=float(self.initial_cash),
            final_value=final_value,
            total_return_pct=total_return_pct,
            total_realized_pnl=float(snap.total_realized_pnl),
            total_commission=float(snap.total_commission),
            n_trades=snap.n_closed_trades,
            win_rate=snap.win_rate,
            sharpe_ratio=snap.sharpe_ratio,
            max_drawdown=snap.max_drawdown,
            trades=self.portfolio.closed_trades,
        )

        results.print_report()
        return results


async def compare_strategies(
    bars: List[Bar],
    strategies: List[AbstractStrategy],
    initial_cash: float = 100_000,
) -> List[BacktestResults]:
    """Run multiple strategies on the same dataset and return ranked results."""
    all_results = []
    for strategy in strategies:
        runner = BacktestRunner(bars=bars, strategy=strategy, initial_cash=initial_cash)
        result = await runner.run()
        all_results.append(result)

    # Rank by Sharpe ratio (or total return if Sharpe unavailable)
    all_results.sort(
        key=lambda r: r.sharpe_ratio if r.sharpe_ratio else r.total_return_pct,
        reverse=True,
    )

    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON RANKING")
    print("=" * 60)
    for i, r in enumerate(all_results, 1):
        sharpe_str = f"{r.sharpe_ratio:.3f}" if r.sharpe_ratio else "N/A"
        print(
            f"  #{i} {r.strategy_id:<35} "
            f"return={r.total_return_pct:+.1f}%  "
            f"sharpe={sharpe_str}  "
            f"dd={r.max_drawdown:.1%}"
        )
    print("=" * 60)

    return all_results
