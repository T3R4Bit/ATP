"""
api/main.py
FastAPI application. All routes registered inside create_app() so they
attach to the correct app instance when dependencies are injected.
"""

from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PortfolioResponse(BaseModel):
    timestamp: str
    cash: float
    portfolio_value: float
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_commission: float
    n_open_positions: int
    n_closed_trades: int
    win_rate: float
    sharpe_ratio: float | None
    max_drawdown: float
    daily_pnl: float


class TradeResponse(BaseModel):
    symbol: str
    strategy_id: str
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    return_pct: float
    entry_time: str
    exit_time: str
    duration_seconds: float


class StrategyInfo(BaseModel):
    strategy_id: str
    name: str
    symbol: str
    timeframe: str
    is_active: bool
    pnl: float


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        if not self.active_connections:
            return
        dead = []
        payload = json.dumps(message, default=str)
        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        await websocket.send_text(json.dumps(message, default=str))


ws_manager = ConnectionManager()


def create_app(portfolio=None, broker=None, strategies=None, feed=None) -> FastAPI:
    """
    App factory. Routes are defined INSIDE this function so they register
    on the returned app instance, not a stale module-level one.
    """

    _state = {
        "portfolio": portfolio,
        "broker": broker,
        "strategies": strategies,
        "feed": feed,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from core.events import EventType, bus

        @bus.subscribe(EventType.BAR)
        async def on_bar_ws(event):
            bar = event.data
            await ws_manager.broadcast({
                "type": "price_update",
                "symbol": bar.symbol,
                "timeframe": bar.timeframe,
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "volume": float(bar.volume),
                "timestamp": bar.timestamp.isoformat(),
            })

        @bus.subscribe(EventType.ORDER_FILLED)
        async def on_fill_ws(event):
            fill = event.data
            await ws_manager.broadcast({
                "type": "fill",
                "symbol": fill.symbol,
                "side": str(fill.side),
                "quantity": float(fill.quantity),
                "price": float(fill.price),
                "commission": float(fill.commission),
                "strategy_id": fill.strategy_id,
                "timestamp": fill.timestamp.isoformat(),
            })

        @bus.subscribe(EventType.PNL_UPDATE)
        async def on_pnl_ws(event):
            p = _state["portfolio"]
            if p:
                snap = p.get_snapshot()
                await ws_manager.broadcast({
                    "type": "portfolio_update",
                    "portfolio_value": float(snap.portfolio_value),
                    "realized_pnl": float(snap.total_realized_pnl),
                    "unrealized_pnl": float(snap.total_unrealized_pnl),
                    "daily_pnl": float(snap.daily_pnl),
                    "win_rate": snap.win_rate,
                    "timestamp": snap.timestamp.isoformat(),
                })

        yield

    app = FastAPI(
        title="Trading Engine API",
        description="Real-time algorithmic trading system",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "ws_clients": len(ws_manager.active_connections),
        }

    @app.get("/portfolio", response_model=PortfolioResponse)
    async def get_portfolio():
        p = _state["portfolio"]
        if not p:
            raise HTTPException(status_code=503, detail="Portfolio not initialized")
        snap = p.get_snapshot()
        return PortfolioResponse(
            timestamp=snap.timestamp.isoformat(),
            cash=float(snap.cash),
            portfolio_value=float(snap.portfolio_value),
            total_realized_pnl=float(snap.total_realized_pnl),
            total_unrealized_pnl=float(snap.total_unrealized_pnl),
            total_commission=float(snap.total_commission),
            n_open_positions=snap.n_open_positions,
            n_closed_trades=snap.n_closed_trades,
            win_rate=snap.win_rate,
            sharpe_ratio=snap.sharpe_ratio,
            max_drawdown=snap.max_drawdown,
            daily_pnl=float(snap.daily_pnl),
        )

    @app.get("/portfolio/trades", response_model=List[TradeResponse])
    async def get_trades(limit: int = 50, strategy_id: str | None = None):
        p = _state["portfolio"]
        if not p:
            raise HTTPException(status_code=503, detail="Portfolio not initialized")
        trades = p.closed_trades
        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        return [
            TradeResponse(
                symbol=t.symbol,
                strategy_id=t.strategy_id,
                entry_price=float(t.entry_price),
                exit_price=float(t.exit_price),
                quantity=float(t.quantity),
                realized_pnl=float(t.realized_pnl),
                return_pct=t.return_pct * 100,
                entry_time=t.entry_time.isoformat(),
                exit_time=t.exit_time.isoformat(),
                duration_seconds=t.duration_seconds,
            )
            for t in sorted(trades, key=lambda x: x.exit_time, reverse=True)[:limit]
        ]

    @app.get("/portfolio/strategy-pnl")
    async def get_strategy_pnl():
        p = _state["portfolio"]
        if not p:
            raise HTTPException(status_code=503, detail="Portfolio not initialized")
        return p.get_strategy_pnl()

    @app.get("/strategies", response_model=List[StrategyInfo])
    async def list_strategies():
        sm = _state["strategies"]
        if not sm:
            return []
        p = _state["portfolio"]
        strategy_pnl = p.get_strategy_pnl() if p else {}
        return [
            StrategyInfo(
                strategy_id=s.strategy_id,
                name=s.__class__.__name__,
                symbol=s.symbol,
                timeframe=s.timeframe,
                is_active=s.is_active,
                pnl=strategy_pnl.get(s.strategy_id, 0.0),
            )
            for s in sm.strategies
        ]

    @app.post("/strategies/{strategy_id}/toggle")
    async def toggle_strategy(strategy_id: str):
        sm = _state["strategies"]
        if not sm:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")
        strategy = sm.get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        if strategy.is_active:
            strategy.deactivate()
            return {"strategy_id": strategy_id, "status": "deactivated"}
        else:
            strategy.activate()
            return {"strategy_id": strategy_id, "status": "activated"}

    @app.get("/market/prices")
    async def get_prices():
        b = _state["broker"]
        if not b:
            return {}
        return {symbol: float(pos.average_cost) for symbol, pos in b.positions.items()}

    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket):
        await ws_manager.connect(websocket)
        p = _state["portfolio"]
        if p:
            snap = p.get_snapshot()
            await ws_manager.send_personal(websocket, {
                "type": "initial_state",
                "portfolio_value": float(snap.portfolio_value),
                "realized_pnl": float(snap.total_realized_pnl),
                "cash": float(snap.cash),
                "n_trades": snap.n_closed_trades,
            })
        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
                    msg = json.loads(data)
                    if msg.get("type") == "pong":
                        pass
                except asyncio.TimeoutError:
                    await ws_manager.send_personal(websocket, {"type": "ping"})
                except WebSocketDisconnect:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            ws_manager.disconnect(websocket)

    return app
