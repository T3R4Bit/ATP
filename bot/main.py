from __future__ import annotations

import argparse

from bot.config import TradingConfig
from bot.dashboard import start_dashboard
from bot.trader import RuntimeState, run_trading_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI crypto trade bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train model")
    train.add_argument("--exchange", default="binance")
    train.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    train.add_argument("--timeframe", default="5m")
    train.add_argument("--years", type=int, default=2)

    trade = sub.add_parser("trade", help="Run trading loop")
    trade.add_argument("--mode", choices=["paper", "live"], default="paper")
    trade.add_argument("--exchange", default="binance")
    trade.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    trade.add_argument("--timeframe", default="5m")
    trade.add_argument("--starting-balance", type=float, default=None, help="Override paper starting balance in USDT")
    trade.add_argument("--dashboard", action="store_true", help="Start local dashboard")
    trade.add_argument("--dashboard-host", default="127.0.0.1")
    trade.add_argument("--dashboard-port", type=int, default=8000)

    return parser.parse_args()


def do_train(args: argparse.Namespace) -> None:
    from bot.data import fetch_and_cache_symbols
    from bot.features import build_training_frame
    from bot.model import save_model, train_model

    raw = fetch_and_cache_symbols(args.exchange, args.symbols, args.timeframe, args.years)
    frame = build_training_frame(raw)
    model, mse = train_model(frame)
    save_model(model)
    print(f"Model trained. Rows={len(frame)} MSE={mse:.10f}")


def do_trade(args: argparse.Namespace) -> None:
    from bot.model import load_model

    cfg = TradingConfig()
    if args.starting_balance is not None:
        cfg.starting_cash_usdt = args.starting_balance

    model = load_model()

    state = None
    if args.dashboard:
        state = RuntimeState(
            mode=args.mode,
            starting_balance=cfg.starting_cash_usdt,
            portfolio_value=cfg.starting_cash_usdt,
            pnl_pct=0.0,
            profit_loss_ratio=None,
        )
        start_dashboard(state, host=args.dashboard_host, port=args.dashboard_port)
        print(f"Dashboard running at http://{args.dashboard_host}:{args.dashboard_port}")

    run_trading_loop(model, args.mode, args.exchange, args.symbols, args.timeframe, cfg, dashboard_state=state)


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        do_train(args)
    elif args.cmd == "trade":
        do_trade(args)


if __name__ == "__main__":
    main()
