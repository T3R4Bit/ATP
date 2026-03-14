import sys

from bot.main import parse_args


def test_trade_starting_balance_arg(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "trade",
            "--mode",
            "paper",
            "--starting-balance",
            "25000",
            "--dashboard",
        ],
    )
    args = parse_args()
    assert args.starting_balance == 25000
    assert args.dashboard is True
