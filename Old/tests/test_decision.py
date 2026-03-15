from bot.config import TradingConfig
from bot.trader import PaperAccount, decide_action, profit_loss_ratio


def test_decide_action_buy():
    cfg = TradingConfig(buy_threshold=0.001, sell_threshold=-0.001)
    assert decide_action(0.002, cfg) == "BUY"


def test_decide_action_sell():
    cfg = TradingConfig(buy_threshold=0.001, sell_threshold=-0.001)
    assert decide_action(-0.002, cfg) == "SELL"


def test_decide_action_hold():
    cfg = TradingConfig(buy_threshold=0.001, sell_threshold=-0.001)
    assert decide_action(0.0, cfg) == "HOLD"


def test_profit_loss_ratio_tracking():
    account = PaperAccount(cash=1000)
    account.buy("BTC/USDT", price=100, amount=1)
    account.sell("BTC/USDT", price=120, amount=0.5)  # +10
    account.sell("BTC/USDT", price=80, amount=0.5)   # -10
    assert account.realized_profit == 10
    assert account.realized_loss == 10
    assert profit_loss_ratio(account) == 1.0


def test_profit_loss_ratio_none_without_losses():
    account = PaperAccount(cash=1000)
    assert profit_loss_ratio(account) is None
