from dataclasses import dataclass
import os


@dataclass
class TradingConfig:
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")
    api_password: str = os.getenv("EXCHANGE_PASSWORD", "")
    starting_cash_usdt: float = float(os.getenv("STARTING_CASH_USDT", "10000"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    buy_threshold: float = float(os.getenv("BUY_THRESHOLD", "0.0015"))
    sell_threshold: float = float(os.getenv("SELL_THRESHOLD", "-0.0015"))
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "30"))
