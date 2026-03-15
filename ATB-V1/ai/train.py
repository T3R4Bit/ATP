"""
ai/train.py
Standalone training script. Run this BEFORE starting the engine
to pre-train models so startup is instant.

Usage:
    python -m ai.train                          # train BTC + ETH
    python -m ai.train --symbols BTCUSDT        # train just BTC
    python -m ai.train --bars 10000             # use more historical data

The trained models are saved to ai/models/ and loaded automatically
by the engine on startup.
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def train_all(symbols: list[str], n_bars: int) -> None:
    from ai.predictor import MarketPredictor

    results = []
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*50}")
        predictor = MarketPredictor(symbol=symbol)
        result = await predictor.train(n_bars=n_bars)
        results.append(result)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n  {r['symbol']}")
        print(f"    Training bars:  {r['n_training_bars']}")
        print(f"    Test bars:      {r['n_test_bars']}")
        print(f"    Train accuracy: {r['train_accuracy']}")
        print(f"    Test accuracy:  {r['test_accuracy']}")
        print(f"    Top features:")
        for feat, imp in r["top_features"]:
            print(f"      {feat:<25} {imp:.4f}")
    print("=" * 60)
    print("\nModels saved to ai/models/")
    print("Start the engine: python main.py --mode paper")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI price predictors")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--bars", type=int, default=5000,
                        help="Number of 1m historical bars to train on")
    args = parser.parse_args()

    asyncio.run(train_all(
        symbols=[s.upper() for s in args.symbols],
        n_bars=args.bars,
    ))