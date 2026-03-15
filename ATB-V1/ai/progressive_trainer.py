"""
ai/progressive_trainer.py
Continuously trains the model on random chunks of historical data.

How it works:
1. Downloads maximum historical data (up to 3 years of 1m bars)
2. Every N minutes, samples a random chunk of history
3. Re-trains or incrementally updates the model on that chunk
4. Evaluates on a held-out recent window (last 7 days always reserved)
5. Saves the model only if it improves on the validation set
6. Runs forever as a background process alongside the trading engine

This mimics how professional quant shops maintain their models —
constant retraining on different market regimes so the model
never gets stuck overfitting one period.

Usage:
    # Run standalone (background process):
    python -m ai.progressive_trainer

    # With options:
    python -m ai.progressive_trainer --symbols BTCUSDT ETHUSDT --interval 30

    # Collect 1 year of data first:
    python -m ai.data_collector --days 365
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import pickle
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ai/training.log"),
    ],
)
logger = logging.getLogger("ai.progressive")

MODEL_DIR = Path("ai/models")
DATA_DIR = Path("ai/data")


# ---------------------------------------------------------------------------
# Chunk sampler
# ---------------------------------------------------------------------------

class ChunkSampler:
    """
    Samples random windows of historical data for training.
    
    Sampling strategies:
    - uniform: equal chance of any period (good for diversity)
    - recent_biased: recent data weighted more heavily (good for regime adaptation)
    - regime_aware: samples from different volatility regimes (good for robustness)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        chunk_size: int = 5000,
        validation_holdout_days: int = 7,
        strategy: str = "recent_biased",
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size

        # Always hold out the most recent N days for validation
        # This is NEVER used for training
        holdout_cutoff = df.index[-1] - timedelta(days=validation_holdout_days)
        self.train_df = df[df.index <= holdout_cutoff]
        self.val_df = df[df.index > holdout_cutoff]

        logger.info(
            f"ChunkSampler: {len(self.train_df):,} training bars, "
            f"{len(self.val_df):,} validation bars (last {validation_holdout_days} days)"
        )

    def sample(self) -> pd.DataFrame:
        """Return a random chunk of training data."""
        n = len(self.train_df)
        if n <= self.chunk_size:
            return self.train_df

        if self.strategy == "uniform":
            start = random.randint(0, n - self.chunk_size)
            return self.train_df.iloc[start : start + self.chunk_size]

        elif self.strategy == "recent_biased":
            # Weight recent data 3x more likely than old data
            # Creates a probability ramp: old=low, recent=high
            max_start = n - self.chunk_size
            weights = np.linspace(1, 3, max_start + 1)
            weights = weights / weights.sum()
            start = np.random.choice(max_start + 1, p=weights)
            return self.train_df.iloc[start : start + self.chunk_size]

        elif self.strategy == "regime_aware":
            # Detect volatility regimes and sample across them
            returns = self.train_df["close"].pct_change().abs()
            rolling_vol = returns.rolling(60).mean()

            # Label each bar: high_vol (top 33%) or low_vol (bottom 33%) or mid
            vol_33 = rolling_vol.quantile(0.33)
            vol_66 = rolling_vol.quantile(0.66)

            # Pick a random regime to focus on this chunk
            regime = random.choice(["low_vol", "mid_vol", "high_vol"])
            if regime == "low_vol":
                mask = rolling_vol <= vol_33
            elif regime == "high_vol":
                mask = rolling_vol >= vol_66
            else:
                mask = (rolling_vol > vol_33) & (rolling_vol < vol_66)

            regime_df = self.train_df[mask]
            if len(regime_df) < self.chunk_size:
                # Fall back to uniform if not enough regime data
                return self.sample_uniform()

            start = random.randint(0, max(0, len(regime_df) - self.chunk_size))
            return regime_df.iloc[start : start + self.chunk_size]

        return self.train_df.iloc[-self.chunk_size:]

    def sample_uniform(self) -> pd.DataFrame:
        n = len(self.train_df)
        start = random.randint(0, max(0, n - self.chunk_size))
        return self.train_df.iloc[start : start + self.chunk_size]

    @property
    def validation_set(self) -> pd.DataFrame:
        return self.val_df


# ---------------------------------------------------------------------------
# Model state tracker
# ---------------------------------------------------------------------------

class ModelState:
    """Tracks training history and decides whether to accept a new model."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.best_val_accuracy: float = 0.0
        self.best_val_auc: float = 0.0
        self.iteration: int = 0
        self.history: list[dict] = []
        self.last_improvement: datetime = datetime.utcnow()

    def should_accept(self, val_accuracy: float, val_auc: float) -> bool:
        """Accept new model if it improves on EITHER metric."""
        return val_accuracy > self.best_val_accuracy or val_auc > self.best_val_auc

    def update(self, val_accuracy: float, val_auc: float, accepted: bool) -> None:
        if accepted:
            self.best_val_accuracy = max(self.best_val_accuracy, val_accuracy)
            self.best_val_auc = max(self.best_val_auc, val_auc)
            self.last_improvement = datetime.utcnow()

        self.iteration += 1
        self.history.append({
            "iteration": self.iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "val_accuracy": val_accuracy,
            "val_auc": val_auc,
            "accepted": accepted,
            "best_accuracy": self.best_val_accuracy,
            "best_auc": self.best_val_auc,
        })

    def save_history(self) -> None:
        path = MODEL_DIR / f"{self.symbol}_training_history.csv"
        pd.DataFrame(self.history).to_csv(path, index=False)

    def hours_since_improvement(self) -> float:
        return (datetime.utcnow() - self.last_improvement).total_seconds() / 3600


# ---------------------------------------------------------------------------
# Single training iteration
# ---------------------------------------------------------------------------

def train_on_chunk(
    chunk: pd.DataFrame,
    val_df: pd.DataFrame,
    symbol: str,
    feature_names: Optional[list[str]] = None,
) -> tuple[object, list[str], float, float]:
    """
    Train one model on a chunk and evaluate on validation set.
    Returns: (model, feature_names, val_accuracy, val_auc)
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from ai.predictor import build_features, build_labels

    # Build features for this chunk
    chunk_features = build_features(chunk)
    chunk_labels = build_labels(chunk, forward_bars=1)

    combined = pd.concat([chunk_features, chunk_labels.rename("target")], axis=1)
    combined = combined.iloc[:-1]
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    X = combined.drop(columns=["target"])
    y = combined["target"]
    X = X.clip(lower=X.quantile(0.001), upper=X.quantile(0.999), axis=1)

    # If feature_names provided (from previous iteration), use same features
    if feature_names:
        available = [f for f in feature_names if f in X.columns]
        X = X[available]
        feature_names = available
    else:
        feature_names = list(X.columns)

    if len(X) < 200 or len(y.unique()) < 2:
        raise ValueError(f"Insufficient data: {len(X)} samples")

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=random.randint(0, 10000),  # different seed each time
        verbosity=0,
    )
    model.fit(X, y, verbose=False)

    # Evaluate on validation set
    val_features = build_features(val_df)
    val_labels = build_labels(val_df, forward_bars=1)
    val_combined = pd.concat([val_features, val_labels.rename("target")], axis=1)
    val_combined = val_combined.iloc[:-1].replace([np.inf, -np.inf], np.nan).dropna()

    X_val = val_combined.drop(columns=["target"])
    y_val = val_combined["target"]

    # Use only features that exist in both
    available_val = [f for f in feature_names if f in X_val.columns]
    X_val = X_val[available_val]

    if len(X_val) < 50:
        raise ValueError("Not enough validation data")

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_acc = float(accuracy_score(y_val, val_preds))
    try:
        val_auc = float(roc_auc_score(y_val, val_probs))
    except Exception:
        val_auc = 0.5

    return model, feature_names, val_acc, val_auc


def save_model(model, feature_names: list[str], symbol: str, state: ModelState) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{symbol}_1m_predictor.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "train_accuracy": state.best_val_accuracy,
            "test_accuracy": state.best_val_accuracy,
            "test_auc": state.best_val_auc,
            "feature_importance": dict(zip(
                feature_names,
                model.feature_importances_.tolist(),
            )),
            "trained_at": datetime.utcnow(),
            "symbol": symbol,
            "iteration": state.iteration,
        }, f)


# ---------------------------------------------------------------------------
# Progressive trainer
# ---------------------------------------------------------------------------

class ProgressiveTrainer:
    """
    Continuously trains models on random historical chunks.
    
    Runs as an async background task so the trading engine
    keeps running while training happens.
    """

    def __init__(
        self,
        symbol: str,
        train_interval_minutes: int = 30,
        chunk_size: int = 5000,
        sampling_strategy: str = "recent_biased",
        max_no_improvement_hours: float = 24.0,
    ) -> None:
        self.symbol = symbol
        self.train_interval_minutes = train_interval_minutes
        self.chunk_size = chunk_size
        self.sampling_strategy = sampling_strategy
        self.max_no_improvement_hours = max_no_improvement_hours
        self._running = False
        self._sampler: Optional[ChunkSampler] = None
        self._state = ModelState(symbol)
        self._current_feature_names: Optional[list[str]] = None

    async def load_data(self) -> bool:
        """Load saved CSV data. Returns False if not enough data available."""
        from ai.data_collector import load_dataset
        try:
            df = load_dataset(self.symbol, "1m")
            if len(df) < 10000:
                logger.warning(
                    f"{self.symbol}: Only {len(df):,} bars available. "
                    f"Run: python -m ai.data_collector --days 365"
                )

            self._sampler = ChunkSampler(
                df=df,
                chunk_size=self.chunk_size,
                validation_holdout_days=7,
                strategy=self.sampling_strategy,
            )
            logger.info(f"{self.symbol}: Loaded {len(df):,} bars for progressive training")
            return True
        except FileNotFoundError:
            logger.error(
                f"{self.symbol}: No data file found. "
                f"Run: python -m ai.data_collector --days 365"
            )
            return False

    def _run_training_iteration(self) -> Optional[dict]:
        """Run one training iteration synchronously (called in executor)."""
        if not self._sampler:
            return None

        chunk = self._sampler.sample()
        val_df = self._sampler.validation_set

        chunk_start = chunk.index[0].strftime("%Y-%m-%d %H:%M")
        chunk_end = chunk.index[-1].strftime("%Y-%m-%d %H:%M")

        try:
            model, feature_names, val_acc, val_auc = train_on_chunk(
                chunk=chunk,
                val_df=val_df,
                symbol=self.symbol,
                feature_names=self._current_feature_names,
            )
        except ValueError as e:
            logger.warning(f"  Skipping iteration: {e}")
            return None

        accepted = self._state.should_accept(val_acc, val_auc)
        self._state.update(val_acc, val_auc, accepted)

        if accepted:
            self._current_feature_names = feature_names
            save_model(model, feature_names, self.symbol, self._state)

        result = {
            "iteration": self._state.iteration,
            "symbol": self.symbol,
            "chunk": f"{chunk_start} → {chunk_end}",
            "chunk_bars": len(chunk),
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "accepted": accepted,
            "best_accuracy": self._state.best_val_accuracy,
            "best_auc": self._state.best_val_auc,
        }

        status = "✓ SAVED" if accepted else "  skip"
        logger.info(
            f"[{self.symbol}] iter={self._state.iteration:04d} {status} | "
            f"chunk={chunk_start} ({len(chunk):,} bars) | "
            f"val_acc={val_acc:.1%} auc={val_auc:.3f} | "
            f"best={self._state.best_val_accuracy:.1%}"
        )

        self._state.save_history()
        return result

    async def run(self) -> None:
        """Main training loop. Runs forever until stopped."""
        self._running = True

        ok = await self.load_data()
        if not ok:
            logger.error(f"{self.symbol}: Cannot start — no data. Exiting trainer.")
            return

        logger.info(
            f"{self.symbol}: Starting progressive training "
            f"(interval={self.train_interval_minutes}m, "
            f"chunk={self.chunk_size} bars, "
            f"strategy={self.sampling_strategy})"
        )

        iteration_count = 0
        while self._running:
            try:
                # Run training in thread pool so it doesn't block the event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._run_training_iteration
                )

                iteration_count += 1

                # Reload data every 24 hours to pick up new bars
                if iteration_count % (24 * 60 // self.train_interval_minutes) == 0:
                    logger.info(f"{self.symbol}: Reloading data to pick up new bars...")
                    await self.load_data()

                # If no improvement in a long time, try switching sampling strategy
                hours_no_improvement = self._state.hours_since_improvement()
                if hours_no_improvement > self.max_no_improvement_hours:
                    strategies = ["uniform", "recent_biased", "regime_aware"]
                    current_idx = strategies.index(self.sampling_strategy)
                    self.sampling_strategy = strategies[(current_idx + 1) % len(strategies)]
                    if self._sampler:
                        self._sampler.strategy = self.sampling_strategy
                    logger.info(
                        f"{self.symbol}: No improvement for {hours_no_improvement:.0f}h, "
                        f"switching to {self.sampling_strategy} sampling"
                    )
                    self._state.last_improvement = datetime.utcnow()  # reset timer

            except Exception as e:
                logger.error(f"{self.symbol}: Training error: {e}", exc_info=True)

            # Wait before next iteration
            await asyncio.sleep(self.train_interval_minutes * 60)

    async def stop(self) -> None:
        self._running = False
        logger.info(f"{self.symbol}: Progressive trainer stopped (iter={self._state.iteration})")


# ---------------------------------------------------------------------------
# Multi-symbol manager
# ---------------------------------------------------------------------------

class ProgressiveTrainingManager:
    """Manages progressive trainers for multiple symbols."""

    def __init__(
        self,
        symbols: list[str],
        train_interval_minutes: int = 30,
        chunk_size: int = 5000,
        sampling_strategy: str = "recent_biased",
    ) -> None:
        self.trainers = {
            symbol: ProgressiveTrainer(
                symbol=symbol,
                train_interval_minutes=train_interval_minutes,
                chunk_size=chunk_size,
                sampling_strategy=sampling_strategy,
            )
            for symbol in symbols
        }
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start all trainers as background tasks."""
        for symbol, trainer in self.trainers.items():
            task = asyncio.create_task(trainer.run(), name=f"trainer_{symbol}")
            self._tasks.append(task)
        logger.info(f"Progressive training started for: {list(self.trainers.keys())}")

    async def stop(self) -> None:
        for trainer in self.trainers.values():
            await trainer.stop()
        for task in self._tasks:
            task.cancel()

    def get_status(self) -> dict:
        return {
            symbol: {
                "iteration": trainer._state.iteration,
                "best_accuracy": f"{trainer._state.best_val_accuracy:.1%}",
                "best_auc": f"{trainer._state.best_val_auc:.3f}",
                "hours_since_improvement": f"{trainer._state.hours_since_improvement():.1f}h",
                "sampling_strategy": trainer.sampling_strategy,
            }
            for symbol, trainer in self.trainers.items()
        }


# ---------------------------------------------------------------------------
# Entry point (run standalone)
# ---------------------------------------------------------------------------

async def main(args) -> None:
    symbols = [s.upper() for s in args.symbols]

    manager = ProgressiveTrainingManager(
        symbols=symbols,
        train_interval_minutes=args.interval,
        chunk_size=args.chunk_size,
        sampling_strategy=args.strategy,
    )

    try:
        await manager.start()
        # Print status every hour
        while True:
            await asyncio.sleep(3600)
            status = manager.get_status()
            logger.info("=" * 55)
            logger.info("PROGRESSIVE TRAINING STATUS")
            for symbol, s in status.items():
                logger.info(
                    f"  {symbol}: iter={s['iteration']} "
                    f"best_acc={s['best_accuracy']} "
                    f"auc={s['best_auc']} "
                    f"no_improvement={s['hours_since_improvement']}"
                )
            logger.info("=" * 55)
    except KeyboardInterrupt:
        logger.info("Stopping progressive trainer...")
        await manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive model trainer")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Minutes between training iterations (default: 30)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000,
        help="Number of bars per training chunk (default: 5000)"
    )
    parser.add_argument(
        "--strategy",
        choices=["uniform", "recent_biased", "regime_aware"],
        default="recent_biased",
        help="Chunk sampling strategy (default: recent_biased)"
    )
    args = parser.parse_args()

    Path("ai/models").mkdir(parents=True, exist_ok=True)
    Path("ai").mkdir(exist_ok=True)

    asyncio.run(main(args))