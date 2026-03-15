"""
ai/predictor.py
AI price direction predictor for 1-minute BTC/ETH candles.

Architecture:
- Fetches historical 1m candles from Binance REST API
- Engineers ~25 features per bar (price action + indicators)
- Trains a gradient boosted classifier (XGBoost) to predict next-bar direction
- Wraps the model as an AbstractStrategy so it plugs straight into the engine
- Saves/loads trained models to disk so you don't retrain every startup

Prediction target:
  1 = next 1m candle closes HIGHER than current close (buy signal)
  0 = next 1m candle closes LOWER or equal (no trade / sell signal)

Why XGBoost over a neural net to start:
- Trains in seconds on CPU with ~5000 bars
- Naturally handles mixed feature scales (no normalization needed)
- Feature importance built in — you can see WHAT it's using
- Easy to swap for LSTM/Transformer in Phase 4 via the same interface

Install deps first:
  pip install xgboost scikit-learn pandas numpy aiohttp
"""

from __future__ import annotations
import asyncio
import logging
import os
import pickle
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw OHLCV data.
    
    All features are either:
    - Normalized (returns, ratios) — scale-invariant across price levels
    - Bounded indicators (RSI 0-100) — no normalization needed
    - Rolling statistics (z-scores) — removes trend bias
    
    This means the same model works whether BTC is at $30k or $100k.
    """
    f = pd.DataFrame(index=df.index)

    # --- Price action features ---
    f["return_1"] = df["close"].pct_change(1)
    f["return_3"] = df["close"].pct_change(3)
    f["return_5"] = df["close"].pct_change(5)
    f["return_10"] = df["close"].pct_change(10)
    f["return_30"] = df["close"].pct_change(30)

    # Candle body and wick ratios
    f["body_size"] = (df["close"] - df["open"]).abs() / df["open"]
    f["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
    f["is_bullish"] = (df["close"] > df["open"]).astype(int)

    # High/low range
    f["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # --- Volume features ---
    f["volume_return"] = df["volume"].pct_change(1)
    f["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    # Volume spike: current bar volume vs recent average
    f["volume_spike"] = (df["volume"] > df["volume"].rolling(10).mean() * 1.5).astype(int)

    # --- RSI (14 period) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    f["rsi_14"] = 100 - (100 / (1 + rs))
    f["rsi_7"] = _rsi(df["close"], 7)

    # RSI momentum — is RSI rising or falling?
    f["rsi_momentum"] = f["rsi_14"].diff(3)

    # --- EMA features ---
    f["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    f["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    f["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Price vs EMAs (normalized)
    f["price_vs_ema9"] = (df["close"] - f["ema_9"]) / f["ema_9"]
    f["price_vs_ema21"] = (df["close"] - f["ema_21"]) / f["ema_21"]
    f["ema9_vs_ema21"] = (f["ema_9"] - f["ema_21"]) / f["ema_21"]

    # --- VWAP ---
    typical = (df["high"] + df["low"] + df["close"]) / 3
    f["vwap"] = (typical * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    f["price_vs_vwap"] = (df["close"] - f["vwap"]) / f["vwap"]

    # --- ATR (volatility) ---
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(com=13, adjust=False).mean()
    f["atr_normalized"] = f["atr_14"] / df["close"]  # % of price

    # --- Bollinger Bands ---
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    f["bb_upper"] = (bb_mid + 2 * bb_std - df["close"]) / df["close"]
    f["bb_lower"] = (df["close"] - (bb_mid - 2 * bb_std)) / df["close"]
    f["bb_position"] = (df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # --- Momentum ---
    f["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    f["momentum_10"] = df["close"] / df["close"].shift(10) - 1

    # --- Rolling z-score of returns (mean reversion signal) ---
    ret = df["close"].pct_change()
    f["return_zscore"] = (ret - ret.rolling(20).mean()) / (ret.rolling(20).std() + 1e-10)

    # Drop EMA/VWAP raw values — keep only normalized ratios
    f = f.drop(columns=["ema_9", "ema_21", "ema_50", "vwap", "atr_14"])

    return f


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def build_labels(df: pd.DataFrame, forward_bars: int = 1) -> pd.Series:
    """
    Label: 1 if close N bars ahead > current close, else 0.
    Default: predict next 1m bar direction.
    """
    future_close = df["close"].shift(-forward_bars)
    return (future_close > df["close"]).astype(int)


# ---------------------------------------------------------------------------
# Data fetcher
# ---------------------------------------------------------------------------

async def fetch_historical_1m(
    symbol: str,
    n_bars: int = 5000,
) -> pd.DataFrame:
    """
    Fetch up to n_bars of 1m candles from Binance REST API.
    Binance max per request is 1000 bars, so we paginate.
    Returns a DataFrame with columns: open, high, low, close, volume
    """
    import aiohttp

    REST_BASE = "https://api.binance.us/api/v3/klines"
    all_bars = []
    end_time = None
    bars_remaining = n_bars

    logger.info(f"Fetching {n_bars} x 1m bars for {symbol}...")

    async with aiohttp.ClientSession() as session:
        while bars_remaining > 0:
            limit = min(bars_remaining, 1000)
            params = {
                "symbol": symbol.upper(),
                "interval": "1m",
                "limit": limit,
            }
            if end_time:
                params["endTime"] = end_time

            async with session.get(REST_BASE, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            if not data:
                break

            all_bars = data + all_bars
            bars_remaining -= len(data)
            end_time = data[0][0] - 1  # go further back in time

            if len(data) < limit:
                break

            await asyncio.sleep(0.2)  # rate limit courtesy

    logger.info(f"  Got {len(all_bars)} bars for {symbol}")

    rows = []
    for k in all_bars:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

class MarketPredictor:
    """
    Trains and serves a price direction prediction model.

    Usage:
        predictor = MarketPredictor(symbol="BTCUSDT")
        await predictor.train(n_bars=5000)
        prediction = predictor.predict(latest_features_dict)
        # prediction = {"direction": 1, "probability": 0.67, "confidence": "medium"}
    """

    MODEL_DIR = Path("ai/models")

    def __init__(self, symbol: str, forward_bars: int = 1) -> None:
        self.symbol = symbol.upper()
        self.forward_bars = forward_bars
        self.model = None
        self.feature_names: List[str] = []
        self.train_accuracy: float = 0.0
        self.test_accuracy: float = 0.0
        self.feature_importance: Dict[str, float] = {}
        self._last_trained: Optional[datetime] = None
        self._df_cache: Optional[pd.DataFrame] = None

    @property
    def model_path(self) -> Path:
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return self.MODEL_DIR / f"{self.symbol}_1m_predictor.pkl"

    async def train(self, n_bars: int = 5000) -> dict:
        """
        Full training pipeline:
        1. Fetch historical data
        2. Build features
        3. Train/test split (80/20, time-ordered — NO shuffling)
        4. Train XGBoost classifier
        5. Evaluate and save
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import accuracy_score, classification_report
        except ImportError:
            raise ImportError(
                "Install required packages: pip install xgboost scikit-learn pandas numpy"
            )

        logger.info(f"Training predictor for {self.symbol}...")

        # 1. Fetch data
        df = await fetch_historical_1m(self.symbol, n_bars=n_bars)
        self._df_cache = df

        # 2. Build features and labels
        features = build_features(df)
        labels = build_labels(df, forward_bars=self.forward_bars)

        # Align and drop NaN rows (from rolling windows)
        combined = pd.concat([features, labels.rename("target")], axis=1)
        combined = combined.iloc[:-self.forward_bars]  # drop rows with no future label
        combined = combined.replace([float("inf"), float("-inf")], float("nan")).dropna()

        X = combined.drop(columns=["target"])
        y = combined["target"]

        # Clip extreme outliers per feature (1st/99th percentile)
        X = X.clip(lower=X.quantile(0.001), upper=X.quantile(0.999), axis=1)

        # Final safety check — drop any remaining NaN/inf rows
        mask = X.replace([float("inf"), float("-inf")], float("nan")).notna().all(axis=1)
        X = X[mask]
        y = y[mask]

        self.feature_names = list(X.columns)

        logger.info(f"  Training on {len(X)} samples, {len(self.feature_names)} features")
        logger.info(f"  Label distribution: {y.mean():.1%} bullish bars")

        # Handle class imbalance with scale_pos_weight
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        logger.info(f"  Class balance: {n_neg} down / {n_pos} up (scale_pos_weight={scale_pos_weight:.2f})")

        # 3. Time-series split — NEVER shuffle financial data
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # 4. Train XGBoost
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            scale_pos_weight=scale_pos_weight,  # fix class imbalance
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # 5. Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        self.train_accuracy = float(accuracy_score(y_train, train_pred))
        self.test_accuracy = float(accuracy_score(y_test, test_pred))

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = dict(
            sorted(
                zip(self.feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        self._last_trained = datetime.utcnow()

        # Save model
        self._save()

        result = {
            "symbol": self.symbol,
            "n_training_bars": len(X_train),
            "n_test_bars": len(X_test),
            "train_accuracy": f"{self.train_accuracy:.1%}",
            "test_accuracy": f"{self.test_accuracy:.1%}",
            "top_features": list(self.feature_importance.items())[:5],
            "trained_at": self._last_trained.isoformat(),
        }

        logger.info(f"  Train accuracy: {self.train_accuracy:.1%}")
        logger.info(f"  Test accuracy:  {self.test_accuracy:.1%}")
        logger.info(f"  Top features:   {list(self.feature_importance.keys())[:5]}")

        return result

    def predict_from_df(self, df: pd.DataFrame) -> Optional[dict]:
        """
        Given a recent OHLCV DataFrame (needs ~60 rows minimum for indicator warmup),
        return a prediction for the NEXT bar.
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return None

        features = build_features(df)
        latest = features.iloc[[-1]][self.feature_names]

        if latest.isnull().any().any():
            return None

        prob = self.model.predict_proba(latest)[0]
        direction = int(prob[1] > 0.5)
        confidence_score = float(max(prob))

        if confidence_score > 0.65:
            confidence = "high"
        elif confidence_score > 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "symbol": self.symbol,
            "direction": direction,          # 1=up, 0=down
            "direction_label": "UP" if direction == 1 else "DOWN",
            "probability_up": float(prob[1]),
            "probability_down": float(prob[0]),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _save(self) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "train_accuracy": self.train_accuracy,
                "test_accuracy": self.test_accuracy,
                "feature_importance": self.feature_importance,
                "trained_at": self._last_trained,
                "symbol": self.symbol,
            }, f)
        logger.info(f"Model saved to {self.model_path}")

    def load(self) -> bool:
        """Load a previously trained model. Returns True if successful."""
        if not self.model_path.exists():
            return False
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.train_accuracy = data["train_accuracy"]
        self.test_accuracy = data["test_accuracy"]
        self.feature_importance = data["feature_importance"]
        self._last_trained = data["trained_at"]
        logger.info(
            f"Loaded model for {self.symbol} "
            f"(trained {self._last_trained}, test acc={self.test_accuracy:.1%})"
        )
        return True


# ---------------------------------------------------------------------------
# AI Strategy — plugs into the trading engine
# ---------------------------------------------------------------------------

class AIStrategy:
    """
    Wraps MarketPredictor as a trading strategy compatible with the engine.

    Only fires signals when:
    - Model confidence is HIGH (>65%) — filters out noise
    - We have enough recent bars for all indicators to be valid

    Sits alongside RSI/EMA strategies. You can compare their performance
    in the portfolio's strategy-pnl breakdown.
    """

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str = "1m",
        min_confidence: str = "high",  # "low", "medium", or "high"
        retrain_hours: int = 24,        # retrain model every N hours
    ) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_confidence = min_confidence
        self.retrain_hours = retrain_hours
        self._active = True
        self._predictor = MarketPredictor(symbol=symbol)
        self._bar_buffer: List[dict] = []  # rolling window of recent bars
        self._buffer_size = 100            # keep last 100 bars for prediction
        self._in_position = False
        self._last_prediction: Optional[dict] = None

    async def initialize(self) -> None:
        """
        Call once at startup. Loads saved model or trains a new one.
        """
        loaded = self._predictor.load()
        if not loaded:
            logger.info(f"No saved model for {self.symbol}, training now...")
            await self._predictor.train(n_bars=5000)
        else:
            # Check if model is stale
            age_hours = (
                datetime.utcnow() - self._predictor._last_trained
            ).total_seconds() / 3600
            if age_hours > self.retrain_hours:
                logger.info(f"Model for {self.symbol} is {age_hours:.0f}h old, retraining...")
                await self._predictor.train(n_bars=5000)

    def on_bar(self, features: Dict[str, Any]) -> Optional[Any]:
        """
        Called on every 1m bar. Buffers recent bars and runs prediction.
        Returns a Signal if confident enough, None otherwise.
        """
        if not self._active or features.get("symbol") != self.symbol:
            return None
        if features.get("timeframe") != self.timeframe:
            return None
        if self._predictor.model is None:
            return None

        # Buffer the bar
        self._bar_buffer.append({
            "open": float(features.get("open", 0)),
            "high": float(features.get("high", 0)),
            "low": float(features.get("low", 0)),
            "close": float(features.get("close", 0)),
            "volume": float(features.get("volume", 0)),
        })
        if len(self._bar_buffer) > self._buffer_size:
            self._bar_buffer.pop(0)

        # Need at least 60 bars for indicators to stabilize
        if len(self._bar_buffer) < 60:
            return None

        # Build DataFrame from buffer and predict
        df = pd.DataFrame(self._bar_buffer)
        prediction = self._predictor.predict_from_df(df)
        if not prediction:
            return None

        self._last_prediction = prediction

        # Filter by confidence
        confidence_levels = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_levels.get(self.min_confidence, 2)
        pred_level = confidence_levels.get(prediction["confidence"], 0)

        if pred_level < min_level:
            return None

        # Build signal
        from core.types import Signal, Side, SignalStrength

        close = Decimal(str(features["close"]))

        if prediction["direction"] == 1 and not self._in_position:
            self._in_position = True
            return Signal(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                side=Side.BUY,
                strength=SignalStrength.MEDIUM,
                timestamp=datetime.utcnow(),
                price=close,
                reason=f"AI predicts UP | prob={prediction['probability_up']:.0%} | conf={prediction['confidence']}",
                metadata=prediction,
            )

        elif prediction["direction"] == 0 and self._in_position:
            self._in_position = False
            return Signal(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                side=Side.SELL,
                strength=SignalStrength.MEDIUM,
                timestamp=datetime.utcnow(),
                price=close,
                reason=f"AI predicts DOWN | prob={prediction['probability_down']:.0%} | conf={prediction['confidence']}",
                metadata=prediction,
            )

        return None

    def on_fill(self, fill) -> None:
        from core.types import Side
        if fill.strategy_id == self.strategy_id and fill.side == Side.SELL:
            self._in_position = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self): self._active = True
    def deactivate(self): self._active = False

    def get_last_prediction(self) -> Optional[dict]:
        return self._last_prediction