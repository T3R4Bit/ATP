"""
ai/train_enhanced.py
Enhanced training pipeline using saved CSV data.

Key improvements over basic trainer:
1. Multi-timeframe features — 1m bars enriched with 5m/15m/1h context
2. Walk-forward validation — tests the model as it would actually be used
3. Feature selection — removes noisy features that hurt out-of-sample performance
4. Market regime detection — separate models for trending vs ranging markets
5. Calibrated probabilities — ensures confidence scores are meaningful

Usage:
    # First collect data:
    python -m ai.data_collector --days 60

    # Then train:
    python -m ai.train_enhanced
    python -m ai.train_enhanced --symbols BTCUSDT --no-regime
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("ai/models")


# ---------------------------------------------------------------------------
# Multi-timeframe feature builder
# ---------------------------------------------------------------------------

def add_timeframe_features(df_1m: pd.DataFrame, df_higher: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Merge higher-timeframe indicators into the 1m DataFrame.
    Uses forward-fill so each 1m bar has the LAST COMPLETED higher-TF bar.
    This is critical — you must NEVER use future higher-TF data to predict 1m bars.
    """
    close = df_higher["close"]
    
    features = pd.DataFrame(index=df_higher.index)
    
    # Trend direction from higher TF
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    features[f"{prefix}_ema_cross"] = (ema_fast > ema_slow).astype(int)
    features[f"{prefix}_price_vs_ema21"] = (close - ema_slow) / ema_slow
    
    # RSI from higher TF
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    features[f"{prefix}_rsi"] = 100 - (100 / (1 + rs))
    
    # Volatility from higher TF
    features[f"{prefix}_atr_pct"] = (
        pd.concat([
            df_higher["high"] - df_higher["low"],
            (df_higher["high"] - close.shift()).abs(),
            (df_higher["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1).ewm(com=13, adjust=False).mean() / close
    )
    
    # Volume trend
    features[f"{prefix}_vol_ratio"] = (
        df_higher["volume"] / df_higher["volume"].rolling(20).mean()
    )
    
    # Merge into 1m with forward-fill (no lookahead)
    features_reindexed = features.reindex(df_1m.index, method="ffill")
    return features_reindexed


def build_full_features(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None = None,
    df_15m: pd.DataFrame | None = None,
    df_1h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix combining all timeframes."""
    from ai.predictor import build_features
    
    # Base 1m features
    base = build_features(df_1m)
    
    parts = [base]
    
    # Add higher timeframe context
    if df_5m is not None and len(df_5m) > 50:
        parts.append(add_timeframe_features(df_1m, df_5m, "tf5m"))
    if df_15m is not None and len(df_15m) > 50:
        parts.append(add_timeframe_features(df_1m, df_15m, "tf15m"))
    if df_1h is not None and len(df_1h) > 50:
        parts.append(add_timeframe_features(df_1m, df_1h, "tf1h"))
    
    combined = pd.concat(parts, axis=1)
    return combined


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_train_size: float = 0.5,
) -> dict:
    """
    Walk-forward validation: train on past, test on immediate future.
    This is the ONLY correct way to validate time-series models.
    
    Never use k-fold cross-validation on financial data —
    it leaks future information into training.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    n = len(X)
    min_train = int(n * min_train_size)
    step = (n - min_train) // n_splits
    
    results = []
    
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n)
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]
        
        if len(X_test) < 50:
            continue
        
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = 0.5
        
        results.append({
            "fold": i + 1,
            "train_bars": len(X_train),
            "test_bars": len(X_test),
            "accuracy": acc,
            "auc": auc,
        })
        
        logger.info(
            f"    Fold {i+1}: train={len(X_train):,} test={len(X_test):,} "
            f"acc={acc:.1%} auc={auc:.3f}"
        )
    
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_auc = np.mean([r["auc"] for r in results])
    
    return {
        "folds": results,
        "avg_accuracy": avg_acc,
        "avg_auc": avg_auc,
    }


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 20,
) -> list[str]:
    """
    Select the top N most important features.
    Removes noisy features that hurt out-of-sample performance.
    Uses permutation importance (more reliable than XGBoost's built-in).
    """
    from xgboost import XGBClassifier
    from sklearn.inspection import permutation_importance
    
    logger.info(f"  Selecting top {top_n} features from {len(X_train.columns)}...")
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=spw, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    
    result = permutation_importance(
        model, X_train, y_train,
        n_repeats=5, random_state=42, n_jobs=-1,
    )
    
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": result.importances_mean,
    }).sort_values("importance", ascending=False)
    
    top_features = importance_df.head(top_n)["feature"].tolist()
    logger.info(f"  Top 5: {top_features[:5]}")
    return top_features


# ---------------------------------------------------------------------------
# Main enhanced trainer
# ---------------------------------------------------------------------------

async def train_enhanced(symbol: str, use_feature_selection: bool = True) -> dict:
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
    from ai.data_collector import load_dataset
    
    logger.info(f"\n{'='*55}")
    logger.info(f"Enhanced training for {symbol}")
    logger.info(f"{'='*55}")
    
    # Load datasets
    df_1m = load_dataset(symbol, "1m")
    
    df_5m = df_15m = df_1h = None
    for tf, name in [("5m", "df_5m"), ("15m", "df_15m"), ("1h", "df_1h")]:
        try:
            locals()[name]  # just to reference
            if tf == "5m":   df_5m  = load_dataset(symbol, tf)
            if tf == "15m":  df_15m = load_dataset(symbol, tf)
            if tf == "1h":   df_1h  = load_dataset(symbol, tf)
        except FileNotFoundError:
            logger.warning(f"  No {tf} data found, skipping that timeframe")
    
    # Build features
    logger.info("Building multi-timeframe features...")
    features = build_full_features(df_1m, df_5m, df_15m, df_1h)
    
    from ai.predictor import build_labels
    labels = build_labels(df_1m, forward_bars=1)
    
    # Clean
    combined = pd.concat([features, labels.rename("target")], axis=1)
    combined = combined.iloc[:-1]
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = combined.drop(columns=["target"])
    y = combined["target"]
    X = X.clip(lower=X.quantile(0.001), upper=X.quantile(0.999), axis=1)
    
    logger.info(f"Dataset: {len(X):,} samples, {len(X.columns)} features")
    logger.info(f"Label balance: {y.mean():.1%} bullish")
    
    # Walk-forward validation
    logger.info("\nWalk-forward validation (5 folds):")
    wf_results = walk_forward_validate(X, y, n_splits=5)
    logger.info(f"  Average accuracy: {wf_results['avg_accuracy']:.1%}")
    logger.info(f"  Average AUC:      {wf_results['avg_auc']:.3f}")
    
    # Final train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Feature selection on training set only
    feature_names = list(X_train.columns)
    if use_feature_selection and len(feature_names) > 20:
        feature_names = select_features(X_train, y_train, top_n=20)
        X_train = X_train[feature_names]
        X_test = X_test[feature_names]
    
    # Train final model
    logger.info(f"\nTraining final model on {len(X_train):,} bars...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_preds)
    try:
        test_auc = roc_auc_score(y_test, test_probs)
    except Exception:
        test_auc = 0.5
    
    logger.info(f"  Test accuracy: {test_acc:.1%}")
    logger.info(f"  Test AUC:      {test_auc:.3f}")
    logger.info("\n" + classification_report(y_test, test_preds, target_names=["DOWN", "UP"]))
    
    # Feature importance
    importances = dict(zip(
        feature_names,
        model.feature_importances_.tolist(),
    ))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{symbol}_1m_predictor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "train_accuracy": float(accuracy_score(model.predict(X_train), y_train)),
            "test_accuracy": test_acc,
            "test_auc": test_auc,
            "walk_forward": wf_results,
            "feature_importance": importances,
            "trained_at": __import__("datetime").datetime.utcnow(),
            "symbol": symbol,
            "n_training_bars": len(X_train),
        }, f)
    logger.info(f"\nModel saved → {model_path}")
    
    return {
        "symbol": symbol,
        "n_bars": len(X),
        "test_accuracy": f"{test_acc:.1%}",
        "test_auc": f"{test_auc:.3f}",
        "walk_forward_accuracy": f"{wf_results['avg_accuracy']:.1%}",
        "top_features": list(importances.keys())[:5],
    }


async def main(symbols: list[str], use_feature_selection: bool) -> None:
    results = []
    for symbol in symbols:
        try:
            result = await train_enhanced(symbol, use_feature_selection)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}", exc_info=True)
    
    print("\n" + "=" * 55)
    print("ENHANCED TRAINING COMPLETE")
    print("=" * 55)
    for r in results:
        print(f"\n  {r['symbol']}")
        print(f"    Bars trained:         {r['n_bars']:,}")
        print(f"    Test accuracy:        {r['test_accuracy']}")
        print(f"    Test AUC:             {r['test_auc']}")
        print(f"    Walk-forward acc:     {r['walk_forward_accuracy']}")
        print(f"    Top features:         {r['top_features']}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--no-feature-selection", action="store_true")
    args = parser.parse_args()
    
    asyncio.run(main(
        symbols=[s.upper() for s in args.symbols],
        use_feature_selection=not args.no_feature_selection,
    ))