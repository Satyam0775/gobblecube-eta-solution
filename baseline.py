#!/usr/bin/env python
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

TRAIN_FILE = "sample_1M.parquet"
DEV_FILE = "dev.parquet"

# 🔥 ADD NEW FEATURES
FEATURES = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
    "month",
    "passenger_count",
    "is_weekend",
    "is_rush_hour",
    "zone_diff"   # 🔥 NEW
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])

    return pd.DataFrame({
        "pickup_zone": df["pickup_zone"].astype("int32"),
        "dropoff_zone": df["dropoff_zone"].astype("int32"),
        "hour": ts.dt.hour.astype("int8"),
        "dow": ts.dt.dayofweek.astype("int8"),
        "month": ts.dt.month.astype("int8"),
        "passenger_count": df["passenger_count"].astype("int8"),

        # 🔥 NEW FEATURES
        "is_weekend": (ts.dt.dayofweek >= 5).astype("int8"),
        "is_rush_hour": ts.dt.hour.isin([7,8,9,17,18,19]).astype("int8"),

        # 🔥 DISTANCE PROXY (very powerful)
        "zone_diff": np.abs(df["pickup_zone"] - df["dropoff_zone"]).astype("int16"),
    })[FEATURES]


def main() -> None:
    train_path = DATA_DIR / TRAIN_FILE
    dev_path = DATA_DIR / DEV_FILE

    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run download_data.py first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)

    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("\nEngineering features...")
    X_train = engineer_features(train)
    y_train = train["duration_seconds"].to_numpy(dtype="float32")

    X_dev = engineer_features(dev)
    y_dev = dev["duration_seconds"].to_numpy(dtype="float32")

    del train, dev

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=300,   # 🔥 slightly more trees
        max_depth=10,       # 🔥 deeper learning
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s")

    print("\nEvaluating...")
    preds = model.predict(X_dev)

    preds = np.clip(preds, 30, 3 * 3600)

    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"🔥 Dev MAE: {mae:.1f} seconds")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()