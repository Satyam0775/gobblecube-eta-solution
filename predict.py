"""Submission interface — this is what Gobblecube's grader imports."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

# Load model once (fast inference)
with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

# Disable feature-name validation for speed (XGBoost optimization)
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None


def predict(request: dict) -> float:
    """
    Predict trip duration in seconds.

    Input:
        {
            "pickup_zone": int,
            "dropoff_zone": int,
            "requested_at": str (ISO format),
            "passenger_count": int,
        }

    Output:
        float (duration in seconds)
    """

    ts = datetime.fromisoformat(request["requested_at"])

    # 🔥 SAME FEATURES AS baseline.py (MUST MATCH ORDER)
    is_weekend = 1 if ts.weekday() >= 5 else 0
    is_rush_hour = 1 if ts.hour in [7, 8, 9, 17, 18, 19] else 0
    zone_diff = abs(int(request["pickup_zone"]) - int(request["dropoff_zone"]))

    x = np.array(
        [[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
            is_weekend,
            is_rush_hour,
            zone_diff,
        ]],
        dtype=np.int32,
    )

    pred = float(_MODEL.predict(x)[0])

    # 🔥 SAFETY CLIP (important for grader stability)
    return float(np.clip(pred, 30, 3 * 3600))