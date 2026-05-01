# Submission Writeup

---

## Your final score

Dev MAE: **321.1 s**

---

## Your approach, in one paragraph

I built an XGBoost regressor trained on a 1M-row random sample of the 2023 NYC TLC dataset. The six baseline features (pickup zone, dropoff zone, hour, day of week, month, passenger count) were extended with four engineered features: a `route_id` encoding each unique origin-destination pair as a single integer (`pickup_zone * 300 + dropoff_zone`), a `rush_hour` binary flag covering AM and PM peak windows (7–10, 17–20), an `is_weekend` flag, and a `zone_pair` sum as a lightweight zone-interaction proxy. The model was tuned to `n_estimators=300`, `max_depth=10`, `learning_rate=0.05` using histogram-based tree building (`tree_method="hist"`) for CPU efficiency. Predictions are clipped to [30, 10800] seconds to eliminate physically impossible outputs. Inference in `predict.py` uses plain NumPy — no pandas overhead — and runs well under the 200 ms constraint.

---

## What you tried that didn't work

**Training on the full 37M-row dataset.** Loading all 12 monthly parquet files and concatenating them into a single DataFrame caused a `MemoryError` on 8–16 GB machines before training even started. I ultimately solved this by building a streaming pipeline that processes one month at a time and appends to output files via PyArrow writers, but the XGBoost model was still trained on the 1M sample because fitting on 37M rows exceeded practical CPU training time.

**Increasing model complexity past a point.** Pushing `n_estimators` beyond 300 and `max_depth` beyond 10 on 1M rows gave diminishing MAE returns while training time grew significantly. The bottleneck appears to be feature information rather than model capacity — the six base features plus four engineered ones don't carry enough signal to justify a much deeper ensemble.

**Zone difference as a distance proxy.** Using `abs(pickup_zone - dropoff_zone)` as a rough distance feature did not meaningfully reduce MAE. Zone IDs are not spatially ordered, so arithmetic differences between them are largely noise. Real geographic distance (Haversine from zone centroids) would be the correct approach here.

---

## Where AI tooling sped you up most

Claude (via claude.ai) was useful in three specific places:

**Debugging the MemoryError pipeline.** Describing the crash context and constraints (pandas-only, no Dask, Windows-compatible) produced a working incremental PyArrow writer approach quickly. This would have taken considerably longer to work out from documentation alone.

**Feature engineering brainstorm.** Prompting with the schema and baseline MAE surfaced the `route_id` integer encoding idea, which turned out to be the highest-leverage single feature. I wouldn't have reached for a zone-pair fingerprint as the first thing to try.

**Inference refactor.** Claude helped restructure `predict.py` to remove pandas from the hot path and compute features directly from the request dict using NumPy, which kept inference latency well within the 200 ms constraint.

Where it fell short: suggestions around model architecture (e.g. adding embedding layers for zone IDs) were directionally correct but required more compute and setup time than available, so they stayed as "next experiments."

---

## Next experiments

1. **Route-level historical averages as a feature.** Precompute mean and median `duration_seconds` per `route_id` from the full training set and join them at training and inference time. The naive zone-pair lookup already scores ~300 s, which suggests per-route statistics carry more signal than any time-based feature added so far.

2. **Real geographic distance.** Compute Haversine distance between zone centroids using the NYC taxi zone shapefile. This replaces the failed zone-difference proxy with something geometrically meaningful.

3. **LightGBM on the full dataset.** The streaming pipeline already builds `train.parquet` incrementally. LightGBM's `Dataset` API supports streaming from disk, which could make training on all 37M rows feasible on a 16 GB machine without loading everything into RAM.

4. **Weather features.** Join NOAA hourly precipitation and temperature for JFK/LGA to each trip by timestamp. Rain events visibly spike NYC taxi durations and are publicly available.

---

## How to reproduce

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download and build data splits (one-time, ~500 MB)
python data/download_data.py

# 3. Train model — writes model.pkl (~3-5 min on CPU)
python baseline.py

# 4. Score on Dev set
python grade.py
```

---

_Total time spent on this challenge: ~6 hours._
