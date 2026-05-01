# ETA Prediction — Gobblecube Take-Home Challenge

Predict NYC taxi trip duration from a ride request. Submission for the Gobblecube ETA Challenge.

---

## Problem

Given a ride request containing pickup zone, dropoff zone, timestamp, and passenger count, predict the trip duration in seconds. The model is evaluated on **Mean Absolute Error (MAE)** against a held-out 2024 slice of NYC Yellow Taxi data.

---

## My Approach

### Data
- **Training**: 1M randomly sampled trips from the 2023 NYC TLC dataset (`sample_1M.parquet`)
- **Evaluation**: Last 2 weeks of 2023 (~1M trips, `dev.parquet`)
- Full 37M-row dataset was not loaded into memory — sampled at read-time to avoid OOM on laptop hardware

### Model
**XGBoost Regressor** (`tree_method="hist"`, CPU-only)

| Hyperparameter    | Value  |
|-------------------|--------|
| `n_estimators`    | 300    |
| `max_depth`       | 10     |
| `learning_rate`   | 0.05   |
| `subsample`       | 0.8    |
| `colsample_bytree`| 0.8    |

### Feature Engineering

| Feature          | Description                                           |
|------------------|-------------------------------------------------------|
| `pickup_zone`    | NYC taxi zone ID (1–265)                              |
| `dropoff_zone`   | NYC taxi zone ID (1–265)                              |
| `hour`           | Hour of pickup (0–23)                                 |
| `dow`            | Day of week (0=Monday … 6=Sunday)                     |
| `month`          | Month of pickup (1–12)                                |
| `passenger_count`| Number of passengers                                  |
| `route_id`       | `pickup_zone * 300 + dropoff_zone` — unique O-D pair fingerprint |
| `rush_hour`      | 1 if hour ∈ {7–10, 17–20}, else 0 — AM/PM peak congestion signal |
| `is_weekend`     | 1 if Saturday or Sunday, else 0                       |
| `zone_pair`      | `pickup_zone + dropoff_zone` — lightweight zone-interaction proxy |

### Inference Design
- `predict.py` uses **no pandas at inference time** — features are computed with plain Python and passed directly to the model as a NumPy array
- Predictions are **clipped to [30, 10800] seconds** to eliminate physically impossible outputs
- Inference is well under the 200 ms constraint on CPU

---

## Results

| Approach                         | Dev MAE     |
|----------------------------------|-------------|
| Global mean baseline             | ~580 s      |
| Zone-pair lookup (no ML)         | ~300 s      |
| GBT baseline (starter repo)      | ~350 s      |
| **This submission (XGBoost)**    | **~321 s**  |

**~29-second improvement over the provided GBT baseline.**

---

## How to Run

### 1. Setup

```bash
git clone <your-repo-url>
cd eta-challenge-starter

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data (one-time, ~500 MB)

```bash
python data/download_data.py
```

### 3. Train model

```bash
python baseline.py
# Writes model.pkl — takes ~3–5 min on a laptop CPU
```

### 4. Score on Dev set

```bash
python grade.py
```

### 5. Run smoke tests

```bash
pytest tests/test_submission.py
```

### 6. Build and test Docker image

```bash
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

---

## Tech Stack

- **Python 3.10+**
- **pandas** — data loading and feature engineering during training
- **numpy** — fast inference without pandas overhead
- **xgboost** — gradient-boosted tree regressor
- **pyarrow** — incremental parquet I/O for memory-efficient data pipeline
- **pytest** — submission contract validation

---

## Memory Engineering

The original pipeline crashed with a `MemoryError` loading ~37M rows. Fixed by:

- Reading one monthly parquet file at a time in `download_data.py`
- Appending cleaned rows to output files via **PyArrow streaming writers** (no full-dataset concat in RAM)
- Building `sample_1M.parquet` via a **streaming reservoir sampler** over `train.parquet`
- Peak RAM during training: **~400–600 MB**

---

## What I'd Try Next

- **Route-level historical averages** — precompute mean/median duration per `route_id` and join as a feature; this is likely the single highest-leverage improvement given the zone-pair lookup already scores ~300 s
- **Cyclic time encoding** — encode `hour` and `dow` as `sin/cos` pairs so the model sees temporal continuity (e.g. 23:00 and 00:00 are adjacent)
- **External features** — NOAA weather data (precipitation, temperature) and NYC public holiday flags
- **LightGBM** — generally faster to train than XGBoost on tabular data of this size; worth benchmarking
- **Train on full dataset** — using the incremental pipeline already built, training on all 37M rows with LightGBM should be feasible on a machine with 16 GB RAM
