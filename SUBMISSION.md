# Submission Writeup

---

## Your final score

Dev MAE: **321.1 s**

---

## Your approach, in one paragraph

An XGBoost regression model is trained on a 1M-row sampled subset of the NYC TLC 2023 dataset to predict trip duration. The baseline feature set (pickup zone, dropoff zone, hour, day of week, month, passenger count) is extended with additional engineered features including a weekend indicator, rush-hour flag, and a route-level proxy using absolute zone difference. The model is configured with `n_estimators=300`, `max_depth=10`, and `learning_rate=0.07` using histogram-based tree construction for efficient CPU training. Predictions are constrained within a valid range of 30 to 10,800 seconds. The inference pipeline is implemented using NumPy for minimal overhead and fast execution.

---

## What you tried that didn't work

Training on the full dataset was not feasible due to memory constraints on local hardware — loading all 12 monthly parquet files caused a `MemoryError` before training could begin. Increasing model depth and number of estimators beyond a certain threshold resulted in marginal MAE improvements while significantly increasing training time, suggesting the bottleneck was feature signal rather than model capacity. Using raw zone identifiers without additional feature engineering produced weaker performance, confirming that zone-level interaction and temporal context features carry meaningful predictive signal.

---

## Where AI tooling sped you up most

AI-assisted workflows accelerated three specific areas: debugging the memory crash in the data pipeline and identifying a streaming, month-by-month processing approach as the fix; brainstorming feature engineering additions beyond the baseline six features; and refactoring the inference path in `predict.py` to eliminate pandas overhead. The tooling was particularly effective at surfacing efficient patterns for handling large tabular datasets within tight compute constraints. It was less useful for architectural decisions that required empirical validation — suggestions like zone embedding layers were directionally reasonable but impractical within the available training time.

---

## Next experiments

- **Geographic distance from zone centroids** — compute Haversine distance between pickup and dropoff zone centroids using the NYC taxi zone shapefile; a geometrically meaningful distance feature should outperform the current zone-difference proxy
- **Route-level historical averages** — precompute mean and median trip duration per origin-destination pair from the full training set and join as features; the naive zone-pair lookup already scores ~300 s, suggesting per-route statistics carry significant untapped signal
- **LightGBM or CatBoost** — benchmark against XGBoost on the same feature set; LightGBM in particular supports efficient training on larger data slices within the same RAM budget
- **External signals** — join NOAA hourly precipitation and temperature data by timestamp; rain events measurably increase NYC taxi durations and the data is publicly available

---

## How to reproduce

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download and build data splits (one-time, ~500 MB)
python data/download_data.py

# Train model — writes model.pkl
python baseline.py

# Score on Dev set
python grade.py
```

---

_Total time spent on this challenge: ~5 hours._
