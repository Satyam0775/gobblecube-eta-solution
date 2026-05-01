#!/usr/bin/env python
"""One-time download & cleanup of NYC TLC 2023 yellow-taxi data.

Produces:
    data/train.parquet       -- 11.5 months of 2023, ~37M trips after cleaning
    data/dev.parquet         -- last 2 weeks of 2023, ~1M trips (for local grading)
    data/sample_1M.parquet   -- 1M-row subset of train for fast iteration

The held-out Eval set (a 2024 slice) is kept by Gobblecube and never distributed.

Memory-efficient design:
    - Processes one month at a time (never loads all 12 months into RAM)
    - Cleans and splits each month independently
    - Appends results to Parquet files using PyArrow streaming writers
    - Peak RAM usage: ~400-600 MB (one month ~3M rows at a time)
    - Compatible with 8-16 GB RAM systems on Windows + CPU (pandas only)

Takes ~5 minutes on a fast connection, ~20 minutes on a slow one.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_URL  = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS    = [f"2023-{m:02d}" for m in range(1, 13)]

DATA_DIR  = Path(__file__).parent
RAW_DIR   = DATA_DIR / "raw"

CUTOFF      = pd.Timestamp("2023-12-18")   # dev = last ~2 weeks of Dec
SAMPLE_SIZE = 1_000_000

# Columns read from each raw parquet — only what we need to minimise I/O
RAW_COLS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
]

# Final schema written to every output parquet (train / dev / sample)
OUTPUT_SCHEMA = pa.schema([
    pa.field("pickup_zone",      pa.int32()),
    pa.field("dropoff_zone",     pa.int32()),
    pa.field("requested_at",     pa.string()),
    pa.field("passenger_count",  pa.int8()),
    pa.field("duration_seconds", pa.float64()),
])


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

def download_month(yyyymm: str) -> Path:
    """Download one monthly parquet if not already cached."""
    RAW_DIR.mkdir(exist_ok=True)
    url = f"{BASE_URL}/yellow_tripdata_{yyyymm}.parquet"
    out = RAW_DIR / f"yellow_{yyyymm}.parquet"
    if out.exists():
        print(f"  cached   {out.name}")
        return out
    print(f"  fetching {url}")
    urlretrieve(url, out)
    return out


# -----------------------------------------------------------------------------
# Per-month clean  (returns a small, already-filtered DataFrame)
# -----------------------------------------------------------------------------

def clean_month(path: Path) -> pd.DataFrame:
    """
    Load, clean and return one month of data.

    Memory note: we read only the 5 required columns, compute duration
    in-place, apply all filters, then discard the raw frame immediately.
    Peak memory for a single month is ~150-300 MB, well within budget.
    """
    raw = pd.read_parquet(path, columns=RAW_COLS)

    duration = (
        raw["tpep_dropoff_datetime"] - raw["tpep_pickup_datetime"]
    ).dt.total_seconds()

    df = pd.DataFrame({
        "pickup_zone":      raw["PULocationID"].astype("int32"),
        "dropoff_zone":     raw["DOLocationID"].astype("int32"),
        "requested_at":     raw["tpep_pickup_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "passenger_count":  raw["passenger_count"].fillna(1).astype("int8"),
        "duration_seconds": duration.astype("float64"),
        "_ts":              raw["tpep_pickup_datetime"],   # used for split; dropped later
    })

    # Discard raw frame immediately to free RAM before applying filter
    del raw, duration

    # Cleaning rules:
    #   - trip duration between 30 seconds and 3 hours
    #   - valid taxi zones (1-265)
    #   - only 2023 records (guards against mislabelled rows in TLC data)
    mask = (
        (df["duration_seconds"] >= 30)
        & (df["duration_seconds"] <= 3 * 3600)
        & (df["pickup_zone"].between(1, 265))
        & (df["dropoff_zone"].between(1, 265))
        & (df["_ts"].dt.year == 2023)
    )
    return df.loc[mask].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Incremental writer helpers
# -----------------------------------------------------------------------------

def _to_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert output columns (no _ts) to a PyArrow table with OUTPUT_SCHEMA."""
    out = df.drop(columns=["_ts"])
    return pa.Table.from_pandas(out, schema=OUTPUT_SCHEMA, preserve_index=False)


def _append(writer: pq.ParquetWriter, df: pd.DataFrame) -> None:
    """Append a pandas DataFrame to an open ParquetWriter (no-op if empty)."""
    if len(df):
        writer.write_table(_to_arrow(df))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    print("Step 1: download monthly parquets")
    paths = [download_month(m) for m in MONTHS]

    # -------------------------------------------------------------------------
    # Step 2 + 3 combined: process month-by-month and stream directly into
    # train.parquet / dev.parquet — no full-dataset concat in RAM.
    # -------------------------------------------------------------------------
    print("\nStep 2 + 3: clean each month and write train / dev incrementally")

    train_path = DATA_DIR / "train.parquet"
    dev_path   = DATA_DIR / "dev.parquet"

    # Remove stale output files so writers always start fresh
    train_path.unlink(missing_ok=True)
    dev_path.unlink(missing_ok=True)

    total_clean = 0
    total_train = 0
    total_dev   = 0

    with (
        pq.ParquetWriter(train_path, OUTPUT_SCHEMA) as train_writer,
        pq.ParquetWriter(dev_path,   OUTPUT_SCHEMA) as dev_writer,
    ):
        for path in paths:
            print(f"  processing {path.name} ...", end=" ", flush=True)

            month_df = clean_month(path)
            total_clean += len(month_df)

            # Split using same CUTOFF as original pipeline
            train_rows = month_df[month_df["_ts"] <  CUTOFF]
            dev_rows   = month_df[month_df["_ts"] >= CUTOFF]

            # Stream rows into the open parquet writers (no in-memory concat)
            _append(train_writer, train_rows)
            _append(dev_writer,   dev_rows)

            total_train += len(train_rows)
            total_dev   += len(dev_rows)

            print(
                f"{len(month_df):>8,} clean rows  "
                f"(train={len(train_rows):,}  dev={len(dev_rows):,})"
            )

            # Explicitly release month frame before next iteration
            del month_df, train_rows, dev_rows

    print(f"\n  cleaned total : {total_clean:,} trips")
    print(f"  train.parquet : {total_train:,} rows  ->  {train_path}")
    print(f"  dev.parquet   : {total_dev:,} rows  ->  {dev_path}")

    # -------------------------------------------------------------------------
    # Step 4: 1M-row sample.
    # Stream train.parquet in batches; keep a rolling reservoir trimmed to
    # 2x SAMPLE_SIZE so we never load the full train set into RAM.
    # -------------------------------------------------------------------------
    print("\nStep 4: build 1M-row training sample (streaming reservoir)")

    sample_path = DATA_DIR / "sample_1M.parquet"
    sample_path.unlink(missing_ok=True)

    pf        = pq.ParquetFile(train_path)
    reservoir = []    # accumulates DataFrames; periodically trimmed
    seen      = 0

    for batch in pf.iter_batches(batch_size=200_000):
        chunk = batch.to_pandas()
        reservoir.append(chunk)
        seen += len(chunk)

        # Once reservoir exceeds 2x target, downsample to SAMPLE_SIZE to keep
        # memory bounded; use random_state for reproducibility across runs.
        if seen > 2 * SAMPLE_SIZE:
            combined  = pd.concat(reservoir, ignore_index=True)
            reservoir = [combined.sample(n=SAMPLE_SIZE, random_state=42)]
            seen      = SAMPLE_SIZE

    # Final draw from whatever remained in the reservoir
    combined = pd.concat(reservoir, ignore_index=True)
    sample   = (
        combined
        .sample(n=min(SAMPLE_SIZE, len(combined)), random_state=42)
        .reset_index(drop=True)
    )

    sample.to_parquet(sample_path, index=False)
    print(f"  sample_1M.parquet: {len(sample):,} rows  ->  {sample_path}")

    print("\nDone. Next: `python baseline.py`")


if __name__ == "__main__":
    main()