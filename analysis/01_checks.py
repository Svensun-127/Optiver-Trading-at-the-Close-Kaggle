"""
Data investigation script for Optiver Trading at the Close competition.

Loads train.csv, profiles all columns, checks missing values, distributions,
target behaviour, and potential data leakage signals. Designed to be
reproducible: run this script standalone and it will print a full report.

Usage:
    python analysis/01_checks.py      # defaults to data/train.csv
    python analysis/01_checks.py --data path/to/train.csv
"""

import argparse
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest dtype that preserves values.

    Iterates over every numeric column in *df*, computes min/max bounds,
    and converts from float64→float32 and int64→int32/16/8 where safe.
    Returns a new DataFrame (does not mutate the input).

    Why it matters in finance:
        The raw CSV stores WAP, bid/ask prices, and imbalance figures as
        float64.  For tree-based models (LightGBM/XGBoost) float32 is
        sufficient and cuts memory footprint roughly in half, which matters
        on a 5.24M-row dataset.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  Memory before downcast : {start_mem:.1f} MB")

    int_cols = df.select_dtypes(include=["int"]).columns
    float_cols = df.select_dtypes(include=["float"]).columns

    for col in float_cols:
        c_min = df[col].min()
        c_max = df[col].max()
        if pd.isna(c_min):
            continue
        df[col] = df[col].astype(np.float32)

    for col in int_cols:
        c_min = df[col].min()
        c_max = df[col].max()
        if pd.isna(c_min):
            continue
        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  Memory after  downcast : {end_mem:.1f} MB")
        print(f"  Reduction               : {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Load the training CSV from *path* with memory-optimised types.

    Returns a pandas DataFrame with `row_id` kept as string (it is an
    identifier, not a feature) and all numeric columns downcast to the
    smallest viable dtype.
    """
    print("=" * 72)
    print("Loading training data ...")
    print(f"  Source: {path}")
    df = pd.read_csv(path, dtype={"row_id": str})
    print(f"  Raw shape: {df.shape}")
    df = reduce_mem_usage(df)
    return df


# ---------------------------------------------------------------------------
# Column-by-column profiling
# ---------------------------------------------------------------------------


def profile_basic(df: pd.DataFrame) -> None:
    """Print overall shape, column list, and dtypes."""
    print("\n" + "=" * 72)
    print("1. Basic Information")
    print("=" * 72)
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Columns list  : {df.columns.tolist()}")
    print(f"\n  Dtypes after type optimisation:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:30s}  {str(dtype):12s}")


def profile_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column missing rates and return as a DataFrame.

    Missing values in far_price / near_price are expected and are tied to
    seconds_in_bucket (the auction imbalance data only becomes available
    after 300 seconds into the closing auction).  We flag any other column
    with >1 % missing as suspicious.
    """
    print("\n" + "=" * 72)
    print("2. Missing Value Analysis")
    print("=" * 72)

    miss = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().sum() / len(df) * 100).values,
    }).sort_values("missing_pct", ascending=False)

    print(miss.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    expected_missing = {"far_price", "near_price"}
    for _, row in miss.iterrows():
        col = row["column"]
        pct = row["missing_pct"]
        if col in expected_missing:
            print(f"  [OK] {col}: {pct:.2f}% missing (expected -- auction-data timing)")
        elif pct > 1.0:
            print(f"  [WARN] {col}: {pct:.2f}% missing -- UNEXPECTED, investigate!")
        else:
            print(f"  [OK] {col}: {pct:.2f}% missing")

    return miss


def profile_numeric(df: pd.DataFrame) -> None:
    """Print describe() for all numeric columns with 5th/95th percentiles."""
    print("\n" + "=" * 72)
    print("3. Numeric Column Summary (5th, 50th, 95th percentiles)")
    print("=" * 72)

    numeric_df = df.select_dtypes(include=["float", "int"])
    desc = numeric_df.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T
    # Keep only the most informative rows
    cols = ["count", "mean", "std", "1%", "5%", "50%", "95%", "99%"]
    print(desc[cols].to_string(float_format=lambda x: f"{x:.4f}"))


def profile_target(df: pd.DataFrame) -> None:
    """
    Analyse target distribution with financial context.

    target represents the 60-second forward price change.  Its distribution
    should be roughly zero-centred but with fatter tails than a normal
    distribution (financial returns are leptokurtic).
    """
    print("\n" + "=" * 72)
    print("4. Target ('target') Analysis")
    print("=" * 72)

    t = df["target"]
    print(f"  Count   : {len(t):,}")
    print(f"  Mean    : {t.mean():.6f}")
    print(f"  Std     : {t.std():.6f}")
    print(f"  Min     : {t.min():.6f}")
    print(f"  1%      : {t.quantile(0.01):.6f}")
    print(f"  5%      : {t.quantile(0.05):.6f}")
    print(f"  50%     : {t.quantile(0.50):.6f}")
    print(f"  95%     : {t.quantile(0.95):.6f}")
    print(f"  99%     : {t.quantile(0.99):.6f}")
    print(f"  Max     : {t.max():.6f}")
    print(f"  Skewness: {t.skew():.6f}")
    print(f"  Kurtosis: {t.kurtosis():.6f}")


# ---------------------------------------------------------------------------
# Time-series structure checks
# ---------------------------------------------------------------------------


def profile_temporal_structure(df: pd.DataFrame) -> Dict[str, int]:
    """Print date_id, time_id, seconds_in_bucket cardinality and patterns."""
    print("\n" + "=" * 72)
    print("5. Temporal Structure")
    print("=" * 72)

    n_dates = df["date_id"].nunique()
    n_stocks = df["stock_id"].nunique()
    n_times = df["time_id"].nunique()

    print(f"  Unique date_ids     : {n_dates}")
    print(f"  Unique stock_ids    : {n_stocks}")
    print(f"  Unique time_ids     : {n_times}")
    print(f"  date_id range       : [{df['date_id'].min()}, {df['date_id'].max()}]")
    print(f"  stock_id range      : [{df['stock_id'].min()}, {df['stock_id'].max()}]")
    print(f"  seconds_in_bucket range: [{df['seconds_in_bucket'].min()}, {df['seconds_in_bucket'].max()}]")

    # Expected: 200 stocks × 481 days = ~96,200 date-stock combos
    # Each combo has 55 rows (seconds_in_bucket 0-540 step 10)
    expected_rows = n_stocks * n_dates * 55
    actual_rows = len(df)
    print(f"\n  Expected rows (200×{n_dates}×55): {expected_rows:,}")
    print(f"  Actual rows                  : {actual_rows:,}")
    print(f"  Difference                   : {actual_rows - expected_rows:,}")

    # Check seconds_in_bucket values
    sec_vals = sorted(df["seconds_in_bucket"].unique())
    print(f"\n  seconds_in_bucket unique values: {len(sec_vals)}")
    print(f"  seconds_in_bucket values: {sec_vals[:5]} ... {sec_vals[-5:]}")

    # Stock-date coverage
    pivot = df.groupby("date_id")["stock_id"].nunique()
    fully_covered = (pivot == n_stocks).sum()
    print(f"\n  Dates with full stock coverage (200 stocks): {fully_covered}/{n_dates}")
    if fully_covered < n_dates:
        bad_dates = pivot[pivot < n_stocks].index.tolist()
        print(f"  Dates with partial coverage: {bad_dates}")

    return {"n_dates": n_dates, "n_stocks": n_stocks, "n_times": n_times}


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------


def check_leakage(df: pd.DataFrame) -> None:
    """
    Check for potential data leakage patterns.

    (a) date_id continuity -- are there gaps that suggest data segmented
        by distinct time periods (trading calendars)?
    (b) target autocorrelation by date_id -- does the mean target per day
        carry exploitable signal?
    (c) far_price / near_price availability vs seconds_in_bucket --
        confirm the documented 300-second threshold.

    These checks are essential because any feature that leaks future
    information must be identified before CV design.
    """
    print("\n" + "=" * 72)
    print("6. Leakage & Structural Checks")
    print("=" * 72)

    # --- (a) date_id continuity ---
    date_ids = sorted(df["date_id"].unique())
    gaps = [date_ids[i] - date_ids[i - 1] for i in range(1, len(date_ids))]
    max_gap = max(gaps) if gaps else 0
    all_one = all(g == 1 for g in gaps)
    print(f"\n  (a) date_id continuity:")
    print(f"      Range  : [{date_ids[0]}, {date_ids[-1]}]")
    print(f"      Count  : {len(date_ids)}")
    print(f"      Max gap: {max_gap}")
    print(f"      All contiguous (gap=1): {all_one}")

    # --- (b) target mean per date ---
    target_by_date = df.groupby("date_id")["target"].agg(["mean", "std", "count"])
    print(f"\n  (b) Target per date_id:")
    print(f"      Mean of daily means : {target_by_date['mean'].mean():.6f}")
    print(f"      Std of daily means  : {target_by_date['mean'].std():.6f}")
    print(f"      Min daily mean      : {target_by_date['mean'].min():.6f}")
    print(f"      Max daily mean      : {target_by_date['mean'].max():.6f}")
    # Flag if daily means drift systematically
    corr = target_by_date["mean"].corr(pd.Series(date_ids))
    print(f"      Correlation(day_idx, daily_mean_target): {corr:.4f}")
    if abs(corr) > 0.1:
        print(f"      [WARN] Non-trivial trend in daily mean target -- investigate")

    # --- (c) far_price / near_price missing rate by seconds_in_bucket ---
    print(f"\n  (c) far_price / near_price missing by seconds_in_bucket:")
    for col in ["far_price", "near_price"]:
        # Check at which seconds_in_bucket the column is populated
        avail = df.groupby("seconds_in_bucket")[col].apply(
            lambda x: x.notna().mean()
        )
        always_na = avail[avail == 0].index.tolist()
        always_present = avail[avail == 1].index.tolist()
        partial = avail[(avail > 0) & (avail < 1)].index.tolist()
        print(f"      {col}:")
        print(f"        Always missing: {always_na}")
        print(f"        Always present: {always_present}")
        if partial:
            print(f"        Partially available: {partial}")
            for s in partial:
                print(f"          seconds={s}: {avail[s]:.4%} present")

    # --- (d) stock_id balance ---
    rows_per_stock = df.groupby("stock_id").size()
    print(f"\n  (d) Rows per stock_id:")
    print(f"      Min  : {rows_per_stock.min()}")
    print(f"      Mean : {rows_per_stock.mean():.0f}")
    print(f"      Max  : {rows_per_stock.max()}")
    if rows_per_stock.min() != rows_per_stock.max():
        print(f"      [WARN] Uneven row count across stocks -- check for data gaps")

    # --- (e) Imbalance buy/sell flag distribution ---
    print(f"\n  (e) imbalance_buy_sell_flag distribution:")
    flag_counts = df["imbalance_buy_sell_flag"].value_counts().sort_index()
    for k, v in flag_counts.items():
        print(f"      {k:5d} : {v:>10,} ({v/len(df)*100:5.2f}%)")

    # --- (f) Feature correlation warnings ---
    print(f"\n  (f) Top feature-feature correlations (|r| > 0.95):")
    numeric_cols = df.select_dtypes(include=["float"]).columns.tolist()
    # Exclude target from the correlation check, use a sample for speed
    corr_cols = [c for c in numeric_cols if c != "target"]
    if len(df) > 100_000:
        sample = df[corr_cols].sample(100_000, random_state=42)
    else:
        sample = df[corr_cols]
    corr_matrix = sample.corr()
    high_corr_pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.95:
                high_corr_pairs.append((corr_cols[i], corr_cols[j], r))
    if high_corr_pairs:
        for a, b, r in high_corr_pairs:
            print(f"      {a} <-> {b}: r = {r:.4f}")
    else:
        print("      None found in sample.")


# ---------------------------------------------------------------------------
# Feature: financial meaning summaries
# ---------------------------------------------------------------------------


def print_feature_descriptions() -> None:
    """Print a brief financial description of every feature column."""
    print("\n" + "=" * 72)
    print("7. Feature Descriptions (Financial Context)")
    print("=" * 72)

    descriptions = [
        ("stock_id",        "Unique identifier for each stock."),
        ("date_id",         "Trading day identifier (integer, sequential)."),
        ("time_id",         "Unique time-bucket ID within the trading day."),
        ("seconds_in_bucket", "Seconds elapsed since the start of the closing auction (0–540)."),
        ("imbalance_size",  "Net order imbalance size at the current auction price. "
                            "Positive values indicate excess buy interest; negative indicates sell pressure."),
        ("imbalance_buy_sell_flag",
                            "Direction of the imbalance: -1 (sell-side pressure), 0 (neutral), 1 (buy-side pressure)."),
        ("reference_price", "Reference price used as baseline for the auction price calculation. "
                            "Usually the last traded price before the closing auction."),
        ("matched_size",    "Total quantity (shares) matched at the current auction price level."),
        ("far_price",       "Best price on the far side of the order book (opposite to near_price). "
                            "Only available when seconds_in_bucket >= 300."),
        ("near_price",      "Best price on the near side of the order book (closer to the spread). "
                            "Only available when seconds_in_bucket >= 300."),
        ("bid_price",       "Best bid price in the limit order book -- highest price a buyer is willing to pay."),
        ("bid_size",        "Total order size (shares) at the best bid price level."),
        ("ask_price",       "Best ask price in the limit order book -- lowest price a seller is willing to accept."),
        ("ask_size",        "Total order size (shares) at the best ask price level."),
        ("wap",             "Weighted Average Price of the current order book. "
                            "Calculated as (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)."),
        ("target",          "60-second forward price change to predict. "
                            "Formula: (future_wap - current_wap) / current_wap * 10000, "
                            "measured 60 seconds after the current snapshot."),
    ]
    for name, desc in descriptions:
        print(f"  - {name:30s}  {desc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(data_path: str) -> None:
    """Run all data checks and print a structured report to stdout."""
    df = load_data(data_path)
    profile_basic(df)
    profile_missing(df)
    profile_numeric(df)
    profile_target(df)
    profile_temporal_structure(df)
    check_leakage(df)
    print_feature_descriptions()
    print("\n" + "=" * 72)
    print("Data investigation complete.")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optiver data investigation -- reproducible checks"
    )
    parser.add_argument(
        "--data", default="data/train.csv",
        help="Path to train CSV (default: data/train.csv)"
    )
    args = parser.parse_args()
    main(args.data)
