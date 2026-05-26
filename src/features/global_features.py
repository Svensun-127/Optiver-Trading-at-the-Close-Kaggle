"""
Global (stock-level aggregation) features for Optiver Trading at the Close.

Computes per-stock daily aggregates, then derives rolling-window statistics,
market-relative strength, lag features, and synthetic index features. All
temporal computations use shift(1) to exclude the current date — strictly
no future leakage.

Feature groups (~32 features):
  1. Stock Historical Statistics (~21) — rolling window (3/5/10 day)
     median/std/ptp/skew/kurt of daily target/wap/imbalance
  2. Market Relative Strength (~4)  — stock return vs market average
  3. Lag Features (~4)              — shift(1), diff, pct_change per stock
  4. Synthetic Index (~3)           — equal-weighted index deviation

Usage:
    from src.features.global_features import GLOBAL_FEATURES, compute_global_features
    X_global = compute_global_features(df)        # df = raw training DataFrame
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Daily aggregation
# ═══════════════════════════════════════════════════════════════════════════


def _daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw tick-level data to (stock_id, date_id) daily level.

    Reduces 5.24M rows to ~96K rows (200 stocks × 481 days) so that
    rolling-window, lag, market, and index computations run efficiently.
    Each aggregate column captures a summary of the stock's behaviour
    during that day's closing auction.

    Returns DataFrame indexed by (stock_id, date_id).
    """
    agg_specs = {
        "target": ["mean", "std"],
        "wap": ["mean", "std"],
        "imbalance_size": ["mean", "std"],
        "reference_price": ["mean"],
        "matched_size": ["mean"],
        "bid_size": ["mean"],
        "ask_size": ["mean"],
        "seconds_in_bucket": ["max"],
    }

    daily = df.groupby(["stock_id", "date_id"]).agg(agg_specs)
    daily.columns = ["_".join(c) for c in daily.columns.values]
    daily["snapshot_count"] = df.groupby(["stock_id", "date_id"]).size()

    return daily


# ═══════════════════════════════════════════════════════════════════════════
# Skewness / kurtosis helpers (no scipy dependency)
# ═══════════════════════════════════════════════════════════════════════════


def _manual_skew(x: np.ndarray) -> float:
    """Sample skewness — bias-corrected, no scipy dependency."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = x.mean()
    sd = np.std(x, ddof=0)
    if sd < 1e-8:
        return 0.0
    z = (x - mean) / sd
    return float((z ** 3).mean() * n * (n - 1) ** 0.5 / (n - 2))


def _manual_kurt(x: np.ndarray) -> float:
    """Sample excess kurtosis — bias-corrected, no scipy dependency."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = x.mean()
    sd = np.std(x, ddof=0)
    if sd < 1e-8:
        return 0.0
    z = (x - mean) / sd
    k = (z ** 4).mean()
    return float(((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)))


# ═══════════════════════════════════════════════════════════════════════════
# Group 1: Stock Historical Rolling Statistics
# ═══════════════════════════════════════════════════════════════════════════


def _compute_historical_rolling(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Per-stock rolling-window statistics of daily aggregates.

    For each stock, computes rolling median, std, ptp (peak-to-peak),
    skewness, and excess kurtosis of daily target mean over 3/5/10 day
    windows, plus median/std of daily WAP and imbalance over 5/10 days.

    **Temporal safety**: uses ``shift(1).rolling(w, min_periods=1)`` so
    the window for date D covers dates [D-w, D-1] — never D itself.
    This is the critical guard against future leakage.

    Financial rationale:
    - Rolling median of past daily target → stock's typical closing-auction
      drift (persistent direction signal).
    - Rolling std/ptp → how volatile the stock's close has been recently
      (a volatile stock is harder to price → wider expected MAE).
    - Rolling skew/kurt → tail-risk profile. A stock with negative skew in
      recent closes may be building sell pressure.
    - WAP rolling stats → recent price trend and stability (trending vs
      mean-reverting behaviour).
    - Imbalance rolling stats → recent order-flow pressure patterns.
    """
    out = pd.DataFrame(index=daily.index)
    daily_sorted = daily.sort_index()
    stocks = daily_sorted.index.get_level_values("stock_id").unique()

    # (column, windows, stats) — concrete spec of what to compute
    specs: list[tuple] = [
        ("target_mean", [3, 5, 10], ["median", "std", "ptp"]),
        ("target_mean", [5, 10], ["skew", "kurt"]),
        ("wap_mean", [5, 10], ["median", "std"]),
        ("imbalance_size_mean", [5, 10], ["median", "std"]),
    ]

    for col, windows, stats in specs:
        for w in windows:
            for stock in stocks:
                idx_mask = daily_sorted.index.get_level_values("stock_id") == stock
                if idx_mask.sum() < 2:          # need at least 2 days to shift
                    continue

                stock_idx = daily_sorted.index[idx_mask]
                s = daily_sorted.loc[stock_idx, col]
                rolled = s.shift(1).rolling(w, min_periods=1)

                for stat in stats:
                    feat = f"{col}_{stat}_{w}d"
                    if stat == "median":
                        out.loc[stock_idx, feat] = rolled.median().values
                    elif stat == "std":
                        out.loc[stock_idx, feat] = rolled.std().values
                    elif stat == "ptp":
                        out.loc[stock_idx, feat] = (
                            rolled.max().values - rolled.min().values
                        )
                    elif stat == "skew":
                        out.loc[stock_idx, feat] = rolled.apply(
                            _manual_skew, raw=False
                        ).values
                    elif stat == "kurt":
                        out.loc[stock_idx, feat] = rolled.apply(
                            _manual_kurt, raw=False
                        ).values

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Group 2: Market Relative Strength
# ═══════════════════════════════════════════════════════════════════════════


def _compute_market_relative_strength(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Stock performance measured relative to the market average.

    For each date, computes the equal-weighted average WAP return across
    all stocks, then measures each stock's deviation from it. Stocks that
    outperform the market during the closing auction tend to show persistent
    relative strength.

    Also computes a rolling z-score of relative strength (5d and 10d) —
    a stock whose relative strength suddenly spikes may be experiencing
    an information event.

    **Temporal safety**: current-date market stats use same-date WAP only
    (no target leakage). Rolling z-scores use shift(1).
    """
    out = pd.DataFrame(index=daily.index)
    daily_sorted = daily.sort_index()

    # 1. Daily WAP return per stock: (wap_today - wap_yesterday) / wap_yesterday
    wap_mean = daily_sorted["wap_mean"]
    wap_lag = wap_mean.groupby("stock_id").shift(1)
    daily_sorted["wap_return"] = (wap_mean - wap_lag) / (wap_lag.abs() + 1e-8)

    # 2. Equal-weighted market average return per date
    market_return = daily_sorted.groupby("date_id")["wap_return"].transform("mean")

    # 3. Relative strength: stock return − market return
    daily_sorted["rel_strength"] = daily_sorted["wap_return"] - market_return
    # clip extreme values (thinly-traded stocks can have huge single-day moves)
    daily_sorted["rel_strength"] = daily_sorted["rel_strength"].clip(-0.5, 0.5)

    # 4. Rolling z-score of relative strength (shift → past only)
    for w in [5, 10]:
        grp = daily_sorted.groupby("stock_id")["rel_strength"]
        rs_mean = grp.transform(
            lambda g: g.shift(1).rolling(w, min_periods=1).mean()
        )
        rs_std = grp.transform(
            lambda g: g.shift(1).rolling(w, min_periods=1).std()
        )
        out[f"rel_strength_z_{w}d"] = (
            (daily_sorted["rel_strength"] - rs_mean) / (rs_std + 1e-8)
        )

    # 5. Raw relative strength (current day — computed from same-day WAP, safe)
    out["rel_strength"] = daily_sorted["rel_strength"]

    # 6. Cumulative relative strength over 5 days
    out["rel_strength_cum5d"] = daily_sorted.groupby("stock_id")["rel_strength"].transform(
        lambda g: g.shift(1).rolling(5, min_periods=1).sum()
    )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Group 3: Lag Features
# ═══════════════════════════════════════════════════════════════════════════


def _compute_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Per-stock lag-1 features: previous day's values and day-over-day changes.

    Simple lag features capture the most recent known state of each stock.
    Because these use ``shift(1)`` within each stock group, they never
    reference the current day's target.

    - **target_mean_lag1**: yesterday's average closing-auction price change.
      If the stock consistently drifts in one direction, this carries signal.
    - **target_mean_diff1**: day-over-day change in the drift. A reversal
      (sign change) often indicates a regime shift.
    - **wap_mean_lag1** / **wap_mean_ret1**: yesterday's price level and
      daily return. Momentum-style signal.
    """
    out = pd.DataFrame(index=daily.index)
    daily_sorted = daily.sort_index()

    grp = daily_sorted.groupby("stock_id")

    # Lag-1 of daily target mean
    out["target_mean_lag1"] = grp["target_mean"].shift(1)
    # Day-over-day difference
    out["target_mean_diff1"] = daily_sorted["target_mean"] - out["target_mean_lag1"]

    # Lag-1 of daily WAP
    out["wap_mean_lag1"] = grp["wap_mean"].shift(1)
    # Daily WAP return
    out["wap_mean_ret1"] = (
        daily_sorted["wap_mean"] - out["wap_mean_lag1"]
    ) / (out["wap_mean_lag1"].abs() + 1e-8)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Group 4: Synthetic Index
# ═══════════════════════════════════════════════════════════════════════════


def _compute_synthetic_index(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weighted market index and per-stock deviations.

    Constructs a simple market barometer from all stocks' WAP, then measures
    how far each stock is from it. Large deviations may indicate stock-specific
    news or liquidity events that affect closing-auction pricing.

    - **market_index**: cross-sectional mean of wap_mean per date.
      A rising index means most stocks are being marked higher.
    - **stock_dev_index**: absolute deviation from the index (dollar terms).
    - **stock_dev_index_pct**: relative deviation (percentage terms). Makes
      the signal comparable across stocks with different nominal prices.

    Uses same-date WAP only — no target leakage.
    """
    out = pd.DataFrame(index=daily.index)
    daily_sorted = daily.sort_index()

    # Equal-weighted market index: average WAP across all stocks per date
    out["market_index"] = daily_sorted.groupby("date_id")["wap_mean"].transform("mean")

    wap = daily_sorted["wap_mean"]
    idx = out["market_index"]

    out["stock_dev_index"] = wap - idx
    out["stock_dev_index_pct"] = (wap - idx) / (idx.abs() + 1e-8)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def compute_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ~30 stock-level aggregation features.

    Workflow:
    1. Aggregate raw tick data to (stock_id, date_id) daily summaries.
    2. Compute four groups of features on the daily DataFrame.
    3. Merge back to the original row-level DataFrame via stock_id + date_id.

    The returned DataFrame has the same index as *df* so it can be aligned
    with base/micro features by row index, exactly like
    ``compute_micro_features``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training data (must contain the standard Optiver columns).

    Returns
    -------
    pd.DataFrame
        Global-feature columns only, same row count and index as *df*.
    """
    # 1. Aggregate to daily level (~96K rows)
    daily = _daily_aggregate(df)

    # 2. Compute feature groups on daily level
    hist = _compute_historical_rolling(daily)
    mkt = _compute_market_relative_strength(daily)
    lag = _compute_lag_features(daily)
    idx = _compute_synthetic_index(daily)

    # Combine all global features into one daily-level DataFrame
    daily_feat = pd.concat([hist, mkt, lag, idx], axis=1)

    # 3. Merge back to full row-level DataFrame
    out = df[["stock_id", "date_id"]].merge(
        daily_feat, on=["stock_id", "date_id"], how="left"
    )

    # Drop merge keys — only feature columns remain
    out.drop(columns=["stock_id", "date_id"], inplace=True)

    # Align index with original df
    out.index = df.index

    # Replace infinities that may arise from division edge cases
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4. Auto-downcast to save memory (global features are float64-heavy)
    for col in out.columns:
        col_type = out[col].dtype
        if col_type == np.float64:
            out[col] = out[col].astype(np.float32)
        elif col_type == np.int64:
            c_min, c_max = out[col].min(), out[col].max()
            if c_min >= -128 and c_max <= 127:
                out[col] = out[col].astype(np.int8)
            elif c_min >= -32768 and c_max <= 32767:
                out[col] = out[col].astype(np.int16)
            else:
                out[col] = out[col].astype(np.int32)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Feature registry — single source of truth for what this module produces.
# ═══════════════════════════════════════════════════════════════════════════

GLOBAL_FEATURES: list[str] = [
    # ── Group 1: Stock Historical Rolling (~21) ─────────────────────────
    # target_mean rolling (3d)
    "target_mean_median_3d",
    "target_mean_std_3d",
    "target_mean_ptp_3d",
    # target_mean rolling (5d)
    "target_mean_median_5d",
    "target_mean_std_5d",
    "target_mean_ptp_5d",
    "target_mean_skew_5d",
    "target_mean_kurt_5d",
    # target_mean rolling (10d)
    "target_mean_median_10d",
    "target_mean_std_10d",
    "target_mean_ptp_10d",
    "target_mean_skew_10d",
    "target_mean_kurt_10d",
    # wap_mean rolling (5d, 10d)
    "wap_mean_median_5d",
    "wap_mean_std_5d",
    "wap_mean_median_10d",
    "wap_mean_std_10d",
    # imbalance_size_mean rolling (5d, 10d)
    "imbalance_size_mean_median_5d",
    "imbalance_size_mean_std_5d",
    "imbalance_size_mean_median_10d",
    "imbalance_size_mean_std_10d",
    # ── Group 2: Market Relative Strength (4) ───────────────────────────
    "rel_strength",
    "rel_strength_z_5d",
    "rel_strength_z_10d",
    "rel_strength_cum5d",
    # ── Group 3: Lag Features (4) ───────────────────────────────────────
    "target_mean_lag1",
    "target_mean_diff1",
    "wap_mean_lag1",
    "wap_mean_ret1",
    # ── Group 4: Synthetic Index (3) ────────────────────────────────────
    "market_index",
    "stock_dev_index",
    "stock_dev_index_pct",
]

# Total: 32 features
assert len(GLOBAL_FEATURES) == 32, f"Expected 32 global features, got {len(GLOBAL_FEATURES)}"
