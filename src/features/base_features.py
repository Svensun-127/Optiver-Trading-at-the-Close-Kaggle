"""
Baseline feature engineering for Optiver Trading at the Close.

Produces the feature matrix used by the baseline LightGBM model. Each function
is documented with its financial rationale so future iterations can understand
the signal each feature is meant to capture.

Design decisions for baseline (per Step 1 findings):
- 4 price columns (ref/bid/ask/wap) are kept — tree models handle collinearity
  better than linear models, and each may capture subtle microstructural signals.
- far_price / near_price are only available at seconds_in_bucket >= 300 (~45%
  of rows). For the baseline we use the community standard approach: fill with a
  sentinel value (-1.0) and add binary availability indicators.
- Derived features are intentionally simple — avoid over-optimising before the
  first CV run establishes a MAE baseline.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# Columns with multi-column gaps in the 220 corrupted rows (Step 1 finding).
_CORRUPTED_COLS: list[str] = [
    "reference_price", "imbalance_size", "matched_size",
    "ask_price", "bid_price", "wap",
]

# Columns that must not be missing for a row to be used for training.
_CRITICAL_COLS: list[str] = [
    "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "bid_price", "bid_size",
    "ask_price", "ask_size", "wap",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows unsuitable for supervised training.

    Drops two categories of rows identified in Step 1:
    1. Rows where *target* is NaN (~88 rows, ~0.002%) — these cannot
       contribute to supervised learning because the label is missing.
    2. The 220 rows that have simultaneous multi-column gaps across the
       six core price/size columns. They represent a single corrupted
       data slice and are too few to justify imputation complexity.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training data (must contain ``target`` and the columns listed
        in ``_CORRUPTED_COLS``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with corrupted rows removed.
    """
    n_before = len(df)

    # 1. Drop rows where target is missing — no label to learn from.
    df = df.dropna(subset=["target"]).copy()

    # 2. Drop the 220-row multi-column corruption slice.
    # All six _CORRUPTED_COLS are NaN simultaneously in these rows.
    corrupted_mask = df[_CORRUPTED_COLS[0]].isna()
    for col in _CORRUPTED_COLS[1:]:
        corrupted_mask = corrupted_mask & df[col].isna()
    df = df[~corrupted_mask]

    n_after = len(df)
    if n_before != n_after:
        print(f"  Data cleaning: dropped {n_before - n_after} rows "
              f"({100 * (n_before - n_after) / n_before:.3f}%)")

    return df


def winsorize_target(y: pd.Series, limit: float = 30.0) -> pd.Series:
    """
    Clip target values to ``[-limit, +limit]``.

    Winsorization caps extreme tail values without removing the row entirely.
    The raw target has min≈-400 and max≈+446 (Step 1 finding), which are
    extreme for a 60-second forward return. These outliers inflate MAE and
    can cause the model to optimise for rare events rather than typical
    price moves.

    The ±30 threshold corresponds to roughly ±3σ (σ=9.45), retaining ~99.7%
    of the distribution while removing the most extreme tails.

    Parameters
    ----------
    y : pd.Series
        Raw target values.
    limit : float
        Symmetric clipping bound (default 30.0).

    Returns
    -------
    pd.Series
        Winsorized target.
    """
    original = y.copy()
    y = y.clip(lower=-limit, upper=limit)
    n_clipped = (original != y).sum()
    if n_clipped > 0:
        print(f"  Target winsorization (±{limit}): clipped "
              f"{n_clipped:,} / {len(y):,} rows ({100 * n_clipped / len(y):.2f}%)")
    return y


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build price-derived features from order-book levels.

    Creates four features that capture the current state of the limit
    order book around the auction price:

    - **bid_ask_spread**: The raw spread (ask_p − bid_p). A wider spread
      indicates lower liquidity and higher implicit transaction costs.
      Large spreads often precede price dislocations.
    - **spread_pct**: Spread normalised by WAP. This allows cross-stock
      comparison — a $0.01 spread means very different things for a
      $10 stock vs a $100 stock.
    - **wap_ref_diff**: WAP deviation from the reference price. Captures
      intra-auction price drift. Positive values mean the current
      auction-clearing price is above the pre-auction reference — bullish
      pressure during the closing auction.
    - **price_momentum**: The simple difference between WAP and reference
      as a ratio, i.e. (wap / reference_price − 1). More interpretable
      than the raw diff for stocks with different nominal prices.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ask_price``, ``bid_price``, ``wap``, and
        ``reference_price``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with four new price-feature columns added.
    """
    eps = 1e-8  # avoid division-by-zero

    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["bid_ask_spread"] / (df["wap"].abs() + eps)
    df["wap_ref_diff"] = df["wap"] - df["reference_price"]
    df["price_momentum"] = df["wap"] / (df["reference_price"].abs() + eps) - 1.0

    return df


def create_size_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build order-size-based features from the order-book depth.

    Creates three features that describe the balance of supply and demand
    in the limit order book:

    - **imbalance_ratio**: ``imbalance_size / matched_size``. Measures
      how much net order imbalance remains relative to what has already
      been matched. A high positive value means buy interest far exceeds
      the matched volume — strong upward pressure.
    - **size_imbalance**: ``bid_size / (bid_size + ask_size)``. The
      proportion of top-of-book depth on the bid side. Values > 0.5 mean
      buyers place more limit orders than sellers at the best prices —
      bullish signal.
    - **depth_total**: ``bid_size + ask_size``. Total displayed depth at
      the best bid/ask levels. Higher depth means a thicker order book
      which generally dampens price impact.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``imbalance_size``, ``matched_size``, ``bid_size``,
        and ``ask_size``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with three new size-feature columns added.
    """
    eps = 1e-8

    df["imbalance_ratio"] = df["imbalance_size"] / (df["matched_size"].abs() + eps)
    df["imbalance_ratio"] = df["imbalance_ratio"].clip(-100, 100)

    df["size_imbalance"] = df["bid_size"] / (df["bid_size"] + df["ask_size"] + eps)
    df["depth_total"] = df["bid_size"] + df["ask_size"]

    return df


def handle_missing_far_near(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill ``far_price`` and ``near_price`` with a sentinel value and add
    binary availability indicators.

    These two columns are only populated for ``seconds_in_bucket >= 300``
    (the second half of the closing auction, ~45% of rows). For tree-based
    models, a sentinel value (-1.0) works well because the model can learn
    a split rule like "far_price < 0 → use the missing-data branch."

    The binary flags ``far_price_avail`` and ``near_price_avail`` give the
    model an explicit signal about data completeness, which can capture
    regime changes between the auction's early (info-poor) and late
    (info-rich) phases.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``far_price``, ``near_price``, and ``seconds_in_bucket``.

    Returns
    -------
    pd.DataFrame
        DataFrame with sentinel-filled values and two new indicator columns.
    """
    df["far_price_avail"] = df["far_price"].notna().astype(np.int8)
    df["near_price_avail"] = df["near_price"].notna().astype(np.int8)

    df["far_price"] = df["far_price"].fillna(-1.0)
    df["near_price"] = df["near_price"].fillna(-1.0)

    return df


def create_far_near_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features derived from far-side and near-side order-book prices.

    - **far_near_spread**: ``far_price − near_price``. Captures the slope
      of the order book beyond the best bid/ask. A wide far-near spread
      indicates that liquidity is thin beyond the best levels —
      large orders would face significant slippage.
    - **far_near_mean**: ``(far_price + near_price) / 2``. A mid-range
      price estimate from the broader order book. When available, it may
      be a more stable reference than WAP for estimating "fair value".

    Both features are only meaningful when the underlying columns are
    available (seconds_in_bucket >= 300). When either is the sentinel
    value (−1), the derived feature is forced to 0 since the signal is
    structurally absent.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``far_price`` and ``near_price`` after sentinel filling.

    Returns
    -------
    pd.DataFrame
        DataFrame with two additional far/near feature columns.
    """
    df["far_near_spread"] = df["far_price"] - df["near_price"]
    df["far_near_mean"] = (df["far_price"] + df["near_price"]) / 2.0

    # Zero out when both are unavailable (sentinel = -1)
    unavailable = (df["far_price"] == -1.0) & (df["near_price"] == -1.0)
    df.loc[unavailable, "far_near_spread"] = 0.0
    df.loc[unavailable, "far_near_mean"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Feature name registry — kept in one place so run.py can write
# feature_list.txt without duplicating the list.
# ---------------------------------------------------------------------------

# Raw numerical features always present in the data.
NUMERICAL_FEATURES: list[str] = [
    "seconds_in_bucket",
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
]

# Conditional raw features (only seconds_in_bucket >= 300).
CONDITIONAL_FEATURES: list[str] = [
    "far_price",
    "near_price",
]

# Derived features created by this module.
DERIVED_FEATURES: list[str] = [
    "bid_ask_spread",
    "spread_pct",
    "wap_ref_diff",
    "price_momentum",
    "imbalance_ratio",
    "size_imbalance",
    "depth_total",
    "far_price_avail",
    "near_price_avail",
    "far_near_spread",
    "far_near_mean",
]

# Categorical features passed to LightGBM's ``categorical_feature``.
CATEGORICAL_FEATURES: list[str] = ["stock_id"]

# All feature columns the model sees (stock_id + numerical + derived).
ALL_FEATURES: list[str] = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + CONDITIONAL_FEATURES
    + DERIVED_FEATURES
)

# Metadata columns needed for CV splitting but NOT fed to the model.
METADATA_COLS: list[str] = ["date_id"]


def build_features(
    df: pd.DataFrame,
    X_micro: Optional[pd.DataFrame] = None,
    X_global: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full feature-engineering pipeline: clean → winsorize → engineer → select.

    Orchestrates the following steps in order:
    1. Drop corrupted rows (missing target + multi-column gap slice).
    2. Winsorize target at ±30 to bound extreme tail events.
    3. Engineer derived features (price spreads, size ratios, far/near signals).
    4. Fill far_price / near_price missing with sentinel + availability flags.
    5. Optionally merge pre-computed micro and global features aligned by row index.
    6. Return ``(X_all, y_winsorized)`` ready for CV splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training DataFrame as loaded from ``data/train.csv``.
    X_micro : pd.DataFrame, optional
        Pre-computed micro-feature DataFrame (from ``compute_micro_features``
        on the **raw** df). Aligned by original row index so only rows that
        survive cleaning are kept.
    X_global : pd.DataFrame, optional
        Pre-computed global-feature DataFrame (from ``compute_global_features``
        on the **raw** df). Aligned by original row index.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix including base + optional micro + global features,
        indexed from 0..n-1 after reset.
    y : pd.Series
        Winsorized target, same index as ``X``.
    """
    print("Building features ...")

    # 1. Clean
    df = clean_data(df)

    # 2. Winsorize target
    y = winsorize_target(df["target"], limit=30.0)

    # 3. Engineer features
    df = create_price_features(df)
    df = create_size_features(df)
    df = handle_missing_far_near(df)
    df = create_far_near_features(df)

    # 4. Select columns the model will use + metadata for CV splitting
    output_cols = ALL_FEATURES + METADATA_COLS
    X = df[output_cols].copy()

    # 5. Merge micro and global features (aligned by original row index)
    feature_count = len(ALL_FEATURES)
    if X_micro is not None:
        from src.features.micro_features import MICRO_FEATURES  # noqa: F811

        X_micro_aligned = X_micro.reindex(X.index)
        X = pd.concat([X, X_micro_aligned], axis=1)
        feature_count += len(MICRO_FEATURES)

    if X_global is not None:
        from src.features.global_features import GLOBAL_FEATURES  # noqa: F811

        X_global_aligned = X_global.reindex(X.index)
        X = pd.concat([X, X_global_aligned], axis=1)
        feature_count += len(GLOBAL_FEATURES)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"  Features built: {feature_count} model features, "
          f"{len(X):,} rows")
    return X, y
