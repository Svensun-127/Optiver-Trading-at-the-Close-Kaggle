"""
Microstructure features for Optiver Trading at the Close.

Numba JIT-accelerated features capturing order-book dynamics, triplet imbalance
patterns, and price-formation microstructure. Follows the championship solution's
three-phase feature engineering approach.

Feature groups (total: ~30):
  1. Triplet Imbalance (8)  — Numba JIT: (max−mid)/(mid−min) asymmetry
  2. Price/Depth (7)        — micro price, urgency, spreads
  3. Aggregate Stats (8)    — row-wise mean/std/skew/kurt of price & size cols
  4. Far/Near Enhanced (5)  — NaN-preserving far/near features
  5. Volume/Flow (5)        — order-flow intensity and imbalance ratios

Usage:
    from src.features.micro_features import MICRO_FEATURES, compute_micro_features
    X_micro = compute_micro_features(df)        # df = raw training DataFrame
    X_all  = pd.concat([X_base, X_micro], axis=1)
"""

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit


# ═══════════════════════════════════════════════════════════════════════════
# Numba JIT-accelerated kernels
# ═══════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _triplet_imbalance(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    (max − mid) / (mid − min + eps) for each triplet (a_i, b_i, c_i).

    Measures asymmetry of three order-book signals. A large positive value
    means the dominant signal far exceeds the median — directional pressure.
    Near-zero means balanced, symmetric order flow.

    Numba JIT compiles this to native code; ~100× faster than a Python loop
    over 5.2M rows.
    """
    n = len(a)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x, y, z = a[i], b[i], c[i]

        # Sort 3 elements → min, mid, max (branch-free would be slower here)
        if x <= y:
            if y <= z:
                mn, md, mx = x, y, z
            elif x <= z:
                mn, md, mx = x, z, y
            else:
                mn, md, mx = z, x, y
        else:
            if x <= z:
                mn, md, mx = y, x, z
            elif y <= z:
                mn, md, mx = y, z, x
            else:
                mn, md, mx = z, y, x

        denom = md - mn + eps
        if denom < eps:
            denom = eps
        out[i] = (mx - md) / denom
    return out


@njit(cache=True)
def _row_stats_4col(values: np.ndarray) -> tuple:
    """
    Row-wise mean, std, skewness, excess-kurtosis for an (n, 4) array.

    The input is a stack of 4 column vectors, e.g.:
        [bid_price, ask_price, wap, reference_price]
    or
        [bid_size, ask_size, matched_size, abs(imbalance_size)]

    Returns (mean, std, skew, kurt) each shape (n,).
    """
    n_rows = values.shape[0]
    mean = np.zeros(n_rows, dtype=np.float64)
    std = np.zeros(n_rows, dtype=np.float64)
    skew = np.zeros(n_rows, dtype=np.float64)
    kurt = np.zeros(n_rows, dtype=np.float64)

    for i in range(n_rows):
        # Mean over 4 columns
        s = values[i, 0] + values[i, 1] + values[i, 2] + values[i, 3]
        m = s * 0.25
        mean[i] = m

        # Std
        d0 = values[i, 0] - m
        d1 = values[i, 1] - m
        d2 = values[i, 2] - m
        d3 = values[i, 3] - m
        var = (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3) * 0.25
        sd = np.sqrt(var)
        std[i] = sd

        if sd > 1e-8:
            inv_sd = 1.0 / sd
            z0, z1 = d0 * inv_sd, d1 * inv_sd
            z2, z3 = d2 * inv_sd, d3 * inv_sd
            skew[i] = (z0 ** 3 + z1 ** 3 + z2 ** 3 + z3 ** 3) * 0.25
            kurt[i] = (z0 ** 4 + z1 ** 4 + z2 ** 4 + z3 ** 4) * 0.25 - 3.0
        else:
            skew[i] = 0.0
            kurt[i] = 0.0

    return mean, std, skew, kurt


@njit(cache=True)
def _log_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """log(a / b) with safe clamping. Numba-accelerated over 5.2M rows."""
    n = len(a)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        num = a[i]
        denom = b[i]
        if num <= 0.0:
            num = eps
        if denom <= 0.0:
            denom = eps
        out[i] = np.log(num / denom)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════════════


def _safe_div(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-8,
    clip_val: Optional[float] = None,
) -> np.ndarray:
    """Element-wise a / (b + eps) with optional symmetric clipping."""
    result = a / (b + eps)
    if clip_val is not None:
        result = np.clip(result, -clip_val, clip_val)
    return result


def compute_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ~30 microstructure features for every row in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training data. Must contain the standard 17 Optiver columns.
        ``far_price`` / ``near_price`` may be NaN (when seconds_in_bucket
        < 300).  Row order is preserved in the output index.

    Returns
    -------
    pd.DataFrame
        Micro-feature columns, same row count and index as *df*.
    """
    out = pd.DataFrame(index=df.index)
    eps = 1e-8

    # Short aliases (raw numpy arrays — Numba-friendly, no pandas overhead)
    bid_p = df["bid_price"].values.astype(np.float64)
    ask_p = df["ask_price"].values.astype(np.float64)
    bid_s = df["bid_size"].values.astype(np.float64)
    ask_s = df["ask_size"].values.astype(np.float64)
    wap_v = df["wap"].values.astype(np.float64)
    ref_p = df["reference_price"].values.astype(np.float64)
    imb_s = df["imbalance_size"].values.astype(np.float64)
    imb_f = df["imbalance_buy_sell_flag"].values.astype(np.float64)
    mat_s = df["matched_size"].values.astype(np.float64)
    far_p = df["far_price"].values.astype(np.float64)
    near_p = df["near_price"].values.astype(np.float64)
    sec_v = df["seconds_in_bucket"].values.astype(np.float64)

    abs_imb = np.abs(imb_s)
    depth = bid_s + ask_s

    # ── Group 1: Triplet Imbalance (8 features) ─────────────────────────
    # Each captures asymmetry between three order-book dimensions.
    # Formula: (max - mid) / (mid - min + eps)
    # Large value → one dimension dominates → directional signal.

    out["tri_flow"] = _triplet_imbalance(         # order-flow asymmetry
        bid_s, ask_s, abs_imb, eps)
    out["tri_price"] = _triplet_imbalance(        # price-formation asymmetry
        bid_p, ask_p, wap_v, eps)
    out["tri_match"] = _triplet_imbalance(        # matching pressure
        bid_s, ask_s, mat_s, eps)
    out["tri_ref_bid"] = _triplet_imbalance(      # bid-side ref deviation
        bid_p, ref_p, wap_v, eps)
    out["tri_exec"] = _triplet_imbalance(         # execution flow
        bid_s, mat_s, abs_imb, eps)
    out["tri_ref_ask"] = _triplet_imbalance(      # ask-side ref deviation
        ask_p, ref_p, wap_v, eps)
    out["tri_depth"] = _triplet_imbalance(         # depth balance
        bid_s, ask_s, depth, eps)

    # Term-structure triplet — only valid when far & near are available.
    # When NaN, LightGBM's native NaN handling routes it to the missing
    # branch, which is the correct behaviour (structurally absent signal).
    out["tri_term"] = np.nan
    valid_far = ~np.isnan(far_p)
    valid_near = ~np.isnan(near_p)
    valid_both = valid_far & valid_near
    if valid_both.any():
        tri_vals = _triplet_imbalance(
            near_p[valid_both], wap_v[valid_both], far_p[valid_both], eps)
        out.loc[valid_both, "tri_term"] = tri_vals

    # ── Group 2: Price / Depth (7 features) ────────────────────────────

    # Micro-price: size-weighted bid-ask midpoint.
    # Closer to the "true" price than a simple mid when the book is lopsided.
    micro_p = (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s + eps)
    out["micro_price"] = micro_p

    # Mid-price: simple average of best bid and ask.
    out["mid_price"] = (bid_p + ask_p) * 0.5

    # Relative spread normalised by mid-price (cross-stock comparable).
    out["price_spread"] = _safe_div(ask_p - bid_p, out["mid_price"].values, eps)

    # Market urgency: wide spread + thin book = urgent execution pressure.
    # sqrt(depth) scales the spread by book thickness — a $0.01 spread in a
    # thin book (100 shares) is far more urgent than in a thick one (10k).
    out["market_urgency"] = _safe_div(
        (ask_p - bid_p) * np.sqrt(depth + eps), wap_v, eps, clip_val=100.0)

    # WAP deviation from micro-price (positive = WAP above fair value).
    out["wap_micro_diff"] = _safe_div(wap_v - micro_p, wap_v, eps)

    # Book slope: price spread per unit depth (slippage proxy).
    out["book_slope"] = _safe_div(ask_p - bid_p, depth, eps)

    # Signed imbalance: preserves buy/sell direction.
    # imbalance_buy_sell_flag: 1 = buy pressure, 0 (or -1) = sell pressure.
    direction = np.where(imb_f == 1, 1.0, -1.0)
    out["signed_imb"] = imb_s * direction

    # Net order intensity: directional imbalance relative to total depth.
    out["order_intensity"] = _safe_div(out["signed_imb"].values, depth, eps)

    # ── Group 3: Aggregate Statistics (8 features) ──────────────────────
    # Row-wise moments across the 4 core price/size dimensions.
    # Captures the shape of each order-book "profile" — a wide dispersion
    # among price levels or sizes often precedes large price moves.

    price_stack = np.column_stack([bid_p, ask_p, wap_v, ref_p])
    p_mean, p_std, p_skew, p_kurt = _row_stats_4col(price_stack)
    out["price_mean"] = p_mean
    out["price_std"] = p_std
    out["price_skew"] = p_skew
    out["price_kurt"] = p_kurt

    size_stack = np.column_stack([bid_s, ask_s, mat_s, abs_imb])
    s_mean, s_std, s_skew, s_kurt = _row_stats_4col(size_stack)
    out["size_mean"] = s_mean
    out["size_std"] = s_std
    out["size_skew"] = s_skew
    out["size_kurt"] = s_kurt

    # ── Group 4: Far / Near Enhanced (5 features) ───────────────────────
    # NaN-preserving variants. Unlike base_features (which fills -1 sentinel),
    # these keep NaN so LightGBM can use its native missing-value handling.
    # This gives the model two different "views" of the same sparse columns.

    out["far_price_nan"] = far_p          # raw, NaN when unavailable
    out["near_price_nan"] = near_p        # raw, NaN when unavailable

    # Far/near spread (NaN when either is missing — structurally correct).
    out["far_near_spread_nan"] = far_p - near_p

    # Far/near deviation from mid-price (normalised).
    mid_v = out["mid_price"].values
    out["far_to_mid"] = _safe_div(far_p - mid_v, mid_v, eps)
    out["near_to_mid"] = _safe_div(near_p - mid_v, mid_v, eps)

    # ── Group 5: Volume / Flow dynamics (5 features) ───────────────────

    # Volume intensity: how much of the book has already been matched.
    out["volume_intensity"] = _safe_div(mat_s, depth, eps)

    # Imbalance flow: net imbalance relative to matched volume.
    out["imb_flow"] = _safe_div(abs_imb, mat_s, eps, clip_val=100.0)

    # Log quote imbalance: log(bid_size / ask_size). Symmetric around zero.
    # A positive value = bid side is thicker (buyers dominate the book).
    out["log_quote_imb"] = _log_ratio(bid_s, ask_s, eps)

    # Bid/ask size ratio (clipped for stability).
    out["bid_ask_size_ratio"] = _safe_div(bid_s, ask_s, eps, clip_val=100.0)

    # Matched-to-depth ratio (execution completion proxy).
    out["matched_depth_ratio"] = _safe_div(mat_s, depth, eps)

    # ── Cleanup ─────────────────────────────────────────────────────────
    # Replace infinities (from division by near-zero) with NaN so LightGBM
    # handles them uniformly.
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Feature registry — single source of truth for what this module produces.
# ═══════════════════════════════════════════════════════════════════════════

MICRO_FEATURES: list[str] = [
    # Triplet Imbalance (8)
    "tri_flow",
    "tri_price",
    "tri_match",
    "tri_ref_bid",
    "tri_exec",
    "tri_ref_ask",
    "tri_depth",
    "tri_term",
    # Price / Depth (8)
    "micro_price",
    "mid_price",
    "price_spread",
    "market_urgency",
    "wap_micro_diff",
    "book_slope",
    "signed_imb",
    "order_intensity",
    # Aggregate Statistics (8)
    "price_mean",
    "price_std",
    "price_skew",
    "price_kurt",
    "size_mean",
    "size_std",
    "size_skew",
    "size_kurt",
    # Far / Near Enhanced (5)
    "far_price_nan",
    "near_price_nan",
    "far_near_spread_nan",
    "far_to_mid",
    "near_to_mid",
    # Volume / Flow (5)
    "volume_intensity",
    "imb_flow",
    "log_quote_imb",
    "bid_ask_size_ratio",
    "matched_depth_ratio",
]

# Total: 34 features
assert len(MICRO_FEATURES) == 34, f"Expected 34 micro features, got {len(MICRO_FEATURES)}"
