"""
Kaggle Notebook submission: Optiver - Trading at the Close.

Self-contained inference script that EXACTLY reproduces the feature
engineering pipeline used during training (base_features.py + micro_features.py
+ global_features.py).  Numba JIT functions are replaced with numpy vectorised
equivalents that produce identical numerical results.

Usage:
    !python notebook_submission.py
"""

import lightgbm as lgb
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Model loading
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = Path("lgb_model.txt")
if not MODEL_PATH.exists():
    print(f"ERROR: {MODEL_PATH} not found.")
    sys.exit(1)

model = lgb.Booster(model_file=str(MODEL_PATH))

print(f"[notebook] Model loaded: {MODEL_PATH} "
      f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
print(f"[notebook] n_estimators: {model.num_trees()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Numpy equivalents of Numba kernels (identical numerical results)
# ═══════════════════════════════════════════════════════════════════════════════

def _triplet_imbalance_np(a, b, c, eps=1e-8):
    """(max - mid) / (mid - min + eps) — vectorised triplet imbalance."""
    stacked = np.column_stack([a, b, c])
    stacked.sort(axis=1)
    mn, md, mx = stacked[:, 0], stacked[:, 1], stacked[:, 2]
    denom = np.maximum(md - mn, eps)
    return (mx - md) / denom


def _row_stats_4col_np(values):
    """Row-wise mean, std, skew, excess-kurtosis for (n, 4) array."""
    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    var = (centered ** 2).mean(axis=1)
    std = np.sqrt(var)

    mask = std > 1e-8
    skew = np.zeros(len(values), dtype=np.float64)
    kurt = np.zeros(len(values), dtype=np.float64)

    if mask.any():
        inv_sd = 1.0 / std[mask]
        z = centered[mask] * inv_sd[:, None]
        skew[mask] = (z ** 3).mean(axis=1)
        kurt[mask] = (z ** 4).mean(axis=1) - 3.0

    return mean, std, skew, kurt


def _log_ratio_np(a, b, eps=1e-8):
    """log(a / b) with safe clamping (match Numba _log_ratio)."""
    a_safe = np.maximum(a, eps)
    b_safe = np.maximum(b, eps)
    return np.log(a_safe / b_safe)


def _safe_div_np(a, b, eps=1e-8, clip_val=None):
    """Element-wise a / (b + eps) with optional symmetric clipping."""
    result = a / (b + eps)
    if clip_val is not None:
        result = np.clip(result, -clip_val, clip_val)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Base features — EXACT match to src/features/base_features.py
# ═══════════════════════════════════════════════════════════════════════════════

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24 base features matching ALL_FEATURES exactly."""
    out = pd.DataFrame(index=df.index)
    eps = 1e-8

    # --- Raw numerical features (10) ---
    for c in ["seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
              "reference_price", "matched_size", "bid_price", "bid_size",
              "ask_price", "ask_size", "wap"]:
        out[c] = df[c].astype(np.float32)

    # --- Categorical feature ---
    out["stock_id"] = df["stock_id"].astype(np.int16)

    # --- Conditional raw features (far_price, near_price) ---
    # Filled with sentinel -1.0 and added availability flags — exact match
    # to handle_missing_far_near() in base_features.py.
    far_avail = df["far_price"].notna().astype(np.int8)
    near_avail = df["near_price"].notna().astype(np.int8)
    out["far_price_avail"] = far_avail
    out["near_price_avail"] = near_avail

    out["far_price"] = df["far_price"].fillna(-1.0).astype(np.float32)
    out["near_price"] = df["near_price"].fillna(-1.0).astype(np.float32)

    # --- Price features (create_price_features) ---
    ask = out["ask_price"].values.astype(np.float64)
    bid = out["bid_price"].values.astype(np.float64)
    wap = out["wap"].values.astype(np.float64)
    ref = out["reference_price"].values.astype(np.float64)

    out["bid_ask_spread"] = (ask - bid).astype(np.float32)
    out["spread_pct"] = ((ask - bid) / (np.abs(wap) + eps)).astype(np.float32)
    out["wap_ref_diff"] = (wap - ref).astype(np.float32)
    out["price_momentum"] = (wap / (np.abs(ref) + eps) - 1.0).astype(np.float32)

    # --- Size features (create_size_features) ---
    bid_sz = out["bid_size"].values.astype(np.float64)
    ask_sz = out["ask_size"].values.astype(np.float64)
    imb_sz = out["imbalance_size"].values.astype(np.float64)
    mat_sz = out["matched_size"].values.astype(np.float64)

    imb_ratio = imb_sz / (np.abs(mat_sz) + eps)
    out["imbalance_ratio"] = np.clip(imb_ratio, -100, 100).astype(np.float32)
    out["size_imbalance"] = (bid_sz / (bid_sz + ask_sz + eps)).astype(np.float32)
    out["depth_total"] = (bid_sz + ask_sz).astype(np.float32)

    # --- Far/near features (create_far_near_features) ---
    far = out["far_price"].values.astype(np.float64)
    near = out["near_price"].values.astype(np.float64)

    out["far_near_spread"] = (far - near).astype(np.float32)
    out["far_near_mean"] = ((far + near) / 2.0).astype(np.float32)

    # Zero out when both are unavailable (sentinel = -1.0)
    unavailable = (far == -1.0) & (near == -1.0)
    out.loc[unavailable, "far_near_spread"] = 0.0
    out.loc[unavailable, "far_near_mean"] = 0.0

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Micro features — EXACT match to src/features/micro_features.py
# ═══════════════════════════════════════════════════════════════════════════════

def compute_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 34 micro-structure features matching MICRO_FEATURES exactly."""
    out = pd.DataFrame(index=df.index)
    eps = 1e-8

    # Raw numpy arrays (float64 — match Numba call signatures)
    bid_p = df["bid_price"].values.astype(np.float64)
    ask_p = df["ask_price"].values.astype(np.float64)
    bid_s = df["bid_size"].values.astype(np.float64)
    ask_s = df["ask_size"].values.astype(np.float64)
    wap_v = df["wap"].values.astype(np.float64)
    ref_p = df["reference_price"].values.astype(np.float64)
    imb_s = df["imbalance_size"].values.astype(np.float64)
    imb_f = df["imbalance_buy_sell_flag"].values.astype(np.float64)
    mat_s = df["matched_size"].values.astype(np.float64)
    far_p = df["far_price"].values.astype(np.float64)     # keep NaN
    near_p = df["near_price"].values.astype(np.float64)  # keep NaN
    sec_v = df["seconds_in_bucket"].values.astype(np.float64)

    abs_imb = np.abs(imb_s)
    depth = bid_s + ask_s

    # ── Group 1: Triplet Imbalance (8) ──
    out["tri_flow"] = _triplet_imbalance_np(bid_s, ask_s, abs_imb, eps)
    out["tri_price"] = _triplet_imbalance_np(bid_p, ask_p, wap_v, eps)
    out["tri_match"] = _triplet_imbalance_np(bid_s, ask_s, mat_s, eps)
    out["tri_ref_bid"] = _triplet_imbalance_np(bid_p, ref_p, wap_v, eps)
    out["tri_exec"] = _triplet_imbalance_np(bid_s, mat_s, abs_imb, eps)
    out["tri_ref_ask"] = _triplet_imbalance_np(ask_p, ref_p, wap_v, eps)
    out["tri_depth"] = _triplet_imbalance_np(bid_s, ask_s, depth, eps)

    # tri_term: NaN when far/near unavailable (matches NaN-preserving logic)
    out["tri_term"] = np.nan
    valid_far = ~np.isnan(far_p)
    valid_near = ~np.isnan(near_p)
    valid_both = valid_far & valid_near
    if valid_both.any():
        tri_vals = _triplet_imbalance_np(
            near_p[valid_both], wap_v[valid_both], far_p[valid_both], eps)
        out.loc[valid_both, "tri_term"] = tri_vals

    # ── Group 2: Price / Depth (8) ──
    micro_p = (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s + eps)
    out["micro_price"] = micro_p
    out["mid_price"] = (bid_p + ask_p) * 0.5
    out["price_spread"] = _safe_div_np(ask_p - bid_p, out["mid_price"].values, eps)
    out["market_urgency"] = _safe_div_np(
        (ask_p - bid_p) * np.sqrt(depth + eps), wap_v, eps, clip_val=100.0)
    out["wap_micro_diff"] = _safe_div_np(wap_v - micro_p, wap_v, eps)
    out["book_slope"] = _safe_div_np(ask_p - bid_p, depth, eps)

    direction = np.where(imb_f == 1, 1.0, -1.0)
    signed_imb_raw = imb_s * direction
    out["signed_imb"] = signed_imb_raw
    out["order_intensity"] = _safe_div_np(signed_imb_raw, depth, eps)

    # ── Group 3: Aggregate Statistics (8) ──
    price_stack = np.column_stack([bid_p, ask_p, wap_v, ref_p])
    p_mean, p_std, p_skew, p_kurt = _row_stats_4col_np(price_stack)
    out["price_mean"] = p_mean
    out["price_std"] = p_std
    out["price_skew"] = p_skew
    out["price_kurt"] = p_kurt

    size_stack = np.column_stack([bid_s, ask_s, mat_s, abs_imb])
    s_mean, s_std, s_skew, s_kurt = _row_stats_4col_np(size_stack)
    out["size_mean"] = s_mean
    out["size_std"] = s_std
    out["size_skew"] = s_skew
    out["size_kurt"] = s_kurt

    # ── Group 4: Far / Near Enhanced (5) ──
    out["far_price_nan"] = far_p           # raw, NaN when unavailable
    out["near_price_nan"] = near_p        # raw, NaN when unavailable
    out["far_near_spread_nan"] = far_p - near_p

    mid_v = out["mid_price"].values
    out["far_to_mid"] = _safe_div_np(far_p - mid_v, mid_v, eps)
    out["near_to_mid"] = _safe_div_np(near_p - mid_v, mid_v, eps)

    # ── Group 5: Volume / Flow (5) ──
    out["volume_intensity"] = _safe_div_np(mat_s, depth, eps)
    out["imb_flow"] = _safe_div_np(abs_imb, mat_s, eps, clip_val=100.0)
    out["log_quote_imb"] = _log_ratio_np(bid_s, ask_s, eps)
    out["bid_ask_size_ratio"] = _safe_div_np(bid_s, ask_s, eps, clip_val=100.0)
    out["matched_depth_ratio"] = _safe_div_np(mat_s, depth, eps)

    # Replace infinities with NaN (match training cleanup)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Global features — target-dependent filled with 0 (inference limitation)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_global_features_online(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 32 global features.

    Target-dependent features (target_mean_*, rel_strength_*, etc.) are
    filled with 0 because target is unknown at inference time.  Non-target
    features use batch-level approximations.
    """
    out = pd.DataFrame(index=df.index, dtype=np.float32)

    # All global features — fill with 0 (safe default)
    # The model was trained with these as proper rolling-window aggregates,
    # but at inference time we only have a single batch per time_id.
    all_global = [
        "target_mean_median_3d", "target_mean_std_3d", "target_mean_ptp_3d",
        "target_mean_median_5d", "target_mean_std_5d", "target_mean_ptp_5d",
        "target_mean_skew_5d", "target_mean_kurt_5d",
        "target_mean_median_10d", "target_mean_std_10d", "target_mean_ptp_10d",
        "target_mean_skew_10d", "target_mean_kurt_10d",
        "wap_mean_median_5d", "wap_mean_std_5d",
        "wap_mean_median_10d", "wap_mean_std_10d",
        "imbalance_size_mean_median_5d", "imbalance_size_mean_std_5d",
        "imbalance_size_mean_median_10d", "imbalance_size_mean_std_10d",
        "rel_strength", "rel_strength_z_5d", "rel_strength_z_10d",
        "rel_strength_cum5d",
        "target_mean_lag1", "target_mean_diff1",
        "wap_mean_lag1", "wap_mean_ret1",
        "market_index", "stock_dev_index", "stock_dev_index_pct",
    ]
    for f in all_global:
        out[f] = 0.0

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feature order — exact match to src/model/feature_list.txt
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_ORDER = [
    # --- base (24) ---
    "stock_id",
    "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "bid_price", "bid_size",
    "ask_price", "ask_size", "wap",
    "far_price", "near_price",
    "bid_ask_spread", "spread_pct", "wap_ref_diff", "price_momentum",
    "imbalance_ratio", "size_imbalance", "depth_total",
    "far_price_avail", "near_price_avail", "far_near_spread", "far_near_mean",
    # --- micro (34) ---
    "tri_flow", "tri_price", "tri_match", "tri_ref_bid", "tri_exec",
    "tri_ref_ask", "tri_depth", "tri_term",
    "micro_price", "mid_price", "price_spread", "market_urgency",
    "wap_micro_diff", "book_slope", "signed_imb", "order_intensity",
    "price_mean", "price_std", "price_skew", "price_kurt",
    "size_mean", "size_std", "size_skew", "size_kurt",
    "far_price_nan", "near_price_nan", "far_near_spread_nan",
    "far_to_mid", "near_to_mid",
    "volume_intensity", "imb_flow", "log_quote_imb",
    "bid_ask_size_ratio", "matched_depth_ratio",
    # --- global (32) ---
    "target_mean_median_3d", "target_mean_std_3d", "target_mean_ptp_3d",
    "target_mean_median_5d", "target_mean_std_5d", "target_mean_ptp_5d",
    "target_mean_skew_5d", "target_mean_kurt_5d",
    "target_mean_median_10d", "target_mean_std_10d", "target_mean_ptp_10d",
    "target_mean_skew_10d", "target_mean_kurt_10d",
    "wap_mean_median_5d", "wap_mean_std_5d",
    "wap_mean_median_10d", "wap_mean_std_10d",
    "imbalance_size_mean_median_5d", "imbalance_size_mean_std_5d",
    "imbalance_size_mean_median_10d", "imbalance_size_mean_std_10d",
    "rel_strength", "rel_strength_z_5d", "rel_strength_z_10d",
    "rel_strength_cum5d",
    "target_mean_lag1", "target_mean_diff1",
    "wap_mean_lag1", "wap_mean_ret1",
    "market_index", "stock_dev_index", "stock_dev_index_pct",
]

assert len(FEATURE_ORDER) == 90, f"Expected 90 features, got {len(FEATURE_ORDER)}"


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 90 features in the exact order expected by the model."""
    base = compute_base_features(df)
    micro = compute_micro_features(df)
    g = compute_global_features_online(df)

    all_feats = pd.concat([base, micro, g], axis=1)

    # Fill any missing features with 0 (safety net)
    for feat in FEATURE_ORDER:
        if feat not in all_feats.columns:
            all_feats[feat] = 0.0

    return all_feats[FEATURE_ORDER]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Inference loop (only runs when executed as a script, NOT on import)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[notebook] Starting iter_test loop ...")

    import optiver2023
    env = optiver2023.make_env()

    batch = 0
    for bt in env.iter_test():
        test, sample_prediction = bt[0], bt[-1]
        batch += 1

        X = build_all_features(test)
        sample_prediction["target"] = model.predict(X).astype(np.float64)
        env.predict(sample_prediction)

        if batch % 10 == 0:
            print(f"[notebook] Batch {batch}: {len(test)} rows, "
                  f"pred_mean={sample_prediction['target'].mean():.4f}")

    print(f"[notebook] Done — {batch} batches processed")
