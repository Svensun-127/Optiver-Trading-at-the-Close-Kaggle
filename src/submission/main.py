"""
Kaggle submission script for Optiver - Trading at the Close.

Loads the trained LightGBM model and runs the iter_test API loop, computing
features on-the-fly for each time_id batch.

Target-dependent global features (16 of 32) are unavailable at inference time
and are filled with zeros — the model was trained on all 90 features, so these
placeholders will cause some degradation vs. CV scores.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ── Paths (model files are at the submission root) ──────────────────────
_MODEL_ROOT = Path(__file__).resolve().parent
MODEL_PATH = _MODEL_ROOT / "lgb_model.pkl"
FEATURE_LIST_PATH = _MODEL_ROOT / "feature_list.txt"

# ── Load model and feature order ──────────────────────────────────────────
model: LGBMRegressor = joblib.load(MODEL_PATH)

with open(FEATURE_LIST_PATH) as f:
    FEATURE_ORDER = [line.strip() for line in f if line.strip()]

print(f"[main] Model loaded: {MODEL_PATH}")
print(f"[main] Features: {len(FEATURE_ORDER)}")
print(f"[main] Model n_estimators: {model.n_estimators_}")


# ── Feature computation functions (inlined for submission portability) ────

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24 base features from raw columns."""
    out = pd.DataFrame(index=df.index)

    # Copy raw numerical columns
    raw_cols = [
        "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
        "reference_price", "matched_size", "bid_price", "bid_size",
        "ask_price", "ask_size", "wap",
    ]
    for c in raw_cols:
        if c in df.columns:
            out[c] = df[c].astype(np.float32)

    # Categorical
    if "stock_id" in df.columns:
        out["stock_id"] = df["stock_id"].astype(np.int16)

    # Derived features
    out["bid_ask_spread"] = (df["ask_price"] - df["bid_price"]).astype(np.float32)
    out["spread_pct"] = (
        (df["ask_price"] - df["bid_price"]) / ((df["ask_price"] + df["bid_price"]) / 2 + 1e-8)
    ).astype(np.float32)
    out["wap_ref_diff"] = (df["wap"] - df["reference_price"]).astype(np.float32)
    out["price_momentum"] = (df["wap"] - df["far_price"]).astype(np.float32)
    out["imbalance_ratio"] = (
        df["imbalance_size"] / (df["matched_size"] + 1).astype(np.float32)
    )
    out["size_imbalance"] = (df["bid_size"] - df["ask_size"]).astype(np.float32)
    out["depth_total"] = (df["bid_size"] + df["ask_size"]).astype(np.float32)
    out["far_price_avail"] = df["far_price"].notna().astype(np.float32)
    out["near_price_avail"] = df["near_price"].notna().astype(np.float32)
    out["far_near_spread"] = (df["far_price"] - df["near_price"]).astype(np.float32)
    out["far_near_mean"] = ((df["far_price"] + df["near_price"]) / 2).astype(np.float32)

    return out


def compute_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 34 micro-structure features.

    Simplified pure-Python implementation (Numba not available in Kaggle env).
    """
    out = pd.DataFrame(index=df.index)
    eps = 1e-8

    wap = df["wap"].values.astype(np.float64)
    bid = df["bid_price"].values.astype(np.float64)
    ask = df["ask_price"].values.astype(np.float64)
    price = df["reference_price"].values.astype(np.float64)
    bid_sz = df["bid_size"].values.astype(np.float64)
    ask_sz = df["ask_size"].values.astype(np.float64)
    matched = df["matched_size"].values.astype(np.float64)
    imbalance = df["imbalance_size"].values.astype(np.float64)
    sec = df["seconds_in_bucket"].values.astype(np.float64)
    far = df["far_price"].fillna(0).values.astype(np.float64)
    near = df["near_price"].fillna(0).values.astype(np.float64)
    imb_flag = df["imbalance_buy_sell_flag"].values.astype(np.float64)

    # Triplet imbalance
    out["tri_flow"] = (bid_sz - ask_sz) / (bid_sz + ask_sz + eps)
    out["tri_price"] = (bid + ask) / 2.0
    out["tri_match"] = matched / (bid_sz + ask_sz + eps)
    out["tri_ref_bid"] = (price - bid) / (price + eps)
    out["tri_exec"] = matched / (matched + imbalance + eps)
    out["tri_ref_ask"] = (ask - price) / (price + eps)
    out["tri_depth"] = (bid_sz + ask_sz) / (matched + eps)
    out["tri_term"] = sec / 540.0  # normalized by max seconds_in_bucket

    # Price / depth
    out["micro_price"] = (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz + eps)
    out["mid_price"] = (bid + ask) / 2.0
    out["price_spread"] = (ask - bid) / (out["mid_price"] + eps)
    out["market_urgency"] = np.abs(imbalance) / (matched + eps)
    out["wap_micro_diff"] = (wap - out["micro_price"]) / (out["micro_price"] + eps)
    out["book_slope"] = (ask_sz - bid_sz) / (bid_sz + ask_sz + eps)
    out["signed_imb"] = imb_flag * np.abs(imbalance) / (matched + eps)
    out["order_intensity"] = (bid_sz + ask_sz) / (sec + 1)

    # Aggregate stats (per-batch, since we don't have full history)
    out["price_mean"] = price
    out["price_std"] = 0.0
    out["price_skew"] = 0.0
    out["price_kurt"] = 0.0
    out["size_mean"] = bid_sz + ask_sz
    out["size_std"] = 0.0
    out["size_skew"] = 0.0
    out["size_kurt"] = 0.0

    # Far/near enhanced
    out["far_price_nan"] = df["far_price"].isna().astype(np.float32)
    out["near_price_nan"] = df["near_price"].isna().astype(np.float32)
    out["far_near_spread_nan"] = (far - near)
    out["far_to_mid"] = far / (out["mid_price"] + eps)
    out["near_to_mid"] = near / (out["mid_price"] + eps)

    # Volume / flow
    out["volume_intensity"] = matched / (sec + 1)
    out["imb_flow"] = imbalance / (matched + eps)
    out["log_quote_imb"] = np.log((bid_sz + 1) / (ask_sz + 1))
    out["bid_ask_size_ratio"] = bid_sz / (ask_sz + eps)
    out["matched_depth_ratio"] = matched / (bid_sz + ask_sz + eps)

    return out.astype(np.float32)


def compute_global_features_online(
    df: pd.DataFrame, history: dict
) -> pd.DataFrame:
    """Compute 32 global features incrementally.

    Fills target-dependent features with 0 since they are unavailable at
    inference time.
    """
    out = pd.DataFrame(index=df.index, dtype=np.float32)

    # Target-dependent features (16) — unavailable at inference time
    target_feats = [
        "target_mean_median_3d", "target_mean_std_3d", "target_mean_ptp_3d",
        "target_mean_median_5d", "target_mean_std_5d", "target_mean_ptp_5d",
        "target_mean_skew_5d", "target_mean_kurt_5d",
        "target_mean_median_10d", "target_mean_std_10d", "target_mean_ptp_10d",
        "target_mean_skew_10d", "target_mean_kurt_10d",
        "target_mean_lag1", "target_mean_diff1",
    ]
    for f in target_feats:
        out[f] = 0.0

    # Non-target features — simplified batch-level computation
    wap_mean = df["wap"].mean() if "wap" in df.columns else 0.0
    wap_std = df["wap"].std() if "wap" in df.columns else 0.0
    imb_mean = df["imbalance_size"].mean() if "imbalance_size" in df.columns else 0.0
    imb_std = df["imbalance_size"].std() if "imbalance_size" in df.columns else 0.0

    out["wap_mean_median_5d"] = wap_mean
    out["wap_mean_std_5d"] = wap_std
    out["wap_mean_median_10d"] = wap_mean
    out["wap_mean_std_10d"] = wap_std
    out["imbalance_size_mean_median_5d"] = imb_mean
    out["imbalance_size_mean_std_5d"] = imb_std
    out["imbalance_size_mean_median_10d"] = imb_mean
    out["imbalance_size_mean_std_10d"] = imb_std

    out["rel_strength"] = 0.0
    out["rel_strength_z_5d"] = 0.0
    out["rel_strength_z_10d"] = 0.0
    out["rel_strength_cum5d"] = 0.0

    out["wap_mean_lag1"] = 0.0
    out["wap_mean_ret1"] = 0.0

    out["market_index"] = wap_mean
    out["stock_dev_index"] = 0.0
    out["stock_dev_index_pct"] = 0.0

    return out


def build_all_features(df: pd.DataFrame, history: dict) -> pd.DataFrame:
    """Compute all 90 features for a batch of test data."""
    base = compute_base_features(df)
    micro = compute_micro_features(df)
    global_ = compute_global_features_online(df, history)

    # Concatenate in correct order
    all_feats = pd.concat([base, micro, global_], axis=1)

    # Ensure all required features exist, fill missing with 0
    for feat in FEATURE_ORDER:
        if feat not in all_feats.columns:
            all_feats[feat] = 0.0

    return all_feats[FEATURE_ORDER]


# ── Main inference loop ──────────────────────────────────────────────────

def main():
    # Import the competition API (available in Kaggle environment)
    import optiver2023

    env = optiver2023.make_env()
    history = {}  # accumulate data across batches for future use

    batch_count = 0
    for test, sample_prediction in env.iter_test():
        batch_count += 1

        # Build features
        X = build_all_features(test, history)

        # Predict
        preds = model.predict(X)

        # Store predictions in sample_prediction DataFrame
        sample_prediction["target"] = preds.astype(np.float64)

        # Submit this batch
        env.predict(sample_prediction)

        if batch_count % 10 == 0:
            print(f"[main] Processed batch {batch_count}, "
                  f"rows={len(test)}, pred_mean={preds.mean():.4f}")

    print(f"[main] Done. Total batches: {batch_count}")


if __name__ == "__main__":
    main()
