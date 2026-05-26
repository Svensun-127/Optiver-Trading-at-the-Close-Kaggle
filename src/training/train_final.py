"""
Train final LightGBM model on ALL training data (no CV fold splitting).

Uses the best hyperparameters from baseline_004 (HPO-tuned).  Holds out the
last 5% of date_ids as a validation set for early stopping, then evaluates
the optimal iteration count.  The final model is saved with ``joblib``.

Usage:
    python src/training/train_final.py
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.base_features import CATEGORICAL_FEATURES
from src.training.utils import MODEL_FEATURES, load_data_and_features


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 64)
    print("  Train Final Model (full data, no CV)")
    print("=" * 64)

    config = load_config("config/baseline_004.yaml")
    model_params = config["model"]["params"].copy()

    # ── 1. Load data + build all 90 features ────────────────────────────
    t0 = time.time()
    X, y, _date_id = load_data_and_features(config)
    print(f"  Data loaded: {len(X):,} rows, {len(MODEL_FEATURES)} features "
          f"({time.time() - t0:.0f}s)")

    # ── 2. Hold-out last 5% date_ids for early stopping ──────────────────
    all_dates = np.sort(X["date_id"].unique())
    n_val_dates = max(1, int(len(all_dates) * 0.05))
    train_dates = all_dates[:-n_val_dates]
    val_dates = all_dates[-n_val_dates:]

    train_mask = X["date_id"].isin(train_dates)
    val_mask = X["date_id"].isin(val_dates)

    X_train = X.loc[train_mask, MODEL_FEATURES]
    y_train = y.loc[train_mask]
    X_val = X.loc[val_mask, MODEL_FEATURES]
    y_val = y.loc[val_mask]

    print(f"  Train: {len(X_train):,} rows ({len(train_dates)} dates)")
    print(f"  Val:   {len(X_val):,} rows ({len(val_dates)} dates)")

    # ── 3. Train with early stopping ─────────────────────────────────────
    print(f"\n  Training with {model_params['n_estimators']} max iterations ...")
    t1 = time.time()

    model = LGBMRegressor(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=100),
        ],
        categorical_feature=CATEGORICAL_FEATURES,
    )

    train_time = time.time() - t1
    best_iter = model.best_iteration_
    val_mae = float(model.best_score_["valid_0"]["l1"])

    print(f"\n  Best iteration: {best_iter}")
    print(f"  Val MAE:        {val_mae:.4f}")
    print(f"  Train time:     {train_time:.0f}s")

    # ── 4. Retrain on ALL data with best_iter trees ──────────────────────
    print(f"\n  Retraining on ALL data with n_estimators={best_iter} ...")
    t2 = time.time()

    final_params = {**model_params, "n_estimators": best_iter}
    final_model = LGBMRegressor(**final_params)
    X_all = X[MODEL_FEATURES]
    final_model.fit(
        X_all, y,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    print(f"  Final train time: {time.time() - t2:.0f}s")

    # ── 5. Save ──────────────────────────────────────────────────────────
    model_dir = Path("src/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lgb_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\n  Model saved: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Feature list for inference alignment
    feat_path = model_dir / "feature_list.txt"
    with open(feat_path, "w") as f:
        for feat in MODEL_FEATURES:
            f.write(f"{feat}\n")
    print(f"  Feature list: {feat_path} ({len(MODEL_FEATURES)} features)")

    print(f"\n{'=' * 64}")
    print(f"  Done. Total: {time.time() - t0:.0f}s")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
