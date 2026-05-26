"""
Training script for CatBoost baseline — Purged Walk-Forward CV.

Mirrors ``src/training/run.py`` but uses CatBoostRegressor instead of
LightGBM.  Shares the same 90-feature matrix and CV folds so OOF
predictions are directly comparable for heterogeneous ensembling.

Usage:
    python src/training/run_catboost.py
    python src/training/run_catboost.py --config catboost_baseline_001
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.base_features import ALL_FEATURES, CATEGORICAL_FEATURES
from src.features.global_features import GLOBAL_FEATURES
from src.features.micro_features import MICRO_FEATURES
from src.training.utils import MODEL_FEATURES, load_data_and_features
from src.validation.splitter import PurgedGroupTimeSeriesSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(name: str = "catboost_baseline_001") -> Dict[str, Any]:
    config_path = Path(f"config/{name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def create_within_fold_val(
    train_dates: np.ndarray, val_frac: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(len(train_dates) * val_frac))
    return train_dates[:-n_val], train_dates[-n_val:]


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    # ── 1. Load & build features ────────────────────────────────────────
    print("=" * 64)
    print(f"Experiment: {config['experiment']['name']}")
    print("=" * 64)

    t_start = time.time()
    X, y, _date_id = load_data_and_features(config)

    # ── 2. Setup CV splitter (identical to LightGBM runs) ────────────────
    cv_cfg = config["cv"]
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=cv_cfg["n_splits"],
        purge_gap=cv_cfg["purge_gap"],
        date_col=cv_cfg["date_col"],
    )

    # ── 3. Cross-validation loop ────────────────────────────────────────
    model_cfg = config["model"]
    n_folds = cv_cfg["n_splits"]
    oof_preds = np.zeros(len(X), dtype=np.float32)
    oof_folds = np.full(len(X), -1, dtype=np.int8)

    fold_metrics: List[Dict[str, Any]] = []

    # CatBoost categorical feature indices within MODEL_FEATURES
    cat_indices = [i for i, f in enumerate(MODEL_FEATURES) if f in CATEGORICAL_FEATURES]

    print(f"\nRunning {n_folds}-fold walk-forward CV (CatBoost) ...")
    print(f"  Purge gap: {cv_cfg['purge_gap']}  |  "
          f"Features: {len(MODEL_FEATURES)}  |  "
          f"Categorical: {len(cat_indices)}  |  "
          f"Rows: {len(X):,}")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
        fold_start = time.time()
        print(f"\n{'─' * 50}")
        print(f"  Fold {fold + 1}/{n_folds}")

        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx].values
        y_val = y.iloc[val_idx].values

        # Within-fold early-stopping split (last 20% of train date_ids)
        train_date_ids = np.sort(X_train_raw["date_id"].unique())
        inner_train_dates, inner_val_dates = create_within_fold_val(
            train_date_ids, val_frac=0.20
        )

        inner_train_mask = X_train_raw["date_id"].isin(inner_train_dates)
        inner_val_mask = X_train_raw["date_id"].isin(inner_val_dates)

        X_inner_train = X_train_raw.loc[inner_train_mask, MODEL_FEATURES]
        X_inner_val = X_train_raw.loc[inner_val_mask, MODEL_FEATURES]
        y_inner_train = y_train[inner_train_mask.values]
        y_inner_val = y_train[inner_val_mask.values]

        X_val = X_val_raw[MODEL_FEATURES]

        print(f"    Train (inner): {len(X_inner_train):,} rows  |  "
              f"Val (inner): {len(X_inner_val):,} rows")
        print(f"    Val (outer):   {len(X_val):,} rows")

        # ── Train CatBoost ──────────────────────────────────────────────
        params = model_cfg["params"].copy()
        # Extract CatBoost-specific fit arguments
        early_stop = model_cfg.get("early_stopping_rounds", 50)
        verbose = params.pop("verbose", 100)
        thread_count = params.pop("thread_count", -1)

        model = CatBoostRegressor(
            **params,
            thread_count=thread_count,
            verbose=verbose,
            early_stopping_rounds=early_stop,
        )

        model.fit(
            X_inner_train, y_inner_train,
            eval_set=(X_inner_val, y_inner_val),
            cat_features=cat_indices,
        )

        # ── Predict & store OOF ─────────────────────────────────────────
        oof_preds[val_idx] = model.predict(X_val).astype(np.float32)
        oof_folds[val_idx] = fold

        # ── Fold metrics ────────────────────────────────────────────────
        fold_mae = float(np.mean(np.abs(oof_preds[val_idx] - y_val)))
        best_iter = model.get_best_iteration()
        elapsed = time.time() - fold_start

        print(f"    Fold {fold} MAE: {fold_mae:.4f}  |  "
              f"Best iter: {best_iter}  |  "
              f"Time: {elapsed:.0f}s")

        fold_metrics.append({
            "fold": fold,
            "mae": round(fold_mae, 6),
            "best_iteration": best_iter,
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "elapsed_seconds": round(elapsed, 1),
        })

    # ── 4. Overall metrics ──────────────────────────────────────────────
    overall_mae = float(np.mean(np.abs(oof_preds - y.values)))
    fold_maes = [m["mae"] for m in fold_metrics]
    overall_mae_std = float(np.std(fold_maes))

    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 64}")
    print(f"  CV complete")
    print(f"  Overall MAE: {overall_mae:.4f} +/- {overall_mae_std:.4f}")
    print(f"  Per-fold MAEs: {fold_maes}")
    print(f"  Total time: {total_elapsed:.0f}s")
    print(f"{'=' * 64}")

    # ── 5. Save outputs ─────────────────────────────────────────────────
    out_dir = setup_output_dir(config["output"]["dir"])

    metrics = {
        "experiment": config["experiment"]["name"],
        "description": config["experiment"]["description"],
        "cv_strategy": "PurgedGroupTimeSeriesSplit",
        "n_splits": n_folds,
        "purge_gap": cv_cfg["purge_gap"],
        "target_winsorization": config["preprocessing"]["target_winsorization"],
        "n_base_features": len(ALL_FEATURES),
        "n_micro_features": len(MICRO_FEATURES),
        "n_global_features": len(GLOBAL_FEATURES),
        "n_features": len(MODEL_FEATURES),
        "fold_metrics": fold_metrics,
        "overall_mae": round(overall_mae, 6),
        "overall_mae_std": round(overall_mae_std, 6),
        "elapsed_seconds": round(total_elapsed, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved: {out_dir / 'metrics.json'}")

    oof_df = pd.DataFrame({
        "row_id": np.arange(len(oof_preds), dtype=np.int32),
        "target": y.values,
        "prediction": oof_preds,
        "fold": oof_folds,
    })
    oof_df.to_csv(out_dir / "oof.csv", index=False)
    print(f"  Saved: {out_dir / 'oof.csv'}  ({len(oof_df):,} rows)")

    with open(out_dir / "feature_list.txt", "w") as f:
        f.write(f"# Features for {config['experiment']['name']}\n")
        f.write(f"# Total: {len(MODEL_FEATURES)} features\n\n")
        for feat in MODEL_FEATURES:
            tag = " (cat)" if feat in CATEGORICAL_FEATURES else ""
            f.write(f"  {feat}{tag}\n")
    print(f"  Saved: {out_dir / 'feature_list.txt'}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CatBoost baseline with purged walk-forward CV."
    )
    parser.add_argument(
        "--config", default="catboost_baseline_001",
        help="Experiment config name (default: catboost_baseline_001)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
