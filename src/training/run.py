"""
Training script for baseline_001 — LightGBM with Purged Walk-Forward CV.

Loads raw data, builds features, runs 5-fold cross-validation, and saves
out-of-fold predictions, per-fold MAE metrics, and the feature list.

Usage:
    python src/training/run.py                     # default config
    python src/training/run.py --config baseline_001
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
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.base_features import ALL_FEATURES, CATEGORICAL_FEATURES
from src.features.global_features import GLOBAL_FEATURES
from src.features.micro_features import MICRO_FEATURES
from src.training.utils import MODEL_FEATURES, load_data_and_features
from src.validation.splitter import PurgedGroupTimeSeriesSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(name: str = "baseline_001") -> Dict[str, Any]:
    """Load a YAML experiment config from ``config/{name}.yaml``."""
    config_path = Path(f"config/{name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_output_dir(path: str) -> Path:
    """Create the output directory (idempotent)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def create_within_fold_val(
    train_dates: np.ndarray, val_frac: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split training date_ids into train and internal validation sets.

    Takes the **last** ``val_frac`` of date_ids (temporally closest to the
    real validation fold) as the internal holdout for early stopping. This
    respects temporal order — we never validate on earlier dates than we
    train on.

    Returns (train_dates, val_dates).
    """
    n_val = max(1, int(len(train_dates) * val_frac))
    return train_dates[:-n_val], train_dates[-n_val:]


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a full CV training run and return results.

    Parameters
    ----------
    config : dict
        Parsed experiment configuration.

    Returns
    -------
    dict
        Results dictionary with keys: ``fold_metrics``, ``overall_mae``,
        ``overall_mae_std``, ``elapsed_seconds``.
    """
    # ── 1. Load & build features ────────────────────────────────────────
    print("=" * 64)
    print(f"Experiment: {config['experiment']['name']}")
    print("=" * 64)

    t_start = time.time()
    X, y, _date_id = load_data_and_features(config)

    # ── 2. Setup CV splitter ────────────────────────────────────────────
    cv_cfg = config["cv"]
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=cv_cfg["n_splits"],
        purge_gap=cv_cfg["purge_gap"],
        date_col=cv_cfg["date_col"],
    )

    # ── 4. Cross-validation loop ────────────────────────────────────────
    model_cfg = config["model"]
    n_folds = cv_cfg["n_splits"]
    oof_preds = np.zeros(len(X), dtype=np.float32)
    oof_folds = np.full(len(X), -1, dtype=np.int8)

    fold_metrics: List[Dict[str, Any]] = []

    print(f"\nRunning {n_folds}-fold walk-forward CV ...")
    print(f"  Purge gap: {cv_cfg['purge_gap']}  |  "
          f"Features: {len(MODEL_FEATURES)}  |  "
          f"Rows: {len(X):,}")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
        fold_start = time.time()
        print(f"\n{'─' * 50}")
        print(f"  Fold {fold + 1}/{n_folds}")

        # Separate model features from metadata columns
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

        # ── Train model ─────────────────────────────────────────────────
        params = model_cfg["params"].copy()
        model = LGBMRegressor(**params)

        model.fit(
            X_inner_train, y_inner_train,
            eval_set=[(X_inner_val, y_inner_val)],
            callbacks=[
                early_stopping(
                    stopping_rounds=model_cfg["early_stopping_rounds"],
                    verbose=False,
                ),
                log_evaluation(period=100),
            ],
            categorical_feature=CATEGORICAL_FEATURES,
        )

        # ── Predict & store OOF ─────────────────────────────────────────
        oof_preds[val_idx] = model.predict(X_val).astype(np.float32)
        oof_folds[val_idx] = fold

        # ── Fold metrics ────────────────────────────────────────────────
        fold_mae = float(np.mean(np.abs(oof_preds[val_idx] - y_val)))
        best_iter = model.best_iteration_ if model.best_iteration_ else params["n_estimators"]
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

    # ── 5. Overall metrics ──────────────────────────────────────────────
    overall_mae = float(np.mean(np.abs(oof_preds - y.values)))
    fold_maes = [m["mae"] for m in fold_metrics]
    overall_mae_std = float(np.std(fold_maes))

    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 64}")
    print(f"  CV complete")
    print(f"  Overall MAE: {overall_mae:.4f} ± {overall_mae_std:.4f}")
    print(f"  Per-fold MAEs: {fold_maes}")
    print(f"  Total time: {total_elapsed:.0f}s")
    print(f"{'=' * 64}")

    # ── 6. Save outputs ─────────────────────────────────────────────────
    out_dir = setup_output_dir(config["output"]["dir"])

    # metrics.json
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

    # oof.csv
    oof_df = pd.DataFrame({
        "row_id": np.arange(len(oof_preds), dtype=np.int32),
        "target": y.values,
        "prediction": oof_preds,
        "fold": oof_folds,
    })
    oof_df.to_csv(out_dir / "oof.csv", index=False)
    print(f"  Saved: {out_dir / 'oof.csv'}  ({len(oof_df):,} rows)")

    # feature_list.txt
    with open(out_dir / "feature_list.txt", "w") as f:
        f.write(f"# Features for {config['experiment']['name']}\n")
        f.write(f"# Total: {len(MODEL_FEATURES)} features "
                f"(base: {len(ALL_FEATURES)}, micro: {len(MICRO_FEATURES)}, "
                f"global: {len(GLOBAL_FEATURES)})\n\n")
        f.write("## Categorical\n")
        for feat in CATEGORICAL_FEATURES:
            f.write(f"  {feat}\n")
        f.write("\n## Numerical (base)\n")
        for feat in ALL_FEATURES:
            if feat in CATEGORICAL_FEATURES:
                continue
            tag = ""
            if feat in config.get("features", {}).get("derived", []):
                tag = "  # derived"
            f.write(f"  {feat}{tag}\n")
        f.write("\n## Micro-structure\n")
        for feat in MICRO_FEATURES:
            f.write(f"  {feat}\n")
        f.write("\n## Global (stock-level aggregation)\n")
        for feat in GLOBAL_FEATURES:
            f.write(f"  {feat}\n")
    print(f"  Saved: {out_dir / 'feature_list.txt'}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM baseline with purged walk-forward CV."
    )
    parser.add_argument(
        "--config", default="baseline_001",
        help="Experiment config name (default: baseline_001)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
