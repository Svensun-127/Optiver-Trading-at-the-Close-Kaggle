"""
Optuna hyperparameter optimisation for LightGBM — Phase 4.

Loads data + 90 features once, pre-computes 5 fold splits, then runs
Optuna trials with 5-fold purged walk-forward CV.  Logs each trial to
CSV, saves the best config, and auto-trains a baseline_004 model.

Usage:
    python src/training/hpo.py                        # 30 trials (default)
    python src/training/hpo.py --n-trials 50          # custom trial count
    python src/training/hpo.py --config hpo_001
"""

import argparse
import gc
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.base_features import CATEGORICAL_FEATURES
from src.training.utils import MODEL_FEATURES, load_data_and_features
from src.validation.splitter import PurgedGroupTimeSeriesSplit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(name: str) -> Dict[str, Any]:
    """Load a YAML experiment config from ``config/{name}.yaml``."""
    config_path = Path(f"config/{name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def precompute_folds(
    X: pd.DataFrame, config: Dict[str, Any]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Pre-compute all fold (train_idx, val_idx) splits.

    Done once before HPO so every trial evaluates on identical folds,
    eliminating CV-partition noise from the search.
    """
    cv_cfg = config["cv"]
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=cv_cfg["n_splits"],
        purge_gap=cv_cfg["purge_gap"],
        date_col=cv_cfg["date_col"],
    )
    return list(splitter.split(X))


def sample_params(
    trial: optuna.Trial, search_space: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert the YAML search_space definition into Optuna ``suggest_*`` calls."""
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["values"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    return params


def create_within_fold_val(
    train_dates: np.ndarray, val_frac: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sorted date_ids into inner-train and inner-val (last *val_frac*)."""
    n_val = max(1, int(len(train_dates) * val_frac))
    return train_dates[:-n_val], train_dates[-n_val:]


# ---------------------------------------------------------------------------
# Main HPO routine
# ---------------------------------------------------------------------------


def run_hpo(config: Dict[str, Any], n_trials_override: int | None = None) -> Dict[str, Any]:
    """
    Execute the full Optuna HPO pipeline.

    1. Load data + features once (via ``load_data_and_features``).
    2. Pre-compute 5 fold indices.
    3. Run *n_trials* Optuna trials, each with 5-fold walk-forward CV.
    4. Save per-trial results → ``outputs/hpo_log.csv``.
    5. Save best hyperparameters → ``config/baseline_004.yaml``.
    6. Auto-train final model → ``outputs/baseline_004/``.

    Returns the best-config dictionary.
    """
    # ── 1. Load data & features (once, reused across all trials) ───────
    print("=" * 64)
    print(f"  HPO: {config['experiment']['name']}")
    print("=" * 64)

    t_load_start = time.time()
    X, y, _date_id = load_data_and_features(config)
    print(f"  Data + features loaded in {time.time() - t_load_start:.0f}s")

    # ── 1b. Subsample stocks for HPO (memory constraint) ───────────────
    stock_frac = config.get("data", {}).get("stock_fraction", 1.0)
    if stock_frac < 1.0:
        rng = np.random.default_rng(42)
        all_stocks = X["stock_id"].unique()
        n_pick = max(1, int(len(all_stocks) * stock_frac))
        picked = rng.choice(all_stocks, size=n_pick, replace=False)
        mask = X["stock_id"].isin(picked)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask.values]
        print(f"  HPO stock subsample: {n_pick}/{len(all_stocks)} stocks  →  {len(X):,} rows")

    # ── 2. Pre-compute fold indices ────────────────────────────────────
    folds = precompute_folds(X, config)
    n_folds = len(folds)

    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {i + 1}/{n_folds}:  train={len(train_idx):,}  val={len(val_idx):,}")

    # ── 3. Setup search ────────────────────────────────────────────────
    search_cfg = config["search"]
    search_space = config["search_space"]
    fixed_params = config["fixed_params"].copy()
    n_trials = n_trials_override or search_cfg["n_trials"]

    param_names = list(search_space.keys())

    log_path = Path(config["output"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    trial_logs: List[Dict[str, Any]] = []

    # ── 4. Optuna objective ────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        trial_params = sample_params(trial, search_space)
        params = {**fixed_params, **trial_params}

        trial_start = time.time()
        fold_maes: List[float] = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train = y.iloc[train_idx].values
            y_val = y.iloc[val_idx].values

            # Within-fold early-stopping split (last 20 % of train dates)
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

            # Train
            model = LGBMRegressor(**params)
            model.fit(
                X_inner_train,
                y_inner_train,
                eval_set=[(X_inner_val, y_inner_val)],
                callbacks=[
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(period=0),
                ],
                categorical_feature=CATEGORICAL_FEATURES,
            )

            y_pred = model.predict(X_val)
            fold_mae = float(np.mean(np.abs(y_pred - y_val)))
            fold_maes.append(fold_mae)

            # Report intermediate value for pruning
            trial.report(fold_mae, fold_i)

            # ── Free fold-level memory ──────────────────────────────
            del X_train_raw, X_val_raw, y_train, y_val
            del X_inner_train, X_inner_val, y_inner_train, y_inner_val
            del X_val, y_pred
            del model
            gc.collect()

            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_mae = float(np.mean(fold_maes))
        trial_time = round(time.time() - trial_start, 1)

        # Log this trial
        log_row = {"trial": trial.number, "mae": round(mean_mae, 6), "time_seconds": trial_time}
        for pname in param_names:
            log_row[pname] = trial_params[pname]
        trial_logs.append(log_row)

        # Write log incrementally (crash-safe)
        pd.DataFrame(trial_logs).to_csv(log_path, index=False)

        print(f"  Trial {trial.number:2d}  MAE={mean_mae:.4f}  time={trial_time:.0f}s  "
              f"lr={trial_params['learning_rate']}  leaves={trial_params['num_leaves']}  "
              f"ff={trial_params['feature_fraction']}  bf={trial_params['bagging_fraction']}")

        return mean_mae

    # ── 5. Run Optuna study (SQLite-backed for crash resilience) ──────
    storage_path = Path("outputs/hpo_study.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{storage_path}",
        heartbeat_interval=60,
        grace_period=120,
    )

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
        interval_steps=1,
    )

    # Resume existing study or create a new one
    study_name = "hpo_001"
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
        )

    remaining = max(0, n_trials - len(study.trials))
    print(f"\n  Search space: {len(search_space)} params  |  "
          f"Existing: {len(study.trials)}  |  To run: {remaining}/{n_trials}  |  "
          f"Folds: {n_folds}  |  "
          f"Sampler: TPE  |  Pruner: Median\n")

    t_search_start = time.time()
    if remaining > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    else:
        print("  Study already has enough trials — skipping search.")

    search_time = time.time() - t_search_start

    # ── 6. Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  HPO complete  —  {n_trials} trials  |  "
          f"Search: {search_time:.0f}s  |  Best MAE: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"{'=' * 64}")
    print(f"  Log:  {log_path}  ({len(trial_logs)} trials)")

    # ── 7. Save best config ────────────────────────────────────────────
    template = load_config("baseline_003")
    best_params = study.best_params

    best_config: Dict[str, Any] = {
        "experiment": {
            "name": "baseline_004",
            "description": (
                f"Phase 4 HPO best — MAE {study.best_value:.4f} from "
                f"{n_trials}-trial Optuna search (hpo_001).  "
                f"Best params: {best_params}"
            ),
            "created": datetime.now().strftime("%Y-%m-%d"),
        },
        "data": template["data"],
        "preprocessing": template["preprocessing"],
        "cv": template["cv"],
        "features": template["features"],
        "model": {
            "type": "lightgbm",
            "params": {**fixed_params, **best_params, "n_jobs": -1},
            "early_stopping_rounds": 50,
            "eval_metric": "mae",
        },
        "output": {
            "dir": "outputs/baseline_004",
            "save_oof": True,
            "save_metrics": True,
            "save_feature_list": True,
        },
    }

    best_cfg_path = Path(config["output"]["best_config_path"])
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Config:  {best_cfg_path}")

    # ── 8. Train best model ────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  Training baseline_004 with best params ...")
    print(f"{'=' * 64}")

    from src.training.run import run_experiment

    run_experiment(best_config)

    return best_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for LightGBM — Phase 4 hyperparameter search."
    )
    parser.add_argument(
        "--config", default="hpo_001",
        help="HPO config name (default: hpo_001)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override n_trials from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_hpo(config, n_trials_override=args.n_trials)


if __name__ == "__main__":
    main()
