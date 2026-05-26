"""
Shared training utilities for the Optiver pipeline.

Provides the canonical data-loading + feature-building sequence used by both
``run.py`` (single experiment) and ``hpo.py`` (hyperparameter search) so the
two scripts never diverge on preprocessing.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.features.base_features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    METADATA_COLS,
    build_features,
)
from src.features.micro_features import (
    MICRO_FEATURES,
    compute_micro_features,
)
from src.features.global_features import (
    GLOBAL_FEATURES,
    compute_global_features,
)
from src.utils import reduce_mem_usage

# Convenience constant — all 90 features the model sees.
MODEL_FEATURES: list[str] = ALL_FEATURES + MICRO_FEATURES + GLOBAL_FEATURES


def load_data_and_features(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Load training data, compress memory, build the full 90-feature matrix.

    Performs the canonical pipeline:
    1. ``pd.read_csv`` → ``reduce_mem_usage``
    2. ``compute_global_features`` (32 stock-level aggregation features)
    3. ``compute_micro_features``  (34 Numba-accelerated micro-structure features)
    4. ``build_features``          (24 base features + cleaning + winsorization)

    Parameters
    ----------
    config : dict
        Experiment / HPO configuration. Must contain ``data.train_path``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (n_samples × 92 columns). The 92 columns are the 90
        MODEL_FEATURES plus 2 METADATA_COLS (``date_id`` and ``stock_id``,
        where stock_id already serves as a categorical model feature).
    y : np.ndarray
        Winsorized target as a 1-d float64 array.
    date_id : np.ndarray
        ``date_id`` values aligned with ``X`` for temporal splitting.
    """
    # 1. Load raw data
    print("\nLoading data ...")
    df = pd.read_csv(config["data"]["train_path"])
    df = reduce_mem_usage(df)
    print(f"  Loaded {len(df):,} rows from {config['data']['train_path']}")

    # 2. Global features must be computed on the raw (uncleaned) df so that
    #    stock-date aggregation boundaries are consistent.
    print("  Computing global features ...")
    X_global = compute_global_features(df)

    # 3. Micro features likewise operate on raw data to preserve NaN far/near.
    print("  Computing micro features ...")
    X_micro = compute_micro_features(df)

    # 4. Base features handle cleaning + winsorization + derived features.
    X, y = build_features(df, X_micro=X_micro, X_global=X_global)

    date_id = X["date_id"].values

    return X, y, date_id
