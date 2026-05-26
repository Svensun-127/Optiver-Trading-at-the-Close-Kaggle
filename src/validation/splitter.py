"""
Purged Group Time Series Cross-Validation Splitter.

Implements a walk-forward cross-validation scheme designed for financial time
series where observations are grouped by trading day.  The purging step removes
training samples that fall within a configurable window on either side of each
validation fold, blocking information leakage from adjacent trading days.

Reference
---------
Community-validated implementation from the Kaggle Optiver competition.
Based on sklearn's TimeSeriesSplit extended with group-aware purging.
"""

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class PurgedGroupTimeSeriesSplit:
    """
    Time-series cross-validator that splits by date group with a purge gap.

    Observations from the same ``date_col`` stay together (never split across
    train / validation).  A purge window of ``purge_gap`` days on each side of
    every validation fold is stripped from the training set so the model cannot
    peek into temporally adjacent sessions.

    Financial context
    -----------------
    In auction-level data, the *target* is a 60-second forward price change.
    While overnight information decay makes cross-day leakage less acute than
    intraday leakage, adjacent trading days may still share momentum regimes,
    overnight gap patterns, or stock-specific liquidity profiles.  Purging one
    day on each side is a minimal-cost safeguard that prevents those subtle
    signals from inflating validation scores.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds (default 5).
    purge_gap : int
        Number of date_id values to remove from the training set on each side
        of the validation fold boundary (default 1).
    date_col : str
        Column name that identifies the trading day (default "date_id").
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 1,
        date_col: str = "date_id",
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.date_col = date_col

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train / validation index splits.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.  Must contain ``self.date_col``.
        y : pd.Series, optional
            Target.  Not used by the splitter but accepted for sklearn compat.
        groups : np.ndarray, optional
            Ignored; the splitter derives groups from ``X[self.date_col]``.

        Yields
        ------
        (train_idx, val_idx) : tuple of np.ndarray
            Integer-position indices for each fold.
        """
        if self.date_col not in X.columns:
            raise KeyError(
                f"date_col '{self.date_col}' not found in X columns: "
                f"{X.columns.tolist()}"
            )

        dates = X[self.date_col].values
        unique_dates = np.unique(dates)

        # Inner TimeSeriesSplit operates on the array of *unique* date ids
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=None,
            test_size=None,
            gap=0,  # we handle purging manually below
        )

        for fold_train_dates, fold_val_dates in tscv.split(unique_dates):
            # Extract the actual date_id values for this fold
            train_dates_raw = unique_dates[fold_train_dates]
            val_dates = unique_dates[fold_val_dates]

            # --- Apply purge gap ---
            # Remove training dates that fall within `purge_gap` of the
            # validation boundary (on BOTH sides of the val window).
            val_min = val_dates.min()
            val_max = val_dates.max()

            purged_train_dates = train_dates_raw[
                (train_dates_raw < val_min - self.purge_gap)
                | (train_dates_raw > val_max + self.purge_gap)
            ]

            if len(purged_train_dates) == 0:
                raise RuntimeError(
                    f"Fold produced empty training set after purging. "
                    f"Reduce purge_gap (currently {self.purge_gap}) or n_splits."
                )

            # Convert date_id values back to row-index positions
            train_idx = np.where(np.isin(dates, purged_train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]

            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        """Return the number of splits (required by sklearn API)."""
        return self.n_splits

    def describe_folds(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return a DataFrame summarising each fold's date ranges and sizes.

        Useful for CV design documentation and sanity-checking that purge gaps
        are wide enough and no date_id appears in both train and val.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ``self.date_col``.

        Returns
        -------
        pd.DataFrame
            Columns: fold, train_start, train_end, train_dates, val_start,
            val_end, val_dates, train_rows, val_rows, purge_gap_applied.
        """
        rows = []
        dates = X[self.date_col].values

        for fold, (train_idx, val_idx) in enumerate(self.split(X)):
            train_dates = np.unique(dates[train_idx])
            val_dates = np.unique(dates[val_idx])

            # Find the expected purge gap between train and val date ranges
            # The purge removes dates from BOTH sides:
            #   train |--purge gap--|  val  |--purge gap--| train
            if len(train_dates) > 0 and len(val_dates) > 0:
                train_max_lt_val = train_dates[train_dates < val_dates.min()]
                train_min_gt_val = train_dates[train_dates > val_dates.max()]
                gap_before = (
                    val_dates.min() - train_max_lt_val.max()
                    if len(train_max_lt_val) > 0
                    else 0
                )
                gap_after = (
                    train_min_gt_val.min() - val_dates.max()
                    if len(train_min_gt_val) > 0
                    else 0
                )
            else:
                gap_before = 0
                gap_after = 0

            rows.append({
                "fold": fold,
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "train_dates": len(train_dates),
                "val_start": val_dates.min(),
                "val_end": val_dates.max(),
                "val_dates": len(val_dates),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "gap_before_val": gap_before,
                "gap_after_val": gap_after,
            })

        return pd.DataFrame(rows)
