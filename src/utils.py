"""
General-purpose utilities for the Optiver pipeline.

Provides memory compression and other helpers shared across modules.
"""

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, float_to_fp32: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest viable dtype.

    ``int64`` → ``int32`` / ``int16`` / ``int8`` based on value range.
    ``float64`` → ``float32`` when ``float_to_fp32=True``.

    Reduces memory footprint by ~50 % on the 5.2M-row Optiver dataset while
    preserving enough precision for tree-based models.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    float_to_fp32 : bool
        If True, downcast float64 → float32.

    Returns
    -------
    pd.DataFrame
        Same data with downcast numeric columns.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == np.float64 and float_to_fp32:
            df[col] = df[col].astype(np.float32)
        elif col_type == np.int64:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    pct = 100 * (start_mem - end_mem) / start_mem
    if pct > 0:
        print(f"  Memory reduced: {start_mem:.1f} MB → {end_mem:.1f} MB "
              f"({pct:.1f}% reduction)")

    return df
