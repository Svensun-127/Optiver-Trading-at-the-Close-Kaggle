"""
Windows-compatible lightweight replacement for the optiver2023 make_env API.

The original competition package (optiver2023/competition.cpython-310.so)
is compiled for Linux only. This module mirrors the MockApi behavior from
public_timeseries_testing_util.py but with no external dependencies beyond
pandas.

Usage (mirrors the competition pattern):
    import src.mock_api

    env = src.mock_api.make_env()
    for test, revealed_targets, sample_submission in env.iter_test():
        pred = model.predict(test)
        env.predict(pred)
"""

from typing import List, Sequence, Tuple

import pandas as pd


class MockApi:
    """Lightweight online-timeseries evaluator.

    Groups input CSV data by a shared column (default: time_id), iterates
    over groups in chronological order, and requires a predict() call for
    each group before advancing.

    Parameters
    ----------
    input_paths : sequence of str
        Paths to CSV files to serve. Must include at least 2 files.
        Typical: [test_path, revealed_targets_path, sample_submission_path].
    group_id_column : str
        Column that defines evaluation groups (default "time_id").
    export_group_id_column : bool
        If True, the group_id column is included in served DataFrames.
    """

    def __init__(
        self,
        input_paths: Sequence[str] = (),
        group_id_column: str = "time_id",
        export_group_id_column: bool = True,
    ):
        self.input_paths = input_paths
        self.group_id_column = group_id_column
        self.export_group_id_column = export_group_id_column

        if len(self.input_paths) < 2:
            raise ValueError(
                f"input_paths must have at least 2 entries, got {len(self.input_paths)}"
            )

        self._status = "initialized"
        self.predictions: List[pd.DataFrame] = []

    def iter_test(self) -> Tuple[pd.DataFrame, ...]:
        """Yield one group's worth of data from all input CSVs at a time.

        Groups are iterated in the order they first appear in the first
        input file. After yielding, the API blocks until predict() is called.
        """
        if self._status != "initialized":
            raise RuntimeError(
                "iter_test() can only be called once. Create a new env for each run."
            )

        dataframes = []
        for pth in self.input_paths:
            dataframes.append(pd.read_csv(pth, low_memory=False))

        group_order = (
            dataframes[0][self.group_id_column]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        dataframes = [df.set_index(self.group_id_column) for df in dataframes]

        for group_id in group_order:
            self._status = "prediction_needed"
            current_data = []
            for df in dataframes:
                cur_df = df.loc[group_id].copy()
                if not isinstance(cur_df, pd.DataFrame):
                    cur_df = pd.DataFrame(
                        {a: b for a, b in zip(cur_df.index.values, cur_df.values)},
                        index=[group_id],
                    )
                    cur_df.index.name = self.group_id_column
                cur_df = cur_df.reset_index(drop=not self.export_group_id_column)
                current_data.append(cur_df)
            yield tuple(current_data)

            while self._status != "prediction_received":
                print(
                    "Waiting for predict() call before advancing iter_test() ...",
                    flush=True,
                )
                yield None  # type: ignore[misc]

    def predict(self, user_predictions: pd.DataFrame) -> None:
        """Register predictions for the current group and unlock the next.

        Parameters
        ----------
        user_predictions : pd.DataFrame
            Must contain columns that match sample_submission format
            (typically "row_id" and "target").
        """
        if self._status == "finished":
            raise RuntimeError("All predictions already submitted.")
        if self._status != "prediction_needed":
            raise RuntimeError("Call iter_test() to get the next batch first.")
        if not isinstance(user_predictions, pd.DataFrame):
            raise TypeError("user_predictions must be a pandas DataFrame.")

        self.predictions.append(user_predictions)
        self._status = "prediction_received"

    @property
    def status(self) -> str:
        """Current lifecycle status: initialized | prediction_needed | prediction_received | finished."""
        return self._status


def make_env(
    input_paths: Sequence[str] = (),
    group_id_column: str = "time_id",
    export_group_id_column: bool = True,
) -> MockApi:
    """Factory function compatible with the competition's make_env() signature.

    Returns a MockApi instance configured with the given paths and settings.
    """
    return MockApi(
        input_paths=input_paths,
        group_id_column=group_id_column,
        export_group_id_column=export_group_id_column,
    )