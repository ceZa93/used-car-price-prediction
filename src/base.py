"""
Base class for DataFrame pipeline stages.

Both DataPreprocessor and FeatureEngineer share the same pattern:
accept a DataFrame, mutate it through chainable methods, and expose
get_data().  This base class eliminates that duplication.
"""
from __future__ import annotations

import pandas as pd


class DataFramePipeline:
    """Thin base for pipeline stages that wrap a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, *, copy: bool = True) -> None:
        self.df: pd.DataFrame = df.copy() if copy else df

    def get_data(self) -> pd.DataFrame:
        return self.df
