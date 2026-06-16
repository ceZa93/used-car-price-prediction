import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, df):
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"FeatureEngineer expects a pandas DataFrame, got {type(df).__name__}"
            )
        if df.empty:
            raise ValueError("FeatureEngineer received an empty DataFrame")
        self.df = df

    def create_features(self):
        print("\n=== CREATING NEW FEATURES ===")

        if 'year' in self.df.columns:
            self.df['car_age'] = 2024 - self.df['year']
            print("car_age: (2024 - year)")
        else:
            logger.warning("Column 'year' not found — cannot create car_age")

        if 'mileage' in self.df.columns and 'car_age' in self.df.columns:
            self.df['km_per_year'] = self.df['mileage'] / self.df['car_age'].replace(0, 1)
            print("km_per_year: mileage / car_age")
        else:
            logger.warning(
                "Columns 'mileage' and/or 'car_age' not found — cannot create km_per_year"
            )

        return self

    def drop_unnecessary_columns(self):
        print("\n=== DROPPING UNNECESSARY COLUMNS ===")

        cols_to_drop = [
            'post_info', 'car_name', 'color', 'emission_class',
            'year', 'favorite', 'views',
        ]

        existing_cols = [col for col in cols_to_drop if col in self.df.columns]

        self.df = self.df.drop(columns=existing_cols)
        print(f"Uklonjene kolone: {existing_cols}")

        return self

    def get_data(self):
        return self.df
