from src.base import DataFramePipeline
from src.utils import REFERENCE_YEAR, log_section, log_step


class FeatureEngineer(DataFramePipeline):
    def __init__(self, df):
        super().__init__(df, copy=False)

    def create_features(self):
        log_section("CREATING NEW FEATURES")

        if 'year' in self.df.columns:
            self.df['car_age'] = REFERENCE_YEAR - self.df['year']
            log_step(f"car_age: ({REFERENCE_YEAR} - year)")

        if 'mileage' in self.df.columns and 'car_age' in self.df.columns:
            self.df['km_per_year'] = self.df['mileage'] / self.df['car_age'].replace(0, 1)
            log_step("km_per_year: mileage / car_age")

        return self

    def drop_unnecessary_columns(self):
        log_section("DROPPING UNNECESSARY COLUMNS")

        cols_to_drop = [
            'post_info', 'car_name', 'color', 'emission_class',
            'year', 'favorite', 'views'
        ]

        existing_cols = [col for col in cols_to_drop if col in self.df.columns]

        self.df = self.df.drop(columns=existing_cols)
        log_step(f"Uklonjene kolone: {existing_cols}")

        return self
