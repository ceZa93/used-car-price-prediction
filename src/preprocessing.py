import logging

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, df):
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("DataPreprocessor expects a pandas DataFrame, got "
                            f"{type(df).__name__}")
        if df.empty:
            raise ValueError("DataPreprocessor received an empty DataFrame")

        self.df = df.copy()
        if 'car_mileage, km' in self.df.columns:
            self.df = self.df.rename(columns={'car_mileage, km': 'mileage'})

    def clean_horsepower(self):
        print("\n=== CLEANING HORSEPOWER ===")
        if 'horsepower' not in self.df.columns:
            logger.warning("Column 'horsepower' not found — skipping clean_horsepower()")
            return self

        self.df['horsepower'] = (
            self.df['horsepower']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

        invalid_count = int(self.df['horsepower'].isna().sum())
        if invalid_count > 0:
            logger.warning(
                "%d rows have non-numeric horsepower values (now NaN)", invalid_count
            )

        print("Horsepower konvertovan u broj")
        return self

    def handle_missing_values(self):
        print("\n=== HANDLING MISSING VALUES ===")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print(f"  {col}: popunjeno sa median")

        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                print(f"  {col}: popunjeno sa 'Unknown'")

        print("\nSvi missing values uklonjeni!")
        return self

    def remove_outliers(self):
        print("\n=== REMOVING OUTLIERS ===")
        before = len(self.df)

        if 'mileage' in self.df.columns:
            self.df = self.df[self.df['mileage'] <= 1000000]
        if 'price' in self.df.columns:
            self.df = self.df[(self.df['price'] >= 100) & (self.df['price'] <= 100000)]

        after = len(self.df)
        removed = before - after
        print(f"Uklonjeni redovi: {removed} (ostalo {after})")

        if after == 0:
            raise ValueError(
                "All rows were removed during outlier filtering. "
                "Check the outlier thresholds or input data."
            )
        return self

    def encode_categorical(self):
        print("\n=== ENCODING CATEGORICAL COLUMNS ===")

        if 'car_name' in self.df.columns:
            self.df['brand'] = self.df['car_name'].astype(str).str.split().str[0].str.upper()
            print("Brand izvucen iz car_name")

        categorical_cols = ['A/C', 'fuel', 'car_type', 'type_of_drive', 'gearbox', 'doors', 'color']

        self.label_encoders = {}
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: encoded")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        try:
            joblib.dump(self.label_encoders, models_dir / "label_encoders.joblib")
        except OSError as exc:
            raise OSError(
                f"Failed to save label encoders to {models_dir / 'label_encoders.joblib'}: {exc}"
            ) from exc

        print("Label encoders sacuvani: models/label_encoders.joblib")
        return self

    def get_data(self):
        return self.df
