import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.base import DataFramePipeline
from src.utils import log_section, log_step, save_artifact


class DataPreprocessor(DataFramePipeline):
    def __init__(self, df):
        super().__init__(df)
        if 'car_mileage, km' in self.df.columns:
            self.df = self.df.rename(columns={'car_mileage, km': 'mileage'})

    def clean_horsepower(self):
        log_section("CLEANING HORSEPOWER")
        if 'horsepower' in self.df.columns:
            self.df['horsepower'] = self.df['horsepower'].astype(str).str.extract(r'(\d+)').astype(float)
            log_step("Horsepower konvertovan u broj")
        return self

    def handle_missing_values(self):
        log_section("HANDLING MISSING VALUES")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                log_step(f"{col}: popunjeno sa median")

        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                log_step(f"{col}: popunjeno sa 'Unknown'")

        log_step("Svi missing values uklonjeni!")
        return self

    def remove_outliers(self):
        log_section("REMOVING OUTLIERS")
        before = len(self.df)

        if 'mileage' in self.df.columns:
            self.df = self.df[self.df['mileage'] <= 1000000]
        if 'price' in self.df.columns:
            self.df = self.df[(self.df['price'] >= 100) & (self.df['price'] <= 100000)]

        after = len(self.df)
        log_step(f"Uklonjeni redovi: {before - after} (ostalo {after})")
        return self

    def encode_categorical(self):
        log_section("ENCODING CATEGORICAL COLUMNS")

        if 'car_name' in self.df.columns:
            self.df['brand'] = self.df['car_name'].astype(str).str.split().str[0].str.upper()
            log_step("Brand izvučen iz car_name")

        categorical_cols = ['A/C', 'fuel', 'car_type', 'type_of_drive', 'gearbox', 'doors', 'color']

        self.label_encoders = {}
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                log_step(f"{col}: encoded")

        save_artifact(self.label_encoders, "label_encoders.joblib")
        return self
