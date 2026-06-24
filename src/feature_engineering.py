import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year=2024):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()

        # Skaliranje year-a umjesto kreiranja car_age
        # year range: 1980 (najstariji) do 2024 (sada)
        if 'year' in df.columns:
            year_numeric = pd.to_numeric(df['year'], errors='coerce')
            # Skaliraj year na raspon 0-1, gdje je 1980=0 i 2024=1
            df['year'] = ((year_numeric - 1980) / (2024 - 1980)).clip(0, 1)

        # Skaliranje sa MinMaxScaler umjesto log transformacija (tree-based modeli)
        if 'car_mileage, km' in df.columns:
            df['car_mileage, km'] = pd.to_numeric(df['car_mileage, km'], errors='coerce').clip(lower=0)

        if 'horsepower' in df.columns:
            df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce').clip(lower=0)

        if 'engine_capacity, cc' in df.columns:
            df['engine_capacity, cc'] = pd.to_numeric(df['engine_capacity, cc'], errors='coerce').clip(lower=0)

        if 'seats_amount' in df.columns:
            df['seats_amount'] = pd.to_numeric(df['seats_amount'], errors='coerce')

        # NAPOMENA: emission_class je IZBAČENA u data_preparation.py jer:
        # - Target encoding je činio previše dominantnim (30%+ importance)
        # - Korisnik ne želi da je koristi za određivanje cene
        # - Nema potrebe da se kreira emission_class_scaled

        # Drži gearbox kao kategoričku - biće target-encoded kasnije!

        # OBRIŠI brand kolonu - redundantna jer je informacija već u car_name!
        # car_name = "Alfa Romeo 145" (marka + model)
        # brand = "ALFA" (samo marka - već u car_name)
        if 'brand' in df.columns:
            df = df.drop(columns=['brand'])

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(list(input_features))
        if hasattr(self, 'feature_names_in_'):
            return np.array(self.feature_names_in_)
        return np.array([])


class CategoryTargetEncoder(BaseEstimator, TransformerMixin):
    """Target-mean encoding za više kategoričkih kolona (radi sa numpy arrays ili DataFrames)."""
    def __init__(self):
        self.encoders_ = {}
        self.columns_ = None

    def fit(self, X, y):
        # Ako dobijam numpy array sa n_features_in_, pravi DataFrame sa dummy names
        if isinstance(X, np.ndarray):
            if self.columns_ is None:
                self.columns_ = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.columns_)
        elif isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()
        else:
            raise ValueError("X mora biti DataFrame ili numpy array")
        
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        self.global_mean_ = float(y_series.mean())
        
        # Fit-uj encoder za svaku kolonu
        for col in X.columns:
            category_series = X[col].astype('string').fillna('Unknown')
            self.encoders_[col] = pd.Series(y_series).groupby(category_series).mean().to_dict()
        
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if self.columns_ is None:
                raise ValueError("Morate prvo fit-ovati encoder")
            X = pd.DataFrame(X, columns=self.columns_)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X mora biti DataFrame ili numpy array")
        
        result = X.copy()
        for col in X.columns:
            if col not in self.encoders_:
                continue
            category_series = X[col].astype('string').fillna('Unknown')
            encoded = category_series.map(self.encoders_[col]).fillna(self.global_mean_)
            result[col] = encoded
        
        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        if self.columns_ is not None:
            return np.array(self.columns_)
        raise ValueError("Encoder nije fit-ovan")


class BrandTargetEncoder(BaseEstimator, TransformerMixin):
    """Target-mean encoding za bilo koju kategoričku kolonu."""
    def __init__(self, column='brand'):
        self.column = column

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[self.column])

        category_series = X[self.column].astype('string').fillna('Unknown')
        self.global_mean_ = float(pd.Series(y).mean())
        self.category_means_ = pd.Series(y).groupby(category_series).mean().to_dict()
        self.feature_names_out_ = np.array([f'{self.column}_encoded'])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[self.column])

        category_series = X[self.column].astype('string').fillna('Unknown')
        encoded = category_series.map(self.category_means_).fillna(self.global_mean_)
        return pd.DataFrame({f'{self.column}_encoded': encoded.values}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_