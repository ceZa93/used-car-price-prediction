"""
MODUL: UCITAVANJE PODATAKA
"""
import pandas as pd


class DataLoader:
    def __init__(self, data_path='data/serbia_car_sales_price_2024_v2.csv'):
        self.data_path = data_path
        self.df = None

    def load(self):
        try:
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found: '{self.data_path}'. "
                "Check the file path and ensure the data directory exists."
            )
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Data file is empty: '{self.data_path}'. "
                "Provide a CSV with at least a header row."
            )
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Failed to parse CSV '{self.data_path}': {exc}"
            ) from exc

        if self.df.empty:
            raise ValueError(
                f"Loaded DataFrame from '{self.data_path}' has 0 rows. "
                "Provide a non-empty dataset."
            )

        print(f"Ucitano: {self.df.shape[0]} redova, {self.df.shape[1]} kolona")
        return self.df

    def show_info(self):
        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded yet. Call load() before show_info()."
            )

        print("\n=== DATASET INFO ===")
        print(self.df.info())
        print("\n=== MISSING VALUES ===")
        print(self.df.isnull().sum())
        print("\n=== FIRST 5 ROWS ===")
        print(self.df.head())
