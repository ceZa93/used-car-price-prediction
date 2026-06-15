"""
MODUL: UČITAVANJE PODATAKA
Zadatak: Učitaj CSV i prikaži osnovne informacije
"""

import pandas as pd

class DataLoader:
    def __init__(self, data_path='data/serbia_car_sales_price_2024_v2.csv'):
        self.data_path = data_path
        self.df = None
    
    def load(self):
        """Učitaj podatke"""
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Učitano: {self.df.shape[0]} redova, {self.df.shape[1]} kolona")
        return self.df
    
    def show_info(self):
        """Prikaži info"""
        print("\n=== DATASET INFO ===")
        print(self.df.info())
        print("\n=== MISSING VALUES ===")
        print(self.df.isnull().sum())
        print("\n=== FIRST 5 ROWS ===")
        print(self.df.head())


if __name__ == "__main__":
    loader = DataLoader()
    loader.load()
    loader.show_info()