import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, data_path='data/serbia_car_sales_price_2024_v2.csv'):
        self.data_path = data_path
        self.df = None
    
    def load_data(self):
        """Učitaj CSV fajl"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset učitan: {self.df.shape[0]} redova, {self.df.shape[1]} kolona")
            return self.df
        except FileNotFoundError:
            print(f"✗ Fajl nije pronađen: {self.data_path}")
            return None
    
    def get_basic_info(self):
        """Osnovne informacije o dataset-u"""
        if self.df is None:
            print("Prvo učitaj podatke sa load_data()")
            return
        
        print("\n=== OSNOVNE INFORMACIJE ===")
        print(f"Redova: {self.df.shape[0]}")
        print(f"Kolona: {self.df.shape[1]}")
        print(f"\nTipovi podataka:\n{self.df.dtypes}")
        print(f"\nMissing vrednosti:\n{self.df.isnull().sum()}")
    
    def get_data(self):
        """Vrati podatke"""
        return self.df


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    loader.get_basic_info()