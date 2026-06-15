"""
MODUL: FEATURE ENGINEERING
Zadatak: Kreiraj nove features iz postojećih i skaliraj podatke
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
    
    def create_age_feature(self):
        """Kreiraj feature: starost automobila"""
        print("\n=== CREATING NEW FEATURES ===")
        
        current_year = 2024
        self.df['car_age'] = current_year - self.df['year']
        print("✓ car_age: (2024 - year)")
        return self
    
    def create_price_per_hp_feature(self):
        """Kreiraj feature: cena po HP"""
        self.df['price_per_hp'] = self.df['price'] / (self.df['horsepower'] + 1)
        print("✓ price_per_hp: price / horsepower")
        return self
    
    def create_price_per_cc_feature(self):
        """Kreiraj feature: cena po CC"""
        self.df['price_per_cc'] = self.df['price'] / (self.df['engine_capacity, cc'] + 1)
        print("✓ price_per_cc: price / engine_capacity")
        return self
    
    def create_mileage_per_year_feature(self):
        """Kreiraj feature: km po godini"""
        self.df['km_per_year'] = self.df['car_mileage, km'] / (self.df['car_age'] + 1)
        print("✓ km_per_year: mileage / car_age")
        return self
    
    def drop_unnecessary_columns(self):
        """Ukloni kolone koje nam nisu potrebne"""
        print("\n=== DROPPING UNNECESSARY COLUMNS ===")
        
        cols_to_drop = ['post_info', 'car_name', 'color', 'emission_class']
        self.df = self.df.drop(columns=cols_to_drop)
        print(f"✓ Uklonjene kolone: {cols_to_drop}")
        return self
    
    def scale_features(self):
        """Skaliraj sve numeričke feature-e"""
        print("\n=== SCALING FEATURES ===")
        
        # Izdvoji numeric kolone (osim target varijable)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Skaliraj sve
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        print(f"✓ Skalirane kolone: {len(numeric_cols)}")
        print(f"✓ StandardScaler: mean=0, std=1")
        
        return self
    
    def get_data(self):
        """Vrati podatke sa novim features"""
        return self.df
    
    def get_scaler(self):
        """Vrati scaler objekat (trebalo bi za test podatke kasnije)"""
        return self.scaler


if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    
    loader = DataLoader()
    df = loader.load()
    
    processor = DataPreprocessor(df)
    df_clean = processor.handle_missing_values() \
                        .clean_horsepower() \
                        .remove_outliers() \
                        .encode_categorical() \
                        .get_data()
    
    engineer = FeatureEngineer(df_clean)
    df_final = engineer.create_age_feature() \
                       .create_price_per_hp_feature() \
                       .create_price_per_cc_feature() \
                       .create_mileage_per_year_feature() \
                       .drop_unnecessary_columns() \
                       .scale_features() \
                       .get_data()
    
    print(f"\n=== FINALNI DATASET SA FEATURES I SCALING ===")
    print(f"Redova: {df_final.shape[0]}")
    print(f"Kolona: {df_final.shape[1]}")
    print(f"\nPrve 5 redova (skalirani):")
    print(df_final.head())
    print(f"\nSve kolone:")
    print(df_final.columns.tolist())