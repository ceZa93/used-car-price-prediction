"""
MODUL: OBRADA PODATAKA
Zadatak: Očisti podatke, radi sa missing values, encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self):
        """Popuni missing vrednosti"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # Numericke kolone - popuni sa median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print(f"✓ {col}: popunjeno sa median")
        
        # Categorical kolone - popuni sa 'Unknown'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                print(f"✓ {col}: popunjeno sa 'Unknown'")
        
        print(f"\n✓ Svi missing values uklonjeni!")
        return self
    
    def clean_horsepower(self):
        """Konvertuj horsepower u broj"""
        print("\n=== CLEANING HORSEPOWER ===")
        
        # Ukloni sve nakon razmaka i "HP"
        # Iz "106 HP (78 kW)" izvuci samo "106"
        self.df['horsepower'] = self.df['horsepower'].str.extract(r'(\d+)').astype(float)
        print("✓ Horsepower konvertovan u broj")
        return self
    
    def remove_outliers(self):
        """Ukloni outliere"""
        print("\n=== REMOVING OUTLIERS ===")
        
        before = len(self.df)
        
        # Ukloni mileage > 1 milion km
        self.df = self.df[self.df['car_mileage, km'] <= 1000000]
        
        # Ukloni cene < 100 ili > 100000
        self.df = self.df[(self.df['price'] >= 100) & (self.df['price'] <= 100000)]
        
        after = len(self.df)
        print(f"✓ Uklonjeni redovi: {before - after} (ostalo {after})")
        return self
    
    def encode_categorical(self):
        """Encoding kategorijskih kolona"""
        print("\n=== ENCODING CATEGORICAL COLUMNS ===")
        
        categorical_cols = ['A/C', 'fuel', 'car_type', 'type_of_drive', 'gearbox', 'doors', 'color']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                print(f"✓ {col}: encoded")
        
        return self
    
    def get_data(self):
        """Vrati čiste podatke"""
        return self.df


if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load()
    
    processor = DataPreprocessor(df)
    df_clean = processor.handle_missing_values() \
                        .clean_horsepower() \
                        .remove_outliers() \
                        .encode_categorical() \
                        .get_data()
    
    print(f"\n=== FINALNI DATASET ===")
    print(f"Redova: {df_clean.shape[0]}")
    print(f"Kolona: {df_clean.shape[1]}")
    print(df_clean.head())