import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        if 'car_mileage, km' in self.df.columns:
            self.df = self.df.rename(columns={'car_mileage, km': 'mileage'})
    
    def clean_horsepower(self):
        """Konvertuj horsepower u broj (radi se PRE handle_missing_values)"""
        print("\n=== CLEANING HORSEPOWER ===")
        if 'horsepower' in self.df.columns:
            self.df['horsepower'] = self.df['horsepower'].astype(str).str.extract(r'(\d+)').astype(float)
            print("✓ Horsepower konvertovan u broj")
        return self

    def handle_missing_values(self):
        """Popuni missing vrednosti"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print(f"✓ {col}: popunjeno sa median")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                print(f"✓ {col}: popunjeno sa 'Unknown'")
        
        print("\n✓ Svi missing values uklonjeni!")
        return self
    
    def remove_outliers(self):
        """Ukloni outliere"""
        print("\n=== REMOVING OUTLIERS ===")
        before = len(self.df)
        
        if 'mileage' in self.df.columns:
            self.df = self.df[self.df['mileage'] <= 1000000]
        if 'price' in self.df.columns:
            self.df = self.df[(self.df['price'] >= 100) & (self.df['price'] <= 100000)]
        
        after = len(self.df)
        print(f"✓ Uklonjeni redovi: {before - after} (ostalo {after})")
        return self
    
    def encode_categorical(self):
        print("\n=== ENCODING CATEGORICAL COLUMNS ===")
        
        if 'car_name' in self.df.columns:
            self.df['brand'] = self.df['car_name'].astype(str).str.split().str[0].str.upper()
            print("✓ Brand izvučen iz car_name")
        
        # 2. Encoding
        # 'brand' se NAMERNO ne label-enkodira ovde: ostaje kao tekst da bi se
        # kasnije (u ModelTrainer-u) primenilo target encoding po prosečnoj ceni,
        # čime brend dobija mnogo jači uticaj na predikciju cene.
        categorical_cols = ['A/C', 'fuel', 'car_type', 'type_of_drive', 'gearbox', 'doors', 'color']
        
        self.label_encoders = {} 
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le # Sačuvaj
                print(f"✓ {col}: encoded")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        joblib.dump(self.label_encoders, models_dir / "label_encoders.joblib")
        print("✓ Label encoders sačuvani: models/label_encoders.joblib")
        return self
    
    def get_data(self):
        return self.df