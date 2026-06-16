import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def create_features(self):
        print("\n=== CREATING NEW FEATURES ===")
        
        if 'year' in self.df.columns:
            self.df['car_age'] = 2024 - self.df['year']
            print("✓ car_age: (2024 - year)")
        
        if 'mileage' in self.df.columns and 'car_age' in self.df.columns:
            # Izbegavanje deljenja sa nulom za nova kola
            self.df['km_per_year'] = self.df['mileage'] / self.df['car_age'].replace(0, 1)
            print("✓ km_per_year: mileage / car_age")
            
        return self

    def drop_unnecessary_columns(self):
        print("\n=== DROPPING UNNECESSARY COLUMNS ===")
        
        cols_to_drop = [
            'post_info', 'car_name', 'color', 'emission_class', 
            'year', 'favorite', 'views'
        ]
        
        existing_cols = [col for col in cols_to_drop if col in self.df.columns]
        
        self.df = self.df.drop(columns=existing_cols)
        print(f"✓ Uklonjene kolone: {existing_cols}")
        
        return self

    def get_data(self):
        return self.df