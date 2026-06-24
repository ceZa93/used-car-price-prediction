"""
MODUL: PRIPREMA PODATAKA
Kombinuje učitavanje sirovih podataka sa čišćenjem i validacijom.
Ključne operacije:
  1. Učitavanje CSV-a
  2. Normalizacija tekstualnih i numeričkih kolona
  3. Ekstrakcija relevantnih atributa (npr. brand iz car_name)
  4. Popunjavanje nedostajućih vrednosti
  5. Uklanjanje outlier-a i nevalidnih zapisa
  6. Otpuštanje irelantnih kolona
"""
import numpy as np
import pandas as pd


class DataPreparation:
    """
    Učitava sirove podatke i primenjuje pipeline-a čišćenja.
    """
    
    def __init__(self, data_path='data/raw/serbia_car_sales_price_2024_v2.csv', reference_year=2024):
        self.data_path = data_path
        self.reference_year = reference_year
        self.df = None

    def load(self):
        """Učitava CSV fajl i ispisu osnovne informacije."""
        self.df = pd.read_csv(self.data_path)
        print(f"OK Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"   Columns: {list(self.df.columns)}")
        return self

    def show_info(self):
        """Prikazuje info o dataset-u (info, missing values, first rows)."""
        print("\n=== DATASET INFO ===")
        print(self.df.info())
        print("\n=== MISSING VALUES ===")
        print(self.df.isnull().sum())
        print("\n=== FIRST 5 ROWS ===")
        print(self.df.head())
        return self

    def _normalize_text_columns(self):
        """Normlizuje tekstualne kolone: trim, zamena praznih sa NaN."""
        text_columns = self.df.select_dtypes(include=['object', 'string']).columns
        for col in text_columns:
            self.df[col] = self.df[col].astype('string').str.strip()
            self.df[col] = self.df[col].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
        return self

    def _coerce_numeric_columns(self):
        """Forsira tekstualne kolone koje trebaju biti numeričke na brojeve."""
        numeric_like_columns = [
            'price',
            'year',
            'mileage',
            'horsepower',
            'engine_capacity, cc',
            'seats_amount',
            'views',
            'favorite',
        ]

        for col in numeric_like_columns:
            if col in self.df.columns:
                extracted = self.df[col].astype('string').str.replace(',', '.', regex=False)
                self.df[col] = pd.to_numeric(extracted.str.extract(r'([-+]?\d*\.?\d+)')[0], errors='coerce')
        return self

    def extract_brand(self):
        """Ekstrauje brend iz car_name (prvi reč, uppercase)."""
        print("\n=== EXTRACTING BRAND ===")
        if 'car_name' in self.df.columns:
            self.df['brand'] = (
                self.df['car_name']
                .astype('string')
                .str.split()
                .str[0]
                .str.upper()
                .fillna('UNKNOWN')
            )
            print("OK Brand extracted from car_name")
        return self

    def clean_horsepower(self):
        """Čisti numeričke kolone: normalizacija teksta i forsiranje na brojeve."""
        print("\n=== CLEANING NUMERIC FIELDS ===")
        self._normalize_text_columns()
        self._coerce_numeric_columns()

        if 'horsepower' in self.df.columns:
            self.df['horsepower'] = self.df['horsepower'].clip(lower=0)
        if 'mileage' in self.df.columns:
            self.df['mileage'] = self.df['mileage'].clip(lower=0)
        if 'engine_capacity, cc' in self.df.columns:
            self.df['engine_capacity, cc'] = self.df['engine_capacity, cc'].clip(lower=0)
        if 'year' in self.df.columns:
            self.df['year'] = self.df['year'].clip(lower=1950, upper=self.reference_year)

        print("OK Text and numeric columns normalized")
        return self

    def handle_missing_values(self):
        """Popunjava nedostajuće vrednosti (median za brojeve, 'Unknown' za tekst)."""
        print("\n=== HANDLING MISSING VALUES ===")

        # NAPOMENA: Brand je redundantan jer je informacija već u car_name!
        # car_name = "Alfa Romeo 145" → brand je samo "ALFA" koji je već u car_name
        # Ne ekstraktujem brand jer će biti izbačen u feature engineering-u

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print(f"OK {col}: filled with median ({missing_count} missing values)")

        categorical_cols = self.df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                print(f"OK {col}: filled with 'Unknown' ({missing_count} missing values)")

        return self

    def remove_outliers(self):
        """Uklanja očigledno nevalidne zapise (npr. cena=0, mileage=9999999)."""
        print("\n=== REMOVING OUTLIERS ===")
        before = len(self.df)

        masks = []

        if 'price' in self.df.columns:
            # IZBACI skupocene automobile (> 25k€) jer model na njima MASIRA!
            # After 25k€ model počinje da ozbiljno greši - to je samo 1.4% automobila
            # 99th percentile je 27,999€ - automobili iznad 25k su ekstremno retki
            masks.append(self.df['price'].between(100, 25000))
            print("   ⚠️  Price filter: 100 € - 25000 € (izbacujem skupocene > 25k€)")
        if 'car_mileage, km' in self.df.columns:
            masks.append(self.df['car_mileage, km'].between(0, 550000))
        if 'horsepower' in self.df.columns:
            # 99% auta ima < 280 KS (raspon 50-500 je premali)
            masks.append(self.df['horsepower'].between(50, 280))
        if 'engine_capacity, cc' in self.df.columns:
            # 99% auta ima < 3000 cc (raspon 500-5000 je premali)
            masks.append(self.df['engine_capacity, cc'].between(500, 3000))
        if 'year' in self.df.columns:
            masks.append(self.df['year'].between(1980, self.reference_year))
        if 'seats_amount' in self.df.columns:
            masks.append(self.df['seats_amount'].between(1, 9))

        if masks:
            combined_mask = np.logical_and.reduce(masks)
            self.df = self.df.loc[combined_mask].copy()

        after = len(self.df)
        print(f"OK Removed rows: {before - after} (remaining {after})")
        return self

    def drop_irrelevant_columns(self):
        """Otpušta kolone koje nisu relevantne za cenu."""
        print("\n=== DROPPING IRRELEVANT COLUMNS ===")
        # Izbacujem emission_class jer:
        # 1. Ima preveliku importance (30%+) što nije realno
        # 2. Korisnik ne želi da je koristi za određivanje cene
        # 3. Target encoding ga pravi previše dominantnim
        # Napomena: 'year' se NE briše jer će biti skaliran u FeatureEngineer (1980-2024 range)
        cols_to_drop = ['post_info', 'views', 'favorite', 'emission_class']
        existing_cols = [col for col in cols_to_drop if col in self.df.columns]
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            print(f"OK Dropped columns: {existing_cols}")
        else:
            print("OK No columns to drop")
        return self

    def prepare(self):
        """Izvršava kompletan pipeline čišćenja."""
        return (
            self.clean_horsepower()
            .handle_missing_values()
            .remove_outliers()
            .drop_irrelevant_columns()
        )

    def get_data(self):
        """Vraća pripremljene podatke kao DataFrame."""
        return self.df


if __name__ == "__main__":
    # Demo: Učitaj podatke i primeni čišćenje
    preparation = DataPreparation()
    preparation.load().show_info()
    preparation.prepare()
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE")
    print("="*50)
    print(f"Final shape: {preparation.df.shape}")
    print(preparation.df.head())
