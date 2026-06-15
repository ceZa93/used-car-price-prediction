"""
MODUL: TRENIRANJE MODELA
Zadatak: Trenira više modela i pronalazi najbolji
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path

class ModelTrainer:
    def __init__(self, df, target_column='price'):
        self.df = df.copy()
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = []
    
    def prepare_data(self):
        """Podeli podatke na X (features) i y (target)"""
        print("\n=== PREPARING DATA ===")
        
        # Ukloni redove sa NaN vrednostima
        self.df = self.df.dropna()
        print(f"✓ Redova posle uklanjanja NaN: {len(self.df)}")
        
        # Izdvoji features (X) i target (y)
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Podeli na train (80%) i test (20%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Train set: {len(self.X_train)} redova")
        print(f"✓ Test set: {len(self.X_test)} redova")
        return self
    
    def train_models(self):
        """Trenira sve modele"""
        print("\n=== TRAINING MODELS ===")
        
        models_dict = {
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
            "SVR": SVR(kernel='rbf')
        }
        
        for model_name, model in models_dict.items():
            print(f"\n--- {model_name} ---")
            
            # Treniranje
            model.fit(self.X_train, self.y_train)
            
            # Predikcija
            y_pred = model.predict(self.X_test)
            
            # Metrике
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R²:   {r2:.4f}")
            
            # Sačuvaj model
            self.models[model_name] = model
            
            # Sačuvaj rezultate
            self.results.append({
                "model": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })
        
        return self
    
    def get_best_model(self):
        """Pronađi najbolji model po R² score"""
        results_df = pd.DataFrame(self.results)
        best = results_df.loc[results_df['r2'].idxmax()]
        
        print("\n" + "="*50)
        print("BEST MODEL")
        print("="*50)
        print(f"Model: {best['model']}")
        print(f"R²:    {best['r2']:.4f}")
        print(f"RMSE:  {best['rmse']:.2f}")
        print(f"MAE:   {best['mae']:.2f}")
        print("="*50)
        
        return best['model'], results_df
    
    def save_best_model(self):
        """Sačuvaj najbolji model"""
        print("\n=== SAVING MODEL ===")
        
        best_model_name, results_df = self.get_best_model()
        best_model = self.models[best_model_name]
        
        # Kreiraj models folder
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Sačuvaj model
        model_path = models_dir / "best_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"✓ Model sačuvan: {model_path}")
        
        # Sačuvaj rezultate kao CSV
        results_path = models_dir / "model_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"✓ Rezultati sačuvani: {results_path}")
        
        return best_model
    
    def show_results(self):
        """Prikaži sve rezultate"""
        results_df = pd.DataFrame(self.results)
        print("\n=== RESULTS SUMMARY ===")
        print(results_df.sort_values(by='r2', ascending=False))


if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # 1. Load
    loader = DataLoader()
    df = loader.load()
    
    # 2. Preprocess
    processor = DataPreprocessor(df)
    df_clean = processor.handle_missing_values() \
                        .clean_horsepower() \
                        .remove_outliers() \
                        .encode_categorical() \
                        .get_data()
    
    # 3. Feature Engineering
    engineer = FeatureEngineer(df_clean)
    df_final = engineer.create_age_feature() \
                       .create_price_per_hp_feature() \
                       .create_price_per_cc_feature() \
                       .create_mileage_per_year_feature() \
                       .drop_unnecessary_columns() \
                       .scale_features() \
                       .get_data()
    
    # 4. Model Training
    trainer = ModelTrainer(df_final)
    trainer.prepare_data() \
           .train_models() \
           .show_results()
    
    # 5. Sačuvaj best model
    trainer.save_best_model()