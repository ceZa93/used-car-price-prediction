import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
    
    def prepare_data(self):
        print("\n=== PREPARING DATA ===")
        self.df = self.df.dropna()
        print(f"✓ Redova posle uklanjanja NaN: {len(self.df)}")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✓ Train set: {len(self.X_train)} redova")
        print(f"✓ Test set: {len(self.X_test)} redova")

        print("\n=== SCALING FEATURES ===")
        feature_columns = self.X_train.columns
        
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=feature_columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=feature_columns)
        print("✓ StandardScaler primenjen (fit na Train setu, transform na Test setu).")
        
        return self
    
    def train_models(self):
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
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R²:   {r2:.4f}")
            
            self.models[model_name] = model
            self.results.append({"model": model_name, "rmse": rmse, "mae": mae, "r2": r2})
        return self
    
    def get_best_model(self):
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
        print("\n=== SAVING MODEL AND SCALER ===")
        best_model_name, results_df = self.get_best_model()
        best_model = self.models[best_model_name]
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(best_model, models_dir / "best_model.joblib")
        print(f"✓ Model sačuvan: models/best_model.joblib")

        joblib.dump(self.scaler, models_dir / "scaler.joblib")
        print(f"✓ Scaler sačuvan: models/scaler.joblib")
        
        results_df.to_csv(models_dir / "model_results.csv", index=False)
        return best_model
    
    def show_results(self):
        results_df = pd.DataFrame(self.results)
        print("\n=== RESULTS SUMMARY ===")
        print(results_df.sort_values(by='r2', ascending=False).to_string(index=False))