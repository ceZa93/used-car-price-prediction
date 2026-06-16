import logging

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

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, df, target_column='price', brand_column='brand'):
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"ModelTrainer expects a pandas DataFrame, got {type(df).__name__}"
            )
        if df.empty:
            raise ValueError("ModelTrainer received an empty DataFrame")

        self.df = df.copy()
        self.target_column = target_column
        self.brand_column = brand_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
        self.brand_price_map = None
        self.brand_global_mean = None

    def prepare_data(self):
        print("\n=== PREPARING DATA ===")
        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        self.df = self.df.dropna()
        if self.df.empty:
            raise ValueError(
                "DataFrame is empty after dropping NaN rows. "
                "Check input data quality."
            )
        print(f"Redova posle uklanjanja NaN: {len(self.df)}")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train set: {len(self.X_train)} redova")
        print(f"Test set: {len(self.X_test)} redova")

        self._encode_brand_by_price()

        print("\n=== SCALING FEATURES ===")
        feature_columns = self.X_train.columns

        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=feature_columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=feature_columns
        )
        print("StandardScaler primenjen (fit na Train setu, transform na Test setu).")

        return self

    def _encode_brand_by_price(self):
        if self.brand_column not in self.X_train.columns:
            return

        print("\n=== TARGET ENCODING BRENDA (po prosecnoj ceni) ===")
        self.brand_global_mean = float(self.y_train.mean())
        self.brand_price_map = (
            self.y_train.groupby(self.X_train[self.brand_column]).mean().to_dict()
        )

        self.X_train = self.X_train.copy()
        self.X_test = self.X_test.copy()
        self.X_train[self.brand_column] = (
            self.X_train[self.brand_column]
            .map(self.brand_price_map)
            .fillna(self.brand_global_mean)
        )
        self.X_test[self.brand_column] = (
            self.X_test[self.brand_column]
            .map(self.brand_price_map)
            .fillna(self.brand_global_mean)
        )
        print(f"Brend enkodiran po prosecnoj ceni ({len(self.brand_price_map)} brendova).")

    def train_models(self):
        print("\n=== TRAINING MODELS ===")
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Training data is not prepared. Call prepare_data() before train_models()."
            )

        models_dict = {
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=15
            ),
            "SVR": SVR(kernel='rbf'),
        }

        for model_name, model in models_dict.items():
            print(f"\n--- {model_name} ---")

            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            except Exception as exc:
                logger.error("Model '%s' failed during training: %s", model_name, exc)
                print(f"  FAILED: {exc}")
                continue

            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R2:   {r2:.4f}")

            self.models[model_name] = model
            self.results.append({"model": model_name, "rmse": rmse, "mae": mae, "r2": r2})

        if not self.results:
            raise RuntimeError("All models failed during training. Cannot proceed.")

        return self

    def get_best_model(self):
        if not self.results:
            raise RuntimeError(
                "No model results available. "
                "Call train_models() before get_best_model()."
            )

        results_df = pd.DataFrame(self.results)
        best = results_df.loc[results_df['r2'].idxmax()]

        print("\n" + "=" * 50)
        print("BEST MODEL")
        print("=" * 50)
        print(f"Model: {best['model']}")
        print(f"R2:    {best['r2']:.4f}")
        print(f"RMSE:  {best['rmse']:.2f}")
        print(f"MAE:   {best['mae']:.2f}")
        print("=" * 50)
        return best['model'], results_df

    def save_best_model(self):
        print("\n=== SAVING MODEL AND SCALER ===")
        best_model_name, results_df = self.get_best_model()
        best_model = self.models[best_model_name]

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        try:
            joblib.dump(best_model, models_dir / "best_model.joblib")
            print("Model sacuvan: models/best_model.joblib")

            joblib.dump(self.scaler, models_dir / "scaler.joblib")
            print("Scaler sacuvan: models/scaler.joblib")

            if self.brand_price_map is not None:
                joblib.dump(
                    {"map": self.brand_price_map, "global_mean": self.brand_global_mean},
                    models_dir / "brand_price_map.joblib",
                )
                print("Brand price map sacuvan: models/brand_price_map.joblib")

            results_df.to_csv(models_dir / "model_results.csv", index=False)
        except OSError as exc:
            raise OSError(
                f"Failed to save model artifacts to '{models_dir}': {exc}"
            ) from exc

        return best_model

    def show_results(self):
        if not self.results:
            raise RuntimeError(
                "No results to show. Call train_models() first."
            )
        results_df = pd.DataFrame(self.results)
        print("\n=== RESULTS SUMMARY ===")
        print(results_df.sort_values(by='r2', ascending=False).to_string(index=False))
