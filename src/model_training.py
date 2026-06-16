import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.utils import log_section, log_step, save_artifact, compute_metrics


class ModelTrainer:
    def __init__(self, df, target_column='price', brand_column='brand'):
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
        log_section("PREPARING DATA")
        self.df = self.df.dropna()
        log_step(f"Redova posle uklanjanja NaN: {len(self.df)}")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        log_step(f"Train set: {len(self.X_train)} redova")
        log_step(f"Test set: {len(self.X_test)} redova")

        self._encode_brand_by_price()

        log_section("SCALING FEATURES")
        feature_columns = self.X_train.columns

        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=feature_columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=feature_columns)
        log_step("StandardScaler primenjen (fit na Train setu, transform na Test setu).")

        return self

    def _encode_brand_by_price(self):
        if self.brand_column not in self.X_train.columns:
            return

        log_section("TARGET ENCODING BRENDA (po prosečnoj ceni)")
        self.brand_global_mean = float(self.y_train.mean())
        self.brand_price_map = self.y_train.groupby(self.X_train[self.brand_column]).mean().to_dict()

        self.X_train = self.X_train.copy()
        self.X_test = self.X_test.copy()
        self.X_train[self.brand_column] = (
            self.X_train[self.brand_column].map(self.brand_price_map).fillna(self.brand_global_mean)
        )
        self.X_test[self.brand_column] = (
            self.X_test[self.brand_column].map(self.brand_price_map).fillna(self.brand_global_mean)
        )
        log_step(f"Brend enkodiran po prosečnoj ceni ({len(self.brand_price_map)} brendova).")

    def train_models(self):
        log_section("TRAINING MODELS")
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

            metrics = compute_metrics(self.y_test, y_pred)

            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE:  {metrics['mae']:.2f}")
            print(f"  R²:   {metrics['r2']:.4f}")

            self.models[model_name] = model
            self.results.append({"model": model_name, **metrics})
        return self

    def get_best_model(self):
        results_df = pd.DataFrame(self.results)
        best = results_df.loc[results_df['r2'].idxmax()]

        print("\n" + "=" * 50)
        print("BEST MODEL")
        print("=" * 50)
        print(f"Model: {best['model']}")
        print(f"R²:    {best['r2']:.4f}")
        print(f"RMSE:  {best['rmse']:.2f}")
        print(f"MAE:   {best['mae']:.2f}")
        print("=" * 50)
        return best['model'], results_df

    def save_best_model(self):
        log_section("SAVING MODEL AND SCALER")
        best_model_name, results_df = self.get_best_model()
        best_model = self.models[best_model_name]

        save_artifact(best_model, "best_model.joblib")
        save_artifact(self.scaler, "scaler.joblib")

        if self.brand_price_map is not None:
            save_artifact(
                {"map": self.brand_price_map, "global_mean": self.brand_global_mean},
                "brand_price_map.joblib",
            )

        results_df.to_csv("models/model_results.csv", index=False)
        return best_model

    def show_results(self):
        results_df = pd.DataFrame(self.results)
        log_section("RESULTS SUMMARY")
        print(results_df.sort_values(by='r2', ascending=False).to_string(index=False))
