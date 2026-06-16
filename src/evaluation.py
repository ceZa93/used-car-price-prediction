import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
)


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(self.X_test)
        self.metrics = {}

        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)

        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        sns.set_theme(style="whitegrid")

    def calculate_metrics(self):
        """Izračunava sve regresione metrike."""
        print("\n=== CALCULATING METRICS ===")

        y_test = np.array(self.y_test)
        y_pred = np.array(self.y_pred)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        medae = float(median_absolute_error(y_test, y_pred))
        evs = float(explained_variance_score(y_test, y_pred))

        residuals = y_test - y_pred
        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals))

        self.metrics = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "mape": round(mape * 100, 2),
            "median_absolute_error": round(medae, 2),
            "explained_variance_score": round(evs, 4),
            "mean_residual": round(mean_residual, 2),
            "std_residual": round(std_residual, 2),
        }

        print(f"  RMSE:                    {self.metrics['rmse']} €")
        print(f"  MAE:                     {self.metrics['mae']} €")
        print(f"  R²:                      {self.metrics['r2']}")
        print(f"  MAPE:                    {self.metrics['mape']}%")
        print(f"  Median Absolute Error:   {self.metrics['median_absolute_error']} €")
        print(f"  Explained Variance:      {self.metrics['explained_variance_score']}")
        print(f"  Mean Residual (bias):    {self.metrics['mean_residual']} €")
        print(f"  Std Residual:            {self.metrics['std_residual']} €")

        return self

    def save_metrics(self, filename="metrics.json"):
        """Čuva metrike u JSON fajl."""
        print("\n=== SAVING METRICS ===")

        if not self.metrics:
            self.calculate_metrics()

        save_path = self.models_dir / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"✓ Metrike sačuvane: {save_path}")
        return self

    def plot_actual_vs_predicted(self):
        print("\n=== PLOTTING ACTUAL VS PREDICTED ===")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, color='blue')

        max_val = max(self.y_test.max(), self.y_pred.max())
        min_val = min(self.y_test.min(), self.y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.title('Stvarne vs Predviđene Cene Automobila')
        plt.xlabel('Stvarne Cene (€)')
        plt.ylabel('Predviđene Cene (€)')

        save_path = self.plots_dir / 'actual_vs_predicted.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Grafik sačuvan: {save_path}")
        return self

    def plot_residuals(self):
        print("\n=== PLOTTING RESIDUALS ===")
        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='purple', bins=50)
        plt.title('Distribucija grešaka (Stvarna cena - Predviđena cena)')
        plt.xlabel('Greška u predikciji (€)')
        plt.ylabel('Broj automobila')

        plt.axvline(x=0, color='r', linestyle='--')

        save_path = self.plots_dir / 'residuals_distribution.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Grafik sačuvan: {save_path}")
        return self

    def plot_feature_importance(self):
        print("\n=== PLOTTING FEATURE IMPORTANCE ===")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            features = self.X_test.columns

            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', hue='Feature', legend=False)
            plt.title('Važnost atributa (Feature Importance)')
            plt.xlabel('Relativna važnost')
            plt.ylabel('Atribut')

            save_path = self.plots_dir / 'feature_importance.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"✓ Grafik sačuvan: {save_path}")
        else:
            print("! Ovaj model ne podržava feature_importances_ (npr. SVR ili KNN). Grafik je preskočen.")

        return self

    def run_all_evaluations(self):
        """Pokreće sve evaluacije: metrike + grafici + čuvanje."""
        self.calculate_metrics()
        self.save_metrics()
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_feature_importance()
