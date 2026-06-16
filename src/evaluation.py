import logging

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        if model is None:
            raise ValueError("ModelEvaluator received a None model")
        if X_test is None or y_test is None:
            raise ValueError("ModelEvaluator requires both X_test and y_test")

        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        try:
            self.y_pred = self.model.predict(self.X_test)
        except Exception as exc:
            raise RuntimeError(
                f"Model prediction failed during evaluation init: {exc}"
            ) from exc

        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)

        sns.set_theme(style="whitegrid")

    def _save_plot(self, fig, filename):
        save_path = self.plots_dir / filename
        try:
            fig.savefig(save_path, bbox_inches='tight')
        except OSError as exc:
            logger.error("Failed to save plot '%s': %s", save_path, exc)
            raise OSError(
                f"Failed to save plot to '{save_path}': {exc}"
            ) from exc
        finally:
            plt.close(fig)
        print(f"Grafik sacuvan: {save_path}")

    def plot_actual_vs_predicted(self):
        print("\n=== PLOTTING ACTUAL VS PREDICTED ===")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.y_test, self.y_pred, alpha=0.5, color='blue')

        max_val = max(self.y_test.max(), self.y_pred.max())
        min_val = min(self.y_test.min(), self.y_pred.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax.set_title('Stvarne vs Predvidjene Cene Automobila')
        ax.set_xlabel('Stvarne Cene (EUR)')
        ax.set_ylabel('Predvidjene Cene (EUR)')

        self._save_plot(fig, 'actual_vs_predicted.png')
        return self

    def plot_residuals(self):
        print("\n=== PLOTTING RESIDUALS ===")
        residuals = self.y_test - self.y_pred

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='purple', bins=50, ax=ax)
        ax.set_title('Distribucija gresaka (Stvarna cena - Predvidjena cena)')
        ax.set_xlabel('Greska u predikciji (EUR)')
        ax.set_ylabel('Broj automobila')
        ax.axvline(x=0, color='r', linestyle='--')

        self._save_plot(fig, 'residuals_distribution.png')
        return self

    def plot_feature_importance(self):
        print("\n=== PLOTTING FEATURE IMPORTANCE ===")

        if not hasattr(self.model, 'feature_importances_'):
            logger.info(
                "Model %s does not support feature_importances_ — skipping plot.",
                type(self.model).__name__,
            )
            print(
                f"Model {type(self.model).__name__} ne podrzava "
                "feature_importances_ — grafik preskocen."
            )
            return self

        importances = self.model.feature_importances_
        features = self.X_test.columns

        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x='Importance', y='Feature', data=fi_df,
            palette='viridis', hue='Feature', legend=False, ax=ax,
        )
        ax.set_title('Vaznost atributa (Feature Importance)')
        ax.set_xlabel('Relativna vaznost')
        ax.set_ylabel('Atribut')

        self._save_plot(fig, 'feature_importance.png')
        return self

    def run_all_evaluations(self):
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_feature_importance()
