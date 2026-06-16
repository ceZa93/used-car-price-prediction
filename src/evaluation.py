import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import log_section, save_plot, PLOTS_DIR


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(self.X_test)

        PLOTS_DIR.mkdir(exist_ok=True)

        sns.set_theme(style="whitegrid")

    def plot_actual_vs_predicted(self):
        log_section("PLOTTING ACTUAL VS PREDICTED")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, color='blue')

        max_val = max(self.y_test.max(), self.y_pred.max())
        min_val = min(self.y_test.min(), self.y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.title('Stvarne vs Predviđene Cene Automobila')
        plt.xlabel('Stvarne Cene (€)')
        plt.ylabel('Predviđene Cene (€)')

        save_plot('actual_vs_predicted.png')
        return self

    def plot_residuals(self):
        log_section("PLOTTING RESIDUALS")
        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='purple', bins=50)
        plt.title('Distribucija grešaka (Stvarna cena - Predviđena cena)')
        plt.xlabel('Greška u predikciji (€)')
        plt.ylabel('Broj automobila')

        plt.axvline(x=0, color='r', linestyle='--')

        save_plot('residuals_distribution.png')
        return self

    def plot_feature_importance(self):
        log_section("PLOTTING FEATURE IMPORTANCE")

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

            save_plot('feature_importance.png')
        else:
            print("! Ovaj model ne podržava feature_importances_ (npr. SVR ili KNN). Grafik je preskočen.")

        return self

    def run_all_evaluations(self):
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_feature_importance()
