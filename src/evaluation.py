"""
MODUL: EVALUACIJA MODELA
Zadatak: Detaljna analiza grešaka i vizuelizacija rezultata najboljeg modela
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(self.X_test)
        
        # Kreiraj folder za čuvanje grafika
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Postavljanje stila za grafike
        sns.set_theme(style="whitegrid")

    def plot_actual_vs_predicted(self):
        """Kreira scatter plot: Stvarne vs Predviđene cene"""
        print("\n=== PLOTTING ACTUAL VS PREDICTED ===")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, color='blue')
        
        # Dodavanje idealne linije (gde je stvarna cena = predviđena cena)
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
        """Kreira histogram reziduala (grešaka)"""
        print("\n=== PLOTTING RESIDUALS ===")
        residuals = self.y_test - self.y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='purple', bins=50)
        plt.title('Distribucija grešaka (Stvarna cena - Predviđena cena)')
        plt.xlabel('Greška u predikciji (€)')
        plt.ylabel('Broj automobila')
        
        # Dodajemo liniju na nuli (gde nema greške)
        plt.axvline(x=0, color='r', linestyle='--')
        
        save_path = self.plots_dir / 'residuals_distribution.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Grafik sačuvan: {save_path}")
        return self

    def plot_feature_importance(self):
        """Prikazuje koji atributi su najviše uticali na cenu (samo za Tree/Forest modele)"""
        print("\n=== PLOTTING FEATURE IMPORTANCE ===")
        
        # Proveravamo da li model ima atribut feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            features = self.X_test.columns
            
            # Kreiramo DataFrame za lakše sortiranje
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
        """Pokreće sve evaluacije odjednom"""
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_feature_importance()