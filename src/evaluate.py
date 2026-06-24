"""
MODUL: EKSPLORATIVNA ANALIZA I EVALUACIJA
Sadrži EDA grafike pre treninga i evaluacijske grafike nakon treninga.

Redosled:
  1. DatasetExplorer - EDA grafici na pripremljenim podacima
  2. ModelEvaluator - grafici i metrike nakon treniranja modela
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


class DatasetExplorer:
    """Eksplorativna analiza sirovih/pripremljenih podataka."""

    def __init__(self, df, target_column='price', output_dir='results/figures'):
        self.df = df.copy()
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style='whitegrid')

    def _save_fig(self, name):
        """Čuva figuru sa automatskim zatvaranjem."""
        path = self.output_dir / name
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=160)
        plt.close()
        print(f'OK Saved plot: {path}')

    def plot_missing_values(self):
        """Grafik nedostajućih vrednosti."""
        missing = self.df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        plt.figure(figsize=(12, 6))
        if missing.empty:
            plt.text(0.5, 0.5, 'Nema nedostajućih vrednosti nakon čišćenja', 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
        else:
            sns.barplot(x=missing.index, y=missing.values, color='#2E86AB')
            plt.xticks(rotation=45, ha='right')
            plt.title('Nedostajuće vrednosti po koloni')
            plt.ylabel('Broj nedostajućih')
            plt.xlabel('Kolona')

        self._save_fig('01_missing_values.png')
        return self

    def plot_target_distribution(self):
        """Raspodela ciljne promenljive."""
        if self.target_column not in self.df.columns:
            return self

        plt.figure(figsize=(12, 6))
        sns.histplot(self.df[self.target_column], bins=40, kde=True, color='#D1495B')
        plt.title('Raspodela ciljne promenljive - cena')
        plt.xlabel('Cena (€)')
        plt.ylabel('Broj oglasa')
        self._save_fig('02_target_distribution.png')
        return self

    def plot_numeric_correlations(self):
        """Korelaciona matrica numeričkih atributa."""
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            return self

        corr = numeric_df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=False, linewidths=0.3)
        plt.title('Korelaciona matrica numeričkih atributa')
        self._save_fig('03_numeric_correlations.png')
        return self

    def plot_numeric_distributions(self):
        """Raspodele numeričkih atributa (boxplot)."""
        numeric_columns = [
            col for col in ['price', 'year', 'mileage', 'horsepower', 'engine_capacity, cc', 'seats_amount']
            if col in self.df.columns
        ]

        if not numeric_columns:
            return self

        rows = int(np.ceil(len(numeric_columns) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, column in zip(axes, numeric_columns):
            sns.boxplot(x=self.df[column], ax=ax, color='#5B8E7D')
            ax.set_title(f'Raspon i anomalije: {column}')
            ax.set_xlabel(column)

        for ax in axes[len(numeric_columns):]:
            ax.axis('off')

        self._save_fig('04_numeric_distributions.png')
        return self

    def plot_brand_price_analysis(self):
        """Analiza cene po brendu."""
        if 'brand' not in self.df.columns or self.target_column not in self.df.columns:
            return self

        brand_stats = (
            self.df.groupby('brand')[self.target_column]
            .agg(['mean', 'median', 'count'])
            .query('count >= 20')
            .sort_values('mean', ascending=False)
            .head(15)
            .reset_index()
        )

        if brand_stats.empty:
            return self

        plt.figure(figsize=(14, 7))
        sns.barplot(data=brand_stats, x='mean', y='brand', palette='viridis')
        plt.title('Prosečna cena po brendu (top 15 po broju oglasa)')
        plt.xlabel('Prosečna cena (€)')
        plt.ylabel('Brend')
        self._save_fig('05_brand_price_analysis.png')
        return self

    def save_summary_tables(self):
        """Čuva CSV tabele sa statistikama."""
        missing = self.df.isna().sum().sort_values(ascending=False).to_frame('missing_values')
        missing.to_csv(self.output_dir / 'missing_values_summary.csv')

        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            numeric_df.corr(numeric_only=True).to_csv(self.output_dir / 'correlation_matrix.csv')

        print(f'OK Saved EDA tables in: {self.output_dir}')
        return self

    def run_all(self):
        """Pokreni sve EDA analize."""
        self.plot_missing_values()
        self.plot_target_distribution()
        self.plot_numeric_correlations()
        self.plot_numeric_distributions()
        self.plot_brand_price_analysis()
        self.save_summary_tables()
        return self


class ModelEvaluator:
    """Evaluacijska analiza i grafici nakon treniranja modela."""

    def __init__(self, model, X_test, y_test, output_dir='results/figures', feature_names_out=None, X_test_scaled=None):
        self.model = model
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.y_pred = self.model.predict(self.X_test)
        self.feature_names_out = feature_names_out
        self.X_test_scaled = X_test_scaled  # Skailirane vrednosti za grafike!

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style='whitegrid')

    def print_metrics(self):
        """Ispisuje sve metrike modela."""
        print("\n=== METRIKE REGRESIONOG MODELA ===")

        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred) * 100
        medae = median_absolute_error(self.y_test, self.y_pred)
        evs = explained_variance_score(self.y_test, self.y_pred)

        print(f"  MAE  (Mean Absolute Error):          {mae:,.2f} €")
        print(f"  RMSE (Root Mean Squared Error):      {rmse:,.2f} €")
        print(f"  MSE  (Mean Squared Error):           {mse:,.2f}")
        print(f"  R²   (Coefficient of Determination): {r2:.4f}")
        print(f"  MAPE (Mean Abs Percentage Error):    {mape:.2f} %")
        print(f"  MedAE (Median Absolute Error):       {medae:,.2f} €")
        print(f"  EVS  (Explained Variance Score):     {evs:.4f}")

        self.metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "MSE": mse,
            "R2": r2,
            "MAPE": mape,
            "MedAE": medae,
            "EVS": evs,
        }
        return self

    def plot_metrics_summary(self):
        """Grafik sažetka metrika."""
        print("\n=== PLOTTING METRICS SUMMARY ===")
        
        # Kreiraj results/metrics folder ako ne postoji
        metrics_dir = Path('results/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)

        labels = ["MAE (€)", "RMSE (€)", "MedAE (€)"]
        values = [self.metrics["MAE"], self.metrics["RMSE"], self.metrics["MedAE"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].barh(labels, values, color=["#2196F3", "#FF9800", "#4CAF50"])
        axes[0].set_title("Greške modela (u €)")
        axes[0].set_xlabel("Vrednost (€)")
        for i, v in enumerate(values):
            axes[0].text(v + max(values) * 0.01, i, f"{v:,.0f}", va="center")

        score_labels = ["R²", "EVS", f"MAPE\n({self.metrics['MAPE']:.1f}%)"]
        score_values = [self.metrics["R2"], self.metrics["EVS"], 1 - self.metrics["MAPE"] / 100]
        colors = ["#2196F3", "#4CAF50", "#FF9800"]
        bars = axes[1].bar(score_labels, score_values, color=colors)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title("Performanse modela (bliže 1.0 = bolje)")
        axes[1].set_ylabel("Skor")
        axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        for bar, v in zip(bars, [self.metrics["R2"], self.metrics["EVS"], 1 - self.metrics["MAPE"] / 100]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")

        plt.tight_layout()
        save_path = metrics_dir / "metrics_summary.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=160)
        plt.close()
        print(f"✅ Saved plot: {save_path}")
        return self

    def plot_actual_vs_predicted(self):
        """Grafik stvarne vs predviđene vrednosti."""
        print("\n=== PLOTTING ACTUAL VS PREDICTED ===")

        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.y_test,
            self.y_pred,
            alpha=0.55,
            color="#1f77b4",
            label="Svaka tačka = jedan automobil",
        )

        max_val = max(self.y_test.max(), self.y_pred.max())
        min_val = min(self.y_test.min(), self.y_pred.min())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Idealno: predikcija = stvarna cena",
        )

        plt.title("Stvarne vs Predviđene Cene Automobila")
        plt.xlabel("Stvarna cena (€)")
        plt.ylabel("Predviđena cena (€)")
        plt.legend(loc="upper left")
        plt.text(
            0.02,
            0.98,
            "Iznad linije = model precenjuje\nIspod linije = model potcenjuje",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        save_path = self.output_dir / "actual_vs_predicted.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=160)
        plt.close()
        print(f"OK Saved plot: {save_path}")
        return self

    def plot_residuals(self):
        """Grafik distribucije grešaka (residuala)."""
        print("\n=== PLOTTING RESIDUALS ===")
        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color="purple", bins=50)
        plt.title("Distribucija grešaka (Stvarna cena - Predviđena cena)")
        plt.xlabel("Greška u predikciji (€)")
        plt.ylabel("Broj automobila")

        plt.axvline(x=0, color="r", linestyle="--")

        save_path = self.output_dir / "residuals_distribution.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=160)
        plt.close()
        print(f"OK Saved plot: {save_path}")
        return self

    def plot_feature_importance(self):
        """Grafik važnosti atributa (permutation importance - procentualno)."""
        print("\n=== PLOTTING FEATURE IMPORTANCE ===")
        
        # Kreiraj results/metrics folder ako ne postoji
        metrics_dir = Path('results/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)

        importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=15,
            random_state=42,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
        )

        fi_df = pd.DataFrame(
            {
                'Feature': self.X_test.columns,
                'Importance': importance.importances_mean,
                'Std': importance.importances_std,
            }
        )
        
        # NOVO: Normalizuj na procentualnu važnost
        total_importance = fi_df['Importance'].sum()
        fi_df['Importance_pct'] = (fi_df['Importance'] / total_importance * 100)
        
        fi_df = fi_df.sort_values(by='Importance_pct', ascending=False)
        top_features = fi_df.head(15)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_features['Feature'], top_features['Importance_pct'], color='#2A9D8F')
        plt.title('Važnost atributa - Relativna važnost (%)')
        plt.xlabel('Procenat važnosti (% od ukupne)')
        plt.ylabel('Atribut')
        
        # Dodaj vrednosti na bar-ovima
        for i, (bar, v) in enumerate(zip(bars, top_features['Importance_pct'])):
            plt.text(v + 0.2, i, f'{v:.1f}%', va='center', fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / 'feature_importance.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=160)
        plt.close()
        # Spremi CSV u results/metrics/ jer je to metrika
        csv_path = metrics_dir / 'feature_importance_table.csv'
        fi_df.to_csv(csv_path, index=False)
        print(f'✅ Saved plot: {save_path}')
        print(f'✅ Saved metrics: {csv_path}')

        return self

    def run_all_evaluations(self):
        """Pokreni sve evaluacijske analize."""
        self.print_metrics()
        self.plot_metrics_summary()
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_feature_importance()
        return self

    def save_processed_data_sample(self):
        """Čuva sample podataka i statistiku."""
        print("\n=== SAVING PROCESSED DATA SAMPLE ===")
        
        # Kreiraj data/processed folder ako ne postoji
        processed_data_dir = Path('data/processed')
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            X_test_df = pd.DataFrame(self.X_test)
        except:
            X_test_df = self.X_test.copy()
        
        X_test_df['actual_price'] = self.y_test.values
        X_test_df['predicted_price'] = self.y_pred
        X_test_df['error_absolute'] = np.abs(self.y_test.values - self.y_pred)
        X_test_df['error_percent'] = (np.abs(self.y_test.values - self.y_pred) / self.y_test.values * 100).round(2)
        
        # Spremi sample od prvih 50 redova -> data/processed/
        sample_df = X_test_df.head(50)
        sample_path = processed_data_dir / 'processed_data_sample.csv'
        sample_df.to_csv(sample_path, index=False)
        print(f"✅ Saved processed data sample: {sample_path}")
        
        # Spremi i statistiku -> data/processed/
        stats = {
            'Total test samples': len(X_test_df),
            'Mean actual price': f"{self.y_test.mean():.2f} €",
            'Median actual price': f"{self.y_test.median():.2f} €",
            'Mean predicted price': f"{self.y_pred.mean():.2f} €",
            'Mean absolute error': f"{np.abs(self.y_test.values - self.y_pred).mean():.2f} €",
            'Median absolute error': f"{median_absolute_error(self.y_test, self.y_pred):.2f} €",
            'Mean error percentage': f"{(np.abs(self.y_test.values - self.y_pred) / self.y_test.values * 100).mean():.2f} %",
        }
        
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        stats_path = processed_data_dir / 'processed_data_stats.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"✅ Saved data statistics: {stats_path}")
        
        # Spremi i vizuelizaciju distribucije nakon skaliranja
        self._plot_scaled_distributions()
        return self

    def _plot_scaled_distributions(self):
        """Grafik distribucija skaliranih vrednosti koje se koriste za treniranje."""
        print("\n=== PLOTTING SCALED FEATURE DISTRIBUTIONS ===")
        
        if self.X_test_scaled is None:
            print("❌ Nema skaliranih vrednosti za prikaz")
            return
        
        try:
            # Koristim skailirane vrednosti
            X_scaled_df = pd.DataFrame(self.X_test_scaled)
            
            # Ako imamo feature_names_out, koristi ih
            if self.feature_names_out:
                feature_names = list(self.feature_names_out)
                if len(feature_names) <= X_scaled_df.shape[1]:
                    X_scaled_df.columns = feature_names[:X_scaled_df.shape[1]]
            else:
                # Kreiraj generic names
                X_scaled_df.columns = [f'Feature_{i}' for i in range(X_scaled_df.shape[1])]
            
            # Uzmi prvo 6 kolona
            cols_to_plot = list(X_scaled_df.columns)[:6]
            X_plot = X_scaled_df[cols_to_plot]
            
            if len(cols_to_plot) == 0:
                print("OK No columns to plot")
                return
            
            print(f"✅ Plotting {len(cols_to_plot)} skaliranih features:")
            for i, col in enumerate(cols_to_plot, 1):
                print(f"   {i}. {col}")
            
            # Kreiraj 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            
            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                data = X_plot[col].dropna()
                
                # Histogram
                ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                
                # Mean line
                mean_val = data.mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                
                # Labels
                ax.set_title(f"{col}\n(skaliran 0-1)", fontsize=11, fontweight='bold')
                ax.set_xlabel("Vrednost")
                ax.set_ylabel("Frekvencija")
                ax.legend()
                ax.grid(alpha=0.3)
            
            # Isključi prazne subplots
            for idx in range(len(cols_to_plot), 6):
                axes[idx].axis('off')
            
            plt.tight_layout()
            save_path = self.output_dir / 'scaled_features_distribution.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=160)
            plt.close()
            print(f"✅ Saved plot: {save_path}")
            
        except Exception as e:
            print(f"❌ ERROR in _plot_scaled_distributions: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Demo
    print("Koristi src.evaluate kao modul u main.py")
