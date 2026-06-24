"""
MODUL: TRENIRANJE I SELEKCIJA MODELA
Kombinuje treniranje više modela, validaciju, hyperparameter tuning, 
feature selection i finalni test na TEST skupu.

Redosled koraka:
  1. prepare_data() - split 70/15/15 (train/val/test)
  2. train_models() - GridSearchCV na svakom modelu sa CV=3
  3. compare_feature_sets() - permutation importance i selekcija top features
  4. refit_final_model() - refit na train+val sa odabranim features
  5. evaluate_on_test() - finalna evaluacija na TEST skupu
  6. save_best_model() - export .joblib i metrike
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor

from src.feature_engineering import FeatureEngineer, CategoryTargetEncoder


class ModelTrainer:
    """
    Trenira, validira, testira i čuva ML modele za predikciju cena.
    """

    def __init__(self, df, target_column='price', reference_year=2024, random_state=42):
        self.df = df.copy()
        self.target_column = target_column
        self.reference_year = reference_year
        self.random_state = random_state

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.models = {}
        self.results = []
        self.training_runs = {}
        self.best_model_name = None
        self.best_params = None
        self.best_pipeline = None
        self.selected_features = None
        self.feature_selection_summary = None
        self.feature_importance_summary = None
        self.use_log_target = True
        
        self.test_metrics = None

    def prepare_data(self):
        """Split 70/15/15 (train/val/test) pre bilo kakvog preprocessing-a."""
        print("\n=== PREPARING DATA (70/15/15 split) ===")
        # Target mora biti numerički pre split-a.
        self.df = self.df.dropna(subset=[self.target_column]).copy()
        self.df[self.target_column] = pd.to_numeric(self.df[self.target_column], errors='coerce')
        self.df = self.df.dropna(subset=[self.target_column]).copy()
        print(f"OK Rows after target cleaning: {len(self.df)}")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # 70% train, 30% temp (15% val + 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=self.random_state,
        )

        # Split temp 50/50 → val i test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=self.random_state,
        )

        self.X_train = X_train.reset_index(drop=True)
        self.X_val = X_val.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_val = y_val.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        print(f"OK Train set:      {len(self.X_train)} rows ({len(self.X_train)/len(X)*100:.0f}%)")
        print(f"OK Validation set: {len(self.X_val)} rows ({len(self.X_val)/len(X)*100:.0f}%)")
        print(f"OK Test set:       {len(self.X_test)} rows ({len(self.X_test)/len(X)*100:.0f}%)")
        return self

    def _set_feature_columns(self, X_reference):
        """Detektuje tip svake kolone nakon feature engineeringa."""
        engineered = FeatureEngineer(reference_year=self.reference_year).fit_transform(X_reference)
        self._numeric_columns = list(engineered.select_dtypes(include=[np.number]).columns)
        # Sve kategoričke kolone (uključujući brand) će biti target-encoded
        self._categorical_columns = list(engineered.select_dtypes(exclude=[np.number]).columns)

    def _build_preprocessor(self):
        """ColumnTransformer: MinMaxScaler za brojeve, Target encoding za kategoričke (sve 0-1)."""
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('target_encode', CategoryTargetEncoder()),
                ('scaler', MinMaxScaler()),  # Sve kategoričke vrednosti se takođe skaliraju 0-1
            ]
        )

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self._numeric_columns),
                ('cat', categorical_transformer, self._categorical_columns),
            ],
            remainder='drop',
            verbose_feature_names_out=True,  # Čuva names sa prefixima
        ).set_output(transform='pandas')  # Čuva DataFrame kroz sve korake

    def _fit_preprocessor_with_target(self, preprocessor, X, y):
        """Fit preprocessor sa target vrednostima (posebno za CategoryTargetEncoder)."""
        # Fit categorical transformator sa y za target encoding
        if len(self._categorical_columns) > 0:
            cat_transformer = preprocessor.named_transformers_['cat']
            target_encoder = cat_transformer.named_steps['target_encode']
            X_cat = X[self._categorical_columns]
            target_encoder.fit(X_cat, y)
        
        # Ostalo se fit-uje normalno
        preprocessor.fit(X, y)
        return preprocessor

    def _build_pipeline(self, estimator):
        """Feature engineering + preprocessing + model."""
        return Pipeline(
            steps=[
                ('feature_engineering', FeatureEngineer(reference_year=self.reference_year)),
                ('preprocessor', self._build_preprocessor()),
                ('model', estimator),
            ]
        )

    def _wrap_target_model(self, pipeline):
        """Primeni log transformaciju na target ako je enabled."""
        if not self.use_log_target:
            return pipeline

        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    def _prefix_param_grid(self, param_grid):
        """Prilagodi parameter grid za TransformedTargetRegressor."""
        if not self.use_log_target:
            return param_grid

        return {f'regressor__{key}': value for key, value in param_grid.items()}

    def _build_pipeline_for_columns(self, estimator, X_reference):
        """Kreira pipeline sa određenim feature set-om."""
        engineered = FeatureEngineer(reference_year=self.reference_year).fit_transform(X_reference)
        numeric_columns = list(engineered.select_dtypes(include=[np.number]).columns)
        categorical_columns = list(engineered.select_dtypes(exclude=[np.number]).columns)

        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('target_encode', CategoryTargetEncoder()),
                ('scaler', MinMaxScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns),
            ],
            remainder='drop',
            verbose_feature_names_out=True,
        ).set_output(transform='pandas')

        return Pipeline(
            steps=[
                ('feature_engineering', FeatureEngineer(reference_year=self.reference_year)),
                ('preprocessor', preprocessor),
                ('model', estimator),
            ]
        )

    def _get_candidate_models(self):
        """Definiši modele za treniranje i njihove hiperparametare."""
        return {
            'Ridge': (
                Ridge(),
                {'model__alpha': [0.3, 1.0, 3.0, 10.0]},
            ),
            'Random Forest': (
                RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                {
                    'model__n_estimators': [250, 400],
                    'model__max_depth': [None, 18],
                    'model__min_samples_leaf': [1, 3],
                },
            ),
            'Extra Trees': (
                ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
                {
                    'model__n_estimators': [300, 500],
                    'model__max_depth': [None, 18],
                    'model__min_samples_leaf': [1, 3],
                },
            ),
            'Gradient Boosting': (
                GradientBoostingRegressor(random_state=self.random_state),
                {
                    'model__n_estimators': [200, 400],
                    'model__learning_rate': [0.03, 0.05],
                    'model__max_depth': [3, 4],
                    'model__subsample': [0.8, 1.0],
                },
            ),
            'Hist Gradient Boosting': (
                HistGradientBoostingRegressor(random_state=self.random_state),
                {
                    'model__max_iter': [200, 400],
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [None, 8],
                    'model__min_samples_leaf': [20, 40],
                },
            ),
        }

    def _evaluate_predictions(self, y_true, y_pred):
        """Izračunaj MAE, RMSE, R²."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def train_models(self):
        """Trenira sve candidate modele sa GridSearchCV na validation skupu."""
        print("\n=== TRAINING AND VALIDATION ===")
        self._set_feature_columns(self.X_train)

        candidate_models = self._get_candidate_models()
        self.training_runs = {}
        self.results = []

        for model_name, (estimator, param_grid) in candidate_models.items():
            print(f"\n--- {model_name} ---")
            base_pipeline = self._build_pipeline(estimator)
            pipeline = self._wrap_target_model(base_pipeline)
            search = GridSearchCV(
                pipeline,
                param_grid=self._prefix_param_grid(param_grid),
                scoring='neg_mean_absolute_error',
                cv=3,
                n_jobs=-1,
                refit=True,
            )
            search.fit(self.X_train, self.y_train)

            val_pred = search.best_estimator_.predict(self.X_val)
            val_mae, val_rmse, val_r2 = self._evaluate_predictions(self.y_val, val_pred)

            print(f"  Best params: {search.best_params_}")
            print(f"  Validation MAE:  {val_mae:,.2f} €")
            print(f"  Validation RMSE: {val_rmse:,.2f} €")
            print(f"  Validation R²:   {val_r2:.4f}")

            self.training_runs[model_name] = search
            self.results.append(
                {
                    'model': model_name,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'best_params': search.best_params_,
                }
            )

        results_df = pd.DataFrame(self.results)
        best_row = results_df.sort_values('val_mae', ascending=True).iloc[0]
        self.best_model_name = best_row['model']
        self.best_params = best_row['best_params']
        self.best_pipeline = self.training_runs[self.best_model_name].best_estimator_

        print("\n=== BEST MODEL AFTER VALIDATION ===")
        print(f"Model: {self.best_model_name}")
        print(f"Validation MAE:  {best_row['val_mae']:,.2f} €")
        print(f"Validation RMSE: {best_row['val_rmse']:,.2f} €")
        print(f"Validation R²:   {best_row['val_r2']:.4f}")
        return self

    def compare_feature_sets(self, top_n=12):
        """Permutation importance: sva features vs top N features."""
        print("\n=== FEATURE SELECTION COMPARISON ===")
        if self.best_pipeline is None:
            raise RuntimeError('Morate prvo trenirati modele.')

        # Full model sa svim features
        full_model = self.best_pipeline
        full_val_pred = full_model.predict(self.X_val)
        full_mae, full_rmse, full_r2 = self._evaluate_predictions(self.y_val, full_val_pred)

        # Permutation importance
        importance = permutation_importance(
            full_model,
            self.X_val,
            self.y_val,
            n_repeats=10,
            random_state=self.random_state,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
        )

        # Uzmi feature names iz X_val (year je sada skaliran, ne car_age)
        feature_names = list(self.X_val.columns)
        
        print(f"\n✓ Feature names: {feature_names}")
        print(f"✓ Importance length: {len(importance.importances_mean)}")

        # Osiguraj da feature_names bude ista dužina kao importance.importances_mean
        if len(feature_names) != len(importance.importances_mean):
            print(f"\n⚠ MISMATCH: {len(feature_names)} features vs {len(importance.importances_mean)} importances")
            # Popuni nedostajuće sa generiranim imenima
            if len(feature_names) < len(importance.importances_mean):
                for i in range(len(feature_names), len(importance.importances_mean)):
                    feature_names.append(f"feature_{i}")
                print(f"✓ Dodan {len(importance.importances_mean) - len(feature_names)} missing feature names")
            else:
                feature_names = feature_names[:len(importance.importances_mean)]
        
        importance_df = pd.DataFrame(
            {
                'feature': feature_names,
                'importance': importance.importances_mean,
                'std': importance.importances_std,
            }
        )

        # ✅ ČUVA: importance_df za grafik
        self.feature_importance_summary = importance_df.copy()

        top_features = importance_df.head(top_n)['feature'].tolist()
        print(f"\n✅ Top {top_n} features: {top_features}")

        # Refit model sa samo top features
        trained_model = self.training_runs[self.best_model_name].best_estimator_
        if self.use_log_target:
            reduced_estimator = trained_model.regressor_.named_steps['model']
        else:
            reduced_estimator = trained_model.named_steps['model']

        reduced_model = reduced_estimator.__class__(**reduced_estimator.get_params())
        reduced_base_pipeline = self._build_pipeline_for_columns(reduced_model, self.X_train[top_features])
        reduced_pipeline = self._wrap_target_model(reduced_base_pipeline)

        if self.use_log_target:
            reduced_pipeline.set_params(**self.best_params)
        else:
            reduced_pipeline.set_params(**self.best_params)
        reduced_pipeline.fit(self.X_train[top_features], self.y_train)

        reduced_val_pred = reduced_pipeline.predict(self.X_val[top_features])
        reduced_mae, reduced_rmse, reduced_r2 = self._evaluate_predictions(self.y_val, reduced_val_pred)

        self.feature_selection_summary = pd.DataFrame(
            [
                {
                    'feature_set': 'all_features',
                    'feature_count': len(self.X_train.columns),
                    'val_mae': full_mae,
                    'val_rmse': full_rmse,
                    'val_r2': full_r2,
                },
                {
                    'feature_set': f'top_{top_n}_features',
                    'feature_count': len(top_features),
                    'val_mae': reduced_mae,
                    'val_rmse': reduced_rmse,
                    'val_r2': reduced_r2,
                },
            ]
        )

        # feature_importance_summary je već assign-ovan gore sa transformisanim kolonama

        # Odaberi najbolji set
        if reduced_mae < full_mae:
            self.best_pipeline = reduced_pipeline
            self.selected_features = top_features
            winner = f'top_{top_n}_features'
        else:
            self.selected_features = list(self.X_train.columns)
            winner = 'all_features'

        print(self.feature_selection_summary.to_string(index=False))
        print(f"OK Selected deployment set: {winner}")
        print(f"FINAL selected_features ({len(self.selected_features)}): {self.selected_features}")
        return self

    def refit_final_model(self):
        """Refit na train+val sa odabranim features."""
        print("\n=== REFITTING FINAL MODEL ===")
        if self.best_pipeline is None:
            raise RuntimeError('Nema modela za refit.')

        X_train_final = pd.concat([self.X_train, self.X_val], ignore_index=True)
        y_train_final = pd.concat([self.y_train, self.y_val], ignore_index=True)

        if self.selected_features is None:
            self.selected_features = list(X_train_final.columns)

        self.best_pipeline.fit(X_train_final[self.selected_features], y_train_final)
        print(f"OK Final model refit on {len(X_train_final)} rows")
        return self

    def evaluate_on_test(self):
        """Finalna evaluacija na TEST skupu (koristi se samo jednom!)."""
        print("\n=== EVALUATING ON TEST SET ===")
        if self.best_pipeline is None:
            raise RuntimeError('Nema istreniranog modela.')

        if self.selected_features is None:
            raise RuntimeError('Nema odabranih features.')

        # Predikcija na test skupu
        y_pred = self.best_pipeline.predict(self.X_test[self.selected_features])

        # Metrике
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        medae = median_absolute_error(self.y_test, y_pred)
        evs = explained_variance_score(self.y_test, y_pred)

        self.test_metrics = {
            'model_name': self.best_model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'MedAE': medae,
            'EVS': evs,
            'test_samples': len(self.y_test),
        }

        print(f"\n--- {self.best_model_name} ---")
        print(f"Test MAE:    {mae:,.2f} €")
        print(f"Test RMSE:   {rmse:,.2f} €")
        print(f"Test R²:     {r2:.4f}")
        print(f"Test MAPE:   {mape:.2f} %")
        print(f"Test MedAE:  {medae:,.2f} €")
        print(f"Test EVS:    {evs:.4f}")
        print(f"Test samples: {len(self.y_test)}")

        return self

    def save_best_model(self):
        """Čuva model, metrike i metadata."""
        print("\n=== SAVING MODEL ARTIFACTS ===")
        if self.best_pipeline is None:
            raise RuntimeError('Nema istreniranog modela za čuvanje.')

        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        # Čuvaj model
        joblib.dump(self.best_pipeline, models_dir / 'best_model.joblib')
        print('OK Model saved: models/best_model.joblib')

        # Izvuci feature names iz preprocessor
        try:
            if self.use_log_target:
                preprocessor = self.best_pipeline.regressor_.named_steps['preprocessor']
            else:
                preprocessor = self.best_pipeline.named_steps['preprocessor']
            feature_names_out = preprocessor.get_feature_names_out().tolist()
        except:
            feature_names_out = None

        # Čuvaj metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_params': self.best_params,
            'selected_features': self.selected_features,
            'feature_names_out': feature_names_out,
            'reference_year': self.reference_year,
            'validation_results': pd.DataFrame(self.results),
            'feature_selection_summary': self.feature_selection_summary,
            'test_metrics': self.test_metrics,
        }
        joblib.dump(metadata, models_dir / 'training_metadata.joblib')
        print('OK Metadata saved: models/training_metadata.joblib')

        # Kreiraj results/metrics folder
        metrics_dir = Path('results/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Čuvaj CSV metrike u results/metrics/
        results_df = pd.DataFrame(self.results).sort_values('val_mae', ascending=True)
        results_df.to_csv(metrics_dir / 'model_results.csv', index=False)
        print(f'✅ Model validation results saved: {metrics_dir / "model_results.csv"}')
        
        if self.feature_importance_summary is not None:
            self.feature_importance_summary.to_csv(metrics_dir / 'feature_importance.csv', index=False)
            print(f'✅ Feature importance saved: {metrics_dir / "feature_importance.csv"}')
        
        if self.feature_selection_summary is not None:
            self.feature_selection_summary.to_csv(metrics_dir / 'feature_selection_comparison.csv', index=False)
            print(f'✅ Feature selection comparison saved: {metrics_dir / "feature_selection_comparison.csv"}')

        # Čuvaj test metrike kao JSON
        if self.test_metrics:
            metrics_json = {k: v for k, v in self.test_metrics.items() if k != 'model_name'}
            with open(models_dir / 'test_metrics.json', 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print('OK Test metrics saved: models/test_metrics.json')

        return self.best_pipeline

    def show_results(self):
        """Prikaži finalne rezultate."""
        results_df = pd.DataFrame(self.results).sort_values(by='val_mae', ascending=True)
        print("\n=== RESULTS SUMMARY (sorted by validation MAE) ===")
        print(results_df[['model', 'val_mae', 'val_rmse', 'val_r2']].to_string(index=False))

        if self.feature_selection_summary is not None:
            print("\n=== FEATURE SET COMPARISON ===")
            print(self.feature_selection_summary.to_string(index=False))
        
        if self.test_metrics:
            print("\n=== TEST SET EVALUATION ===")
            print(f"Model: {self.test_metrics['model_name']}")
            print(f"MAE:  {self.test_metrics['MAE']:,.2f} €")
            print(f"RMSE: {self.test_metrics['RMSE']:,.2f} €")
            print(f"R²:   {self.test_metrics['R2']:.4f}")
        
        return self


if __name__ == "__main__":
    # Demo
    print("Koristi src.train kao modul u main.py")
