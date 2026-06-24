"""
GLAVNA SKRIPTA ZA POKRETANJE CELOG ML PIPELINE-A

Redosled izvršavanja:
  1. DataPreparation  → Učitaj i očisti podatke
  2. DatasetExplorer  → EDA analiza (pre treniranja)
  3. ModelTrainer     → Treniranje, selekcija modela, finalni test
  4. ModelEvaluator   → Grafici i detaljna evaluacija
"""
from pathlib import Path

from src.data_preparation import DataPreparation
from src.evaluate import DatasetExplorer, ModelEvaluator
from src.train import ModelTrainer


if __name__ == "__main__":
    print("\n" + "="*70)
    print("USED CAR PRICE PREDICTION - KOMPLETAN ML PIPELINE")
    print("="*70)

    # ================================================================
    # 1. PRIPREMA PODATAKA
    # ================================================================
    print("\n" + "="*70)
    print("FAZA 1: PRIPREMA PODATAKA")
    print("="*70)
    
    preparation = DataPreparation(data_path='data/raw/serbia_car_sales_price_2024_v2.csv')
    preparation.load().show_info()
    preparation.prepare()
    
    df_clean = preparation.get_data()
    
    # Spremi sample čistih podataka
    from pathlib import Path
    results_dir = Path('results/figures')
    results_dir.mkdir(parents=True, exist_ok=True)
    df_clean.head(100).to_csv(results_dir / 'cleaned_data_sample.csv', index=False)
    print(f"\nOK Saved cleaned data sample: results/figures/cleaned_data_sample.csv")

    # ================================================================
    # 2. EKSPLORATIVNA ANALIZA (pre treniranja)
    # ================================================================
    print("\n" + "="*70)
    print("FAZA 2: EKSPLORATIVNA ANALIZA PODATAKA (EDA)")
    print("="*70)
    
    explorer = DatasetExplorer(df_clean, output_dir='results/figures')
    explorer.run_all()

    # ================================================================
    # 3. TRENIRANJE I SELEKCIJA MODELA
    # ================================================================
    print("\n" + "="*70)
    print("FAZA 3: TRENIRANJE I SELEKCIJA MODELA")
    print("="*70)
    
    trainer = ModelTrainer(df_clean)
    trainer.prepare_data() \
           .train_models() \
           .compare_feature_sets(top_n=12) \
           .refit_final_model() \
           .evaluate_on_test() \
           .show_results()

    # Čuvanje modela i metadata
    trainer.save_best_model()

    # ================================================================
    # 4. DETALJNO EVALUACIJA I GRAFICI
    # ================================================================
    print("\n" + "="*70)
    print("FAZA 4: EVALUACIJA MODELA - GRAFICI I ANALIZE")
    print("="*70)

    # Učitaj feature names iz metadata
    import joblib
    metadata = joblib.load('models/training_metadata.joblib')
    feature_names_out = metadata.get('feature_names_out', None)
    
    # Dobij skailirane vrednosti za test skup
    # Prvo fit prepreocesor na train skupu (da ne dođe do data leakage)
    if trainer.use_log_target:
        preprocessor = trainer.best_pipeline.regressor_.named_steps['preprocessor']
    else:
        preprocessor = trainer.best_pipeline.named_steps['preprocessor']
    
    # Prvo feature engineering na test skupu
    fe = trainer.best_pipeline.regressor_.named_steps['feature_engineering'] if trainer.use_log_target else trainer.best_pipeline.named_steps['feature_engineering']
    X_test_fe = fe.transform(trainer.X_test[trainer.selected_features])
    
    # Skailiraj sa preprocessor-om
    X_test_scaled = preprocessor.transform(X_test_fe)
    
    evaluator = ModelEvaluator(
        trainer.best_pipeline,
        trainer.X_test[trainer.selected_features],
        trainer.y_test,
        output_dir='results/figures',
        feature_names_out=feature_names_out,
        X_test_scaled=X_test_scaled  # Prosleđi skailirane vrednosti!
    )
    evaluator.run_all_evaluations()
    evaluator.save_processed_data_sample()

    # ================================================================
    # ZAVRŠETAK
    # ================================================================
    print("\n" + "="*70)
    print("PIPELINE KOMPLETIRAN USPEŠNO!")
    print("="*70)
    print("\nIzlazni fajlovi:")
    print("  ✓ models/best_model.joblib - Sačuvan finalni model")
    print("  ✓ models/training_metadata.joblib - Metadata i feature list")
    print("  ✓ models/model_results.csv - Poređenje svih modela")
    print("  ✓ models/feature_importance.csv - Permutation importance")
    print("  ✓ models/test_metrics.json - Finalne metrike")
    print("  ✓ results/figures/ - EDA i evaluacijski grafici")
    print("\nPokretanje Streamlit aplikacije:")
    print("  $ streamlit run app/ui.py")
    print("="*70 + "\n")