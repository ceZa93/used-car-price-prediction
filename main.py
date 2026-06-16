"""
GLAVNA SKRIPTA ZA POKRETANJE CELOG PIPELINE-A
"""
import logging
import sys

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # 1. Load Data
    loader = DataLoader()
    df = loader.load()

    # 2. Preprocess
    processor = DataPreprocessor(df)
    df_clean = processor.clean_horsepower() \
                        .handle_missing_values() \
                        .remove_outliers() \
                        .encode_categorical() \
                        .get_data()

    # 3. Feature Engineering
    engineer = FeatureEngineer(df_clean)
    df_final = engineer.create_features() \
                       .drop_unnecessary_columns() \
                       .get_data()

    # 4. Model Training
    trainer = ModelTrainer(df_final)
    trainer.prepare_data() \
           .train_models() \
           .show_results()

    # 5. Save model
    best_model = trainer.save_best_model()

    # 6. Evaluation
    print("\n" + "=" * 50)
    print("STARTING MODEL EVALUATION")
    print("=" * 50)

    evaluator = ModelEvaluator(best_model, trainer.X_test, trainer.y_test)
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline prekinut od strane korisnika.")
        sys.exit(130)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
