import sys
import os
import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.config import DATA_PATH, MODEL_SAVE_DIR
from src.utils.logger import get_logger
from src.data.data_loader import load_data
from src.data.preprocessor import split_data
from src.models.model_factory import get_models
from src.models.train import find_best_model
from src.models.evaluate import generate_evaluation_artifacts

logger = get_logger(__name__)

def main():
    logger.info("Starting Fraud Detection Pipeline")

    try:
        # 1. Load Data
        df = load_data(DATA_PATH)
        
        # 2. Split Data
        X_train, X_test, y_train, y_test = split_data(df)
        
        # 3. Get Models
        models = get_models(y_train)
        
        # 4. Train, Evaluate CV, and Find Best Model
        best_model, best_model_name, results_df = find_best_model(
            models, X_train, y_train, X_test, y_test
        )
        
        # 5. Save best model
        model_path = os.path.join(MODEL_SAVE_DIR, f"{best_model_name.replace(' ', '_')}.joblib")
        joblib.dump(best_model, model_path)
        logger.info(f"Saved the best model to {model_path}")
        
        # 6. Generate final evaluation artifacts on test set
        generate_evaluation_artifacts(best_model, best_model_name, X_test, y_test)
        
        logger.info("Pipeline executed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
