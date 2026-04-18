import os

# Project root is the parent directory of the `src` folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.dirname(PROJECT_ROOT)  # parent of fraud_detection_prod

DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 3
EVAL_THRESHOLD = 0.3

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
