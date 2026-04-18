import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and performs basic integrity checks.
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Drop duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    if initial_shape != final_shape:
        logger.info(f"Dropped {initial_shape[0] - final_shape[0]} duplicate rows. New shape: {df.shape}")

    return df
