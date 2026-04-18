import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def split_data(df: pd.DataFrame, target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataframe into training and testing sets.
    """
    logger.info(f"Splitting data into features (X) and target ({target_col})")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    logger.info(f"Original class distribution:\n{y.value_counts(normalize=True)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
