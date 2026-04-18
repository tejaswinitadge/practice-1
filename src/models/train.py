import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from typing import Dict, Tuple, Any
from src.config import CV_SPLITS, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model_cv(
    model: Any, 
    model_name: str, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates a single model using Stratified K-Fold CV on train data and evaluates on test data.
    """
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    cv_precision, cv_recall, cv_f1, cv_pr_auc = [], [], [], []

    # Cross validation loop
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        
        cv_precision.append(report["1"]["precision"])
        cv_recall.append(report["1"]["recall"])
        cv_f1.append(report["1"]["f1-score"])
        
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)[:, 1]
        else:
            y_score = model.decision_function(X_val)
            
        cv_pr_auc.append(average_precision_score(y_val, y_score))

    # Evaluate on test set (requires training on full X_train)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    if hasattr(model, "predict_proba"):
        y_test_score = model.predict_proba(X_test)[:, 1]
    else:
        y_test_score = model.decision_function(X_test)

    test_roc_auc = roc_auc_score(y_test, y_test_score)
    test_pr_auc = average_precision_score(y_test, y_test_score)

    result_dict = {
        "Model": model_name,
        "CV Precision": np.mean(cv_precision),
        "CV Recall": np.mean(cv_recall),
        "CV F1": np.mean(cv_f1),
        "CV PR-AUC": np.mean(cv_pr_auc) if cv_pr_auc else None,
        "Test Precision": test_report["1"]["precision"],
        "Test Recall": test_report["1"]["recall"],
        "Test F1": test_report["1"]["f1-score"],
        "Test ROC-AUC": test_roc_auc,
        "Test PR-AUC": test_pr_auc
    }
    logger.info(f"Finished evaluating {model_name} | CV PR-AUC: {result_dict['CV PR-AUC']:.4f}")
    return result_dict


def find_best_model(
    models: Dict[str, Any], 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Any, str, pd.DataFrame]:
    """
    Evaluates multiple models and selects the best one based on CV PR-AUC.
    """
    results = []
    logger.info(f"Starting evaluation of {len(models)} models...")

    for model_name, model in models.items():
        res = evaluate_model_cv(model, model_name, X_train, y_train, X_test, y_test)
        results.append(res)
    
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="CV PR-AUC", ascending=False)
    
    logger.info(f"Model Comparison:\n{results_df_sorted.to_string()}")
    
    best_model_name = results_df_sorted.iloc[0]["Model"]
    logger.info(f"Best Model Based on CV PR-AUC: {best_model_name}")
    
    return models[best_model_name], best_model_name, results_df_sorted
