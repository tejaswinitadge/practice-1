from typing import Dict
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# Use imblearn pipeline for SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
import pandas as pd

def get_models(y_train: pd.Series) -> Dict[str, SklearnPipeline]:
    """
    Returns a dictionary of all models evaluated in the project.
    
    Args:
        y_train: Training target variable, needed to compute scale_pos_weight for XGBoost.
        
    Returns:
        A dictionary mapping model names to instantiated Pipelines.
    """
    models = {}

    # Calculate scale factor for XGBoost
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # 1. Baselines
    models["Baseline Logistic Regression"] = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="saga", max_iter=200, tol=1e-2, n_jobs=-1, random_state=42))
    ])

    models["Baseline Random Forest"] = SklearnPipeline([
        ("model", RandomForestClassifier(n_estimators=30, max_depth=6, min_samples_split=10, 
                                         min_samples_leaf=5, n_jobs=-1, random_state=42))
    ])

    models["Baseline XGBoost"] = SklearnPipeline([
        ("model", XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, 
                                tree_method="hist", n_jobs=-1, random_state=42, eval_metric="logloss"))
    ])

    models["Baseline SVM"] = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(max_iter=1000, random_state=42))
    ])

    # 2. Cost-Sensitive
    models["Cost-Sensitive Logistic Regression"] = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="saga", max_iter=200, tol=1e-2, class_weight="balanced", 
                                     n_jobs=-1, random_state=42))
    ])

    models["Cost-Sensitive Random Forest"] = SklearnPipeline([
        ("model", RandomForestClassifier(n_estimators=30, max_depth=6, class_weight="balanced", 
                                         n_jobs=-1, random_state=42))
    ])

    models["Cost-Sensitive SVM"] = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    models["Cost-Sensitive XGBoost"] = SklearnPipeline([
        ("model", XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, subsample=0.8, 
                                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, 
                                tree_method="hist", n_jobs=-1, random_state=42, eval_metric="logloss"))
    ])

    # 3. SMOTE + Tomek
    models["SMOTE + Tomek + LR"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("resample", SMOTETomek(random_state=42)),  
        ("model", LogisticRegression(solver="liblinear", max_iter=2000, random_state=42))
    ])

    models["SMOTE + Tomek + RF"] = ImbPipeline([
        ("resample", SMOTETomek(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42))
    ])

    models["SMOTE + Tomek + SVM"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("resample", SMOTETomek(random_state=42)),
        ("model", LinearSVC(max_iter=2000, random_state=42))
    ])

    models["SMOTE + Tomek + XGB"] = ImbPipeline([
        ("resample", SMOTETomek(random_state=42)),
        ("model", XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, 
                                n_jobs=-1, random_state=42, eval_metric="logloss"))
    ])

    return models
