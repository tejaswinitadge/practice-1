import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from typing import Any
import os
from src.config import PROJECT_ROOT, EVAL_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_evaluation_artifacts(model: Any, model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Generates confusion matrix and precision-recall curve, and performs threshold tuning.
    Saves plots to the artifacts directory.
    """
    artifact_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    y_test_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_test_score = model.predict_proba(X_test)[:, 1]
    else:
        y_test_score = model.decision_function(X_test)

    # 1. Print Final Classification Report
    logger.info("===== FINAL TEST RESULTS =====")
    report = classification_report(y_test, y_test_pred)
    logger.info(f"\n{report}")
    logger.info(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_score):.4f}")
    logger.info(f"Test PR-AUC: {average_precision_score(y_test, y_test_score):.4f}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Test Set\n({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

    # 3. Precision-Recall Curve Plot
    precision, recall, _ = precision_recall_curve(y_test, y_test_score)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve\n({model_name})")
    pr_path = os.path.join(artifact_dir, "precision_recall_curve.png")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    logger.info(f"Saved precision-recall curve plot to {pr_path}")

    # 4. Threshold tuning
    y_threshold_pred = (y_test_score >= EVAL_THRESHOLD).astype(int)
    logger.info(f"===== Threshold = {EVAL_THRESHOLD} =====")
    thresh_report = classification_report(y_test, y_threshold_pred)
    logger.info(f"\n{thresh_report}")

