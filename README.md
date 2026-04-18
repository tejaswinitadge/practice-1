# Fraud Detection Project

A production-ready pipeline for detecting credit card fraud using various machine learning techniques including cost-sensitive learning and SMOTE-Tomek oversampling.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `creditcard.csv` is located in the data directory specified in `src/config.py` (which currently defaults to the parent folder's location).

## Running the Pipeline

Execute the pipeline using:
```bash
python scripts/run_pipeline.py
```

This will:
- Load the dataset.
- Train several models (Baselines, Cost-Sensitive, SMOTE+Tomek) using stratified K-fold cross-validation.
- Select the best model based on Precision-Recall AUC (PR-AUC).
- Train the best model on the complete training set.
- Evaluate it on the test set.
- Save the trained model to `models/` dir and logs to `logs/` dir.
