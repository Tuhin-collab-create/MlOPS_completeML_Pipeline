import os
import numpy as np
import pandas as pd
import logging
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, "model_evaluation.log")

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %I:%M %p")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(file_path, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
def load_model(model_path: str):
    """Load a trained model from disk."""
    try:
        with open(model_path,'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {model_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error occurred while loading model from {model_path}: {e}")
        raise

def load_data(file_path: str):
    """Load data from processed subfolder."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
    
def evaluate_model(clf, X_test, y_test):
    """Evaluating the model """
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] 
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        matrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
        logger.info(f"Model evaluation metrics calculated --> {matrics_dict}")
        return matrics_dict
    except ValueError as e:
        logger.error(f"ValueError during model evaluation: {e}")
        raise   
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    
def save_metrics(metrics:dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved successfully to {file_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found error while saving metrics to {file_path}: {e}")
        raise
    
def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        X_test = test_data.drop(columns=['target']).values
        y_test = test_data['target'].values
        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()