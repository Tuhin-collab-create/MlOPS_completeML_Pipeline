import os
import pandas as pd
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

# Logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, "model_training.log")

logger = logging.getLogger("model_training")
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

def load_params(params_path: str):
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info(f"Parameters loaded successfully from {params_path}")
            return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {params_path}: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {params_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters from {params_path}: {e}")
        raise

    
# loading data for model training
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
    
def train_model(X_train, y_train,params):
    """
    Train a Random Forest model and save it to disk.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train do not match.")
        logger.info("Starting model training with params: %s",params)
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=params["random_state"])
        logger.info("Model training started with samples %d samples",X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return clf
    except ValueError as e:
        logger.error(f"ValueError during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
def save_model(model, file_path: str):
    """
    Save model to disk.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {file_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found error while saving model to {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise

def main():
    try:
        params = load_params("params.yaml")['model_training']
        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.drop(columns=['target']).values
        y_train = train_data['target'].values
        clf = train_model(X_train, y_train, params)
        save_model(clf, "models/model.pkl")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        raise 

if __name__ == "__main__":
    main()