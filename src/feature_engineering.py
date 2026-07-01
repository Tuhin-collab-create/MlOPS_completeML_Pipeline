import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

import yaml


# Logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, "feature-engineering.log")

logger = logging.getLogger("feature-engineering")
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
    
# Adding params.yaml file to read the parameters
def load_params(params_path: str):
    """Loading parameters from a YAML file."""
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
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

def load_data(file_path: str):
    """Load data from interim subfolder."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def apply_tfidf(train,test,max_features):
    """
    applying Tfidf vectorizer to the text column"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train = train['text'].values
        x_test = test['text'].values
        y_train = train['target'].values
        y_test = test['target'].values
        
        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)   
        
        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['target'] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['target'] = y_test
        logger.info("TF-IDF vectorization applied successfully.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str):
    """
    Save the DataFrame to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise
    
def main():
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_path, "data", "interim", "train_processed.csv")
        test_path = os.path.join(base_path, "data", "interim", "test_processed.csv")
        train = load_data(train_path)
        test = load_data(test_path)
        train_df, test_df = apply_tfidf(train, test, max_features)
        save_data(train_df, os.path.join(base_path, "data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join(base_path, "data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()