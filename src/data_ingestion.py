import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, "data_ingestion.log")

logger = logging.getLogger("Data_ingestion")
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
        



# Data Loader
def load_data(data_url:str):
    try:
        df = pd.read_csv(data_url, encoding='latin-1') 
        logger.debug('Data Loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.debug('Failed to parse during loadng the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise

# preprocessing
def preprocess_data(df):
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)
        df.rename(columns ={'v1':'target','v2':'text'},inplace = True)
        logger.debug('data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing. %s',e)
        raise

#Saving train and test data
def save_data(train_data,test_data,data_path):
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index = False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index = False)
        logger.debug("train and test data is saved to,%s",raw_data_path)
    except Exception as e:
        logger.error('unexpected error occured while saving the data, %s',e)
    
def main():
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/Tuhin-collab-create/spam_data/main/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion, %s',e)
        print(f'Error: {e}')

if __name__ == "__main__":
    main()