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
        data_path = 'https://raw.githubusercontent.com/Tuhin-collab-create/spam_data/main/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=0.20,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion, %s',e)
        print(f'Error: {e}')

if __name__ == "__main__":
    main()