import os
import logging
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Global objects
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ---------------- Logging ----------------

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, "data_preprocessing.log")

logger = logging.getLogger("Data_preprocessing")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M %p"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(file_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------- Text Cleaning ----------------

def transform_text(text):

    text = str(text).lower()

    words = nltk.word_tokenize(text)

    words = [word for word in words if word.isalnum()]

    words = [
        word for word in words
        if word not in stop_words and word not in string.punctuation
    ]

    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


# ---------------- Preprocessing ----------------

def preprocess_df(df, text='text', target='target'):
    try:
        logger.debug("Preprocessing started")

        df[target] = df[target].map({'ham': 0, 'spam': 1})
        logger.debug("Target encoded")

        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")

        df[text] = df[text].apply(transform_text)
        logger.debug("Text transformed")

        return df

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


# ---------------- Main ----------------

def main(text='text', target='target'):

    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        logger.debug("Data loaded successfully")

        train_preprocessed = preprocess_df(train_data, text, target)
        test_preprocessed = preprocess_df(test_data, text, target)

        data_path = "./data/interim"
        os.makedirs(data_path, exist_ok=True)

        train_preprocessed.to_csv(
            os.path.join(data_path, "train_processed.csv"),
            index=False
        )

        test_preprocessed.to_csv(
            os.path.join(data_path, "test_processed.csv"),
            index=False
        )

        logger.debug("Processed data saved successfully")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)

    except pd.errors.EmptyDataError as e:
        logger.error("Empty file: %s", e)

    except Exception as e:
        logger.error("Failed to complete preprocessing: %s", e)


if __name__ == "__main__":
    main()