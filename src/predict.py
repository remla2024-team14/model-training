import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config_reader import ConfigReader
from utils import load_variable
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lib_ml.preprocessing import TextPreprocessor
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pickle

directories = ConfigReader().params["directories"]
MODEL_PATH = directories["model_path"]

def load_test_dataset(file_path):
    with open('outputs/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    urls = [line.strip().split('\t')[1] for line in lines if '\t' in line]
    labels = [line.strip().split('\t')[0] for line in lines if '\t' in line]

    preprocessor = TextPreprocessor(
        config={'lower': True, 'char_level': True, 'oov_token': '-n-', 'sequence_length': 200})
    preprocessor.tokenizer = tokenizer
    x_test = preprocessor.transform_text(urls)

    if x_test is not None and len(x_test) > 0:
        encoder = LabelEncoder()
        y_test = encoder.fit_transform(labels)
        return x_test, y_test
    else:
        logging.error("No valid data to process.")
        return None, None
def load_tf_model():
    model = load_model(MODEL_PATH)
    return model
def predict(model, x_test, y_test):
    if x_test is not None and y_test is not None:
        y_pred = model.predict(x_test, batch_size=1000)
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_test = y_test.reshape(-1, 1)

        print('Classification Report:')
        print(classification_report(y_test, y_pred_binary, zero_division=1))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred_binary))
        print('Accuracy:', accuracy_score(y_test, y_pred_binary))
    else:
        print("No predictions to make, as input data is not valid.")

def main():
    model = load_tf_model()
    x_test, y_test = load_test_dataset('data/val.txt')
    if x_test is not None and y_test is not None:
        predict(model, x_test, y_test)
    else:
        print("Failed to load or process test data.")

if __name__ == "__main__":
    main()