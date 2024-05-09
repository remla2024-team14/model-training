
"""Module for evaluating the trained model on test set"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config_reader import ConfigReader
from utils import load_variable
import logging
from lib_ml.preprocessing import TextPreprocessor
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

directories = ConfigReader().params["directories"]
MODEL_PATH = directories["model_path"]
model = load_model(MODEL_PATH)
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

directories = ConfigReader().params["directories"]
MODEL_PATH, TOKENIZED_PATH = directories["model_path"], directories["tokenized_outputs_dir"]


def load_test_dataset():
    try:
        # Assume raw_x_test and raw_y_test are the correct raw data filenames
        x_test = load_variable("raw_x_test.txt")
        y_test = load_variable("raw_y_test.txt")

        # Apply preprocessing if necessary
        # Assuming that TextPreprocessor has methods like fit_text and transform_text
        preprocessor = TextPreprocessor(config={'lower': True, 'char_level': True, 'oov_token': '-n-', 'sequence_length': 200})
        preprocessor.fit_text(x_test)  # Only if you need to fit the tokenizer
        x_test = preprocessor.transform_text(x_test)

        # Assuming LabelEncoder usage if labels are categorical
        encoder = LabelEncoder()
        y_test = encoder.fit_transform(y_test)  # Ensure labels are fitted correctly or pre-fitted

        return x_test, y_test
    except Exception as e:
        logging.error(f"Failed to load or process test datasets: {str(e)}")
        raise


def load_tf_model():
    """
    Load the trained TensorFlow model.

    Returns:
        keras.models.Model: The loaded TensorFlow model.
    """
    model = load_model(MODEL_PATH)
    return model


def predict(model, x_test, y_test):
    y_pred = model.predict(x_test, batch_size=1000)
    print("Raw predictions:", y_pred)

    # Consider using a different threshold or method to determine class labels
    threshold = 0.5  # Adjust based on analysis
    y_pred_binary = (y_pred > threshold).astype(int)
    y_test = y_test.reshape(-1, 1)

    print('Classification Report:')
    try:
        report = classification_report(y_test, y_pred_binary, zero_division=1)
        print(report)
    except ValueError as e:
        print("Error in generating report:", e)

    print('Confusion Matrix:')
    try:
        confusion_mat = confusion_matrix(y_test, y_pred_binary)
        print(confusion_mat)
        print('Accuracy:', accuracy_score(y_test, y_pred_binary))
    except ValueError as e:
        print("Error in generating confusion matrix:", e)

def main():
    """
    Load the trained model, test dataset, and make predictions.
    """
    model = load_tf_model()
    x_test, y_test = load_test_dataset()

    predict(model, x_test, y_test)


if __name__ == "__main__":
    main()
