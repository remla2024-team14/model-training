from os.path import join

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config_reader import ConfigReader

directories = ConfigReader().params["directories"]
MODEL_PATH, TOKENIZED_PATH = directories["model_path"], directories["tokenized_outputs_dir"]

def load_variable(filename):
    file_path = join(TOKENIZED_PATH, filename + ".txt")
    return np.loadtxt(file_path)


def load_test_dataset():
    x_test = load_variable("x_test")
    y_test = load_variable("y_test")

    return x_test, y_test


def load_tf_model():
    model = load_model(MODEL_PATH)
    return model


def predict(model, x_test, y_test):
    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))

def main():
    model = load_tf_model()
    x_test, y_test = load_test_dataset()

    predict(model, x_test, y_test)

if __name__ == "__main__":
    main()