"""Module for defining model architecture and training the model"""

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.metrics import Precision, Recall
from utils import load_variable
import os
from os.path import join
import json

import numpy as np

from config_reader import ConfigReader

directories = ConfigReader().params["directories"]

TOKENIZED_PATH = directories["tokenized_outputs_dir"]
MODEL_PATH = directories["model_path"]
METRICS_PATH = join(directories["metrics_file"])

metrics_dir = os.path.dirname(METRICS_PATH)
os.makedirs(metrics_dir, exist_ok=True)


def load_tokenized_data():
    """
    Load tokenized data and character index from files.

    Returns:
        tuple: A tuple containing tokenized data and character index:
            - x_train (numpy.ndarray): Tokenized and padded sequences of input training data.
            - y_train (numpy.ndarray): Encoded labels for training data.
            - x_val (numpy.ndarray): Tokenized and padded sequences of input validation data.
            - y_val (numpy.ndarray): Encoded labels for validation data.
            - char_index (dict): A dictionary mapping characters to their indices.
    """
    x_train = load_variable("x_train")
    y_train = load_variable("y_train")
    x_val = load_variable("x_val")
    y_val = load_variable("y_val")

    with open(join(TOKENIZED_PATH, "char_index.txt"), 'r') as file:
        data = file.read()

    char_index = json.loads(data)

    return x_train, y_train, x_val, y_val, char_index


def define_params():
    """
    Define model parameters by reading them from a configuration file.

    Returns:
        dict: A dictionary containing model parameters.
    """
    return ConfigReader().params["model_params"]


def define_model(params, char_index):
    """
    Define the architecture of the model as CNN with dropout layers for model regularization.

    Args:
        params (dict): A dictionary containing model parameters.
        char_index (dict): A dictionary mapping characters to their indices.

    Returns:
        keras.models.Sequential: The defined model.
    """
    model = Sequential()

    voc_size = len(char_index.keys())
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories']) - 1, activation='sigmoid'))

    return model


def train_model(model, params, x_train, y_train, x_val, y_val):
    """Train the defined model.

    Args:
        model (keras.models.Sequential): The model to train.
        params (dict): A dictionary containing model parameters.
        x_train (numpy.ndarray): Tokenized and padded sequences of input training data.
        y_train (numpy.ndarray): Encoded labels for training data.
        x_val (numpy.ndarray): Tokenized and padded sequences of input validation data.
        y_val (numpy.ndarray): Encoded labels for validation data.

    Returns:
        keras.models.Sequential: The trained model.
    """
    model.compile(loss=params['loss_function'],
                  optimizer=params['optimizer'], metrics=['accuracy', Precision(), Recall()])

    hist = model.fit(x_train, y_train,
                     batch_size=params['batch_train'],
                     epochs=params['epoch'],
                     shuffle=True,
                     validation_data=(x_val, y_val))

    metrics = {
        "accuracy": hist.history['accuracy'][-1],
        "val_accuracy": hist.history['val_accuracy'][-1],
        "precision": hist.history['precision'][-1],
        "val_precision": hist.history['val_precision'][-1],
        "recall": hist.history['recall'][-1],
        "val_recall": hist.history['val_recall'][-1],
        "loss": hist.history['loss'][-1],
        "val_loss": hist.history['val_loss'][-1]
    }

    with open(METRICS_PATH, 'w') as json_file:
        json.dump(metrics, json_file)

    return model


def main():
    """
    Main function to define, train, and save the model.
    """
    params = define_params()
    x_train, y_train, x_val, y_val, char_index = load_tokenized_data()
    model = define_model(params, char_index)

    model = train_model(model, params, x_train, y_train, x_val, y_val)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
