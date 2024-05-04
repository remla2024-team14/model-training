from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.metrics import Precision, Recall
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

def load_variable(filename):
    file_path = join(TOKENIZED_PATH, filename + ".txt")
    return np.loadtxt(file_path)


def load_tokenized_data():
    x_train = load_variable("x_train")
    y_train = load_variable("y_train")
    x_val = load_variable("x_val")
    y_val = load_variable("y_val")

    with open(join(TOKENIZED_PATH, "char_index.txt"), 'r') as file:
        data = file.read()

    char_index = json.loads(data)

    return x_train, y_train, x_val, y_val, char_index


def define_params():
    return ConfigReader().params["model_params"]


def define_model(params, char_index):
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
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy', Precision(), Recall()])

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
    params = define_params()
    x_train, y_train, x_val, y_val, char_index = load_tokenized_data()
    model = define_model(params, char_index)

    model = train_model(model, params, x_train, y_train, x_val, y_val)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
