"""Module for defining model architecture and training the model"""
# import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.metrics import Precision, Recall
from utils import load_variable
import os
from os.path import join
import json
from config_reader import ConfigReader
from sklearn.preprocessing import LabelEncoder
import logging
import pickle
from lib_ml.preprocessing import TextPreprocessor

directories = ConfigReader().params["directories"]

TOKENIZED_PATH = directories["tokenized_outputs_dir"]
MODEL_PATH = directories["model_path"]
METRICS_PATH = join(directories["metrics_file"])

metrics_dir = os.path.dirname(METRICS_PATH)
os.makedirs(metrics_dir, exist_ok=True)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w',
                    encoding='utf-8')


def load_data():
    try:
        raw_x_train = load_variable("raw_x_train.txt")
        raw_y_train = load_variable("raw_y_train.txt")
        raw_x_val = load_variable("raw_x_val.txt")
        raw_y_val = load_variable("raw_y_val.txt")

        config = {
            'lower': True,
            'char_level': True,
            'oov_token': '-n-',
            'sequence_length': 200
        }

        preprocessor = TextPreprocessor(config)
        combined_texts = [str(text) for text in raw_x_train + raw_x_val]
        preprocessor.fit_text(combined_texts)

        x_train = preprocessor.transform_text(raw_x_train)
        x_val = preprocessor.transform_text(raw_x_val)
        char_index = preprocessor.tokenizer.word_index

        encoder = LabelEncoder()
        all_labels = raw_y_train + raw_y_val
        encoder.fit(all_labels)
        y_train = encoder.transform(raw_y_train)
        y_val = encoder.transform(raw_y_val)

        return x_train, y_train, x_val, y_val, char_index, preprocessor
    except Exception as e:
        logging.error(f"Failed to load or process data: {e}")
        raise


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


def train_model(model, params, x_train, y_train, x_val, y_val, preprocessor):
    model.compile(loss=params['loss_function'],
                  optimizer=params['optimizer'], metrics=['accuracy', Precision(), Recall()])

    hist = model.fit(x_train, y_train,
                     batch_size=params['batch_train'],
                     epochs=params['epoch'],
                     shuffle=True,
                     validation_data=(x_val, y_val))

    tokenizer_path = 'outputs/tokenizer.pkl'
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(preprocessor.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    x_train, y_train, x_val, y_val, char_index, preprocessor = load_data()
    model = define_model(params, char_index)

    model = train_model(model, params, x_train, y_train, x_val, y_val, preprocessor)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
