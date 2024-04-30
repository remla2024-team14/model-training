from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from os.path import join
import json

import numpy as np

TOKENIZED_PATH = "outputs/tokenized"
MODEL_PATH = "outputs/model.h5"


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
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': 5000,
              'batch_test': 5000,
              'categories': ['phishing', 'legitimate'],
              'char_index': None,
              'epoch': 30,
              'embedding_dimension': 50,
              'dataset_dir': "../outputs/tokenized/"}

    return params


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
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                     batch_size=params['batch_train'],
                     epochs=params['epoch'],
                     shuffle=True,
                     validation_data=(x_val, y_val)
                     )

    return model


def main():
    params = define_params()
    x_train, y_train, x_val, y_val, char_index = load_tokenized_data()
    model = define_model(params, char_index)

    model = train_model(model, params, x_train, y_train, x_val, y_val)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
