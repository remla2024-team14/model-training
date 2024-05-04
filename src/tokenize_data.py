from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

import json
import os
from os.path import join

from config_reader import ConfigReader

directories = ConfigReader().params["directories"]

INPUTS_DIR, OUTPUTS_DIR = directories["raw_outputs_dir"], directories["tokenized_outputs_dir"]
FILE_NAMES = ["x_train", "y_train", "x_test", "y_test", "x_val", "y_val"]


def _load_data(file_dir):
    vars = {}

    for filename in FILE_NAMES:
        filename = "raw_" + filename
        with open(join(file_dir, filename + ".txt"), 'r') as filehandle:
            vars[filename] = json.load(filehandle)

    return vars["raw_x_train"], vars["raw_x_val"], vars["raw_y_train"], vars["raw_y_val"], vars["raw_x_test"], vars[
        "raw_y_test"]


def tokenize(dataset_input_dir):
    raw_x_train, raw_x_val, raw_y_train, raw_y_val, raw_x_test, raw_y_test = _load_data(dataset_input_dir)

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, char_index


def main():
    # Ensure input and output directories exist
    os.makedirs(INPUTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test, char_index = tokenize(INPUTS_DIR)

    save_variable(x_train, "x_train")
    save_variable(y_train, "y_train")
    save_variable(x_val, "x_val")
    save_variable(y_val, "y_val")
    save_variable(x_test, "x_test")
    save_variable(y_test, "y_test")
    with open(join(OUTPUTS_DIR, "char_index.txt"), 'w') as file:
        json.dump(char_index, file)


def save_variable(variable, filename):
    np.savetxt(join(OUTPUTS_DIR, filename + ".txt"), variable, fmt='%d')


if __name__ == "__main__":
    main()
