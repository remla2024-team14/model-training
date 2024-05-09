"""Module for tokenizing raw train and test datasets"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

import json
import os
from os.path import join
from lib_ml.preprocessing import TextPreprocessor
from config_reader import ConfigReader

directories = ConfigReader().params["directories"]

INPUTS_DIR, OUTPUTS_DIR = directories["raw_outputs_dir"], directories["tokenized_outputs_dir"]
FILE_NAMES = ["x_train", "y_train", "x_test", "y_test", "x_val", "y_val"]


def _load_data(file_dir):
    """Load raw data (see FILE_NAMES) to be tokenized into dictionary.

    Args:
        file_dir (str): The directory path containing the raw train, test and val datasets.

    Returns:
        dict: A dictionary where keys are the FILE_NAMES (with 'raw_' prefix added) and
            values are the contents of the corresponding text files loaded as JSON.
    """
    data_dict = {}

    for filename in FILE_NAMES:
        filename = "raw_" + filename
        with open(join(file_dir, filename + ".txt"), 'r') as filehandle:
            data_dict[filename] = json.load(filehandle)

    return data_dict


def tokenize(dataset_input_dir):
    """Tokenize the raw text data from the given directory.

    Args:
        dataset_input_dir (str): The directory path containing the raw datasets.

    Returns:
        tuple: A tuple containing the tokenized data and associated information:
            - x_train (numpy.ndarray): Tokenized and padded sequences of input training data.
            - y_train (numpy.ndarray): Encoded labels for training data.
            - x_val (numpy.ndarray): Tokenized and padded sequences of input validation data.
            - y_val (numpy.ndarray): Encoded labels for validation data.
            - x_test (numpy.ndarray): Tokenized and padded sequences of input test data.
            - y_test (numpy.ndarray): Encoded labels for test data.
            - char_index (dict): A dictionary mapping characters to their indices.
    """
    data_dict = _load_data(dataset_input_dir)
    raw_x_train, raw_y_train = data_dict['raw_x_train'], data_dict['raw_y_train']
    raw_x_val, raw_y_val = data_dict['raw_x_val'], data_dict['raw_y_val']
    raw_x_test, raw_y_test = data_dict['raw_x_test'], data_dict['raw_y_test']

    config = {
        'lower': True,
        'char_level': True,
        'oov_token': '-n-',
        'sequence_length': 200
    }
    preprocessor = TextPreprocessor(config)
    preprocessor.fit_text(raw_x_train + raw_x_val + raw_x_test)
    char_index = preprocessor.tokenizer.word_index

    # Tokenize and pad sequences
    x_train = preprocessor.transform_text(raw_x_train)
    x_val = preprocessor.transform_text(raw_x_val)
    x_test = preprocessor.transform_text(raw_x_test)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, char_index


def main():
    """
    Main function for saving tokenized data.

    This function performs the following steps:
    1. Ensures input and output directories exist.
    2. Tokenizes input data using TextPreprocessor.
    3. Saves tokenized data and associated information.
    """

    os.makedirs(INPUTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test, char_index = tokenize(INPUTS_DIR)

    save_variable(x_train, "x_train")
    save_variable(y_train, "y_train")
    save_variable(x_val, "x_val")
    save_variable(y_val, "y_val")
    save_variable(x_test, "x_test")
    save_variable(y_test, "y_test")

    with open(join(OUTPUTS_DIR, "char_index.json"), 'w') as file:
        json.dump(char_index, file, ensure_ascii=False)


def save_variable(variable, filename):

    path = join(OUTPUTS_DIR, filename + ".txt")
    if isinstance(variable, np.ndarray):
        np.savetxt(path, variable, fmt='%d')
    else:
        # For non-NumPy arrays, save in json or other formats
        with open(path, 'w') as f:
            json.dump(variable, f)

if __name__ == "__main__":
    main()
