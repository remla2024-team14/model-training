"""Module containing global helper utility functions"""

import numpy as np
from config_reader import ConfigReader
from os.path import join

directories = ConfigReader().params["directories"]
TOKENIZED_PATH = directories["tokenized_outputs_dir"]


def load_variable(filename):
    """
    Load a variable from a text file.

    Args:
        filename (str): The name of the file to load from (without extension).

    Returns:
        numpy.ndarray: The variable loaded from the file.
    """
    file_path = join(TOKENIZED_PATH, filename + ".txt")
    return np.loadtxt(file_path)
