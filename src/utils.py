from config_reader import ConfigReader
from os.path import join
import os

print("Current Working Directory:", os.getcwd())
directories = ConfigReader().params["directories"]
TOKENIZED_PATH = directories["tokenized_outputs_dir"]
# RAW_DATA_PATH = directories["raw_outputs_dir"]


def load_variable(filename):
    base_path = "/home/runner/work/model-training/model-training/"
    file_path = os.path.join(base_path, "outputs/raw", filename)
    # file_path = join(RAW_DATA_PATH, filename)
    # data = pd.read_csv(file_path, header=None)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data
