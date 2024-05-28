from src.config_reader import ConfigReader
from os.path import join
import re

directories = ConfigReader().params["directories"]
TOKENIZED_PATH = directories["tokenized_outputs_dir"]
RAW_DATA_PATH = directories["raw_outputs_dir"]


def load_variable(filename):
    file_path = join(RAW_DATA_PATH, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        data = re.findall(r'https?://[^\s,]+', content)
    return data
