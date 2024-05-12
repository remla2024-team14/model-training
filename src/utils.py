from config_reader import ConfigReader
from os.path import join, dirname, realpath
# import os

directories = ConfigReader().params["directories"]
TOKENIZED_PATH = directories["tokenized_outputs_dir"]
RAW_DATA_PATH = directories["raw_outputs_dir"]


def load_variable(filename):
    base_dir = dirname(realpath(__file__))  # 获取当前文件的绝对路径
    file_path = join(base_dir, 'outputs', 'raw', filename)
    # file_path = join(RAW_DATA_PATH, filename)
    # print(f"Attempting to load file at: {file_path}")  # Debugging line
    # data = pd.read_csv(file_path, header=None)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data
