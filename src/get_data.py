"""Module for loading raw train and test datasets, possibly from remote"""

import json
import os
from os.path import join
import boto3
import urllib

from src.config_reader import ConfigReader

directories = ConfigReader().params["directories"]
BASE_DIR, OUTPUTS_DIR = directories["base_dir"], directories["raw_outputs_dir"]
REMOTE = ConfigReader().params["remote_download"]


def fetch_data_publicly():
    files_and_urls = {
        'train.txt': 'https://team14awsbucket.s3.amazonaws.com/train.txt',
        'test.txt': 'https://team14awsbucket.s3.amazonaws.com/test.txt',
        'val.txt': 'https://team14awsbucket.s3.amazonaws.com/val.txt'
    }

    # Ensure the data directory exists
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # Download data from public URLs
    for local_file, url in files_and_urls.items():
        local_path = os.path.join(BASE_DIR, local_file)
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {url} to {local_path}")


if REMOTE:
    fetch_data_publicly()

train_dir = join(BASE_DIR, "train.txt")
test_dir = join(BASE_DIR, "test.txt")
val_dir = join(BASE_DIR, "val.txt")

train = [line.strip() for line in open(train_dir, "r").readlines()[1:]]
raw_x_train = [line.split("\t")[1] for line in train]
raw_y_train = [line.split("\t")[0] for line in train]

test = [line.strip() for line in open(test_dir, "r").readlines()]
raw_x_test = [line.split("\t")[1] for line in test]
raw_y_test = [line.split("\t")[0] for line in test]

val = [line.strip() for line in open(val_dir, "r").readlines()]
raw_x_val = [line.split("\t")[1] for line in val]
raw_y_val = [line.split("\t")[0] for line in val]

OUTPUTS_DIR = 'outputs/raw'

# check directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

for partition in ["train", "test", "val"]:
    file_path_x = join(OUTPUTS_DIR, f"raw_x_{partition}.txt")
    file_path_y = join(OUTPUTS_DIR, f"raw_y_{partition}.txt")

    with open(file_path_x, 'w') as filehandle:
        json.dump(globals()[f"raw_x_{partition}"], filehandle)

    with open(file_path_y, 'w') as filehandle:
        json.dump(globals()[f"raw_y_{partition}"], filehandle)


def get_data():
    return raw_x_train, raw_y_train, raw_x_val, raw_y_val, raw_x_test, raw_y_test
