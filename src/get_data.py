import json
import os
from os.path import join
import boto3
from dotenv import load_dotenv

from config_reader import ConfigReader

directories = ConfigReader().params["directories"]
BASE_DIR, OUTPUTS_DIR = directories["base_dir"], directories["raw_outputs_dir"]
REMOTE = ConfigReader().params["remote_download"]

load_dotenv()

if REMOTE:
    s3 = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    s3.download_file('url-phishing-data', 'train.txt', 'data/train.txt')
    s3.download_file('url-phishing-data', 'test.txt', 'data/test.txt')
    s3.download_file('url-phishing-data', 'val.txt', 'data/val.txt')

train_dir, test_dir, val_dir = join(BASE_DIR, "train.txt"), join(BASE_DIR, "test.txt"), join(BASE_DIR, "val.txt")

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
