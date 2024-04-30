import json
from os.path import join

FILE_DIR = "outputs"

with open(join(FILE_DIR, "train.txt"), 'r') as filehandle:
    train_array = json.load(filehandle)

with open(join(FILE_DIR, "test.txt"), 'r') as filehandle:
    test_array = json.load(filehandle)

with open(join(FILE_DIR, "val.txt"), 'r') as filehandle:
    val_array = json.load(filehandle)