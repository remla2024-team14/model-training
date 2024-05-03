import json


class ConfigReader:

    def __init__(self, file_path="config.json"):
        with open(file_path) as param_file:
            self.params = json.load(param_file)
