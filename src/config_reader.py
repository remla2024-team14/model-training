"""Module providing a ConfigReader for loading in relevant model training and evaluation parameters"""

import json


class ConfigReader:
    """ 
    A class to read in configuration parameters, including paths to use for outputting models, metrics etc.
    """
    def __init__(self, file_path="config.json"):
        """
        Initialize the ConfigReader with the path to the JSON configuration file.

        Args:
            file_path (str, optional): The path to the JSON configuration file.
                Defaults to "config.json".
        """
        with open(file_path) as param_file:
            self.params = json.load(param_file)
