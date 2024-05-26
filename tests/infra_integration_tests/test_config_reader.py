import unittest
import json
from src.config_reader import ConfigReader


# Integration test
class TestConfigReader(unittest.TestCase):
    def setUp(self):
        self.config_reader = ConfigReader().params
        with open("config.json") as param_file:
            self.test_json = json.load(param_file)

    def test_config_reader(self):
        self.assertEqual(self.config_reader, self.test_json)


if __name__ == "__main__":
    unittest.main()
