import unittest
from src.get_data import fetch_data_remotely
import os
import shutil


# Integration test
class TestFetchData(unittest.TestCase):
    def setUp(self, folder_path="data"):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been deleted.")
        else:
            print(f"Folder '{folder_path}' does not exist.")

    def test_fetch_data_remotely(self):
        fetch_data_remotely()
        self.assertTrue(os.path.exists("data"))
        self.assertTrue(os.path.isfile("data/train.txt"))
        self.assertTrue(os.path.isfile("data/val.txt"))
        self.assertTrue(os.path.isfile("data/test.txt"))


if __name__ == "__main__":
    unittest.main()
