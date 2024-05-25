import unittest
from src.predict import load_test_dataset, load_tf_model, predict


class TestPredict(unittest.TestCase):

    def test_predict(self):
        model = load_tf_model()
        x_test, y_test = load_test_dataset('data/val.txt')
        binary_predictions = predict(model, x_test, y_test)
        self.assertTrue(x_test is not None)
        self.assertTrue(y_test is not None)


if __name__ == "__main__":
    unittest.main()
