import unittest
from src.predict import load_test_dataset, load_tf_model, predict


# Integration test
class TestPredict(unittest.TestCase):

    def test_predict(self):
        model = load_tf_model()
        x_test, y_test = load_test_dataset('data/val.txt')
        binary_predictions = predict(model, x_test, y_test)
        self.assertTrue(x_test is not None)
        self.assertTrue(y_test is not None)
        self.assertEqual(binary_predictions.shape[0], y_test.shape[0])


if __name__ == "__main__":
    unittest.main()
