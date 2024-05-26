import unittest
from src.define_train_model import load_data, define_model, define_params


# Integration test
class TestDefineTrainModel(unittest.TestCase):

    def test_define_train_model(self):
        x_train, y_train, x_val, y_val, char_index, preprocessor = load_data()
        params = define_params()

        model = define_model(params, char_index)
        output_layer = model.layers[-1]

        self.assertEqual(x_train.shape, x_val.shape)
        self.assertEqual(y_train.shape, y_val.shape)
        self.assertTrue(output_layer.output_shape[-1] == 1)


if __name__ == "__main__":
    unittest.main()
