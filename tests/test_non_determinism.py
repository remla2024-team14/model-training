import numpy as np
import tensorflow as tf
from src.define_train_model import train_model, load_data, define_params, define_model


# This test belongs under the category "Model Development"
def test_non_deterministic_behavior():
    def train_and_evaluate(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x_train, y_train, x_val, y_val, char_index, preprocessor = load_data()

        # Ensure data size matches before training
        min_train_length = min(len(x_train), len(y_train))
        x_train = x_train[:min_train_length]
        y_train = y_train[:min_train_length]
        min_val_length = min(len(x_val), len(y_val))
        x_val = x_val[:min_val_length]
        y_val = y_val[:min_val_length]

        params = define_params()
        model = define_model(params, char_index)
        model = train_model(model, params, x_train, y_train, x_val, y_val, preprocessor)
        return model.evaluate(x_val, y_val)

    seed_1 = train_and_evaluate(1)
    seed_2 = train_and_evaluate(2)

    assert np.allclose(seed_1, seed_2, atol=0.1), "Model performance varies significantly with different random seeds."


if __name__ == "__main__":
    test_non_deterministic_behavior()
