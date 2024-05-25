import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from sklearn.model_selection import train_test_split
from src.define_train_model import train_model, define_params, define_model, load_data


def test_model_on_data_slices():
    x_train, y_train, x_val, y_val, char_index, preprocessor = load_data()
    params = define_params()

    # print(f"x_train size: {len(x_train)}, y_train size: {len(y_train)}")
    # print(f"x_val size: {len(x_val)}, y_val size: {len(y_val)}")

    # if len(x_train) < 20 or len(x_val) < 20:
    #    raise ValueError("Not enough data to perform train/test split. Please ensure the dataset is large enough.")

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    model = define_model(params, char_index)
    model = train_model(model, params, x_train, y_train, x_val, y_val, preprocessor)

    x_slice, _, y_slice, _ = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

    overall_score = model.evaluate(x_val, y_val)
    slice_score = model.evaluate(x_slice, y_slice)

    assert np.allclose(overall_score, slice_score, atol=0.1), "Model performance on data slice differs significantly from overall performance."


if __name__ == "__main__":
    test_model_on_data_slices()
