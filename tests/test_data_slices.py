import numpy as np
from sklearn.model_selection import train_test_split
from src.define_train_model import train_model, define_params, define_model, load_data


# This test belongs under the category "Model Development"
def test_model_on_data_slices():
    x_train, y_train, x_val, y_val, char_index, preprocessor = load_data()
    params = define_params()

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    model = define_model(params, char_index)
    model = train_model(model, params, x_train, y_train, x_val, y_val, preprocessor)

    x_slice, _, y_slice, _ = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

    overall_score = model.evaluate(x_val, y_val)
    slice_score = model.evaluate(x_slice, y_slice)

    assert np.allclose(overall_score, slice_score, atol=0.1), \
        "Model performance on data slice differs significantly from overall performance."


if __name__ == "__main__":
    test_model_on_data_slices()
