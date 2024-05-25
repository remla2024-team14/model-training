import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from src.define_train_model import train_model, define_params, define_model, load_data

def test_mutamorphic_behavior():
    x_train, y_train, x_val, y_val, char_index, preprocessor = load_data()
    params = define_params()
    model = define_model(params, char_index)
    model = train_model(model, params, x_train, y_train, x_val, y_val, preprocessor)

    # Create a validation dataset with slight perturbations
    noise_factor = 0.1
    x_val_perturbed = x_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val.shape)
    x_val_perturbed = np.clip(x_val_perturbed, 0, 1)

    # Evaluating model performance on raw and perturbed validation data
    original_score = model.evaluate(x_val, y_val)
    perturbed_score = model.evaluate(x_val_perturbed, y_val)

    print(f"Original score: {original_score}")
    print(f"Perturbed score: {perturbed_score}")

    assert np.allclose(original_score, perturbed_score, atol=0.1), "Model performance changes significantly on perturbed data."

if __name__ == "__main__":
    test_mutamorphic_behavior()
