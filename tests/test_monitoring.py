# Test to measure how much memory it takes to setup the model and make 1 prediction, see readme
import pytest
import time

from tensorflow.keras.models import load_model
from lib_ml import TextPreprocessor


@pytest.fixture
def model_setup(request):
    input_url = request.param
    preprocessor = TextPreprocessor()

    preprocessor.fit_text([input_url])
    processed_texts = preprocessor.transform_text([input_url])

    model = load_model('models/model.h5')
    return model, processed_texts


@pytest.mark.parametrize("model_setup", ["aoiwjdao.com"], indirect=True)
def test_model_memory_usage(model_setup):
    model, url = model_setup
    result = model.predict(url)
    assert result > 0.5


@pytest.mark.parametrize("model_setup", ["aoiwjdao.com"], indirect=True)
def test_model_latency(model_setup):
    model, processed_texts = model_setup

    start_time = time.time()

    model.predict(processed_texts)

    end_time = time.time()

    expected_time = 0.5
    assert (end_time - start_time) < expected_time
    result_time = end_time - start_time
    print(f"Prediction time: {result_time}s")
    assert result_time < expected_time, "Model prediction took longer than expected."
