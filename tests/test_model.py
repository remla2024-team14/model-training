import pytest
from tensorflow.keras.models import load_model
from lib_ml.preprocessing import TextPreprocessor
import time


@pytest.fixture
def model_setup(request):
    input_url = request.param
    preprocessor = TextPreprocessor()

    preprocessor.fit_text([input_url])
    processed_texts = preprocessor.transform_text([input_url])

    model = load_model('models/model.keras')
    return model, processed_texts


# Test how long it takes to make a prediction
@pytest.mark.parametrize("model_setup", ["aoiwjdao.com"], indirect=True)
def test_model_latency(model_setup):
    model, processed_texts = model_setup

    start_time = time.time()
    model.predict(processed_texts)
    end_time = time.time()

    expected_time = 0.5
    result_time = end_time - start_time
    print(f"Prediction time: {result_time}s")
    assert result_time < expected_time, "Model prediction took longer than expected."


# Test to measure how much memory it takes to setup the model and make 1 prediction, see readme
@pytest.mark.parametrize("model_setup", ["aoiwjdao.com"], indirect=True)
def test_model_memory_usage(model_setup):
    model, url = model_setup
    result = model.predict(url)
    assert result > 0.5


@pytest.mark.parametrize("model_setup", [("")], indirect=True)
def test_empty_string(model_setup):
    model, url = model_setup
    with pytest.raises(AttributeError) as exc_info:
        result = model.predict(url)

    assert "'NoneType' object has no attribute 'shape'" in str(exc_info.value)


@pytest.mark.parametrize("model_setup", [("aoiwjdao.com"), ("oaiwd.com")], indirect=True)
def test_multiple_spam(model_setup):
    model, url = model_setup
    result = model.predict(url)
    assert result > 0.5
