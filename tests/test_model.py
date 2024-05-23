import pytest
from tensorflow.keras.models import load_model
from lib_ml.preprocessing import TextPreprocessor
import time


@pytest.fixture
def model():

    model = load_model('models/model.keras')

    return model


def test_model_latency(model):
    input_url = "aoiwjdao.com"
    preprocessor = TextPreprocessor()

    preprocessor.fit_text(input_url)
    processed_texts = preprocessor.transform_text(input_url)

    start_time = time.time()

    model.predict(processed_texts)

    end_time = time.time()

    expected_time = 0.5
    assert (end_time - start_time) < expected_time
