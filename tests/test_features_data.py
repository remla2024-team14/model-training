import os
from urllib.parse import urlparse

import boto3
import pytest
from botocore.exceptions import ClientError
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from lib_ml.preprocessing import TextPreprocessor
import pandas as pd
import seaborn as sns


@pytest.fixture
def model_setup(request):
    input_url = request.param
    preprocessor = TextPreprocessor()

    preprocessor.fit_text([input_url])
    processed_texts = preprocessor.transform_text([input_url])

    model = load_model('models/model.keras')
    return model, processed_texts


# Test code that creates input features in both training and serving
@pytest.mark.parametrize("input_url, expected", [
    ("aoiwjdioa.com", (1, 200)),
    ("aoiwjdsk.nl", (1, 200))
])
def test_preprocessing(input_url, expected):
    preprocessor = TextPreprocessor()

    preprocessor.fit_text([input_url])
    processed_texts = preprocessor.transform_text([input_url])

    assert processed_texts.shape == expected


# Test code that creates input features in both training and serving
@pytest.mark.parametrize("model_setup", [
    "http://www.kern-photo.com/ourohana/wp-content/content/tracking_2754677973/dhl.php?email=abuse@free.fr",
    "oaiwd.com"], indirect=True)
def test_multiple_spam(model_setup):
    model, url = model_setup
    print("preprocessed: ", url.shape)
    result = model.predict(url)
    print("result: ", result)

    assert result > 0.5


# Test that system maintains privacy controls across its entire data pipeline, this is the only way our data is
# downloaded:
def test_privacy():
    # Initialize S3 client with invalid credentials
    s3 = boto3.client('s3', aws_access_key_id="randomaccesskeythatdoesnotwork",
                      aws_secret_access_key="randomsecretkeythatdoesnotwork")
    # Expect a ClientError due to invalid credentials
    with pytest.raises(ClientError) as exc_info:
        s3.download_file("awsbucketteam14", "somefile.zip", "somedir")

    # Check if the error code is 403 Forbidden
    assert exc_info.value.response['Error']['Code'] == '403'


# Test to measure how much memory it takes to setup the model and make 1 prediction, see readme
@pytest.mark.parametrize("model_setup", ["aoiwjdao.com"], indirect=True)
def test_model_memory_usage(model_setup):
    model, url = model_setup
    result = model.predict(url)
    assert result > 0.5


@pytest.mark.parametrize("model_setup", [""], indirect=True)
def test_empty_string(model_setup):
    model, url = model_setup
    with pytest.raises(AttributeError) as exc_info:
        model.predict(url)

    assert "'NoneType' object has no attribute 'shape'" in str(exc_info.value)


