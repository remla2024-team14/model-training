from urllib.parse import urlparse
from botocore.exceptions import ClientError
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from lib_ml.preprocessing import TextPreprocessor
import boto3
import os
import pytest
import pandas as pd
import seaborn as sns


@pytest.fixture
def model_setup(request):
    input_url = request.param
    preprocessor = TextPreprocessor()

    preprocessor.fit_text([input_url])
    processed_texts = preprocessor.transform_text([input_url])

    model = load_model('models/model.h5')
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


# Creates features for our dataset to show how testing features could be done
@pytest.fixture()
def create_df():
    def read_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data1 = file.read().splitlines()
        return data1

    urls = read_data("data/train.txt")

    data = {'class': [], 'url': []}
    for entry in urls:
        parts = entry.split('\t', 1)
        if len(parts) == 2:
            class_label, url = parts
            data['class'].append(class_label)
            data['url'].append(url.strip())

    df = pd.DataFrame(data)

    def count_segments(segmented_url):
        parsed_url = urlparse(segmented_url)
        path = parsed_url.path
        return path.count('/') if path else 0

    # Add new features
    df['no_char'] = df['url'].apply(len)
    df['segments'] = df['url'].apply(count_segments)

    return df


output_dir = './outputs/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def plot_kde(df, x_col, hue_col, title, xlabel, ylabel, filename, fill=True, alpha=0.5, palette="viridis", linewidth=2):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=x_col, hue=hue_col, fill=fill, alpha=alpha, palette=palette, linewidth=linewidth)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()


# Create plots of the features and save them in outputs/plots
def test_no_char_vs_no_segments(create_df):
    df = create_df
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='no_char', y='segments', hue='class', style='class', palette='viridis', s=100)
    plt.title('Correlation between URL Length and Number of Segments')
    plt.xlabel('Number of Characters (no_char)')
    plt.ylabel('Number of Segments (segments)')
    plt.legend(title='URL Class')
    plt.grid(True)
    plt.savefig(f'{output_dir}/no_char_vs_segments.png')
    plt.close()


def test_no_char(create_df):
    df = create_df
    plot_kde(df, 'no_char', 'class', 'Distribution of URL Character Count by Class', 'Number of Characters (no_char)',
             'Density', 'no_char_distribution.png')


def test_segments(create_df):
    df = create_df
    plot_kde(df, 'segments', 'class', 'Distribution of URL Segment Count by Class', 'Number of Segments', 'Density',
             'segment_distribution.png')


def test_data_distribution(create_df):
    df = create_df
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 4))
    ax = sns.countplot(data=df, x='class', palette='viridis')
    plt.title('Distribution of Phishing and Legitimate URLs')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    total = float(len(df))

    for p in ax.patches:
        height = p.get_height()

        percentage = '{:.1f}%'.format(100 * height / total)

        ax.text(p.get_x() + p.get_width() / 2., height + 3, percentage, ha="center")

    plt.savefig(f'{output_dir}/class_distribution.png')
    plt.close()
