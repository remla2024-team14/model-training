# URL Fishing - Team 14

## About The Project

This is a project for course Release Engineering for ML Applications (CS4295) at Delft University of Technology.

## Getting Started

### Installation

1. Clone the repository

```
https://github.com/remla2024-team14/model-training.git
```

2. Setup (and activate) your environment

```python
# using pip

pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

### How To: Dependency Management with Poetry

In your virtual environment, run `pip install pipx` followed by `python -m pipx install poetry`. Then, run `python -m pipx ensurepath` and restart your terminal/IDE.

After re-opening your terminal or IDE, you should now be able to run poetry. Test this by simply writing the command `poetry`. Then ensure poetry is up to date by running `pip install --upgrade poetry`.

To install the defined dependencies for this project, run:

```
poetry install
```

### How To: Run DVC Pipeline

To run the DVC pipeline (as configured in `dvc.yaml`), firstly make sure you have DVC installed in your working environment.

To run the pipeline, simply use the command `dvc repro`.

Check the [DVC documentation](https://dvc.org/doc/start) for further details and additional possibilities.

### How To: DVC Remotes

If you would like to remotely download the data, you need an *AWS access key ID* and an *AWS secret access key*, which you should add in a local `.env` file with the following format:

```
AWS_ACCESS_KEY_ID=<aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<aws_secret_access_key>
AWS_BUCKET_NAME=<bucket_name>
```

If you want to setup a remote run:

```
dvc remote add -d myremote s3://<bucket>/<key>
```

Next, add:

```
dvc remote modify --local <myremote-name> access_key_id '<aws_access_key_id>'
dvc remote modify --local <myremote-name> secret_access_key '<aws_secret_access_key>'
```

You can push artefacts to the remote by running `dvc push`. Similarly, pulling from the remote can be performed by running `dvc pull`.

## How To: DVC Experiment Management

In this project, DVC is also used to report metrics and keep track of different experiments/models.

Run the experiment using `dvc exp run`. See the difference by running `dvc metrics diff`.Please install the latest version of dvc locally`pip install -U dvc`,otherwise it may lead to version incompatibility and other problems.

Whenever anything is changed in the project, a new experiment can be run and the experiment log can be checked using `dvc exp show`.

All metrics will be generated to an output file named `metrics.json`.

## Code Quality

This project uses the following linters to display code quality information:

- Pylint
- Flake8

NOTE: we obtained perfect scores for both Pylint and Flake8.

### Pylint

To run Pylint on a specific file, use `pylint src/<file_name>` or `pylint src/` to analyse the full directory.
It should output something as:

```
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```

In Pylint's configuration file `pylintrc`, we have thoroughly analysed linter rules and made the following modifications to adapt it to this specific ML project:

- We allow the following commonly used variable names in ML: `X_train`,`Y_train`,
  `X_test` and
  `Y_test`
- To discourage non-informative variable names, we defined a set of bad names: `bad-names=foo, baz, toto, tutu, tata, x, y, z, var, vars`
- We extend the list of exceptions that will emit a warning with `ArithmeticError`, `BufferError` and `LookupError` - especially common in ML projects
- We ignore files that are either auto-generated or do not contain Python code: `ignore=CVS, .git, __pycache__, build, dist, .gitignore, requirements.txt, config.json`
- We only show warnings with high confidence levels and those that lead to inference errors (`confidence=HIGH, INFERENCE_FAILURE`)

### Flake8

To analyse our Python code using Flake8, we run `flake8 --max-line-length 100`. This will configure the maximum allowed line length to 100 (in line with Pylint), instead of the 88 which is the default.

### PyTest for ML Testing

#### Test Setup

When using PyTest, passing arguments to methods require `@pytest.mark.parametrize` and `pytest.fixture`.
PyTest will run the test for each parameter.

Since our dataset does not contain features, we made two features `no_char` which is the length of the URL and `segments` which is length of the path of the URL.

#### Running Tests

We can use PyTest-monitor to check the memory usage.
*Note: `pytest-monitor` updates its SQLite database incrementally, so delete monitor.db file between test runs.*

To run the tests run `pytest --db ./monitor.db` or `pytest` if you do not want the monitor.db which is used to view things such as memory usage.

To view memory usage of each test in terminal run `sqlite3 ./monitor.db`, then `.headers on`, then `.mode column` and at last `select ITEM, MEM_USAGE from TEST_METRICS ORDER BY MEM_USAGE DESC LIMIT 10;`

The plots for the features and data are stored in outputs/plots. 

#### Test Adequacy

To inspect test coverage, run `pytest --cov=TEST_DIR`. It will generate something like:

```
---------- coverage: platform darwin, python 3.11.6-final-0 ----------
Name                                                       Stmts   Miss  Cover
------------------------------------------------------------------------------
tests/infra_integration_tests/test_config_reader.py           12     1    58%
tests/infra_integration_tests/test_define_train_model.py      11     0    78%
tests/infra_integration_tests/test_get_data.py                18     0    94%
tests/infra_integration_tests/test_predict.py                 12     0    87%
tests/test_features_data.py                                  105     1    65%
tests/test_monitoring.py                                      28     0    86%
------------------------------------------------------------------------------
TOTAL                                                        186    131    78%
```

- `Stmts`: The total number of statements in the package.
- `Miss`: The number of statements that were not executed during testing.
- `Cover`: The percentage of statements that were executed during testing.


## Contributors

- Dani Rogmans
- Justin Luu
- Yang Li
- Nadine Kuo
