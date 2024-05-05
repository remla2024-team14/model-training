# URL Fishing - Team 14

## About The Project

This is a project for course Release Engineering for ML Applications (CS4295) at Delft University of Technology.

## Getting Started

### Installation

1. Clone the repository

```
git clone https://github.com/nadinekuo/URL-Fishing-CS4295.git
```

2. Setup (and activate) your environment

```python
# using pip

pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt

# TODO: update with Poetry later
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

Run the experiment using `dvc exp run`. See the difference by running `dvc metrics diff`.

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


## Contributors

- Dani Rogmans
- Justin Luu
- Yang Li
- Nadine Kuo
