# URL-Fishing-CS4295
Project for course Release Engineering for ML Applications (CS4295) at Delft University of Technology

If you would like to remotely download the data, you need an AWS access key ID and an AWS secret access key, which you should add in a local .env file with the following format:
You should also set the bucket name:

AWS_ACCESS_KEY_ID=aws_access_key_id
AWS_SECRET_ACCESS_KEY=aws_secret_access_key
AWS_BUCKET_NAME=bucket_name


Please make sure to install the requirements.

To run the dvc pipeline make sure you have dvc installed.
To run the pipeline use:
dvc repro

DVC remote:

If you want to setup a remote:
dvc remote add -d myremote s3://<bucket>/<key>

then add:

dvc remote modify --local myremote access_key_id 'aws_access_key_id'
dvc remote modify --local myremote secret_access_key 'aws_secret_access_key'

you can push/pull by running:

- dvc push
- dvc pull