# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Conventional Commits](https://www.conventionalcommits.org/) with [Commitizen](https://github.com/commitizen/cz-cli).

## v0.3.0 (2025-10-21)

### Feat

- **powerbi**: correct background in reports
- **powerbi**: add backgrounds report
- **powerbi**: add powerbi operational report
- **continuous_prediction**: add custom model prediction
- **batch_prediction**: add hourly/daily batch inference
- **data_processor**: add possibility between training and production inference
- **dab**: improve tags in DAB jobs
- **config**: add parameter batch_inference_table
- **dab**: change cluster id and automatic fallback on defined cluster for dev, acc, prd
- **monitoring**: add notebook for monitoring
- **dab**: add job for monitoring
- **dab**: add name for monitoring table
- **model_serving**: do not raise error if timeout
- **databricks_utils**: update module and test to new version of cli > 0.20
- **taskfile**: adapt task target for streamlit
- **streamlit**: add monitoring to merge with regular app
- **pyproject**: add optional ui packages for streamlit
- **pyproject**: add optional ui packages for streamlit
- add streamlit app for monitoring
- add monitoring of endpoint with Lakehouse
- add monitoring table name in modelserving
- **pyproject**: change databricks cli version from 0.18 to >0.2
- **model_serving**: add AI Gateway configuration to enable lakehouse monitoring
- **streamlit**: remove custom package
- replace DL of mlflow model by calling the serving endpoint
- **streamlit**: add configuration for DAB for streamlit app
- **streamlit**: add streamlit for demo
- **monitoring**: add module monitoring from course code hub
- **bundle**: add job for training and retraining of custom model
- add deploy_custom_model_serving script
- **basic_model**: add conversion to pandas dataframe if input in predict is a spark dataframe
- **custom_model**: add custom model in modelling
- add curl request from command line with automatically get the token
- add config for custom model
- add config for custom model
- add an initial and weekly training in databricks jobs
- **taskfile**: add target run-process-new-data
- **process_data**: add ability to overwrite/append/upsert data in Unity Catalog
- **dab**: add post commit status to databricks.yml
- **dab**: add deploy serving task in databricks.yml
- **dab**: add train_and_register step
- **dab**: initiate databricks bundle with step 1 to preprocess new data
- **data_processor**: add synthetic data generator
- **config**: add endpoint_name and endpoint_fe_name to global config of the project
- **deploy_model**: remove unused arguments version_model in testing the API call
- add get_databricks_token and is_databricks to utils
- add script to deploy endpoint on databricks
- add model serving module
- **feature_engineering**: improve train and test script for fe
- improve feature_lookup_model module with model name for fe
- **tests**: simplify task command for test
- **config**: add model_name for fe
- add script to test register fe model
- **taskfile**: add target fe_train_register_model for feature engineering
- update uv lock file
- add unit test to marvelous package locally
- **config**: add feature_table_name, feature_function_name and experiment_name_fe for feature engineering
- add script for feature engineering and training model
- add feature_engineering.py module for feature lookup and feature pyfunc
- **pyproject**: add databricks-feature-engineering and databricks-feature-lookup packages
- **data_processor**: keep Booking_ID in columns
- **config**: add feature_table_name and feature_function_name
- add feature_engineering.py module for feature lookup and feature pyfunc
- **data_processor**: only remove columns and drop dupplicates in data processing
- improve list of extensions for VSCode

### Fix

- **dab**: change inference app name to hotel-resa-caotrido because of bug no more than 30 characters app name
- **dab**: correct app name to avoid dupplication name and add suffix of environment
- **custom_model**: change prediction name form Cancelled to Canceled
- **continuous_prediction**: correct endpoint name
- **dab**: switch to pause on schedule on dev, acc et prd
- **test**: correct tests with parameter batch_inference_table added
- **cd**: add  --force-lock due to failed cd previous
- **bundle**: correct cluster id in prd
- **cd**: correct contents permissions from read to write for tagging
- **databricks_utils**: revert back databricks-cli from 0.2 to 0.18
- **custom_model**: correct error in registering whl in jobs
- **databricks_utils**: connection to databricks connect with new version
- solved merge conflict
- **cd**: correct cd to main and acc and prd environment
- **cd**: deploy on only acc + use AWS credentials
- **basic_model**: correct unit test
- **basic_model**: correct the retrieval of the latest model
- correct custom serving call for custom models endpoint
- **serving**: custom model, correct artifact for serving and add uv build for whl to be loaded
- correct get_token if exist in .env for basic model serving call
- **model_serving**: change timeout of waiting deployment from 600 to 1200 sec (case in free edition much longer)
- **deploy_serving_model**: correct get token in .env if exist and deploy without aws credentials if does not exist
- correct predic_custom_model for homogeneity in the code
- remove model_improved = True
- **config**: change filename to Hotel_Reservations.csv
- correct label assignation 0 and 1 and get latest model
- **data_processor**: correct unit test errors
- **data_processor**: correct save to catalog
- **basic_model**: return latest_version when register_model
- **model_serving**: improve wait_until_ready if endpoint does not exist, continue
- **synthetic_data**: align data types in script for generating data
- **synthetic_data**: align data types in script for generating data
- **notebooks**: add branch for local execution in process_data
- **pyproject**: correct librarie name to hotel_reservation* with * to include all submodules
- **feature_engineering**: correct with get temporary databricks token
- add serving as module with init.py
- **taskfile**: clean env in dev-install to avoid collision with test
- **pyproject**: correct package name include = ["mlops_course"]
- script for fe are functionnal on databricks
- correct prediction in fe to string value
- **config**: correct model name
- **pyproject**: fix version of packages for databricks
- correct bugs in feature_lookup_modell.py module
- **basic_model**: add possibility if no model exist to upload directly

### Refactor

- **powerbi**: switch to import mode & refactor calculation
- **dab**: rename continous_prediction.yml file
- **dab**: reorganize in simple .yml file in resources
- **visualization**: delete old module
- clean up init and rename module visualization
- rename test_register_fe_model into predict_register_model script
- rename basic model script and module as basic_model_xxx
- **taskfile**: rename run tasks for homogeneity
- **serving_model**: move scripts to functional test and add target in taskfile
- **notebooks**: move utilities notebooks to utils/ subfolder for clarity
- **marvelous**: delete marvelous legacy module to move it to utils/ module
- **scripts**: move actual scripts to notebook to prepare scripts for DAB
- **deploy_model**: separate serving and testing endpoint and refactor function in module
- **model_serving**: transfer wait_until_ready function to the method class
- **test**: put unit test in tests/unit_test
- clean train_register_fe_model.py script from unused code
- **taskfile**: refactor task to not delete venv
- **package**: rename package from mlops_course to hotel_reservation
- add vscode settings to git tracking
- **scripts**: switch env with argparse argument

## v0.2.0 (2025-10-06)

### Feat

- adapt script to ipython or cli execution
- **basic_model**: add comparison perf of old and new model
- add task for creating experiment workspace and train_register models
- **wiki**: add wiki project documentation
- add script to create and delete experiment for script train_and_register_model.py
- add scripts to train, log experiment and register model in UC
- add basic_model.py module from course
- **uv**: update uv lock file
- **taskfile**: add command to display databricks config
- **pyproject**: add boto3 to dep
- **devbox**: add act to local compute github actions

### Fix

- force .env to overwrite environment variable (e.g., PROFILE)
- add marvelous common module to avoid issue with github (to be corrected)
- **config**: change experiment folder name to avoid dupplication with other students
- **taskfile**: update task command with right env (dev, test) for each task

### Refactor

- add __init__ file for module marvelous

## v0.1.0 (2025-10-06)

### Feat

- add test, qa-lines-count and pc to target in taskfile
- **taskfile**: add clean target to clean up repo from tmp files
- **test**: add test for timer.py
- **test**: add test for data_processor
- **test**: add test for config.py
- add gitlab configuration from Cao standard squeleton
- add install, demo, upload and process data task in taskfile
- robust spark loading session locally or on cli
- add databricks utils module to load spark session locally or with databricks connect from cli
- add env loader to read from .env file
- add processing data scripts and module
- add utilities to catch time execution
- add config.py module to read yaml config file
- add devbox isolation for the setup
- add running on databricks session and not only spark in demo
- add .coveragec config
- add .env template
- add module and script to delete data from unity catalog
- add script and module to upload data in unity catalog from kaggle
- add project configuration yaml file
- update uv.lock file
- add additional packages and add config for commitizen
- add databricks.yml configuration for cohort4
- add standard squeleton for data science project

### Fix

- **pyproject**: correct version to be read by commitizen
- **pyproject**: correct version_files to version.txt
- **pyproject**: correct package name include = ["mlops_course"]
- **uv**: update uv lock file
- **pyproject**: reorganize dev dep
- **ci**: correct ci target to test
- correct typo on cmds
- change from python 3.11 to 3.12
- adapt to use token or profile in authentification

### Refactor

- **commitizen**: delete skip ci when bumping version
- remove and add to .gitignore .coveragec and devbox.lock
- move ReadMe from projects to docs/
