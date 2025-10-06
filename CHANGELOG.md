# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Conventional Commits](https://www.conventionalcommits.org/) with [Commitizen](https://github.com/commitizen/cz-cli).

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
