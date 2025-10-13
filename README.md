# 🏨 Hotel Reservation - End-to-End MLOps with Databricks

[![Course Badge](https://img.shields.io/badge/COURSE-MARVELOUS--MLOPS--COHORT4-003189?style=for-the-badge&logo=Databricks&logoColor=FF3621)](https://databricks.com)
[![Platform Badge](https://img.shields.io/badge/PLATFORM-DATABRICKS-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![Lang Badge](https://img.shields.io/badge/LANGUAGE-PYTHON_3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](#)
[![Infra Badge](https://img.shields.io/badge/ENV-DEVBOX_|_UV_|_TASKFILE-6d7cff?style=for-the-badge&logo=dev.to&logoColor=white)](#)

An **end-to-end MLOps project** developed as part of the *Marvelous MLOps Databricks Course (Cohort 4)*.
It automates the lifecycle of a **hotel reservation classification model**, from **data ingestion & feature engineering** to **model training, registration, and deployment** on Databricks.


## 🧠 Project Description

This repository demonstrates:

* **Reproducible ML pipelines** using **Databricks, MLflow**, and **LightGBM**
* **Multi-environment management** across **DEV / ACC / PRD**
* **Data ingestion → preprocessing → feature engineering → model training**
* **Databricks Feature Store & Feature Lookup** integration
* **Task automation** with Devbox + UV + Taskfile
* **CI/CD** with GitHub Actions & GitLab CI (including docs build)
* **Testing & QA** with pre-commit, Ruff, and Pytest (Spark/Delta mocked)


## 🧰 Technology Stack

### Core Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-9cf?style=for-the-badge&logo=python&logoColor=white)

### Dev Tools
![Devbox](https://img.shields.io/badge/Devbox-6d7cff?style=for-the-badge&logo=dev.to&logoColor=white)
![Taskfile](https://img.shields.io/badge/Taskfile-231F20?style=for-the-badge&logo=gnu-bash&logoColor=white)
![UV](https://img.shields.io/badge/UV_Package_Manager-181717?style=for-the-badge&logo=pypi&logoColor=white)

### Version Control & CI/CD
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![GitLab](https://img.shields.io/badge/gitlab-%23181717.svg?style=for-the-badge&logo=gitlab&logoColor=white)
![Pre-commit](https://img.shields.io/badge/Pre--commit-FFBB00?style=for-the-badge&logo=git&logoColor=white)
![Commitizen](https://img.shields.io/badge/Commitizen-1E90FF?style=for-the-badge&logo=git&logoColor=white)


## ⚙️ Installation & Setup (Taskfile)

```bash
# 1. Clone
git clone <your-repo-url>
cd hotel_reservation

# 2. Install tooling
task install

# 3. Create & sync Python env
task dev-install

# 4. Configure environment
cp .env.template .env
# → update with GitHub token / Databricks creds

# 5. Run demo
task demo
````


## 🧩 Project Overview

### 📂 Structure

```
hotel_reservation/
│
├── src/hotel_reservation/
│   ├── data/            # Data ingestion & upload
│   ├── feature/         # Feature engineering (DataProcessor, FE Store utils)
│   ├── model/           # Model training & registry
│   ├── utils/           # Config, env loader, timing
│   └── vizualization/   # Placeholder for visualizations
│
├── scripts/
│   ├── run_upload_data.py          # Upload CSV to Databricks volume
│   ├── run_process_data.py         # Create train/test tables
│   ├── run_create_mlflow_workspace.py
│   ├── run_cleanup_mlflow_experiments.py
│   ├── train_register_model.py     # Train & register baseline model
│   └── train_register_fe_model.py  # NEW: Train with Feature Store + FE lookup
│
├── data/                  # Local datasets
├── docs/                  # Documentation (built via GitLab CI)
├── tests/                 # Unit tests (Pytest + Spark/Delta mocks)
├── project_config.yml     # Config per env (dev/acc/prd)
├── Taskfile.yml           # Tasks (lint, demo, fe_train_register_model…)
├── pyproject.toml         # Dependencies & metadata
└── .github / .gitlab/     # CI/CD configs
```


## 🚀 Key Features

* **End-to-End Workflow**: data upload → feature engineering → training → registry → prediction
* **Feature Store integration**: `train_register_fe_model.py` builds feature tables & lookup functions
* **Environment aware**: `project_config.yml` defines `dev/acc/prd` with separate catalogs & schemas
* **Robust testing**: mocks PySpark & Delta, enabling fast local testing
* **Docs pipeline**: CI builds & publishes Sphinx docs from `docs/hotel_reservation`


## 🧪 Development Workflow

| Command                        | Description                           |
| ------------------------------ | ------------------------------------- |
| `task dev-install`             | Setup dev dependencies                |
| `task demo`                    | Run demo training pipeline            |
| `task run-upload-data`         | Upload dataset to Databricks volume   |
| `task run-process-data`        | Create train/test Delta tables        |
| `task train-register-model`    | Train & register baseline model       |
| `task fe_train_register_model` | **NEW** Train with Feature Store & FE |
| `task lint`                    | Run linters & pre-commit hooks        |
| `task clean`                   | Cleanup env & temp files              |


## 🧱 Prerequisites

* **Mandatory**: Linux/macOS, Python 3.12+, Databricks account/workspace, `task`, `devbox`, `uv`
* **Recommended**: Docker for isolated testing, CI setup (GitHub/GitLab)


## 🧾 Configuration

Example (`project_config.yml`):

```yaml
dev:
  catalog_name: mlops_dev
  schema_name: caotrido
  volume_name: data
  raw_data_file: "Hotel Reservations.csv"
  train_table: hotel_reservations_train_set
  test_table: hotel_reservations_test_set
  feature_table_name: hotel_reservations_features
  feature_function_name: hotel_reservations_feature_fn
  experiment_name_fe: /Shared/hotel_reservations/fe_experiment
```

Switch envs with:

```bash
task run-upload-data -- --env=dev
task fe_train_register_model -- --env=prd
```


## 📊 End-to-End Workflow (Mermaid)

```mermaid
flowchart TD
    A[Raw Data CSV] --> B[Upload to Databricks Volume]
    B --> C[Process Data → Train/Test Delta Tables]
    C --> D["Feature Engineering Pipeline (optional)"]
    D --> E["Feature Store: Table + Function (optional)"]
    E --> F["Model Training (Logistic Regression)"]
    F --> G[MLflow Tracking: params, metrics, artifacts]
    G --> H[MLflow Registry / Unity Catalog]
    H --> I[Load Latest Production Model]
    I --> J[Batch/Online Prediction]
```


## 🧑‍💻 Contributing

```bash
git checkout -b feature/<your-feature>
task lint
cz commit
git push origin feature/<your-feature>
```

Then open a **Merge Request / Pull Request**.


## 📜 License

Proprietary © 2025 — *Marvelous MLOps Course / Cao Tri Do*
For **educational and internal use only**.
