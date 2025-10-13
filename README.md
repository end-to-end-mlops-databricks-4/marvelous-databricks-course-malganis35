# 🏨 Hotel Reservation - End-to-End MLOps with Databricks

[![Course Badge](https://img.shields.io/badge/COURSE-MARVELOUS--MLOPS--COHORT4-003189?style=for-the-badge\&logo=Databricks\&logoColor=FF3621)](https://databricks.com)
[![Platform Badge](https://img.shields.io/badge/PLATFORM-DATABRICKS-FF3621?style=for-the-badge\&logo=databricks\&logoColor=white)](https://databricks.com)
[![Lang Badge](https://img.shields.io/badge/LANGUAGE-PYTHON_3.12-3670A0?style=for-the-badge\&logo=python\&logoColor=ffdd54)](#)
[![Infra Badge](https://img.shields.io/badge/ENV-DEVBOX_|_UV_|_TASKFILE-6d7cff?style=for-the-badge\&logo=dev.to\&logoColor=white)](#)

An **end-to-end MLOps project** developed as part of the *Marvelous MLOps Databricks Course (Cohort 4)*.
This project automates the full lifecycle of a **hotel reservation classification model**, from **data ingestion** to **model deployment** on Databricks.


## 🧠 Project Description

This repository demonstrates how to:

* Build reproducible ML pipelines using **Databricks, MLflow**, and **scikit-learn**
* Manage configurations across **DEV / ACC / PRD environments**
* Handle **data ingestion, preprocessing, and model training**
* Use **Devbox + UV + Taskfile** for environment and task management
* Integrate **CI/CD pipelines** with **GitHub Actions** and **GitLab CI**

## 🧰 Technological Stack

### **Core Stack**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge\&logo=python\&logoColor=ffdd54)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge\&logo=databricks\&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge\&logo=mlflow\&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-9cf?style=for-the-badge\&logo=python\&logoColor=white)

### **Dev Tools**

![Devbox](https://img.shields.io/badge/Devbox-6d7cff?style=for-the-badge\&logo=dev.to\&logoColor=white)
![Taskfile](https://img.shields.io/badge/Taskfile-231F20?style=for-the-badge\&logo=gnu-bash\&logoColor=white)
![UV](https://img.shields.io/badge/UV_Package_Manager-181717?style=for-the-badge\&logo=pypi\&logoColor=white)

### **Version Control & CI/CD**

![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge\&logo=github\&logoColor=white)
![GitLab](https://img.shields.io/badge/gitlab-%23181717.svg?style=for-the-badge\&logo=gitlab\&logoColor=white)
![Pre-commit](https://img.shields.io/badge/Pre--commit-FFBB00?style=for-the-badge\&logo=git\&logoColor=white)
![Commitizen](https://img.shields.io/badge/Commitizen-1E90FF?style=for-the-badge\&logo=git\&logoColor=white)


## ⚙️ Installation & Setup (with Taskfile)

### 1. Clone the project

```bash
git clone <your-repo-url>
cd malganis35_cohort4
```

### 2. Install dependencies and tools

```bash
task install
```

This installs `uv`, `task`, and `devbox`.

### 3. Create and sync the Python environment

```bash
task dev-install
```

This creates a `.venv` and installs all **dev dependencies** listed in `pyproject.toml`.

### 4. Set up environment variables

Copy the example environment file:

```bash
cp .env.template .env
```

Update it with your **GitHub Personal Access Token** or Databricks credentials:

```bash
GIT_TOKEN=<your_github_PAT>
```

### 5. Run demo scripts

Run a quick demo to verify the setup:

```bash
task demo
```

Or upload data and process it in the Databricks workspace:

```bash
task run-upload-data
task run-process-data
```

## 🧩 Project Overview

### 📂 Project Structure

```
malganis35_cohort4/
│
├── src/mlops_course/
│   ├── data/              # Data ingestion and upload utilities
│   ├── feature/           # Feature engineering and preprocessing
│   ├── model/             # Model training and registry logic
│   ├── utils/             # Config, environment, timing helpers
│   └── vizualization/     # Visualization utilities (placeholder)
│
├── scripts/               # Execution scripts (upload, cleanup, process)
├── data/                  # Raw, processed, and external data
├── docs/                  # Course and internal documentation
├── tests/                 # Unit tests (pytest)
├── project_config.yml     # Multi-env Databricks config (dev, acc, prd)
├── Taskfile.yml           # Task automation definitions
├── devbox.json            # Devbox environment setup
├── pyproject.toml         # Dependencies and metadata
└── .github / .gitlab/     # CI/CD configurations
```

## 🚀 Key Features

* **End-to-End MLOps Workflow**: From data upload → model training → MLflow registry.
* **Multi-Environment Configuration**: `dev`, `acc`, and `prd` managed in `project_config.yml`.
* **Databricks Integration**: Uses `databricks-sdk` and `databricks-connect` for remote operations.
* **Reproducibility**: Fully reproducible via `Devbox` and `UV` environments.
* **Automated QA**: Pre-commit hooks, Ruff linting, and unit tests integrated.
* **Task Automation**: Common commands managed by `Taskfile.yml`.

## 🧪 Development Workflow

| Command                 | Description                          |
| ----------------------- | ------------------------------------ |
| `task dev-install`      | Sync all dev dependencies            |
| `task demo`             | Run demo model training              |
| `task run-upload-data`  | Upload data to Databricks volume     |
| `task run-process-data` | Execute preprocessing pipeline       |
| `task lint`             | Run `pre-commit` hooks               |
| `task clean`            | Remove temporary files and venv      |
| `task digest`           | Generate git digest for repo summary |


## 🧱 Prerequisites

**Mandatory**

* Linux/macOS environment
* Python 3.12+
* Databricks account and workspace (Free or Premium)
* `task` installed: ```sudo apt install task```
* `devbox` and `uv` can be installed using: ```task install```

**Recommended**

* Docker (for isolated environment testing)
* GitHub or GitLab CI enabled

## 🧾 Configuration

Main environment configurations are defined in:

```yaml
# project_config.yml
dev:
  catalog_name: mlops_dev
  schema_name: caotrido
  volume_name: data
  raw_data_file: "Hotel Reservations.csv"
  train_table: hotel_reservations_train_set
  test_table: hotel_reservations_test_set
```

Switch between environments (`dev`, `acc`, `prd`) by passing flags in your Task commands.

## 🧑‍💻 Contributing

1. Create a new feature branch:

   ```bash
   git checkout -b feature/<your-feature>
   ```
2. Run linters and tests before pushing:

   ```bash
   task lint
   ```
3. Commit with **Commitizen** format:

   ```bash
   cz commit
   ```
4. Push and open a merge request or pull request.

## 📜 License

Proprietary © 2025 — *Marvelous MLOps Course / Cao Tri Do*
For educational and internal use only.
