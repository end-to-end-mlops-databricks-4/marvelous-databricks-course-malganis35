# 🛠 Taskfile — Project Automation Guide

This project uses [`Taskfile`](https://taskfile.dev/) to automate development, testing, and deployment workflows.
It acts like a modern Makefile — simple, cross-platform, and dependency-aware.

---

## 📦 Installation

Before using any tasks, make sure the [`task`](https://taskfile.dev/installation/) CLI tool is installed.

### Install Task
```bash
# Ubuntu / Debian
sudo apt install task

# Or via script (recommended)
sh -c "$(curl -fsSL https://taskfile.dev/install.sh)"
````

### Verify

```bash
task --version
```

---

## 🧭 Why Taskfile?

`Taskfile.yml` provides a **consistent and reproducible way** to manage your project:

* 🔁 Set up and sync Python environments with [`uv`](https://github.com/astral-sh/uv)
* 🧪 Run data pipelines, model training, and evaluation
* 🧹 Clean the workspace safely
* 🚀 Deploy to Databricks or Streamlit
* ✅ Run pre-commit hooks and linting

No need to memorize long shell commands — just run:

```bash
task <task-name>
```

---

## 🚀 Quick Start

### 1. Install dependencies and create the environment

```bash
task install
task create-venv
task dev-install
```

### 2. Run preprocessing or training scripts

```bash
task run_process_data
task run_train_register_basic_model
```

### 3. Make predictions using a registered model

```bash
task run_predict_basic_model
```

### 4. Deploy and test serving endpoints

```bash
task run_deploy_basic_model_serving
task run_test_basic_model_serving
```

### 5. Run tests and lint checks

```bash
task test
task lint
```

### 6. Clean everything

```bash
task clean
```

---

## 📋 Available Tasks

| Task                                | Description                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| **install**                         | Install project-level tools (`task`, `uv`, `devbox`).                                 |
| **create-venv**                     | Create a virtual environment using `uv` with Python 3.12.                             |
| **dev-install**                     | Recreate `.venv` and install all development dependencies.                            |
| **test-install**                    | Install only the test dependencies.                                                   |
| **demo**                            | Run the demonstration script (`notebooks/demo.py`).                                   |
| **run_upload_data**                 | Upload and prepare data for the project using `.env` and `project_config.yml`.        |
| **run_process_data**                | Execute the preprocessing pipeline.                                                   |
| **run_process_new_data**            | Process newly arrived datasets.                                                       |
| **run_create_mlflow_workspace**     | Create Databricks directories for MLflow experiments.                                 |
| **run_train_register_basic_model**  | Train, log, and register a **basic model** in MLflow.                                 |
| **run_train_register_custom_model** | Train and register a **custom model** (e.g., logistic regression with probabilities). |
| **run_train_register_fe_model**     | Train and register a model with **feature engineering** lookup.                       |
| **run_predict_basic_model**         | Download and make predictions using the latest basic model.                           |
| **run_predict_custom_model**        | Download and make predictions using the latest custom model.                          |
| **run_predict_fe_model**            | Make predictions with the latest feature-engineered model.                            |
| **run_deploy_basic_model_serving**  | Deploy a basic model as a Databricks serving endpoint.                                |
| **run_deploy_custom_model_serving** | Deploy a custom model as a Databricks serving endpoint.                               |
| **run_test_basic_model_serving**    | Test the deployed basic model serving endpoint.                                       |
| **run_test_custom_model_serving**   | Test the deployed custom model serving endpoint.                                      |
| **run_local_streamlit_app**         | Run the Streamlit app locally for testing.                                            |
| **deploy_streamlit_app**            | Deploy the Streamlit app to Databricks.                                               |
| **lint / pc**                       | Run all pre-commit hooks via `uv run pre-commit run --all-files`.                     |
| **test**                            | Run unit tests with `pytest` and generate a coverage report.                          |
| **qa-lines-count**                  | Count lines of code for all `.py` files (excluding `.venv`).                          |
| **clean**                           | Remove caches, build artifacts, and temporary files.                                  |
| **clean-databricks**                | Clean data, schemas, and experiments from Databricks.                                 |
| **digest**                          | Generate a Git digest of the repository using `gitingest`.                            |
| **databrickscfg**                   | Display the contents of the `~/.databrickscfg` file.                                  |
| **help / default**                  | List all available tasks with their descriptions.                                     |

---

## 🧹 Maintenance Commands

### Clean your environment

```bash
task clean
```

Removes:

* `.venv/`, `_artifact/`, `.pytest_cache/`, `.ruff_cache/`
* `__pycache__`, `.egg-info/`
* Node modules and coverage artifacts

### Count lines of code per file (identifying big files)

```bash
task qa-lines-count
```

Displays the number of lines per `.py` file, sorted ascending.

---

## 🧠 Tips & Best Practices

* Always run `task dev-install` after editing dependencies in `pyproject.toml`.
* Use `task test` locally before pushing changes to ensure builds and tests pass.
* Use `task lint` locally before pushing changes to ensure code is properly linted.
* Ensure your `~/.databrickscfg` file is configured before running Databricks tasks.
* You can reuse these tasks in CI/CD pipelines:

```yaml
# Example: GitHub Actions
- name: Run tests
  run: task test
```

---

## 🌟 Example Workflow

```bash
# 1️⃣ Setup environment
task install
task dev-install

# 2️⃣ Upload & process data
task run_upload_data
task run_process_data

# 3️⃣ Train & register model
task run_train_register_basic_model

# 4️⃣ Predict & evaluate
task run_predict_basic_model

# 5️⃣ Deploy and test model serving
task run_deploy_basic_model_serving
task run_test_basic_model_serving

# 6️⃣ Clean up environment
task clean
```

---

## 🆘 Getting Help

To list all available tasks:

```bash
task help
```

Or simply run:

```bash
task
```
