# 🛠 Taskfile Introduction

This project uses [`Taskfile`](https://taskfile.dev/) to automate common development, testing, and deployment workflows.
`Taskfile.yml` acts like a Makefile but with modern syntax, dependency resolution, and cross-platform support.

---

## 📦 Installing Task

Before you use any tasks, you need to install the `task` CLI tool:

👉 Install it from: [https://taskfile.dev/installation/](https://taskfile.dev/installation/)

Verify your installation with:

```bash
task --version

## 🧭 Why Taskfile?

Taskfile provides a consistent and reliable way to:

- Set up virtual environments and install dependencies
- Run tests and generate coverage reports
- Format, lint, and clean your codebase
- Serve reports or run MLflow locally
- Manage multiple dev/test environments with `uv`

✨ No more copy-pasting shell commands — just run short, repeatable tasks!

---

## 🚀 Usage

Once Task is installed, you can run any task like this:

```bash
task <task-name>
```

For example, to sync your development environment:

```bash
task sync-dev
```

Run unit tests with coverage:
```bash
task run-unit-tests
```
Clean up build artifacts and caches:

```bash
task clean
```

## 📋 Available Tasks

| Task                                     | Description |
|------------------------------------------|-------------|
| `create-venv`                            | Creates a virtual environment using `uv` with Python 3.11.9. |
| `sync-dev`                               | Deletes and recreates the virtual environment, then installs all **dev dependencies** from `pyproject.toml`. |
| `sync-test`                              | Deletes and recreates the virtual environment, then installs only the **test dependencies**. |
| `lint`                                   | Runs `pre-commit` hooks on all files using `uv run`. |
| `run-unit-tests`                         | Runs `pytest` with coverage tracking, stores results in the `test-reports/` directory. |
| `clean`                                  | Removes build artifacts, cache files, `.pyc` files, and temporary test files. |
| `build`                                  | Builds the project using `uv build` (uses `setuptools`). |
| `serve-coverage-report`                  | Serves the HTML coverage report at [http://localhost:8000](http://localhost:8000). |
| `serve-latest-registered-model-locally`  | Downloads the latest registered MLflow model from Databricks and serves it locally via `mlflow models serve`. |
| `help` or `default`                      | Lists all defined tasks with their descriptions. |
