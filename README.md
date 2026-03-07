# REPLACE_WITH_PROJECT_NAME

REPLACE_WITH_ONE_SENTENCE_DESCRIPTION

## Overview

A minimal, reproducible Python project template using a `src` layout, modern tooling and Python 3.13 or newer.

## Directory layout

```text
.
├── src/
│   └── REPLACE_WITH_PROJECT_NAME/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_template_scaffolding.py
├── .github/
│   ├── actions/setup-uv-env/
│   │   └── action.yml
│   └── workflows/
│       └── main.yml
├── .pre-commit-config.yaml
├── noxfile.py
├── pyproject.toml
└── uv.lock
```

## What is included

| File / directory | Purpose |
|---|---|
| `src/` | Project package in a `src` layout. Contains a placeholder directory to be renamed. |
| `tests/` | Automated tests. Ships with a single scaffolding test to validate the tooling. |
| `pyproject.toml` | Project metadata, Python version constraint, dev dependencies and tool configuration. |
| `.pre-commit-config.yaml` | Local pre-commit hooks that run pytest (with 100% coverage), ruff, mypy and bandit. |
| `noxfile.py` | Nox sessions for ruff, mypy and bandit, targeting the `src` and `tests` directories. |
| `.github/workflows/main.yml` | CI workflow that runs pre-commit, nox and the test suite on Ubuntu, macOS and Windows. |
| `.github/actions/setup-uv-env/` | Composite action that installs uv and the requested Python version. |

## Tooling summary

| Tool | Role |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Dependency management and virtual environment creation. |
| [pytest](https://docs.pytest.org/) | Test runner with coverage reporting via pytest-cov. |
| [ruff](https://docs.astral.sh/ruff/) | Linting and formatting in a single tool. |
| [mypy](https://mypy.readthedocs.io/) | Static type checking. |
| [bandit](https://bandit.readthedocs.io/) | Security-focused static analysis. |
| [nox](https://nox.thea.codes/) | Task runner for linting, type checking and security scanning sessions. |
| [pre-commit](https://pre-commit.com/) | Git hook manager that runs checks before each commit. |

## Prerequisites

- **Python 3.13+** installed on your system.
- **[uv](https://docs.astral.sh/uv/)** installed for dependency and environment management.
- **Git** for version control, hooks and the CI workflow.

## Getting started

### 1. Create your project from the template

Clone or copy the repository, then replace the placeholders with your own values:

- Rename the `src/REPLACE_WITH_PROJECT_NAME/` directory to your package name.
- In `pyproject.toml` and `README.md`, replace `REPLACE_WITH_PROJECT_NAME` with your package name.
- In `pyproject.toml` and `README.md`, replace `REPLACE_WITH_ONE_SENTENCE_DESCRIPTION` with a short project description.

### 2. Set up the development environment

```bash
uv venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
uv sync --group dev
```

### 3. Install the pre-commit hooks

```bash
uv run pre-commit install
```

After this, every `git commit` will automatically run the full hook suite (tests, linting, type checking, security scanning). The commit is rejected if any check fails or test coverage drops below 100%.

### 4. Run the test suite

```bash
uv run pytest
```

### 5. Run linting and static analysis

Run individual nox sessions:

```bash
uv run nox -s ruff
uv run nox -s mypy
uv run nox -s bandit
```

Or run all default sessions at once:

```bash
uv run nox
```

## Continuous integration

The GitHub Actions workflow (`.github/workflows/main.yml`) triggers on pushes to `main` and on pull requests. It runs across Ubuntu, macOS and Windows with Python 3.13 and executes:

1. All nox sessions (ruff, mypy, bandit).
2. The test suite with coverage output in XML format.

## Licence

[MIT](LICENSE)
