Title: Microsoft Copilot: Your AI companion

URL Source: https://copilot.microsoft.com/shares/tasks/p38nBoXd6a2TyTN6EGx9p

Markdown Content:
Deep Research

Upgrade Guide for Migrating Python Project from pip to PDM

## Modernizing a Python 3.10 Project: A Comprehensive Step-by-Step Guide to Migrating from pip to PDM

## Introduction

Modern Python development increasingly emphasizes **reproducibility, environment isolation, and adherence to packaging standards**. While `pip` and `requirements.txt` have long been the de facto tools for dependency management, they lack features such as deterministic lockfiles, robust environment automation, and first-class support for the `pyproject.toml` standard. **PDM (Python Development Master)** is a modern Python package and dependency manager that addresses these gaps by providing a unified, PEP-compliant workflow for managing dependencies, environments, and project metadata.

This report provides a **comprehensive, step-by-step guide** for upgrading an existing Python 3.10 project from a traditional `pip`-based workflow to PDM. It covers installation, configuration, dependency migration, environment management, CI/CD integration, best practices for reproducibility, and caveats to ensure a smooth and robust transition. The guide is tailored for developers familiar with Python and `pip`, and is structured to facilitate both immediate migration and long-term maintainability.

1. Understanding PDM: Motivation and Core Concepts

---

### 1.1 Why Modernize with PDM?

**PDM** is designed to bring Python project management in line with modern standards and workflows seen in other ecosystems (such as Node.js and Rust). Its key advantages over traditional `pip` workflows include:

- **PEP 621 and PEP 582 Compliance:** PDM uses `pyproject.toml` as the single source of truth for project metadata and dependencies, supporting the latest Python packaging standards.

- **Automated Environment Management:** PDM can automatically create and manage isolated environments, either via virtualenv or the PEP 582 `__pypackages__` directory, reducing manual setup and activation steps.

- **Deterministic Dependency Resolution:** Through its lockfile (`pdm.lock`), PDM ensures reproducible builds and installations across all environments.

- **Integrated Python Version Management:** PDM can manage and install multiple Python interpreters, ensuring projects use the correct version.

- **Flexible Dependency Grouping:** Supports dev dependencies, optional extras, and custom groups for fine-grained control.

- **Extensible and Fast:** PDM offers a plugin system and a high-performance dependency resolver.

**In summary, PDM unifies dependency management, environment isolation, and packaging metadata under a single, standards-based tool, streamlining both development and deployment workflows.**

2. Installing and Configuring PDM

---

### 2.1 Installing PDM

PDM requires **Python 3.9 or higher** to run (your project can target lower versions, but PDM itself must be run with 3.9+). There are several installation methods:

- **Recommended (Cross-platform):**

bash ```
curl -sSL https://pdm-project.org/install-pdm.py | python3 -

````
This script installs PDM into your user site (`$HOME/.local/bin` on Unix, `%APPDATA%\Python\Scripts` on Windows). Ensure this directory is in your `PATH`.

*   **Homebrew (macOS):**

bash ```
brew install pdm
````

- **pipx (isolated environment):**

bash ```
pipx install pdm

````
*   **pip (user site):**

bash ```
pip install --user pdm
````

- **Windows (PowerShell):**

powershell ```
powershell -ExecutionPolicy ByPass -c "irm https://pdm-project.org/install-pdm.py | py -"

```

After installation, verify with:

bash

```

pdm --version

````

You should see output similar to `PDM, version 2.26.1`.

### 2.2 Initial Configuration

PDM can be configured globally or per-project. Key configuration options include:

*   **Virtual Environment Backend:** Choose between `virtualenv`, `venv`, or `conda`:

bash ```
pdm config venv.backend virtualenv  # or venv, conda
````

- **Default to Virtualenv or PEP 582:** By default, PDM uses virtualenv. To use PEP 582:

bash ```
pdm config python.use_venv false

```
*   **Python Interpreter Management:** PDM can install and manage Python interpreters (see Section 4).

**Tip:** For CI/CD or shared environments, consider setting configuration options in the project’s `pdm.toml` or via environment variables for consistency.

3. Migrating Dependencies to pyproject.toml
-------------------------------------------

### 3.1 Overview of Migration Paths

Most existing Python projects declare dependencies in either `requirements.txt` (for applications) or `setup.py` (for libraries). Migrating to PDM involves converting these to the standardized `pyproject.toml` format.

#### Supported Migration Sources:

*   `requirements.txt`

*   `setup.py`

*   `Pipfile` (Pipenv)

*   Poetry’s `pyproject.toml`

*   Flit’s `pyproject.toml`

### 3.2 Using `pdm init` and `pdm import`

#### 3.2.1 Initializing pyproject.toml

Navigate to your project root and run:

bash

```

pdm init

````

This interactive command will:

*   Prompt for the Python interpreter (see Section 4).

*   Ask if you want to create a virtual environment.

*   Collect project metadata (name, version, license, author, etc.).

*   Ask if you want to import dependencies from existing files.

**If PDM detects**`requirements.txt`**or**`setup.py`**, it will offer to import them automatically.**

#### 3.2.2 Importing Dependencies

*   **From requirements.txt:**

bash ```
pdm import requirements.txt
````

- **From setup.py:**

bash ```
pdm import setup.py

````
> _Note: Importing from_`setup.py`_executes the file with the project interpreter. Ensure it is trusted and that_`setuptools`_is installed in the environment._

PDM will parse the dependencies and add them to the `[project] dependencies` section in `pyproject.toml`. It will also create a `pdm.lock` file with the resolved dependency tree.

#### 3.2.3 Edge Cases and Manual Adjustments

*   **Unsupported Syntax:** Some pip-specific flags in `requirements.txt` (e.g., `-r`, `-c`, `-e`) are not supported. You may need to manually add editable or constraint dependencies.

*   **Version Specifiers:** PDM supports PEP 508 specifiers. If your `requirements.txt` uses non-standard syntax, adjust accordingly.

*   **Dev Dependencies:** If you have a `requirements-dev.txt`, import it as a development group:

bash ```
pdm import requirements-dev.txt --group dev
````

- **Optional/Extra Dependencies:** For optional features, use `[project.optional-dependencies]` in `pyproject.toml`.

#### 3.2.4 Example: Migrating requirements.txt

Suppose your `requirements.txt` contains:

Code

```
requests==2.31.0
numpy>=1.21.0
```

After running `pdm import requirements.txt`, your `pyproject.toml` will include:

toml

```
[project]
dependencies = [
    "requests==2.31.0",
    "numpy>=1.21.0"
]
```

And a `pdm.lock` file will be generated with the full dependency tree.

#### 3.2.5 Migrating from setup.py

PDM can extract metadata and dependencies from `setup.py`, but complex dynamic logic may require manual intervention. For projects with `setup.cfg`, consider using tools like `ini2toml` for conversion, or migrate to `pyproject.toml` manually.

4. Managing Python Interpreters and Virtual Environments

---

### 4.1 Python Interpreter Selection

PDM allows you to specify and manage the Python interpreter for your project, ensuring compatibility and reproducibility.

- **Selecting an Interpreter:**

bash ```
pdm use 3.10

````
This sets the interpreter for the project and stores the path in `.pdm-python`.

*   **Listing Available Interpreters:**

bash ```
pdm python list
````

- **Installing a New Interpreter:**

bash ```
pdm python install 3.10.8

````
PDM downloads and installs the specified version using `python-build-standalone`.

*   **.python-version Integration:** If a `.python-version` file is present (as used by `pyenv`), PDM will use its value.

*   **Changing the Interpreter:**

bash ```
pdm use /path/to/python3.10
````

**Best Practice:** Set the `requires-python` field in `pyproject.toml` to match your project’s compatibility:

toml

```
[project]
requires-python = ">=3.10,<3.11"
```

This ensures that dependency resolution and lockfiles are consistent with your intended Python version.

### 4.2 Virtual Environment Modes

PDM supports two primary modes for environment isolation:

#### 4.2.1 Virtualenv Mode (Default)

- **Creates a**`.venv`**directory** in the project root.

- **Dependencies are installed into the virtual environment.**

- **Well-supported by IDEs and tooling.**

- **Recommended for most projects.**

**Creating/Reusing a Virtualenv:**

bash

```
pdm venv create 3.10
pdm use --venv in-project
```

**Switching to a Named Virtualenv:**

bash

```
pdm venv create --name testenv 3.10
pdm use --venv testenv
```

#### 4.2.2 PEP 582 (`__pypackages__`) Mode

- **Dependencies are installed into a**`__pypackages__/<major.minor>/lib`**directory** in the project root.

- **No activation required; scripts run with**`pdm run`**automatically use the correct environment.**

- **Not yet a Python standard (PEP 582 is rejected), but supported by PDM for Node.js-style workflows.**

- **IDE support may require manual configuration.**

**Enabling PEP 582:**

bash

```
pdm config python.use_venv false
```

**Note:** For global PEP 582 support, run:

bash

```
eval "$(pdm --pep582)"
```

and add this to your shell profile.

#### 4.2.3 Comparison Table

| Mode       | Isolation | Activation Needed | IDE Support | Directory Used           | Recommended For         |
| ---------- | --------- | ----------------- | ----------- | ------------------------ | ----------------------- |
| Virtualenv | Strong    | Yes (optional)    | Excellent   | .venv                    | Most projects           |
| PEP 582    | Moderate  | No                | Manual      | **pypackages**/<ver>/lib | Node.js-style workflows |

**Analysis:** Virtualenv mode is more mature and better supported by IDEs and external tools. PEP 582 is convenient for rapid prototyping and for developers coming from JavaScript backgrounds, but may require extra configuration for full IDE integration.

### 4.3 Environment Management Commands

- **Show Current Environment:**

bash ```
pdm info

````
*   **List All Virtualenvs:**

bash ```
pdm venv list
````

- **Activate a Virtualenv:**

bash ```
eval $(pdm venv activate)

````
*   **Remove a Virtualenv:**

bash ```
pdm venv remove <name>
````

**Tip:** PDM does not require manual activation for most workflows; use `pdm run <command>` to execute scripts in the project environment.

5. Managing Dependencies and Dependency Groups

---

### 5.1 Adding, Updating, and Removing Dependencies

- **Add a Dependency:**

bash ```
pdm add requests

````
*   **Add Multiple Dependencies:**

bash ```
pdm add numpy pandas matplotlib
````

- **Add with Version Constraints:**

bash ```
pdm add "django>=4.2"

````
*   **Remove a Dependency:**

bash ```
pdm remove requests
````

- **Update All Dependencies:**

bash ```
pdm update

````
*   **Update Specific Packages:**

bash ```
pdm update fastapi pydantic
````

**All changes are reflected in both**`pyproject.toml`**and**`pdm.lock`**for reproducibility.**

### 5.2 Dependency Groups: Dev, Optional, and Custom

PDM supports **dependency groups** for organizing dependencies by purpose:

- **Default (runtime) dependencies:**`[project] dependencies`

- **Development dependencies:**`[dependency-groups] dev = [...]`

- **Custom groups:**`[dependency-groups] test = [...]`, `[dependency-groups] lint = [...]`

- **Optional (extras):**`[project.optional-dependencies]`

**Adding Dev Dependencies:**

bash

```
pdm add --dev pytest mypy black
```

This creates a `[dependency-groups] dev = [...]` section.

**Adding Custom Groups:**

bash

```
pdm add -dG test pytest pytest-cov
pdm add -dG lint flake8 black
```

**Adding Optional Extras:**

bash

```
pdm add -G web flask jinja2
```

This creates a `[project.optional-dependencies] web = [...]` section.

#### 5.2.1 Installing Specific Groups

- **Production only:**

bash ```
pdm install --prod

````
*   **Production + specific optional group:**

bash ```
pdm install -G web
````

- **Production + specific dev group:**

bash ```
pdm install -G test

````
*   **All dependencies:**

bash ```
pdm install
````

**Analysis:** Dependency groups allow for clear separation of runtime, development, and optional dependencies, improving maintainability and reducing environment bloat in production deployments.

### 5.3 Lockfile Management and Reproducibility

PDM’s `pdm.lock` file records the exact versions of all dependencies and sub-dependencies, ensuring **deterministic, reproducible installs**.

- **Creating/Updating the Lockfile:**

bash ```
pdm lock

````
*   **Checking Lockfile Consistency:**

bash ```
pdm lock --check
````

- **Refreshing Hashes:**

bash ```
pdm lock --refresh

````

**Best Practice:** Always commit both `pyproject.toml` and `pdm.lock` to version control for reproducibility across teams and CI/CD environments.

#### 5.3.1 Platform- and Python-Version-Specific Lockfiles

For projects with platform- or Python-version-specific dependencies, PDM supports targeted lockfiles:

*   **Lock for a Specific Python Version:**

bash ```
pdm lock --python=">=3.10,<3.11"
````

- **Lock for a Specific Platform:**

bash ```
pdm lock --platform=linux

````
*   **Lock for Multiple Targets:**

bash ```
pdm lock --platform=linux --python="==3.10.*" --lockfile=py310-linux.lock
````

- **Append to Existing Lockfile:**

bash ```
pdm lock --platform=windows --python="==3.10.\*" --append

```

**Analysis:** This feature is crucial for projects with conditional dependencies or for ensuring reproducibility across heterogeneous deployment targets.

6. Updating Scripts, CLI Commands, and Project Metadata
-------------------------------------------------------

### 6.1 Project Scripts and Entry Points

PDM supports two mechanisms for scripts:

*   **[project.scripts]:** For defining console script entry points (installed with the package).

*   **[tool.pdm.scripts]:** For defining development and automation tasks (similar to npm scripts or Makefile).

**Example:**

toml

```

[project.scripts]
mycli = "my_package.cli:main"

[tool.pdm.scripts]
test = "pytest tests"
lint = "flake8 src"
all = {composite = ["lint", "test"]}

```

**Running Scripts:**

bash

```

pdm run test
pdm run lint
pdm run all

```

Scripts defined in `[tool.pdm.scripts]` can be composite, shell, or Python function calls, and support environment variables, working directories, and argument placeholders.

### 6.2 Updating Documentation

*   **Replace pip/venv instructions** with PDM equivalents in `README.md` and developer onboarding docs.

*   **Document the use of**`pdm install`**,**`pdm run`**, and dependency group conventions.**

*   **Update any references to**`requirements.txt`**or**`setup.py`**to point to**`pyproject.toml`**and**`pdm.lock`**.**

**Example Table for Documentation Update:**

| Old Command | New PDM Command | Notes |
| --- | --- | --- |
| `pip install -r requirements.txt` | `pdm install` | Installs from lockfile |
| `python -m venv venv` | `pdm init` (with venv) | PDM manages venv creation |
| `pip freeze > requirements.txt` | `pdm lock` | Lockfile is `pdm.lock` |
| `pip install package` | `pdm add package` | Adds and locks dependency |
| `python script.py` | `pdm run python script.py` | Runs in project environment |

**Analysis:** Clear documentation ensures team members and contributors adopt the new workflow consistently.

7. Integrating PDM with CI/CD Pipelines
---------------------------------------

### 7.1 Updating CI/CD Workflows

**Key Steps:**

1.   **Install PDM in the CI environment.**

2.   **Restore or cache the virtual environment and PDM cache for faster builds.**

3.   **Install dependencies using**`pdm install`**or**`pdm sync`**.**

4.   **Run tests and scripts using**`pdm run`**.**

**Example: GitHub Actions Workflow**

yaml

```

jobs:
test:
runs-on: ubuntu-latest
strategy:
matrix:
python-version: ['3.10']
steps: - uses: actions/checkout@v4 - name: Set up PDM
uses: pdm-project/setup-pdm@v4
with:
python-version: ${{ matrix.python-version }} - name: Install dependencies
run: pdm install - name: Run tests
run: pdm run pytest tests

```

**For other CI platforms (GitLab, Azure DevOps):**

*   **Cache the**`.venv`**directory** (or `__pypackages__` for PEP 582) and the PDM cache directory (see `pdm config cache_dir`).

*   **Set environment variables** as needed for cache paths and interpreter selection.

### 7.2 Caching and Performance Considerations

*   **Cache**`.venv`**or**`__pypackages__` by Python version and architecture to avoid redundant installs.

*   **Cache PDM’s internal cache directory** for faster dependency resolution.

*   **Do not cache pip’s cache**; PDM does not use it.

**Example: Azure DevOps Cache Task**

yaml

```

- task: Cache@2
  inputs:
  key: 'venv | "$(Agent.OS)" | "$(python.version)"'
  path: .venv

```

**Analysis:** Proper caching can significantly reduce CI build times and improve reliability.

### 7.3 Exporting requirements.txt  for Legacy Workflows

If you need to support environments that require `requirements.txt` (e.g., Docker builds without PDM), export from the lockfile:

bash

```

pdm export -o requirements.txt

```

**Note:** The exported file will reflect the current environment (Python version, platform). For multi-platform support, generate separate exports as needed.

8. Best Practices for Compatibility and Reproducibility
-------------------------------------------------------

### 8.1 Version Control

*   **Always commit**`pyproject.toml`**and**`pdm.lock`**.**

*   **Do not commit**`.pdm-python`**(interpreter path) or**`.venv`**/**`__pypackages__`**directories.**

*   **Optionally commit**`pdm.toml`**for shared configuration.**

### 8.2 Environment Isolation

*   **Avoid sharing environments between projects.** PDM will remove unlisted dependencies when syncing, which can cause conflicts if environments are reused.

*   **Use one environment per project** for maximum isolation.

### 8.3 Lockfile Hygiene

*   **Regenerate the lockfile** whenever dependencies or Python version constraints change.

*   **Use platform- and version-specific lockfiles** for projects with conditional dependencies.

### 8.4 Dependency Overrides

*   **Use**`[tool.pdm.resolution.overrides]` in `pyproject.toml` to force specific versions when necessary.

*   **Pass constraint files via CLI for organization-wide overrides.**

### 8.5 Python Version Management

*   **Declare**`requires-python` in `pyproject.toml` to ensure all dependencies are compatible.

*   **Use PDM’s Python installer** to manage interpreter versions for CI/CD and onboarding.

### 8.6 IDE and Tooling Integration

*   **VS Code:** Add `__pypackages__/<major.minor>/lib` to `python.autoComplete.extraPaths` and `python.analysis.extraPaths` in `.vscode/settings.json` for PEP 582 mode.

*   **PyCharm:** Mark `__pypackages__/<major.minor>/lib` as a source root and configure the interpreter accordingly.

*   **Linters/Formatters:** Configure to use the PDM-managed environment or point to the correct Python interpreter.

**Example: VS Code settings for PEP 582**

json

```

{
"python.autoComplete.extraPaths": ["__pypackages__/3.10/lib"],
"python.analysis.extraPaths": ["__pypackages__/3.10/lib"]
}

```

**Analysis:** Proper IDE configuration ensures that code completion, linting, and debugging work seamlessly with PDM-managed environments.

9. Caveats, Limitations, and Rollback Strategies
------------------------------------------------

### 9.1 Known Limitations

*   **PEP 582 is not a Python standard:** Some tools and IDEs may not recognize `__pypackages__` without manual configuration.

*   **Editable Installs:** Only allowed in development dependency groups.

*   **Exporting requirements.txt:** The export reflects the current environment; multi-platform exports require separate commands.

*   **Dynamic Metadata:** Some fields in `pyproject.toml` can be marked as dynamic, but not all build backends support this fully.

*   **Legacy Tooling:** Some older tools expect `requirements.txt` or `setup.py`; maintain these as needed during transition.

### 9.2 Rollback and Coexistence

*   **Coexistence:** You can maintain `requirements.txt` and `setup.py` alongside `pyproject.toml` during migration. However, avoid editing dependencies in multiple places to prevent drift.

*   **Rollback:** If issues arise, revert to the previous workflow by restoring `requirements.txt` and using `pip` as before. Ensure environments are cleaned to avoid conflicts.

### 9.3 Testing the Migration

**Checklist:**

*   [ ] All dependencies are present and correctly resolved in `pyproject.toml` and `pdm.lock`.

*   [ ] Scripts and entry points work as expected via `pdm run`.

*   [ ] CI/CD pipelines pass with PDM-based workflows.

*   [ ] IDE integration is functional.

*   [ ] Exported `requirements.txt` (if needed) matches expectations.

*   [ ] No unintentional environment pollution or missing packages.

**Validation Steps:**

*   Test locally by removing `.venv` or `__pypackages__` and reinstalling with `pdm install`.

*   Run all project scripts and tests via `pdm run`.

*   Validate in CI/CD and production environments.

10. Conclusion
--------------

**Migrating from pip to PDM modernizes your Python project, aligning it with current best practices for reproducibility, environment isolation, and standards compliance.** By following the steps outlined in this guide—installing and configuring PDM, migrating dependencies, managing environments, updating scripts and CI/CD, and adhering to best practices—you can ensure a smooth transition and set your project up for robust, maintainable development.

**PDM’s strengths lie in its standards-based approach, flexible environment management, and deterministic dependency resolution.** While there are some caveats (notably around PEP 582 and legacy tooling), the benefits for modern Python development are substantial. With careful planning, thorough testing, and clear documentation, your team can fully leverage PDM’s capabilities for both current and future projects.

Appendix: Key PDM Commands Reference
------------------------------------

| Task | Command Example | Notes |
| --- | --- | --- |
| Install PDM | `curl -sSL https://pdm-project.org/install-pdm.py | python3 -` | User site install |
| Initialize PDM in existing project | `pdm init` | Interactive setup |
| Import requirements.txt | `pdm import requirements.txt` | Converts to pyproject.toml |
| Import setup.py | `pdm import setup.py` | Extracts metadata and dependencies |
| Add dependency | `pdm add requests` | Updates pyproject.toml and lockfile |
| Add dev dependency | `pdm add --dev pytest` | Adds to dev group |
| Remove dependency | `pdm remove requests` | Updates pyproject.toml and lockfile |
| Update dependencies | `pdm update` | Updates all, or specify packages |
| Install dependencies | `pdm install` | Installs from lockfile |
| Run script | `pdm run <script>` | Uses project environment |
| Export requirements.txt | `pdm export -o requirements.txt` | For legacy workflows |
| Create virtualenv | `pdm venv create 3.10` | Named or in-project |
| Use PEP 582 mode | `pdm config python.use_venv false` | Node.js-style isolation |
| Show environment info | `pdm info` | Interpreter, environment, etc. |
| List installed packages | `pdm list` | Supports group filtering |
| Lock dependencies for platform/version | `pdm lock --platform=linux --python="==3.10.*"` | For platform-specific lockfiles |

**By adopting PDM, your Python project will benefit from modern, reproducible, and maintainable workflows, positioning your team for success in both development and deployment.**

References
----------

19

*   Introduction - PDM
*   pdm · PyPI
*   Geek Cafe
*   Introduction - PDM
*   Python Interpreter Management | pdm-project/pdm | DeepWiki
*   Is there a simple way to convert setup.py to pyproject.toml
*   Manage Dependencies - PDM
*   Configuring setuptools using pyproject.toml files - setuptools 80.9.0 ...
*   Working with PEP 582 - PDM
*   PEP 582 – Python local packages directory | peps.python.org
*   Dependency Groups - Python Packaging User Guide
*   Lock file - PDM
*   are cross-platform lockfiles deprecated? #3037 - GitHub
*   Entries added to lockfile by - GitHub
*   PDM Scripts - PDM
*   Cache pdm environments in azure devops pipeline #2935 - GitHub
*   Advanced Usage - PDM
*   Need requirements.txt with fixed package versions - GitHub
*   Debugging a PDM-managed project in VSCode - GitHub
```
