# Marie AI scripts

Run project automation from the repository root through `scripts/`. This is the
canonical location for human-invoked build, development, operations, release,
and verification utilities.

## Quick index

| Area | Commands |
| --- | --- |
| Verification and formatting | `test-scheduler.sh`, `black.sh`, `test-vllm.py` |
| Python and CUDA environment | `setup-py312-torch212-cu130.sh` |
| Wheels and packaging | `fetch-wheels.sh`, `build-wasm-compilers.sh`, `repackage_tool.py`, `repackage_llama_index.py` |
| Containers and local services | `build-container.sh`, `start_documentdb.sh`, `setup-s3-users.sh`, `hyperdx-init-user.sh` |
| Releases and versions | `release.sh`, `get-versions.sh`, `get-last-release-note.py`, `prepend_version_json.py`, `update-version.sh` |
| Developer utilities | `devbot.sh`, `update-autocomplete-cli.py` |

Use each command's help output for its supported arguments. For example:

```bash
scripts/fetch-wheels.sh help
scripts/setup-py312-torch212-cu130.sh help
scripts/hyperdx-init-user.sh --help
```

## Repackage Python module tool

This tool automates the process of **cloning**, **updating**, and **repackaging** a Python module into a new namespace. It is **generic**, allowing transformations for any repository, and is **configurable** for different module structures.

---

## 🚀 Features
✅ **Works from CLI & Python Script** – Can be executed as a standalone command-line tool or integrated into a Python script.  
✅ **Git Update Support** – Updates an existing repository instead of re-cloning every time.  
✅ **Namespace Transformation** – Dynamically modifies import paths based on configuration.  
✅ **Blacklist Functionality** – Excludes specific directories or files from transformation.  
✅ **Generic & Configurable** – Can be applied to **any** repository with customizable transformations.  

---

## 🛠️ Installation
Ensure you have Python 3 installed.

⚙️ Usage

You can use this tool either from the command line or as a Python script.
🔹 Running from Command Line

```bash
python scripts/repackage_tool.py --repo-url https://github.com/run-llama/llama_index \
                         --branch main \
                         --source-namespace llama_index.core \
                         --target-namespace marie.core \
                         --blacklist tests examples \
                         --output-dir marie_output
```

🔹 Running as a Python Script
    
```python
from repackage_tool import update_or_clone_repository, repackage_module

# Step 1: Clone or update the repository
update_or_clone_repository(
    "https://github.com/run-llama/llama_index", destination="repo_clone", branch="main"
)

# Step 2: Repackage the module
repackage_module(
    source_dir="repo_clone",
    source_namespace="llama_index.core",
    target_namespace="marie.core",
    blacklist=["tests", "examples"],
    output_dir="marie_output",
)
```


🔄 Example Transformation

Before (llama_index.core)

```python
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager
```

After (marie.core)

```python
from marie.core.base.llms.types import ChatMessage
from marie.core.callbacks import CallbackManager
```

