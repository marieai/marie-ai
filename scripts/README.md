# 📌 Repackage Python Module Tool

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
python repackage_tool.py --repo-url https://github.com/run-llama/llama_index \
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


