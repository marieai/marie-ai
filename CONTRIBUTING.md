# Contribute to MarieAI(🦊)

# Coding standards

To ensure the readability of our code, we stick to a few conventions:

* We format python files using black.
* For linting, we use flake8.
* For sorting imports, we use isort.


The `setup.cfg` and `pyproject.toml` already contain the proper configuration for these tools. 
If you are working in a modern IDE (such a VSCode), which integrates these tools, the options will be picked up.
If you are working from the command line, you should use these tools form the root of this repository or via a `make` file.



We also provide a .pre-commit-config.yaml file, which enables you to use pre-commit to automatically perform the required linting/formatting actions before you commit your changes. 
You can install it using
```shell
pip install pre-commit
pre-commit install
```