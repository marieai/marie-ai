---
sidebar_position: 1
---

# Contribute to 🦊-Marie

# Coding standards

To ensure the readability of our code, we stick to a few conventions:

* We format python files using black.
* For linting, we use flake8.
* For sorting imports, we use isort.


The `setup.cfg` and `pyproject.toml` already contain the proper configuration for these tools. 
If you are working in a modern IDE (such a VSCode), which integrates these tools, the options will be picked up.
If you are working from the command line, you should use these tools form the root of this repository or via a `make` file.


https://towardsdatascience.com/how-to-add-git-hooks-for-your-python-projects-using-the-pre-commit-framework-773acc3b28a7

https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates

https://flake8.pycqa.org/en/latest/