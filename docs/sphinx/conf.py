import os
import sys
from os import path

sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.

project = 'Marie'
copyright = '2023, Greg'
author = 'Greg'

# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary', 'sphinx_markdown_builder', 'autoapi.extension']


# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.1.2"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "sphinx.ext.autosectionlabel",
    # "sphinx_panels",
    # "myst_parser",
    "sphinx_markdown_builder",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'build']

autoapi_dirs = ['../marie_server']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']

# Marie specific configuration
os.environ["MARIE_DEFAULT_MOUNT"] = "/mnt/marie"
