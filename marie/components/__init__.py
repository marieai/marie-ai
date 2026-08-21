"""
Avoid package-level re-exports here.

Importing a specific component submodule should not eagerly import unrelated
classifiers, indexers, OCR helpers, or TroCR dependencies.
"""
