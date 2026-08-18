"""
Avoid package-level re-exports here.

Importing a specific component submodule should not eagerly import unrelated
classifiers, indexers, OCR helpers, or TroCR dependencies.
"""


def __getattr__(name: str):
    """Load legacy component exports only when they are requested."""
    if name == 'TransformersDocumentClassifier':
        from marie.components.document_classifier.transformers import (
            TransformersDocumentClassifier,
        )

        return TransformersDocumentClassifier
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
