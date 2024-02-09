import unittest

from marie import DocumentArray
from marie.pipe.namedentity import NamedEntityPipelineComponent


def test_predict():
    component = NamedEntityPipelineComponent(
        name="NamedEntity", document_indexers={}, logger=None
    )

    documents = DocumentArray(
        [
            {"text": "This is a document", "tags": {"classification": "class1"}},
            {"text": "Another document", "tags": {"classification": "class2"}},
            {"text": "Yet another document", "tags": {"classification": "class1"}},
        ]
    )
    words = [
        ["This", "is", "a", "document"],
        ["Another", "document"],
        ["Yet", "another", "document"],
    ]
    boxes = [
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]],
        [[0, 0, 10, 10], [20, 20, 30, 30]],
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
    ]

    result = component.predict(documents, words=words, boxes=boxes)

    # assertEqual(len(result.documents), 3)  # The same documents should be returned
