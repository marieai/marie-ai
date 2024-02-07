import unittest

from marie.models.document import DocumentArray
from marie.pipe.namedentity import NamedEntityPipelineComponent


class TestNamedEntityPipelineComponent(unittest.TestCase):
    def setUp(self):
        self.component = NamedEntityPipelineComponent(
            name="NamedEntity", document_indexers={}, logger=None
        )

    def test_extract_named_entity(self):
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

        result = self.component.extract_named_entity(documents, words, boxes)

        self.assertEqual(
            len(result), 0
        )  # No document indexers provided, so no extraction should happen

    def test_predict(self):
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

        result = self.component.predict(documents, words=words, boxes=boxes)

        self.assertEqual(
            len(result.documents), 3
        )  # The same documents should be returned


if __name__ == '__main__':
    unittest.main()
