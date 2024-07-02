from docarray.documents.legacy import LegacyDocument

from marie.utils.pydantic import patch_pydantic_schema_2x


def test_legacy_schema():
    LegacyDocument.schema = classmethod(patch_pydantic_schema_2x)
    legacy_doc_schema = LegacyDocument.schema()
    print(legacy_doc_schema)
